"""
Timed-chunk async client for UR5 + GR00T.

Applies the lerobot async-inference concept (https://huggingface.co/docs/lerobot/async)
to CHUNK-mode execution:

  lerobot async keeps an action queue, triggers inference BEFORE the queue
  empties, and blends the new chunk into the not-yet-executed actions. That
  needs to know which timestep the robot is currently executing. With
  one-action-at-a-time dispatch the queue itself tells you; in chunk mode the
  whole trajectory lives inside scaled_joint_trajectory_controller, so there
  is nothing to inspect.

  Solution here: estimate the executing timestep from wall-clock time. The
  controller runs waypoints at fixed dt cadence, so

      executing_step(t) = anchor_step + (t - anchor_wall) / dt

  where (anchor_step, anchor_wall) is recorded at every trajectory dispatch.

Cycle (robot never pauses):
  1. dispatch chunk as ONE FollowJointTrajectory goal (spline interpolation
     -> smooth, same as simple client chunk mode)
  2. sleep until  chunk_end - inference_latency_est - margin
  3. capture obs, tag it with estimated timestep s_obs, run blocking
     inference — robot keeps executing the old chunk meanwhile
  4. when the new chunk returns:
       - drop actions whose timestep is already in the past
       - blend overlapping timesteps against the still-pending tail of the
         previous chunk (lerobot aggregate functions)
       - boundary-blend the first waypoints from the estimated current
         commanded position (no positional jump)
       - dispatch as a new goal -> controller preempts the old one mid-flight

Compared to ur5_gr00t_async_client.py (failed attempt): no sub-chunk
dispatching, no receiver/control thread pair racing on a queue. One thread,
one preemption per inference cycle, full-horizon trajectories throughout.
"""

import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import rclpy
import tyro
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from gr00t.eval.robot import RobotInferenceClient
from pynput import keyboard
from scipy.signal import savgol_filter
from trajectory_msgs.msg import JointTrajectoryPoint

from filter_utils import blend_chunk_boundary, rts_smoother_chunk, savgol_chunk
from ur5_gr00t_simple_client import (
    GRIPPER_THRESHOLD,
    HOME_JOINT_POSITIONS,
    HOME_TOLERANCE,
    TASK,
    UR5SensorNode,
    build_obs_dict,
)

AGGREGATE_FUNCTIONS = {
    "latest_only":      lambda old, new: new,
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,  # favor newer
    "average":          lambda old, new: 0.5 * (old + new),
    "conservative":     lambda old, new: 0.7 * old + 0.3 * new,  # favor older
}


@dataclass
class ArgsConfig:
    # Connection
    host: str = "localhost"
    port: int = 5555

    # Task
    lang: str | None = None            # None -> random task from TASK list
    action_horizon: int = 16
    dt: float = 0.15
    gripper_max_effort: float = 50.0

    # Async timing
    trigger_margin_steps: int = 2      # start inference this many steps before
                                       # (estimated) inference completion would
                                       # overrun the current chunk's end
    extra_delay_steps: int = 1         # skip this many extra actions on dispatch
                                       # to cover goal-send + controller latency
    min_dispatch_steps: int = 4        # always dispatch at least this many
                                       # waypoints, even if inference ran long

    # Overlap blending (lerobot aggregate)
    aggregate_fn_name: Literal["latest_only", "weighted_average", "average", "conservative"] = "weighted_average"

    # Within-chunk smoothing (same options as simple client)
    chunk_filter: Literal["none", "savgol", "rts"] = "none"
    chunk_filter_window: int = 7
    chunk_filter_polyorder: int = 3
    chunk_filter_q: float = 1e-3
    chunk_filter_r: float = 1e-4

    # Boundary blending at preemption point
    boundary_blend: bool = True
    boundary_blend_steps: int = 4

    # Frame buffer selection: 0 = current frame, -N = N frames ago
    buffer: int = 0

    # Diagnostics
    save_images: bool = False
    verbose: bool = True


class TimedChunkClient:
    def __init__(self, args: ArgsConfig, sensor: UR5SensorNode):
        self.args = args
        self.sensor = sensor
        self.client = RobotInferenceClient(host=args.host, port=args.port)

        # --- Wall-clock anchor: robot was at (float) timestep anchor_step at
        # wall time anchor_wall; executing_step(t) = anchor_step + (t - anchor_wall)/dt
        self.anchor_step: float = 0.0
        self.anchor_wall: float | None = None
        self.pending: dict[int, np.ndarray] = {}   # timestep -> commanded arm pos (6,)
        self.pending_last_ts: int = -1

        self.t_infer_ema: float | None = None      # EMA of inference latency (s)

        self._grip_stop: threading.Event | None = None
        self._grip_thread: threading.Thread | None = None
        self._last_grip_sent = 0.0

        self.aggregate_fn = AGGREGATE_FUNCTIONS[args.aggregate_fn_name]

        # keyboard flags
        self.inferring = False
        self.returning_home = False
        self.quit_flag = False

    # ------------------------------------------------------------------
    # Timestep estimation
    # ------------------------------------------------------------------

    def executing_step(self, t: float | None = None) -> float:
        """Estimated (float) timestep the robot is executing at wall time t."""
        if self.anchor_wall is None:
            return self.anchor_step
        if t is None:
            t = time.perf_counter()
        # Clamp to the dispatched trajectory's end: once the chunk finishes the
        # controller holds position, time no longer advances the timestep.
        est = self.anchor_step + (t - self.anchor_wall) / self.args.dt
        return min(est, float(self.pending_last_ts)) if self.pending_last_ts >= 0 else est

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _send_trajectory(self, arm_actions: np.ndarray):
        """Fire-and-forget FollowJointTrajectory goal. New goal preempts old."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.sensor.scaled_joint_names_reordered
        # Waypoint velocities switch the controller's interpolation from
        # linear to cubic splines — without them every waypoint (and the
        # preemption splice) is a velocity discontinuity.
        arm_actions = np.atleast_2d(arm_actions)
        if arm_actions.shape[0] >= 2:
            velocities = np.gradient(arm_actions, self.args.dt, axis=0)
        else:
            velocities = np.zeros_like(arm_actions)
        for i, positions in enumerate(arm_actions):
            point = JointTrajectoryPoint()
            point.positions = np.atleast_1d(positions).tolist()
            point.velocities = velocities[i].tolist()
            t = (i + 1) * self.args.dt
            point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(point)
        self.sensor._trajectory_action.send_goal_async(goal)

    def _gripper_paced(self, grip_chunk: list, stop: threading.Event):
        for g in grip_chunk:
            if stop.is_set():
                return
            if abs(g - self._last_grip_sent) > GRIPPER_THRESHOLD:
                self.sensor.send_gripper_command(g, max_effort=self.args.gripper_max_effort)
                self._last_grip_sent = g
            stop.wait(self.args.dt)

    def _start_gripper(self, grip_chunk: list):
        if self._grip_stop is not None:
            self._grip_stop.set()
        self._grip_stop = threading.Event()
        self._grip_thread = threading.Thread(
            target=self._gripper_paced, args=(grip_chunk, self._grip_stop), daemon=True
        )
        self._grip_thread.start()

    def _stop_gripper(self):
        if self._grip_stop is not None:
            self._grip_stop.set()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_chunk_filter(self, arm: np.ndarray, grip: np.ndarray):
        a = self.args
        H = arm.shape[0]
        if a.chunk_filter == "rts" and H >= 2:
            arm = rts_smoother_chunk(arm, dt=a.dt, q=a.chunk_filter_q, r=a.chunk_filter_r)
            grip = rts_smoother_chunk(grip[:, np.newaxis], dt=a.dt,
                                      q=a.chunk_filter_q, r=a.chunk_filter_r).squeeze(1)
        elif a.chunk_filter == "savgol" and H > a.chunk_filter_window:
            arm = savgol_chunk(arm, a.chunk_filter_window, a.chunk_filter_polyorder)
            grip = savgol_filter(grip, a.chunk_filter_window, a.chunk_filter_polyorder)
        return arm, grip

    # ------------------------------------------------------------------
    # Observation / inference
    # ------------------------------------------------------------------

    def _capture_obs(self, cycle: int):
        img1 = self.sensor.get_azure_kinect_image(self.args.buffer)
        img2 = self.sensor.get_wfov_image(self.args.buffer)
        state = self.sensor.get_joint_state()
        if img1 is None or img2 is None or state is None:
            return None
        # get_joint_state() is already canonical [pan, lift, elbow, w1..w3, grip]
        # (callback keys /joint_states by name) — no reindex needed.
        if self.args.save_images:
            cv2.imwrite(f"inference_images/cycle_{cycle:04d}_k4a.jpg",
                        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"inference_images/cycle_{cycle:04d}_wfov.jpg",
                        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        return build_obs_dict(img1, img2, state, self.args.lang)

    def _infer(self, obs) -> tuple[np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        action_dict = self.client.get_action(obs)
        t_infer = time.perf_counter() - t0
        self.t_infer_ema = (t_infer if self.t_infer_ema is None
                            else 0.7 * self.t_infer_ema + 0.3 * t_infer)
        arm = np.atleast_2d(action_dict["action.ur5_arm"])                      # (H, 6)
        grip = np.atleast_1d(action_dict["action.gripper"]).flatten()           # (H,)
        return arm, grip, t_infer

    # ------------------------------------------------------------------
    # Trigger timing
    # ------------------------------------------------------------------

    def _wait_until_trigger(self) -> bool:
        """Sleep until it is time to capture obs for the next inference.

        Trigger point = chunk end minus (estimated inference latency + margin),
        so the new chunk lands just before the old one runs out.
        Returns False if interrupted by a keyboard flag.
        """
        if self.anchor_wall is None or self.pending_last_ts < 0:
            return True
        est_lat = self.t_infer_ema if self.t_infer_ema is not None else 1.0
        t_end = self.anchor_wall + (self.pending_last_ts - self.anchor_step) * self.args.dt
        t_trig = t_end - est_lat - self.args.trigger_margin_steps * self.args.dt
        while time.perf_counter() < t_trig:
            if self.quit_flag or self.returning_home or not self.inferring:
                return False
            time.sleep(0.005)
        return True

    # ------------------------------------------------------------------
    # One inference->dispatch cycle
    # ------------------------------------------------------------------

    def step(self, cycle: int) -> bool:
        """Run one capture -> infer -> merge -> dispatch cycle."""
        obs = self._capture_obs(cycle)
        if obs is None:
            time.sleep(0.05)
            return False
        # action[i] of the returned chunk targets absolute timestep s_obs + 1 + i
        s_obs = int(round(self.executing_step()))

        try:
            arm_all, grip_all, t_infer = self._infer(obs)
        except Exception as e:
            print(f"[infer] error: {e}")
            time.sleep(0.1)
            return False

        H = min(self.args.action_horizon, arm_all.shape[0])
        arm_all, grip_all = arm_all[:H], grip_all[:H]

        t_ret = time.perf_counter()
        if self.anchor_wall is None:
            start_idx = 0          # first dispatch: robot idle, use full chunk
            s_now = float(s_obs)
        else:
            s_now = self.executing_step(t_ret)
            # skip actions whose timestep is already in the past (+ latency pad)
            start_idx = int(np.floor(s_now)) + 1 + self.args.extra_delay_steps - (s_obs + 1)
            start_idx = max(0, min(start_idx, H - self.args.min_dispatch_steps))

        first_ts = s_obs + 1 + start_idx
        arm_tail = arm_all[start_idx:].copy()     # (S, 6)
        grip_tail = grip_all[start_idx:].copy()   # (S,)
        new_ts = np.arange(first_ts, first_ts + arm_tail.shape[0])

        # --- lerobot-style overlap blending against still-pending old actions ---
        n_overlap = 0
        for k, ts in enumerate(new_ts):
            old = self.pending.get(int(ts))
            if old is not None:
                arm_tail[k] = self.aggregate_fn(old, arm_tail[k])
                n_overlap += 1

        # --- within-chunk smoothing ---
        arm_tail, grip_tail = self._apply_chunk_filter(arm_tail, grip_tail)

        # --- boundary blend from estimated current commanded position ---
        if self.args.boundary_blend and self.pending:
            cur = int(np.floor(s_now))
            prev_pos = self.pending.get(cur)
            if prev_pos is None:
                prev_pos = self.pending.get(cur + 1)
            prev_prev = self.pending.get(cur - 1)
            if prev_pos is not None:
                prev_vel = ((prev_pos - prev_prev) / self.args.dt
                            if prev_prev is not None else np.zeros(6))
                arm_tail = blend_chunk_boundary(
                    arm_tail, prev_pos, prev_vel, self.args.dt,
                    self.args.boundary_blend_steps)

        # --- dispatch (preempts old goal) + re-anchor ---
        t_dispatch = time.perf_counter()
        self._send_trajectory(arm_tail)
        self.anchor_step = float(first_ts - 1)   # robot ~at first_ts-1 now,
        self.anchor_wall = t_dispatch            # reaches first_ts after dt
        self.pending = {int(ts): arm_tail[k] for k, ts in enumerate(new_ts)}
        self.pending_last_ts = int(new_ts[-1])
        self._start_gripper(grip_tail.tolist())

        if self.args.verbose:
            print(f"[cycle {cycle}] infer={t_infer:.3f}s  s_obs={s_obs}  "
                  f"s_now={s_now:.1f}  start_idx={start_idx}  "
                  f"dispatched={arm_tail.shape[0]}  overlap_blended={n_overlap}")
        return True

    # ------------------------------------------------------------------
    # Home / reset
    # ------------------------------------------------------------------

    def reset_state(self):
        self.anchor_wall = None
        self.anchor_step = 0.0
        self.pending = {}
        self.pending_last_ts = -1
        self._last_grip_sent = 0.0

    def go_home(self):
        self._stop_gripper()
        state = self.sensor.get_joint_state()
        if state is not None:
            # already canonical [pan, lift, elbow, w1..w3, grip] — no reindex
            if not np.allclose(state[:6], HOME_JOINT_POSITIONS, atol=HOME_TOLERANCE):
                # home goal preempts whatever trajectory is running
                self.sensor.send_single_action_scaled_joint(HOME_JOINT_POSITIONS, dt=3.0, wait=True)
        self.sensor.send_gripper_command(0.0, max_effort=self.args.gripper_max_effort)
        self.reset_state()


def main(args: ArgsConfig):
    assert -4 <= args.buffer <= 0, f"--buffer must be in [-4, 0], got {args.buffer}"
    if args.chunk_filter == "savgol":
        assert args.chunk_filter_window % 2 == 1, "chunk_filter_window must be odd"
        assert args.chunk_filter_polyorder < args.chunk_filter_window

    rclpy.init()
    sensor = UR5SensorNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sensor,), daemon=True)
    spin_thread.start()

    need_frames = abs(args.buffer) + 1
    print(f"Waiting for sensor data (need {need_frames} buffered frame(s))...")
    while (sensor.get_joint_state() is None
           or len(sensor.k4a_buffer) < need_frames
           or len(sensor.wfov_buffer) < need_frames):
        time.sleep(0.1)
    print("Sensors ready.")

    ctl = TimedChunkClient(args, sensor)
    assert ctl.client.ping(), "Server not reachable"
    print("Modality config:", ctl.client.get_modality_config())

    print("Moving to home position before inference...")
    ctl.go_home()
    print("Home position reached.")

    use_random_task = args.lang is None
    task_idx = 0
    if use_random_task:
        task_idx = random.randrange(len(TASK))
        args.lang = TASK[task_idx]
    print(f"Task [{task_idx}]: {args.lang}")

    def on_press(key):
        try:
            ch = key.char
        except AttributeError:
            return
        if ch == 's':
            ctl.inferring = True
            print("[KB] Inference started")
        elif ch == 'p':
            ctl.inferring = False
            print("[KB] Inference paused")
        elif ch == 'h':
            ctl.returning_home = True
            print("[KB] Returning home")
        elif ch == 'q':
            ctl.quit_flag = True
            print("[KB] Quit requested")

    kb_listener = keyboard.Listener(on_press=on_press)
    kb_listener.start()
    print("Keyboard ready: s=start  p=pause  h=home  q=quit")

    if args.save_images:
        os.makedirs("inference_images", exist_ok=True)

    try:
        cycle = 0
        while not ctl.quit_flag:
            if ctl.returning_home:
                ctl.go_home()
                if use_random_task:
                    task_idx = random.randrange(len(TASK))
                    args.lang = TASK[task_idx]
                print(f"Home pose reached. Next task [{task_idx}]: {args.lang}")
                ctl.returning_home = False
                continue

            if not ctl.inferring:
                time.sleep(0.1)
                continue

            # Sleep until the early-trigger point of the executing chunk;
            # bail out early if a keyboard flag flipped meanwhile.
            if not ctl._wait_until_trigger():
                continue

            if ctl.step(cycle):
                cycle += 1
    finally:
        ctl._stop_gripper()
        kb_listener.stop()
        sensor.stop()
        sensor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
