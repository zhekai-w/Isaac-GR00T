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

One thread, one preemption per inference cycle, full-horizon trajectories
throughout: no sub-chunk dispatching and no receiver/control thread pair racing
on a shared queue.
"""

import random
import os
import threading
import time
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import tyro
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from gr00t.eval.robot import RobotInferenceClient
from pynput import keyboard
from trajectory_msgs.msg import JointTrajectoryPoint

from filter_utils import AGGREGATE_FUNCTIONS, apply_chunk_filter, blend_chunk_boundary
from ur5_client_common import (
    GRIPPER_THRESHOLD,
    HOME_JOINT_POSITIONS,
    HOME_TOLERANCE,
    TASK,
    UR5SensorNode,
    build_obs_dict,
    make_keyboard_handler,
    init_sensor_and_wait,
    shutdown_sensor,
)

# Floor on the first waypoint's time_from_start. A point due sooner than this
# would be a step the controller has to take in one update cycle.
MIN_LEAD = 0.02



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

    # Executing-timestep source: measured from controller_state, falling back to
    # wall-clock extrapolation (which assumes nothing slowed the arm down).
    use_measured_step: bool = True
    use_speed_scaling: bool = True
    ctrl_state_window: int = 4         # +-timesteps searched around the estimate
    ctrl_state_timeout: float = 0.2    # s; older controller_state is ignored
    ctrl_state_max_resid: float = 0.05 # rad; worse match falls back to extrapolation

    # Overlap blending (lerobot aggregate)
    aggregate_fn_name: Literal["ramp", "latest_only", "weighted_average", "average", "conservative"] = "weighted_average"

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
    # Write timed_chunk_log.npz for scripts/compare_logs.py (pass --log to enable).
    log: bool = False


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
        self._n_est_fallback = 0                   # executing_step() calls with no
                                                   # usable controller_state match

        # Diagnostic log, keyed by timestep and overwritten on each dispatch: a later
        # chunk supersedes the tail of an earlier one, and appending both would put
        # duplicate timesteps in the commanded sequence.
        self._cmd: dict[int, tuple] = {}           # ts -> (due_wall, pos(6,), cycle)
        self._log_cycles: list[tuple] = []
        self._log_grip_t: list[float] = []

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
        """Which (fractional) timestep the arm is executing at wall time t.

        Measured, not guessed: the controller publishes its desired position and
        `self.pending` maps timestep -> commanded position, so the two can be matched.
        Falls back to extrapolation when controller_state is absent/stale/ambiguous.
        """
        if t is None:
            t = time.perf_counter()
        est = self._extrapolated_step(t)
        if not self.args.use_measured_step:
            return est
        meas = self._measured_step(est, t)
        if meas is None:
            self._n_est_fallback += 1
            return est
        return meas

    def _extrapolated_step(self, t: float) -> float:
        """Open-loop estimate: waypoints elapse at 1/dt, scaled by the speed factor.

        Without the scaling this is wrong, not merely imprecise — JTC advances
        `traj_time_ += period * scaling_factor_`, so at 50% the estimate drifts
        without bound.
        """
        if self.anchor_wall is None:
            return self.anchor_step
        scale = self.sensor.speed_scaling if self.args.use_speed_scaling else 1.0
        est = self.anchor_step + (t - self.anchor_wall) * scale / self.args.dt
        # Clamp to the dispatched trajectory's end: once the chunk finishes the
        # controller holds position, time no longer advances the timestep.
        return min(est, float(self.pending_last_ts)) if self.pending_last_ts >= 0 else est

    def _measured_step(self, est: float, t: float) -> float | None:
        """Locate the controller's desired position within `pending`.

        Windowed around `est` — joint poses repeat over a task, so a global
        nearest-waypoint match would teleport to an unrelated part of the trajectory.
        """
        cs = self.sensor.ctrl_state
        if cs is None or not self.pending:
            return None
        t_cs, desired = cs
        age = t - t_cs
        if not (-0.05 <= age <= self.args.ctrl_state_timeout):
            return None

        w = self.args.ctrl_state_window
        best = None
        for a in range(int(np.floor(est)) - w, int(np.ceil(est)) + w):
            p0, p1 = self.pending.get(a), self.pending.get(a + 1)
            if p0 is None or p1 is None:
                continue
            v = p1 - p0
            n = float(v @ v)
            if n < 1e-12:
                continue
            u = min(1.0, max(0.0, float((desired - p0) @ v) / n))
            resid = float(np.linalg.norm(p0 + u * v - desired))
            if best is None or resid < best[0]:
                best = (resid, a + u)
        if best is None or best[0] > self.args.ctrl_state_max_resid:
            # Poor match: right after a dispatch the controller is still bridging
            # from the old trajectory, so its desired position is on no segment of
            # the new `pending` at all.
            return None

        # Advance for the age of the sample, and refuse a match that disagrees with
        # the open-loop estimate by more than the search window — that would be a
        # mismatch found in a repeated pose, not a correction.
        scale = self.sensor.speed_scaling if self.args.use_speed_scaling else 1.0
        step = best[1] + age * scale / self.args.dt
        if abs(step - est) > w:
            return None
        return min(step, float(self.pending_last_ts)) if self.pending_last_ts >= 0 else step

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _send_trajectory(self, arm_actions: np.ndarray, lead: float | None = None):
        """Fire-and-forget FollowJointTrajectory goal. New goal preempts old.

        `lead` = seconds until the FIRST waypoint. The arm sits at a fractional
        timestep and first_ts is 1-2 steps ahead (extra_delay_steps), so a flat dt
        (lead=None, old behavior) compresses that into one step — up to 2x speed at
        every chunk boundary.
        """
        if lead is None:
            lead = self.args.dt
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
            t = lead + i * self.args.dt
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
                if self.args.log and self.sensor.js_recording:
                    self._log_grip_t.append(time.perf_counter())
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
        return apply_chunk_filter(
            arm, grip, a.chunk_filter, a.dt, q=a.chunk_filter_q, r=a.chunk_filter_r,
            window=a.chunk_filter_window, polyorder=a.chunk_filter_polyorder)

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
        # Time left in the chunk, from where the arm actually is and at the rate it
        # is actually running — not from the dispatch anchor at nominal speed.
        now = time.perf_counter()
        scale = max(0.05, self.sensor.speed_scaling if self.args.use_speed_scaling else 1.0)
        t_end = now + (self.pending_last_ts - self.executing_step(now)) * self.args.dt / scale
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
            want = int(np.floor(s_now)) + 1 + self.args.extra_delay_steps - (s_obs + 1)
            # `min_dispatch_steps` caps how far into the chunk we may skip, so when
            # inference outruns the chunk this clamp yields a first waypoint that
            # lies BEHIND the arm — a rewind command. Clamp as before, then check
            # the result against where the arm actually is (below, at dispatch).
            start_idx = max(0, min(want, H - self.args.min_dispatch_steps))

        first_ts = s_obs + 1 + start_idx
        arm_tail = arm_all[start_idx:].copy()     # (S, 6)
        grip_tail = grip_all[start_idx:].copy()   # (S,)
        new_ts = np.arange(first_ts, first_ts + arm_tail.shape[0])

        # --- lerobot-style overlap blending against still-pending old actions ---
        # "ramp" needs the overlap counted first — its weight depends on position.
        overlap_idx = [k for k, ts in enumerate(new_ts) if int(ts) in self.pending]
        n_overlap = len(overlap_idx)
        for i, k in enumerate(overlap_idx):
            old = self.pending[int(new_ts[k])]
            if self.aggregate_fn is None:                  # "ramp": 0 -> 1 across it
                a = (i + 1) / (n_overlap + 1)
                arm_tail[k] = (1.0 - a) * old + a * arm_tail[k]
            else:
                arm_tail[k] = self.aggregate_fn(old, arm_tail[k])

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
        # `lead` = true remaining travel to the first waypoint, recomputed at dispatch
        # rather than at t_ret.
        t_dispatch = time.perf_counter()
        s_disp = self.executing_step(t_dispatch)
        gap_steps = first_ts - s_disp
        if self.anchor_wall is not None and gap_steps < 0.5:
            # First waypoint at or behind the arm: dispatching would rewind, and the
            # chunk is stale as a prediction anyway. Drop it and re-infer.
            print(f"[stale] chunk dropped: infer={t_infer:.2f}s left the first "
                  f"waypoint {-gap_steps:.1f} steps behind the arm. Raise --dt or "
                  f"--action-horizon, or lower --min-dispatch-steps "
                  f"(H={H}, min_dispatch={self.args.min_dispatch_steps}).")
            if self.args.log and self.sensor.js_recording:
                self._log_cycles.append((
                    t_dispatch, t_infer, float(s_obs), float(s_now), float(start_idx),
                    0.0, 0.0, 0.0, self.sensor.speed_scaling, 1.0,
                ))
            return False
        lead = max(MIN_LEAD, gap_steps * self.args.dt)
        self._send_trajectory(arm_tail, lead=lead)
        self.anchor_step = float(first_ts)       # waypoint first_ts is reached
        self.anchor_wall = t_dispatch + lead     # exactly `lead` from now
        self.pending = {int(ts): arm_tail[k] for k, ts in enumerate(new_ts)}
        self.pending_last_ts = int(new_ts[-1])
        self._start_gripper(grip_tail.tolist())

        if self.args.log and self.sensor.js_recording:
            for k, ts in enumerate(new_ts):
                self._cmd[int(ts)] = (t_dispatch + lead + k * self.args.dt,
                                      arm_tail[k].copy(), cycle)
            self._log_cycles.append((
                t_dispatch, t_infer, float(s_obs), float(s_now), float(start_idx),
                lead, float(n_overlap), float(arm_tail.shape[0]),
                self.sensor.speed_scaling, 0.0,   # 0 = dispatched, 1 = dropped stale
            ))

        if self.args.verbose:
            print(f"[cycle {cycle}] infer={t_infer:.3f}s  s_obs={s_obs}  "
                  f"s_now={s_now:.1f}  start_idx={start_idx}  lead={lead*1000:.0f}ms  "
                  f"dispatched={arm_tail.shape[0]}  overlap_blended={n_overlap}  "
                  f"scale={self.sensor.speed_scaling:.2f}  "
                  f"est_fallbacks={self._n_est_fallback}")
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

    sensor, spin_thread = init_sensor_and_wait(need_frames=abs(args.buffer) + 1)

    if args.use_measured_step:
        if sensor.ctrl_state is None:
            print("[WARN] no controller_state received — falling back to wall-clock "
                  "estimation. Check that scaled_joint_trajectory_controller is "
                  "active and state_publish_rate > 0.")
        else:
            print("controller_state OK — executing timestep is measured, not estimated")
    if args.use_speed_scaling:
        print(f"speed scaling: {sensor.speed_scaling:.2f}"
              + ("" if abs(sensor.speed_scaling - 1.0) < 0.01
                 else "  <- arm is NOT at full speed (pendant slider / safety limit)"))

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

    def kb_start():
        ctl.inferring = True
        if args.log:
            sensor.js_recording = True
        print("[KB] Inference started")

    def kb_pause():
        ctl.inferring = False
        sensor.js_recording = False
        print("[KB] Inference paused")

    def kb_home():
        ctl.returning_home = True
        print("[KB] Returning home")

    def kb_quit():
        ctl.quit_flag = True
        print("[KB] Quit requested")

    kb_listener = keyboard.Listener(on_press=make_keyboard_handler({
        's': kb_start, 'p': kb_pause, 'h': kb_home, 'q': kb_quit
    }))
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
        sensor.js_recording = False
        if args.log and ctl._cmd:
            ts = np.array(sorted(ctl._cmd))
            np.savez(
                "timed_chunk_log.npz",
                t=np.array([ctl._cmd[k][0] for k in ts]),
                step=ts,
                cmd=np.array([ctl._cmd[k][1] for k in ts]),
                chunk=np.array([ctl._cmd[k][2] for k in ts]),
                cycles=np.array(ctl._log_cycles) if ctl._log_cycles else np.zeros((0, 10)),
                grip_t=np.array(ctl._log_grip_t),
                js=np.array(sensor.js_log) if sensor.js_log else np.zeros((0, 7)),
                # `cmd` here is what was DISPATCHED — post rts, post overlap blend,
                # post boundary blend. simple_client logs the raw chunk instead, so
                # compare_logs.py has to know which stage it is looking at.
                cmd_stage=np.array("dispatched"),
                dt=np.array(args.dt),
                action_horizon=np.array(args.action_horizon),
                aggregate_fn_name=np.array(args.aggregate_fn_name),
                chunk_filter=np.array(args.chunk_filter),
                control_interface=np.array("jtc_action"),
            )
            print(f"Diagnostic log saved: timed_chunk_log.npz "
                  f"({len(ts)} waypoints, {len(ctl._log_cycles)} cycles, "
                  f"{len(sensor.js_log)} js samples)")
        sensor.stop()
        shutdown_sensor(sensor)


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
