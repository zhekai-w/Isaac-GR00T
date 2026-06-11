"""
Streaming async client for UR5 + GR00T.

Two-thread architecture:
  - Action receiver thread: captures observations, runs inference (blocking ZMQ call),
    blends incoming action chunks against pending queue entries, merges into queue.
  - Control loop thread (main): pops one TimedAction per dt tick, sends arm + gripper
    commands non-blocking via send_single_action_scaled_joint(wait=False).

Because we drain the queue one action at a time, the queue's remaining contents
are always a precise record of what the robot has NOT yet started executing.
When a new inference chunk arrives, we blend only against those pending timesteps,
which eliminates the "action feedback estimation" problem entirely.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import rclpy
from scipy.signal import savgol_filter
import tyro
from gr00t.eval.robot import RobotInferenceClient

from filter_utils import OneEuroFilter, rts_smoother_chunk, savgol_chunk
from ur5_gr00t_simple_client import UR5SensorNode, HOME_JOINT_POSITIONS


GRIPPER_THRESHOLD = 0.005


@dataclass
class TimedAction:
    timestep: int
    arm: np.ndarray   # (6,)
    gripper: float


AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only":      lambda old, new: new,
    "average":          lambda old, new: 0.5 * (old + new),
    "conservative":     lambda old, new: 0.7 * old + 0.3 * new,
}


class Gr00tStreamingClient:
    """Async client that never pauses for inference.

    Control loop pops one action per `dt` and streams it to the UR5 via
    ``send_single_action_scaled_joint(wait=False)`` + gripper command.
    Inference runs in a background daemon thread; results are blended into
    the shared queue with exact timestep alignment.
    """

    def __init__(
        self,
        sensor: UR5SensorNode,
        host: str = "localhost",
        port: int = 5555,
        lang: str = "pick up the object",
        action_horizon: int = 16,
        chunk_size_threshold: float = 0.4,
        dt: float = 0.05,
        wait: bool = False,
        aggregate_fn_name: str = "conservative",
        chunk_filter: str = "none",
        chunk_filter_q: float = 1e-3,
        chunk_filter_r: float = 1e-4,
        chunk_filter_window: int = 7,
        chunk_filter_polyorder: int = 3,
        filter: bool = False,
        filter_mincutoff: float = 1.0,
        filter_beta: float = 0.1,
        api_token: str | None = None,
    ):
        self.client = RobotInferenceClient(host=host, port=port, api_token=api_token)
        self.sensor = sensor
        self.lang = lang
        self.action_horizon = action_horizon
        self.chunk_size_threshold = chunk_size_threshold
        self.dt = dt
        self.wait = wait
        self.aggregate_fn = AGGREGATE_FUNCTIONS[aggregate_fn_name]

        # Within-chunk smoothing (applied on the full chunk from inference, before queue insertion)
        self.chunk_filter = chunk_filter
        self.chunk_filter_q = chunk_filter_q
        self.chunk_filter_r = chunk_filter_r
        self.chunk_filter_window = chunk_filter_window
        self.chunk_filter_polyorder = chunk_filter_polyorder

        # Thread sync
        self.action_queue: deque[TimedAction] = deque()
        self.queue_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.must_go = threading.Event()
        self.must_go.set()
        self.start_barrier = threading.Barrier(2)

        # State tracking
        self.latest_executed_step = -1
        self.latest_step_lock = threading.Lock()
        self.max_queue_size = action_horizon

        # Per-step low-pass filters (applied on the control-loop side)
        self._use_filter = filter
        if filter:
            freq = 1.0 / dt
            self._arm_filters = [OneEuroFilter(freq, filter_mincutoff, filter_beta) for _ in range(6)]
            self._grip_filter = OneEuroFilter(freq, filter_mincutoff, filter_beta)

        # Last sent gripper (for threshold deadband)
        self._last_grip_sent = 0.0

    # ------------------------------------------------------------------
    # Observation / Inference
    # ------------------------------------------------------------------

    def _build_obs_dict(self, img1, img2, state):
        return {
            "video.azure_kinect": img1[np.newaxis, ...],
            "video.wfov": img2[np.newaxis, ...],
            "state.ur5_arm": state[:6][np.newaxis, :].astype(np.float64),
            "state.gripper": state[6:7][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.lang],
        }

    def _get_action_chunk(self, img1, img2, state, base_step):
        """Blocking ZMQ inference; returns chunk-smoothed list[TimedAction] aligned to base_step."""
        obs = self._build_obs_dict(img1, img2, state)
        raw = self.client.get_action(obs)

        # Extract full chunk arrays (H, 6) and (H,)
        arm_all = np.array([np.atleast_1d(raw["action.ur5_arm"][i]) for i in range(self.action_horizon)])
        grip_all = np.array([np.atleast_1d(raw["action.gripper"][i])[0] for i in range(self.action_horizon)])

        # Within-chunk batch smoothing (same as simple_client/async_client)
        if self.chunk_filter == "rts":
            arm_all = rts_smoother_chunk(arm_all, dt=self.dt, q=self.chunk_filter_q, r=self.chunk_filter_r)
            grip_all = rts_smoother_chunk(grip_all[:, np.newaxis], dt=self.dt, q=self.chunk_filter_q, r=self.chunk_filter_r).squeeze(1)
        elif self.chunk_filter == "savgol":
            arm_all = savgol_chunk(arm_all)
            grip_all = savgol_filter(grip_all, self.chunk_filter_window, self.chunk_filter_polyorder)

        timed_actions = []
        for i in range(self.action_horizon):
            timed_actions.append(TimedAction(timestep=base_step + i, arm=arm_all[i], gripper=grip_all[i]))
        return timed_actions

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _aggregate_into_queue(self, incoming: list[TimedAction]):
        """Merge *incoming* (new chunk) into the queue, blending overlapping timesteps."""
        with self.queue_lock:
            with self.latest_step_lock:
                latest = self.latest_executed_step

            # Only pending actions remain (not yet dispatched)
            existing = {a.timestep: a for a in self.action_queue if a.timestep > latest}

            for new_a in incoming:
                if new_a.timestep <= latest:
                    continue
                if new_a.timestep in existing:
                    old = existing[new_a.timestep]
                    existing[new_a.timestep] = TimedAction(
                        timestep=new_a.timestep,
                        arm=self.aggregate_fn(old.arm, new_a.arm),
                        gripper=float(self.aggregate_fn(old.gripper, new_a.gripper)),
                    )
                else:
                    existing[new_a.timestep] = new_a

            self.action_queue = deque(sorted(existing.values(), key=lambda a: a.timestep))
            self.max_queue_size = max(self.max_queue_size, len(self.action_queue))

    def _queue_depleted(self) -> bool:
        with self.queue_lock:
            size = len(self.action_queue)
        if self.max_queue_size <= 0:
            return True
        return size / self.max_queue_size <= self.chunk_size_threshold

    def _queue_empty(self) -> bool:
        with self.queue_lock:
            return len(self.action_queue) == 0

    # ------------------------------------------------------------------
    # Thread 1 — Action Receiver (background)
    # ------------------------------------------------------------------

    def _action_receiver_loop(self):
        """Daemon thread: captures obs, runs inference (blocking), fills queue."""
        self.start_barrier.wait()

        while not self.shutdown_event.is_set():
            should_infer = self._queue_depleted()
            force = self.must_go.is_set() and self._queue_empty()

            if not (should_infer or force):
                time.sleep(0.001)
                continue

            # Base step = next timestep the robot hasn't started yet.
            # Because the queue tracks precisely what's pending, we don't need
            # wall-clock estimation — the queue itself is the ground truth.
            with self.latest_step_lock:
                base_step = self.latest_executed_step + 1

            img1 = self.sensor.get_azure_kinect_image()
            img2 = self.sensor.get_wfov_image()
            state = self.sensor.get_joint_state()
            if state is None or img1 is None or img2 is None:
                time.sleep(0.01)
                continue
            # Reorder to training convention: [pan, lift, elbow, w1, w2, w3, gripper]
            state = state[[5, 0, 1, 2, 3, 4, 6]]

            try:
                timed_actions = self._get_action_chunk(img1, img2, state, base_step)
            except Exception:
                time.sleep(0.1)
                continue

            self._aggregate_into_queue(timed_actions)

            if force:
                self.must_go.clear()
            self.must_go.set()

    # ------------------------------------------------------------------
    # Thread 2 — Control Loop (main)
    # ------------------------------------------------------------------

    def _control_loop(self, num_steps: int):
        """Main thread: pops one TimedAction per `dt` and streams to hardware."""
        self.start_barrier.wait()

        for _ in range(num_steps):
            if self.shutdown_event.is_set():
                break

            loop_start = time.perf_counter()

            action: TimedAction | None = None
            with self.queue_lock:
                if self.action_queue:
                    timed_action = self.action_queue.popleft()
                    action = timed_action
                    with self.latest_step_lock:
                        self.latest_executed_step = timed_action.timestep

            if action is not None:
                arm = action.arm
                grip = action.gripper

                # Optional per-step low-pass filter
                if self._use_filter:
                    arm = np.array([self._arm_filters[j](arm[j]) for j in range(6)])
                    grip = self._grip_filter(grip)

                self.sensor.send_single_action_scaled_joint(arm, dt=self.dt, wait=self.wait)

                # Gripper with deadband
                if abs(grip - self._last_grip_sent) > GRIPPER_THRESHOLD:
                    self.sensor.send_gripper_command(float(grip))
                    self._last_grip_sent = grip

            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, self.dt - elapsed))

    def run(self, num_steps: int = 1000):
        """Start both threads and run for `num_steps` control iterations."""
        assert self.client.ping(), "GR00T server not reachable"

        receiver = threading.Thread(target=self._action_receiver_loop, daemon=True)
        receiver.start()

        try:
            self._control_loop(num_steps)
        finally:
            self.shutdown_event.set()
            receiver.join(timeout=2.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class ArgsConfig:
    # Connection
    host: str = "localhost"
    port: int = 5555
    api_token: str | None = None

    # Task
    lang: str = "place the small cube on the red box."
    num_steps: int = 1000

    # Async core
    action_horizon: int = 16
    chunk_size_threshold: float = 0.5
    dt: float = 0.05
    wait: bool = False
    aggregate_fn_name: Literal["weighted_average", "latest_only", "average", "conservative"] = "conservative"

    # Within-chunk batch smoothing (applied to full inference chunk before queue insertion)
    chunk_filter: Literal["none", "savgol", "rts"] = "rts"
    chunk_filter_q: float = 1e-3
    chunk_filter_r: float = 1e-4
    chunk_filter_window: int = 7
    chunk_filter_polyorder: int = 3

    # Per-step low-pass filter
    filter: bool = False
    filter_mincutoff: float = 1.0
    filter_beta: float = 0.1


def main(args: ArgsConfig):
    rclpy.init()
    sensor = UR5SensorNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sensor,), daemon=True)
    spin_thread.start()

    print("Waiting for sensor data...")
    while sensor.get_joint_state() is None or len(sensor.k4a_buffer) == 0 or len(sensor.wfov_buffer) == 0:
        time.sleep(0.1)
    print("Sensors ready.")

    print("Moving to home position...")
    sensor.send_single_action_scaled_joint(HOME_JOINT_POSITIONS, dt=3.0, wait=True)
    sensor.send_gripper_command(0.0)
    print("Home position reached.")

    client = Gr00tStreamingClient(
        sensor=sensor,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
        lang=args.lang,
        action_horizon=args.action_horizon,
        chunk_size_threshold=args.chunk_size_threshold,
        dt=args.dt,
        wait=args.wait,
        aggregate_fn_name=args.aggregate_fn_name,
        chunk_filter=args.chunk_filter,
        chunk_filter_q=args.chunk_filter_q,
        chunk_filter_r=args.chunk_filter_r,
        chunk_filter_window=args.chunk_filter_window,
        chunk_filter_polyorder=args.chunk_filter_polyorder,
        filter=args.filter,
        filter_mincutoff=args.filter_mincutoff,
        filter_beta=args.filter_beta,
    )

    try:
        client.run(num_steps=args.num_steps)
    finally:
        sensor.stop()
        sensor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
