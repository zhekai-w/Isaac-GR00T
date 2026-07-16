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
        lookahead: int = 4,
        control_interface: str = "forward_position",
        stream_hz: float = 125.0,
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
        self.lookahead = max(1, lookahead)
        self.control_interface = control_interface
        self.stream_hz = stream_hz
        self.aggregate_fn = AGGREGATE_FUNCTIONS[aggregate_fn_name]

        # forward_position (servoj) streaming: the control loop sets a target
        # segment per tick; a high-rate thread interpolates along it and
        # publishes setpoints to the forward_position_controller.
        self._segment_lock = threading.Lock()
        self._segment: tuple[float, float, np.ndarray, np.ndarray] | None = None  # (t0, dur, from, to)
        self._last_target: np.ndarray | None = None

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

        # Diagnostic log: per-tick wall time, timestep, commanded and measured positions
        self._log_t: list[float] = []
        self._log_step: list[int] = []
        self._log_cmd: list[np.ndarray] = []
        self._log_meas: list[np.ndarray] = []
        self._log_prof: list[tuple] = []
        self._log_grip_t: list[float] = []

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
    # Thread 3 — forward_position (servoj) streamer
    # ------------------------------------------------------------------

    def _fpc_streamer_loop(self):
        """Publishes interpolated setpoints at stream_hz along the current
        segment (set by the control loop once per dt). This is the moveit_servo
        pattern: servoj consumes a dense setpoint stream and its lookahead_time
        smooths the remaining discretization."""
        period = 1.0 / self.stream_hz
        while not self.shutdown_event.is_set():
            tick = time.perf_counter()
            with self._segment_lock:
                seg = self._segment
            if seg is not None:
                t0, dur, frm, to = seg
                alpha = min(1.0, max(0.0, (tick - t0) / dur))
                self.sensor.publish_fpc(frm + alpha * (to - frm))
            elapsed = time.perf_counter() - tick
            time.sleep(max(0.0, period - elapsed))

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
            window: list[np.ndarray] = []
            with self.queue_lock:
                if self.action_queue:
                    timed_action = self.action_queue.popleft()
                    action = timed_action
                    # Receding-horizon window: current action plus the next few
                    # pending ones (peeked, not popped — they are re-sent with
                    # shifted times next tick until actually executed).
                    window = [timed_action.arm] + [
                        a.arm for a in list(self.action_queue)[: self.lookahead - 1]
                    ]
                    with self.latest_step_lock:
                        self.latest_executed_step = timed_action.timestep

            t_pop = time.perf_counter()

            t_send = t_grip = t_log = t_pop
            if action is not None:
                grip = action.gripper

                # Optional per-step low-pass filter (applied to the executed point)
                if self._use_filter:
                    window[0] = np.array([self._arm_filters[j](window[0][j]) for j in range(6)])
                    grip = self._grip_filter(grip)

                if self.control_interface == "forward_position":
                    # Hand the new target to the servoj streamer thread: it
                    # interpolates from the previous target over dt at stream_hz.
                    frm = self._last_target if self._last_target is not None else window[0]
                    with self._segment_lock:
                        self._segment = (time.perf_counter(), self.dt, frm, window[0])
                    self._last_target = window[0]
                else:
                    # Finite-difference velocities so the spline passes through
                    # each point at speed instead of stopping. Last point gets
                    # zero velocity: it is superseded before being reached, and
                    # JTC rejects nonzero end velocity by default.
                    velocities = None
                    if len(window) > 1:
                        diffs = [(window[i + 1] - window[i]) / self.dt for i in range(len(window) - 1)]
                        velocities = diffs + [np.zeros(6)]
                    self.sensor.send_window_scaled_joint(window, dt=self.dt, velocities_window=velocities)
                t_send = time.perf_counter()

                # Gripper with deadband
                if abs(grip - self._last_grip_sent) > GRIPPER_THRESHOLD:
                    self.sensor.send_gripper_command(float(grip))
                    self._last_grip_sent = grip
                    self._log_grip_t.append(time.perf_counter())
                t_grip = time.perf_counter()

                meas = self.sensor.get_joint_state()
                self._log_t.append(time.perf_counter())
                self._log_step.append(action.timestep)
                self._log_cmd.append(window[0].copy())
                self._log_meas.append(
                    meas[[5, 0, 1, 2, 3, 4]].copy() if meas is not None else np.full(6, np.nan)
                )
                t_log = time.perf_counter()

            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            sleep_target = max(0.0, self.dt - elapsed)
            time.sleep(sleep_target)
            t_wake = time.perf_counter()
            self._log_prof.append((
                t_pop - loop_start,      # pop (incl. lock wait)
                t_send - t_pop,          # send_window_scaled_joint
                t_grip - t_send,         # gripper
                t_log - t_grip,          # joint-state read + log append
                sleep_target,            # requested sleep
                t_wake - loop_start - elapsed - sleep_target,  # sleep overshoot
            ))

    def run(self, num_steps: int = 1000):
        """Start both threads and run for `num_steps` control iterations."""
        assert self.client.ping(), "GR00T server not reachable"

        # High-rate /joint_states recording for diagnostics
        self.sensor.js_log = []
        self.sensor.js_recording = True

        receiver = threading.Thread(target=self._action_receiver_loop, daemon=True)
        receiver.start()

        if self.control_interface == "forward_position":
            streamer = threading.Thread(target=self._fpc_streamer_loop, daemon=True)
            streamer.start()

        try:
            self._control_loop(num_steps)
        finally:
            self.shutdown_event.set()
            receiver.join(timeout=2.0)
            self.sensor.js_recording = False
            if self._log_t:
                np.savez(
                    "streaming_log.npz",
                    t=np.array(self._log_t),
                    step=np.array(self._log_step),
                    cmd=np.array(self._log_cmd),
                    meas=np.array(self._log_meas),
                    prof=np.array(self._log_prof),
                    grip_t=np.array(self._log_grip_t),
                    js=np.array(self.sensor.js_log) if self.sensor.js_log else np.zeros((0, 7)),
                )
                print(f"Diagnostic log saved: streaming_log.npz ({len(self._log_t)} ticks)")
                prof = np.array(self._log_prof)
                names = ["pop", "send", "grip", "log", "sleep_req", "sleep_over"]
                print("per-tick section times (ms): median / p95 / max")
                for i, n in enumerate(names):
                    col = prof[:, i] * 1000
                    print(f"  {n:10s} {np.median(col):7.1f} / {np.percentile(col, 95):7.1f} / {col.max():7.1f}")


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
    # Number of queued actions sent per goal (current + lookahead-1 peeked ahead)
    lookahead: int = 4
    # "forward_position": servoj streaming via forward_position_controller at
    # stream_hz (smooth; requires that controller to be active).
    # "jtc_topic": multi-point windows on the scaled JTC command topic.
    control_interface: Literal["forward_position", "jtc_topic"] = "forward_position"
    stream_hz: float = 125.0
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


def home_via_fpc(sensor: UR5SensorNode, duration: float = 4.0, hz: float = 125.0):
    """Move to home by streaming a smoothstep interpolation to forward_position_controller."""
    state = sensor.get_joint_state()
    cur = state[[5, 0, 1, 2, 3, 4]].astype(float)
    target = np.asarray(HOME_JOINT_POSITIONS, dtype=float)
    n = max(1, int(duration * hz))
    for i in range(1, n + 1):
        a = i / n
        s = 3 * a * a - 2 * a * a * a  # smoothstep: zero velocity at both ends
        sensor.publish_fpc(cur + s * (target - cur))
        time.sleep(1.0 / hz)


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
    if args.control_interface == "forward_position":
        home_via_fpc(sensor)
    else:
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
        lookahead=args.lookahead,
        control_interface=args.control_interface,
        stream_hz=args.stream_hz,
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

    # First streamed segment starts from the home pose we just reached
    client._last_target = np.asarray(HOME_JOINT_POSITIONS, dtype=float)

    try:
        client.run(num_steps=args.num_steps)
    finally:
        sensor.stop()
        sensor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
