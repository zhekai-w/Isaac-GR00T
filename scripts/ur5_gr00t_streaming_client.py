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

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pynput import keyboard
import tyro

from filter_utils import AGGREGATE_FUNCTIONS, ButterworthLPF, OneEuroFilter, apply_chunk_filter
from gr00t.eval.robot import RobotInferenceClient
from ur5_client_common import (
    GRIPPER_THRESHOLD,
    HOME_JOINT_POSITIONS,
    UR5SensorNode,
    build_obs_dict,
    home_via_fpc,
    init_sensor_and_wait,
    shutdown_sensor,
)


@dataclass
class TimedAction:
    timestep: int
    arm: np.ndarray   # (6,)
    gripper: float


def _sample_segment(seg, t: float, interp: str = "hermite", brake_time: float = 0.05):
    """Evaluate the current setpoint segment at wall time `t`.

    seg = (t0, dur, p0, v0, p1, v1). Returns (position, velocity).

    "linear": the original lerp — piecewise-constant velocity, holds at p1 once
    the segment expires. Kept bit-for-bit so it can serve as the A/B control.

    "hermite": cubic Hermite through (p0, v0) -> (p1, v1), so velocity is
    continuous across waypoints (the next segment inherits this one's endpoint
    position *and* derivative). Past the segment end a constant-deceleration
    brake tail runs for `brake_time` seconds instead of extrapolating the cubic,
    which would run away; the tail is C1 at u=0 and settles to a fixed point, so
    the pause path still holds position by republishing.
    """
    t0, dur, p0, v0, p1, v1 = seg
    s = (t - t0) / dur

    if interp == "linear":
        a = min(1.0, max(0.0, s))
        vel = (p1 - p0) / dur if 0.0 <= s < 1.0 else np.zeros_like(p0)
        return p0 + a * (p1 - p0), vel

    if s <= 0.0:
        return p0.copy(), v0.copy()
    if s < 1.0:
        s2 = s * s
        s3 = s2 * s
        pos = ((2 * s3 - 3 * s2 + 1) * p0
               + (s3 - 2 * s2 + s) * dur * v0
               + (-2 * s3 + 3 * s2) * p1
               + (s3 - s2) * dur * v1)
        vel = ((6 * s2 - 6 * s) * (p0 - p1) / dur
               + (3 * s2 - 4 * s + 1) * v0
               + (3 * s2 - 2 * s) * v1)
        return pos, vel

    # Overrun (control loop late, or paused): brake to rest from v1.
    if brake_time <= 0.0:
        return p1.copy(), np.zeros_like(p1)
    u = (s - 1.0) * dur
    if u < brake_time:
        return p1 + v1 * (u - u * u / (2.0 * brake_time)), v1 * (1.0 - u / brake_time)
    return p1 + v1 * (brake_time / 2.0), np.zeros_like(p1)


def _sleep_until(deadline: float, slack: float = 0.005) -> float:
    """Sleep until `deadline` (perf_counter), returning the signed error.

    time.sleep() overshoots by however long GIL reacquisition takes — measured at
    22 ms here, against a 70 ms period. Busy-waiting the last `slack` seconds makes
    the wake independent of that. slack=0 = plain sleep.
    """
    if slack > 0.0:
        rough = deadline - slack - time.perf_counter()
        if rough > 0.0:
            time.sleep(rough)
        while time.perf_counter() < deadline:
            pass
    else:
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
    return time.perf_counter() - deadline




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
        chunk_size_threshold: float = 0.2,
        dt: float = 0.05,
        wait: bool = False,
        lookahead: int = 30,
        control_interface: str = "jtc_topic",
        jtc_end_velocity: str = "continue",
        jtc_absolute_timing: bool = True,
        use_speed_scaling: bool = True,
        stream_hz: float = 125.0,
        tick_slack: float = 0.005,
        aggregate_fn_name: str = "ramp",
        segment_interp: str = "hermite",
        hermite_tension: float = 1.0,
        hermite_monotone: bool = True,
        brake_time: float = 0.05,
        max_tangent_vel: float = 1.5,
        stream_filter: str = "butter",
        stream_filter_cutoff: float = 8.0,
        stream_filter_beta: float = 0.05,
        chunk_filter: str = "none",
        chunk_filter_q: float = 1e-3,
        chunk_filter_r: float = 1e-4,
        chunk_filter_window: int = 7,
        chunk_filter_polyorder: int = 3,
        filter: bool = False,
        filter_mincutoff: float = 1.0,
        filter_beta: float = 0.1,
        log: bool = False,
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
        self.jtc_end_velocity = jtc_end_velocity
        # Absolute waypoint schedule for jtc_topic (see _window_times).
        self.jtc_absolute_timing = jtc_absolute_timing
        self._sched_anchor: tuple[int, float] | None = None
        self._min_lead = 0.005      # s; a point due sooner than this is dropped
        self._sched_period = 0.0    # step period the anchor was built with
        self.use_speed_scaling = use_speed_scaling
        self._n_late_drop = 0
        self.stream_hz = stream_hz
        self.tick_slack = tick_slack
        self.aggregate_fn_name = aggregate_fn_name
        # None = "ramp", handled positionally in _aggregate_into_queue
        self.aggregate_fn = AGGREGATE_FUNCTIONS[aggregate_fn_name]

        # forward_position (servoj) streaming: the control loop sets a target
        # segment per tick; a high-rate thread interpolates along it and
        # publishes setpoints to the forward_position_controller.
        self.segment_interp = segment_interp
        self.hermite_tension = hermite_tension
        self.hermite_monotone = hermite_monotone
        self.brake_time = brake_time
        self.max_tangent_vel = max_tangent_vel
        self._segment_lock = threading.Lock()
        # (t0, dur, p0, v0, p1, v1) — v0/v1 unused in "linear" mode.
        self._segment: tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._last_target: np.ndarray | None = None
        # Set on resume-from-pause: the next segment re-anchors on measured state.
        self._reseed_stream = False

        # Output-stage low-pass, applied to the dense stream_hz setpoint stream
        # right before publish_fpc. This is the moveit_servo ButterworthFilterPlugin
        # position: filtering *after* the setpoint generator, not on sparse waypoints.
        self._stream_filter = stream_filter
        self._stream_filter_cutoff = stream_filter_cutoff
        self._stream_filter_beta = stream_filter_beta
        self._stream_filter_lock = threading.Lock()
        self._stream_lpf_obj = (
            ButterworthLPF(6, stream_hz, stream_filter_cutoff)
            if stream_filter == "butter" else None
        )
        self._stream_euro: list[OneEuroFilter] | None = None
        if stream_filter == "oneeuro":
            self._build_stream_euro()

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

        # Keyboard control flags (s=start, p=pause, h=home, q=quit)
        self.inferring = False
        self.returning_home = False
        self.quit_flag = False
        # Bumped on every home/reset so an in-flight inference started before
        # the reset cannot merge its (now stale) timesteps into the fresh queue.
        self._epoch = 0

        # Per-step filter, control-loop side. Incompatible with the dense output
        # filter: it rewrites window[0] but not window[1], so the Hermite tangent
        # would mix filtered and unfiltered points into a spurious velocity.
        if filter and stream_filter != "none":
            print(f"[WARN] --filter ignored: superseded by --stream-filter {stream_filter}")
            filter = False
        self._use_filter = filter
        self._filter_freq = 1.0 / dt
        self._filter_mincutoff = filter_mincutoff
        self._filter_beta = filter_beta
        if filter:
            self._reset_filters()

        # Last sent gripper (for threshold deadband)
        self._last_grip_sent = 0.0
        self.log = log

        # Diagnostic log: per-tick wall time, timestep, commanded and measured positions
        self._log_t: list[float] = []
        self._log_step: list[int] = []
        self._log_cmd: list[np.ndarray] = []
        self._log_meas: list[np.ndarray] = []
        self._log_prof: list[tuple] = []
        self._log_grip_t: list[float] = []
        # Published setpoint stream at stream_hz. _log_cmd above is the *waypoint*
        # stream (control-loop rate) and is blind to everything the interpolator
        # and the output filter do, so smoothness has to be measured here.
        self._log_merge: list[tuple] = []
        self._log_stream_t: list[float] = []
        self._log_stream_cmd: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Observation / Inference
    # ------------------------------------------------------------------

    def _build_obs_dict(self, img1, img2, state):
        return build_obs_dict(img1, img2, state, self.lang)

    def _get_action_chunk(self, img1, img2, state, base_step):
        """Blocking ZMQ inference; returns chunk-smoothed list[TimedAction] aligned to base_step."""
        obs = self._build_obs_dict(img1, img2, state)
        raw = self.client.get_action(obs)

        # Extract full chunk arrays (H, 6) and (H,)
        arm_all = np.array([np.atleast_1d(raw["action.ur5_arm"][i]) for i in range(self.action_horizon)])
        grip_all = np.array([np.atleast_1d(raw["action.gripper"][i])[0] for i in range(self.action_horizon)])

        arm_all, grip_all = apply_chunk_filter(
            arm_all, grip_all, self.chunk_filter, self.dt,
            q=self.chunk_filter_q, r=self.chunk_filter_r,
            window=self.chunk_filter_window, polyorder=self.chunk_filter_polyorder)

        timed_actions = []
        for i in range(self.action_horizon):
            timed_actions.append(TimedAction(timestep=base_step + i, arm=arm_all[i], gripper=grip_all[i]))
        return timed_actions

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _aggregate_into_queue(self, incoming: list[TimedAction]):
        """Merge *incoming* into the queue, blending overlapping timesteps.

        Fixed-weight functions apply the same old/new mix everywhere in the overlap,
        so the queue steps at BOTH ends of it on every merge. "ramp" sweeps 0 -> 1
        across the overlap instead: continuous with what is executing and with the
        pure-new region beyond. Chunk filtering runs before this, so nothing
        downstream smooths those steps.
        """
        with self.queue_lock:
            with self.latest_step_lock:
                latest = self.latest_executed_step

            # Only pending actions remain (not yet dispatched)
            existing = {a.timestep: a for a in self.action_queue if a.timestep > latest}

            fresh = [a for a in incoming if a.timestep > latest]
            n_dropped = len(incoming) - len(fresh)
            overlap = sorted(a.timestep for a in fresh if a.timestep in existing)
            n_overlap = len(overlap)
            ramp_pos = {ts: (i + 1) / (n_overlap + 1) for i, ts in enumerate(overlap)}

            for new_a in fresh:
                old = existing.get(new_a.timestep)
                if old is None:
                    existing[new_a.timestep] = new_a
                    continue
                if self.aggregate_fn is None:      # "ramp"
                    a = ramp_pos[new_a.timestep]
                    arm = (1.0 - a) * old.arm + a * new_a.arm
                    grip = (1.0 - a) * old.gripper + a * new_a.gripper
                else:
                    arm = self.aggregate_fn(old.arm, new_a.arm)
                    grip = self.aggregate_fn(old.gripper, new_a.gripper)
                existing[new_a.timestep] = TimedAction(
                    timestep=new_a.timestep, arm=arm, gripper=float(grip))

            self.action_queue = deque(sorted(existing.values(), key=lambda a: a.timestep))
            self.max_queue_size = max(self.max_queue_size, len(self.action_queue))

        # Merge events, for correlating roughness against chunk boundaries.
        if self.sensor.js_recording:
            self._log_merge.append((time.perf_counter(), latest, n_overlap,
                                    len(fresh) - n_overlap, n_dropped))

    # ------------------------------------------------------------------
    # Absolute waypoint schedule (jtc_topic)
    # ------------------------------------------------------------------

    def _step_period(self) -> float:
        """Wall-clock seconds per waypoint, corrected for controller speed scaling.

        JTC advances its clock as `traj_time_ += period * scaling_factor_`, so at 50%
        a waypoint takes 2*dt of wall time. Both the tick period and the schedule must
        stretch: scaling only the schedule outruns the arm, scaling only the period
        leaves the controller a schedule it cannot meet.
        """
        if not self.use_speed_scaling:
            return self.dt
        return self.dt / min(1.0, max(0.05, float(self.sensor.speed_scaling)))

    def _window_times(self, ts0: int, window: list, now: float):
        """Slice `window` and time each point against a fixed schedule.

        Waypoint `ts` is due at ``anchor_wall + (ts - anchor_step) * period``, decided
        once. The alternative — ``(i+1)*dt`` from the publish instant — hands the
        trajectory clock to whenever this thread woke, so tick jitter becomes velocity
        modulation. Here a late publish only shortens the first segment.

        Already-due points are dropped, not clamped: clamping re-stretches the
        timeline, which is what this exists to avoid.
        """
        period = self._step_period()
        # Re-anchor when the speed factor moves: the anchor encodes a rate, so
        # applying a new rate to an old anchor would jump the whole schedule.
        if (self._sched_anchor is None
                or abs(period - self._sched_period) > 1e-6):
            self._sched_anchor = (ts0, now + period)
        a_step, a_wall = self._sched_anchor
        self._sched_period = period

        due = [a_wall + (ts0 + i - a_step) * period for i in range(len(window))]
        if due[-1] - now < self._min_lead:
            # More than a whole window behind (long inference stall, pause/resume
            # race). Chasing that would fire catch-up motion; restart the clock.
            self._sched_anchor = (ts0, now + period)
            a_step, a_wall = self._sched_anchor
            due = [a_wall + (ts0 + i - a_step) * period for i in range(len(window))]

        keep = next(i for i, d in enumerate(due) if d - now >= self._min_lead)
        self._n_late_drop += keep
        return window[keep:], [due[i] - now for i in range(keep, len(due))]

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
            # Paused or homing: no inference, no queue growth.
            if not self.inferring or self.returning_home:
                time.sleep(0.02)
                continue

            should_infer = self._queue_depleted()
            force = self.must_go.is_set() and self._queue_empty()

            if not (should_infer or force):
                time.sleep(0.001)
                continue

            # Base step = next timestep the robot hasn't started yet.
            # Because the queue tracks precisely what's pending, we don't need
            # wall-clock estimation — the queue itself is the ground truth.
            epoch = self._epoch
            with self.latest_step_lock:
                base_step = self.latest_executed_step + 1

            img1 = self.sensor.get_azure_kinect_image()
            img2 = self.sensor.get_wfov_image()
            state = self.sensor.get_joint_state()
            if state is None or img1 is None or img2 is None:
                time.sleep(0.01)
                continue
            # get_joint_state() is already canonical [pan, lift, elbow, w1..w3, grip]
            # (callback keys /joint_states by name) — no reindex needed.

            try:
                timed_actions = self._get_action_chunk(img1, img2, state, base_step)
            except Exception:
                time.sleep(0.1)
                continue

            # A home/pause landed while this inference was in flight — its
            # timesteps refer to a queue generation that no longer exists.
            if self._epoch != epoch:
                continue

            self._aggregate_into_queue(timed_actions)

            if force:
                self.must_go.clear()
            self.must_go.set()

    # ------------------------------------------------------------------
    # Thread 3 — forward_position (servoj) streamer
    # ------------------------------------------------------------------

    def _build_stream_euro(self):
        self._stream_euro = [
            OneEuroFilter(self.stream_hz, 1.0, self._stream_filter_beta)
            for _ in range(6)
        ]

    def _reset_stream_filter(self, x0):
        """Seed the output filter to x0 so the next published sample doesn't kick."""
        x0 = np.asarray(x0, dtype=float)
        with self._stream_filter_lock:
            if self._stream_filter == "butter":
                self._stream_lpf_obj.reset(x0)
            elif self._stream_filter == "oneeuro":
                self._build_stream_euro()
                for j in range(6):
                    self._stream_euro[j](float(x0[j]))  # prime

    def _stream_lpf(self, q: np.ndarray) -> np.ndarray:
        with self._stream_filter_lock:
            if self._stream_filter == "butter":
                return self._stream_lpf_obj(q)
            if self._stream_filter == "oneeuro":
                return np.array([self._stream_euro[j](float(q[j])) for j in range(6)])
        return q

    def _outgoing_velocity(self, p0, p1, p2):
        """Catmull-Rom tangent at p1 with Fritsch-Carlson monotone limiting.

        p2 is the next pending waypoint; without one, coast at half chord speed.
        Only the OUTGOING tangent is limited — clipping the inherited v0 would
        re-break the C1 continuity this exists to provide, and v0 is already bounded
        (it was a limited v1 when created).
        """
        d1 = (p1 - p0) / self.dt
        if p2 is None:
            return np.clip(0.5 * d1, -self.max_tangent_vel, self.max_tangent_vel)
        d2 = (p2 - p1) / self.dt
        v = self.hermite_tension * 0.5 * (d1 + d2)
        if self.hermite_monotone:
            lim = 3.0 * np.minimum(np.abs(d1), np.abs(d2))
            v = np.where(np.sign(d1) != np.sign(d2), 0.0, np.clip(v, -lim, lim))
        return np.clip(v, -self.max_tangent_vel, self.max_tangent_vel)

    def _fpc_streamer_loop(self):
        """Publish interpolated setpoints at stream_hz along the current segment.

        moveit_servo pattern: servoj consumes a dense stream and its lookahead_time
        smooths the remaining discretization.
        Per tick: sample segment -> output low-pass -> publish_fpc.
        """
        period = 1.0 / self.stream_hz
        deadline = time.perf_counter() + period
        while not self.shutdown_event.is_set():
            tick = time.perf_counter()
            with self._segment_lock:
                seg = self._segment
            if seg is not None:
                q, _ = _sample_segment(seg, tick, self.segment_interp, self.brake_time)
                q = self._stream_lpf(q)
                self.sensor.publish_fpc(q)
                if self.sensor.js_recording:
                    self._log_stream_t.append(tick)
                    self._log_stream_cmd.append(np.asarray(q, dtype=float).copy())
            # Same absolute-deadline scheduling as the control loop, and it matters
            # more here: at 125 Hz the period is 8 ms, so the GIL-reacquisition
            # overshoot that cost the control loop 30% would swallow this one whole.
            _sleep_until(deadline, min(self.tick_slack, 0.5 * period))
            deadline += period
            if time.perf_counter() > deadline:
                deadline = time.perf_counter() + period

    # ------------------------------------------------------------------
    # Keyboard control
    # ------------------------------------------------------------------

    def _on_press(self, key):
        try:
            ch = key.char
        except AttributeError:
            return
        if ch == 's':
            # Re-anchor on measured state: during the pause the arm may have been
            # freedriven, so the held setpoint no longer matches reality and
            # resuming without a reseed would snap it back.
            self._reseed_stream = True
            self._sched_anchor = None   # schedule restarts from the resume instant
            self.inferring = True
            self.sensor.js_recording = self.log
            print("[KB] Inference started")
        elif ch == 'p':
            self.inferring = False
            self.sensor.js_recording = False
            print("[KB] Inference paused")
        elif ch == 'h':
            self.inferring = False
            self.returning_home = True
            print("[KB] Returning home")
        elif ch == 'q':
            self.quit_flag = True
            self.shutdown_event.set()
            print("[KB] Quit requested")

    # ------------------------------------------------------------------
    # Home / reset
    # ------------------------------------------------------------------

    def _reset_filters(self):
        self._arm_filters = [
            OneEuroFilter(self._filter_freq, self._filter_mincutoff, self._filter_beta)
            for _ in range(6)
        ]
        self._grip_filter = OneEuroFilter(
            self._filter_freq, self._filter_mincutoff, self._filter_beta
        )

    def go_home(self):
        """Drop all pending actions, move to home, reset timestep bookkeeping.

        Called from the control loop, so nothing else is driving the arm; the
        segment is cleared first so the fpc streamer thread stops publishing
        while home_via_fpc owns the setpoint stream.
        """
        with self._segment_lock:
            self._segment = None
        with self.queue_lock:
            self.action_queue.clear()
        with self.latest_step_lock:
            self.latest_executed_step = -1
        self._sched_anchor = None       # timesteps restart at 0, so must the clock
        self._epoch += 1
        self.max_queue_size = self.action_horizon
        self.must_go.set()

        if self.control_interface == "forward_position":
            home_via_fpc(self.sensor)
        else:
            self.sensor.send_single_action_scaled_joint(HOME_JOINT_POSITIONS, dt=3.0, wait=True)
        self.sensor.send_gripper_command(0.0)

        # Anchor on where the arm actually ended up, not on the nominal home pose:
        # a home move that did not fully converge would otherwise show up as a jump
        # on the first segment after the reset.
        meas = self.sensor.get_joint_state()
        home = (meas[:6].astype(float) if meas is not None
                else np.asarray(HOME_JOINT_POSITIONS, dtype=float))
        self._last_target = home
        self._reset_stream_filter(home)
        self._reseed_stream = False
        self._last_grip_sent = 0.0
        if self._use_filter:
            self._reset_filters()

    # ------------------------------------------------------------------
    # Thread 2 — Control Loop (main)
    # ------------------------------------------------------------------

    def _control_loop(self, num_steps: int):
        """Main thread: pops one TimedAction per `dt` and streams to hardware."""
        self.start_barrier.wait()

        executed = 0
        # Absolute deadlines: `sleep(dt - elapsed)` can only make a tick longer, so
        # overshoot becomes permanent playback stretch. A deadline lets the next tick
        # absorb it.
        deadline = time.perf_counter() + self._step_period()
        while executed < num_steps:
            if self.shutdown_event.is_set() or self.quit_flag:
                break

            if self.returning_home:
                self.go_home()
                self.returning_home = False
                print("Home pose reached. Press 's' to resume.")
                deadline = time.perf_counter() + self._step_period()
                continue

            if not self.inferring:
                # Paused: hold position. The fpc streamer keeps republishing the
                # last segment, so the arm stays where it is.
                time.sleep(0.05)
                deadline = time.perf_counter() + self._step_period()
                continue

            executed += 1
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
                    tgt = window[0]
                    nxt = window[1] if len(window) > 1 else None
                    now = time.perf_counter()
                    with self._segment_lock:
                        seg = self._segment
                        if seg is None or self.segment_interp == "linear":
                            p0 = self._last_target if self._last_target is not None else tgt
                            v0 = np.zeros(6)
                        else:
                            # Sample the live segment rather than trusting bookkeeping:
                            # this makes C0/C1 continuity exact regardless of how late
                            # or jittery this tick was.
                            p0, v0 = _sample_segment(seg, now, self.segment_interp, self.brake_time)
                        if self._reseed_stream:
                            meas = self.sensor.get_joint_state()
                            if meas is not None:
                                p0, v0 = meas[:6].astype(float), np.zeros(6)
                                self._reset_stream_filter(p0)
                            self._reseed_stream = False
                        v1 = (self._outgoing_velocity(p0, tgt, nxt)
                              if self.segment_interp == "hermite" else np.zeros(6))
                        # Segment duration must equal the tick period, or the
                        # interpolator finishes early and runs its brake tail.
                        self._segment = (now, self._step_period(), p0, v0, tgt, v1)
                    self._last_target = tgt
                else:
                    # Finite-difference velocities so the spline passes through each
                    # point at speed. jtc_end_velocity "zero" commands a stop at the
                    # window end on EVERY republish — a velocity modulation locked to
                    # the replan rate. "continue" carries the last chord velocity.
                    win, times = window, None
                    if self.jtc_absolute_timing:
                        win, times = self._window_times(
                            action.timestep, window, time.perf_counter())
                    velocities = None
                    if len(win) > 1:
                        diffs = [(win[i + 1] - win[i]) / self.dt for i in range(len(win) - 1)]
                        end_v = np.zeros(6) if self.jtc_end_velocity == "zero" else diffs[-1]
                        velocities = diffs + [end_v]
                    self.sensor.send_window_scaled_joint(
                        win, dt=self.dt, velocities_window=velocities, times_window=times)
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
                    meas[:6].copy() if meas is not None else np.full(6, np.nan)
                )
                t_log = time.perf_counter()

            # Maintain control frequency against the absolute schedule.
            sleep_target = deadline - time.perf_counter()
            tick_err = _sleep_until(deadline, self.tick_slack)
            deadline += self._step_period()
            now = time.perf_counter()
            if now > deadline:
                # Fell behind by more than a whole tick (long inference stall, a
                # blocking send). Re-anchor rather than firing catch-up ticks
                # back-to-back, which would play the trajectory *faster* than dt.
                deadline = now + self._step_period()
            self._log_prof.append((
                t_pop - loop_start,      # pop (incl. lock wait)
                t_send - t_pop,          # send_window_scaled_joint
                t_grip - t_send,         # gripper
                t_log - t_grip,          # joint-state read + log append
                sleep_target,            # requested sleep
                tick_err,                # deadline error (was: sleep overshoot)
            ))

    def run(self, num_steps: int = 1000):
        """Start both threads and run for `num_steps` control iterations."""
        assert self.client.ping(), "GR00T server not reachable"

        # High-rate /joint_states recording for diagnostics (gated by s/p)
        self.sensor.js_log = []
        self.sensor.js_recording = False

        kb_listener = keyboard.Listener(on_press=self._on_press)
        kb_listener.start()
        print("Keyboard ready: s=start  p=pause  h=home  q=quit")

        receiver = threading.Thread(target=self._action_receiver_loop, daemon=True)
        receiver.start()

        if self.control_interface == "forward_position":
            streamer = threading.Thread(target=self._fpc_streamer_loop, daemon=True)
            streamer.start()

        try:
            self._control_loop(num_steps)
        finally:
            self.shutdown_event.set()
            kb_listener.stop()
            receiver.join(timeout=2.0)
            self.sensor.js_recording = False
            if self.log and self._log_t:
                np.savez(
                    "streaming_log.npz",
                    t=np.array(self._log_t),
                    step=np.array(self._log_step),
                    cmd=np.array(self._log_cmd),
                    meas=np.array(self._log_meas),
                    prof=np.array(self._log_prof),
                    grip_t=np.array(self._log_grip_t),
                    js=np.array(self.sensor.js_log) if self.sensor.js_log else np.zeros((0, 7)),
                    merge=(np.array(self._log_merge) if self._log_merge
                           else np.zeros((0, 5))),
                    aggregate_fn_name=np.array(self.aggregate_fn_name),
                    stream_t=np.array(self._log_stream_t),
                    stream_cmd=(np.array(self._log_stream_cmd)
                                if self._log_stream_cmd else np.zeros((0, 6))),
                    # Config stamp so a saved log is self-identifying in A/B plots.
                    segment_interp=np.array(self.segment_interp),
                    stream_filter=np.array(self._stream_filter),
                    control_interface=np.array(self.control_interface),
                    jtc_end_velocity=np.array(self.jtc_end_velocity),
                    jtc_absolute_timing=np.array(self.jtc_absolute_timing),
                    speed_scaling=np.array(self.sensor.speed_scaling),
                    stream_filter_cutoff=np.array(self._stream_filter_cutoff),
                    dt=np.array(self.dt),
                    stream_hz=np.array(self.stream_hz),
                )
                # The dense setpoint stream only exists on the forward_position path;
                # under jtc_topic the controller interpolates, so there is nothing to
                # sample and 0 is the expected count.
                extra = (f", {len(self._log_stream_t)} stream samples"
                         if self.control_interface == "forward_position"
                         else " (no dense stream in jtc_topic mode)")
                print(f"Diagnostic log saved: streaming_log.npz "
                      f"({len(self._log_t)} ticks{extra})")
                if self.jtc_absolute_timing:
                    print(f"waypoints dropped as already-due: {self._n_late_drop} "
                          f"(schedule kept, frames dropped — nonzero means the loop "
                          f"could not keep up with dt)")
                prof = np.array(self._log_prof)
                names = ["pop", "send", "grip", "log", "sleep_req", "tick_err"]
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
    # Infer when the queue drops to this fraction of its high-water mark. Lower =
    # fewer merges, smaller overlap, staler observations. 0.4 -> 0.2 with ramp took
    # commanded rms jerk 38.9 -> 4.5.
    chunk_size_threshold: float = 0.2
    dt: float = 0.05
    wait: bool = False
    # Waypoints per published trajectory (current + lookahead-1 peeked). Capped by
    # what is pending, so > action_horizon means "the whole queue". This is the
    # runway the controller keeps interpolating if a tick is late.
    lookahead: int = 30
    # "forward_position": servoj streaming via forward_position_controller at
    # stream_hz (smooth; requires that controller to be active).
    # "jtc_topic": multi-point windows on the scaled JTC command topic.
    control_interface: Literal["forward_position", "jtc_topic"] = "jtc_topic"
    stream_hz: float = 125.0
    # Busy-wait tail per tick (s). 0 = plain sleep (old behavior, for A/B).
    tick_slack: float = 0.005
    # GIL hold time before yielding. Default 0.005 is the same order as tick_err.
    gil_switch_interval: float = 0.001
    aggregate_fn_name: Literal["ramp", "weighted_average", "latest_only", "average", "conservative"] = "ramp"

    # Setpoint generation between waypoints (forward_position only).
    # "linear" reproduces the original lerp exactly — use it as the A/B control.
    # "hermite" is C1 across waypoints, so velocity no longer steps every dt.
    segment_interp: Literal["linear", "hermite"] = "hermite"
    hermite_tension: float = 1.0     # 1.0 = Catmull-Rom, 0.0 = stop at each waypoint
    hermite_monotone: bool = True    # Fritsch-Carlson limiting on the outgoing tangent
    brake_time: float = 0.05         # s, deceleration tail when a segment overruns
    max_tangent_vel: float = 1.5     # rad/s spline-sanity clip (UR5 hw limit 3.1416)

    # Output-stage low-pass on the dense stream_hz setpoint stream. This is where
    # moveit_servo puts its ButterworthFilterPlugin; filtering at waypoint rate
    # instead (see --filter) gets undone by the interpolator.
    stream_filter: Literal["none", "butter", "oneeuro"] = "butter"
    stream_filter_cutoff: float = 8.0   # Hz; group delay ~ sqrt(2)/(2*pi*fc) = 28 ms
    stream_filter_beta: float = 0.05    # oneeuro only

    # jtc_topic only: velocity of the LAST point in each republished window.
    # "zero" = old behavior (commands a stop every replan); "continue" =
    # carry the last chord velocity so the spline is not always braking.
    jtc_end_velocity: Literal["zero", "continue"] = "continue"

    # Put time_from_start on a fixed wall clock instead of "(i+1)*dt from now", so
    # a late publish shortens the first segment instead of shifting the whole future.
    jtc_absolute_timing: bool = True

    # Stretch tick period AND schedule by the controller's speed-scaling factor
    # (pendant slider, safety slowdown). See _step_period.
    use_speed_scaling: bool = True

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

    # Write streaming_log.npz for scripts/compare_logs.py (pass --log to enable).
    log: bool = False


def main(args: ArgsConfig):
    if args.gil_switch_interval > 0:
        sys.setswitchinterval(args.gil_switch_interval)

    sensor, spin_thread = init_sensor_and_wait()

    print("Moving to home position...")
    if args.control_interface == "forward_position":
        home_via_fpc(sensor)
    else:
        sensor.send_single_action_scaled_joint(HOME_JOINT_POSITIONS, dt=1.0, wait=False)
    sensor.send_gripper_command(0.0)
    if args.use_speed_scaling and abs(sensor.speed_scaling - 1.0) > 0.01:
        print(f"[WARN] controller speed scaling is {sensor.speed_scaling:.2f} — the arm "
              f"will run at {sensor.speed_scaling*100:.0f}% and the schedule is being "
              f"stretched to match. Check the pendant slider if this is unintended.")
    print("Home position reached. Press 's' to start inference.")

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
        tick_slack=args.tick_slack,
        aggregate_fn_name=args.aggregate_fn_name,
        segment_interp=args.segment_interp,
        hermite_tension=args.hermite_tension,
        hermite_monotone=args.hermite_monotone,
        brake_time=args.brake_time,
        max_tangent_vel=args.max_tangent_vel,
        stream_filter=args.stream_filter,
        stream_filter_cutoff=args.stream_filter_cutoff,
        stream_filter_beta=args.stream_filter_beta,
        jtc_end_velocity=args.jtc_end_velocity,
        jtc_absolute_timing=args.jtc_absolute_timing,
        use_speed_scaling=args.use_speed_scaling,
        chunk_filter=args.chunk_filter,
        chunk_filter_q=args.chunk_filter_q,
        chunk_filter_r=args.chunk_filter_r,
        chunk_filter_window=args.chunk_filter_window,
        chunk_filter_polyorder=args.chunk_filter_polyorder,
        filter=args.filter,
        filter_mincutoff=args.filter_mincutoff,
        filter_beta=args.filter_beta,
        log=args.log,
    )

    _meas = sensor.get_joint_state()
    _start = (_meas[:6].astype(float) if _meas is not None
              else np.asarray(HOME_JOINT_POSITIONS, dtype=float))
    client._last_target = _start
    client._reset_stream_filter(_start)

    try:
        client.run(num_steps=args.num_steps)
    finally:
        shutdown_sensor(sensor)


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
