import collections
import threading
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from gr00t.eval.robot import RobotInferenceClient
from trajectory_msgs.msg import JointTrajectoryPoint
from filter_utils import rts_smoother_chunk, savgol_chunk

# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class TimedAction:
    """A single action tagged with its logical timestep."""
    timestep: int
    action: np.ndarray  # shape (7,) for UR5


@dataclass
class FPSTracker:
    """Tracks actual control loop frequency."""
    target_fps: float
    _count: int = 0
    _start_time: float = None

    def tick(self) -> float:
        self._count += 1
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now
            return 0.0
        elapsed = now - self._start_time
        return (self._count - 1) / elapsed if elapsed > 0 else 0.0


# ==============================================================================
# Aggregate Functions
# ==============================================================================

AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,  # favor newer
    "latest_only":      lambda old, new: new,
    "average":          lambda old, new: 0.5 * (old + new),
    "conservative":     lambda old, new: 0.7 * old + 0.3 * new,  # favor older
    "none":           lambda old, new: old,  # ignore new
}


# ==============================================================================
# Async Client
# ==============================================================================

class Gr00tAsyncClient:
    """
    Two-thread async client for GR00T inference over ZMQ.

    Thread 1 (action_receiver): captures obs, calls ZMQ get_action (blocking),
             merges returned action chunk into a shared queue with temporal blending.
    Thread 2 (control_loop): chunk mode — drains action_horizon actions, applies
             filter, sends as one trajectory; single mode — pops one action per tick.
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        modality_keys=None,
        lang="pick up the object",
        action_horizon=16,
        chunk_size_threshold=0.7,
        dt=0.05,
        aggregate_fn_name="conservative",
        send_mode="chunk",
        chunk_dispatch_size=4,
        chunk_filter="rts",
        chunk_filter_q=1e-3,
        chunk_filter_r=1e-4,
        chunk_filter_window=7,
        chunk_filter_polyorder=3,
        api_token=None,
    ):
        self.client = RobotInferenceClient(host=host, port=port, api_token=api_token)
        self.modality_keys = modality_keys or ["ur5_arm", "gripper"]
        self.lang = lang
        self.action_horizon = action_horizon
        self.chunk_size_threshold = chunk_size_threshold
        self.dt = dt
        self.aggregate_fn = AGGREGATE_FUNCTIONS[aggregate_fn_name]
        self.send_mode = send_mode
        self.chunk_dispatch_size = max(1, min(chunk_dispatch_size, action_horizon))
        self.chunk_filter = chunk_filter
        self.chunk_filter_q = chunk_filter_q
        self.chunk_filter_r = chunk_filter_r
        self.chunk_filter_window = chunk_filter_window
        self.chunk_filter_polyorder = chunk_filter_polyorder

        # Thread synchronization
        self.action_queue = collections.deque()
        self.queue_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.must_go = threading.Event()
        self.must_go.set()  # initially armed
        self.start_barrier = threading.Barrier(2)

        # State tracking
        self.latest_executed_step = -1
        self.latest_step_lock = threading.Lock()
        self.global_timestep = 0
        self.max_queue_size = action_horizon

        # Inference rate tracker
        self._inference_fps = FPSTracker(target_fps=1.0 / dt)
        self._obs_count = 0

        # Robot reference (set in run())
        self.robot = None

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

    def _get_action_chunk(self, img1, img2, state):
        """Send obs via ZMQ (BLOCKS), return list of TimedAction."""
        obs = self._build_obs_dict(img1, img2, state)
        raw = self.client.get_action(obs)
        # raw = {"action.ur5_arm": (16, 6), "action.gripper": (16, 1)}

        base_step = self.global_timestep
        timed_actions = []
        for i in range(self.action_horizon):
            flat = np.concatenate(
                [np.atleast_1d(raw[f"action.{k}"][i]) for k in self.modality_keys],
                axis=0,
            )  # shape (7,)
            timed_actions.append(TimedAction(timestep=base_step + i, action=flat))
        return timed_actions

    # ------------------------------------------------------------------
    # Queue Management
    # ------------------------------------------------------------------

    def _aggregate_into_queue(self, incoming):
        """Merge incoming actions into the queue, blending overlapping timesteps."""
        with self.queue_lock:
            existing = {a.timestep: a for a in self.action_queue}

        with self.latest_step_lock:
            latest = self.latest_executed_step

        for new_a in incoming:
            if new_a.timestep <= latest:
                continue  # already executed
            if new_a.timestep in existing:
                old_action = existing[new_a.timestep].action
                blended = self.aggregate_fn(old_action, new_a.action)
                existing[new_a.timestep] = TimedAction(new_a.timestep, blended)
            else:
                existing[new_a.timestep] = new_a

        with self.queue_lock:
            self.action_queue = collections.deque(
                sorted(existing.values(), key=lambda a: a.timestep)
            )
            self.max_queue_size = max(self.max_queue_size, len(self.action_queue))

    def _queue_depleted(self):
        """Check if queue is low enough to trigger new inference."""
        with self.queue_lock:
            size = len(self.action_queue)
        if self.max_queue_size <= 0:
            return True
        return size / self.max_queue_size <= self.chunk_size_threshold

    def _queue_empty(self):
        with self.queue_lock:
            return len(self.action_queue) == 0

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_chunk_filter(self, arm_chunk: np.ndarray) -> np.ndarray:
        """Filter arm trajectory chunk (H, 6) before sending."""
        H = arm_chunk.shape[0]
        if self.chunk_filter == "rts" and H >= 2:
            return rts_smoother_chunk(arm_chunk, dt=self.dt,
                                      q=self.chunk_filter_q, r=self.chunk_filter_r)
        if self.chunk_filter == "savgol" and H >= self.chunk_filter_window:
            return savgol_chunk(arm_chunk, self.chunk_filter_window, self.chunk_filter_polyorder)
        return arm_chunk  # "none" or too-small sub-chunk

    # ------------------------------------------------------------------
    # Thread 1: Action Receiver
    # ------------------------------------------------------------------

    def _action_receiver_thread(self):
        """Daemon thread: captures obs, calls ZMQ inference, fills queue."""
        self.start_barrier.wait()
        print("[action_receiver] Thread started")

        while not self.shutdown_event.is_set():
            should_infer = self._queue_depleted()
            force = self.must_go.is_set() and self._queue_empty()

            if not (should_infer or force):
                time.sleep(0.001)
                continue

            img1, img2, state = self.robot.get_observation()
            infer_fps = self._inference_fps.tick()
            self._obs_count += 1
            if self._obs_count % 5 == 0:
                print(f"[inference_rate] {infer_fps:.2f} Hz  (camera fps logged separately)")

            try:
                timed_actions = self._get_action_chunk(img1, img2, state)
            except Exception as e:
                print(f"[action_receiver] Inference error: {e}")
                time.sleep(0.1)
                continue

            self._aggregate_into_queue(timed_actions)

            if force:
                self.must_go.clear()
            self.must_go.set()

            print(f"[action_receiver] Chunk merged, queue size: {len(self.action_queue)}")

    # ------------------------------------------------------------------
    # Thread 2: Control Loop
    # ------------------------------------------------------------------

    def _control_loop_thread(self, num_steps):
        """Main thread: dispatches to chunk or single mode."""
        self.start_barrier.wait()
        print(f"[control_loop] Thread started (mode={self.send_mode}, filter={self.chunk_filter})")
        fps = FPSTracker(target_fps=1.0 / self.dt)

        if self.send_mode == "chunk":
            self._run_chunk_mode(num_steps, fps)
        else:
            self._run_single_mode(num_steps, fps)



    def _run_chunk_mode(self, num_steps, fps):
        """Async chunk mode: dispatch small sub-chunks so the queue stays partially
        full. This keeps all three async mechanisms active:
          - early trigger fires while queue still has actions to execute,
          - inference runs on the receiver thread in parallel with dispatch,
          - new chunks land while old ones are still queued → fusion happens.
        """
        sub_size = self.chunk_dispatch_size
        steps_executed = 0
        sub_idx = 0

        while steps_executed < num_steps and not self.shutdown_event.is_set():
            # Wait until a sub-chunk's worth of actions is ready
            while not self.shutdown_event.is_set():
                with self.queue_lock:
                    qsize = len(self.action_queue)
                if qsize >= sub_size:
                    break
                time.sleep(0.001)

            if self.shutdown_event.is_set():
                break

            # Drain sub_size actions (leaves the rest in queue for fusion)
            chunk = []
            with self.queue_lock:
                take = min(sub_size, len(self.action_queue), num_steps - steps_executed)
                for _ in range(take):
                    timed = self.action_queue.popleft()
                    chunk.append(timed.action)
                    with self.latest_step_lock:
                        self.latest_executed_step = timed.timestep
                    self.global_timestep = timed.timestep + 1

            if not chunk:
                continue

            chunk_arr  = np.array(chunk)                              # (S, 7)
            arm_chunk  = self._apply_chunk_filter(chunk_arr[:, :6])   # (S, 6)
            grip_chunk = chunk_arr[:, 6].tolist()                     # list of S floats

            dispatch_start = time.perf_counter()
            self.robot.send_chunk(arm_chunk, grip_chunk)
            steps_executed += len(chunk)
            sub_idx += 1

            # Pace at the sub-chunk's wall-clock duration so subsequent
            # dispatches preempt the controller at the right cadence and the
            # receiver thread has time to fuse fresh chunks into the queue.
            elapsed = time.perf_counter() - dispatch_start
            time.sleep(max(0.0, len(chunk) * self.dt - elapsed))

            if sub_idx % 5 == 0:
                print(f"[control_loop] Sub-chunk {sub_idx} "
                      f"({steps_executed}/{num_steps} steps), FPS: {fps.tick():.1f}")

    def _run_single_mode(self, num_steps, fps):
        """Pop one action per tick at fixed rate."""
        for step in range(num_steps):
            if self.shutdown_event.is_set():
                break
            loop_start = time.perf_counter()

            action = None
            with self.queue_lock:
                if self.action_queue:
                    timed_action = self.action_queue.popleft()
                    action = timed_action.action
                    with self.latest_step_lock:
                        self.latest_executed_step = timed_action.timestep
                    self.global_timestep = timed_action.timestep + 1

            if action is not None:
                self.robot.send_action(action)

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, self.dt - elapsed))

            if step % 100 == 0:
                measured_fps = fps.tick()
                print(f"[control_loop] Step {step}, FPS: {measured_fps:.1f}")

    def run(self, robot, num_steps=1000):
        """
        Start async inference loop.

        Args:
            robot: Object with:
                   - get_observation() -> (img1_azure_kinect, img2_wfov, state)
                   - send_chunk(arm_chunk, grip_chunk) for chunk mode
                   - send_action(action) for single mode
            num_steps: Total control steps. In chunk mode, rounded down to
                       the nearest action_horizon multiple.
        """
        self.robot = robot
        assert self.client.ping(), "Server not reachable"

        receiver = threading.Thread(target=self._action_receiver_thread, daemon=True)
        receiver.start()

        try:
            self._control_loop_thread(num_steps)
        finally:
            self.shutdown_event.set()
            receiver.join(timeout=2.0)
            print("[async_client] Shutdown complete")


@dataclass
class AsyncArgsConfig:
    # Connection
    host: str = "localhost"
    port: int = 5555
    api_token: Optional[str] = None

    # Task
    lang: str = "place the large cube on the orange box."
    num_steps: int = 1000

    # Async core
    action_horizon: int = 16
    chunk_size_threshold: float = 0.5
    dt: float = 0.05
    aggregate_fn_name: Literal["weighted_average", "latest_only", "average", "conservative", "none"] = "conservative"
    send_mode: Literal["single", "chunk"] = "chunk"
    chunk_dispatch_size: int = 4      # chunk mode: # actions per trajectory dispatch
                                      # smaller = more responsive + more fusion; larger = smoother

    # Within-chunk smoothing
    chunk_filter: Literal["none", "savgol", "rts"] = "rts"
    chunk_filter_window: int = 7      # savgol: odd, < action_horizon
    chunk_filter_polyorder: int = 3   # savgol: < chunk_filter_window
    chunk_filter_q: float = 1e-3      # rts: process noise
    chunk_filter_r: float = 1e-4      # rts: measurement noise


def main(args: AsyncArgsConfig):
    import rclpy
    from ur5_gr00t_simple_client import UR5SensorNode, send_gripper_paced, HOME_JOINT_POSITIONS

    class UR5Robot:
        def __init__(self, sensor: UR5SensorNode, dt: float):
            self.sensor = sensor
            self.dt = dt

        def get_observation(self):
            img1  = self.sensor.get_azure_kinect_image()  # (360, 640, 3) uint8
            img2  = self.sensor.get_wfov_image()           # (360, 640, 3) uint8
            state = self.sensor.get_joint_state()          # (7,) float64
            # reorder to training convention: [pan, lift, elbow, w1, w2, w3, gripper]
            state = state[[5, 0, 1, 2, 3, 4, 6]]
            return img1, img2, state

        def send_chunk(self, arm_chunk, grip_chunk):
            """Fire-and-forget arm trajectory + gripper dispatch.

            Submits the FollowJointTrajectory goal asynchronously (no waits on
            acceptance or completion) so the control loop can immediately dispatch
            the next sub-chunk while the controller spline-interpolates this one.
            New goals preempt the previous trajectory at the controller level.
            """
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = self.sensor.scaled_joint_names_reordered
            for i, positions in enumerate(arm_chunk):
                point = JointTrajectoryPoint()
                point.positions = np.atleast_1d(positions).tolist()
                t = (i + 1) * self.dt
                point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
                goal.trajectory.points.append(point)
            self.sensor._trajectory_action.send_goal_async(goal)
            grip_thread = threading.Thread(
                target=send_gripper_paced,
                args=(self.sensor, grip_chunk, self.dt, 50.0),
                daemon=True,
            )
            grip_thread.start()

        def send_action(self, action):
            """Single-step fallback (single mode)."""
            self.sensor.send_single_action_scaled_joint(action[:6], dt=self.dt, wait=False)
            self.sensor.send_gripper_command(float(action[6]))

    rclpy.init()
    sensor = UR5SensorNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sensor,), daemon=True)
    spin_thread.start()

    print("Waiting for sensor data...")
    while (sensor.get_joint_state() is None
           or len(sensor.k4a_buffer) == 0
           or len(sensor.wfov_buffer) == 0):
        time.sleep(0.1)
    print("Sensors ready.")

    print("Moving to home position...")
    sensor.send_single_action_scaled_joint(HOME_JOINT_POSITIONS, dt=3.0, wait=True)
    sensor.send_gripper_command(0.0)
    print("Home position reached.")

    robot = UR5Robot(sensor, dt=args.dt)
    client = Gr00tAsyncClient(
        host=args.host,
        port=args.port,
        api_token=args.api_token,
        lang=args.lang,
        action_horizon=args.action_horizon,
        chunk_size_threshold=args.chunk_size_threshold,
        dt=args.dt,
        aggregate_fn_name=args.aggregate_fn_name,
        send_mode=args.send_mode,
        chunk_dispatch_size=args.chunk_dispatch_size,
        chunk_filter=args.chunk_filter,
        chunk_filter_q=args.chunk_filter_q,
        chunk_filter_r=args.chunk_filter_r,
        chunk_filter_window=args.chunk_filter_window,
        chunk_filter_polyorder=args.chunk_filter_polyorder,
    )

    try:
        client.run(robot, num_steps=args.num_steps)
    finally:
        sensor.stop()
        sensor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(AsyncArgsConfig))
