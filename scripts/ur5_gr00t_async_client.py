import collections
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from gr00t.eval.robot import RobotInferenceClient

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
}


# ==============================================================================
# Async Client
# ==============================================================================

class Gr00tAsyncClient:
    """
    Two-thread async client for GR00T inference over ZMQ.

    Thread 1 (action_receiver): captures obs, calls ZMQ get_action (blocking),
             merges returned action chunk into a shared queue with temporal blending.
    Thread 2 (control_loop): pops one action per tick at fixed rate, sends to robot.
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        modality_keys=None,
        language_instruction="pick up the object",
        actions_per_chunk=16,
        chunk_size_threshold=0.5,
        environment_dt=0.05,
        aggregate_fn_name="weighted_average",
        api_token=None,
    ):
        self.client = RobotInferenceClient(host=host, port=port, api_token=api_token)
        self.modality_keys = modality_keys or ["ur5_arm", "gripper"]
        self.language_instruction = language_instruction
        self.actions_per_chunk = actions_per_chunk
        self.chunk_size_threshold = chunk_size_threshold
        self.environment_dt = environment_dt
        self.aggregate_fn = AGGREGATE_FUNCTIONS[aggregate_fn_name]

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
        self.max_queue_size = actions_per_chunk

        # Robot reference (set in run())
        self.robot = None

    # ------------------------------------------------------------------
    # Observation / Inference
    # ------------------------------------------------------------------

    def _build_obs_dict(self, img1, img2, state):
        return {
            "video.azure_kinect": img1[np.newaxis, ...],
            "video.realsense": img2[np.newaxis, ...],
            "state.ur5_arm": state[:6][np.newaxis, :].astype(np.float64),
            "state.gripper": state[6:7][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.language_instruction],
        }

    def _get_action_chunk(self, img1, img2, state):
        """Send obs via ZMQ (BLOCKS), return list of TimedAction."""
        obs = self._build_obs_dict(img1, img2, state)
        raw = self.client.get_action(obs)
        # raw = {"action.ur5_arm": (16, 6), "action.gripper": (16, 1)}

        base_step = self.global_timestep
        timed_actions = []
        for i in range(self.actions_per_chunk):
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
    # Thread 1: Action Receiver
    # ------------------------------------------------------------------

    def _action_receiver_thread(self):
        """Daemon thread: captures obs, calls ZMQ inference, fills queue."""
        self.start_barrier.wait()
        print("[action_receiver] Thread started")

        while not self.shutdown_event.is_set():
            # Decide whether to run inference
            should_infer = self._queue_depleted()
            force = self.must_go.is_set() and self._queue_empty()

            if not (should_infer or force):
                time.sleep(0.001)  # yield to avoid busy-wait
                continue

            # Capture observation from robot
            # NOTE: robot read operations must be thread-safe
            img1, img2, state = self.robot.get_observation()

            # ZMQ call -- BLOCKS until server returns action chunk
            try:
                timed_actions = self._get_action_chunk(img1, img2, state)
            except Exception as e:
                print(f"[action_receiver] Inference error: {e}")
                time.sleep(0.1)
                continue

            # Merge into queue with blending
            self._aggregate_into_queue(timed_actions)

            if force:
                self.must_go.clear()
            # Re-arm: next time queue empties, force inference again
            self.must_go.set()

            print(f"[action_receiver] Chunk merged, queue size: {len(self.action_queue)}")

    # ------------------------------------------------------------------
    # Thread 2: Control Loop
    # ------------------------------------------------------------------

    def _control_loop_thread(self, num_steps):
        """Main thread: pops one action per tick at fixed rate."""
        self.start_barrier.wait()
        print("[control_loop] Thread started")
        fps = FPSTracker(target_fps=1.0 / self.environment_dt)

        for step in range(num_steps):
            if self.shutdown_event.is_set():
                break
            loop_start = time.perf_counter()

            # Pop one action
            action = None
            with self.queue_lock:
                if self.action_queue:
                    timed_action = self.action_queue.popleft()
                    action = timed_action.action
                    with self.latest_step_lock:
                        self.latest_executed_step = timed_action.timestep
                    self.global_timestep = timed_action.timestep + 1

            # Execute or hold
            if action is not None:
                self.robot.send_action(action)
            # else: robot holds last position (no new command)

            # Fixed-rate sleep
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, self.environment_dt - elapsed))

            if step % 100 == 0:
                measured_fps = fps.tick()
                print(f"[control_loop] Step {step}, FPS: {measured_fps:.1f}")

    def run(self, robot, num_steps=1000):
        """
        Start async inference loop.

        Args:
            robot: Object with get_observation() -> (img1, img2, state)
                   and send_action(action: np.ndarray) methods.
            num_steps: Total control steps to execute.
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


if __name__ == "__main__":
    # Example: replace with your actual robot interface
    class UR5Robot:
        def get_observation(self):
            img1 = get_azure_kinect_image()   # (360, 640, 3) uint8
            img2 = get_realsense_image()      # (360, 640, 3) uint8
            state = get_joint_state()          # (7,) float64
            return img1, img2, state

        def send_action(self, action):
            # action is np.ndarray shape (7,)
            # action[:6] = joint targets, action[6] = gripper
            send_joint_command(action[:6])
            send_gripper_command(action[6])

    robot = UR5Robot()
    client = Gr00tAsyncClient(
        host="localhost",
        port=5555,
        actions_per_chunk=16,
        chunk_size_threshold=0.5,
        environment_dt=0.05,          # 20 Hz
        aggregate_fn_name="weighted_average",
        language_instruction="place the large cube on the orange box.",
    )
    client.run(robot, num_steps=1000)  # run for 1000 ticks = 50 seconds @ 20 Hz