# GR00T N1.5 Client Guide: Simple & Async Patterns (UR5)

This guide covers two client patterns for deploying GR00T N1.5 inference on a UR5 robot arm, both using the existing `inference_service.py` ZMQ server. No server modifications needed.

---

## Prerequisites

1. Start the inference server:
```bash
python scripts/inference_service.py --server \
    --model-path <your_finetuned_model_or_nvidia/GR00T-N1.5-3B> \
    --data-config <your_ur5_data_config> \
    --embodiment-tag new_embodiment \
    --port 5555
```

2. UR5 robot connected with Robotiq gripper and cameras (Azure Kinect + RealSense).

## UR5 Modality Setup

From `dataset/meta/modality.json`:

| GR00T Key | Source | Shape |
|---|---|---|
| `state.ur5_arm` | Joint positions [0:6] | (1, 6) |
| `state.gripper` | Gripper position [6:7] | (1, 1) |
| `action.ur5_arm` | Target joints [0:6] | (16, 6) |
| `action.gripper` | Target gripper [6:7] | (16, 1) |
| `video.azure_kinect` | cam1 RGB | (1, 360, 640, 3) |
| `video.realsense` | cam2 RGB | (1, 360, 640, 3) |
| `annotation.human.task_description` | Language instruction | list[str] |

- **Control frequency**: 20 Hz (`environment_dt = 0.05s`)
- **Action horizon**: 16 steps (GR00T default)
- **Action dim**: 7 total (6 arm + 1 gripper)
- **Modality keys**: `["ur5_arm", "gripper"]`

---

## 1. Simple Client (Synchronous)

### Architecture

```
[capture obs] --> [ZMQ inference (BLOCKING)] --> [execute 16 actions] --> repeat
                       ~200ms                      16 x 50ms = 800ms
                    robot idles                   robot moves
```

### Code

```python
import time
import numpy as np
from gr00t.eval.robot import RobotInferenceClient

# ---- Config ----
HOST = "localhost"
PORT = 5555
MODALITY_KEYS = ["ur5_arm", "gripper"]
ACTION_HORIZON = 16        # how many of the 16 actions to execute
DT = 0.05                  # 20 Hz, matches dataset FPS
LANG = "place the large cube on the orange box."

# ---- Client ----
client = RobotInferenceClient(host=HOST, port=PORT)
assert client.ping(), "Server not reachable"
print("Modality config:", client.get_modality_config())


def build_obs_dict(img1, img2, state):
    """
    Build GR00T observation dict from raw sensor data.

    Args:
        img1: Azure Kinect RGB image, shape (360, 640, 3), uint8
        img2: RealSense RGB image, shape (360, 640, 3), uint8
        state: Joint state, shape (7,), float64
               [shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan, gripper]
    """
    return {
        "video.azure_kinect": img1[np.newaxis, ...],                # (1, 360, 640, 3)
        "video.realsense": img2[np.newaxis, ...],                   # (1, 360, 640, 3)
        "state.ur5_arm": state[:6][np.newaxis, :].astype(np.float64),   # (1, 6)
        "state.gripper": state[6:7][np.newaxis, :].astype(np.float64),  # (1, 1)
        "annotation.human.task_description": [LANG],
    }


def concat_action(action_dict, step):
    """Concatenate action modalities at a given timestep into flat array."""
    return np.concatenate(
        [np.atleast_1d(action_dict[f"action.{k}"][step]) for k in MODALITY_KEYS],
        axis=0,
    )  # shape (7,)


# ---- Control Loop ----
NUM_CYCLES = 100  # number of inference cycles

for cycle in range(NUM_CYCLES):
    # 1. Capture observation from robot
    img1 = get_azure_kinect_image()   # your camera function, returns (360, 640, 3) uint8
    img2 = get_realsense_image()      # your camera function, returns (360, 640, 3) uint8
    state = get_joint_state()         # your robot function, returns (7,) float64

    # 2. Send observation, get action chunk (BLOCKS during inference)
    obs = build_obs_dict(img1, img2, state)
    action_dict = client.get_action(obs)
    # action_dict = {"action.ur5_arm": (16, 6), "action.gripper": (16, 1)}

    # 3. Execute actions sequentially
    for step in range(ACTION_HORIZON):
        action = concat_action(action_dict, step)  # (7,)
        send_action_to_robot(action)                # your robot command function
        time.sleep(DT)

    print(f"Cycle {cycle}: executed {ACTION_HORIZON} actions")
```

### Timing

```
Cycle 1:
  [inference ~200ms] [a0 a1 a2 ... a15 @ 50ms each = 800ms] = ~1000ms total
                      ▲ robot idle                            ▲ next inference starts

Total per cycle: ~1s (200ms inference + 800ms execution)
At 16 actions/cycle: effective 16 Hz action rate during execution, but 0 Hz during inference
```

---

## 2. Async Client (Action Queue + Temporal Blending)

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                    CLIENT                            │
│                                                      │
│  Thread 1: action_receiver          Thread 2: control│
│  ┌────────────────────────┐   ┌────────────────────┐ │
│  │ capture obs             │   │ pop 1 action/tick  │ │
│  │ ZMQ get_action (blocks) │   │ send to robot      │ │
│  │ merge into queue        │──►│ sleep(env_dt)      │ │
│  │ with temporal blending  │   │ repeat @ 20 Hz     │ │
│  └────────────────────────┘   └────────────────────┘ │
└──────────────────────────────────────────────────────┘
                │
                │ ZMQ REQ/REP
                ▼
┌──────────────────────────────────────────────────────┐
│  inference_service.py (existing, no changes)         │
└──────────────────────────────────────────────────────┘
```

### Temporal Blending

When a new action chunk overlaps with actions already in the queue, overlapping timesteps are blended:

```
Chunk A:   [A0  A1  A2  A3  A4  A5  A6  A7  ...]
                                ▲ queue at 50%, new inference triggered
Chunk B:                       [B4  B5  B6  B7  B8  B9  B10 B11 ...]

Executed:   A0  A1  A2  A3  ?4  ?5  ?6  ?7  B8  B9  B10 B11

Where ?4 = 0.3 * A4 + 0.7 * B4  (weighted_average favors newer prediction)
```

| Timestep | Chunk A | Chunk B | Executed |
|---|---|---|---|
| 0-3 | A0-A3 | -- | A0-A3 |
| 4-7 | A4-A7 | B4-B7 | blend(A, B) |
| 8-11 | -- | B8-B11 | B8-B11 |

### Code

```python
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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


# ==============================================================================
# Usage
# ==============================================================================

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
```

---

## 3. Comparison

| | Simple | Async |
|---|---|---|
| **Threads** | 1 (single) | 2 (receiver + control) |
| **Robot idle during inference** | Yes (~200ms per cycle) | No |
| **Action queue** | None | deque with lock |
| **Temporal blending** | No | Yes (configurable) |
| **Overlap between chunks** | No, sequential | Yes, controlled by `chunk_size_threshold` |
| **Control frequency** | Interrupted by inference | Constant at `1/environment_dt` |
| **Complexity** | Beginner | Intermediate |
| **When to use** | Testing, evaluation, slow tasks | Real-time deployment, fast tasks |

## 4. Timing Diagrams

### Simple Client
```
Time:  0ms        200ms                    1000ms       1200ms
       │           │                         │            │
       [inference]  [a0  a1  a2 ... a15]     [inference]  [a0 ...
       ▲ idle       ▲ executing @ 20Hz       ▲ idle
```

### Async Client
```
Receiver: [obs→inference→merge]·····[obs→inference→merge]·····[obs→infer...
Control:  [a0][a1][a2][a3][a4][a5][a6][a7][a8][a9][a10][a11][a12]...
          ▲ continuous @ 20 Hz, never stops
                           ▲ new chunk blended in seamlessly
```

## 5. Config Reference

| Parameter | Default | Description |
|---|---|---|
| `host` | `"localhost"` | ZMQ server address |
| `port` | `5555` | ZMQ server port |
| `actions_per_chunk` | `16` | Actions used from each inference (GR00T outputs 16) |
| `chunk_size_threshold` | `0.5` | Trigger new inference when queue <= 50% full |
| `environment_dt` | `0.05` | Control loop period in seconds (0.05 = 20 Hz) |
| `aggregate_fn_name` | `"weighted_average"` | Blending strategy for overlapping timesteps |
| `api_token` | `None` | Optional auth token for ZMQ server |

### Aggregate Functions

| Name | Formula | Use case |
|---|---|---|
| `weighted_average` | `0.3 * old + 0.7 * new` | Default. Favors newer predictions |
| `latest_only` | `new` | No blending, just replace |
| `average` | `0.5 * (old + new)` | Equal weight |
| `conservative` | `0.7 * old + 0.3 * new` | Favors older predictions, smoother |

## 6. Notes

- **ZMQ thread safety**: Only the receiver thread uses the ZMQ socket. ZMQ REQ sockets are NOT thread-safe, so the control loop never touches it.
- **First inference latency**: The queue starts empty. The robot holds position until the first action chunk arrives (~200-500ms). The `must_go` mechanism ensures this happens immediately.
- **Robot thread safety**: `get_observation()` is called from the receiver thread while `send_action()` is called from the control loop. Ensure your robot driver supports concurrent reads and writes (UR5 RTDE interface does).
- **Choosing `actions_per_chunk`**: Use all 16 for maximum lookahead, or fewer (e.g., 8-12) for more frequent re-inference with newer observations.
- **Choosing `chunk_size_threshold`**: Lower values (0.3) mean more overlap and smoother blending but more inference calls. Higher values (0.7) mean less compute but choppier transitions.
