"""
Shared utilities for UR5 + GR00T client variants.

All three client scripts (simple, streaming, timed_chunk) share the same
ROS2 sensor node, camera setup, gripper pacing, keyboard handling, and
observation-dict construction. This module centralises that to avoid
duplication.
"""

import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory, GripperCommand
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64, Float64MultiArray

# Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A


JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

WIDTH, HEIGHT = 640, 360
WFOV_DEVICE = 2
WFOV_DEVICE_PATH = f"/dev/video{WFOV_DEVICE}"

_WFOV_V4L2_DEFAULTS = [
    ("brightness",                 0),
    ("contrast",                  32),
    ("saturation",                64),
    ("hue",                        0),
    ("white_balance_automatic",    1),
    ("gamma",                    100),
    ("gain",                       0),
    ("power_line_frequency",       1),
    ("white_balance_temperature", 4600),
    ("sharpness",                  2),
    ("backlight_compensation",     1),
    ("auto_exposure",              3),
    ("exposure_time_absolute",   157),
    ("exposure_dynamic_framerate", 0),
]


def _wfov_init_camera():
    for name, value in _WFOV_V4L2_DEFAULTS:
        subprocess.run(
            ["v4l2-ctl", "-d", WFOV_DEVICE_PATH, "-c", f"{name}={value}"],
            capture_output=True,
        )


GRIPPER_THRESHOLD = 0.005       # metres (~5 mm dead-band)
RELEASE_ENTER_THRESHOLD = 0.1
RELEASE_EXIT_THRESHOLD = 0.05

HOME_JOINT_POSITIONS = np.array([
    np.deg2rad(90.0),   # shoulder_pan_joint
    np.deg2rad(-89.71),  # shoulder_lift_joint
    np.deg2rad(96.66),  # elbow_joint
    np.deg2rad(-96.91), # wrist_1_joint
    np.deg2rad(-89.70), # wrist_2_joint
    np.deg2rad(0.0),    # wrist_3_joint
])

HOME_TOLERANCE = 0.05

TASK = [
    "place mango in the basket.",
    "place mango in the wooden plate.",
    "place mango in the white plate.",
    "place apple in the basket.",
    "place apple in the wooden plate.",
    "place apple in the white plate.",
    "place green pepper in the basket.",
    "place green pepper in the wooden plate.",
    "place green pepper in the white plate.",
]


def build_obs_dict(img1, img2, state, lang):
    """
    Build GR00T observation dict from raw sensor data.

    Args:
        img1: Azure Kinect RGB image, shape (360, 640, 3), uint8
        img2: WFOV USB camera RGB image, shape (360, 640, 3), uint8
        state: Joint state, shape (7,), float64
               [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper]
        lang: Task description string.
    """
    return {
        "video.azure_kinect": img1[np.newaxis, ...],
        "video.wfov": img2[np.newaxis, ...],
        "state.ur5_arm": state[:6][np.newaxis, ...].astype(np.float64),
        "state.gripper": state[6:7][np.newaxis, ...].astype(np.float64),
        "annotation.human.task_description": [lang],
    }


class UR5SensorNode(Node):
    """ROS2 node wrapping cameras, joint-state subscriptions, and action clients."""

    def __init__(self):
        super().__init__('ur5_gr00t_client')

        # --- Joint state buffers ---
        self.latest_joint_position = None
        self.joint_lock = threading.Lock()
        self.latest_gripper_position = 0.0
        self.gripper_lock = threading.Lock()
        self._js_names = None
        self._js_index = None
        self.js_recording = False
        self.js_log = []

        # --- Controller state + speed scaling ---
        self.ctrl_state = None      # (perf_counter, desired positions in JOINT_ORDER)
        self.speed_scaling = 1.0    # 1.0 = full speed
        self._cs_names = None
        self._cs_index = None

        self.create_subscription(JointState, "/joint_states", self._jointstate_callback, 1)
        self.create_subscription(JointState, "/gripper/joint_states", self._gripper_callback, 1)

        self._trajectory_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        )
        self._jt_stream_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            1,
        )
        self._fpc_pub = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            1,
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            "/scaled_joint_trajectory_controller/controller_state",
            self._ctrl_state_callback, 1)
        self.create_subscription(
            Float64, "/speed_scaling_state_broadcaster/speed_scaling",
            self._speed_scaling_callback, 1)

        self._gripper_action = ActionClient(
            self,
            GripperCommand,
            "/gripper/robotiq_gripper_controller/gripper_cmd",
        )

        # Joint order matching the dataset and data_collect.py
        self.scaled_joint_names_reordered = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.scaled_joint_names = [
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "shoulder_pan_joint",
        ]
        self.joint_names = self.scaled_joint_names  # for simple-client compatibility

        # --- Azure Kinect init ---
        k4a_config = Config()
        k4a_config.color_resolution = pyk4a.ColorResolution.RES_1080P
        k4a_config.depth_mode = pyk4a.DepthMode.OFF
        k4a_config.camera_fps = pyk4a.FPS.FPS_30
        k4a_config.synchronized_images_only = False
        self.k4a = PyK4A(k4a_config)
        self.k4a.start()

        # --- Camera buffers + threads ---
        self.k4a_buffer = deque(maxlen=5)
        self.k4a_lock = threading.Lock()
        self.camera_running = True
        self._k4a_frame_count = 0
        self._k4a_fps_t0 = None
        self._wfov_frame_count = 0
        self._wfov_fps_t0 = None

        # --- WFOV USB camera init ---
        _wfov_init_camera()
        self.wfov_cap = cv2.VideoCapture(WFOV_DEVICE)
        self.wfov_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.wfov_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
        self.wfov_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
        if not self.wfov_cap.isOpened():
            raise RuntimeError(f"Failed to open WFOV camera at {WFOV_DEVICE_PATH}")

        self.wfov_buffer = deque(maxlen=5)
        self.wfov_lock = threading.Lock()

        threading.Thread(target=self._k4a_loop, daemon=True).start()
        threading.Thread(target=self._wfov_loop, daemon=True).start()

    # -- ROS2 callbacks --
    def _jointstate_callback(self, msg):
        if msg.name != self._js_names:
            try:
                self._js_index = [list(msg.name).index(j) for j in JOINT_ORDER]
            except ValueError:
                return
            self._js_names = msg.name
        idx = self._js_index
        p = msg.position
        pos = np.array([p[i] for i in idx], dtype=np.float32)
        with self.joint_lock:
            self.latest_joint_position = pos
        if self.js_recording:
            self.js_log.append((time.perf_counter(), *pos))

    def _ctrl_state_callback(self, msg):
        """Latest desired (not measured) joint position from the trajectory controller."""
        src = None
        if len(msg.reference.positions) >= 6:
            src = msg.reference
        elif hasattr(msg, "desired") and len(msg.desired.positions) >= 6:
            src = msg.desired
        if src is None:
            return
        if msg.joint_names != self._cs_names:
            try:
                self._cs_index = [list(msg.joint_names).index(j) for j in JOINT_ORDER]
            except ValueError:
                return
            self._cs_names = msg.joint_names
        p = src.positions
        self.ctrl_state = (time.perf_counter(),
                           np.array([p[i] for i in self._cs_index], dtype=np.float64))

    def _speed_scaling_callback(self, msg):
        self.speed_scaling = float(msg.data) / 100.0

    def _gripper_callback(self, msg):
        with self.gripper_lock:
            self.latest_gripper_position = msg.position[0]

    # -- Camera threads --
    def _k4a_loop(self):
        while self.camera_running:
            try:
                capture = self.k4a.get_capture()
                if capture is None or capture.color is None:
                    continue
                rgb = capture.color[:, :, 2::-1]
                rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                with self.k4a_lock:
                    self.k4a_buffer.append(np.array(rgb, dtype=np.uint8))
                self._k4a_frame_count += 1
                now = time.perf_counter()
                if self._k4a_fps_t0 is None:
                    self._k4a_fps_t0 = now
                elif self._k4a_frame_count % 150 == 0:
                    fps = self._k4a_frame_count / (now - self._k4a_fps_t0)
                    print(f"[camera_fps] azure_kinect: {fps:.1f} Hz")
            except Exception as e:
                self.get_logger().error(f"K4A error: {e}")
                time.sleep(0.01)

    def _wfov_loop(self):
        while self.camera_running:
            try:
                ret, frame = self.wfov_cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                h, w = frame.shape[:2]
                target_ratio = WIDTH / HEIGHT
                if w / h > target_ratio:
                    crop_w = int(h * target_ratio)
                    x = (w - crop_w) // 2
                    frame = frame[:, x:x + crop_w]
                else:
                    crop_h = int(w / target_ratio)
                    y = (h - crop_h) // 2
                    frame = frame[y:y + crop_h, :]
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with self.wfov_lock:
                    self.wfov_buffer.append(frame)
                self._wfov_frame_count += 1
                now = time.perf_counter()
                if self._wfov_fps_t0 is None:
                    self._wfov_fps_t0 = now
                elif self._wfov_frame_count % 150 == 0:
                    fps = self._wfov_frame_count / (now - self._wfov_fps_t0)
                    print(f"[camera_fps] wfov:         {fps:.1f} Hz")
            except Exception as e:
                self.get_logger().error(f"WFOV error: {e}")
                time.sleep(0.01)

    # -- Public getters --
    def get_azure_kinect_image(self, buffer_idx: int = 0):
        """Return frame from k4a ring buffer. buffer_idx=0 -> newest, -N -> N frames ago."""
        with self.k4a_lock:
            if len(self.k4a_buffer) == 0:
                return None
            idx = max(-(len(self.k4a_buffer)), buffer_idx - 1)
            return self.k4a_buffer[idx].copy()

    def get_wfov_image(self, buffer_idx: int = 0):
        """Return frame from WFOV ring buffer. buffer_idx=0 -> newest, -N -> N frames ago."""
        with self.wfov_lock:
            if len(self.wfov_buffer) == 0:
                return None
            idx = max(-(len(self.wfov_buffer)), buffer_idx - 1)
            return self.wfov_buffer[idx].copy()

    def get_joint_state(self):
        with self.joint_lock:
            arm = self.latest_joint_position
        with self.gripper_lock:
            gripper = self.latest_gripper_position
        if arm is None:
            return None
        return np.append(arm, gripper).astype(np.float64)

    # -- Action / command senders (kept from simple_client) --
    def send_single_action(self, arm_positions, dt, wait: bool = False):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = np.atleast_1d(arm_positions).tolist()
        point.time_from_start = Duration(sec=int(dt), nanosec=int((dt % 1) * 1e9))
        goal.trajectory.points.append(point)
        future = self._trajectory_action.send_goal_async(goal)

        if not wait:
            return True
        while not future.done():
            time.sleep(0.01)
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)
        result = result_future.result().result
        return result.error_code == FollowJointTrajectory.Result.SUCCESSFUL

    def send_single_action_scaled_joint(self, arm_positions, dt: float, wait: bool = False,
                                         timeout_sec: float = 0.3, velocities=None):
        if not self._trajectory_action.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning("UR action server not ready")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.scaled_joint_names_reordered
        point = JointTrajectoryPoint()
        point.positions = np.atleast_1d(arm_positions).tolist()
        if velocities is not None:
            point.velocities = np.atleast_1d(velocities).tolist()
        point.time_from_start = Duration(sec=int(dt), nanosec=int((dt % 1) * 1e9))
        goal.trajectory.points.append(point)

        send_future = self._trajectory_action.send_goal_async(goal)
        if not wait:
            return True
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=dt + timeout_sec)
        if not send_future.done():
            self.get_logger().error("send_goal_async timed out")
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=dt + timeout_sec)
        if not result_future.done():
            self.get_logger().error("get_result_async timed out")
            return False
        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            return True
        self.get_logger().warn(
            f"Trajectory finished with error_code={result.error_code}, "
            f"error_string={result.error_string}"
        )
        return False

    def send_window_scaled_joint(self, arm_window, dt: float, velocities_window=None,
                                  times_window=None):
        traj = JointTrajectory()
        traj.joint_names = self.scaled_joint_names_reordered
        for i, positions in enumerate(arm_window):
            point = JointTrajectoryPoint()
            point.positions = np.atleast_1d(positions).tolist()
            if velocities_window is not None:
                point.velocities = np.atleast_1d(velocities_window[i]).tolist()
            t = (i + 1) * dt if times_window is None else float(times_window[i])
            point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            traj.points.append(point)
        self._jt_stream_pub.publish(traj)
        return True

    def send_chunk_action(self, arm_actions, dt):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.scaled_joint_names_reordered
        for i, positions in enumerate(arm_actions):
            point = JointTrajectoryPoint()
            point.positions = np.atleast_1d(positions).tolist()
            t = (i + 1) * dt
            point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(point)
        future = self._trajectory_action.send_goal_async(goal)
        while not future.done():
            time.sleep(0.01)
        goal_handle = future.result()
        time.sleep(1.0)
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory chunk goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)

    def publish_fpc(self, positions):
        msg = Float64MultiArray()
        msg.data = np.atleast_1d(positions).astype(float).tolist()
        self._fpc_pub.publish(msg)

    def send_chunk_action_fpc(self, arm_actions, dt):
        for positions in arm_actions:
            msg = Float64MultiArray()
            msg.data = np.atleast_1d(positions).tolist()
            self._fpc_pub.publish(msg)
            time.sleep(dt)

    def send_gripper_command(self, position, max_effort=50.0):
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = max_effort
        self._gripper_action.send_goal_async(goal)

    def stop(self):
        self.camera_running = False
        self.k4a.stop()
        self.wfov_cap.release()


# --- Gripper pacing ---
def send_gripper_paced(sensor, grip_chunk: list, dt: float, effort: float) -> None:
    """Send gripper commands synchronized with arm waypoint timing."""
    prev = grip_chunk[0]
    sensor.send_gripper_command(prev, max_effort=effort)
    for g in grip_chunk[1:]:
        time.sleep(dt)
        if abs(g - prev) > GRIPPER_THRESHOLD:
            sensor.send_gripper_command(g, max_effort=effort)
            prev = g


# --- Keyboard handler factory ---
def make_keyboard_handler(callbacks: dict):
    """Create a keyboard listener that dispatches 's','p','h','q' to callbacks."""
    def on_press(key):
        try:
            ch = key.char
        except AttributeError:
            return
        if ch in callbacks:
            callbacks[ch]()
    return on_press


# --- Sensor init/shutdown ---
def init_sensor_and_wait(need_frames: int = 1):
    """Initialize rclpy, create UR5SensorNode, spin thread, wait for first data.

    need_frames must cover the caller's --buffer depth: get_*_image(-N) clamps to
    the oldest frame it has, so starting with a half-full ring silently hands back
    a newer frame than asked for instead of blocking.
    """
    rclpy.init()
    sensor = UR5SensorNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sensor,), daemon=True)
    spin_thread.start()
    print(f"Waiting for sensor data (need {need_frames} buffered frame(s))...")
    while (sensor.get_joint_state() is None
           or len(sensor.k4a_buffer) < need_frames
           or len(sensor.wfov_buffer) < need_frames):
        time.sleep(0.1)
    print("Sensors ready.")
    return sensor, spin_thread


def shutdown_sensor(sensor):
    """Clean shutdown of the sensor node and ROS context."""
    sensor.stop()
    sensor.destroy_node()
    rclpy.shutdown()


def home_via_fpc(sensor: UR5SensorNode, duration: float = 1.0, hz: float = 125.0):
    """Move to home by streaming a smoothstep interpolation to forward_position_controller."""
    state = sensor.get_joint_state()
    cur = state[:6].astype(float)
    target = np.asarray(HOME_JOINT_POSITIONS, dtype=float)
    n = max(1, int(duration * hz))
    period = 1.0 / hz
    for i in range(1, n + 1):
        tick = time.perf_counter()
        a = i / n
        s = 3 * a * a - 2 * a * a * a  # smoothstep: zero velocity at both ends
        sensor.publish_fpc(cur + s * (target - cur))
        elapsed = time.perf_counter() - tick
        time.sleep(max(0.0, period - elapsed))
