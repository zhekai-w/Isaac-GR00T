import os
import random
import subprocess
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import tyro
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray
from gr00t.eval.robot import RobotInferenceClient
from filter_utils import OneEuroFilter, savgol_chunk, rts_smoother_chunk, blend_chunk_boundary
from scipy.signal import savgol_filter
from pynput import keyboard

# Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A

# Realsense
import pyrealsense2 as rs

WIDTH, HEIGHT = 640, 360
WFOV_DEVICE = 0
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
GRIPPER_THRESHOLD = 0.005  # metres (~5 mm dead-band)
RELEASE_ENTER_THRESHOLD = 0.1
RELEASE_EXIT_THRESHOLD = 0.05

# HOME_JOINT_POSITIONS = np.array([
#      1.4191266298294067,  # shoulder_pan_joint
#     -1.749514404927389,   # shoulder_lift_joint
#      1.754384994506836,   # elbow_joint
#     -1.6118515173541468,  # wrist_1_joint
#     -1.5746586958514612,  # wrist_2_joint
#     -0.12216407457460576, # wrist_3_joint
# ])

HOME_JOINT_POSITIONS = np.array([
     np.deg2rad(90.0),  # shoulder_pan_joint
     np.deg2rad(-89.71),   # shoulder_lift_joint
     np.deg2rad(96.66),   # elbow_joint
     np.deg2rad(-96.91),  # wrist_1_joint
     np.deg2rad(-89.70),  # wrist_2_joint
     np.deg2rad(0.0), # wrist_3_joint
])

HOME_TOLERANCE = 0.05

TASK = ["place apple in the basket.",
        "place apple in the wooden plate.",
        "place apple in the white plate.",
        "place mango in the basket.",
        "place mango in the wooden plate.",
        "place mango in the white plate.",
        "place green pepper in the basket.",
        "place green pepper in the wooden plate.",
        "place green pepper in the white plate."]

@dataclass
class ArgsConfig:
    host: str = "localhost"
    port: int = 5555
    lang: str | None = None
    action_horizon: int = 16
    dt: float = 0.15
    num_cycles: int = 300
    gripper_max_effort: float = 50.0
    send_mode: Literal["single", "chunk"] = "chunk"
    single_duration: float = 0.3        # trajectory duration per waypoint (single mode)
    single_last_duration: float = 0.2   # duration for last N waypoints (with blocking wait)
    single_wait_last_n: int = 2         # how many trailing steps to block on
    filter: bool = False
    filter_mincutoff: float = 1.0  # Hz — lower = more smoothing
    filter_beta: float = 0.1       # speed coefficient — higher = less lag when moving fast
    # Within-chunk polynomial smoothing
    chunk_filter: Literal["none", "savgol", "rts"] = "none"
    chunk_filter_window: int = 7      # savgol: must be odd and < action_horizon
    chunk_filter_polyorder: int = 3   # savgol: must be < chunk_filter_window
    chunk_filter_q: float = 1e-3      # rts: process noise (larger = trust measurements more)
    chunk_filter_r: float = 1e-4      # rts: measurement noise (larger = smooth more)
    # Between-chunk boundary blending
    boundary_blend: bool = False
    boundary_blend_steps: int = 4     # cosine ramp over first N waypoints (N * dt seconds)
    # Frame buffer selection: 0 = current frame, -N = N frames ago (range [-4, 0])
    buffer: int = 0
    # Controller for chunk mode: "scaled_joint_trajectory_controller" uses action server,
    # "forward_position_controller" publishes Float64MultiArray directly (no interpolation)
    controller: Literal["scaled_joint_trajectory_controller", "forward_position_controller"] = "scaled_joint_trajectory_controller"


def build_obs_dict(img1, img2, state, lang):
    """
    Build GR00T observation dict from raw sensor data.

    Args:
        img1: Azure Kinect RGB image, shape (360, 640, 3), uint8
        img2: RealSense RGB image, shape (360, 640, 3), uint8
        img2: WFOV USB camera RGB image, shape (360, 640, 3), uint8
        state: Joint state, shape (7,), float64
               [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, gripper]
    """
    return {
        "video.azure_kinect": img1[np.newaxis, ...],
        # "video.realsense": img2[np.newaxis, ...],
        "video.wfov": img2[np.newaxis, ...],
        "state.ur5_arm": state[:6][np.newaxis, ...].astype(np.float64),
        "state.gripper": state[6:7][np.newaxis, ...].astype(np.float64),
        "annotation.human.task_description": [lang],
    }


class UR5SensorNode(Node):
    def __init__(self, controller: str = "scaled_joint_trajectory_controller"):
        super().__init__('ur5_gr00t_client')
        self._controller = controller

        # --- Joint state buffers ---
        self.latest_joint_position = None
        self.joint_lock = threading.Lock()
        self.latest_gripper_position = 0.0
        self.gripper_lock = threading.Lock()

        # ROS2 subscribers
        self.create_subscription(JointState, "/joint_states", self._jointstate_callback, 1)
        self.create_subscription(JointState, "/gripper/joint_states", self._gripper_callback, 1)

        # Action client for scaled_joint_trajectory_controller
        self._trajectory_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        )
        # Publisher for forward_position_controller (chunk mode only)
        if controller == "forward_position_controller":
            self._fpc_pub = self.create_publisher(
                Float64MultiArray,
                "/forward_position_controller/commands",
                1,
            )
        else:
            self._fpc_pub = None
        # Action client: Robotiq gripper
        self._gripper_action = ActionClient(
            self,
            GripperCommand,
            "/gripper/robotiq_gripper_controller/gripper_cmd",
        )
        # Match joint order from data_collect.py (UR driver /joint_states order)
        self.joint_names = [
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "shoulder_pan_joint",
        ]

        # Train another model with reordered joints
        self.scaled_joint_names_reordered = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Joint order matching the dataset and eval_policy_hardware.py
        self.scaled_joint_names = [
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "shoulder_pan_joint",
        ]

        # --- Azure Kinect init ---
        k4a_config = Config()
        k4a_config.color_resolution = pyk4a.ColorResolution.RES_1080P
        k4a_config.depth_mode = pyk4a.DepthMode.OFF
        k4a_config.camera_fps = pyk4a.FPS.FPS_30
        k4a_config.synchronized_images_only = False
        self.k4a = PyK4A(k4a_config)
        self.k4a.start()

        # --- RealSense init ---
        # self.rs_pipeline = rs.pipeline()
        # rs_config = rs.config()
        # rs_config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, 30)
        # self.rs_pipeline.start(rs_config)

        # --- Camera buffers + threads ---
        self.k4a_buffer = deque(maxlen=5)  # newest at right; img1 (Azure Kinect)
        self.k4a_lock = threading.Lock()
        # self.latest_rs_image = None
        # self.rs_lock = threading.Lock()
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
            raise RuntimeError(f"Failed to open WFOV camera at /dev/video{WFOV_DEVICE}")

        self.wfov_buffer = deque(maxlen=5)  # newest at right; img2 (WFOV)
        self.wfov_lock = threading.Lock()

        threading.Thread(target=self._k4a_loop, daemon=True).start()
        # threading.Thread(target=self._rs_loop, daemon=True).start()
        threading.Thread(target=self._wfov_loop, daemon=True).start()

    # -- ROS2 callbacks --
    def _jointstate_callback(self, msg):
        with self.joint_lock:
            self.latest_joint_position = np.array(list(msg.position), dtype=np.float32)

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

    # def _rs_loop(self):
    #     while self.camera_running:
    #         try:
    #             frames = self.rs_pipeline.wait_for_frames()
    #             color_frame = frames.get_color_frame()
    #             if not color_frame:
    #                 continue
    #             with self.rs_lock:
    #                 self.latest_rs_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
    #         except RuntimeError as e:
    #             self.get_logger().error(f"RS error: {e}")
    #             time.sleep(0.01)

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
        """Return frame from k4a ring buffer. buffer_idx=0 → newest, -N → N frames ago."""
        with self.k4a_lock:
            if len(self.k4a_buffer) == 0:
                return None
            idx = max(-(len(self.k4a_buffer)), buffer_idx - 1)
            return self.k4a_buffer[idx].copy()

    def get_realsense_image(self):
        with self.rs_lock:
            return self.latest_rs_image.copy() if self.latest_rs_image is not None else None

    def get_wfov_image(self, buffer_idx: int = 0):
        """Return frame from WFOV ring buffer. buffer_idx=0 → newest, -N → N frames ago."""
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

    def send_single_action(self, arm_positions, dt, wait: bool = False):
        """Send a single joint position as a 1-point trajectory."""
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
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            return True
        return False

    def send_single_action_scaled_joint(self, arm_positions, dt: float, wait: bool = False,
                                         timeout_sec: float = 0.3, velocities=None):
        """Send a single joint position as a 1-point trajectory.

        With wait=True the call blocks until the controller finishes executing the
        point, so time_from_start=dt naturally paces replay at the correct rate.
        Pass velocities (6-element array) to avoid stop-and-go jitter between frames.
        """

        if not self._trajectory_action.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning(f"UR action server not ready: {self.ur_action_name}")
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

        rclpy.spin_until_future_complete(self, send_future, timeout_sec=dt+timeout_sec)
        if not send_future.done():
            self.get_logger().error("send_goal_async timed out")
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=dt+timeout_sec)
        if not result_future.done():
            self.get_logger().error("get_result_async timed out")
            return False

        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            return True
        else:
            self.get_logger().warn(
                f"Trajectory finished with error_code={result.error_code}, "
                f"error_string={result.error_string}"
            )
            return False

    def send_chunk_action(self, arm_actions, dt):
        """Send full action chunk as one trajectory goal; controller spline-interpolates."""
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

    def send_chunk_action_fpc(self, arm_actions, dt):
        """Send chunk waypoints to forward_position_controller at fixed dt intervals."""
        for positions in arm_actions:
            msg = Float64MultiArray()
            msg.data = np.atleast_1d(positions).tolist()
            self._fpc_pub.publish(msg)
            time.sleep(dt)

    def send_gripper_command(self, position, max_effort=50.0):
        """Send a gripper position command (fire-and-forget)."""
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = max_effort
        self._gripper_action.send_goal_async(goal)

    def stop(self):
        self.camera_running = False
        self.k4a.stop()
        # self.rs_pipeline.stop()
        self.wfov_cap.release()


def savgol_chunk(arm_chunk: np.ndarray, window_length: int = 7, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smooth over time axis of arm trajectory chunk (H, 6)."""
    return savgol_filter(arm_chunk, window_length, polyorder, axis=0)


def rts_smoother_chunk(arm_chunk: np.ndarray, dt: float = 0.15,
                        q: float = 1e-3, r: float = 1e-4) -> np.ndarray:
    """RTS (Rauch-Tung-Striebel) Kalman smoother on arm trajectory chunk.

    Constant-velocity state model per joint: state = [position, velocity].
    q: process noise — larger = more responsive, less smooth.
    r: measurement noise — larger = more smoothing.
    arm_chunk: (H, N) — works for any number of joints N.
    Returns (H, N) smoothed positions.
    """
    H, n_joints = arm_chunk.shape
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Hm = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    R = np.array([[r]])

    smoothed = np.zeros_like(arm_chunk)
    for j in range(n_joints):
        z = arm_chunk[:, j]
        # Forward Kalman pass
        xs = np.zeros((H, 2))
        Ps = np.zeros((H, 2, 2))
        x = np.array([z[0], 0.0])
        P = np.eye(2)
        for t in range(H):
            x = F @ x
            P = F @ P @ F.T + Q
            S = (Hm @ P @ Hm.T + R)[0, 0]
            K = (P @ Hm.T) / S
            x = x + K.flatten() * (z[t] - (Hm @ x)[0])
            P = (np.eye(2) - K @ Hm) @ P
            xs[t], Ps[t] = x, P
        # Backward RTS smoother pass
        sm = xs.copy()
        Psm = Ps.copy()
        for t in range(H - 2, -1, -1):
            P_pred = F @ Ps[t] @ F.T + Q
            G = Ps[t] @ F.T @ np.linalg.inv(P_pred)
            sm[t] = xs[t] + G @ (sm[t + 1] - F @ xs[t])
            Psm[t] = Ps[t] + G @ (Psm[t + 1] - P_pred) @ G.T
        smoothed[:, j] = sm[:, 0]
    return smoothed


def blend_chunk_boundary(arm_chunk: np.ndarray, prev_end_pos: np.ndarray,
                          prev_end_vel: np.ndarray, dt: float,
                          blend_steps: int = 4) -> np.ndarray:
    """Cosine-ramp first blend_steps waypoints from previous chunk's terminal state.

    Eliminates hard positional jumps at chunk boundaries.
    prev_end_pos: (N,) last commanded position of previous chunk.
    prev_end_vel: (N,) estimated velocity at end of previous chunk (units/s).
    """
    blended = arm_chunk.copy()
    n = min(blend_steps, arm_chunk.shape[0])
    for step in range(n):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (step + 1) / n))  # 0 → 1
        predicted = prev_end_pos + prev_end_vel * dt * (step + 1)
        blended[step] = (1.0 - alpha) * predicted + alpha * arm_chunk[step]
    return blended


def send_gripper_paced(sensor, grip_chunk: list, dt: float, effort: float) -> None:
    """Send gripper commands synchronized with arm waypoint timing."""
    prev = grip_chunk[0]
    sensor.send_gripper_command(prev, max_effort=effort)
    for g in grip_chunk[1:]:
        time.sleep(dt)
        if abs(g - prev) > GRIPPER_THRESHOLD:
            sensor.send_gripper_command(g, max_effort=effort)
            prev = g


def main(args: ArgsConfig):
    assert -4 <= args.buffer <= 0, f"--buffer must be in [-4, 0], got {args.buffer}"

    client = RobotInferenceClient(host=args.host, port=args.port)
    assert client.ping(), "Server not reachable"
    print("Modality config:", client.get_modality_config())

    # --- Init ROS2 + sensor node ---
    rclpy.init()
    sensor = UR5SensorNode(controller=args.controller)
    spin_thread = threading.Thread(target=rclpy.spin, args=(sensor,), daemon=True)
    spin_thread.start()

    # Wait for first readings
    need_frames = abs(args.buffer) + 1
    print(f"Waiting for sensor data (need {need_frames} buffered frame(s))...")
    while (sensor.get_joint_state() is None
           or len(sensor.k4a_buffer) < need_frames
           or len(sensor.wfov_buffer) < need_frames):
        time.sleep(0.1)
    print("Sensors ready.")

    print("Moving to home position before inference...")
    sensor.send_single_action_scaled_joint(HOME_JOINT_POSITIONS, dt=3.0, wait=True)
    sensor.send_gripper_command(0.0, max_effort=args.gripper_max_effort)
    print("Home position reached.")

    # Validate chunk filter args
    if args.chunk_filter == "savgol":
        assert args.chunk_filter_window % 2 == 1, "chunk_filter_window must be odd"
        assert args.chunk_filter_window < args.action_horizon, \
            f"chunk_filter_window ({args.chunk_filter_window}) must be < action_horizon ({args.action_horizon})"
        assert args.chunk_filter_polyorder < args.chunk_filter_window, \
            "chunk_filter_polyorder must be < chunk_filter_window"

    # --- Filters (one per joint, persistent across cycles) ---
    dt = args.dt
    freq = 1.0 / args.dt
    if args.filter:
        arm_filters = [OneEuroFilter(freq, args.filter_mincutoff, args.filter_beta) for _ in range(6)]
        grip_filter = OneEuroFilter(freq, args.filter_mincutoff, args.filter_beta)

    def on_object_released(cycle, step=None, gripper_value=0.0):
        nonlocal returning_home
        print(f"Object released at cycle {cycle}, gripper={gripper_value:.3f}")
        print("Returning to home pose...")
        returning_home = True

    def is_at_home(state, tolerance=HOME_TOLERANCE):
        return np.allclose(state[:6], HOME_JOINT_POSITIONS, atol=tolerance)

    prev_gripper = 0.0
    returning_home = False
    inferring = False
    quit_flag = False
    use_random_task = args.lang is None
    task_idx = 0
    if use_random_task:
        task_idx = random.randrange(len(TASK))
        args.lang = TASK[task_idx]
    print(f"Task [{task_idx}]: {args.lang}")
    prev_chunk_end_pos: np.ndarray | None = None  # (6,) last commanded arm position
    prev_chunk_end_vel: np.ndarray | None = None  # (6,) estimated velocity at end of chunk

    def on_press(key):
        nonlocal inferring, returning_home, quit_flag
        try:
            ch = key.char
        except AttributeError:
            return
        if ch == 's':
            inferring = True
            print("[KB] Inference started")
        elif ch == 'p':
            inferring = False
            print("[KB] Inference paused")
        elif ch == 'h':
            returning_home = True
            print("[KB] Returning home")
        elif ch == 'q':
            quit_flag = True
            print("[KB] Quit requested")

    kb_listener = keyboard.Listener(on_press=on_press)
    kb_listener.start()
    print("Keyboard ready: s=start  p=pause  h=home  q=quit")

    os.makedirs("inference_images", exist_ok=True)

    # --- Control loop ---
    try:
        cycle = 0
        while cycle < args.num_cycles:
            if quit_flag:
                break
            if not inferring and not returning_home:
                time.sleep(0.1)
                continue

            state = sensor.get_joint_state()
            state_reordered = state[[5, 0, 1, 2, 3, 4, 6]]

            # --- Return-home path: skip inference ---
            if returning_home:
                if not is_at_home(state_reordered):
                    sensor.send_single_action_scaled_joint(HOME_JOINT_POSITIONS, dt=3.0, wait=True)
                    sensor.send_gripper_command(0.0, max_effort=args.gripper_max_effort)
                if use_random_task:
                    task_idx = random.randrange(len(TASK))
                    args.lang = TASK[task_idx]
                print(f"Home pose reached. Next task [{task_idx}]: {args.lang}")
                args.dt = dt
                returning_home = False
                prev_chunk_end_pos = None
                prev_chunk_end_vel = None
                continue

            else:
                # time.sleep(0.2)
                img1 = sensor.get_azure_kinect_image(args.buffer)
                img2 = sensor.get_wfov_image(args.buffer)

                if img1 is not None:
                    cv2.imwrite(f"inference_images/cycle_{cycle:04d}_k4a.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                if img2 is not None:
                    cv2.imwrite(f"inference_images/cycle_{cycle:04d}_wfov.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

                obs = build_obs_dict(img1, img2, state_reordered, args.lang)

                t0 = time.perf_counter()
                action_dict = client.get_action(obs)
                t_infer = time.perf_counter() - t0

                arm_actions = np.atleast_2d(action_dict["action.ur5_arm"])   # (H, 6)
                gripper_actions = np.atleast_1d(action_dict["action.gripper"]).flatten()  # (H,)

                arm_chunk = []
                grip_chunk = []
                for i in range(args.action_horizon):
                    arm_pos = arm_actions[i]
                    grip_pos = gripper_actions[i]
                    if args.filter:
                        arm_pos = np.array([arm_filters[j](arm_pos[j]) for j in range(6)])
                        grip_pos = grip_filter(grip_pos)
                    arm_chunk.append(arm_pos)
                    grip_chunk.append(grip_pos)

                # --- Within-chunk smoothing ---
                arm_chunk_arr = np.array(arm_chunk)    # (H, 6)
                grip_chunk_arr = np.array(grip_chunk)  # (H,)

                if args.chunk_filter == "savgol":
                    arm_chunk_arr = savgol_chunk(arm_chunk_arr, args.chunk_filter_window, args.chunk_filter_polyorder)
                    grip_chunk_arr = savgol_filter(grip_chunk_arr, args.chunk_filter_window, args.chunk_filter_polyorder)
                elif args.chunk_filter == "rts":
                    arm_chunk_arr = rts_smoother_chunk(arm_chunk_arr, dt=args.dt, q=args.chunk_filter_q, r=args.chunk_filter_r)
                    grip_chunk_arr = rts_smoother_chunk(grip_chunk_arr[:, np.newaxis], dt=args.dt, q=args.chunk_filter_q, r=args.chunk_filter_r).squeeze(1)

                arm_chunk = arm_chunk_arr
                grip_chunk = grip_chunk_arr.tolist()

            # --- Boundary blending + state update (skip when overriding to home) ---
            if not returning_home:
                arm_arr = np.array(arm_chunk)
                if args.boundary_blend and prev_chunk_end_pos is not None:
                    arm_arr = blend_chunk_boundary(arm_arr, prev_chunk_end_pos, prev_chunk_end_vel, args.dt, args.boundary_blend_steps)
                    arm_chunk = arm_arr
                prev_chunk_end_pos = arm_arr[-1].copy()
                prev_chunk_end_vel = (arm_arr[-1] - arm_arr[-2]) / args.dt

            if args.send_mode == "chunk":
                grip_thread = threading.Thread(
                    target=send_gripper_paced,
                    args=(sensor, grip_chunk, args.dt, args.gripper_max_effort),
                    daemon=True,
                )
                grip_thread.start()
                if args.controller == "forward_position_controller":
                    sensor.send_chunk_action_fpc(arm_chunk, args.dt)
                else:
                    sensor.send_chunk_action(arm_chunk, args.dt)
                grip_thread.join()

            elif args.send_mode == "single":
                CHUNK_SIZE = args.action_horizon
                WAIT_FROM = CHUNK_SIZE - 2  # frame index within chunk where wait flips to True
                last_grip_sent = grip_chunk[0] - 2 * GRIPPER_THRESHOLD
                for i, (arm_pos, grip_pos) in enumerate(zip(arm_chunk, grip_chunk)):
                    pos_in_chunk = i % CHUNK_SIZE
                    is_last_in_chunk = (pos_in_chunk >= WAIT_FROM) or (i == len(arm_chunk) - 1)
                    # was_holding = prev_gripper > RELEASE_ENTER_THRESHOLD
                    # is_holding = grip_pos > RELEASE_ENTER_THRESHOLD
                    # if was_holding and not is_holding:
                    #     on_object_released(cycle, step=i, gripper_value=grip_pos)
                    # prev_gripper = grip_pos
                    if abs(grip_pos - last_grip_sent) > GRIPPER_THRESHOLD:
                        sensor.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)
                        last_grip_sent = grip_pos
                    if not is_last_in_chunk:
                        sensor.send_single_action_scaled_joint(arm_pos, dt=args.dt, wait=False)
                    else:
                        sensor.send_single_action_scaled_joint(arm_pos, dt=args.dt+0.3, wait=True)

            # else:  # "single"
            #     n = len(arm_chunk)
            #     for i, (arm_pos, grip_pos) in enumerate(zip(arm_chunk, grip_chunk)):
            #         is_last = i >= n - args.single_wait_last_n
            #         dt_step = args.single_last_duration if is_last else args.single_duration
            #         sensor.send_single_action(arm_pos, dt_step, wait=is_last)
            #         sensor.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)

            print(f"Cycle {cycle}: inference={t_infer:.3f}s")
            cycle += 1
    finally:
        kb_listener.stop()
        sensor.stop()
        sensor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    config = tyro.cli(ArgsConfig)
    main(config)
