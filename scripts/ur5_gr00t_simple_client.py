import time
import threading
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
from gr00t.eval.robot import RobotInferenceClient
from eval_policy_hardware import OneEuroFilter

# Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A

# Realsense
import pyrealsense2 as rs

WIDTH, HEIGHT = 640, 360
GRIPPER_THRESHOLD = 0.005  # metres (~5 mm dead-band)
RELEASE_ENTER_THRESHOLD = 0.1
RELEASE_EXIT_THRESHOLD = 0.05 

HOME_JOINT_POSITIONS = np.array([
    -1.749514404927389,  # shoulder_lift_joint
     1.754384994506836,  # elbow_joint
    -1.6118515173541468, # wrist_1_joint
    -1.5746586958514612, # wrist_2_joint
    -0.12216407457460576, # wrist_3_joint
     1.4191266298294067,  # shoulder_pan_joint
])
HOME_TOLERANCE = 0.05


@dataclass
class ArgsConfig:
    host: str = "localhost"
    port: int = 5555
    lang: str = "place the small cube on the red box."
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


def build_obs_dict(img1, img2, state, lang):
    """
    Build GR00T observation dict from raw sensor data.

    Args:
        img1: Azure Kinect RGB image, shape (360, 640, 3), uint8
        img2: RealSense RGB image, shape (360, 640, 3), uint8
        state: Joint state, shape (7,), float64
               [shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan, gripper]
    """
    return {
        "video.azure_kinect": img1[np.newaxis, ...],
        "video.realsense": img2[np.newaxis, ...],
        "state.ur5_arm": state[:6][np.newaxis, ...].astype(np.float64),
        "state.gripper": state[6:7][np.newaxis, ...].astype(np.float64),
        "annotation.human.task_description": [lang],
    }


class UR5SensorNode(Node):
    def __init__(self):
        super().__init__('ur5_gr00t_client')

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
        k4a_config.depth_mode = pyk4a.DepthMode.NFOV_UNBINNED
        k4a_config.camera_fps = pyk4a.FPS.FPS_30
        k4a_config.synchronized_images_only = True
        self.k4a = PyK4A(k4a_config)
        self.k4a.start()

        # --- RealSense init ---
        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, 30)
        self.rs_pipeline.start(rs_config)

        # --- Camera buffers + threads ---
        self.latest_k4a_image = None
        self.k4a_lock = threading.Lock()
        self.latest_rs_image = None
        self.rs_lock = threading.Lock()
        self.camera_running = True

        threading.Thread(target=self._k4a_loop, daemon=True).start()
        threading.Thread(target=self._rs_loop, daemon=True).start()

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
                    self.latest_k4a_image = np.array(rgb, dtype=np.uint8)
            except Exception as e:
                self.get_logger().error(f"K4A error: {e}")
                time.sleep(0.01)

    def _rs_loop(self):
        while self.camera_running:
            try:
                frames = self.rs_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                with self.rs_lock:
                    self.latest_rs_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            except RuntimeError as e:
                self.get_logger().error(f"RS error: {e}")
                time.sleep(0.01)

    # -- Public getters --
    def get_azure_kinect_image(self):
        with self.k4a_lock:
            return self.latest_k4a_image.copy() if self.latest_k4a_image is not None else None

    def get_realsense_image(self):
        with self.rs_lock:
            return self.latest_rs_image.copy() if self.latest_rs_image is not None else None

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
        goal.trajectory.joint_names = self.scaled_joint_names
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
        goal.trajectory.joint_names = self.joint_names
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

    def send_gripper_command(self, position, max_effort=50.0):
        """Send a gripper position command (fire-and-forget)."""
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = max_effort
        self._gripper_action.send_goal_async(goal)

    def stop(self):
        self.camera_running = False
        self.k4a.stop()
        self.rs_pipeline.stop()


def main(args: ArgsConfig):
    client = RobotInferenceClient(host=args.host, port=args.port)
    assert client.ping(), "Server not reachable"
    print("Modality config:", client.get_modality_config())

    # --- Init ROS2 + sensor node ---
    rclpy.init()
    sensor = UR5SensorNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sensor,), daemon=True)
    spin_thread.start()

    # Wait for first readings
    print("Waiting for sensor data...")
    while sensor.get_joint_state() is None or sensor.get_azure_kinect_image() is None or sensor.get_realsense_image() is None:
        time.sleep(0.1)
    print("Sensors ready.")

    # --- Filters (one per joint, persistent across cycles) ---
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

    # --- Control loop ---
    try:
        for cycle in range(args.num_cycles):
            time.sleep(0.5)
            img1 = sensor.get_azure_kinect_image()
            img2 = sensor.get_realsense_image()
            state = sensor.get_joint_state()

            # state_reordered = state[[5, 0, 1, 2, 3, 4, 6]]
            # obs = build_obs_dict(img1, img2, state_reordered, args.lang)
            obs = build_obs_dict(img1, img2, state, args.lang)

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

            if returning_home:
                if is_at_home(state):
                    print("Home pose reached, resuming normal operation...")
                    returning_home = False
                else:
                    arm_chunk = [HOME_JOINT_POSITIONS.copy()] * args.action_horizon
                    grip_chunk = [0.0] * args.action_horizon  # open gripper

            if args.send_mode == "chunk":
                for i, grip_pos in enumerate(grip_chunk):
                    was_holding = prev_gripper > RELEASE_ENTER_THRESHOLD
                    is_holding = grip_pos > RELEASE_ENTER_THRESHOLD
                    if was_holding and not is_holding:
                        on_object_released(cycle, step=i, gripper_value=grip_pos)
                    prev_gripper = grip_pos
                    sensor.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)
                sensor.send_chunk_action(arm_chunk, args.dt)

            elif args.send_mode == "single":
                CHUNK_SIZE = 16
                WAIT_FROM = 14  # frame index within chunk where wait flips to True
                last_grip_sent = grip_chunk[0] - 2 * GRIPPER_THRESHOLD
                for i, (arm_pos, grip_pos) in enumerate(zip(arm_chunk, grip_chunk)):
                    pos_in_chunk = i % CHUNK_SIZE
                    is_last_in_chunk = (pos_in_chunk >= WAIT_FROM) or (i == len(arm_chunk) - 1)
                    was_holding = prev_gripper > RELEASE_ENTER_THRESHOLD
                    is_holding = grip_pos > RELEASE_ENTER_THRESHOLD
                    if was_holding and not is_holding:
                        on_object_released(cycle, step=i, gripper_value=grip_pos)
                    prev_gripper = grip_pos
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
    finally:
        sensor.stop()
        sensor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    config = tyro.cli(ArgsConfig)
    main(config)
