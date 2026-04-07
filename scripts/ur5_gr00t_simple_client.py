import time
import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from gr00t.eval.robot import RobotInferenceClient

# Azure Kinect
import pyk4a
from pyk4a import Config, PyK4A

# Realsense
import pyrealsense2 as rs

HOST = "localhost"
PORT = 5555
MODALITY_KEY = ["ur5_arm", "gripper"]
ACTION_HORIZON = 16
DT = 0.05
LANG = "placeholder"

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
        "video.azure_kinect": img1[np.newaxis, ...],
        "video.realsense": img2[np.newaxis, ...],
        "state.ur5_arm": state[:6][np.newaxis, ...].astype(np.float64),
        "state.gripper": state[6:7][np.newaxis, ...].astype(np.float64),
        "annotation.human.task_description": [LANG],
    }

def concat_action(action_dict, step):
    return np.concatenate(
        [np.atleast_1d(action_dict[f"action.{k}"][step]) for k in MODALITY_KEY], 
        axis=0,
    )

WIDTH, HEIGHT = 640, 360


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
        # Match joint order from data_collect.py (UR driver /joint_states order)
        self.joint_names = [
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

    def send_single_action(self, arm_positions):
        """Send a single joint position as a 1-point trajectory (fire-and-forget)."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = arm_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=int(DT * 1e9))
        goal.trajectory.points.append(point)
        self._trajectory_action.send_goal_async(goal)

    def stop(self):
        self.camera_running = False
        self.k4a.stop()
        self.rs_pipeline.stop()

def main():
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

    # --- Control loop ---
    NUM_CYCLES = 100

    try:
        for cycle in range(NUM_CYCLES):
            img1 = sensor.get_azure_kinect_image()
            img2 = sensor.get_realsense_image()
            state = sensor.get_joint_state()

            obs = build_obs_dict(img1, img2, state)

            t0 = time.perf_counter()
            action_dict = client.get_action(obs)
            t_infer = time.perf_counter() - t0

            for step in range(ACTION_HORIZON):
                arm_action = np.atleast_1d(action_dict["action.ur5_arm"][step])
                sensor.send_single_action(arm_action)
                time.sleep(DT)

            print(f"Cycle {cycle}: inference={t_infer:.3f}s")
    finally:
        sensor.stop()
        sensor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()