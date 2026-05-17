"""
Evaluate a GR00T policy by replaying dataset observations through the model
and sending the predicted actions to a real UR5 + Robotiq 2F-85.

This is for validating that policy outputs and hardware behavior match
(open-loop replay from dataset, closed-loop execution on robot).

Example:
    # Using inference server:
    python scripts/eval_policy_hardware.py --dataset-path demo_data/robot_sim.PickNPlace/

    # Using local checkpoint:
    python scripts/eval_policy_hardware.py --model-path /path/to/checkpoint --dataset-path /path/to/data
"""

import time
import threading
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from filter_utils import OneEuroFilter, savgol_chunk, rts_smoother_chunk, blend_chunk_boundary


@dataclass
class ArgsConfig:
    host: str = "localhost"
    port: int = 5555
    plot: bool = False
    modality_keys: List[str] = field(default_factory=lambda: ["ur5_arm", "gripper"])
    data_config: str = "ur5_2f85_arm_gripper"
    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    model_path: str = None
    denoising_steps: int = 4
    action_horizon: int = None
    dt: float = 0.05
    steps: int = 150
    trajs: int = 1
    start_traj: int = 0
    gripper_max_effort: float = 50.0
    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    save_plot_path: str = None
    plot_state: bool = False
    filter: bool = True
    filter_mincutoff: float = 1.0  # Hz — lower = more smoothing
    filter_beta: float = 0.1       # speed coefficient — higher = less lag when moving fast
    send_mode: Literal["single", "chunk"] = "chunk"
    # Within-chunk polynomial smoothing
    chunk_filter: Literal["none", "savgol", "rts"] = "none"
    chunk_filter_window: int = 7      # savgol: must be odd and < action_horizon
    chunk_filter_polyorder: int = 3   # savgol: must be < chunk_filter_window
    chunk_filter_q: float = 1e-3      # rts: process noise (larger = trust measurements more)
    chunk_filter_r: float = 1e-4      # rts: measurement noise (larger = smooth more)
    # Between-chunk boundary blending
    boundary_blend: bool = False
    boundary_blend_steps: int = 4     # cosine ramp over first N waypoints (N * dt seconds)


class UR5HardwareNode(Node):
    def __init__(self):
        super().__init__("eval_policy_hardware")

        # Action client: UR5 arm trajectory
        self._traj_action = ActionClient(
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
        # Publisher: forward_position_controller (used for single-step mode)
        self._fwd_pos_pub = self.create_publisher(
            Float64MultiArray,
            "/forward_position_controller/commands",
            1,
        )

        # Joint order matching data_collect.py
        self.scaled_joint_names = [
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "shoulder_pan_joint",
        ]

    def send_single_action(self, arm_positions):
        """Send a single joint position via forward_position_controller (fire-and-forget)."""
        msg = Float64MultiArray()
        arm_positions = np.atleast_1d(arm_positions)
        # reorder for forward_position_controller 
        msg.data = arm_positions[[5, 0, 1, 2, 3, 4]].tolist()  # shoulder_pan last → first
        self._fwd_pos_pub.publish(msg)

    def send_single_action_scaled_joint(self, arm_positions, dt, wait: bool = False):
        """Send a single joint position as a 1-point trajectory."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.scaled_joint_names
        point = JointTrajectoryPoint()
        point.positions = np.atleast_1d(arm_positions).tolist()
        point.time_from_start = Duration(sec=int(dt), nanosec=int((dt % 1) * 1e9))
        goal.trajectory.points.append(point)
        future = self._traj_action.send_goal_async(goal)

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

    def send_chunk_action(self, arm_actions, dt):
        """Send full action chunk as one trajectory goal; controller spline-interpolates."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.scaled_joint_names
        for i, positions in enumerate(arm_actions):
            point = JointTrajectoryPoint()
            point.positions = np.atleast_1d(positions).tolist()
            t = (i + 1) * dt
            point.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(point)

        future = self._traj_action.send_goal_async(goal)
        while not future.done():
            time.sleep(0.01)
        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory chunk goal rejected")
            return

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.01)

    def send_gripper_command(self, position, max_effort=50.0):
        """Send a gripper position command (fire-and-forget)."""
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = max_effort
        self._gripper_action.send_goal_async(goal)


def main(args: ArgsConfig):
    data_config = load_data_config(args.data_config)

    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config")

    # --- Build policy ---
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    modality = policy.get_modality_config()
    print("Modality config:", modality)

    # --- Load dataset ---
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Total trajectories: {len(dataset.trajectory_lengths)}")
    print(f"All trajectories: {dataset.trajectory_lengths}")
    print(f"Running on trajs with modality keys: {args.modality_keys}")

    # --- Init ROS2 ---
    rclpy.init()
    ur5 = UR5HardwareNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(ur5,), daemon=True)
    spin_thread.start()

    # --- Filters (one per joint, persistent across steps) ---
    freq = 1.0 / args.dt
    if args.filter:
        arm_filters = [OneEuroFilter(freq, args.filter_mincutoff, args.filter_beta) for _ in range(6)]
        grip_filter = OneEuroFilter(freq, args.filter_mincutoff, args.filter_beta)

    # --- Open-loop replay ---
    all_inference_times = []
    total_start = time.perf_counter()

    try:
        for traj_id in range(args.start_traj, args.start_traj + args.trajs):
            print(f"Running trajectory: {traj_id}")

            for step in range(0, args.steps, args.action_horizon):
                data_point = dataset.get_step_data(traj_id, step)

                # Inference
                t0 = time.perf_counter()
                action_dict = policy.get_action(data_point)
                t_infer = time.perf_counter() - t0
                all_inference_times.append(t_infer)

                # Extract arm and gripper actions
                arm_actions = np.atleast_2d(action_dict["action.ur5_arm"])   # (H, 6)
                gripper_actions = np.atleast_1d(action_dict["action.gripper"]).flatten()  # (H,)

                # Apply filters to all steps upfront
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

                if args.send_mode == "chunk":
                    for grip_pos in grip_chunk:
                        ur5.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)
                    ur5.send_chunk_action(arm_chunk, args.dt)
                else:  # "single"
                    for arm_pos, grip_pos in zip(arm_chunk, grip_chunk):
                        # ur5.send_single_action(arm_pos)
                        ur5.send_single_action_scaled_joint(arm_pos, dt=0.3, wait=False)
                        ur5.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)
                        # time.sleep(args.dt)

                print(f"  Step {step}: inference={t_infer:.4f}s")
    finally:
        ur5.destroy_node()
        rclpy.shutdown()

    total_elapsed = time.perf_counter() - total_start

    print("\n--- Timing Summary ---")
    print(f"Total inference calls: {len(all_inference_times)}")
    print(f"Mean inference time:   {np.mean(all_inference_times):.4f}s")
    print(f"Min inference time:    {np.min(all_inference_times):.4f}s")
    print(f"Max inference time:    {np.max(all_inference_times):.4f}s")
    print(f"Total wall time:       {total_elapsed:.2f}s")
    print("Done")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)
