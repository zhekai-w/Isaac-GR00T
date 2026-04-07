# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro

sys.path.insert(0, os.path.dirname(__file__))
from eval_policy_hardware import OneEuroFilter

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Modality keys to evaluate."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.
    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    steps: int = 150
    """Number of steps to evaluate."""

    trajs: int = 1
    """Number of trajectories to evaluate."""

    start_traj: int = 0
    """Start trajectory to evaluate."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str = None
    """Path to save the plot."""

    plot_state: bool = False
    """Whether to plot the state."""

    filter: bool = False
    """Apply One Euro Filter to predicted actions before plotting."""

    filter_mincutoff: float = 1.0
    """One Euro Filter min cutoff frequency (Hz). Lower = more smoothing."""

    filter_beta: float = 0.1
    """One Euro Filter speed coefficient. Higher = less lag when moving fast."""

    dt: float = 0.05
    """Time step between actions (seconds), used to set filter frequency."""


def main(args: ArgsConfig):
    data_config = load_data_config(args.data_config)

    # Set action_horizon from data config if not provided
    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config '{args.data_config}'")

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

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)

    all_mse = []
    all_inference_times = []
    total_start = time.perf_counter()

    for traj_id in range(args.start_traj, args.start_traj + args.trajs):
        print("Running trajectory:", traj_id)

        # Time each inference step manually
        traj_inference_times = []
        for step_count in range(args.steps):
            if step_count % args.action_horizon == 0:
                data_point = dataset.get_step_data(traj_id, step_count)
                t0 = time.perf_counter()
                _ = policy.get_action(data_point)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                traj_inference_times.append(elapsed)
                print(f"  Step {step_count}: inference took {elapsed:.4f}s")

        all_inference_times.extend(traj_inference_times)

        # Build per-dim filters for this trajectory (reset state between trajs)
        filters = None
        if args.filter:
            # Determine action dim from a sample step
            sample = dataset.get_step_data(traj_id, 0)
            action_dim = sum(
                np.atleast_1d(sample[f"action.{key}"][0]).shape[0]
                for key in args.modality_keys
            )
            freq = 1.0 / args.dt
            filters = [OneEuroFilter(freq, args.filter_mincutoff, args.filter_beta) for _ in range(action_dim)]

        # Run the full eval (with MSE + plotting)
        mse = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            plot_state=args.plot_state,
            save_plot_path=args.save_plot_path,
            filters=filters,
        )
        print("MSE:", mse)
        all_mse.append(mse)

    total_elapsed = time.perf_counter() - total_start

    print("\n--- Timing Summary ---")
    print(f"Total inference calls: {len(all_inference_times)}")
    print(f"Mean inference time:   {np.mean(all_inference_times):.4f}s")
    print(f"Min inference time:    {np.min(all_inference_times):.4f}s")
    print(f"Max inference time:    {np.max(all_inference_times):.4f}s")
    print(f"Total wall time:       {total_elapsed:.2f}s")
    print(f"Average MSE across all trajs: {np.mean(all_mse)}")
    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
