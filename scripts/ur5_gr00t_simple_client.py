import os
import random
import time
import threading
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import tyro
from pynput import keyboard

from ur5_client_common import (
    GRIPPER_THRESHOLD,
    HOME_JOINT_POSITIONS,
    HOME_TOLERANCE,
    TASK,
    build_obs_dict,
    send_gripper_paced,
    make_keyboard_handler,
    init_sensor_and_wait,
    shutdown_sensor,
)
from filter_utils import OneEuroFilter, apply_chunk_filter, blend_chunk_boundary
from gr00t.eval.robot import RobotInferenceClient


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
    single_duration: float = 0.3
    single_last_duration: float = 0.2
    single_wait_last_n: int = 2
    filter: bool = False
    filter_mincutoff: float = 1.0
    filter_beta: float = 0.1
    chunk_filter: Literal["none", "savgol", "rts"] = "none"
    chunk_filter_window: int = 7
    chunk_filter_polyorder: int = 3
    chunk_filter_q: float = 1e-3
    chunk_filter_r: float = 1e-4
    boundary_blend: bool = False
    boundary_blend_steps: int = 4
    buffer: int = 0
    controller: Literal["scaled_joint_trajectory_controller", "forward_position_controller"] = "scaled_joint_trajectory_controller"
    log: bool = False


def main(args: ArgsConfig):
    assert -4 <= args.buffer <= 0, f"--buffer must be in [-4, 0], got {args.buffer}"

    client = RobotInferenceClient(host=args.host, port=args.port)
    assert client.ping(), "Server not reachable"
    print("Modality config:", client.get_modality_config())

    sensor, spin_thread = init_sensor_and_wait(need_frames=abs(args.buffer) + 1)

    print("Moving to home position before inference...")
    sensor.send_single_action_scaled_joint(HOME_JOINT_POSITIONS, dt=0.3, wait=True)
    sensor.send_gripper_command(0.0, max_effort=args.gripper_max_effort)
    print("Home position reached.")

    if args.chunk_filter == "savgol":
        assert args.chunk_filter_window % 2 == 1, "chunk_filter_window must be odd"
        assert args.chunk_filter_window < args.action_horizon, \
            f"chunk_filter_window ({args.chunk_filter_window}) must be < action_horizon ({args.action_horizon})"
        assert args.chunk_filter_polyorder < args.chunk_filter_window, \
            "chunk_filter_polyorder must be < chunk_filter_window"

    dt = args.dt
    freq = 1.0 / args.dt
    if args.filter:
        arm_filters = [OneEuroFilter(freq, args.filter_mincutoff, args.filter_beta) for _ in range(6)]

    returning_home = False
    inferring = False
    quit_flag = False
    use_random_task = args.lang is None
    task_idx = 0
    if use_random_task:
        task_idx = random.randrange(len(TASK))
        args.lang = TASK[task_idx]
    print(f"Task [{task_idx}]: {args.lang}")
    prev_chunk_end_pos = None
    prev_chunk_end_vel = None

    global_step = 0
    log_t = []
    log_step = []
    log_chunk = []
    log_cmd = []
    if args.log:
        sensor.js_log = []

    def on_object_released(cycle, step=None, gripper_value=0.0):
        nonlocal returning_home
        print(f"Object released at cycle {cycle}, gripper={gripper_value:.3f}")
        print("Returning to home pose...")
        returning_home = True

    def is_at_home(state, tolerance=HOME_TOLERANCE):
        return np.allclose(state[:6], HOME_JOINT_POSITIONS, atol=tolerance)

    prev_gripper = 0.0

    def kb_start():
        nonlocal inferring
        inferring = True
        if args.log:
            sensor.js_recording = True
        print("[KB] Inference started")

    def kb_pause():
        nonlocal inferring
        inferring = False
        if args.log:
            sensor.js_recording = False
        print("[KB] Inference paused")

    def kb_home():
        nonlocal returning_home
        returning_home = True
        print("[KB] Returning home")

    def kb_quit():
        nonlocal quit_flag
        quit_flag = True
        print("[KB] Quit requested")

    kb_listener = keyboard.Listener(on_press=make_keyboard_handler({
        's': kb_start, 'p': kb_pause, 'h': kb_home, 'q': kb_quit
    }))
    kb_listener.start()
    print("Keyboard ready: s=start  p=pause  h=home  q=quit")

    os.makedirs("inference_images", exist_ok=True)

    try:
        cycle = 0
        while True:
            if quit_flag:
                break
            if not inferring and not returning_home:
                time.sleep(0.1)
                continue

            state = sensor.get_joint_state()

            if returning_home:
                if not is_at_home(state):
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
                img1 = sensor.get_azure_kinect_image(args.buffer)
                img2 = sensor.get_wfov_image(args.buffer)

                if img1 is not None:
                    cv2.imwrite(f"inference_images/cycle_{cycle:04d}_k4a.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                if img2 is not None:
                    cv2.imwrite(f"inference_images/cycle_{cycle:04d}_wfov.jpg", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

                obs = build_obs_dict(img1, img2, state, args.lang)

                t0 = time.perf_counter()
                action_dict = client.get_action(obs)
                t_infer = time.perf_counter() - t0

                arm_actions = np.atleast_2d(action_dict["action.ur5_arm"])
                gripper_actions = np.atleast_1d(action_dict["action.gripper"]).flatten()

                if args.log:
                    t_cycle_start = time.perf_counter()
                    for i in range(args.action_horizon):
                        log_t.append(t_cycle_start + (i + 1) * args.dt)
                        log_step.append(global_step)
                        log_chunk.append(cycle)
                        log_cmd.append(np.asarray(arm_actions[i], dtype=np.float64).copy())
                        global_step += 1

                arm_chunk = []
                grip_chunk = []
                for i in range(args.action_horizon):
                    arm_pos = arm_actions[i]
                    grip_pos = gripper_actions[i]
                    if args.filter:
                        arm_pos = np.array([arm_filters[j](arm_pos[j]) for j in range(6)])
                    arm_chunk.append(arm_pos)
                    grip_chunk.append(grip_pos)

                arm_chunk_arr = np.array(arm_chunk)
                arm_chunk_arr, _ = apply_chunk_filter(
                    arm_chunk_arr, np.asarray(grip_chunk, dtype=float),
                    args.chunk_filter, args.dt,
                    q=args.chunk_filter_q, r=args.chunk_filter_r,
                    window=args.chunk_filter_window, polyorder=args.chunk_filter_polyorder)
                arm_chunk = arm_chunk_arr

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
                WAIT_FROM = CHUNK_SIZE - 2
                last_grip_sent = grip_chunk[0] - 2 * GRIPPER_THRESHOLD
                for i, (arm_pos, grip_pos) in enumerate(zip(arm_chunk, grip_chunk)):
                    pos_in_chunk = i % CHUNK_SIZE
                    is_last_in_chunk = (pos_in_chunk >= WAIT_FROM) or (i == len(arm_chunk) - 1)
                    if abs(grip_pos - last_grip_sent) > GRIPPER_THRESHOLD:
                        sensor.send_gripper_command(grip_pos, max_effort=args.gripper_max_effort)
                        last_grip_sent = grip_pos
                    if not is_last_in_chunk:
                        sensor.send_single_action_scaled_joint(arm_pos, dt=args.dt, wait=False)
                    else:
                        sensor.send_single_action_scaled_joint(arm_pos, dt=args.dt + 0.3, wait=True)

            print(f"Cycle {cycle}: inference={t_infer:.3f}s")
            cycle += 1
    finally:
        kb_listener.stop()
        if args.log:
            sensor.js_recording = False
            if log_t:
                np.savez(
                    "simple_client_log.npz",
                    t=np.array(log_t),
                    step=np.array(log_step),
                    chunk=np.array(log_chunk),
                    cmd=np.array(log_cmd),
                    js=np.array(sensor.js_log) if sensor.js_log else np.zeros((0, 7)),
                )
                print(f"Diagnostic log saved: simple_client_log.npz ({len(log_t)} waypoints)")
        shutdown_sensor(sensor)


if __name__ == '__main__':
    config = tyro.cli(ArgsConfig)
    main(config)
