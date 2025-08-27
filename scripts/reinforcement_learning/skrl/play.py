# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO","MAAMP"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# from utils.plotter import Plotter, initCanvas
import sys
import os
import numpy as np
# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotter import Plotter, initCanvas
import matplotlib.pyplot as plt
en_plot = 1


# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    if not isinstance(experiment_cfg, dict):
        print("[WARNING] Experiment configuration is not a dictionary. Using default configuration.")
        return
    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    # print("args_cli.use_pretrained_checkpoint",args_cli.use_pretrained_checkpoint)
    # print("args_cli.checkpoint",args_cli.checkpoint)
    # print("algorithm",algorithm)
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    print(f"[INFO] Loading checkpoint from directory: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    # print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # print("dir env ",dir(env))
    # for name in dir(env):
    #     attr = getattr(env, name)
    #     if callable(attr):
    #         print(f"{name}  --> function/method")
    #     else:
    #         print(f"{name}  --> value: {attr}")



    if en_plot:
        plt.ion()
        torque_names = [
            'left_leg_pitch', 'right_leg_pitch', 'left_leg_roll', 'right_leg_roll',
            'left_leg_yaw', 'right_leg_yaw', 'left_knee', 'right_knee',
            'left_ankle_pitch', 'right_ankle_pitch', 'left_ankle_roll', 'right_ankle_roll'
        ]

        # 创建 2 行 6 列的画布（总共 12 个子图）
        # initCanvas(2, 6, 100)
        # plotters = [Plotter(i, name) for i, name in enumerate(torque_names)]

        # ----------------------
        # 外骨骼关节
        # ----------------------
        # Update this list to have 6 elements instead of 4
        exo_torque_names = ['left_ankle', 'right_ankle', 'left_ankle_combined', 'right_ankle_combined', 'left_exo', 'right_exo']

        # 创建 3 行 2 列的画布（总共 2 个子图）
        initCanvas(3, 2, 100)
        exo_plotters = [Plotter(i, name) for i, name in enumerate(exo_torque_names)]
    # reset environment
    obs, _ = env.reset()
    timestep = 0
    last_actions = None
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
       
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}

                # actions["exo"][:, 0] = actions["humanoid"][:, 8]   # 第 8 个维度 → exo 第 0 项
                # actions["exo"][:, 1] = actions["humanoid"][:, 9]  # 第 9 个维度 → exo 第 1 项
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])


            #plot actions data
            # for i in range(12):  # 遍历动作维度
            #     tmp = actions["humanoid"][0, i].detach().cpu().item()
            #     scaled_actions[i] = tmp * action_scale[i]    + action_offset[i]

            # obs, _, _, _, _ = env.step(actions)
            obs, rew, term, trunc, extras = env.step(actions)
            # print("joint names ",extras["joint_names"])
            # print("len(extras[ankle_torques]",len(extras["ankle_torques"]))
            # print("extras[ankle_torques]",extras["ankle_torques"].shape)
            # print("len(extras[joint_names])",len(extras["joint_names"]))

            # if en_plot:
                # 绘制全身关节
                # for joint_idx in range(len(plotters)):
                #     if joint_idx < len(extras["joint_names"]):
                #         plotters[joint_idx].plotLine(
                #             env.extras["applied_torque"][0, joint_idx].item(),
                #             labels=['action']
                #         )

                # 绘制外骨骼关节
                # exo_plotters[0].plotLine( env.extras["ankle_torques"][0, 0].item(),labels=['pos'])
                # exo_plotters[1].plotLine( env.extras["ankle_torques"][0, 1].item(),labels=['pos'])
                # exo_plotters[2].plotLine( env.extras["left_ankle_hight"][0].item(),labels=['pos'])
                # exo_plotters[3].plotLine(     env.extras["right_ankle_hight"][0].item(),labels=['pos'])
                # exo_plotters[4].plotLine( env.extras["ankle_angle"][0, 0].item(),labels=['pos'])
                # exo_plotters[5].plotLine( env.extras["ankle_angle"][0, 1].item(),labels=['pos'])
                
            import csv
            with open("extras_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([extras["ankle_angle"].tolist(),
                                extras["ankle_torques"].tolist(),
                                extras["left_imu_lin_acc"].tolist(),
                                extras["left_imu_ang_vel"].tolist(),
                                extras["right_imu_lin_acc"].tolist(),
                                extras["right_imu_ang_vel"].tolist(),
                                extras["left_ankle_hight"].tolist(),
                                extras["right_ankle_hight"].tolist(),
                                ])


        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
