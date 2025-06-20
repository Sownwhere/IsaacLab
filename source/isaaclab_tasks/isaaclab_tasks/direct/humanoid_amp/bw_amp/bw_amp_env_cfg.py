# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING
from .bw_cfg import BW_CFG


from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../motions")


@configclass
class BwAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""
    
    # reward
    rew_termination = -1.0
    rew_action_l2 = -0.5
    rew_joint_pos_limits = -0.5
    rew_joint_acc_l2 = -0.001
    rew_joint_vel_l2 = -0.001

    # env
    episode_length_s = 10.0
    decimation = 2

    # spaces
    observation_space = 49+6 #TODO
    action_space = 12
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 49+6

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "base_link"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = BW_CFG.replace(prim_path="/World/envs/env_.*/Robot")


# @configclass
# class BwAmpDanceEnvCfg(BwAmpEnvCfg):
#     motion_file = os.path.join(MOTIONS_DIR, "Bw_dance.npz")
    
@configclass
class BwAmpWalkEnvCfg(BwAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "bw_walk_npy/bw.npz")