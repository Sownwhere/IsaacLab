import os
from dataclasses import MISSING

from .hexo_cfg import HEXO_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg,SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../motions")

@configclass
class HexoEnvCfg(DirectMARLEnvCfg):
    # reward
    rew_termination = -1.0
    rew_action_l2 = -0.5
    rew_joint_pos_limits = -0.5
    rew_joint_acc_l2 = -0.001
    rew_joint_vel_l2 = -0.001
    rew_exo_torque = -0.5
    exo_torque_limit = 10.0
    # env
    decimation = 2
    episode_length_s = 10.0

    possible_agents = ["exo", "humanoid" ]
    action_spaces = {"exo": 2, "humanoid": 12}
    observation_spaces = {"exo": 4, "humanoid": 49+6}
    state_space = -1
    num_amp_observations = 2
    # 7 + 3 + 3 + 12 + 12+ 12+  12+6
    amp_observation_space = 49+6 

    early_termination = True
    termination_height = 0.5

    motion_file: str = MISSING
    reference_body = "base_link"
    reset_strategy = "random"

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot_cfg: ArticulationCfg = HEXO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    exo_dof_name = ["left_ankle_pitch_joint",
                    "right_ankle_pitch_joint"]
    humanoid_dof_name = [
                          "left_leg_pitch_joint",
                          "left_leg_roll_joint",
                          "left_leg_yaw_joint",
                          "left_knee_joint",
                          "left_ankle_pitch_joint",
                          "left_ankle_roll_joint",
                          "right_leg_pitch_joint",
                          "right_leg_roll_joint",
                          "right_leg_yaw_joint",
                          "right_knee_joint",
                          "right_ankle_pitch_joint",
                          "right_ankle_roll_joint"
                        ]

    # # action scales
    # humanoid_action_scale = 100.0  # [N]
    # exo_action_scale = 50.0  # [Nm]
          
@configclass
class HexoWalkEnvCfg(HexoEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "bw_walk_npy/bw_20250807_153247.npz")
