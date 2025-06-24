import os
from .hexo_cfg import HEXO_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../motions")

@configclass
class HexoEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    possible_agents = ["humanoid", "exo"]
    # action_spaces = {"humanoid": 1, "exo": 1}
    # observation_spaces = {"humanoid": 2, "exo": 2}
    state_space = -1

    # spaces
    # observation_space = 49+6 #TODO
    observation_spaces = {"humanoid": 49+6, "exo": 4}
    action_spaces = {"humanoid": 12, "exo": 2}
    # action_space = 12
    # state_space = 0
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


    # robot
    robot_cfg: ArticulationCfg = HEXO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    humanoid_dof_name = [
        'left_leg_pitch_joint',
        'left_leg_roll_joint',
        'left_leg_yaw_joint',
        'left_knee_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
        'right_leg_pitch_joint',
        'right_leg_roll_joint',
        'right_leg_yaw_joint',
        'right_knee_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint'
    ]
    exo_dof_name = 'right_ankle_roll_joint'

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    initial_humanoid_angle_range = [-0.25, 0.25]  # the range in which the humanoid angle is sampled from on reset [rad]
    initial_exo_angle_range = [-0.25, 0.25]  # the range in which the exo angle is sampled from on reset [rad]

    # action scales
    humanoid_action_scale = 100.0  # [N]
    exo_action_scale = 50.0  # [Nm]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_humanoid_pos = 0
    rew_scale_humanoid_vel = -0.01
    rew_scale_exo_pos = -1.0
    rew_scale_exo_vel = -0.01
    rew_termination = -1.0
    rew_action_l2 = -0.5
    rew_joint_pos_limits = -0.5
    rew_joint_acc_l2 = -0.001
    rew_joint_vel_l2 = -0.001

    # env
    episode_length_s = 10.0
    decimation = 2

@configclass
class HexoWalkEnvCfg(HexoEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "bw_walk_npy/bw.npz")