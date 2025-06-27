from .hexo_cfg import HEXO_CFG
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class HexoEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    possible_agents = ["humanoid", "exo"]
    action_spaces = {"humanoid": 1, "exo": 1}
    observation_spaces = {"humanoid": 2, "exo": 2}
    state_space = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = HEXO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    humanoid_dof_name = "right_leg_roll_joint"
    exo_dof_name = "left_leg_roll_joint"

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