# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents
# from .exo import BwExoEnv
# from .exo import BwExoEnvCfg
##
# Register Gym environments.
##

gym.register(
    id="Bw",
    entry_point=f"{__name__}.bw_amp.bw_amp_env:BwAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bw_amp.bw_amp_env_cfg:BwAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_bw_walk_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bw_walk_amp_cfg.yaml",
    },
)


gym.register(
    id="Bw-Exo",
    entry_point=f"{__name__}.bw_exo_amp.bw_exo_amp_env:BwExoAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bw_exo_amp.bw_exo_amp_env_cfg:BwExoAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_bw_exo_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_bw_exo_amp_cfg.yaml",
    },
)


gym.register(
    id="exo",
    entry_point=f"{__name__}.exo.bw_exo_env:BwExoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exo.bw_exo_env_cfg:BwExoEnvCfg",

        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_exo_ppo_cfg.yaml",
    },
)

gym.register(
    id="motor",
    entry_point=f"{__name__}.motor.motor_env:MotorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.motor.motor_env:MotorEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_exo_ppo_cfg.yaml",
    },
)


gym.register(
    id="marl",
    entry_point=f"{__name__}.multiple_agent.ma_env:MaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multiple_agent.ma_env_cfg:Exo1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_exo_ppo_cfg.yaml",
    },
)

gym.register(
    id="hexo",
    entry_point=f"{__name__}.hexo.hexo_env:HexoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexo.hexo_env_cfg:HexoEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_mippo_cfg_entry_point":f"{agents.__name__}:skrl_mippo_cfg.yaml",
    },
)
