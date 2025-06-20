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
    id="Isaac-Humanoid-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-AMP-Run-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Humanoid-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G1-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_env_cfg:G1AmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_dance_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_dance_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-G11-AMP-Walk-Direct-v11",
    entry_point=f"{__name__}.g1.g1_amp_env:G1AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1.g1_amp_env_cfg:G1AmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_g1_walk_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_g1_walk_amp_cfg.yaml",
    },
)



gym.register(
    id="Isaac-Bw-AMP-Walk-Direct-v0",
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
    id="Isaac-Cartpole-Direct-v0",
    entry_point=f"{__name__}.exo.bw_exo_env:BwExoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exo.bw_exo_env:BwExoEnvCfg",

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