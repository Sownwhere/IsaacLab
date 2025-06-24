# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from .hexo_env_cfg import HexoEnvCfg
from ..motions.python import MotionLoader

class HexoEnv(DirectMARLEnv):
    
    cfg: HexoEnvCfg
    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        return self._motion_loader.sample_states(num_samples)
    def __init__(self, cfg: HexoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        
        self._humanoid_dof_idx, _ = self.robot.find_joints(self.cfg.humanoid_dof_name)
        self._exo_dof_idx, _ = self.robot.find_joints(self.cfg.exo_dof_name)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
                
        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes  
        # key_body_names = ["base_link"]  
        key_body_names = [ 
            # 'left_leg_pitch_link',
            # 'left_leg_roll_link',
            'left_leg_yaw_link',
            'left_knee_link',
            # 'left_ankle_pitch_link',
            'left_ankle_roll_link',

            # 'right_leg_pitch_link',
            # 'right_leg_roll_link',
            'right_leg_yaw_link',
            'right_knee_link',
            # 'right_ankle_pitch_link',
            'right_ankle_roll_link',
        ]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        # Used to for reset strategy
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )



    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(
            self.actions["humanoid"] * self.cfg.humanoid_action_scale, joint_ids=self._humanoid_dof_idx
        )
        self.robot.set_joint_effort_target(
            self.actions["exo"] * self.cfg.exo_action_scale, joint_ids=self._exo_dof_idx
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {
            "humanoid": torch.cat(
                (
                    self.joint_pos[:, self._humanoid_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._humanoid_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            ),
            "exo": torch.cat(
                (
                    self.joint_pos[:, self._exo_dof_idx[0]].unsqueeze(dim=1),
                    self.joint_vel[:, self._exo_dof_idx[0]].unsqueeze(dim=1),
                ),
                dim=-1,
            ),
        }
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_humanoid_vel,
            self.cfg.rew_scale_exo_vel,
            self.joint_vel[:, self._humanoid_dof_idx[0]],
            self.joint_vel[:, self._exo_dof_idx[0]],
            math.prod(self.terminated_dict.values()),
        )
        return total_reward

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._humanoid_dof_idx]) > 100, dim=1)


        terminated = {agent: out_of_bounds for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._exo_dof_idx] += sample_uniform(
            self.cfg.initial_exo_angle_range[0] * math.pi,
            self.cfg.initial_exo_angle_range[1] * math.pi,
            joint_pos[:, self._exo_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_humanoid_vel: float,
    rew_scale_exo_vel: float,
    humanoid_vel: torch.Tensor,
    exo_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()

    rew_humanoid_vel = rew_scale_humanoid_vel * torch.sum(torch.abs(humanoid_vel).unsqueeze(dim=1), dim=-1)
    rew_exo_vel = rew_scale_exo_vel * torch.sum(torch.abs(exo_vel).unsqueeze(dim=1), dim=-1)

    total_reward = {
        "humanoid": rew_alive + rew_termination + rew_humanoid_vel,
        "exo": rew_alive + rew_termination + rew_exo_vel,
    }
    return total_reward
