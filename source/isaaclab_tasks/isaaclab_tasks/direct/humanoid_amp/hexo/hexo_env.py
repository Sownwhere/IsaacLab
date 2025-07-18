# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import math
import torch
from collections.abc import Sequence


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.utils.math import quat_rotate

from .hexo_env_cfg import HexoEnvCfg
from ..motions.python import MotionLoader

class HexoEnv(DirectMARLEnv):
    cfg: HexoEnvCfg

    def __init__(self, cfg: HexoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._humanoid_dof_idx, _ = self.robot.find_joints(self.cfg.humanoid_dof_name)
        self._exo_dof_idx, _ = self.robot.find_joints(self.cfg.exo_dof_name)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        print("self._motion_loader:", self._motion_loader)

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
        print("self.motion_key_body_indexes: " ,self.motion_key_body_indexes)
        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = {k: v.clone() for k, v in actions.items()}


    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(
            self.actions["humanoid"] * self.cfg.humanoid_action_scale, joint_ids=self._humanoid_dof_idx
        )
        self.robot.set_joint_effort_target(
            self.actions["exo"] * self.cfg.exo_action_scale, joint_ids=self._exo_dof_idx
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # humanoid：使用 compute_obs 获取完整观测
        humanoid_obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )
        # print("++++++++++++++++")
        # === 维护 AMP 历史 buffer ===
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = humanoid_obs.clone()  # 最新的放在最前面

        # 存入 extras 供外部使用
        self.extras = {
            "amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)
        }
        # print(self.extras["amp_obs"].shape)
        # print("++++++++++++++++")

        # exo：保持原来的简单观测
        exo_obs = torch.cat(
            (
                self.joint_pos[:, self._exo_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._exo_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._exo_dof_idx[1]].unsqueeze(dim=1),
                self.joint_vel[:, self._exo_dof_idx[1]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        # print("++++++++++++++++")
        # 组合返回
        return {
            "exo": exo_obs,
            "humanoid": humanoid_obs,
            
        }


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

    # def _reset_idx(self, env_ids: Sequence[int] | None):
    #     if env_ids is None:
    #         env_ids = self.robot._ALL_INDICES
    #     super()._reset_idx(env_ids)

    #     joint_pos = self.robot.data.default_joint_pos[env_ids]
    #     joint_pos[:, self._exo_dof_idx] += sample_uniform(
    #         self.cfg.initial_exo_angle_range[0] * math.pi,
    #         self.cfg.initial_exo_angle_range[1] * math.pi,
    #         joint_pos[:, self._exo_dof_idx].shape,
    #         joint_pos.device,
    #     )
    #     joint_vel = self.robot.data.default_joint_vel[env_ids]

    #     default_root_state = self.robot.data.default_root_state[env_ids]
    #     default_root_state[:, :3] += self.scene.env_origins[env_ids]

    #     self.joint_pos[env_ids] = joint_pos
    #     self.joint_vel[env_ids] = joint_vel

    #     self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    #     self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    #     self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["base_link"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)

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

@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs