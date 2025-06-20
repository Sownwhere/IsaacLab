# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from .motor_cfg import MOTOR_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


@configclass
class MotorEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 2
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = MOTOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    motor1_name = "motor1_joint"
    motor2_name = "motor2_joint"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.0, replicate_physics=True)

    # reset
    max_cart_pos = 100.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    velocity_scale = 0.00001

class MotorEnv(DirectRLEnv):
    cfg: MotorEnvCfg

    def __init__(self, cfg: MotorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motor1_idx, _ = self.Motor.find_joints(self.cfg.motor1_name)
        self._motor2_idx, _ = self.Motor.find_joints(self.cfg.motor2_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.Motor.data.joint_pos
        self.joint_vel = self.Motor.data.joint_vel

    def _setup_scene(self):
        self.Motor = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["Motor"] = self.Motor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()
        # print(self.actions.shape)

    def _apply_action(self) -> None:
        self.actions[:,0] =0.001
        self.actions[:,1] =0.0
        self.Motor.set_joint_effort_target(self.actions[:,0].unsqueeze(-1), joint_ids=self._motor1_idx)
        self.Motor.set_joint_effort_target(self.actions[:,1].unsqueeze(-1), joint_ids=self._motor2_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._motor1_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._motor1_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._motor2_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._motor2_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.velocity_scale,
            self.joint_pos[:, self._motor1_idx[0]],
            self.joint_vel[:, self._motor1_idx[0]],
            self.joint_pos[:, self._motor2_idx[0]],
            self.joint_vel[:, self._motor2_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.Motor.data.joint_pos
        self.joint_vel = self.Motor.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._motor1_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(self.joint_vel[:, :] < 0, dim=1)

        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._motor2_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.Motor._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.Motor.data.default_joint_pos[env_ids]
        joint_pos[:, self._motor2_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._motor2_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.Motor.data.default_joint_vel[env_ids]

        default_root_state = self.Motor.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.Motor.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.Motor.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.Motor.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    velocity_scale: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    #当速度等于0.2 的时候，奖励最大
    # ✅ 新增：速度接近1时的奖励、
    velocity_error = torch.square(cart_vel - 20.0)
    target_speed_reward = torch.clamp(10000.0 - velocity_error, min=-3.0)  # 保证非负
    rew_target_speed = velocity_scale * target_speed_reward.squeeze(-1) * 100 
    # print("rew_target_speed", rew_target_speed)
    # ✅ 新增： pole_pos  = 0 奖励最大
    # pole_pos_reward = torch.square(pole_pos - 1)
    # rew_pole_pos = torch.clamp(3.14 - pole_pos_reward, min=-2.0)
    # # print("rew_pole_pos", rew_pole_pos.shape)

    total_reward = rew_target_speed  + rew_alive + rew_termination
    return total_reward
