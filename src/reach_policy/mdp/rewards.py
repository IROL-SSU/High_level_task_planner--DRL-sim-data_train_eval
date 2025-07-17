#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer, ContactSensor
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, matrix_from_quat
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reaching_rew(env: ManagerBasedRLEnv,
                 robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
                 ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                 command_name: str = MISSING,):
    

    robot: RigidObject = env.scene[robot_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Extract the end-effector's position and orientation in world frame
    ee_pos_w = ee.data.target_pos_w[..., 0, :]
    ee_quat_w = ee.data.target_quat_w[..., 0, :]

    # Extract the desired position in body frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # print(f"ee_pos_w: {ee_pos_w}")
    # print(f"des_pos_b: {des_pos_b}")
    # print(f"des_pos_w: {des_pos_w}")

    # print(command[:, :])

    distance = torch.norm((des_pos_w - ee_pos_w), dim=-1, p=2)
    reward = torch.exp(-10.0 * distance)

    return reward

def align_ee_target(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:      

        # extract the asset (to enable type hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        # quat_err = quat_error_magnitude(self._initial_ee_quat[..., 0, :], ee_tcp_quat)
        # return 1.0 - torch.tanh(quat_err)
        ee_tcp_quat = ee.data.target_quat_w[..., 0, :]
        ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)

        des_quat_b = command[:, 3:7]
        des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
        des_rot_mat = matrix_from_quat(des_quat_w)

        ee_tcp_x = ee_tcp_rot_mat[..., 0]
        ee_tcp_z = ee_tcp_rot_mat[..., 2]
        des_z = des_rot_mat[..., 2]

        # print(f"x: {init_rot_mat[..., 0]}, y: {init_rot_mat[..., 1]}, z: {init_rot_mat[..., 2]}")

        align_z = torch.bmm(ee_tcp_x.unsqueeze(1), des_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        return torch.sign(align_z) * align_z**2


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    return quat_error_magnitude(curr_quat_w, des_quat_w)


def joint_acc_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)
