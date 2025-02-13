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
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reaching_rew(env: ManagerBasedRLEnv, 
                 ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                 command_name: str = MISSING,):
    
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Extract the end-effector's position and orientation in world frame
    ee_pos_w = ee.data.target_pos_w[..., 0, :]
    ee_quat_w = ee.data.target_quat_w[..., 0, :]

    # Extract the desired position in body frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(ee_pos_w, ee_quat_w, des_pos_b)

    distance = torch.norm((des_pos_w - ee_pos_w), dim=-1, p=2)
    reward = torch.exp(-1.2 * distance)

    return reward


def orientation_command_error(env: ManagerBasedRLEnv, 
                              command_name: str = MISSING, 
                              ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(ee.data.target_quat_w[:, 0, :], des_quat_b)
    curr_quat_w = ee.data.target_quat_w[:, 0, :]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)