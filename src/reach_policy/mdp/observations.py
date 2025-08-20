from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_unique
from omni.isaac.lab.sensors import FrameTransformerData, ContactSensorData
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def ee_pose_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    robot: Articulation = env.scene[robot_cfg.name]
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos_w = ee_tf_data.target_pos_w[..., 0, :]
    ee_quat_w = ee_tf_data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w, ee_quat_w
    )
    
    # print(f"position: {ee_pos_b}")
    # print(f"orientation: {ee_quat_b}")
    # print(f"robot_joint_vel: {robot.data.joint_vel}")
    return  torch.concat((ee_pos_b, ee_quat_b), dim=1)

def MA_joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joint_nums: float = 6) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(f"joint: {asset.data.joint_pos[:, :6]}")
    return asset.data.joint_pos[:, :joint_nums] - asset.data.default_joint_pos[:, :joint_nums]


def MA_joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(f"joint velocity: {asset.data.joint_vel[:, :6]}")
    return asset.data.joint_vel[:, :6] - asset.data.default_joint_vel[:, :6]