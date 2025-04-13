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
