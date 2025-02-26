from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import MISSING

from omni.isaac.lab.assets import RigidObject, Articulation, RigidObjectCollection
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, matrix_from_quat, euler_xyz_from_quat, quat_mul, transform_points, quat_error_magnitude

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def reward_for_hand_reaching(env: ManagerBasedRLEnv,
                object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                object_id_dict_rev: dict = MISSING,
                ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_link_pos_w[torch.arange(env.scene.num_envs), target_ids]

    

    ee_pos_w = ee.data.target_pos_w


    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] 
    offset_pos[:, 1] = offset_pos[:, 1] - 0.07
    offset_pos[:, 2] = offset_pos[:, 2] + 0.07

    distance = torch.norm((offset_pos[:, :3] - ee_pos_w[..., 0,:3]), dim=-1, p=2)


    reward = torch.exp(-1.2 * distance)
    
    return reward

def reward_for_hand_ori(env: ManagerBasedRLEnv,
                        object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                        object_id_dict_rev: dict = MISSING,
                        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]

    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    
   # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_link_pos_w[torch.arange(env.scene.num_envs), target_ids]

    ee_pos_w = ee.data.target_pos_w


    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] 
    offset_pos[:, 1] = offset_pos[:, 1] - 0.07
    offset_pos[:, 2] = offset_pos[:, 2] + 0.07

    distance = torch.norm((offset_pos[:, :3] - ee_pos_w[..., 0,:3]), dim=-1, p=2)

    hand_ori = robot.data.joint_pos[:, 5]
    hand_init_ori = robot.data.default_joint_pos[:, 5]

    joint_pos_error = torch.abs(hand_init_ori- hand_init_ori)

    reward_for_ori_pose = 1.0 - torch.tanh(joint_pos_error/2.0)

    return torch.where(distance < 0.2, reward_for_ori_pose, 0)