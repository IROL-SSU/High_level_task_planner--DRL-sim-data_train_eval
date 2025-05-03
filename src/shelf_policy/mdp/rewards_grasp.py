#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation, RigidObjectCollection
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer, ContactSensor
from omni.isaac.lab.utils.math import combine_frame_transforms, matrix_from_quat, euler_xyz_from_quat, quat_mul, transform_points
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reward_for_hand_reaching(env: ManagerBasedRLEnv,
                object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                alpha: float = -10.0,):
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids].clone()
    ee_pos_w = ee.data.target_pos_w.clone()

    
    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] 
    offset_pos[:, 1] = offset_pos[:, 1]
    offset_pos[:, 2] = offset_pos[:, 2] + 0.05

    distance = torch.norm((offset_pos[:, :3] - ee_pos_w[..., 0,:3]), dim=-1, p=2)
 
    reward = torch.exp(alpha * distance)
    
    # print(reward)
    
    # print(f"distance: {distance}")
    # print(f"offset pos: {offset_pos}")
    # print(f"ee pos: {ee_pos_w}")

    return reward


def align_ee_target(env: ManagerBasedRLEnv,
                    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
                    ) -> torch.Tensor:      
    
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    shelf: RigidObject = env.scene[shelf_cfg.name]
    

    shelf_quat = shelf.data.root_quat_w[:, :4]
    ee_tcp_quat = ee.data.target_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    shelf_rot_mat = matrix_from_quat(shelf_quat)

    shelf_z_axis = shelf_rot_mat[..., 2]
    ee_tcp_z_axis = ee_tcp_rot_mat[..., 2]

    align = torch.bmm(ee_tcp_z_axis.unsqueeze(1), shelf_z_axis.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    
    return torch.sign(align) * align**2

def grasp_object(
    env:ManagerBasedRLEnv, 
    threshold: float, 
    open_joint_pos: float, 
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    offset_pos = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids].clone()

    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    # print("gripper: {}".format(gripper_joint_pos))
    
    offset_pos[:,0] = offset_pos[:, 0] 
    offset_pos[:,1] = offset_pos[:, 1] 
    offset_pos[:,2] = offset_pos[:, 2] + 0.05
    
    distance = torch.norm(offset_pos - ee_tcp_pos, dim=-1, p=2)

    is_close = distance <= threshold

    return is_close * torch.sum(gripper_joint_pos - open_joint_pos, dim=-1)

def object_lift(env: ManagerBasedRLEnv, 
                threshold: float, 
                object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    offset_pos = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids].clone()
    offset_pos[:,0] = offset_pos[:, 0] 
    offset_pos[:,1] = offset_pos[:, 1] 
    offset_pos[:,2] = offset_pos[:, 2] + 0.05
    
    distance = torch.norm(offset_pos - ee_frame.data.target_pos_w[..., 0, :], dim=-1, p=2)

    return torch.where(offset_pos[:, 2]> threshold, 1.0, 0.0)

def homing_reward(env: ManagerBasedRLEnv,
                  gripper_cfg: SceneEntityCfg,
                  object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                  ):
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # obtain the desired and current positions
    offset_pos = target.data.root_pos_w.clone()
    offset_pos[:,0] = offset_pos[:, 0] 
    offset_pos[:,1] = offset_pos[:, 1] 
    offset_pos[:,2] = offset_pos[:, 2] + 0.03
    
    distance = torch.norm(offset_pos - ee_frame.data.target_pos_w[..., 0, :], dim=-1, p=2)


    gripper_joint_pos = env.scene[gripper_cfg.name].data.joint_pos[:, gripper_cfg.joint_ids]
    
    joint_pos_error = torch.sum(torch.abs(robot.data.joint_pos[:, : 5] - robot.data.default_joint_pos[:, :5]), dim=1)
    reward_for_home_pose = 1.0 - torch.tanh(joint_pos_error/2.0)

    # print(f"gripper_joint: {torch.sum(gripper_joint_pos, dim=-1)}")
    # print(f"distance: {distance}")
    
    return torch.where(torch.sum(gripper_joint_pos, dim=-1) > 0.4,  (offset_pos[:, 2] > 1.03)*(offset_pos[:, 2] < 1.08)*reward_for_home_pose, 0)
    
class shelf_Collision(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")
        wrist_frame_cfg= SceneEntityCfg("wrist_frame")
        finger_frame_cfg = SceneEntityCfg("finger_frame")

        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._finger: FrameTransformer = env.scene[finger_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self._wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
        self._initial_shelf_pos = self._shelf.data.default_root_state[:, :3] + env.scene.env_origins


    
    def __call__(self, env: ManagerBasedRLEnv,):

        collision = self.shelf_collision_penalty(env)
        collision_dynamic = self.shelf_dynamic_penalty(env)
        return collision + collision_dynamic

    def shelf_collision_penalty(self,env: ManagerBasedRLEnv,) -> torch.Tensor:

        shelf_vel = self._shelf.data.root_vel_w
        shelf_delta = self._shelf.data.root_pos_w - self._initial_shelf_pos

        moved = torch.where((torch.norm(shelf_delta , dim=-1, p=2) + torch.norm(shelf_vel , dim=-1, p=2))> 0.005, 1.0, 0.0)
        return moved

    def shelf_dynamic_penalty(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        shelf_pos_w = self._shelf.data.root_pos_w .clone()
        shelf_pos_w[:,2] = shelf_pos_w[:, 2] + 1.06

        distance = torch.norm(shelf_pos_w - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)
        zeta = torch.where(distance < 0.2, 1, 0)
        dst_l_shelf = self._finger.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])
        dst_r_shelf = self._finger.data.target_pos_w[..., 1, 2] - (shelf_pos_w[:,2])
        dst_wrist_shelf = self._wrist.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])


        reward_l = 1 - dst_l_shelf / 0.02
        reward_r = 1 - dst_r_shelf / 0.02
        reward_wrist = 1 - dst_wrist_shelf / 0.08


        reward_l = torch.clamp(reward_l, 0, 1)
        reward_r = torch.clamp(reward_r, 0, 1)
        reward_wrist = torch.clamp(reward_wrist, 0, 1)

        R = zeta * (reward_l + reward_r + reward_wrist)

        return R
    
