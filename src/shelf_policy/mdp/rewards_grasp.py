#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer, ContactSensor
from omni.isaac.lab.utils.math import combine_frame_transforms, matrix_from_quat, euler_xyz_from_quat, quat_mul, transform_points
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def reaching_rew(env: ManagerBasedRLEnv,
                object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
                ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
    target: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_cur_pos_w = target.data.root_pos_w[:, :3]   
    ee_pos_w = ee.data.target_pos_w[..., 0, :]

    
    offset_pos = obj_cur_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] 
    offset_pos[:, 1] = offset_pos[:, 1]
    offset_pos[:, 2] = offset_pos[:, 2] + 0.03

    distance = torch.norm((offset_pos - ee_pos_w), dim=-1, p=2)

    reward = torch.exp(-1.2 * distance)
    
    # print(reward)
    
    # print(f"distance: {distance}")
    # print(f"offset pos: {offset_pos}")
    # print(f"ee pos: {ee_pos_w}")

    return reward


def align_ee_target(env: ManagerBasedRLEnv,
                    object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"),
                    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
                    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),) -> torch.Tensor:      
    
    target: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    shelf: RigidObject = env.scene[shelf_cfg.name]
    
    offset_pos = target.data.root_pos_w.clone()
    offset_pos[:,0] = offset_pos[:, 0] 
    offset_pos[:,1] = offset_pos[:, 1] 
    offset_pos[:,2] = offset_pos[:, 2] + 0.03

    shelf_quat = shelf.data.root_quat_w[:, :4]
    ee_tcp_quat = ee.data.target_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    shelf_rot_mat = matrix_from_quat(shelf_quat)

    shelf_z_axis = shelf_rot_mat[..., 2]
    ee_tcp_x_axis = ee_tcp_rot_mat[..., 0]

    align = torch.bmm(ee_tcp_x_axis.unsqueeze(1), shelf_z_axis.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    
    return torch.sign(align) * align**2

def grasp_handle(
     env:ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    offset_pos = env.scene["cup2"].data.root_pos_w.clone()
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
                object_cfg: SceneEntityCfg = SceneEntityCfg("cup2"), 
                ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    offset_pos = obj.data.root_pos_w.clone()
    offset_pos[:,0] = offset_pos[:, 0] 
    offset_pos[:,1] = offset_pos[:, 1] 
    offset_pos[:,2] = offset_pos[:, 2] + 0.03
    
    distance = torch.norm(offset_pos - ee_frame.data.target_pos_w[..., 0, :], dim=-1, p=2)

    return torch.where(offset_pos[..., 2]> threshold, 1.0, 0.0)

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
        object_cfg = SceneEntityCfg("cup2")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")
        wrist_frame_cfg= SceneEntityCfg("wrist_frame")
        finger_frame_cfg = SceneEntityCfg("finger_frame")
        shelf_contact_cfg = SceneEntityCfg("shelf_contact")

        
        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._finger: FrameTransformer = env.scene[finger_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self._wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
        self._initial_shelf_pos = self._shelf.data.default_root_state[:, :3] + env.scene.env_origins
        self._shelf_contact: ContactSensor = env.scene[shelf_contact_cfg.name]

        self._target_last_w = self._target.data.root_pos_w.clone()

    
    def __call__(self, env: ManagerBasedRLEnv,):

        collision = self.shelf_collision_pentaly(env)
        collision_dynamic = self.shelf_dynamic_penalty(env)
        return collision + collision_dynamic

    def shelf_collision_pentaly(self,env: ManagerBasedRLEnv,) -> torch.Tensor:

        shelf_vel = self._shelf.data.root_lin_vel_w
        shelf_delta = self._shelf.data.root_pos_w - self._initial_shelf_pos
        net_force = torch.norm(self._shelf_contact.data.net_forces_w[...,0,:], dim=1)

        moved = torch.where((torch.norm(shelf_delta , dim=-1, p=2) + torch.norm(shelf_vel , dim=-1, p=2))> 0.005, 1.0, 0.0)
        touched = torch.where(net_force>80, 1.0, 0.0 )
        return moved + touched

    def shelf_dynamic_penalty(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        shelf_pos_w = self._shelf.data.root_pos_w .clone()
        shelf_pos_w[:,2] = shelf_pos_w[:, 2] + 0.98

        distance = torch.norm(shelf_pos_w - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)
        zeta = torch.where(distance < 0.2, 1, 0)
        dst_l_shelf = self._finger.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])
        dst_r_shelf = self._finger.data.target_pos_w[..., 1, 2] - (shelf_pos_w[:,2])
        dst_wrist_shelf = self._wrist.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])


        reward_l = 1 - dst_l_shelf / 0.02
        reward_r = 1 - dst_r_shelf / 0.02
        reward_wrist = 1 - dst_wrist_shelf / 0.07


        reward_l = torch.clamp(reward_l, 0, 1)
        reward_r = torch.clamp(reward_r, 0, 1)
        reward_wrist = torch.clamp(reward_wrist, 0, 1)

        R = zeta * (reward_l + reward_r + reward_wrist)

        return R
    
    
class Object_drop(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        object2_cfg = SceneEntityCfg("cup2")
        ee_frame_cfg = SceneEntityCfg("ee_frame")


        self._target: RigidObject = env.scene[object_cfg.name]
        self._target2: RigidObject = env.scene[object2_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.1]) #0.0 0.0 0.07

    def __call__(self, env:ManagerBasedRLEnv,):
        
        drop = self.object_drop(env)
        drop2 = self.object_drop2(env)
        vel = self.object_velocity(env)
        
        return drop + drop2 + vel

    def object_drop(self, env: ManagerBasedRLEnv,)-> torch.Tensor:

        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]
        delta_z = 1.08 - offset_pos[..., 2] #0.73

        penalty_object = torch.where(delta_z > 0.01, 1, 0)
        return penalty_object
    
    def object_drop2(self, env: ManagerBasedRLEnv,)-> torch.Tensor:

        offset_pos = transform_points(self._top_offset,self._target2.data.root_pos_w, self._target2.data.root_state_w[:, 3:7] )[..., 0 , :]

        delta_z = 1.08 - offset_pos[..., 2] #0.73

        penalty_object = torch.where(delta_z > 0.01, 1, 0)
        return penalty_object
    
    def object_velocity(self, env: ManagerBasedRLEnv,)-> torch.Tensor:
        object_lin_vel_w = self._target.data.root_lin_vel_w.clone()
        object_lin_vel_norm = torch.norm(object_lin_vel_w, dim=-1, p=2)
        penalty = torch.where(object_lin_vel_norm > 1, 1, 0)
        return penalty