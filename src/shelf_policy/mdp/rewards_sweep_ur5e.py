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
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids]
    ee_pos_w = ee.data.target_pos_w

    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] + 0.01
    offset_pos[:, 1] = offset_pos[:, 1] - 0.09
    offset_pos[:, 2] = offset_pos[:, 2] + 0.05

    distance = torch.norm((offset_pos[:, :3] - ee_pos_w[..., 0,:3]), dim=-1, p=2)


    # print(f"object: {offset_pos}")
    # print(f"hand: {ee_pos_w}")
    # print(f"distance: {distance}")


    reward = torch.exp(-1.2 * distance)
    
    return reward

class ee_Align(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection")
        self.object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._initial_ee_quat = self._ee.data.target_quat_w.clone()
        

    def __call__(self, env: ManagerBasedRLEnv,):

        align = self.align_ee_target(env)

        return align

    def align_ee_target(self, env: ManagerBasedRLEnv,) -> torch.Tensor:      

        reset_mask = env.episode_length_buf == 1
        self._initial_ee_quat[reset_mask] = self._ee.data.target_quat_w[reset_mask, :].clone()
        ee_tcp_quat = self._ee.data.target_quat_w[..., 0, :]
        
        # quat_err = quat_error_magnitude(self._initial_ee_quat[..., 0, :], ee_tcp_quat)
        # return 1.0 - torch.tanh(quat_err)

        ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
        init_rot_mat = matrix_from_quat(self._initial_ee_quat[..., 0, :])

        init_ee_y = init_rot_mat[..., 1]
        ee_tcp_y = ee_tcp_rot_mat[..., 1]

        align_y = torch.bmm(ee_tcp_y.unsqueeze(1), init_ee_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return torch.sign(align_y) * align_y**2
    
def pushing_target(env: ManagerBasedRLEnv, 
                   command_name: str,
                   object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),):
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    ee_frame_cfg = SceneEntityCfg("ee_frame")

    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids]
    target_lin_vel_w = object_collection.data.object_lin_vel_w[torch.arange(env.scene.num_envs), target_ids]

    ee_pos_w = ee.data.target_pos_w[..., 0, :]
    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] + 0.01
    offset_pos[:, 1] = offset_pos[:, 1] - 0.09
    offset_pos[:, 2] = offset_pos[:, 2] + 0.05

    distance = torch.norm((des_pos_w - target_pos_w), dim=-1, p=2)
    zeta_m = torch.where((torch.norm(offset_pos - ee_pos_w, dim=-1, p=2)) < 0.04 , 1, 0)
    vel_rew = torch.where(torch.abs(target_lin_vel_w[:, 1]) < 0.04, 4 * torch.abs(target_lin_vel_w[:, 1]) , -1)
    reward = torch.where(distance < 0.03, 1.5, zeta_m *((1 - distance/0.15) + vel_rew))

    # print(f"offset pos: {offset_pos}")
    # print(f"ee pos: {ee_pos_w}")

    # print(f"ee_distance: {torch.norm(offset_pos - ee_pos_w, dim=-1, p=2)}")
    # print(f"current position: {curr_pos_w}")
    # print(f"goal_position: {des_pos_w}")
    # print(f"pushing distance: {distance}")

    return reward

def pushing_bonus(env: ManagerBasedRLEnv, 
                command_name: str,
                object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
):
    
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids]

    
    distance = torch.norm((des_pos_w - target_pos_w), dim=-1, p=2)


    return torch.where(distance < 0.03, 1, 0)

def homing_reward(env: ManagerBasedRLEnv,
                  command_name: str,
                  object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    robot: Articulation = env.scene[asset_cfg.name]
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]

    command = env.command_manager.get_command(command_name)

    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids]
    
    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    distance = torch.norm((des_pos_w - target_pos_w), dim=-1, p=2)
    joint_pos_error = torch.sum(torch.abs(robot.data.joint_pos[:, : 6] - robot.data.default_joint_pos[:, :6]), dim=1)
    reward_for_home_pose = 1.0 - torch.tanh(joint_pos_error/2.0)
    
    # print(f"joint error: {joint_pos_error}")
    # reward_for_home_pose = torch.exp(-0.5 * joint_pos_error)
    # print(f"joint reward: {torch.where(distance < 0.04, reward_for_home_pose, 0)}")
    
    return torch.where(distance < 0.03, reward_for_home_pose, 0)

    
def object_collision(env: ManagerBasedRLEnv,
                object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),)-> torch.Tensor:
    
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    objects_lin_vel_w = object_collection.data.object_lin_vel_w.clone()

    objects_lin_vel_w[torch.arange(env.scene.num_envs), target_ids,:] = 0.0

    reward = torch.tanh(torch.sum(torch.abs(objects_lin_vel_w),dim=(1,2))*10)

    return reward

    


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

        collision = self.shelf_collision_pentaly(env)
        collision_dynamic = self.shelf_dynamic_penalty(env)
        return collision + collision_dynamic

    def shelf_collision_pentaly(self,env: ManagerBasedRLEnv,) -> torch.Tensor:

        shelf_vel = self._shelf.data.root_lin_vel_w
        shelf_delta = self._shelf.data.root_pos_w - self._initial_shelf_pos

        moved = torch.where((torch.norm(shelf_delta , dim=-1, p=2) + torch.norm(shelf_vel , dim=-1, p=2))> 0.005, 1.0, 0.0)
        return moved

    def shelf_dynamic_penalty(self, env: ManagerBasedRLEnv,) -> torch.Tensor:
        shelf_pos_w = self._shelf.data.root_pos_w .clone()
        shelf_pos_w[:,2] = shelf_pos_w[:, 2] + 1.0

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
    
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, :6]), dim=1)