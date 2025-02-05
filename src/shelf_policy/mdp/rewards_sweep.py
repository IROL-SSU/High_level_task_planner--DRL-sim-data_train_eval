from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms, matrix_from_quat, euler_xyz_from_quat, quat_mul, transform_points, quat_error_magnitude

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

class ee_Align(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._initial_ee_quat = self._ee.data.target_quat_w.clone()
        

    def __call__(self, env: ManagerBasedRLEnv,):

        align = self.align_ee_target(env)

        return align

    def align_ee_target(self, env: ManagerBasedRLEnv,) -> torch.Tensor:      
        
        offset_pos = self._target.data.root_pos_w.clone()
        offset_pos[:,0] = offset_pos[:, 0] 
        offset_pos[:,1] = offset_pos[:, 1] - 0.09
        offset_pos[:,2] = offset_pos[:, 2] + 0.05

        # distance = torch.norm(offset_pos - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)

        reset_mask = env.episode_length_buf == 1
        self._initial_ee_quat[reset_mask] = self._ee.data.target_quat_w[reset_mask, :].clone()
        ee_tcp_quat = self._ee.data.target_quat_w[..., 0, :]
        
        # quat_err = quat_error_magnitude(self._initial_ee_quat[..., 0, :], ee_tcp_quat)
        # return 1.0 - torch.tanh(quat_err)

        ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
        init_rot_mat = matrix_from_quat(self._initial_ee_quat[..., 0, :])

        init_ee_z = init_rot_mat[..., 2]
        ee_tcp_z = ee_tcp_rot_mat[..., 2]

        align_z = torch.bmm(ee_tcp_z.unsqueeze(1), init_ee_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return torch.sign(align_z) * align_z**2 

def reaching_rew(env: ManagerBasedRLEnv,
                object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
                ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
    target: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_cur_pos_w = target.data.root_pos_w[:, :3]
    # # Fingertips position: (num_envs, n_fingertips, 3)
    # ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    # lfinger_pos = ee_fingertips_w[..., 0, :]
    # rfinger_pos = ee_fingertips_w[..., 1, :]
    
    ee_pos_w = ee.data.target_pos_w[..., 0, :]

    
    offset_pos = obj_cur_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] 
    offset_pos[:, 1] = offset_pos[:, 1] - 0.09
    offset_pos[:, 2] = offset_pos[:, 2] + 0.06

    distance = torch.norm((offset_pos - ee_pos_w), dim=-1, p=2)

    reward = torch.exp(-1.2 * distance)
    
    # print(f"offset pos: {offset_pos}")
    # print(f"ee pos: {rfinger_pos}")

    return reward
    

def pushing_target(env: ManagerBasedRLEnv, command_name: str,):
    object_cfg = SceneEntityCfg("cup")
    ee_frame_cfg = SceneEntityCfg("ee_frame")

    target: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    curr_pos_w = target.data.root_pos_w[:, :3]  # type: ignore

    curr_v_w = target.data.root_lin_vel_w[:, :3]
    
    ee_pos_w = ee.data.target_pos_w[..., 0, :]
    offset_pos = curr_pos_w.clone()
    offset_pos[:, 0] = offset_pos[:, 0] 
    offset_pos[:, 1] = offset_pos[:, 1] - 0.09
    offset_pos[:, 2] = offset_pos[:, 2] + 0.06

    distance = torch.norm((des_pos_w - curr_pos_w), dim=-1, p=2)
    zeta_m = torch.where((torch.norm(offset_pos - ee_pos_w, dim=-1, p=2)) < 0.03 , 1, 0)
    vel_rew = torch.where(torch.abs(curr_v_w[:, 1]) < 0.5, 4 * torch.abs(curr_v_w[:, 1]) , -1)
    reward = (1 - distance/0.15) + vel_rew

    # print(f"offset pos: {offset_pos}")
    # print(f"ee pos: {ee_pos_w}")

    # print(f"ee_distance: {torch.norm(offset_pos - ee_pos_w, dim=-1, p=2)}")
    # print(f"current position: {curr_pos_w}")
    # print(f"goal_position: {des_pos_w}")
    # print(f"pushing distance: {distance}")
    return torch.where(distance < 0.03, reward, zeta_m * reward)

def pushing_bonus(env: ManagerBasedRLEnv, 
                  command_name: str,
                  object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
                  ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
    
    target: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    curr_pos_w = target.data.root_pos_w[:, :3]  
    distance = torch.norm((des_pos_w - curr_pos_w), dim=-1, p=2)

    return torch.where(distance < 0.03, 1, 0)

def homing_reward(env: ManagerBasedRLEnv,
                  command_name: str,
                  object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    
    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    curr_pos_w = target.data.root_pos_w[:, :3]  
    distance = torch.norm((des_pos_w - curr_pos_w), dim=-1, p=2)
    joint_pos_error = torch.sum(torch.abs(robot.data.joint_pos[:, : 6] - robot.data.default_joint_pos[:, :6]), dim=1)
    reward_for_home_pose = 1.0 - torch.tanh(joint_pos_error/2.0)
    
    # print(f"joint error: {joint_pos_error}")
    # reward_for_home_pose = torch.exp(-0.5 * joint_pos_error)
    # print(f"joint reward: {torch.where(distance < 0.04, reward_for_home_pose, 0)}")
    
    return torch.where(distance < 0.03, reward_for_home_pose, 0)

# def homing_reward(env: ManagerBasedRLEnv,
#                   object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
#                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#                   ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame") 
#                   ):
#     ee: FrameTransformer = env.scene[ee_frame_cfg.name]
#     robot: Articulation = env.scene[asset_cfg.name]
#     target: RigidObject = env.scene[object_cfg.name]

#     target_command = env.command_manager.get_command("target_goal_pos")
#     ee_command = env.command_manager.get_command("ee_goal_pos")

#     object_pos_w = target.data.root_pos_w[:, :3].clone()
#     obj_distance = torch.norm(object_pos_w - target_command[:, :3], dim=-1, p=2)

#     vector_to_initial_pos = ee.data.target_pos_w[..., 0, :3] - ee_command[:, :3]
#     ee_distance = torch.norm(vector_to_initial_pos, dim=-1, p=2)

#     reward_for_home_pose = torch.exp(-1.2 * ee_distance)

    
#     return torch.where(obj_distance < 0.04, reward_for_home_pose, 0)

def homing_bonus(env: ManagerBasedRLEnv,
                  command_name: str,
                  object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),):
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]

    command = env.command_manager.get_command(command_name)
    
    # obtain the desired and current positions
    des_pos_w = command[:, :3]
    curr_pos_w = target.data.root_pos_w[:, :3]  
    distance = torch.norm((des_pos_w - curr_pos_w), dim=-1, p=2)
    joint_pos_error = torch.sum(torch.abs(robot.data.joint_pos[:, : 6] - robot.data.default_joint_pos[:, :6]), dim=1)
    # print(robot.data.joint_names)
    
    homing_bonus_far = torch.where(joint_pos_error < 1.0, 0.3, 0) 
    homing_bonus_close = torch.where(joint_pos_error < 0.5, 0.7, 0)
    homing_bonus = homing_bonus_far + homing_bonus_close
    return torch.where(distance < 0.04, homing_bonus, 0)

# def homing_bonus(env: ManagerBasedRLEnv,
#                   object_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
#                   ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
#     ee: FrameTransformer = env.scene[ee_frame_cfg.name]
#     target: RigidObject = env.scene[object_cfg.name]

#     target_command = env.command_manager.get_command("target_goal_pos")
#     ee_command = env.command_manager.get_command("ee_goal_pos")

#     object_pos_w = target.data.root_pos_w[:, :3].clone()
#     obj_distance = torch.norm(object_pos_w - target_command[:, :3], dim=-1, p=2)

#     vector_to_initial_pos = ee.data.target_pos_w[..., 0, :3] - ee_command[:, :3]
#     ee_distance = torch.norm(vector_to_initial_pos, dim=-1, p=2)

#     homing_bonus_far = torch.where(ee_distance < 0.1, 0.5, 0) 
#     homing_bonus_close = torch.where(ee_distance < 0.02, 1, 0)
#     homing_bonus = homing_bonus_far + homing_bonus_close

#     return torch.where(obj_distance < 0.04, homing_bonus, 0)
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, :6]), dim=1)

class Home_pose(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")          

        self._target: RigidObject = env.scene[self.object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._initial_ee_pos = self._ee.data.target_pos_w.clone() 
    
    def __call__(self, env: ManagerBasedRLEnv,):
        homing = self.home_pose(env)
        return homing
    
    def home_pose(self, env:ManagerBasedRLEnv) -> torch.Tensor:

        # current object state
        object_pos_w = self._target.data.root_pos_w[:, :3].clone()

        # initial object & ee state
        reset_mask = env.episode_length_buf == 1
        self._initial_ee_pos[reset_mask] = self._ee.data.target_pos_w[reset_mask, :].clone()

        command = env.command_manager.get_command("target_goal_pos")
        obj_distance = torch.norm(object_pos_w - command[:, :3], dim=-1, p=2)
        vector_to_initial_pos = self._initial_ee_pos[..., 0, :3] - self._ee.data.target_pos_w[..., 0, :3]
        ee_distance = torch.norm(vector_to_initial_pos, dim=-1, p=2)
        # print(f"ee distance: {ee_distance}")
        reward_for_home_pose = torch.exp(-1.2 * ee_distance)
        return torch.where(obj_distance<0.04, reward_for_home_pose, 0)
    
class Homing_bonus(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")          

        self._target: RigidObject = env.scene[self.object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._initial_ee_pos = self._ee.data.target_pos_w.clone() 
    
    def __call__(self, env: ManagerBasedRLEnv,):
        homing = self.home_bonus(env)
        return homing
    
    def home_bonus(self, env:ManagerBasedRLEnv) -> torch.Tensor:

        # current object state
        object_pos_w = self._target.data.root_pos_w[:, :3].clone()

        # initial object & ee state
        reset_mask = env.episode_length_buf == 1
        self._initial_ee_pos[reset_mask] = self._ee.data.target_pos_w[reset_mask, :].clone()

        command = env.command_manager.get_command("target_goal_pos")
        obj_distance = torch.norm(object_pos_w - command[:, :3], dim=-1, p=2)
        vector_to_initial_pos = self._initial_ee_pos[..., 0, :3] - self._ee.data.target_pos_w[..., 0, :3]
        ee_distance = torch.norm(vector_to_initial_pos, dim=-1, p=2)
        homing_bonus_far = torch.where(ee_distance < 0.1, 0.5, 0) 
        homing_bonus_close = torch.where(ee_distance < 0.02, 1, 0)
        homing_bonus = homing_bonus_far + homing_bonus_close
        return torch.where(obj_distance<0.02, homing_bonus, 0)

class shelf_Collision(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        shelf_cfg = SceneEntityCfg("shelf")
        wrist_frame_cfg= SceneEntityCfg("wrist_frame")
        finger_frame_cfg = SceneEntityCfg("finger_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]
        self._finger: FrameTransformer = env.scene[finger_frame_cfg.name]
        self._shelf: RigidObject = env.scene[shelf_cfg.name]
        self._wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
        self._initial_shelf_pos = self._shelf.data.default_root_state[:, :3] + env.scene.env_origins

        self._target_last_w = self._target.data.root_pos_w.clone()

    
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
        shelf_pos_w[:,2] = shelf_pos_w[:, 2] + 0.98

        distance = torch.norm(shelf_pos_w - self._ee.data.target_pos_w[..., 0, :], dim=-1, p=2)
        zeta = torch.where(distance < 0.2, 1, 0)
        dst_l_shelf = self._finger.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])
        dst_r_shelf = self._finger.data.target_pos_w[..., 1, 2] - (shelf_pos_w[:,2])
        dst_wrist_shelf = self._wrist.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])


        reward_l = 1 - dst_l_shelf / 0.02
        reward_r = 1 - dst_l_shelf / 0.02
        reward_wrist = 1 - dst_wrist_shelf / 0.07


        reward_l = torch.clamp(reward_l, 0, 1)
        reward_r = torch.clamp(reward_r, 0, 1)
        reward_wrist = torch.clamp(reward_wrist, 0, 1)

        R = zeta * (reward_l + reward_r + reward_wrist)
        
        # print(dst_ee_shelf)
        # print(f"ee: {reward_ee}")
        # print(f"wrist: {reward_wrist}")
        
        # print(f"reward: {R}")


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

        self._target_last_w = self._target.data.root_pos_w.clone()

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