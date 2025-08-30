from __future__ import annotations
from dataclasses import MISSING

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, RigidObjectCollection, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, transform_points, euler_xyz_from_quat
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import FrameTransformer, ContactSensor

from src_utils.shelf_utils import normalize_angle

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def success_sweeping(env: ManagerBasedRLEnv,
                     object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     command_name: str = "target_goal_pos",
                     sweeping_threshold: float = 0.02,
                     homing_threshold: float = 0.5):
    
    robot: Articulation = env.scene[asset_cfg.name]

    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_w = command[:, :3]

    # Get the target IDs directly from the environment tensor
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs, )

    # Get the world state(position, orientation, linear velocity, angular velocity); R^13
    target_pos_w = object_collection.data.object_pos_w[torch.arange(env.scene.num_envs), target_ids]

    # joint_pos_error = torch.sum(torch.abs(robot.data.joint_pos[:, : 6] - robot.data.default_joint_pos[:, :6]), dim=1)
    
    distance = torch.norm((des_pos_w - target_pos_w), dim=-1, p=2)

    # print(joint_pos_error)

    # success = (distance < sweeping_threshold) & (joint_pos_error < homing_threshold)

    # print((joint_pos_error < homing_threshold))
    # print(success)

    return (distance < sweeping_threshold)


def drop_object_termination(env: ManagerBasedRLEnv,
                            object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                            params: float = -1.0,
                            height_condition: float = MISSING,
                            rotation_condition: float = MISSING,
                            ):

    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    target_ids = env.target_id.squeeze(-1).long()  # Shape: (num_envs,)

    objects_pose = object_collection.data.object_link_state_w  # (N, num_objects, 13)

    num_envs, num_objects = objects_pose.shape[:2]  # 환경 개수, 물체 개수

    # ✅ 모든 물체의 높이값 가져오기 (z축 위치)
    heights = objects_pose[..., :, 2]  # (N, num_objects)

    # ✅ 물체가 떨어졌는지 확인
    is_dropped = heights < height_condition  # (N, num_objects) -> Bool 텐서

    # ✅ 모든 물체의 quaternion (N, num_objects, 4)
    quat_tensor = objects_pose[..., :, 3:7].reshape(-1, 4)  # (N * num_objects, 4)

    # ✅ 벡터 연산으로 quaternion → Euler 변환
    roll, pitch, _ = euler_xyz_from_quat(quat_tensor)  # (N * num_objects,)

    # ✅ 원래 차원으로 복구 (N, num_objects)
    roll = roll.view(num_envs, num_objects)
    pitch = pitch.view(num_envs, num_objects)

    roll = normalize_angle(roll)
    pitch = normalize_angle(pitch)

    # if params < 0:
    #     roll[torch.arange(env.scene.num_envs), target_ids] = 0.0
    #     pitch[torch.arange(env.scene.num_envs), target_ids] = 0.0

    
    # ✅ 물체가 넘어졌는지 확인 (roll, pitch가 특정 값 이상이면 뒤집힌 것으로 간주)
    is_flipped = (torch.abs(roll) > rotation_condition) | (torch.abs(pitch) > rotation_condition)

    # ✅ 하나라도 물체가 떨어지거나 넘어졌다면 episode 종료
    episode_done = torch.any(is_dropped | is_flipped, dim=1)  # (N,)

    
    return episode_done  # (N,) -> 환경별 episode 종료 여부


def shelf_collision_termination(env: ManagerBasedRLEnv,
                                shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
                                finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
                                wrist_frame_cfg: SceneEntityCfg = SceneEntityCfg("wrist_frame"),
                                threshold: float = MISSING):
    
    shelf: RigidObject = env.scene[shelf_cfg.name]
    finger: FrameTransformer = env.scene[finger_frame_cfg.name]
    wrist: FrameTransformer = env.scene[wrist_frame_cfg.name]
    shelf_pos_w = shelf.data.root_pos_w .clone()
    shelf_pos_w[:,2] = shelf_pos_w[:, 2] + 1.05
    
    
    dst_l_shelf = finger.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])
    dst_r_shelf = finger.data.target_pos_w[..., 1, 2] - (shelf_pos_w[:,2])
    dst_wrist_shelf = wrist.data.target_pos_w[..., 0, 2] - (shelf_pos_w[:,2])
    # shelf_contact: ContactSensor = env.scene[shelf_contact_cfg.name]


    # print(contact_sensor.data.net_forces_w)
    shelf_vel = shelf.data.root_vel_w
    shelf_vel.sum()
    
    termination = (torch.norm(shelf_vel , dim=-1, p=2)> threshold) | (dst_l_shelf < 0.01) | (dst_r_shelf < 0.01) | (dst_wrist_shelf < 0.07)
    

    # print(shelf_contact.data.net_forces_w[:,0, 2])
    return termination

def hand_velocity_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.1):
    robot: Articulation = env.scene[asset_cfg.name]

    #ee lin vel
    ee_lin_vel_w = robot.data.body_state_w[:, 12,7:10].clone()
    ee_ang_vel_w = robot.data.body_state_w[:, 12, 10:13].clone()
    termination = (torch.norm(ee_lin_vel_w, dim=-1, p=2) > threshold) |  (torch.norm(ee_ang_vel_w, dim=-1, p=2) > 2.0)

    return termination




