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

def hand_velocity_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.1):
    robot: Articulation = env.scene[asset_cfg.name]

    #ee lin vel
    ee_lin_vel_w = robot.data.body_state_w[:, 10,7:10].clone()
    ee_ang_vel_w = robot.data.body_state_w[:, 10, 10:13].clone()
    termination = (torch.norm(ee_lin_vel_w, dim=-1, p=2) > threshold) |  (torch.norm(ee_ang_vel_w, dim=-1, p=2) > 2.0)
    
    # print(robot.data.body_pos_w[:,10,:])
    return termination


def joint_velocity_termination(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float=0.5):
    robot: Articulation = env.scene[asset_cfg.name]

    return torch.any(torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids]) > threshold, dim=1)


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