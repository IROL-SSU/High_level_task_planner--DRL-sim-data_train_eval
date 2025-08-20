# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING

import torch
import numpy as np
from random import shuffle, choice
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, RigidObjectCollection
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, random_yaw_orientation
from omni.isaac.lab.sensors import FrameTransformerData

from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def randomize_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_array: tuple,
    asset_dict: dict = MISSING,
    object_width_dict: dict = MISSING,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
    ceiling_height: int = MISSING,
    task_mode: str = MISSING
) -> None:
    
    rows, cols = len(pose_array[0]), len(pose_array[0][0])
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]


    if task_mode == "grasping":
        target_object_name = choice(["cup_1", "bottle_1", "can_1"])
        
    elif task_mode == "sweeping_right":
        target_object_name = choice(list(asset_dict.keys()))

    target_object_id = object_collection.find_objects(name_keys=target_object_name)
    target_width = torch.tensor(object_width_dict[target_object_name], dtype=torch.float32, device=env.device)
    env.target_id[env_ids, 0] = target_object_id[0].to(env.target_id.dtype)
    env.target_width[env_ids, 0] = target_width
    
    
    asset_keys_list: list = list(asset_dict.keys())

    pose_array_tensor = torch.tensor(pose_array, device=env.device)

    # Orientation randomization 미적용
    orientations = torch.empty((env_ids.shape[0], 4), device=env.device)
    orientations[:, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    velocities = torch.zeros((env_ids.shape[0],6), device=env.device)


    shuffle(asset_keys_list)

    for index, asset_name in enumerate(asset_keys_list):
        if asset_name == target_object_name:
            target_index = index

        pose_instance = pose_array_tensor[0, index // cols, index % cols]
        position_uncertainty = torch.empty(2, device=env.device).uniform_(-0.02, 0.02)
        positions = pose_instance[:3] + env.scene.env_origins[env_ids, 0:3]
        positions[:,:2] = positions[:,:2] + position_uncertainty
        orientations[:, :] = random_yaw_orientation(num=len(env_ids), device=env.device)
        object_ids = object_collection.find_objects(name_keys=asset_name)

        object_collection.write_object_link_state_to_sim(
            torch.cat((positions, orientations, velocities), dim=1).unsqueeze(1),
            env_ids=env_ids,
            object_ids=object_ids[0]
        )

    if task_mode == "sweeping_right":
        adjacent_indices = sweeping_right_mode(target_index, rows, cols, env.device)

        for adjacent in adjacent_indices:
            object_index = adjacent[0] * cols + adjacent[1]
            object_name = asset_keys_list[object_index]

            pose_instance = pose_array_tensor[0, adjacent[0], adjacent[1]]
            positions = pose_instance[:3] + env.scene.env_origins[env_ids, 0:3]
            positions[:, 2] = ceiling_height  # 높이 변경

            object_ids = object_collection.find_objects(name_keys=object_name)
            object_collection.write_object_link_pose_to_sim(
                torch.cat((positions, orientations), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0]
            )

    if task_mode == "grasping":

        adjacent_indices = grasping_mode(target_index, rows, cols, env.device)

        for adjacent in adjacent_indices:
            object_index = adjacent[0] * cols + adjacent[1]
            object_name = asset_keys_list[object_index]

            pose_instance = pose_array_tensor[0, adjacent[0], adjacent[1]]
            positions = pose_instance[:3] + env.scene.env_origins[env_ids, 0:3]
            positions[:, 2] = ceiling_height  # 높이 변경

            object_ids = object_collection.find_objects(name_keys=object_name)
            object_collection.write_object_link_pose_to_sim(
                torch.cat((positions, orientations), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0]
            )


def sweeping_right_mode(target_index: int, rows: int, cols: int, device):
    """
    Identify objects in front, right, and diagonal positions of the target.
    If the target is in the last row, include all objects in the front rows.
    Returns results as a GPU Tensor.
    """
    target_row = target_index // cols
    target_col = target_index % cols

    # (1) 앞쪽 찾기 (모든 앞쪽 행 포함)
    if target_row > 0:
        front_rows = torch.arange(target_row - 1, -1, -1, device=device)
        front_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col)), dim=1)
    else:
        front_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (2) 오른쪽 찾기
    if target_col < cols - 1:
        right_index = torch.tensor([[target_row, target_col + 1]], device=device)
    else:
        right_index = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (3) 우측 대각선 찾기
    if target_row > 0 and target_col < cols - 1:
        diagonal_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col + 1)), dim=1)
    else:
        diagonal_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

    # (4) 모든 결과를 GPU Tensor로 결합
    adjacent_array = torch.cat((front_indices, right_index, diagonal_indices), dim=0)

    return adjacent_array


def grasping_mode(target_index: int, rows: int, cols: int, device):
    """
    Identify objects in front, right, and diagonal positions of the target.
    If the target is in the last row, include all objects in the front rows.
    Returns results as a GPU Tensor.
    """
    target_row = target_index // cols
    target_col = target_index % cols

    # (1) 앞쪽 찾기 (모든 앞쪽 행 포함)
    if target_row > 0:
        front_rows = torch.arange(target_row - 1, -1, -1, device=device)
        front_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col)), dim=1)
    else:
        front_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

    return front_indices
