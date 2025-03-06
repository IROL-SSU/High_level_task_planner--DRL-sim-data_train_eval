# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING

import torch
import numpy as np
from random import shuffle, choice, choices
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, RigidObjectCollection
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import FrameTransformerData

from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    

def randomize_scene(
    env,
    env_ids: torch.Tensor,
    pose_array: torch.Tensor,
    asset_dict: dict,
    object_id_dict: dict,
    object_id_dict_rev: dict,
    category_mapping: dict,
    ceiling_height: float,  # ✅ 배치되지 않은 오브젝트를 올릴 높이
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
) -> None:
    """
    Isaac Lab 환경에서 유사도 기반 가중치 배치를 수행하고, 배치되지 않은 물체들을 ceiling_height로 이동하는 함수.

    Args:
        env: Isaac Lab 환경 객체
        env_ids: 현재 활성화된 환경 ID (Tensor)
        pose_array: (1, rows, cols, 7) 형태의 Pose 정보 (torch.Tensor)
        asset_dict: 배치할 오브젝트 리스트
        object_id_dict: 객체 이름 → ID 매핑
        object_id_dict_rev: ID → 객체 이름 매핑
        category_mapping: 카테고리별 객체 리스트
        ceiling_height: 배치되지 않은 물체가 이동할 z축 높이
        object_collection_cfg: Isaac Lab에서 사용할 객체 컬렉션 설정
        spawn_probability: Target Object가 1~2번째 행에 배치될 확률
        visibility_probability: Target Object 앞을 비울 확률
    """
    
    spawn_probability: float = 0.2,
    visibility_probability: float = 0.2,

    num_rows, num_cols = pose_array.shape[1:3]
    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]

    # 1️⃣ Target Object 랜덤 선택 및 배치 규칙 적용
    target_object = choice(list(asset_dict.keys()))
    target_category = target_object.split("_")[0]  # 카테고리 추출
    target_object_id = object_id_dict[target_object]

    if torch.rand(1).item() < spawn_probability:
        target_row_idx = torch.randint(0, 2, (1,)).item()
    else:
        target_row_idx = num_rows - 1

    target_col_idx = torch.randint(0, num_cols, (1,)).item()
    target_position = pose_array[0, target_row_idx, target_col_idx, :3]
    target_rotation = pose_array[0, target_row_idx, target_col_idx, 3:]

    placement_list = [(target_object, target_position, target_rotation)]
    used_positions = {(target_row_idx, target_col_idx)}

    empty_positions = set()
    if torch.rand(1).item() < visibility_probability:
        for row in range(target_row_idx - 1, -1, -1):
            empty_positions.add((row, target_col_idx))

    def place_items_with_weights(items, candidate_positions, position_weights):
        """아이템을 가중치 기반으로 배치"""
        while items and candidate_positions:
            weighted_pos = choices(candidate_positions, weights=position_weights, k=1)[0]
            if weighted_pos not in used_positions and weighted_pos not in empty_positions:
                row_idx, col_idx = weighted_pos
                item = items.pop(0)
                position = pose_array[0, row_idx, col_idx, :3]
                rotation = pose_array[0, row_idx, col_idx, 3:]
                placement_list.append((item, position, rotation))
                used_positions.add(weighted_pos)

    # 🏷 같은 카테고리(0.8) 배치
    same_category_items = category_mapping[target_category].copy()
    same_category_items.remove(target_object)
    shuffle(same_category_items)

    same_category_positions = []
    position_weights = []
    for row_idx in range(target_row_idx - 1, -1, -1):
        for col_offset in [-1, 0, 1]:
            col_idx = target_col_idx + col_offset
            if 0 <= col_idx < num_cols:
                same_category_positions.append((row_idx, col_idx))
                position_weights.append(5.0 if col_offset == 0 else 1.0)

    place_items_with_weights(same_category_items, same_category_positions, position_weights)

    # 🏷 유사한 카테고리(0.5) 배치
    similar_category = None
    if target_category in ["cup", "mug"]:
        similar_category = "mug" if target_category == "cup" else "cup"
    elif target_category in ["bottle", "can"]:
        similar_category = "can" if target_category == "bottle" else "bottle"

    if similar_category:
        similar_category_items = category_mapping[similar_category].copy()
        shuffle(similar_category_items)

        similar_category_positions = []
        position_weights = []
        for col_offset in [-1, 1]:
            col_idx = target_col_idx + col_offset
            if 0 <= col_idx < num_cols:
                similar_category_positions.append((target_row_idx, col_idx))
                position_weights.append(5.0)
                adj_col_idx = col_idx + (1 if col_offset == -1 else -1)
                if 0 <= adj_col_idx < num_cols:
                    similar_category_positions.append((target_row_idx, adj_col_idx))
                    position_weights.append(1.0)

        place_items_with_weights(similar_category_items, similar_category_positions, position_weights)

    # 🏷 배치되지 않은 오브젝트 처리
    placed_objects = {obj[0] for obj in placement_list}  # 이미 배치된 오브젝트 목록
    remaining_objects = set(asset_dict.keys()) - placed_objects  # 배치되지 않은 오브젝트들

    for obj_name in remaining_objects:
        pose_instance = pose_array[0, 0, 0].clone()  # 기본 위치 사용
        position = pose_instance[:3]
        position[2] = ceiling_height  # ✅ z축을 ceiling_height로 변경

        rotation = pose_instance[3:]
        placement_list.append((obj_name, position, rotation))

    # 2️⃣ Isaac Lab의 `RigidObjectCollection`을 활용하여 오브젝트 배치
    for object_name, position, rotation in placement_list:
        position = torch.tensor(position, device=env.device)
        rotation = torch.tensor(rotation, device=env.device)

        noise = torch.rand(3, device=env.device) * 0.02 - 0.01
        noise[2] = 0.0  # ✅ z축 노이즈 제거
        position += noise

        angle = torch.rand(1, device=env.device) * 360.0
        sin_angle = torch.sin(torch.deg2rad(angle / 2))
        cos_angle = torch.cos(torch.deg2rad(angle / 2))
        quaternion = torch.tensor([cos_angle, 0.0, 0.0, sin_angle], device=env.device)

        object_ids = object_collection.find_objects(name_keys=object_name)
        object_collection.write_object_link_pose_to_sim(
            torch.cat((position, quaternion)).unsqueeze(0),
            env_ids=env_ids,
            object_ids=object_ids[0]
        )

    print(f"[DEBUG] {len(placement_list)} objects placed, {len(remaining_objects)} moved to ceiling height.")