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
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import FrameTransformerData

from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class Object_randomization(ManagerTermBase):

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)

        self.asset_pose = np.zeros(1)

    def __call__(self,
                 env: ManagerBasedRLEnv,
                 env_ids: torch.Tensor,
                 asset_dict: dict = MISSING,
                 pose_array: torch.Tensor = MISSING,
                 object_id_dict: dict = MISSING,
                 object_id_dict_rev: dict = MISSING,
                 object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                ):
        
        rows, cols = pose_array.shape[1], pose_array.shape[2]
        object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
        
        env.target_id = object_id_dict[choice(list(asset_dict.keys()))]
        target_obj = object_id_dict_rev[str(env.target_id)]

        object_id_list: list = list(asset_dict.keys())

        # 우선 orientation randomization은 안줌
        orientations = torch.empty((env_ids.shape[0], 4) , device=env.device)
        orientations[:, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

        for i, asset_cfg in enumerate(object_id_list):

            if asset_cfg == target_obj:
                    target_cell = i
            pose_instance = pose_array[0, i // cols, i % cols]

            positions = pose_instance[:3] + env.scene.env_origins[env_ids, 0:3]
            

            object_id = object_collection.find_objects(name_keys=asset_cfg)

            object_collection.write_object_link_pose_to_sim(torch.cat((positions, orientations), dim=1).unsqueeze(1), env_ids=env_ids, object_ids=object_id[0])
        
        adjacent_indices = self.sweeping_right_mode(target_cell, rows, cols, env.device)

        for value in adjacent_indices:
            obj_id = value[0] * cols + value[1]
            obj = object_id_list[obj_id]
            pose_instance = pose_array[0, value[0], value[1]]
            positions = pose_instance[:3] + env.scene.env_origins[env_ids, 0:3]
            positions[:, 2] = 1.3  # 높이 변경
            object_id = object_collection.find_objects(name_keys=obj)
            object_collection.write_object_link_pose_to_sim(torch.cat((positions, orientations), dim=1).unsqueeze(1), env_ids=env_ids, object_ids=object_id[0])


    def sweeping_right_mode(self, target_id: int, rows: int, cols: int, device):
        """
        Identify objects in front, right, and diagonal positions of the target.
        If the target is in the last row, include all objects in the front rows.
        Returns results as a GPU Tensor.
        """
        target_row = target_id // cols
        target_col = target_id % cols

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
        

        

        


        







