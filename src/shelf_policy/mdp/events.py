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
                 pose_array: np.ndarray = MISSING,
                 object_id_dict: dict = MISSING,
                 object_id_dict_rev: dict = MISSING,
                 object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                ):
        

        rows, cols = pose_array.shape[1], pose_array.shape[2]
        object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
        
        env.target_id = object_id_dict[choice(list(asset_dict.keys()))]
        target_obj = object_id_dict_rev[str(env.target_id)]
        
        for cur_env in env_ids.tolist():
            object_id_list: list = list(asset_dict.keys())
            shuffle(object_id_list)
            target_cell = 0
        
            for i, asset_cfg in enumerate(object_id_list):
                # Write pose to simulation
                if asset_cfg == target_obj:
                    target_cell = i

                pose_instance = pose_array[0, i // cols, i % cols]
                pose_tensor = torch.tensor([pose_instance], device=env.device)

                positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
                orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
                object_id = object_collection.find_objects(name_keys=asset_cfg)

                object_collection.write_object_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device), object_ids=object_id[0])
                # asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device))

            adjacent_indices = self.sweeping_right_mode(target_cell, rows, cols)
            # print(f"env: {cur_env}")
            # print(f"target_obj: {target_obj}")
            # print(f"target_cell: {target_cell}")
            # print(f"adjacent_indices: {adjacent_indices}")
            # print(f"pose_array: {pose_array}")

            for value in adjacent_indices:
                id = value[0] * cols + value[1]
                obj = object_id_list[id]
                pose_instance = pose_array[0, value[0], value[1]]
                pose_tensor = torch.tensor([pose_instance], device=env.device)
                positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
                positions[0, 2] = 1.3
                orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
                object_id = object_collection.find_objects(name_keys=obj)
                object_collection.write_object_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device), object_ids=object_id[0])
        
                
        
    def sweeping_right_mode(self, target_id: int, rows: int, cols: int) -> np.ndarray:
        """
        Identify objects in front, right, and diagonal positions of the target.
        If the target is in the last row, include all objects in the front rows.
        Returns results as a NumPy array.
        """
        target_row = target_id // cols
        target_col = target_id % cols

        # (1) 앞쪽 찾기 (모든 앞쪽 행 포함)
        front_rows = np.arange(target_row - 1, -1, -1) if target_row > 0 else np.array([])
        front_indices = np.column_stack((front_rows, np.full_like(front_rows, target_col))) if front_rows.size else np.empty((0, 2), dtype=int)

        # (2) 오른쪽 찾기
        right_index = np.array([[target_row, target_col + 1]]) if target_col < cols - 1 else np.empty((0, 2), dtype=int)

        # (3) 우측 대각선 찾기
        diagonal_indices = np.column_stack((front_rows, np.full_like(front_rows, target_col + 1))) if (target_row > 0 and target_col < cols - 1) else np.empty((0, 2), dtype=int)

        # (4) 모든 결과를 NumPy 배열로 결합
        adjacent_array = np.vstack((front_indices, right_index, diagonal_indices))

        return adjacent_array

        

        


        







