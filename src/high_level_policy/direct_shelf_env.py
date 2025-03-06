# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from llvmlite.binding.value import Visibility
import torch
from collections.abc import Sequence
import random
import os
import numpy as np

from omni.isaac.lab_assets.cart_double_pendulum import CART_DOUBLE_PENDULUM_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
    Articulation,
    RigidObject,
    RigidObjectCollection,
)
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg

from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from src_utils.shelf_utils import load_yaml_config, load_and_reshape_pose
from random import shuffle, choice


@configclass
class DirectShelfEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 150
    episode_length_s = 5.0
    action_space = [{3}, {12}]
    observation_space = 1
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    shelf: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack3.usd",
            mass_props=MassPropertiesCfg(mass=100),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        debug_vis=False,
    )

    # YAML 파일 로드
    object_cfgs = load_yaml_config(
        yaml_path="src/shelf_policy/params/environment_highlevel.yaml"
    )

    rigid_obj_dict = {}
    # 객체 정보 및 Pose 정보 가져오기
    object_path_dict = object_cfgs["objects"]
    object_pose_dict = object_cfgs["pose"]
    object_id_dict = object_cfgs["id"]
    object_id_dict_rev = {str(v): k for k, v in object_id_dict.items()}
    # 크기(키 개수) 비교 후 에러 발생
    if len(object_path_dict) != len(object_pose_dict):
        raise ValueError(
            f"Error: Object count mismatch! "
            f"objects({len(object_path_dict)}) != pose({len(object_pose_dict)})"
        )

    for key, value in object_path_dict.items():
        rigid_obj: RigidObjectCfg = RigidObjectCfg(
            prim_path=os.path.join("/World/envs/env_.*/", f"{key}"),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=object_pose_dict[key][:3], rot=object_pose_dict[key][3:]
            ),
            spawn=UsdFileCfg(
                usd_path=value,
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=MassPropertiesCfg(mass=0.3),
            ),
        )

        rigid_obj_dict[key] = rigid_obj

    object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=rigid_obj_dict
    )

    object_path_dict = object_cfgs["objects"]
    object_pose_dict = object_cfgs["pose"]
    object_id_dict = object_cfgs["id"]
    object_id_dict_rev = {str(v): k for k, v in object_id_dict.items()}
    object_category = object_cfgs["category"]

    pose_array = load_and_reshape_pose(object_pose_dict)
    asset_dict: dict = rigid_obj_dict

    target_row_index = 3
    spawn_probability = 0.15
    visibility_probability = 0.2


class DirectShelfEnv(DirectRLEnv):
    cfg: DirectShelfEnvCfg

    def __init__(
        self, cfg: DirectShelfEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_id = torch.zeros(self.num_envs, 1, device=self.device)
        self.shelf_object_config = torch.full((self.num_envs, 3, 4), -1, device=self.device) # 각 환경별로 shelf의 object 위치(object id가 0부터 시작하므로 -1로 초기화)
        self.shelf_front_object = torch.full((self.num_envs, 4), -1, device=self.device) # 각 환경별로 shelf의 앞쪽 object id

        self.action_commands = torch.tensor(
            [
                [0, 0, 1.05],  # Action 0
                [0, 0.21, 0],  # Action 1
                [0, -0.21, 0],  # Action 2
            ],
            device=self.device,
        )

    def _setup_scene(self):

        self._shelf = RigidObject(self.cfg.shelf)
        self._object_collection = RigidObjectCollection(self.cfg.object_collection)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.to(torch.int)
        

        random_ids = torch.randint(0, 12, (self.num_envs,), device='cuda:0')
        self.actions[:, 1] = random_ids
        policy = self.actions[:, 0]
        items = self.actions[:, 1]
        processed_position = self.action_commands[policy]
        # 올바르게 인덱스 지정 (차원 문제 해결)
        self._processed_items = items.long().unsqueeze(-1)  # (2, 1) 형태

        # 인덱싱 시 중복 방지
        self._obj_state_w = self._object_collection.data.object_state_w[
            torch.arange(self._object_collection.data.object_state_w.shape[0]),
            self._processed_items.squeeze(-1),
        ].clone()

        self._obj_state_w[:, :3] = self._obj_state_w[:, :3] + processed_position[:, :3]

        # if self.actions[0, 0] == 0:
        #     self.shelf_object_config[:, 2, update_idx] = random_object_id

        


        self._object_collection.write_object_state_to_sim(
            object_state=self._obj_state_w.unsqueeze(1),
            object_ids=self._processed_items,
        )

    def _apply_action(self) -> None:
        pass
        

    def _get_observations(self) -> dict:
        obs = torch.zeros(self.num_envs, device=self.device)

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return torch.zeros(self.num_envs), time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)
        rows, cols = len(self.cfg.pose_array[0]), len(self.cfg.pose_array[0][0])

        # 사용자 입력 기준의 target_row_index를 배열 인덱스로 변환
        if np.random.rand() < self.cfg.spawn_probability:
            adjusted_target_row_index = np.random.choice(
                [0, 1]
            )  # 첫 번째(0) 또는 두 번째(1) 행 선택
        else:
            adjusted_target_row_index = (
                self.cfg.target_row_index - 1
            )  # 사람이 1~5로 입력한 값을 0~4로 변환

        random_row = adjusted_target_row_index  # 0부터 rows-1까지 랜덤
        random_col = torch.randint(0, cols, (1,)).item()  # 0부터 cols-1까지 랜덤

        target_object_id = self.cfg.object_id_dict[
            choice(list(self.cfg.asset_dict.keys()))
        ]

        self.target_id[env_ids, 0] = target_object_id

        target_object_name = self.cfg.object_id_dict_rev[str(target_object_id)]

        target_category = self.get_category(target_object_name)
        same_category_items = self.cfg.object_category[target_category].copy()
        random.shuffle(same_category_items)

        similar_category = None
        if target_category in ["cup", "mug"]:
            similar_category = "mug" if target_category == "cup" else "cup"
        elif target_category in ["bottle", "can"]:
            similar_category = "can" if target_category == "bottle" else "bottle"

        similar_category_items = self.cfg.object_category[similar_category].copy()
        random.shuffle(similar_category_items)

        other_categories = set(self.cfg.object_category.keys()) - {
            target_category,
            similar_category,
        }
        other_category_items = []
        for cat in other_categories:
            other_category_items.extend(self.cfg.object_category[cat])
        random.shuffle(other_category_items)

        # 위치별로 배치할 오브젝트 리스트 생성
        placement_list = []
        used_items = {}

        empty_positions = set()
        if np.random.rand() < self.cfg.visibility_probability:
            for row_idx in range(random_row - 1, -1, -1):  # 타겟 객체보다 앞쪽 행(row)
                empty_positions.add((row_idx, random_col))

        # 이미 사용된 위치를 추적하기 위한 집합 # 타겟 위치 추가
        placement_list.append(((random_row, random_col), target_object_name))
        used_positions = {(random_row, random_col)}

        used_positions.update(empty_positions)

        def place_items_with_weights(items, candidate_positions, position_weights):
            """아이템을 가중치 기반으로 배치하고 중복 발생 시 다른 유효한 자리를 재탐색."""

            while items and candidate_positions:
                # 가중치 기반으로 위치 선택
                weighted_pos = random.choices(
                    population=candidate_positions, weights=position_weights, k=1
                )[0]

                if weighted_pos not in used_positions:
                    # 중복되지 않은 경우 배치
                    item = items.pop(0)
                    if item != target_object_name:
                        placement_list.append((weighted_pos, item))
                        used_positions.add(weighted_pos)

                        # 선택된 위치를 후보와 가중치에서 제거
                        idx = candidate_positions.index(weighted_pos)
                        candidate_positions.pop(idx)
                        position_weights.pop(idx)
                else:
                    # 중복된 경우 후보와 가중치에서 해당 위치만 제거
                    idx = candidate_positions.index(weighted_pos)
                    candidate_positions.pop(idx)
                    position_weights.pop(idx)
            if items:
                for item in items:
                    if item != target_object_name:
                        used_items[item] = 1

        # 같은 카테고리 (0.8) 배치
        same_category_positions = []
        for row_idx in range(
            random_row - 1, -1, -1
        ):  # 타겟보다 뒤쪽(행 번호가 작은 방향)
            for col_offset in [-1, 0, 1]:  # 타겟 열 주변의 좌(-1), 정면(0), 우(1)
                col_idx = random_col + col_offset  # 열 계산
                if 0 <= col_idx < cols:  # 유효한 열인지 확인
                    same_category_positions.append((row_idx, col_idx))  # 위치 저장

        # 중심 열에 더 높은 가중치를 부여
        position_weights = [
            5.0 if pos[1] == random_col else 1.0 for pos in same_category_positions
        ]
        place_items_with_weights(
            same_category_items, same_category_positions, position_weights
        )

        # 유사한 카테고리 (0.5) 배치
        similar_category_positions = []
        similar_cols = [random_col - 1, random_col + 1]
        position_weights = []

        for col_idx in similar_cols:
            if 0 <= col_idx < cols:
                for row_idx in range(rows):
                    similar_category_positions.append((row_idx, col_idx))
                    position_weights.append(5.0)

                    # 좌, 우로 확장
                    adj_col_idx = col_idx + (1 if col_idx == random_col - 1 else -1)
                    if 0 <= adj_col_idx < cols:
                        similar_category_positions.append((row_idx, adj_col_idx))
                        position_weights.append(1.0)

        place_items_with_weights(
            similar_category_items, similar_category_positions, position_weights
        )

        # 카테고리 0.8과 0.5에서 사용된 열 추적
        used_columns = {random_col}  # 타겟 열 포함
        used_columns.update(
            [pos[1] for pos in same_category_positions]
        )  # 0.8에서 사용된 열 추가
        used_columns.update(
            [pos[1] for pos in similar_category_positions]
        )  # 0.5에서 사용된 열 추가

        # 다른 카테고리 (0.1) 배치
        other_category_positions = []
        available_columns = [
            col_idx for col_idx in range(cols) if col_idx not in used_columns
        ]

        for col_idx in available_columns:  # 사용되지 않은 열에서만 선택
            for row_idx in range(rows):
                other_category_positions.append((row_idx, col_idx))

        position_weights = [1.0] * len(other_category_positions)  # 균등 가중치
        place_items_with_weights(
            other_category_items, other_category_positions, position_weights
        )

        # asset_keys_list: list = list(self.cfg.asset_dict.keys())
        rest_objects = list(used_items.keys())

        pose_array_tensor = torch.tensor(self.cfg.pose_array, device=self.device)

        # Orientation randomization 미적용
        orientations = torch.empty((env_ids.shape[0], 4), device=self.device)
        orientations[:, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        velocities = torch.zeros((env_ids.shape[0], 6), device=self.device)

        for (row_idx, col_idx), object_name in placement_list:
            index = self.cfg.object_id_dict[object_name]
            pose_instance = pose_array_tensor[0, row_idx, col_idx]
            positions = pose_instance[:3] + self.scene.env_origins[env_ids, 0:3]
            object_ids = self._object_collection.find_objects(name_keys=object_name)
            self._object_collection.write_object_link_state_to_sim(
                torch.cat((positions, orientations, velocities), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0],
            )

        for index, object_name in enumerate(rest_objects):
            pose_instance = pose_array_tensor[0, index // cols, index % cols]
            positions = pose_instance[:3] + self.scene.env_origins[env_ids, 0:3]

            positions[:, 2] = 1.8
            object_ids = self._object_collection.find_objects(name_keys=object_name)
            self._object_collection.write_object_link_state_to_sim(
                torch.cat((positions, orientations, velocities), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0],
            )

        # 1. placement_list에서 좌표와 object name 추출
        # 좌표 리스트와 object name 리스트로 분리함
        coords_list = [list(pos) for pos, _ in placement_list]  # shape: (N, 2)
        obj_names_list = [obj_name for _, obj_name in placement_list]  # 길이: N
        
        # 2. 좌표 텐서 생성
        coords = torch.tensor(coords_list, device=self.device)  # (N, 2)
        num_rows = self.shelf_object_config.shape[1] # shelf_object_config의 행 개수
        coords[:, 0] = (num_rows - 1) - coords[:, 0] # placement_list의 좌표는 왼쪽 아래가 0,0이므로, shelf_object_config (왼쪽 위가 0,0)에 맞추기 위해 행 인덱스 반전
        obj_ids_list = [self.cfg.object_id_dict.get(obj_name, 0) for obj_name in obj_names_list] # object name을 object id로 변환
        object_ids_tensor = torch.tensor(obj_ids_list, device=self.device) # object id 리스트를 텐서로 변환
        expanded_coords = coords.unsqueeze(0).expand(env_ids.size(0), -1, -1)
        expanded_object_ids = object_ids_tensor.unsqueeze(0).expand(env_ids.size(0), -1)
        
        # 값 채워넣기
        self.shelf_object_config[env_ids] = -1 # 초기화 하는 환경만 -1로 초기화
        self.shelf_object_config[env_ids.unsqueeze(1),
                           expanded_coords[:, :, 0],
                           expanded_coords[:, :, 1]] = expanded_object_ids
        



    def get_category(self, item_name):
        for category, items in self.cfg.object_category.items():
            if item_name in items:
                return category
        return None

        # for index, asset_name in enumerate(asset_keys_list):
        #     if asset_name == target_object_name:
        #         target_index = index

        #     pose_instance = pose_array_tensor[0, index // cols, index % cols]
        #     positions = pose_instance[:3] + self.scene.env_origins[env_ids, 0:3]
        #     object_ids = self._object_collection.find_objects(name_keys=asset_name)
        #     self._object_collection.write_object_link_state_to_sim(
        #         torch.cat((positions, orientations, velocities), dim=1).unsqueeze(1),
        #         env_ids=env_ids,
        #         object_ids=object_ids[0],
        #     )