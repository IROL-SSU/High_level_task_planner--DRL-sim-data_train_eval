# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import random
import os

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
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
import numpy as np


from torchvision.models.segmentation import fcn_resnet50
from torch import nn
from PIL import Image


@configclass
class HighlevelDirectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 80 # 기존 2
    episode_length_s = 7.0 # 기존 5.0
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
            usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack2.usd",
            mass_props=MassPropertiesCfg(mass=100),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        debug_vis=False,
    )

    # YAML 파일 로드
    object_cfgs = load_yaml_config(
        yaml_path="src/shelf_policy/params/environment.yaml"
    )
    
    # 카메라
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.48, 0.0, 1.27), rot=(0.0, 0.0169289, 0.0, 0.9998567), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        width=640,
        height=480,
    )
    write_image_to_file = True
    
    mean=[0.650085384273529, 0.6429061861377292, 0.6164081222491794]
    std=[0.1687737046184826, 0.1750711261287332, 0.19902168659025637]
    
    MODEL_PATH = "/home/haneul/IsaacLab_IROL/src/High_level_policy_direct/FCN_model/best_model.pth"

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


class HighlevelDirectEnv(DirectRLEnv):
    cfg: HighlevelDirectEnvCfg

    def __init__(
        self, cfg: HighlevelDirectEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_id = torch.zeros(self.num_envs, 1, device=self.device)
        self.previous_distribution = torch.zeros(self.num_envs, 640, device=self.device)
        self.column_distribution = torch.zeros(self.num_envs, 4, device=self.device)
        
        class FCNModel(nn.Module):
            def __init__(self):
                super(FCNModel, self).__init__()
                # weights=None로 설정하여 pretrained 모델 사용하지 않음
                self.model = fcn_resnet50(weights=None)
                # 분류기 마지막 레이어를 원하는 클래스 수로 변경 (여기서는 예시로 12 클래스 사용)
                self.model.classifier[4] = nn.Conv2d(512, 12, kernel_size=1)

            def forward(self, x):
                return self.model(x)['out']
        
        self.fcn_model = FCNModel()
        # 모델 파일 경로 (필요에 따라 cfg에 추가하거나 상수로 관리 가능)
        state_dict = torch.load(self.cfg.MODEL_PATH, map_location=self.device)
        # 불필요한 aux_classifier 관련 파라미터는 필터링
        filtered_state_dict = {k: v for k, v in state_dict.items() if "aux_classifier" not in k}
        self.fcn_model.load_state_dict(filtered_state_dict, strict=False)
        self.fcn_model = self.fcn_model.to(self.device)

    def _setup_scene(self):

        self._shelf = RigidObject(self.cfg.shelf)
        self._object_collection = RigidObjectCollection(self.cfg.object_collection)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg, translation=(-5, 0, 10))
        light_cfg2 = sim_utils.DomeLightCfg(intensity=2700.0, color=(1.0, 1.0, 1.0))
        light_cfg2.func("/World/Light2", light_cfg2, translation=(1.2, 0, 1.4))
        
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.zeros(self.num_envs, 3, device=self.device)

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        camera_data = self._tiled_camera.data.output["rgb"]  # (N, H, W, C)
        camera_data = camera_data / 255.0
        camera_data = camera_data.permute(0, 3, 1, 2)
        camera_data = Normalize(mean=self.cfg.mean, std=self.cfg.std)(camera_data)
        camera_data = camera_data.to(self.device)  # GPU로 이동
        target_id = self.target_id.squeeze(-1).long()  # (num_envs, 1) → (num_envs,)
        self.fcn_model.eval()
        with torch.no_grad():
            fcn_output = self.fcn_model(camera_data)  # (N, num_classes, H, W)
        pred = fcn_output[torch.arange(self.num_envs, device=self.device), target_id, :, :]
        
        # pred: (num_envs, H, 640)
        B, H, W = pred.shape
        # 각 환경별 전체 최대값 계산 → (B, 1, 1)
        max_vals = pred.view(B, -1).max(dim=1)[0].view(B, 1, 1)
        # 각 환경마다 최대값이 0보다 큰 경우에만 정규화 수행
        normalized_preds = torch.where(max_vals > 0, pred / max_vals, pred)
        gain = 2.0  # 비선형 가중치 적용을 위한 gain 값
        weighted_preds = normalized_preds * torch.exp(-gain * (1 - normalized_preds))
        # y축(행 방향, 즉 H축)으로 합산 → 각 환경별 분포: (B, W)
        current_distributions = weighted_preds.sum(dim=1)  # shape: (num_envs, 640)
        
        # 각 환경별 분포를 개별적으로 업데이트 (이전 분포는 (num_envs, 640))
        gamma = 0.7  # 현재 분포 반영 가중치
        final_distribution = gamma * current_distributions + (1 - gamma) * self.previous_distribution
        # 업데이트 후, 최종 분포를 그대로 (num_envs, 640) 형태로 유지
        self.previous_distribution = final_distribution.detach()
        
        col1 = final_distribution[:, 0:185].max(dim=1)[0]   # 첫 번째 구간의 최대값 (shape: (num_envs,))
        col2 = final_distribution[:, 185:320].max(dim=1)[0]   # 두 번째 구간
        col3 = final_distribution[:, 320:455].max(dim=1)[0]   # 세 번째 구간
        col4 = final_distribution[:, 455:640].max(dim=1)[0]   # 네 번째 구간
        
        self.column_distribution = torch.stack([col1, col2, col3, col4], dim=1)
        
        # print(f"Final distribution shape: {final_distribution.shape}")
        # print(f"Column distribution shape: {self.column_distribution.shape}")
        print(f"Colum distribution: {self.column_distribution[0]}")
        print("-----------------------------")
        
        if self.cfg.write_image_to_file:
            pred_to_save = pred.unsqueeze(-1)
            save_images_to_file(pred_to_save, f"fcn_output.png")

                
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
        
        self.previous_distribution[env_ids, :].zero_()
        self.column_distribution[env_ids, :].zero_()
        
        
        rows, cols = len(self.cfg.pose_array[0]), len(self.cfg.pose_array[0][0])
        random_row = torch.randint(0, rows, (1,)).item()  # 0부터 rows-1까지 랜덤
        random_col = torch.randint(0, cols, (1,)).item()  # 0부터 cols-1까지 랜덤

        target_object_id = self.cfg.object_id_dict[
            choice(list(self.cfg.asset_dict.keys()))
        ]

        self.target_id[env_ids, 0] = target_object_id

        target_object_name = self.cfg.object_id_dict_rev[str(target_object_id)]
        print(f"Target object: {target_object_name}")
        target_category = self.get_category(target_object_name)
        same_category_items = self.cfg.object_category[target_category].copy()

        similar_category = None
        if target_category in ["cup", "mug"]:
            similar_category = "mug" if target_category == "cup" else "cup"
        elif target_category in ["bottle", "can"]:
            similar_category = "can" if target_category == "bottle" else "bottle"

        similar_category_items = self.cfg.object_category[similar_category].copy()

        other_categories = set(self.cfg.object_category.keys()) - {
            target_category,
            similar_category,
        }
        other_category_items = []
        for cat in other_categories:
            other_category_items.extend(self.cfg.object_category[cat])

        # 위치별로 배치할 오브젝트 리스트 생성
        placement_list = []
        used_items = {}

        # 이미 사용된 위치를 추적하기 위한 집합 # 타겟 위치 추가
        placement_list.append(((random_row, random_col), target_object_name))
        used_positions = {(random_row, random_col)}

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
                if 0 <= col_idx < rows:  # 유효한 열인지 확인
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
            if 0 <= col_idx < rows:
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

    def get_category(self, item_name):
        for category, items in self.cfg.object_category.items():
            if item_name in items:
                return category
        return None