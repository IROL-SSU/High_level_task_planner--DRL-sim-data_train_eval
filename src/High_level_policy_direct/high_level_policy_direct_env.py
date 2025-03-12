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
from omni.isaac.lab.utils.math import combine_frame_transforms, transform_points, euler_xyz_from_quat
from src_utils.shelf_utils import normalize_angle

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
    decimation = 10 # 기존 2 / 60(확인용) / 10(학습용)
    episode_length_s = 4.0 # 기존 5.0
    action_space = [{3}, {4}]
    observation_space = 13
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
    write_image_to_file = False
    
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
    
    target_row_index = 3
    spawn_probability = 0.1
    visibility_probability = 0.1
    
    # reward scales
    traget_grasping = 100.0
    hp_sweeping_right = 30.0
    hp_sweeping_left = 30.0
    hp_grasping = 10.0
    
    # penalty scales
    lp_grasping = -5.0
    lp_sweeping_right = -5.0
    lp_sweeping_left = -5.0
    traget_sweeping = -20.0
    empty_action = -10.0
    grasping_w_n_sweeping = -10.0
    sweeping_again = -10.0
    
    sweeping_no_enough = -10.0


class HighlevelDirectEnv(DirectRLEnv):
    cfg: HighlevelDirectEnvCfg

    def __init__(
        self, cfg: HighlevelDirectEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_id = torch.zeros(self.num_envs, 1, device=self.device)
        self.previous_distribution = torch.zeros(self.num_envs, 640, device=self.device) # FCN 출력에서 가져온 이전 step에서의 640 column 분포
        self.column_distribution = torch.zeros(self.num_envs, 4, device=self.device) # 4개의 column에 대한 distribution 최대 값들
        
        self.shelf_object_config = torch.full((self.num_envs, 3, 4), -1, device=self.device) # 각 환경별로 shelf의 object 위치(object id가 0부터 시작하므로 -1로 초기화)
        self.shelf_front_object = torch.full((self.num_envs, 4), -1, device=self.device) # 각 환경별로 shelf의 앞쪽 object id
        self.shelf_front_object_distance = torch.zeros(self.num_envs, 4, device=self.device) # 각 환경별로 shelf의 앞쪽 object까지의 거리
        
        self.previous_shelf_object_config = torch.full((self.num_envs, 3, 4), -1, device=self.device) # 이전 step에서의 shelf의 object 위치(object id가 0부터 시작하므로 -1로 초기화)
        self.previous_shelf_front_object = torch.full((self.num_envs, 4), -1, device=self.device) # 이전 step에서의 shelf의 앞쪽 object id
        self.previous_shelf_front_object_distance = torch.zeros(self.num_envs, 4, device=self.device) # 이전 step에서의 shelf의 앞쪽 object까지의 거리
        self.previous_column_distribution = torch.zeros(self.num_envs, 4, device=self.device) # 이전 step에서의 4개의 column에 대한 distribution 최대 값들
        
        self.action_policy = torch.full((self.num_envs, 1), 0, device=self.device) # 현재 step에서의 action policy
        self.action_column = torch.full((self.num_envs, 1), 0, device=self.device) # 현재 step에서 선택된 action column
        
        self.previous_action_policy = torch.full((self.num_envs, 1), 0, device=self.device) # 이전 step에서의 action policy (초기값은 0)
        self.previous_action_column = torch.full((self.num_envs, 1), 0, device=self.device) # 이전 step에서의 action column (초기값은 0)
        
        self.target_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) # 각 환경별로 target grasped 여부

        
        self.action_commands = torch.tensor(
            [
                [0, 0, 1.05],  # Action 0
                [0, 0.21, 0],  # Action 1
                [0, -0.21, 0],  # Action 2
            ],
            device=self.device,
        )
        
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
        self.actions = actions.to(torch.int)
        
        # random_policy = torch.full((self.num_envs, ), 0, device=self.device)
        # random_policy = torch.randint(0, 3, (self.num_envs,), device='cuda:0')
        # self.actions[:, 0] = random_policy
        # random_ids = torch.randint(0, 4, (self.num_envs,), device='cuda:0')
        # self.actions[:, 1] = random_ids
        
        policy = self.actions[:, 0]
        # print(f"Policy: {policy}")
        item_idx = self.actions[:, 1]
        # print(f"Item idx: {item_idx}")
        
        policy = torch.clamp(policy, min=0, max=2)
        # print(f"Policy: {policy}")
        item_idx = torch.clamp(item_idx, min=0, max=3)
        # print(f"Item idx: {item_idx}")
        
        self.action_policy = policy.unsqueeze(-1).clone()
        # print(f"Previous action policy: {self.action_policy}")
        self.action_column = item_idx.unsqueeze(-1).clone()
        # print(f"Previous action column: {self.action_column}")
        processed_position = self.action_commands[policy]
        
        env_indices = torch.arange(self.num_envs, device=self.device)  # 각 환경에서 shelf_front_object (shape: (num_envs, 4))의 item_idx에 해당하는 object id를 선택
        selected_object_ids = self.shelf_front_object[env_indices, item_idx.long()] # selected_object_ids: (num_envs,) – 각 env에서 선택된 column의 object id
        # print(f"shelf_front_object: {self.shelf_front_object}")
        # print(f"Selected object ids: {selected_object_ids}")
        valid_mask = selected_object_ids != -1 # 유효한 object id가 있는지 체크 (유효하면 -1이 아님)
        
        # 전체 환경에 대한 object state (shape: [num_envs, num_objects, state_dim])
        all_obj_state = self._object_collection.data.object_state_w.clone()

        # 전체 환경에 대해 업데이트할 state를 담을 텐서를 생성 (shape: [num_envs, state_dim])
        # 여기서는 state_dim이 7이라고 가정합니다.
        update_state = torch.zeros(self.num_envs, all_obj_state.size(-1), device=self.device)

        # valid한 환경과 invalid한 환경의 인덱스를 구합니다.
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        invalid_indices = torch.nonzero(~valid_mask, as_tuple=False).squeeze(-1)

        # valid 환경: 선택된 object id에 해당하는 state를 가져와 이동 벡터(processed_position)를 더합니다.
        if valid_indices.numel() > 0:
            valid_obj_ids = selected_object_ids[valid_indices].long()  # shape: [n_valid]
            # all_obj_state[env, obj, :] 선택 -> shape: [n_valid, state_dim]
            update_state[valid_indices] = all_obj_state[valid_indices, valid_obj_ids, :].clone()
            # x,y,z (앞 3 요소)에 processed_position을 더함
            update_state[valid_indices, :3] += processed_position[valid_indices, :3]
        
        # invalid 환경: object id가 -1인 경우에는 dummy 업데이트 (여기서는 0번 object state를 사용)
        if invalid_indices.numel() > 0:
            update_state[invalid_indices] = all_obj_state[invalid_indices, 0, :].clone()
            # 변화가 없도록 processed_position은 더하지 않습니다.

        # 전체 환경에 대해 full_object_ids: 유효하지 않은 환경은 dummy로 0번 object id 사용
        full_object_ids = selected_object_ids.clone().unsqueeze(-1)  # shape: [num_envs, 1]
        if invalid_indices.numel() > 0:
            full_object_ids[invalid_indices] = 0  # dummy object id

        # 최종적으로, 업데이트된 update_state를 [num_envs, 1, state_dim]으로 만들어 업데이트합니다.
        self._object_collection.write_object_state_to_sim(
            object_state=update_state.unsqueeze(1),  # [num_envs, 1, state_dim]
            object_ids=full_object_ids                # [num_envs, 1]
        )
            
        ## Shelf 업데이트: Grasping/Sweeping action에 따라 shelf_object_config 변경
        num_envs, num_rows, num_cols = self.shelf_object_config.shape
        rows_tensor = torch.arange(num_rows, device=self.device).view(1, num_rows, 1) # (1, num_rows, 1)
        mask = (self.shelf_object_config != -1) # (num_envs, num_rows, num_cols)
        candidate = torch.where(mask, rows_tensor.expand_as(self.shelf_object_config), torch.full_like(self.shelf_object_config, -1)) # 각 환경, 각 열에서 front row (최대 row 인덱스)
        max_row_indices = candidate.max(dim=1)[0] # (num_envs, num_cols)
        
        front_rows = max_row_indices[env_indices, item_idx.long()] # 각 환경마다 선택한 column의 front row 인덱스 가져옴
        valid_mask2 = (front_rows != -1) # 선택한 column에 유효한 object가 있는지 확인
        
        # 0: grasping → 해당 위치의 object를 제거 (즉, -1 할당)
        grasp_mask = (policy == 0) & valid_mask2
        grasp_envs = env_indices[grasp_mask]
        grasp_rows = front_rows[grasp_mask]
        grasp_cols = item_idx[grasp_mask]
        self.shelf_object_config[grasp_envs, grasp_rows, grasp_cols] = -1
        
        # 1: sweeping right → object를 오른쪽 열로 이동
        # (a) 정상적인 경우: item_idx < (num_cols - 1)
        sweep_right_mask = (policy == 1) & valid_mask2 & (item_idx < (num_cols - 1))
        sweep_right_envs = env_indices[sweep_right_mask]
        sweep_right_rows = front_rows[sweep_right_mask]
        sweep_right_cols = item_idx[sweep_right_mask]
        sweep_right_new_cols = sweep_right_cols + 1
        obj_ids_to_move = self.shelf_object_config[sweep_right_envs, sweep_right_rows, sweep_right_cols]
        self.shelf_object_config[sweep_right_envs, sweep_right_rows, sweep_right_new_cols] = obj_ids_to_move
        self.shelf_object_config[sweep_right_envs, sweep_right_rows, sweep_right_cols] = -1
        
        # (b) 잘못된 경우: item_idx == (num_cols - 1) → 이동할 column이 없으므로, 해당 객체를 제거
        sweep_right_invalid_mask = (policy == 1) & valid_mask2 & (item_idx == (num_cols - 1))
        sweep_right_invalid_envs = env_indices[sweep_right_invalid_mask]
        sweep_right_invalid_rows = front_rows[sweep_right_invalid_mask]
        sweep_right_invalid_cols = item_idx[sweep_right_invalid_mask]
        self.shelf_object_config[sweep_right_invalid_envs, sweep_right_invalid_rows, sweep_right_invalid_cols] = -1
        
        # 2: sweeping left → object를 왼쪽 열로 이동
        # (a) 정상적인 경우: item_idx > 0
        sweep_left_mask = (policy == 2) & valid_mask2 & (item_idx > 0)
        sweep_left_envs = env_indices[sweep_left_mask]
        sweep_left_rows = front_rows[sweep_left_mask]
        sweep_left_cols = item_idx[sweep_left_mask]
        sweep_left_new_cols = sweep_left_cols - 1
        obj_ids_to_move = self.shelf_object_config[sweep_left_envs, sweep_left_rows, sweep_left_cols]
        self.shelf_object_config[sweep_left_envs, sweep_left_rows, sweep_left_new_cols] = obj_ids_to_move
        self.shelf_object_config[sweep_left_envs, sweep_left_rows, sweep_left_cols] = -1
        
        # (b) 잘못된 경우: item_idx == 0 → 이동할 column이 없으므로, 해당 객체를 제거
        sweep_left_invalid_mask = (policy == 2) & valid_mask2 & (item_idx == 0)
        sweep_left_invalid_envs = env_indices[sweep_left_invalid_mask]
        sweep_left_invalid_rows = front_rows[sweep_left_invalid_mask]
        sweep_left_invalid_cols = item_idx[sweep_left_invalid_mask]
        self.shelf_object_config[sweep_left_invalid_envs, sweep_left_invalid_rows, sweep_left_invalid_cols] = -1

        # # print(self.shelf_object_config)
            

    def _apply_action(self) -> None:
        self._object_collection.update(dt=self.scene.physics_dt)
        # policy = self.actions[:, 0]
        # items = self.actions[:, 1]

        # processed_position = self.action_commands[policy]
        # processed_items = items
        
        # # Apply actions
        # cur_pos = self._object_collection.data.object_pos_w[]
        # self._object_collection.write_object_state_to_sim()
        pass

    def _get_observations(self) -> dict:
        camera_data = self._tiled_camera.data.output["rgb"]  # (N, H, W, C)
        # image_to_image = camera_data[0].detach()
        # image_to_image = image_to_image.cpu().numpy()
        # plt.figure(figsize=(8, 6))
        # plt.imshow(image_to_image)
        # plt.colorbar()  # 컬러 바 추가 (값의 강도 확인 가능)
        # plt.title("Predicted Mask (First Environment)")
        # plt.axis("off")  # 축 없애기
        # plt.show()
        
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
        self.previous_column_distribution = self.column_distribution.clone()
        
        # print(f"Colum distribution: {self.column_distribution}")
        
        ## shelf의 정면에서 보이는 object id 계산
        num_envs, num_rows, num_cols = self.shelf_object_config.shape # shelf_object_config 텐서의 shape 정보를 가져옴
        rows = torch.arange(num_rows, device=self.device).view(1, num_rows, 1) # 각 row 인덱스를 생성 (shape: (1, num_rows, 1)) – 여기서 row 인덱스는 0부터 num_rows-1
        mask = (self.shelf_object_config != -1)  # shape: (num_envs, num_rows, num_cols) // -1이 아닌 위치를 True로 마스킹
        candidate = torch.where(mask, rows.expand_as(self.shelf_object_config), torch.full_like(self.shelf_object_config, -1)) # shape: (num_envs, num_rows, num_cols) // -1이 아닌 위치에 row 인덱스를 배치
        max_row_indices = candidate.max(dim=1)[0]  # shape: (num_envs, num_cols)
        max_row_indices_clamped = torch.clamp(max_row_indices, min=0) # 음수를 0으로 클램핑
        shelf_front_object = self.shelf_object_config.gather(dim=1, index=max_row_indices_clamped.unsqueeze(1)).squeeze(1) # shape: (num_envs, num_cols)
        no_valid = (max_row_indices == -1) # 유효한 row 인덱스가 없는 경우 True, 그렇지 않으면 False
        shelf_front_object = torch.where(no_valid, torch.full_like(shelf_front_object, -1), shelf_front_object) # 유효한 row 인덱스가 없는 경우 -1로 채움
        self.shelf_front_object = shelf_front_object # shelf_front_object를 저장
        
        ## shelf의 정면에서 보이는 object 까지의 거리 계산
        num_envs2, num_rows2, num_cols2 = self.shelf_object_config.shape  # 예: (num_envs, 3, 4)
        rows2 = torch.arange(num_rows2, device=self.device).view(1, num_rows, 1)  # (1, 3, 1)
        mask2 = (self.shelf_object_config != -1)  # (num_envs, 3, 4)
        candidate2 = torch.where(mask2, rows.expand_as(self.shelf_object_config), torch.full_like(self.shelf_object_config, -1))
        max_row_indices2 = candidate2.max(dim=1)[0]  # shape: (num_envs, num_cols)
        mapping = torch.tensor([1.30, 1.17, 1.04], device=self.device, dtype=torch.float32)
        valid_mask = (max_row_indices2 >= 0)
        row_indices_clamped = torch.clamp(max_row_indices2, min=0)
        shelf_front_object_distance = mapping[row_indices_clamped]
        noise = torch.empty_like(shelf_front_object_distance).uniform_(-0.01, 0.01)
        # shelf_front_object_distance = torch.where(valid_mask, shelf_front_object_distance + noise, shelf_front_object_distance)
        shelf_front_object_distance = torch.where(valid_mask, shelf_front_object_distance + noise, torch.zeros_like(shelf_front_object_distance))
        self.shelf_front_object_distance = shelf_front_object_distance
        # print(f"Shelf front object distance: {self.shelf_front_object_distance}")
        # print(f"Shelf front object: {self.shelf_front_object}")
        
        self.previous_shelf_object_config = self.shelf_object_config.clone()
        self.previous_shelf_front_object = shelf_front_object.clone()
        self.previous_shelf_front_object_distance = shelf_front_object_distance.clone()
        
        # print(f"previous_shelf_front_object: {self.previous_shelf_front_object}")
        # print(f"previous_shelf_front_object_distance: {self.previous_shelf_front_object_distance}")
        
        if self.cfg.write_image_to_file:
            pred_to_save = pred.unsqueeze(-1)
            save_images_to_file(pred_to_save, f"fcn_output.png")
            
            # pred_to_image = pred[0].detach()
            # pred_to_image = pred_to_image.cpu().numpy()
            # pred_to_image = (pred_to_image * 255)
            # pred_to_image = np.clip(pred_to_image, 0, 255).astype(np.uint8)
            
            # plt.figure(figsize=(8, 6))
            # plt.imshow(pred_to_image, cmap="gray")  # Grayscale 컬러맵 유지
            # plt.colorbar()  # 컬러 바 추가 (값의 강도 확인 가능)
            # plt.title("Predicted Mask (First Environment)")
            # plt.axis("off")  # 축 없애기
            # plt.show()
                
        shelf_obs = torch.cat(
            (
                self.column_distribution,           # (num_envs, 4)
                self.shelf_front_object_distance,   # (num_envs, 4)
                self.shelf_front_object,            # (num_envs, 4)
                self.target_id                     # (num_envs, 1)
            ),
            dim=-1  # 마지막 차원에서 연결 / 최종 shape: (num_envs, 13)
        )
        # print(f"Shelf obs shape: {shelf_obs.shape}")
        # print(f"Shelf obs: {shelf_obs}")
        # print("-----------------------------")
        obs = {"policy": shelf_obs}
        return obs

    def _get_rewards(self) -> torch.Tensor:
        ## target grasping reward
        # 1. 이전 step의 행동 정보를 추출 (각 환경별로 policy와 sweeping column 인덱스)
        pol = self.action_policy.squeeze(-1) # shape: (num_envs,)
        # print(f"Policy: {pol}")
        col = self.action_column.squeeze(-1).long() # shape: (num_envs,), 값은 0~3 중 하나
        # print(f"Column: {col}")
        
        # 2. 이전 step의 shelf front object 정보에서, 각 환경의 'col' 인덱스에 해당하는 object id를 추출
        shelf_obj = self.previous_shelf_front_object.gather(dim=1, index=col.unsqueeze(1)).squeeze(1) # shape: (num_envs,)
        
        # 3. 현재 target id는 self.target_id에 저장되어 있으며, shape은 (num_envs, 1) / 이전 step에서의 action이 target grasping인 경우에만 reward 부여
        target = self.target_id.squeeze(-1) # shape: (num_envs,)
        
        target_grasp_condition = (pol == 0) & (shelf_obj != -1) & (shelf_obj == target)
        self.target_grasped = self.target_grasped | target_grasp_condition

        target_grasping_reward = self.cfg.traget_grasping * target_grasp_condition.float()
        
        
        ## sweeping right reward
        # 1. 조건 1: 이전 action이 sweeping right (policy==1)이고, 이전 column이 3이 아닌 경우
        swr_cond1 = (pol == 1) & (col != 3)
        
        # 2. 조건 2: 이전 column 분포에서 최대값
        argmax_col = torch.argmax(self.previous_column_distribution, dim=1)
        swr_cond2 = (argmax_col == col)
        # 이전 step의 column 분포에서 상위 2개 인덱스 추출
        # swr_top2 = torch.topk(self.previous_column_distribution, k=2, dim=1)
        # swr_cond2 = (col == swr_top2.indices[:, 0]) | (col == swr_top2.indices[:, 1])
        
        # 3.  환경에서 이전 action column과 이전 action column+1에 해당하는 거리를 각각 추출해서 밀려고 하는 column에 공간이 있는지 확인
        valid_right = (col < 3)
        # print(f"colum: {col}")
        # print(f"Valid right: {valid_right}")
        dist_current = self.previous_shelf_front_object_distance.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)
        # print(f"Dist current: {dist_current}")
        dist_current = dist_current + torch.tensor(0.05, device=self.device)
        # print(f"Dist current: {dist_current}")
        safe_index = torch.where(valid_right, col + 1, torch.zeros_like(col))
        # print(f"Safe index: {safe_index}")
        dist_right = self.previous_shelf_front_object_distance.gather(
            dim=1, index=safe_index.unsqueeze(1).long()
        ).squeeze(1)
        dist_right = torch.where(valid_right, dist_right, torch.zeros_like(dist_right))
        prev_obj_right = self.previous_shelf_front_object.gather(dim=1, index=safe_index.unsqueeze(1).long()).squeeze(1)
        swr_cond3 = valid_right & (((dist_right > 0) & (dist_right > dist_current)) | (prev_obj_right == -1))
        
        # 4. 조건 4: 이전 step의 shelf_front_object에서, 선택된 column(col)에 해당하는 object id가 -1이 아니어야 함
        previous_obj2 = self.previous_shelf_front_object.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)
        swr_cond4 = (previous_obj2 != -1)
        
        # 최종 조건
        sweeping_right_condition = swr_cond1 & swr_cond2 & swr_cond3 & swr_cond4
        # print(f"Sweeping right condition: {sweeping_right_condition}")
        sweeping_right_reward = self.cfg.hp_sweeping_right * sweeping_right_condition.float()
        
        
        ## sweeping left reward
        # 1. 조건 1: 이전 action이 sweeping left (policy==2)이고, 이전 column이 0이 아닌 경우
        swl_cond1 = (pol == 2) & (col != 0)

        # 2. 조건 2: 이전 step의 column 분포에서 각 환경마다 최대값의 인덱스가 이전 action column과 동일
        argmax_col1 = torch.argmax(self.previous_column_distribution, dim=1)  # (num_envs,)
        swl_ond2 = (argmax_col1 == col)
        # 이전 step의 column 분포에서 상위 2개 인덱스 추출
        # swl_top2 = torch.topk(self.previous_column_distribution, k=2, dim=1)
        # swl_ond2 = (col == swl_top2.indices[:, 0]) | (col == swl_top2.indices[:, 1])

        # 3.  환경에서 이전 action column과 이전 action column-1에 해당하는 거리를 각각 추출해서 밀려고 하는 column에 공간이 있는지 확인
        valid_left = (col > 0)
        dist_current2 = self.previous_shelf_front_object_distance.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)  # (num_envs,)
        dist_current2 = dist_current2 + torch.tensor(0.05, device=self.device)
        safe_index2 = torch.where(valid_left, col - 1, torch.zeros_like(col))
        dist_left = self.previous_shelf_front_object_distance.gather(
            dim=1, index=safe_index2.unsqueeze(1).long()
        ).squeeze(1)
        dist_left = torch.where(valid_left, dist_left, torch.zeros_like(dist_left))
        prev_obj_left = self.previous_shelf_front_object.gather(dim=1, index=safe_index2.unsqueeze(1).long()).squeeze(1)
        sw1_cond3 = valid_left & (((dist_left > 0) & (dist_left > dist_current2)) | (prev_obj_left == -1))

        # 4. 조건 4: 이전 step의 shelf_front_object에서, 선택된 column(col)에 해당하는 object id가 -1이 아니어야 함
        previous_obj3 = self.previous_shelf_front_object.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)
        swl_cond4 = (previous_obj3 != -1)

        # 최종 조건
        sweeping_left_condition = swl_cond1 & swl_ond2 & sw1_cond3 & swl_cond4
        # print(f"Sweeping left condition: {sweeping_left_condition}")
        sweeping_left_reward = self.cfg.hp_sweeping_left * sweeping_left_condition.float()
        
        
        ## grasping reward
        # 1. 조건 1: 이전 행동이 grasping (policy == 0) / 이전 step에서 선택된 object가 target이 아닌 경우
        gra_cond1 = (pol == 0) & (shelf_obj != target)
        
        # 2. 조건 2: 이전 step의 column 분포에서, 각 환경의 최대값 인덱스가 이전 선택 열(prev_col)과 같아야 함
        argmax_col2 = torch.argmax(self.previous_column_distribution, dim=1)  # shape: (num_envs,)
        gra_cond2 = (argmax_col2 == col)
        # gra_top2 = torch.topk(self.previous_column_distribution, k=2, dim=1)
        # gra_cond2 = (col == gra_top2.indices[:, 0]) | (col == gra_top2.indices[:, 1])
        
        # 3. 조건 3: 좌우 coulmn에 object가 있어야 함
        d_curr = self.previous_shelf_front_object_distance.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)
        d_curr = d_curr + torch.tensor(0.05, device=self.device)
        # 조건 확인을 위한 인덱스 계산
        valid_left2 = (col > 0)
        valid_right2 = (col < 3)
        safe_index_left  = torch.where(valid_left2,  col - 1, torch.zeros_like(col))
        safe_index_right = torch.where(valid_right2, col + 1, torch.zeros_like(col))
        # 오른쪽 column에 object가 있는지 확인을 위한 거리 계산
        d_left  = torch.where(valid_left2,
                      self.previous_shelf_front_object_distance.gather(dim=1, index=safe_index_left.unsqueeze(1).long()).squeeze(1),
                      torch.zeros_like(d_curr))
        # 왼쪽 column에 object가 있는지 확인을 위한 거리 계산
        d_right = torch.where(valid_right2,
                      self.previous_shelf_front_object_distance.gather(dim=1, index=safe_index_right.unsqueeze(1).long()).squeeze(1),
                      torch.zeros_like(d_curr))
        # 조건 3 계산
        gra_cond3 = torch.zeros_like(d_curr, dtype=torch.bool)
        gra_cond3 = torch.where(col == 0, (d_right > 0) & (d_right <= d_curr), gra_cond3)
        gra_cond3 = torch.where(col == 3, (d_left > 0) & (d_left <= d_curr), gra_cond3)
        mask_mid = (col > 0) & (col < 3)
        gra_cond3 = torch.where(mask_mid, ((d_left > 0) & (d_left <= d_curr)) & ((d_right > 0) & (d_right <= d_curr)), gra_cond3)
        
        # 4. 조건 4: 이전 step의 shelf_front_object에서, 선택된 column(col)에 해당하는 object id가 -1이 아니어야 함
        previous_obj4 = self.previous_shelf_front_object.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)
        gra_cond4 = (previous_obj4 != -1)
        
        # # 최종 조건
        grasp_condition = gra_cond1 & gra_cond2 & gra_cond3 & gra_cond4
        grasping_reward = self.cfg.hp_grasping * grasp_condition.float()
        
        # print(f"previous_shelf_object_config: {self.previous_shelf_object_config}")
        # print(f"current_shelf_object_config: {self.shelf_object_config}")
        # print(f"previous_shelf_fornt_distance: {self.previous_shelf_front_object_distance}")
        # print(f"graasping condition 1: {gra_cond1}")
        # print(f"graasping condition 2: {gra_cond2}")
        # print(f"Grasping condition 3: {gra_cond3}")
        # print(f"Grasping condition 4: {gra_cond4}")
        # print(f"graasping condition: {grasp_condition}")
        # print("-------------------------------------------------")
        
        
        
        #### 패널티 함수 ####
        ## 낮은 확율 분포 grasping 패널티
        argmax_col3 = torch.argmax(self.previous_column_distribution, dim=1)
        grasping_penalty_condition = (pol == 0) & (argmax_col3 != col)
        # low_col1 = torch.topk(self.previous_column_distribution, k=2, dim=1, largest=False)
        # grasping_penalty_condition = (pol == 0) & ((col == low_col1.indices[:, 0]) | (col == low_col1.indices[:, 1]))
        grasping_penalty = self.cfg.lp_grasping * grasping_penalty_condition.float()
        
        ## 낮은 확율 분포 sweeping right 패널티
        argmax_col4 = torch.argmax(self.previous_column_distribution, dim=1)  # shape: (num_envs,)
        sweeping_right_penalty_condition = (pol == 1) & (argmax_col4 != col)
        # low_col2 = torch.topk(self.previous_column_distribution, k=2, dim=1, largest=False)
        # sweeping_right_penalty_condition = (pol == 1) & ((col == low_col2.indices[:, 0]) | (col == low_col2.indices[:, 1]))
        sweeping_right_penalty = self.cfg.lp_sweeping_right * sweeping_right_penalty_condition.float()
        
        ## 낮은 확울 분포 sweeping left 패널티
        argmax_col5 = torch.argmax(self.previous_column_distribution, dim=1)
        sweeping_left_penalty_condition = (pol == 2) & (argmax_col5 != col)
        # low_col3 = torch.topk(self.previous_column_distribution, k=2, dim=1, largest=False)
        # sweeping_left_penalty_condition = (pol == 2) & ((col == low_col3.indices[:, 0]) | (col == low_col3.indices[:, 1]))
        sweeping_left_penalty = self.cfg.lp_sweeping_left * sweeping_left_penalty_condition.float()
        
        ## target sweeping 패널티
        selected_obj_ids = self.previous_shelf_front_object.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)
        target2 = self.target_id.squeeze(-1)
        target_sweeping_condition = ((pol == 1) | (pol == 2)) & (selected_obj_ids == target2)
        target_sweeping_penalty = self.cfg.traget_sweeping * target_sweeping_condition.float()
        
        ## empty column penalty (물체가 없는 column을 선택했을 때 패널티)
        no_object_condition = (self.previous_shelf_front_object.gather(dim=1, index=col.unsqueeze(1)).squeeze(1) == -1)
        no_object_penalty = self.cfg.empty_action * no_object_condition.float()
        
        ## sweeping이 가능하나 grasping을 선택한 경우 패널티
        # 1. 조건 1: grasping을 선택하였으나, 오른쪽으로 sweeping이 가능한 경우
        g_n_swr_cond1 = (pol == 0) & (col != 3)
        grasping_w_n_right_sweeping_condition = g_n_swr_cond1 & swr_cond2 & swr_cond3 & swr_cond4
        # 2. 조건 2: grasping을 선택하였으나, 왼쪽으로 sweeping이 가능한 경우
        g_n_swl_cond1 = (pol == 0) & (col != 0)
        grasping_w_n_left_sweeping_condition = g_n_swl_cond1 & swl_ond2 & sw1_cond3 & swl_cond4
        # 최종 조건
        grasping_w_n_sweeping_condition = grasping_w_n_right_sweeping_condition | grasping_w_n_left_sweeping_condition
        grasping_w_n_sweeping_penalty = self.cfg.grasping_w_n_sweeping * grasping_w_n_sweeping_condition.float()
        
        ## 이전 step에서 sweeping을 했던 object를 다시 sweeping하는 경우 패널티
        # sweeping right → sweeping left 인 경우
        safe_previous_col_right = torch.where(self.previous_action_column.squeeze(-1) < 3, self.previous_action_column.squeeze(-1) + 1, self.previous_action_column.squeeze(-1))
        swr_to_swl_penalty_condition = (
            (self.previous_action_policy.squeeze(-1) == 1) &  # 이전 step이 sweeping right
            (self.previous_action_column.squeeze(-1) < 3) &     # 이전 column이 3 미만
            (self.action_policy.squeeze(-1) == 2) &             # 현재 step이 sweeping left
            (self.action_column.squeeze(-1) == safe_previous_col_right)  # 현재 선택된 column이 이전 step safe_previous_col과 같음
        )
        # sweeping left → sweeping right 인 경우
        safe_previous_col_left = torch.where(self.previous_action_column.squeeze(-1) > 0,self.previous_action_column.squeeze(-1) - 1,self.previous_action_column.squeeze(-1))
        penalty_condition_left = (
            (self.previous_action_policy.squeeze(-1) == 2) &  # 이전 스텝이 sweeping left
            (self.previous_action_column.squeeze(-1) > 0) &     # 이전 column이 0보다 큼
            (self.action_policy.squeeze(-1) == 1) &             # 현재 스텝이 sweeping right
            (self.action_column.squeeze(-1) == safe_previous_col_left)  # 현재 선택된 column이 이전 스텝의 (col - 1)과 같음
        )
        #최종조건
        sweeping_again_penalty_condition = swr_to_swl_penalty_condition | penalty_condition_left
        sweeping_again_penalty = self.cfg.sweeping_again * sweeping_again_penalty_condition.float()
        
        total_reward = (target_grasping_reward + sweeping_right_reward + sweeping_left_reward + grasping_reward + grasping_penalty + sweeping_right_penalty + sweeping_left_penalty + target_sweeping_penalty + no_object_penalty + grasping_w_n_sweeping_penalty + sweeping_again_penalty)
        
        self.previous_action_policy = self.action_policy.clone()
        self.previous_action_column = self.action_column.clone()
        
        return total_reward
        
        # return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self, height_condition: float = 1.04 , rotation_condition: float= 0.9) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # objects_pose = self._object_collection.data.object_state_w.clone()  # (N, num_objects, 13)

        # num_envs, num_objects = objects_pose.shape[:2]  # 환경 개수, 물체 개수

        # # 모든 물체의 높이값 가져오기 (z축 위치)
        # heights = objects_pose[..., :, 2]  # (N, num_objects)
        # # print(f"Heights: {heights}")
        
        # # 물체가 떨어졌는지 확인
        # is_dropped = heights < height_condition  # (N, num_objects) -> Bool 텐서

        # # 모든 물체의 quaternion (N, num_objects, 4)
        # quat_tensor = objects_pose[..., :, 3:7].reshape(-1, 4)  # (N * num_objects, 4)

        # # 벡터 연산으로 quaternion → Euler 변환
        # roll, pitch, _ = euler_xyz_from_quat(quat_tensor)  # (N * num_objects,)

        # # 원래 차원으로 복구 (N, num_objects)
        # roll = roll.view(num_envs, num_objects)
        # pitch = pitch.view(num_envs, num_objects)

        # roll = normalize_angle(roll)
        # pitch = normalize_angle(pitch)

        # # 물체가 넘어졌는지 확인 (roll, pitch가 특정 값 이상이면 뒤집힌 것으로 간주)
        # is_flipped = (torch.abs(roll) > rotation_condition) | (torch.abs(pitch) > rotation_condition)

        # # 하나라도 물체가 떨어지거나 넘어졌다면 episode 종료
        # episode_done = torch.any(is_dropped | is_flipped, dim=1)  # (N,)
        
        
        
        
        ## ---------------------------------- ##
        ## 오브젝트 배열 기반 터미네이션 작성 ##
        pol = self.action_policy.squeeze(-1)
        col = self.action_column.squeeze(-1).long()
        shelf_obj = self.previous_shelf_front_object.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)
        target = self.target_id.squeeze(-1)
        
        num_envs, num_rows, num_cols = self.previous_shelf_object_config.shape  # (num_envs, 3, 4)
        env_indices = torch.arange(num_envs, device=self.device)
        
        ## sweeping right termination
        # 조건 1: col이 3이면 터미네이션 (workspace 밖으로 나감)
        term_spr_cond1 = (pol == 1) & (col == 3)

        # 조건 2: policy가 sweeping right이고, col이 3 미만인 경우
        valid = (pol == 1) & (col < 3)
        # 안전하게 오른쪽 셀을 선택하기 위해 valid인 경우에만 col+1, 그렇지 않으면 0번 인덱스를 사용
        safe_index = torch.where(valid, col + 1, torch.zeros_like(col))

        # 각 환경의 선택된 column의 front row 인덱스를 계산
        rows_tensor = torch.arange(num_rows, device=self.device).view(1, num_rows, 1)  # (1, num_rows, 1)
        mask = (self.previous_shelf_object_config != -1)  # (num_envs, num_rows, num_cols)
        candidate = torch.where(mask, rows_tensor.expand_as(self.previous_shelf_object_config),
                                torch.full_like(self.previous_shelf_object_config, -1))
        front_rows = candidate.max(dim=1)[0]  # (num_envs, num_cols)
        # 각 환경별로, 선택된 col의 front row 인덱스를 추출
        selected_front_rows = front_rows.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)  # (num_envs,)

        # 오른쪽 셀의 object id를 GPU에서 안전하게 추출 (여기서 selected_front_rows를 사용)
        env_indices = torch.arange(num_envs, device=self.device)
        right_cell = self.previous_shelf_object_config[env_indices, selected_front_rows, safe_index]

        # 최종 조건: valid 환경에 한해서, 오른쪽 셀의 값이 -1이 아니어야 함.
        term_spr_cond2 = valid & (right_cell != -1)

        termination_spr = term_spr_cond1 | term_spr_cond2
        
        
        ## sweeping left termination
        # 조건 1: sweeping left action이고, 선택한 col이 0이면 터미네이션 (workspace 밖으로 나감)
        term_swl_cond1 = (pol == 2) & (col == 0)

        # 조건 2: policy가 sweeping left이고, col이 0 미만이 아닌 (즉, col > 0) 경우
        valid_left = (pol == 2) & (col > 0)
        # 안전하게 왼쪽 셀 인덱스를 선택: valid_left인 경우에만 col - 1, 아니면 0번 인덱스 사용
        safe_index2 = torch.where(valid_left, col - 1, torch.zeros_like(col))
        # 각 환경의 선택된 column의 front row 인덱스를 계산 (이전 단계의 shelf_object_config 사용)
        rows_tensor2 = torch.arange(num_rows, device=self.device).view(1, num_rows, 1)  # (1, num_rows, 1)
        mask2 = (self.previous_shelf_object_config != -1)  # (num_envs, num_rows, num_cols)
        candidate2 = torch.where(mask2, rows_tensor2.expand_as(self.previous_shelf_object_config),
                                torch.full_like(self.previous_shelf_object_config, -1))
        front_rows2 = candidate2.max(dim=1)[0]  # (num_envs, num_cols)
        # 각 환경별로, 선택된 col의 front row 인덱스를 추출
        selected_front_rows2 = front_rows2.gather(dim=1, index=col.unsqueeze(1)).squeeze(1)  # (num_envs,)

        # 안전하게 왼쪽 셀의 object id를 추출: 이전 shelf_object_config의 safe_index2 위치 사용
        env_indices = torch.arange(num_envs, device=self.device)
        left_cell = self.previous_shelf_object_config[env_indices, selected_front_rows2, safe_index2]
        # 최종 조건: valid_left인 환경에 대해, 왼쪽 셀의 값이 -1이 아니어야 터미네이션 (즉, 물체가 존재하면 터미네이션)
        term_swl_cond2 = valid_left & (left_cell != -1)

        termination_swl = term_swl_cond1 | term_swl_cond2
        
        ## Target grasping termination
        termination_target = self.target_grasped  # shape: (num_envs,), bool
        
        termination = termination_spr | termination_swl | termination_target

        return termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self.previous_distribution[env_ids, :].zero_()
        self.column_distribution[env_ids, :].zero_()
        
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
        # print("-------------------new episode-------------------")
        # print(f"Target object name: {target_object_name}")
        # print(f"Target object id: {target_object_id}")

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
            
        # print(placement_list)
        # 1. placement_list에서 좌표와 object name 추출
        # 좌표 리스트와 object name 리스트로 분리함
        coords_tuple, obj_names_tuple = zip(*placement_list)
        
        # 2. 좌표 텐서 생성
        coords = torch.tensor(coords_tuple, device=self.device)  # (N, 2)
        num_rows = self.shelf_object_config.shape[1] # shelf_object_config의 행 개수
        coords[:, 0] = (num_rows - 1) - coords[:, 0] # placement_list의 좌표는 왼쪽 아래가 0,0이므로, shelf_object_config (왼쪽 위가 0,0)에 맞추기 위해 행 인덱스 반전
        obj_ids_tuple = tuple(map(lambda name: self.cfg.object_id_dict.get(name, 0), obj_names_tuple)) # object name을 object id로 변환
        object_ids_tensor = torch.tensor(obj_ids_tuple, device=self.device) # object id 리스트를 텐서로 변환
        expanded_coords = coords.unsqueeze(0).expand(env_ids.size(0), -1, -1)
        expanded_object_ids = object_ids_tensor.unsqueeze(0).expand(env_ids.size(0), -1)
        
        # 값 채워넣기
        self.shelf_object_config[env_ids] = -1 # 초기화 하는 환경만 -1로 초기화
        self.shelf_object_config[env_ids.unsqueeze(1),
                           expanded_coords[:, :, 0],
                           expanded_coords[:, :, 1]] = expanded_object_ids
        
        self.previous_shelf_object_config[env_ids] = self.shelf_object_config[env_ids].clone()
        
        self.action_policy[env_ids] = torch.zeros_like(self.action_policy[env_ids])
        self.action_column[env_ids] = torch.zeros_like(self.action_column[env_ids])
        self.previous_action_policy[env_ids] = torch.zeros_like(self.previous_action_policy[env_ids])
        self.previous_action_column[env_ids] = torch.zeros_like(self.previous_action_column[env_ids])
        
        self.target_grasped[env_ids] = False
        
        # print(self.shelf_object_config)
        # print(f"previous_shelf_object_config: {self.previous_shelf_object_config}")
        # print("------------------------------------")
        

    def get_category(self, item_name):
        for category, items in self.cfg.object_category.items():
            if item_name in items:
                return category
        return None#