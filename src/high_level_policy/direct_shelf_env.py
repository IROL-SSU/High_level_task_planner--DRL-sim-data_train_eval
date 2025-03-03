# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
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


@configclass
class DirectShelfEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
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
            usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack.usd",
            mass_props=MassPropertiesCfg(mass=100),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        debug_vis=False,
    )

    # YAML 파일 로드
    object_cfgs = load_yaml_config(
        yaml_path="src/shelf_policy/params/environment_KTH.yaml"
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

    pose_array = load_and_reshape_pose(object_pose_dict)
    asset_dict: dict = rigid_obj_dict


class DirectShelfEnv(DirectRLEnv):
    cfg: DirectShelfEnvCfg

    def __init__(
        self, cfg: DirectShelfEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self.target_id = torch.zeros(self.num_envs, 1, device=self.device)

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
        self.actions = torch.zeros(self.num_envs, 3, device=self.device)

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

        target_object_id = self.cfg.object_id_dict[
            choice(list(self.cfg.asset_dict.keys()))
        ]

        self.target_id[env_ids, 0] = target_object_id

        target_object_name = self.cfg.object_id_dict_rev[str(target_object_id)]

        asset_keys_list: list = list(self.cfg.asset_dict.keys())

        pose_array_tensor = torch.tensor(self.cfg.pose_array, device=self.device)

        # Orientation randomization 미적용
        orientations = torch.empty((env_ids.shape[0], 4), device=self.device)
        orientations[:, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        velocities = torch.zeros((env_ids.shape[0], 6), device=self.device)

        shuffle(asset_keys_list)

        for index, asset_name in enumerate(asset_keys_list):
            if asset_name == target_object_name:
                target_index = index

            pose_instance = pose_array_tensor[0, index // cols, index % cols]
            positions = pose_instance[:3] + self.scene.env_origins[env_ids, 0:3]
            object_ids = self._object_collection.find_objects(name_keys=asset_name)
            self._object_collection.write_object_link_state_to_sim(
                torch.cat((positions, orientations, velocities), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0],
            )
