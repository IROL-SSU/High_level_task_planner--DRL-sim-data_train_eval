# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING

import torch
import numpy as np
from random import shuffle
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import FrameTransformerData

from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class Object_randomization(ManagerTermBase):

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):

        super().__init__(cfg, env)

        self.asset_pose = np.array([[-0.65, -0.20, 0.98], [-0.65, 0.0, 0.98], [-0.65, 0.20, 0.98]], dtype=np.float32)

    def __call__(self,
                 env: ManagerBasedRLEnv,
                 env_ids: torch.Tensor,
                 asset_dict: dict = MISSING,
                 ):
        
        for cur_env in env_ids.tolist():
            objects: list = asset_dict["objects"]
            shuffle(objects)
        
            for i, asset_cfg in enumerate(objects):
                # Write pose to simulation

                asset = env.scene[asset_cfg]

                pose_tensor = torch.tensor([self.asset_pose[i]], device=env.device)
                positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
                orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)

                asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device))
        


        

        


        







