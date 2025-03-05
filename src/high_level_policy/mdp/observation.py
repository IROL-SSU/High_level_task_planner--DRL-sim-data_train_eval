# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from dataclasses import MISSING

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation, RigidObjectCollection
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_unique
from omni.isaac.lab.sensors import FrameTransformerData, ContactSensorData
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

from random import choice

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def none_observation(env: ManagerBasedRLEnv,):
    
    obs = torch.zeros((env.num_envs, 1), device=env.device)

    return obs

