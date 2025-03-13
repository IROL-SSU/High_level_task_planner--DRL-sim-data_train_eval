from __future__ import annotations
from dataclasses import MISSING

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, RigidObjectCollection
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, transform_points, euler_xyz_from_quat
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import FrameTransformer, ContactSensor

from src_utils.shelf_utils import normalize_angle

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def drop_object_termination(env: ManagerBasedRLEnv,
                            object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
                            height_condition: float = MISSING,
                            rotation_condition: float = MISSING):

    object_collection: RigidObjectCollection = env.scene[object_collection_cfg.name]

    objects_pose = object_collection.data.object_link_state_w  # (N, num_objects, 13)

    num_envs, num_objects = objects_pose.shape[:2]  # 환경 개수, 물체 개수

    # ✅ 모든 물체의 높이값 가져오기 (z축 위치)
    heights = objects_pose[..., :, 2]  # (N, num_objects)

    # ✅ 물체가 떨어졌는지 확인
    is_dropped = heights < height_condition  # (N, num_objects) -> Bool 텐서

    # ✅ 모든 물체의 quaternion (N, num_objects, 4)
    quat_tensor = objects_pose[..., :, 3:7].reshape(-1, 4)  # (N * num_objects, 4)

    # ✅ 벡터 연산으로 quaternion → Euler 변환
    roll, pitch, _ = euler_xyz_from_quat(quat_tensor)  # (N * num_objects,)

    # ✅ 원래 차원으로 복구 (N, num_objects)
    roll = roll.view(num_envs, num_objects)
    pitch = pitch.view(num_envs, num_objects)

    roll = normalize_angle(roll)
    pitch = normalize_angle(pitch)

    # ✅ 물체가 넘어졌는지 확인 (roll, pitch가 특정 값 이상이면 뒤집힌 것으로 간주)
    is_flipped = (torch.abs(roll) > rotation_condition) | (torch.abs(pitch) > rotation_condition)

    # ✅ 하나라도 물체가 떨어지거나 넘어졌다면 episode 종료
    episode_done = torch.any(is_dropped | is_flipped, dim=1)  # (N,)

    
    return episode_done  # (N,) -> 환경별 episode 종료 여부


def shelf_collision_termination(env: ManagerBasedRLEnv,
                                shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
                                threshold: float = MISSING):
    
    shelf: RigidObject = env.scene[shelf_cfg.name]

    shelf_vel = shelf.data.root_vel_w
    shelf_vel.sum()

    return torch.norm(shelf_vel , dim=-1, p=2)> threshold



