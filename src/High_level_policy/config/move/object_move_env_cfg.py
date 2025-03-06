# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg, ArticulationCfg, RigidObjectCollectionCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg

from High_level_policy import mdp
from High_level_policy.high_level_policy_env_cfg import HighlevelEnvCfg
import torch
import os

##
# Pre-defined configs
##

from src_utils.shelf_utils import load_yaml_config, load_and_reshape_pose

@configclass
class MoveHighlevelEnvCfg(HighlevelEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # YAML 파일 로드
        object_cfgs = load_yaml_config(yaml_path="src/shelf_policy/params/environment.yaml")


        rigid_obj_dict = {}
        # 객체 정보 및 Pose 정보 가져오기
        object_path_dict = object_cfgs["objects"]
        object_pose_dict = object_cfgs["pose"]
        object_id_dict = object_cfgs["id"]
        object_id_dict_rev = {str(v): k for k, v in object_id_dict.items()}
        # 크기(키 개수) 비교 후 에러 발생
        if len(object_path_dict) != len(object_pose_dict):
            raise ValueError(f"Error: Object count mismatch! "
                            f"objects({len(object_path_dict)}) != pose({len(object_pose_dict)})")
        
        for key, value in object_path_dict.items():
            rigid_obj: RigidObjectCfg=RigidObjectCfg(prim_path=os.path.join("{ENV_REGEX_NS}", f"{key}"),
                                                    init_state=RigidObjectCfg.InitialStateCfg(pos=object_pose_dict[key][:3], rot=object_pose_dict[key][3:]),
                                                    spawn=UsdFileCfg(usd_path=value,
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
            
        # Set Cup as object
        self.scene.object_collection= RigidObjectCollectionCfg(rigid_objects=rigid_obj_dict)

        self.events.object_spawn.params["asset_dict"] = rigid_obj_dict
        self.events.object_spawn.params["pose_array"] = torch.tensor(load_and_reshape_pose(object_pose_dict), device=self.sim.device)
        self.events.object_spawn.params["object_id_dict"] = object_id_dict
        self.events.object_spawn.params["object_id_dict_rev"] = object_id_dict_rev
        self.events.object_spawn.params["ceiling_height"] = 1.8

        # print(self.scene.target)
        
        
@configclass
class MoveHighlevelEnvCfg_PLAY(MoveHighlevelEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
