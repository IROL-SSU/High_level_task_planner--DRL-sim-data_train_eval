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

from shelf_policy import mdp
from shelf_policy.shelf_multi_obj_env_cfg import ShelfEnvCfg
import torch
import os

##
# Pre-defined configs
##

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from shelf_policy.asset.ur5e import UR5e_CFG
from src_utils.shelf_utils import load_yaml_config, load_and_reshape_pose

@configclass
class UR5eShelfEnvCfg(ShelfEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = UR5e_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint"], 
            scale=0.5, 
            use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
            open_command_expr={"left_outer_knuckle_joint": 0.0, "right_outer_knuckle_joint": 0.0},
            close_command_expr={"left_outer_knuckle_joint": 0.5, "right_outer_knuckle_joint": 0.5},
        )


        # YAML 파일 로드
        object_cfgs = load_yaml_config(yaml_path="src/shelf_policy/params/environment_KTH.yaml")


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


        
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_arg2f_base_link_01",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.14],
                    ),
                ),
            ],
        )

        self.scene.finger_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_arg2f_base_link_01",
                    name="l_finger",
                    offset=OffsetCfg(
                        pos=(0.0, -0.07, 0.11),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_arg2f_base_link_01",
                    name="r_finger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.07, 0.11),
                    ),
                ),
            ],
        )
        
        self.scene.wrist_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_arg2f_base_link_01",
                    name="wrist",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, -0.14),
                    ),
                ),
            ],
        )

        # asset dict
        # asset_dict: dict = {"objects": ["target", "cup2", "cup3"]}


        self.observations.policy.target_obs_state.params["object_id_dict_rev"] = object_id_dict_rev
        self.observations.policy.goal_pos.params["command_name"] = "target_goal_pos"

        self.events.object_spawn.params["asset_dict"] = rigid_obj_dict
        
        self.events.object_spawn.params["pose_array"] = load_and_reshape_pose(object_pose_dict)
        self.events.object_spawn.params["object_id_dict"] = object_id_dict
        self.events.object_spawn.params["object_id_dict_rev"] = object_id_dict_rev
        self.events.object_spawn.params["ceiling_height"] = 1.8
        self.events.object_spawn.params["task_mode"] = "sweeping_right"

        self.commands.target_goal_pos.asset_dict = rigid_obj_dict
        self.commands.target_goal_pos.object_id_dict_rev = object_id_dict_rev
        self.commands.target_goal_pos.asset_name = "object_collection"

        self.rewards.reaching.params["object_id_dict_rev"] = object_id_dict_rev

        self.terminations.object_drop.params["height_condition"] = 0.99
        self.terminations.object_drop.params["rotation_condition"] = 0.9

        


@configclass
class UR5eShelfEnvCfg_PLAY(UR5eShelfEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
