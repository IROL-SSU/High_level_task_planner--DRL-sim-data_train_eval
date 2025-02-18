# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from reach_policy.reach_cfg import ReachEnvCfg
import torch
import math



##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from reach_policy.asset.ur3 import UR3_CFG


@configclass
class UR3ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = UR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
       
        # Set actions for the specific robot type
        self.actions.arm_action = mdp.JointVelocityActionCfg(
            asset_name="robot", 
            joint_names=["shoulder_pan_joint",
                        "shoulder_lift_joint",
                        "elbow_joint",
                        "wrist_1_joint",
                        "wrist_2_joint",
                        "wrist_3_joint"], 
            scale=1.0, 
            use_default_offset=True
        )
        
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
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",
                    name="ee",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0),),
                ),
            ],
        )

        self.rewards.align_ee.params["asset_cfg"].body_names = ["tool0"]
        
        self.commands.ee_pose.body_name = "tool0"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        
        

@configclass
class UR3ReachEnvCfg_PLAY(UR3ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False