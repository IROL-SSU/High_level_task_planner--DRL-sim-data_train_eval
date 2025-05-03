# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

# Configuration
##


UR5e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"omniverse://localhost/Library/Shelf/Robots/UR5e/UR5e_v3.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.79505),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,  # -1.7540559 / -1.6
            "shoulder_lift_joint": -2.2,  # -1.27409 / -1.9
            "elbow_joint": 2.2,  # 1.3439 / 1.9
            "wrist_1_joint": 0.0,  # 0.0
            "wrist_2_joint": 1.57,  # 1.5708 / 1.57
            "wrist_3_joint": 1.57,  # 1.5708 / 2.1
            "finger_joint": 0.0,  # 0.0
            "right_outer_knuckle_joint": 0.0,  # 0.0
            "right_outer_finger_joint": 0.0,
            "right_inner_finger_joint":0.0,
            "right_inner_finger_knuckle_joint":0.0,
            "left_outer_finger_joint":0.0,
            "left_inner_finger_knuckle_joint":0.0,
            "left_inner_finger_joint":0.0,

        },
        joint_vel={
            "shoulder_pan_joint": 0.0,  # -1.7540559 / -1.6
            "shoulder_lift_joint": 0.0,  # -1.27409 / -1.9
            "elbow_joint": 0.0,  # 1.3439 / 1.9
            "wrist_1_joint": 0.0,  # 0.0
            "wrist_2_joint": 0.0,  # 1.5708 / 1.57
            "wrist_3_joint": 0.0,  # 1.5708 / 2.1
            "finger_joint": 0.0,  # 0.0
            "right_outer_knuckle_joint": 0.0,  # 0.0
            "right_outer_finger_joint": 0.0,
            "right_inner_finger_joint":0.0,
            "right_inner_finger_knuckle_joint":0.0,
            "left_outer_finger_joint":0.0,
            "left_inner_finger_knuckle_joint":0.0,
            "left_inner_finger_joint":0.0,

        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            velocity_limit={
                "shoulder_pan_joint": 3.14,
                "shoulder_lift_joint":3.14,
                "elbow_joint": 3.14,
                "wrist_1_joint":6.28,
                "wrist_2_joint":6.28,
                "wrist_3_joint":6.28,
            },
            effort_limit=87.0,
            stiffness=261,
            damping=26.1,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint","left_outer_finger_joint","left_inner_finger_knuckle_joint","left_inner_finger_joint", "right_outer_knuckle_joint", "right_outer_finger_joint", "right_inner_finger_joint", "right_inner_finger_knuckle_joint"],
            effort_limit=200.0,
            velocity_limit=2.0,
            stiffness=2000,
            damping=1000,
        ),
    },
)

UR5e_HIGH_PD_CFG = UR5e_CFG.copy()
UR5e_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR5e_HIGH_PD_CFG.actuators["arm"].stiffness = 400.0
UR5e_HIGH_PD_CFG.actuators["arm"].damping = 80.0

