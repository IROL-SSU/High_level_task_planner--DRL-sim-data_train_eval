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

##
# Configuration
##


UR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"omniverse://localhost/Library/Shelf/Robots/UR3/ur3_2f85.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.1, 0.0, 0.79505),
        rot=(0.0 ,0.0 ,0.0 ,1.0),
        joint_pos={
            "shoulder_pan_joint": 0.0, # -1.7540559 / -1.6
            "shoulder_lift_joint": -2.0, # -1.27409 / -1.9
            "elbow_joint": 2.0, # 1.3439 / 1.9
            "wrist_1_joint": 0.0, # 0.0 
            "wrist_2_joint": 1.57, # 1.5708 / 1.57
            "wrist_3_joint": -0.8, # 1.5708 / 2.1
            "left_outer_knuckle_joint": 0.0, # 0.0
            "right_outer_knuckle_joint": 0.0, # 0.0  
            
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint",
                              "shoulder_lift_joint",
                              "elbow_joint",],
            velocity_limit=3.14,
            effort_limit=87.0,
            stiffness=800,
            damping=40,
        ),
        
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint",
                              "wrist_2_joint",
                              "wrist_3_joint"],
            velocity_limit=6.28,
            effort_limit=87.0,
            stiffness=800,
            damping=40,            
        ),
        
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_outer_knuckle_joint",
            "right_outer_knuckle_joint"],
            effort_limit=200.0,
            velocity_limit=0.5,
            stiffness=200,
            damping=20
        ),
    },
)

