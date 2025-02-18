# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, ContactSensorCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import shelf_policy.mdp as mdp

##
# Scene definition
##

@configclass
class ShelfSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    mount = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mount",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"omniverse://localhost/Library/Shelf/Arena/thor_table.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.79505), rot=(1.0, 0.0, 0.0, 0.0),),
    )
    
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/gorilla_rack_ur3.usd",
                                   activate_contact_sensors= True, mass_props=MassPropertiesCfg(mass=100)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.65, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=False,
    )
    # contact sensor
    shelf_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Shelf", update_period=0.0, history_length=6, debug_vis=False)

    # robots
    robot: ArticulationCfg = MISSING

    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    finger_frame: FrameTransformerCfg = MISSING
    wrist_frame: FrameTransformerCfg = MISSING
    
    # objects
    cup: RigidObjectCfg = MISSING
    cup2: RigidObjectCfg = MISSING

##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: ActionTerm = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.rl_joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.rl_joint_vel_rel)
        object_pose = ObsTerm(func=mdp.target_position_in_robot_root_frame)
        ee_pos = ObsTerm(func=mdp.ee_pos_r)
        ee_quat = ObsTerm(func=mdp.ee_quat)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # reset_object_position = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x":(-0.1, 0.1), "y": (-0.05, 0.05),"yaw":(-180, 180)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("cup", body_names="Cup"),
    #     },
    # )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    reaching_object = RewTerm(func=mdp.rewards_grasp.reaching_rew, params={}, weight=2.0)
    align_ee = RewTerm(func=mdp.rewards_grasp.align_ee_target, params={}, weight=2.0)
    grasp_object = RewTerm(func=mdp.rewards_grasp.grasp_handle, weight=7.0,
        params={
            "threshold": 0.03,
            "open_joint_pos": MISSING,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )
    # lifting_object = RewTerm(func=mdp.rewards_grasp.object_lift, params={"threshold": 1.05}, weight=10.0)
    homing_after_grasp = RewTerm(func=mdp.rewards_grasp.homing_reward, 
                                 params={"gripper_cfg": SceneEntityCfg("robot", joint_names=MISSING)}, weight=10.0)
    
    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # collision penalty
    # shelf_collision = RewTerm(func=mdp.rewards_grasp.shelf_Collision, params={}, weight=-0.2)
    object_drop = RewTerm(func=mdp.rewards_grasp.Object_drop, weight=-0.2)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_drop = DoneTerm(func=mdp.Object_drop_Termination, time_out=True, params={"condition": 1.04})
    object_drop2 = DoneTerm(func=mdp.Object2_drop_Termination, time_out=True, params={"condition": 1.04})
    object_vel = DoneTerm(func = mdp.Object_vel_Termination, time_out=True)
    # shelf_contact = DoneTerm(func=mdp.shelf_collision_termination, time_out=True, params={"threshold": 90.0})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class ShelfEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ShelfSceneCfg = ShelfSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: None
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 32
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
