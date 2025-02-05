# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sensors.contact_sensor import ContactSensorCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectShelfSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    wrist_frame: FrameTransformerCfg = MISSING
    # cup_offset_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    cup: RigidObjectCfg = MISSING
    cup2: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.8, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/Table.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/Shelf3.usd", mass_props=MassPropertiesCfg(mass=50)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=True,
    )

     # robot mount
    mount_cfg =AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mount",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_goal_pos = mdp.ObjectGoalPosCommandCfg(
        asset_name="cup",
        init_pos_offset=(0.0, -0.15, 0.0),
        update_goal_on_success=False,
        position_success_threshold=0.03,
        debug_vis=True
    )

    ee_goal_pos = mdp.EEGoalPosCommandCfg(
        asset_name = "ee_frame",
        init_pos_offset=(0.0, 0.0, 0.0),
        update_goal_on_success=False,
        position_success_threshold=0.03,
        debug_vis=True
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.ActionTermCfg = MISSING
    finger_joint_pos: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.rl_joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.rl_joint_vel_rel)
        object_pose = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        ee_pos = ObsTerm(func=mdp.ee_pos_r)
        ee_quat = ObsTerm(func=mdp.ee_quat)
        target_goal_position = ObsTerm(func=mdp.target_goal_pose, params={"command_name": "target_goal_pos"})
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
    #         "pose_range": {"x": (0.0, 0.0), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("cup", body_names="SM_Cup_empty"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # task terms
    reaching_object = RewTerm(func=mdp.rewards_sweep.reaching_rew, params={"command_name": "target_goal_pos"}, weight=2.0)
    align_ee = RewTerm(func=mdp.rewards_sweep.ee_Align, params={}, weight=2.0)
    sweeping_object = RewTerm(func=mdp.rewards_sweep.pushing_target, 
                              params={"command_name": "target_goal_pos"}, 
                              weight=5.0)
                              
    sweeping_bonus = RewTerm(func=mdp.rewards_sweep.pushing_bonus, params={"command_name": "target_goal_pos"}, weight=5.0)
    homing_after_sweep = RewTerm(func=mdp.rewards_sweep.homing_reward, params={"command_name": "target_goal_pos"}, weight=7.0)
    # homing_bonus = RewTerm(func=mdp.rewards_sweep.homing_bonus, params={"command_name":"target_goal_pos"}, weight=7.0)

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.rewards_sweep.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # collision penalty
    shelf_collision = RewTerm(func=mdp.rewards_sweep.shelf_Collision, params={}, weight=-0.6)
    object_drop = RewTerm(func=mdp.rewards_sweep.Object_drop, weight=-0.2)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_drop = DoneTerm(func=mdp.Object_drop_Termination, time_out=True)
    object_drop2 = DoneTerm(func=mdp.Object2_drop_Termination, time_out=True)
    object_vel = DoneTerm(func = mdp.Object_vel_Termination, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )
    
    # homing = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "homing_after_sweep", "weight" : 10.0, "num_steps": 50000}
    # )



##
# Environment configuration
##


@configclass
class ShelfSweepingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectShelfSceneCfg = ObjectShelfSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
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
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.friction_correlation_distance = 0.00625
