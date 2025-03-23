# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
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
import torch

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
        spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack.usd", mass_props=MassPropertiesCfg(mass=100)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    finger_frame: FrameTransformerCfg = MISSING
    wrist_frame: FrameTransformerCfg = MISSING

    #Objects
    object_collection: RigidObjectCollectionCfg = MISSING



    

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    target_goal_pos = mdp.DynamicObjectGoalPosCommandCfg(
        asset_name=MISSING,
        asset_dict=MISSING,
        object_id_dict_rev=MISSING,
        init_pos_offset=(0.0, 0.15, 0.0),
        debug_vis=True,)


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
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        actions = ObsTerm(func=mdp.last_action)
        target_obs_state = ObsTerm(func=mdp.MA_object_position_in_RRF, params={"object_id_dict_rev": MISSING}, noise = Unoise(n_min=-0.01, n_max=0.01))
        ee_pos = ObsTerm(func=mdp.ee_pos_r)
        ee_quat = ObsTerm(func=mdp.ee_quat_r)
        goal_pos = ObsTerm(func=mdp.MA_target_goal_command, params={"command_name": MISSING})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    object_spawn = EventTerm(func=mdp.randomize_scene, 
                             params={"asset_dict": MISSING, "pose_array":MISSING, "object_id_dict": MISSING, "object_id_dict_rev": MISSING, "ceiling_height": MISSING, "task_mode": MISSING}, mode="reset")



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.rewards_sweep_ur5e.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reaching = RewTerm(
        func=mdp.rewards_sweep_ur5e.reward_for_hand_reaching,
        weight=3.0,
        params={"object_id_dict_rev": MISSING}
    )

    orientation = RewTerm(
        func=mdp.rewards_sweep_ur5e.ee_Align,
        weight=3.0,
        params={},
    )

    sweeping_object = RewTerm(func=mdp.rewards_sweep_ur5e.pushing_target, 
                              params={"command_name": "target_goal_pos"}, 
                              weight=5.0)
    
    # sweeping_bonus = RewTerm(func=mdp.rewards_sweep_ur5e.pushing_bonus, params={"command_name": "target_goal_pos"}, weight=7.0)

    homing_after_sweep = RewTerm(func=mdp.rewards_sweep_ur5e.homing_reward, params={"command_name": "target_goal_pos"}, weight=12.0)

    shelf_collision = RewTerm(func=mdp.rewards_sweep_ur5e.shelf_Collision, params={}, weight=-0.3)

    object_collision = RewTerm(func=mdp.rewards_sweep_ur5e.object_collision, params={}, weight=-0.7)

    # object_flip = RewTerm(func=mdp.rewards_sweep_ur5e.object_flip, params={}, weight=-0.2)

    # ee_vel_penalty = RewTerm(func=mdp.rewards_sweep_ur5e.ee_vel_penalty, params={}, weight = -1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_drop = DoneTerm(func=mdp.drop_object_termination, time_out=True, params={"height_condition":MISSING})
    shelf_collision = DoneTerm(func=mdp.shelf_collision_termination,time_out=True, params={"threshold": 0.1})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -2e-1, "num_steps": 30000}
    )

    # ee_vel = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "ee_vel_penalty", "weight": -4e-1, "num_steps": 10000})

    # shelf_penalty = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "shelf_collision", "weight": -3e-1, "num_steps": 20000})


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
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3.0

        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2 ** 17
