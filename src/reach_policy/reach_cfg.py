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

import reach_policy.mdp as mdp



##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
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
    # shelf = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Shelf",
    #     spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack.usd", mass_props=MassPropertiesCfg(mass=100),),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    #     debug_vis=False,
    # )
    # robots
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING


    


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.77),
            pos_y=(-0.2, 0.2),
            pos_z=(0.3, 0.4),
            roll=(0.0, 0.0),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(0.0, 0.0),
        ),
    )
    

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.MA_joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.MA_joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pose = ObsTerm(func=mdp.ee_pose_b)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    reaching_object = RewTerm(func=mdp.rewards.reaching_rew, params={"command_name": "ee_pose"}, weight=2.0)
    align_ee = RewTerm(func=mdp.rewards.align_ee_target, params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"}, weight=2.0)
    
    # task terms
    # end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    # )
    # end_effector_position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    # )
    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_penalty,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # shelf_collision = DoneTerm(func=mdp.shelf_collision_termination,time_out=False, params={"threshold": 0.1})
    hand_velocity = DoneTerm(func=mdp.joint_velocity_termination, time_out=False, params={"threshold": 0.9})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.1, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.1, "num_steps": 4500}
    )

    # joint_acc = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_acc", "weight": -0.001, "num_steps": 4500}
    # )

##
# Environment configuration
##

@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 32
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
