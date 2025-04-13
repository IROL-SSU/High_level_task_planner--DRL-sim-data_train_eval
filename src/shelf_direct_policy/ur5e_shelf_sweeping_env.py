# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_unique, matrix_from_quat, euler_xyz_from_quat
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject, RigidObjectCollectionCfg, RigidObjectCollection, AssetBase, AssetBaseCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import MassPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.sensors import FrameTransformerCfg, FrameTransformer
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip

import omni.isaac.lab.utils.string as string_utils

from src_utils.shelf_utils import load_yaml_config, load_and_reshape_pose
from src_utils.shelf_utils import normalize_angle

import torch
import os
from random import shuffle, choice
import gymnasium as gym


@configclass
class UR5eShelfSweepEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.0  # 500 timesteps
    decimation = 2
    action_space = 7
    observation_space = 52
    state_space = 0
    debug_vis = True


    # simulation
    sim: SimulationCfg = SimulationCfg(dt= 0.01,render_interval=decimation)
        
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"omniverse://localhost/Library/Shelf/Robots/UR5e/UR5e_v3.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0),
            activate_contact_sensors=False,),
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
                    "left_inner_finger_joint":0.0,},),
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
                        "shoulder_pan_joint": 3.14, #3.14
                        "shoulder_lift_joint": 3.14, #3.14
                        "elbow_joint": 3.14, #3.14
                        "wrist_1_joint": 6.28, #6.28
                        "wrist_2_joint": 6.28, #6.28
                        "wrist_3_joint": 6.28, #6.28
                    },
                    effort_limit=87.0,
                    stiffness=261,
                    damping=26.1,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["finger_joint","left_outer_finger_joint","left_inner_finger_knuckle_joint","left_inner_finger_joint", "right_outer_knuckle_joint", "right_outer_finger_joint", "right_inner_finger_joint", "right_inner_finger_knuckle_joint"],
                    effort_limit=200.0,
                    velocity_limit=0.5,
                    stiffness=2000,
                    damping=1000,),},)
    
    gripper_joint = ["finger_joint",
                    "right_outer_knuckle_joint",
                    "left_outer_finger_joint",
                    "left_inner_finger_knuckle_joint",
                    "left_inner_finger_joint",  
                    "right_outer_finger_joint", 
                    "right_inner_finger_joint", 
                    "right_inner_finger_knuckle_joint"]
    
    gripper_open_command_expr = {"finger_joint": 0.0, 
                               "right_outer_knuckle_joint": 0.0,
                               "left_inner_finger_knuckle_joint": 0.0,
                               "left_inner_finger_joint": 0.0, 
                               "left_outer_finger_joint": 0.0,
                               "right_outer_finger_joint": 0.0,
                               "right_inner_finger_joint": 0.0,
                               "right_inner_finger_knuckle_joint": 0.0}
    
    gripper_close_command_expr = {"finger_joint": 0.5, 
                                "right_outer_knuckle_joint": 0.5,
                                "left_inner_finger_knuckle_joint": -0.5,
                                "left_inner_finger_joint": -0.5, 
                                "left_outer_finger_joint": 0.0,
                                "right_outer_finger_joint": 0.0,
                                "right_inner_finger_joint": 0.5,
                                "right_inner_finger_knuckle_joint": -0.5}

    # shelf
    shelf = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Shelf",
        spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack.usd", mass_props=MassPropertiesCfg(mass=100)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=False,
    )


    marker_cfg =FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    ee_frame : FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/robotiq_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.13, 0.0, 0.0],
                    ),
                ),
            ],
    )

    finger_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/robotiq_base_link",
                    name="l_finger",
                    offset=OffsetCfg(
                        pos=(0.13, 0.07, 0.0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/robotiq_base_link",
                    name="r_finger",
                    offset=OffsetCfg(
                        pos=(0.13, -0.07, 0.0),
                    ),
                ),
            ],
        )

    wrist_frame : FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/robotiq_base_link",
                    name="wrist",
                    offset=OffsetCfg(
                        pos=(-0.14, 0.0, 0.0),
                    ),
                ),
            ],
    )

    object_cfgs = load_yaml_config(yaml_path="src/shelf_direct_policy/params/environment.yaml")


    rigid_obj_dict = {}
    # 객체 정보 및 Pose 정보 가져오기
    object_path_dict = object_cfgs["objects"]
    object_pose_dict = object_cfgs["pose"]
    object_id_dict = object_cfgs["id"]
    object_id_dict_rev = {str(v): k for k, v in object_id_dict.items()}
    pose_array = load_and_reshape_pose(object_pose_dict)
    ceiling_height = 1.8

    

    # 크기(키 개수) 비교 후 에러 발생
    if len(object_path_dict) != len(object_pose_dict):
        raise ValueError(f"Error: Object count mismatch! "
                        f"objects({len(object_path_dict)}) != pose({len(object_pose_dict)})")
    
    for key, value in object_path_dict.items():
        rigid_obj: RigidObjectCfg=RigidObjectCfg(prim_path=os.path.join("/World/envs/env_.*/", f"{key}"),
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
                                                                    mass_props=MassPropertiesCfg(mass=0.5),
                                                                ),
                                                            )
        
        rigid_obj_dict[key] = rigid_obj

    
    object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(rigid_objects= rigid_obj_dict)



    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    action_scale = 0.5
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.5
    hand_ori_reward_scale = 1.5
    sweeping_reward_scale = 6.0
    homing_reward_scale = 12.0
    action_penalty_scale = -0.0001
    velocity_penalty_scale = -0.0001
    soft_velocity_penalty_scale = -0.0001
    shelf_collision_penalty_scale = -0.3
    object_collision_penalty_scale = -0.2

    # termination conditions
    height_condition = 1.04
    rotation_condition = 0.9
    shelf_condition = 0.1

    curriculum_step = 10000

   

    joint_limit_soft_ratio = 0.2


class UR5eShelfSweepEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: UR5eShelfSweepEnvCfg

    def __init__(self, cfg: UR5eShelfSweepEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.set_debug_vis(self.cfg.debug_vis)

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )


        # initialize target_object id
        self.target_id = torch.zeros((self.scene.num_envs, 1), device=self.device)
        self.des_pos_offset = torch.tensor((0.0, 0.15, 0.0), dtype=torch.float, device=self.device)

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :6, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :6, 1].to(device=self.device)

        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        self._gripper_joint_ids, self._gripper_joint_names = self._robot.find_joints(self.cfg.gripper_joint)
        self._gripper_num_joints = len(self._gripper_joint_ids)

        # create  tensors for raw and processed actions for gripper 
        self._gripper_raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._gripper_processed_actions = torch.zeros(self.num_envs, self._gripper_num_joints, device=self.device)

        #parse gripper open command
        self._open_command = torch.zeros(self._gripper_num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.gripper_open_command_expr, self._gripper_joint_names
        )
        if len(index_list) != self._gripper_num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        
        self._open_command[index_list] = torch.tensor(value_list, device=self.device).unsqueeze(0)

        self._close_command = torch.zeros(self._gripper_num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.gripper_close_command_expr, self._gripper_joint_names
        )
        if len(index_list) != self._gripper_num_joints:
            raise ValueError(
                f"Could not resolve all joints for the action term. Missing: {set(self._joint_names) - set(name_list)}"
            )
        
        self._close_command[index_list] = torch.tensor(value_list, device=self.device).unsqueeze(0)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.init_gripper_x_axis = torch.tensor([-1, 0, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.init_gripper_y_axis = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))
        self.init_gripper_z_axis = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32).repeat((self.num_envs, 1))


        self._dist_reward_scale = self.cfg.dist_reward_scale
        self._hand_ori_reward_scale = self.cfg.hand_ori_reward_scale
        self._sweeping_reward_scale = self.cfg.sweeping_reward_scale
        self._homing_reward_scale = self.cfg.homing_reward_scale
        self._action_penalty_scale = self.cfg.action_penalty_scale
        self._velocity_penalty_scale = self.cfg.velocity_penalty_scale
        self._soft_velocity_penalty_scale = self.cfg.soft_velocity_penalty_scale
        self._shelf_collision_penalty_scale = self.cfg.shelf_collision_penalty_scale
        self._object_collision_penalty_scale = self.cfg.object_collision_penalty_scale

        

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._shelf = RigidObject(self.cfg.shelf)
        self._object_collections = RigidObjectCollection(self.cfg.object_collection)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self._finger_frame =  FrameTransformer(self.cfg.finger_frame)
        self._wrist_frame = FrameTransformer(self.cfg.wrist_frame)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["shelf"] = self._shelf
        self.scene.rigid_object_collections["objects"] = self._object_collections
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.sensors["fingers"] = self._finger_frame
        self.scene.sensors["wrist"] = self._wrist_frame

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        

        

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        # self._actions = actions.clone().clamp(-1.0, 1.0)

        self._actions = actions.clone().clamp(-3.14, 3.14)
        # targets = self._robot.data.default_joint_pos[:, :6] + self.robot_dof_speed_scales * self.dt * self._actions[:, :6] * self.cfg.action_scale
        
    
        self.robot_dof_targets[:, :6] = self._actions[:, :6] * self.cfg.action_scale + self._robot.data.default_joint_pos[:, :6]
        self.robot_dof_targets[:, :6] = torch.clamp(self.robot_dof_targets[:, :6], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # self.robot_dof_targets[:, :6] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        if self._actions[:, -1].dtype == torch.bool:
            binary_mask = self._actions[:, -1] == 0
        
        else:
            binary_mask = self._actions[:, -1] < 0

        binary_mask = binary_mask.unsqueeze(-1)

        self.robot_dof_targets[:, 6:] = torch.where(binary_mask, self._close_command, self._open_command)


    def _apply_action(self):                     
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # post-physics step calls

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_pose = self._object_collections.data.object_link_state_w
        _, num_objects = object_pose.shape[:2]

        heights = object_pose[..., :, 2]

        is_dropped = heights < self.cfg.height_condition

        quat_tensor = object_pose[..., :, 3:7].reshape(-1, 4)

        roll, pitch, _ = euler_xyz_from_quat(quat_tensor)

        # ✅ 원래 차원으로 복구 (N, num_objects)
        roll = roll.view(self.num_envs, num_objects)
        pitch = pitch.view(self.num_envs, num_objects)

        roll = normalize_angle(roll)
        pitch = normalize_angle(pitch)

        # ✅ 물체가 넘어졌는지 확인 (roll, pitch가 특정 값 이상이면 뒤집힌 것으로 간주)
        is_flipped = (torch.abs(roll) > self.cfg.rotation_condition) | (torch.abs(pitch) > self.cfg.rotation_condition)

        shelf_vel = self._shelf.data.root_vel_w

        shelf_collision = torch.norm(shelf_vel , dim=-1, p=2) > self.cfg.shelf_condition

        object_collision = torch.any(is_dropped, dim=1)

    
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        died = object_collision  

        # success = torch.full_like(torch.zeros(self.num_envs, device=self.device), False, dtype=torch.bool)
        return  died ,truncated

    def _get_rewards(self) -> torch.Tensor:
        return self._compute_rewards(self._dist_reward_scale,
                                     self._hand_ori_reward_scale,
                                     self._sweeping_reward_scale,
                                     self._homing_reward_scale,
                                     self._action_penalty_scale,
                                     self._velocity_penalty_scale,
                                     self._soft_velocity_penalty_scale,
                                     self._shelf_collision_penalty_scale,
                                     self._object_collision_penalty_scale)
        

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        def sweeping_right_mode(target_index: int, rows: int, cols: int, device):
            """
            Identify objects in front, right, and diagonal positions of the target.
            If the target is in the last row, include all objects in the front rows.
            Returns results as a GPU Tensor.
            """
            target_row = target_index // cols
            target_col = target_index % cols

            # (1) 앞쪽 찾기 (모든 앞쪽 행 포함)
            if target_row > 0:
                front_rows = torch.arange(target_row - 1, -1, -1, device=device)
                front_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col)), dim=1)
            else:
                front_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

            # (2) 오른쪽 찾기
            if target_col < cols - 1:
                right_index = torch.tensor([[target_row, target_col + 1]], device=device)
            else:
                right_index = torch.empty((0, 2), dtype=torch.int64, device=device)

            # (3) 우측 대각선 찾기
            if target_row > 0 and target_col < cols - 1:
                diagonal_indices = torch.stack((front_rows, torch.full_like(front_rows, target_col + 1)), dim=1)
            else:
                diagonal_indices = torch.empty((0, 2), dtype=torch.int64, device=device)

            # (4) 모든 결과를 GPU Tensor로 결합
            adjacent_array = torch.cat((front_indices, right_index, diagonal_indices), dim=0)

            return adjacent_array
        
        if self.common_step_counter > self.cfg.curriculum_step:
            self._action_penalty_scale = -0.1
            self._velocity_penalty_scale = -0.1
            self._soft_velocity_penalty_scale = -0.1

        joint_pos = self._robot.data.default_joint_pos[env_ids] 
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # shelf  in
        shelf_init_state = self._shelf.data.default_root_state.clone()[env_ids]
        shelf_init_state[:, 0:3] += self.scene.env_origins[env_ids]
        self._shelf.write_root_pose_to_sim(shelf_init_state[:, 0:7], env_ids=env_ids)
        self._shelf.write_root_velocity_to_sim(shelf_init_state[:, 7:], env_ids=env_ids)
        # self._shelf.reset()


        # object_shuffle
        rows, cols = len(self.cfg.pose_array[0]), len(self.cfg.pose_array[0][0])

        target_object_name = choice(list(self.cfg.rigid_obj_dict.keys()))

        target_object_id = self._object_collections.find_objects(name_keys=target_object_name)

        self.target_id[env_ids, 0] = target_object_id[0].to(self.target_id.dtype)

        asset_keys_list: list = list(self.cfg.rigid_obj_dict.keys())

        pose_array_tensor = torch.tensor(self.cfg.pose_array, device=self.device)

        # Orientation randomization 미적용
        orientations = torch.empty((env_ids.shape[0], 4), device=self.device)
        orientations[:, :] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        velocities = torch.zeros((env_ids.shape[0],6), device=self.device)


        shuffle(asset_keys_list)

        for index, asset_name in enumerate(asset_keys_list):
            if asset_name == target_object_name:
                target_index = index

            pose_instance = pose_array_tensor[0, index // cols, index % cols]
            positions = pose_instance[:3] + self.scene.env_origins[env_ids, 0:3]


            object_ids = self._object_collections.find_objects(name_keys=asset_name)

            self._object_collections.write_object_link_state_to_sim(
                torch.cat((positions, orientations, velocities), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0]
            )

        
        adjacent_indices = sweeping_right_mode(target_index, rows, cols, self.device)

        for adjacent in adjacent_indices:
            object_index = adjacent[0] * cols + adjacent[1]
            object_name = asset_keys_list[object_index]

            pose_instance = pose_array_tensor[0, adjacent[0], adjacent[1]]
            positions = pose_instance[:3] + self.scene.env_origins[env_ids, 0:3]
            positions[:, 2] = self.cfg.ceiling_height  # 높이 변경

            object_ids = self._object_collections.find_objects(name_keys=object_name)
            self._object_collections.write_object_link_pose_to_sim(
                torch.cat((positions, orientations), dim=1).unsqueeze(1),
                env_ids=env_ids,
                object_ids=object_ids[0]
            )

        # set goal position of target object
        target_ids = self.target_id.squeeze(-1).long()

        if self.common_step_counter < 1: 
            self._desired_pos_w = self._object_collections.data.object_link_state_w[torch.arange(self.num_envs), target_ids, :3] + self.des_pos_offset


        self._desired_pos_w[env_ids, :] = self._object_collections.data.object_link_state_w[env_ids, target_ids[env_ids], :3] + self.des_pos_offset


        

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
   
        self._joint_pos = self._robot.data.joint_pos[:, :]
        self._joint_vel = self._robot.data.joint_vel[:, :]
        target_ids = self.target_id.squeeze(-1).long()

        target_state_w = self._object_collections.data.object_state_w[torch.arange(self.num_envs), target_ids]

        self._object_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            target_state_w[:, :3],
            target_state_w[:, 3:7]
        )

        ee_pos_w = self._ee_frame.data.target_pos_w[:, 0, :]
        ee_quat_w = self._ee_frame.data.target_quat_w[:, 0, :]

        self._ee_pos_b, self._ee_quat_b = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3],
            self._robot.data.root_state_w[:, 3:7],
            ee_pos_w,
            ee_quat_w
        )

        obs = torch.cat(
            (self._joint_pos,
             self._joint_vel,
             self._object_pos_b,
             self._ee_pos_b,
             self._ee_quat_b,
             self._desired_pos_w,
             self._previous_actions), dim=-1)
        
        return {"policy": obs}

    # auxiliary methods

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
                """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

                current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/body_pose"
                )
                """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

                # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
                goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                

                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._object_collections.is_initialized:
            return
        
        # update the markers

        # -- current body pose

        # Get the target IDs directly from the environment tensor
        target_ids = self.target_id.squeeze(-1).long()  # Shape: (num_envs,)

        # Get the world state(position, orientation, linear velocity, angular velocity); R^13
        body_link_state_w = self._object_collections.data.object_link_state_w[torch.arange(self.num_envs), target_ids]

        self.current_pose_visualizer.visualize(body_link_state_w[..., :3], body_link_state_w[..., 3:7])
        # -- goal pose
        self.goal_pose_visualizer.visualize(self._desired_pos_w[..., :3], self._object_collections.data.default_object_state[torch.arange(self.num_envs), target_ids, 3:7])
    



    def _compute_rewards(self,
                         dist_reward_scale,
                         hand_ori_reward_scale,
                         sweeping_reward_scale,
                         homing_reward_scale,
                         action_penalty_scale,
                         joint_vel_penalty_scale,
                         soft_joint_vel_penalty_scale,
                         shelf_collision_penalty_scale,
                         object_collision_penalty_scale):
        # distance from hand to the drawer
        target_ids = self.target_id.squeeze(-1).long()

        target_pos_w = self._object_collections.data.object_pos_w[torch.arange(self.num_envs), target_ids].clone()
        target_lin_vel_w = self._object_collections.data.object_lin_vel_w[torch.arange(self.num_envs), target_ids].clone()
        ee_pos_w = self._ee_frame.data.target_pos_w.clone()
        ee_quat_w = self._ee_frame.data.target_quat_w.clone()

        objects_lin_vel_w = self._object_collections.data.object_lin_vel_w.clone()
        objects_lin_vel_w[torch.arange(self.scene.num_envs), target_ids, :] = 0.0

        shelf_pos_w = self._shelf.data.root_pos_w.clone()
        shelf_pos_w[:, 2] = shelf_pos_w[:, 2] + 1.05
        
        offset_pos_w = target_pos_w.clone()
        offset_pos_w[:, 0] = offset_pos_w[:, 0] 
        offset_pos_w[:, 1] = offset_pos_w[:, 1] - 0.07
        offset_pos_w[:, 2] = offset_pos_w[:, 2] + 0.09

        # reaching reward
        d = torch.norm((offset_pos_w[:, :3]- ee_pos_w[..., 0, :3]), dim=-1, p=2)
        dist_reward = torch.exp(-1.2 * d)

        # hand orientation reward
        ee_tcp_rot_mat = matrix_from_quat(ee_quat_w[..., 0, :])
        ee_tcp_y = ee_tcp_rot_mat[..., 1]
        align_y = torch.bmm(ee_tcp_y.unsqueeze(1), self.init_gripper_y_axis.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        ori_reward = torch.sign(align_y) * align_y ** 2

        # sweeping reward
        goal_distance = torch.norm((self._desired_pos_w - target_pos_w), dim=-1, p=2)
        zeta_m = torch.where(torch.norm(offset_pos_w-ee_pos_w[..., 0, :3], dim=-1, p=2)<0.04, 1, 0)
        vel_rew = torch.where(torch.abs(target_lin_vel_w[:, 1])< 0.1, 10 *target_lin_vel_w[:, 1], -1)
        sweeping_reward = torch.where(goal_distance<0.03, 1.5, zeta_m *((1 - goal_distance/0.15) + vel_rew))

        # homing reward
        joint_pos_error = torch.sum(torch.abs(self._robot.data.joint_pos[:, : 6] - self._robot.data.default_joint_pos[:, :6]), dim=1)
        reward_for_home_pose = torch.where(goal_distance < 0.03, torch.exp(-0.5 * joint_pos_error), 0)

        # object collision penalty
        object_collision_penalty = torch.tanh(torch.sum(torch.abs(objects_lin_vel_w), dim=(1,2))*2)
        # print(torch.sum(torch.abs(objects_lin_vel_w), dim=(1,2)))

        # shelf collision penalty
        shelf_distance = torch.norm(shelf_pos_w - ee_pos_w[..., 0, :3], dim=-1, p=2)
        zeta = torch.where(shelf_distance< 0.2, 1, 0)
        dst_l_shelf = self._finger_frame.data.target_pos_w[..., 0, 2] - shelf_pos_w[:, 2]
        dst_r_shelf = self._finger_frame.data.target_pos_w[..., 1, 2] - shelf_pos_w[:, 2]
        dst_wrist_shelf = self._wrist_frame.data.target_pos_w[..., 0, 2] - shelf_pos_w[:, 2]

        reward_l = 1 - dst_l_shelf / 0.02
        reward_r = 1 - dst_r_shelf / 0.02
        reward_wrist = 1 - dst_wrist_shelf / 0.08

        reward_l = torch.clamp(reward_l, 0, 1)
        reward_r = torch.clamp(reward_r, 0, 1)
        reward_wrist = torch.clamp(reward_wrist, 0, 1)

        shelf_collision_penalty = zeta * (reward_l + reward_r + reward_wrist)

        # joint_vel_l2
        joint_vel_l2_penalty = torch.sum(torch.square(self._robot.data.joint_vel[:, :6]), dim=1)

        # joint velocity soft limit penalty
        out_of_limits = (torch.abs(self._robot.data.joint_vel[:, :6]) - self._robot.data.soft_joint_vel_limits[:, :6] * self.cfg.joint_limit_soft_ratio)
        out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
        joint_vel_limit_penalty = torch.sum(out_of_limits, dim=1)

        # action_rate_l2
        action_rate_penalty = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        rewards = (dist_reward_scale * dist_reward
                   + hand_ori_reward_scale * ori_reward
                   + sweeping_reward_scale * sweeping_reward
                   + homing_reward_scale * reward_for_home_pose
                   + object_collision_penalty_scale * object_collision_penalty
                   + shelf_collision_penalty_scale * shelf_collision_penalty
                   + action_penalty_scale * action_rate_penalty
                   + joint_vel_penalty_scale * joint_vel_l2_penalty
                   + soft_joint_vel_penalty_scale * joint_vel_limit_penalty)

        if torch.any(rewards.isnan()):
            raise ValueError("Rewards cannot be NAN")

        self.extras["log"] = {
            "reaching_reward": (dist_reward_scale * dist_reward).mean(),
            "hand_ori_reward": (hand_ori_reward_scale * ori_reward).mean(),
            "sweeping_reward": (sweeping_reward_scale * sweeping_reward).mean(),
            "homing_reward": (homing_reward_scale * reward_for_home_pose).mean(),
            "object_collision_penalty": (object_collision_penalty_scale * object_collision_penalty).mean(),
            "shlef_collision_penalty": (shelf_collision_penalty_scale * shelf_collision_penalty).mean(),
            "action_penalty": (action_penalty_scale * action_rate_penalty).mean(),
            "joint_velocity_penalty": (joint_vel_penalty_scale * joint_vel_l2_penalty).mean(),
            "joint_limit_penalty": (soft_joint_vel_penalty_scale * joint_vel_limit_penalty).mean()
        }

        return rewards

