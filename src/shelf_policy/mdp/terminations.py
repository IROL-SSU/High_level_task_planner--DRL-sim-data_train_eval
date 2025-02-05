from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, transform_points
from omni.isaac.lab.managers import SceneEntityCfg, ManagerTermBase
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.sensors import FrameTransformer, ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold



class Object_drop_Termination(ManagerTermBase):
    def __init__(self, cfg: DoneTerm , env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.1])

    def __call__(self, env:ManagerBasedRLEnv, condition: float):
        drop = self.object_drop(env, condition=condition)

        return drop

    def object_drop(self, env: ManagerBasedRLEnv, condition: float)-> torch.Tensor:
        
        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]

        # print("cup1: {}".format(offset_pos[:, 2]))

        return offset_pos[..., 2] < condition #0.762


class Object2_drop_Termination(ManagerTermBase):
    def __init__(self, cfg: DoneTerm , env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup2")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.1])

    def __call__(self, env:ManagerBasedRLEnv, condition: flaot):
        drop = self.object_drop(env, condition=condition)

        return drop

    def object_drop(self, env: ManagerBasedRLEnv, condition: float)-> torch.Tensor:
        
        ee_pos_w = self._ee.data.target_pos_w[..., 0, :]
        obj_pos = self._target.data.root_pos_w.clone()
        obj_pos[:, 0] = obj_pos[:, 0] 
        obj_pos[:, 1] = obj_pos[:, 1]
        obj_pos[:, 2] = obj_pos[:, 2] + 0.03
        
        distance = torch.norm((obj_pos - ee_pos_w), dim=-1, p=2)
        
        offset_pos = transform_points(self._top_offset,self._target.data.root_pos_w, self._target.data.root_state_w[:, 3:7] )[..., 0 , :]

        # print("cup2: {}".format(offset_pos[:, 2]))
        

        return torch.where(distance < 0.03, False , offset_pos[..., 2] < condition) #0.762
    

class Object_vel_Termination(ManagerTermBase):
    def __init__(self, cfg: DoneTerm , env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg = SceneEntityCfg("cup")
        ee_frame_cfg = SceneEntityCfg("ee_frame")
        

        self._target: RigidObject = env.scene[object_cfg.name]
        self._ee: FrameTransformer = env.scene[ee_frame_cfg.name]

        self._target_last_w = self._target.data.root_pos_w.clone()

        self._top_offset = torch.zeros((env.num_envs, 3), device=env.device)
        self._top_offset[:, :3] = torch.tensor([0.0, 0.0, 0.07])

    def __call__(self, env:ManagerBasedRLEnv,):
        term = self.object_vel(env)

        return term

    def object_vel(self, env: ManagerBasedRLEnv,)-> torch.Tensor:
        

        object_vel = self._target.data.root_lin_vel_w.clone()
        velocity = torch.norm(object_vel, dim=-1, p=2)

        return velocity > 2.0
    

def shelf_collision_termination(
    env: ManagerBasedRLEnv,
    threshold: float = 10.0,
    contact_sensor = SceneEntityCfg("shelf_contact")
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    shelf_contact: ContactSensor = env.scene[contact_sensor.name]

    net_force = torch.norm(shelf_contact.data.net_forces_w[...,0,:], dim=1)
    
    # print(f"net_force: {net_force}")

    # rewarded if the object is lifted above the threshold
    return net_force > threshold