"""Sub-module containing command generators for goal position for objects"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, Articulation, RigidObjectCollection
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers.visualization_markers import VisualizationMarkers

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from .commands_cfg import ObjectGoalPosCommandCfg, EEGoalPosCommandCfg, DynamicObjectGoalPosCommandCfg

class DynamicObjectGoalPosCommand(CommandTerm):
    """
    Command term that generates position command for target object manipulation task.

    This command term generates 3D position commands for the object. 
    """

    cfg: DynamicObjectGoalPosCommandCfg
    """Configuration for the command term"""

    def __init__(self, 
                 cfg: DynamicObjectGoalPosCommandCfg, 
                 env: ManagerBasedRLEnv,):
        """
        Initialize the command term class.

        Args:
        cfg: The configuration parameters for the command term.
        env: The environment object
        """
        # initialize the bse class
        super().__init__(cfg, env)

        self.object_collection: RigidObjectCollection = env.scene[cfg.asset_name]
        
        self.asset_dict: dict = cfg.asset_dict

        self.object_id_dict_rev = cfg.object_id_dict_rev

        target_obj = self.object_id_dict_rev[str(env.target_id)]

        self.target_id = self.object_collection.find_objects(name_keys=target_obj)

        self.target_init_state_w = self.object_collection.data.default_object_state[:, self.target_id[0]]

        self.init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)

        self.pos_command_e = self.target_init_state_w[..., 0, :3] + self.init_pos_offset

        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins

    def _resample_command(self, env_ids: Sequence[int]):
        target_obj = self.object_id_dict_rev[str(self._env.target_id)]

        self.target_id = self.object_collection.find_objects(name_keys=target_obj)

        self.target_init_state_w = self.object_collection.data.object_link_state_w[:, self.target_id[0]]

        self.pos_command_e = self.target_init_state_w[..., 0, :3] + self.init_pos_offset - self._env.scene.env_origins
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    @property
    def command(self) -> torch.Tensor:
        """
        The desired goal pose in the environment frame. Shpe is (num_envs, 7)
        """
        return torch.cat((self.pos_command_w, self.target_init_state_w[..., 0, :3]), dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
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
        if not self.object_collection.is_initialized:
            return
        
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pos_command_w[..., :3], self.target_init_state_w[...,0, 3:7])

        # -- current body pose
        body_link_state_w = self.object_collection.data.object_link_state_w[:, self.target_id[0]] 
        self.current_pose_visualizer.visualize(body_link_state_w[...,0,:3], body_link_state_w[..., 0,3:7])



class ObjectGoalPosCommand(CommandTerm):
    """
    Command term that generates position command for target object manipulation task.

    This command term generates 3D position commands for the object.
    """

    cfg: ObjectGoalPosCommandCfg
    """Configuration for the command term"""

    def __init__(self, cfg: ObjectGoalPosCommandCfg, env: ManagerBasedRLEnv):
        """
        Initialize the command term class.

        Args:
        cfg: The configuration parameters for the command term.
        env: The environment object
        """
        # initialize the bse class
        super().__init__(cfg, env)

        # object
        self.target: RigidObject = env.scene[cfg.asset_name]



        # create buffers to store the command
        # -- command: (x, y, z)

        self.init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        self.pos_command_e = self.target.data.default_root_state[:, :3] + self.init_pos_offset
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins

        # -- orientation: (w, x, y, z)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0


    def __str__(self) -> str:
        msg = "ObjectGoalPosCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg
    

    """
    Properties
    """
    @property
    def command(self) -> torch.Tensor:
        """
        The desired goal pose in the environment frame. Shpe is (num_envs, 7)
        """
        return torch.cat((self.pos_command_w, self.quat_command_w), dim=-1)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        self.pos_command_w[env_ids, :] = self.target.data.root_state_w[env_ids, :3] + self.init_pos_offset

    def _update_command(self):
        pass


class EEGoalPosCommand(CommandTerm):
    """
    Command term that generates position command for target object manipulation task.

    This command term generates 3D position commands for the object. 
    """

    cfg: EEGoalPosCommandCfg
    """Configuration for the command term"""

    def __init__(self, cfg: EEGoalPosCommandCfg, env: ManagerBasedRLEnv):
        """
        Initialize the command term class.

        Args:
        cfg: The configuration parameters for the command term.
        env: The environment object
        """
        # initialize the bse class
        super().__init__(cfg, env)

        # robot
        self.ee: FrameTransformer = env.scene[cfg.asset_name]



        # create buffers to store the command
        # -- command: (x, y, z)

        self.init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        self.pos_command_w = self.ee.data.target_pos_w[..., 0, :]

        # -- orientation: (w, x, y, z)
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0
        self.count = 0

    def __str__(self) -> str:
        msg = "EEGoalPosCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg
    

    """
    Properties
    """
    @property
    def command(self) -> torch.Tensor:
        """
        The desired goal pose in the environment frame. Shpe is (num_envs, 7)
        """
        return torch.cat((self.pos_command_w, self.quat_command_w), dim=-1)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        self.pos_command_w[env_ids, :] = self.ee.data.target_pos_w[env_ids, 0,:]

    def _update_command(self):
        pass