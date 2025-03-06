# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.assets import RigidObjectCollection

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import discreteactioncfg



class DiscreteAction(ActionTerm):
    """
    Base class for discrete actions.
    """

    cfg: discreteactioncfg.DiscreteActionCfg
    """The configuration of the action term."""
    _asset: RigidObjectCollection
    """The rigidcollection asset on which the action term is applied."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: discreteactioncfg.DiscreteActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
        )

        self.env = env

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)

        # parse grasp command
        self._grasp_command = torch.zeros(3, device=self.device)
        self._grasp_command[:] = torch.tensor(self.cfg.grasping_expr, device=self.env.device)

        # parse sweeping_right command
        self._sweep_r_command = torch.zeros(3, device=self.device)
        self._sweep_r_command[:] = torch.tensor(self.cfg.sweeping_right_expr, device=self.env.device)

        # parse_sweeping_left command
        self._sweep_l_command = torch.zeros(3, device=self.device)
        self._sweep_l_command = torch.tensor(self.cfg.sweeping_left_expr, device=self.env.device)

        self.action_choices = torch.stack([
            self._grasp_command,
            self._sweep_r_command,
            self._sweep_l_command])


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        action_indices = actions[:, 0]

        # compute the command
        self._processed_actions[:, 0] = actions[:, 1]
        self._processed_actions[:, 1:4] = self.action_choices[action_indices]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids, :] = 0

    def apply_actions(self):
        # Get the target IDs directly from the environment tensor
        target_ids = self._processed_actions[:, 0].squeeze(-1).long()  # Shape: (num_envs,)

        # Get the world state(position, orientation, linear velocity, angular velocity); R^13
        target_state_w = self._asset.data.object_state_w[torch.arange(self.env.scene.num_envs), target_ids].clone()

        target_state_w[:, :3] = target_state_w[:, :3] + self._processed_actions[:, 1:4]

        self._asset.write_object_link_state_to_sim(target_state_w,
                                                   object_ids=target_ids)