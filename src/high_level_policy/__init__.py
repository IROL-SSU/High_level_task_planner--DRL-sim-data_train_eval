# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import discrete_policy_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##


gym.register(
    id="Isaac-High-Level-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": discrete_policy_env_cfg.DiscreteActionShelfEnvCfg,
    },
    disable_env_checker=True,
)