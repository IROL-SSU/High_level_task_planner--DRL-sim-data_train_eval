# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Shelf-UR5e-Grasp-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.UR5eShelfGraspingEnvCfg
,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ShelfGraspingPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Shelf-UR5e-Grasp-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.UR5eShelfGraspingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.ShelfGraspingPPORunnerCfg,
    },
    disable_env_checker=True,
)