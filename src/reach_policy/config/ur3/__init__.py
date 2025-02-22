# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import agents, joint_vel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##


gym.register(
    id="Isaac-UR3-Reach-Vel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_vel_env_cfg.UR3ReachEnvCfg,
       "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UR3ReachPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-UR3-Reach-Pos-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.UR3ReachEnvCfg,
       "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UR3ReachPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-UR3-Reach-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_vel_env_cfg.UR3ReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UR3ReachPPORunnerCfg,
    },
    disable_env_checker=True,
)
