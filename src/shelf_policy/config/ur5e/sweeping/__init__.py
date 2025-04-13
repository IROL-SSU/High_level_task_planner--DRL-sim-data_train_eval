# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, joint_pos_env_cfg, joint_vel_env_cfg, ik_abs_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Shelf-UR5e-MultiObj-Test-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.UR5eShelfEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.UR5eSweepingRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Shelf-UR5e-MultiObj-Vel-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_vel_env_cfg.UR5eShelfEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg_vel.UR5eSweepingRunnerCfg,
    },
    disable_env_checker=True,
)


gym.register(
    id="Isaac-Shelf-UR5e-MultiObj-IK-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.UR5eShelfEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg_ik.UR5eSweepingRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"
    },
    disable_env_checker=True,
)
