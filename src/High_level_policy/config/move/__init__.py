# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, object_move_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

# gym.register(
#     id="Isaac-High-Level-Policy-Test-v0",
#     entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": object_move_env_cfg.MoveHighlevelEnvCfg,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HighLevelPolicyRunnerCfg,
#     },
#     disable_env_checker=True,
# )