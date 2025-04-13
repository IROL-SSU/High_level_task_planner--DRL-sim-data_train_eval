# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR5e-Shelf-Sweep-Direct-v0",
    entry_point=f"{__name__}.ur5e_shelf_sweeping_env:UR5eShelfSweepEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_shelf_sweeping_env:UR5eShelfSweepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR5eShelfSweepPPORunnerCfg"

    },
)
