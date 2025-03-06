# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

##
# Register Gym environments.
##

##
# Joint Position Control
##


gym.register(
    id="Isaac-shelf-Direct-v0",
    entry_point=f"{__name__}.direct_shelf_env:DirectShelfEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct_shelf_env:DirectShelfEnvCfg",
    },
)
