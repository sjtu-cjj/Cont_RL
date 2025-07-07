# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

#training envs
gym.register(
    id="Cont-RL-Velocity-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Cont-RL-Velocity-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Cont-RL-Velocity-Flat-Unitree-Go2-ContRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg_ContRL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Cont-RL-Velocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Cont-RL-Velocity-Rough-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


# test envs
gym.register(
    id="Cont-RL-Velocity-Flat-Unitree-Go2-Test-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.test_env_cfg:UnitreeGo2FlatTestEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# streamingAC envs
gym.register(
    id="Cont-RL-Velocity-Rough-Unitree-Go2-ContRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contRL_env_cfg:UnitreeGo2RoughContRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_streamingAC_cfg:UnitreeGo2RoughStreamingRunnerCfg",
    },
)

gym.register(
    id="Cont-RL-Velocity-Rough-Unitree-Go2-ContRL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contRL_env_cfg:UnitreeGo2RoughContRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_streamingAC_cfg:UnitreeGo2RoughStreamingRunnerCfg",
    },
)

gym.register(
    id="Cont-RL-Velocity-Flat-Unitree-Go2-ContRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contRL_env_cfg:UnitreeGo2FlatContRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_streamingAC_cfg:UnitreeGo2FlatStreamingRunnerCfg",
    },
)

gym.register(
    id="Cont-RL-Velocity-Flat-Unitree-Go2-ContRL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.contRL_env_cfg:UnitreeGo2FlatContRLEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_streamingAC_cfg:UnitreeGo2FlatStreamingRunnerCfg",
    },
)

# reset envs
gym.register(
    id="Cont-RL-Unitree-Go2-ContRL-reset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_env_cfg:UnitreeGo2ResetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_reset_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)
gym.register(
    id="Cont-RL-Unitree-Go2-ContRL-reset-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_env_cfg:UnitreeGo2ResetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_reset_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)