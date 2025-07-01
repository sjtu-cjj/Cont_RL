# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import math
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp  # noqa: F401, F403

from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

#from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from Cont_RL.tasks.manager_based.cont_rl.cont_RL.contRL_env_cfg import ContRLRoughEnvCfg
##
# Pre-defined configs
##
from Cont_RL.assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughContRLEnvCfg(ContRLRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

       # 创建损伤配置
        damaged_robot_cfg = self._create_damaged_config_from_env()
        self.scene.robot = damaged_robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

    def _create_damaged_config_from_env(self):
        """从环境变量创建损伤配置"""
        damaged_joint = os.getenv('DAMAGED_JOINT', None)
        
        joint_mapping = {
            "FL_hip_joint": "FL_hip", "FL_thigh_joint": "FL_thigh", "FL_calf_joint": "FL_calf",
            "FR_hip_joint": "FR_hip", "FR_thigh_joint": "FR_thigh", "FR_calf_joint": "FR_calf",
            "RL_hip_joint": "RL_hip", "RL_thigh_joint": "RL_thigh", "RL_calf_joint": "RL_calf",
            "RR_hip_joint": "RR_hip", "RR_thigh_joint": "RR_thigh", "RR_calf_joint": "RR_calf"
        }
        
        if not damaged_joint:
            return UNITREE_GO2_CFG
        
        actuator_key = joint_mapping.get(damaged_joint, damaged_joint)
        
        if actuator_key not in UNITREE_GO2_CFG.actuators:
            print(f"❌ 错误：关节 '{damaged_joint}' 不存在，使用默认配置")
            return UNITREE_GO2_CFG
        
        # 方法1b：复制并修改
        new_actuators = UNITREE_GO2_CFG.actuators.copy()
        new_actuators[actuator_key] = UNITREE_GO2_CFG.actuators[actuator_key].replace(
            stiffness=0.0,
            damping=0.0
        )
        
        print(f"✅ 损伤关节: {damaged_joint} -> {actuator_key}")
        return UNITREE_GO2_CFG.replace(actuators=new_actuators)
        


@configclass
class UnitreeGo2RoughContRLEnvCfg_PLAY(UnitreeGo2RoughContRLEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        #self.scene.num_envs = 50
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            # self.scene.terrain.terrain_generator.num_rows = 5
            # self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2FlatContRLEnvCfg(UnitreeGo2RoughContRLEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 创建损伤配置
        damaged_robot_cfg = self._create_damaged_config_from_env()
        self.scene.robot = damaged_robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # # ========== 针对关节失效的奖励调整 ==========
        # # 1. 调整任务目标权重 - 更注重前进而非转向
        # self.rewards.track_lin_vel_xy_exp.weight = 2.0  # 提高前进奖励
        # self.rewards.track_ang_vel_z_exp.weight = 0.0   # 降低转向要求，因为RR_hip失效影响转向
        
        # # 2. 放宽稳定性要求 - 容忍必要的补偿动作
        # self.rewards.lin_vel_z_l2.weight = -1.0         # 减少垂直速度惩罚（原-2.0）
        # self.rewards.ang_vel_xy_l2.weight = -0.02       # 减少俯仰侧翻惩罚（原-0.05）
        # self.rewards.flat_orientation_l2.weight = -1.0  # 减少姿态惩罚（原-2.5）
        
        # # 3. 调整动作平滑性 - 允许更大的补偿动作
        # self.rewards.action_rate_l2.weight = -0.005     # 减少动作变化惩罚（原-0.01）
        # self.rewards.dof_acc_l2.weight = -1.0e-7        # 减少加速度惩罚（原-2.5e-7）
        
        # # 4. 重新设计步态奖励 - 适应三足/异常步态
        # self.rewards.feet_air_time.weight = 0.0         # 降低足部滞空要求（原0.25）
        
        # # 5. 移除或调整不适用的奖励
        # self.rewards.hip_joint_error.weight = 0.0      # 降低髋关节误差惩罚（原-1.0）
        # self.rewards.stand_still.weight = -1.0         # 降低静止惩罚（原-2.0）
        
        
        # # 7. 能效相关 - 考虑到补偿需要更多能量
        # self.rewards.energy_consumption.weight = -1e-5  # 减少能耗惩罚（原-2e-5）
        # self.rewards.dof_torques_l2.weight = -0.0001    # 减少扭矩惩罚（原-0.0002）
        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25


        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        # make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5

    


class UnitreeGo2FlatContRLEnvCfg_PLAY(UnitreeGo2FlatContRLEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

