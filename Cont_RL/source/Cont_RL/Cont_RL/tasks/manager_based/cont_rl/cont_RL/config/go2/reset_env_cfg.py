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
from Cont_RL.tasks.manager_based.cont_rl.cont_RL.reset_env_cfg import StandUpEnvCfg
##
# Pre-defined configs
##
from Cont_RL.assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2ResetEnvCfg(StandUpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # 设置Go2机器人配置，修改初始位姿为趴着的状态
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 覆盖初始位姿：让机器人以趴着的姿态开始（在安全关节限制内）
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.04)  # 稍微提高高度，避免穿透地面
        self.scene.robot.init_state.joint_pos = {
            # 髋关节：在安全范围内 (限制: ±1.0472弧度 = ±60度)
            "FL_hip_joint": 0.0523,     # 约3度 ✓
            "FR_hip_joint": -0.0523,    # 约-3度 ✓
            "RL_hip_joint": 0.3491,     # 约20度 ✓
            "RR_hip_joint": -0.3491,    # 约-20度 ✓
            
            # 大腿关节：趴着姿态 (前腿限制: -1.5708~3.4907, 后腿限制: -0.5236~4.5379)
            "FL_thigh_joint": 1.2,      # 约69度 ✓
            "FR_thigh_joint": 1.2,      # 约69度 ✓  
            "RL_thigh_joint": 1.2,      # 约69度 ✓
            "RR_thigh_joint": 1.2,      # 约69度 ✓
            
            # 小腿关节：在安全限制内 (限制: -2.7227~-0.83776弧度)
            ".*_calf_joint": -2.78,      # 约-143度，在安全范围内 ✓
        }
    
        
        # ========== 场景配置 ==========
        # 改为平地，起身任务不需要复杂地形
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # 移除高度扫描器，起身任务不需要
        self.scene.height_scanner = None
        if hasattr(self.observations.policy, 'height_scan'):
            self.observations.policy.height_scan = None
        
        # 关闭地形课程学习
        self.curriculum.terrain_levels = None
        
        # ========== 动作配置 ==========
        # 减少动作幅度，起身需要更精细的控制
        self.actions.joint_pos.scale = 0.25

        # ========== 事件配置：从定义的趴着姿态开始，微小随机化 ==========
        
        # 移除不必要的推力事件
        self.events.random_push = None
        
        # 调整质量随机化
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        
        # 覆盖重置参数：从趴着姿态开始，微小随机化
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.05, 0.05),          # 很小的位置随机化 ±5cm
                "y": (-0.05, 0.05),          # 很小的位置随机化 ±5cm
                "z": (0.04, 0.04),           # 基座高度微调 ±1cm
                "roll": (-0.05, 0.05),       # 微小滚转 ±3度
                "pitch": (-0.05, 0.05),      # 微小俯仰 ±3度  
                "yaw": (-0.1, 0.1)           # 微小朝向变化 ±6度
            },
            "velocity_range": {
                "x": (-0.05, 0.05),   "y": (-0.05, 0.05),   "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
            },
        }
        
        # 关节微小随机化：在初始位姿基础上添加微小扰动
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)  # ±5%的关节角度随机化
        self.events.reset_robot_joints.params["velocity_range"] = (-0.05, 0.05)  # 很小的初始关节速度

        # ========== 奖励配置：重新调整以促进起身行为 ==========
        
        # 调整起身任务的奖励权重 - 大幅增强起身动机
        if hasattr(self.rewards, 'base_height'):
            self.rewards.base_height.weight = 8.0  # 大幅增强高度奖励
            
        if hasattr(self.rewards, 'upright_posture'):
            self.rewards.upright_posture.weight = 6.0  # 大幅增强姿态奖励
            
        if hasattr(self.rewards, 'stand_success'):
            self.rewards.stand_success.weight = 20.0  # 巨大的成功奖励
            
        if hasattr(self.rewards, 'feet_contact'):
            self.rewards.feet_contact.weight = 3.0   # 增强接触奖励
        
        # 大幅增强基座接触惩罚，强制机器人离开地面
        if hasattr(self.rewards, 'base_contact_penalty'):
            self.rewards.base_contact_penalty.weight = -8.0  # 强烈惩罚趴着
        
        # 减小其他惩罚项，鼓励尝试动作
        self.rewards.dof_torques_l2.weight = -1e-6    # 减小扭矩惩罚
        self.rewards.action_rate_l2.weight = -0.005   # 减小动作变化惩罚
        self.rewards.ang_vel_xy_l2.weight = -0.02     # 减小角速度惩罚
        self.rewards.lin_vel_z_l2.weight = -0.1       # 减小垂直速度惩罚
        self.rewards.dof_acc_l2.weight = -5e-8        # 减小加速度惩罚
        
        # 保持安全惩罚以避免硬件损坏
        self.rewards.dof_pos_limits.weight = -0.2
        self.rewards.dof_vel_limits.weight = -0.2
        self.rewards.dof_torque_limits.weight = -0.2

        # ========== 终止条件：适配起身任务 ==========
        
        # 移除基座接触立即终止（起身过程中基座接触是正常的）
        self.terminations.base_contact = None


@configclass
class UnitreeGo2ResetEnvCfg_PLAY(UnitreeGo2ResetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ========== 播放场景配置 ==========
        # 少量环境用于测试观察
        self.scene.num_envs = 4  # 可以同时观察多个起身尝试
        self.scene.env_spacing = 3.0  # 增加间距便于观察
        
        # 简化地形设置
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        # ========== 播放时的行为配置 ==========
        # 关闭观测噪声，获得更稳定的表现
        self.observations.policy.enable_corruption = False
        
        # 移除训练时的随机干扰事件
        if hasattr(self.events, 'base_external_force_torque'):
            self.events.base_external_force_torque = None
        if hasattr(self.events, 'random_push'):
            self.events.random_push = None
        
