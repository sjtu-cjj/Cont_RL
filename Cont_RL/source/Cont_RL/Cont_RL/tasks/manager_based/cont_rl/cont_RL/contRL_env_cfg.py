# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from .velocity_env_cfg import MySceneCfg, ActionsCfg, ObservationsCfg, EventCfg, RewardsCfg, TerminationsCfg, CurriculumCfg

import Cont_RL.tasks.manager_based.cont_rl.cont_RL.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##



##
# Environment configuration
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.0, 0.0), heading=(0.0, 0.0)
        ),
    )
##

@configclass
class RewardsCfg:
    """MDP的奖励项配置"""
    
    # ========== 任务目标奖励 ==========
    # 线性速度跟踪奖励：鼓励机器人在xy平面上跟踪目标速度，权重1.0表示这是主要任务目标
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # 角速度跟踪奖励：鼓励机器人跟踪绕z轴的目标角速度（转向），权重1.0表示转向同样重要
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # ========== 稳定性惩罚 ==========
    # 垂直速度惩罚：防止机器人在z轴方向上有过大的速度（跳跃或下沉），保持水平移动
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # xy轴角速度惩罚：防止机器人发生俯仰和侧翻，保持身体稳定
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    
    # ========== 能效和平滑性惩罚 ==========
    # 关节扭矩惩罚：鼓励使用较小的扭矩，提高能效并减少电机负载
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # 关节加速度惩罚：减少急剧的关节运动，使动作更平滑，减少震动
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # 动作变化率惩罚：防止相邻时间步的动作差异过大，使控制更平滑
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 能量消耗惩罚：基于扭矩和速度计算功率消耗，鼓励高效的运动模式
    energy_consumption = RewTerm(
        func=mdp.energy_consumption, 
        weight=-2e-5, 
        params={"joint_names": [".*"], "scale": 1.0}
    )
    
    # ========== 行为规范惩罚 ==========
    # 静止惩罚：当指令速度很小时，惩罚机器人的不必要运动，避免原地"颤抖"
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
        params={
            "command_name": "base_velocity", 
            "lin_threshold": 0.1, 
            "ang_threshold": 0.2
        }
    )
    # 髋关节误差惩罚：惩罚髋关节偏离理想位置，维持正确的腿部姿态
    hip_joint_error = RewTerm(
        func=mdp.dof_error_named,
        weight=-1.0,
        params={
            "joint_names": ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
            "scale": 1.0
        }
    )
    
    # ========== 步态质量奖励 ==========
    # 足部滞空时间奖励：鼓励适当的步态，足部在空中停留合适时间，形成自然的行走节奏
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    # 不期望接触惩罚：防止大腿和小腿接触地面，只允许足部接触，避免"爬行"
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh|.*_calf"), "threshold": 1.0},
    )

    # ========== 硬件安全限制 ==========
    # 关节位置限制惩罚：防止关节超出物理限制角度，保护硬件安全
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.4)
    # 关节速度限制惩罚：防止关节转速过快，避免电机过载和硬件损坏
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.4, params={"soft_ratio": 0.8})
    #dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.4)
    # 关节扭矩限制惩罚：防止输出扭矩超过电机额定值，保护电机和减速器
    dof_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.4)

    # ========== 可选惩罚项（当前权重为0，可根据需要启用）==========
    # 平躺姿态惩罚：惩罚机器人身体过度倾斜，保持直立姿态（当前未启用）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    # 关节速度L2惩罚：额外的关节速度平滑项（当前未启用）
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    # 基座高度惩罚：鼓励机器人保持特定高度0.45m，防止蹲得太低或跳得太高（当前未启用）
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=0.0, params={"target_height": 0.45})
    #base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=0.0)


@configclass
class ContRLRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
