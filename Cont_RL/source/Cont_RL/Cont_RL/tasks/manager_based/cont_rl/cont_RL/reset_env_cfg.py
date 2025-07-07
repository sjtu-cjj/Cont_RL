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

import Cont_RL.tasks.manager_based.cont_rl.cont_RL.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    
    # 起身任务不需要速度指令，可以留空或简化
    # 为了保持兼容性，我们保留一个简单的零速度指令
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(100.0, 100.0),  # 很长时间不重新采样
        rel_standing_envs=1.0,  # 全部环境都是站立指令
        rel_heading_envs=0.0,   # 不需要朝向指令
        heading_command=False,  # 关闭朝向指令
        debug_vis=False,        # 关闭可视化
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Events for stand-up task."""

    # startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    # reset: 让机器人趴在地上
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 位置稍微随机
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                # 趴在地上，基座贴近地面
                "z": (0.04, 0.04),  
                # 基本趴着，可以稍微有点侧倾
                "roll": (-0.01, 0.01),    # 约 ±11度，轻微侧倾
                "pitch": (-0.01, 0.01),   # 约 ±6度，轻微前后倾
                # 朝向随机
                "yaw": (-math.pi, math.pi),
            },
            # 基本静止状态
            "velocity_range": {
                "x": (-0.1, 0.1),  "y": (-0.1, 0.1),  "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            },
        },
    )

    # 重置关节到趴着的姿态，稍有随机
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),  # 相对保守的关节角度范围，接近趴着的姿态
            "velocity_range": (-0.1, 0.1),  # 很小的初始关节速度，基本静止
        },
    )

    # 暂时关闭外力干扰，专注于基本起身动作学习
    # random_push = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(20.0, 30.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (-100.0, 100.0),
    #         "torque_range": (-50.0, 50.0),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the stand-up task."""
    
    # ========== 主要任务奖励：起身 ==========
    
    # 1️⃣ 抬升机身高度奖励 (最重要)
    base_height = RewTerm(
        func=mdp.base_height_reward,
        weight=3.0,
        params={"target_height": 0.42}
    )
    
    # 2️⃣ 保持直立姿态奖励 (很重要)
    upright_posture = RewTerm(
        func=mdp.upright_posture_reward,
        weight=2.5,
        params={"std": 0.4}
    )
    
    # 3️⃣ 成功站立奖励 (一次性大奖励)
    stand_success = RewTerm(
        func=mdp.stand_up_success_reward,
        weight=10.0,
        params={
            "height_thresh": 0.38,
            "angle_thresh": 0.3
        }
    )
    
    # 4️⃣ 足部接触奖励 (稳定性)
    feet_contact = RewTerm(
        func=mdp.feet_contact_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "min_contacts": 3
        }
    )
    
    # ========== 行为规范和安全约束 ==========
    
    # 平滑控制惩罚
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    
    # 能效惩罚 (较小，因为起身需要较大扭矩)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5e-6)
    
    # 能量消耗惩罚
    energy_consumption = RewTerm(
        func=mdp.energy_consumption, 
        weight=-1e-5, 
        params={"joint_names": [".*"], "scale": 1.0}
    )
    
    # 避免不期望的接触 (身体其他部位)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh|.*_calf"), "threshold": 1.0},
    )
    
    # 硬件安全限制
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.3)
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.3, params={"soft_ratio": 0.8})
    dof_torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.3)
    
    # ========== 稳定性奖励 ==========
    
    # 减少俯仰/滚转角速度 (避免剧烈摆动)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    
    # 减少垂直速度波动 (平稳起身)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    
    # ========== 可选的辅助奖励 ==========
    
    # 关节加速度惩罚 (平滑运动)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    
        # 基座接触惩罚（防止一直趴着）
    base_contact_penalty = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    
    # ========== 目标姿态引导奖励（权重较小）==========
    # 引导机器人向标准站立姿态移动，但权重设置得比较小
    joint_target_posture = RewTerm(
        func=mdp.joint_target_posture_reward,
        weight=0.5,  # 权重较小，作为引导性奖励
        params={
            "target_joint_pos": {
                "FL_hip_joint": 0.1,
                "FL_thigh_joint": 0.8, 
                "FL_calf_joint": -1.5,
                "FR_hip_joint": -0.1,
                "FR_thigh_joint": 0.8,
                "FR_calf_joint": -1.5,
                "RL_hip_joint": 0.1,
                "RL_thigh_joint": 1.0,
                "RL_calf_joint": -1.5,
                "RR_hip_joint": -0.1,
                "RR_thigh_joint": 1.0,
                "RR_calf_joint": -1.5,
            },
            "std": 0.8  # 比较宽松的标准差，避免过于严格
        }
    )


@configclass
class TerminationsCfg:
    """Termination terms for the stand-up task."""

    # 超时终止 (给足时间让机器人尝试起身)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 成功站立后可选择终止 (可以让它继续保持平衡，也可以立即终止节省时间)
    # 注释掉下面这行可以让机器人站起来后继续保持平衡
    # stand_success = DoneTerm(
    #     func=mdp.stand_up_success_reward,  # 重用奖励函数作为终止条件
    #     params={"height_thresh": 0.38, "angle_thresh": 0.3}
    # )
    
    # 基座长时间接触地面 (防止机器人一直趴着不动) - 更宽松
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 50.0},  # 较高阈值
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the stand-up task."""

    # 起身任务可以不需要复杂地形，注释掉地形课程学习
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    # 可选：实现起身难度课程学习，比如从较小的初始角度开始，逐渐增加到完全倒地
    # 这需要在mdp中实现对应的课程学习函数
    pass


##
# Environment configuration
##


@configclass  
class StandUpEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stand-up recovery environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=2.5)  # 减少环境数量，起身任务更复杂
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
        self.episode_length_s = 15.0  # 缩短episode长度，起身应该在15秒内完成
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

        # 起身任务不需要地形课程学习，保持平地
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


