# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with RSL-RL.
标准RSL-RL强化学习训练脚本

主要功能：
- 支持多种强化学习算法（PPO, SAC等）
- 分布式多GPU训练支持
- 完整的实验管理（日志、视频录制、检查点）
- 灵活的配置系统（Hydra + CLI参数）
- 多种环境类型支持（单/多智能体，直接/管理器模式）
"""

import platform
from importlib.metadata import version

# =============================================================================
# 版本检查和依赖管理
# =============================================================================
# 注释掉的旧版本检查（RSL-RL 2.3.0）
# if version("rsl-rl-lib") != "2.3.0":
#     if platform.system() == "Windows":
#         cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", "rsl-rl-lib==2.3.0"]
#     else:
#         cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", "rsl-rl-lib==2.3.0"]
#     print(
#         f"Please install the correct version of RSL-RL.\nExisting version is: '{version('rsl-rl-lib')}'"
#         " and required version is: '2.3.0'.\nTo install the correct version, run:"
#         f"\n\n\t{' '.join(cmd)}\n"
#     )
#     exit(1)

# 当前使用的版本检查（RSL-RL 2.3.1）
# 确保使用正确版本的RSL-RL库，保证训练环境的一致性和兼容性
if version("rsl-rl-lib") != "2.3.1":
    # 根据操作系统提供相应的安装命令
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", "rsl-rl-lib==2.3.1"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", "rsl-rl-lib==2.3.1"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{version('rsl-rl-lib')}'"
        " and required version is: '2.3.1'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip  # 自定义CLI参数处理模块


# =============================================================================
# 命令行参数定义
# =============================================================================
# 创建命令行参数解析器，用于配置训练参数
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")

# 视频录制相关参数
parser.add_argument("--video", action="store_true", default=False, 
                   help="Record videos during training. 是否录制训练视频")
parser.add_argument("--video_length", type=int, default=200, 
                   help="Length of the recorded video (in steps). 录制视频长度（步数）")
parser.add_argument("--video_interval", type=int, default=2000, 
                   help="Interval between video recordings (in steps). 视频录制间隔（步数）")

# 环境配置参数
parser.add_argument("--num_envs", type=int, default=None, 
                   help="Number of environments to simulate. 并行仿真环境数量")
parser.add_argument("--task", type=str, default=None, 
                   help="Name of the task. 任务名称")
parser.add_argument("--seed", type=int, default=None, 
                   help="Seed used for the environment. 环境随机种子")

# 训练配置参数
parser.add_argument("--max_iterations", type=int, default=None, 
                   help="RL Policy training iterations. 强化学习策略训练迭代次数")
parser.add_argument("--distributed", action="store_true", default=False, 
                   help="Run training with multiple GPUs or nodes. 使用多GPU或多节点分布式训练")

# 添加RSL-RL特定的CLI参数（如学习率、批量大小等）
cli_args.add_rsl_rl_args(parser)
# 添加AppLauncher的CLI参数（如设备选择、无头模式等）
AppLauncher.add_app_launcher_args(parser)

# 解析已知参数和未知参数（未知参数传递给Hydra配置系统）
args_cli, hydra_args = parser.parse_known_args()

# 如果启用视频录制，自动启用相机渲染
if args_cli.video:
    args_cli.enable_cameras = True

# 为Hydra配置系统清理sys.argv，只保留Hydra相关参数
sys.argv = [sys.argv[0]] + hydra_args

# 启动Isaac Lab应用和Omniverse仿真器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# =============================================================================
# 核心库导入
# =============================================================================
import gymnasium as gym  # OpenAI Gym接口
import os
import torch
from datetime import datetime

# RSL-RL核心组件：标准的在策略强化学习训练器
from rsl_rl.runners import OnPolicyRunner

# Isaac Lab环境相关导入
from isaaclab.envs import (
    DirectMARLEnv,                    # 直接多智能体强化学习环境
    DirectMARLEnvCfg,                # 多智能体环境配置
    DirectRLEnvCfg,                  # 直接单智能体RL环境配置
    ManagerBasedRLEnvCfg,            # 基于管理器的RL环境配置
    multi_agent_to_single_agent,      # 多智能体转单智能体工具函数
)

# 工具函数导入
from isaaclab.utils.dict import print_dict  # 字典美化打印
from isaaclab.utils.io import dump_pickle, dump_yaml  # 配置序列化

# RSL-RL与Isaac Lab的集成组件
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# 任务定义模块导入
import isaaclab_tasks  # noqa: F401  # Isaac Lab标准任务
from isaaclab_tasks.utils import get_checkpoint_path  # 检查点路径工具
from isaaclab_tasks.utils.hydra import hydra_task_config  # Hydra任务配置装饰器

import Cont_RL.tasks  # noqa: F401  # 自定义连续学习任务

# =============================================================================
# PyTorch性能优化设置
# =============================================================================
# 启用TF32以提高GPU计算性能（在Ampere架构GPU上）
torch.backends.cuda.matmul.allow_tf32 = True      # 矩阵乘法使用TF32
torch.backends.cudnn.allow_tf32 = True            # 卷积操作使用TF32
torch.backends.cudnn.deterministic = False        # 允许非确定性操作以提高性能
torch.backends.cudnn.benchmark = False            # 禁用cuDNN基准测试以节省内存


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """
    主训练函数，使用RSL-RL代理进行训练
    
    Args:
        env_cfg: 环境配置（支持多种环境类型）
        agent_cfg: 智能体和训练器配置
    """
    # =============================================================================
    # 配置参数覆盖和合并
    # =============================================================================
    # 使用CLI参数覆盖Hydra配置文件中的设置
    # CLI参数具有最高优先级，可以在运行时灵活调整配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    
    # 环境数量配置：CLI参数优先，否则使用配置文件默认值
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # 最大训练迭代次数配置
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # =============================================================================
    # 环境和设备配置
    # =============================================================================
    # 设置环境随机种子，确保实验可重现性
    # 注意：某些随机化在环境初始化时发生，所以在这里设置种子
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # =============================================================================
    # 分布式训练配置
    # =============================================================================
    # 多GPU/多节点分布式训练设置
    if args_cli.distributed:
        # 为每个进程分配不同的GPU设备
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # 为不同进程设置不同的随机种子，确保数据多样性
        # 避免所有进程生成相同的随机数序列
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # =============================================================================
    # 实验日志系统设置
    # =============================================================================
    # 实验日志根目录：logs/rsl_rl/实验名称/
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # 具体运行的日志目录：时间戳格式，便于区分不同运行
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    
    # 如果指定了运行名称，添加到目录名中
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # =============================================================================
    # Isaac环境创建
    # =============================================================================
    # 创建Isaac Lab仿真环境
    # render_mode设置：视频录制时使用rgb_array模式，否则不渲染
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 多智能体环境转换：如果是多智能体环境且算法只支持单智能体，进行转换
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # =============================================================================
    # 断点续训设置
    # =============================================================================
    # 在创建新日志目录之前保存续训路径
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # =============================================================================
    # 环境包装器链设置
    # =============================================================================
    # 环境包装器按以下顺序应用：
    # 1. 基础Isaac环境（物理仿真）
    # 2. 多智能体转换（如需要）
    # 3. 视频录制包装器（可选）
    # 4. RSL-RL适配包装器（必需）
    
    # 视频录制包装器配置
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),           # 视频保存路径
            "step_trigger": lambda step: step % args_cli.video_interval == 0,    # 录制触发条件
            "video_length": args_cli.video_length,                              # 视频长度
            "disable_logger": True,                                             # 禁用额外日志
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL环境包装器：将Isaac Lab环境适配为RSL-RL接口
    # clip_actions: 是否将动作限制在有效范围内
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # =============================================================================
    # 训练器创建和初始化
    # =============================================================================
    # 创建标准的在策略强化学习训练器（OnPolicyRunner）
    # 支持PPO, A2C, TRPO等在策略算法
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 将Git仓库状态记录到日志中，便于版本控制和实验追踪
    runner.add_git_repo_to_log(__file__)
    
    # 如果是续训，加载之前保存的模型检查点
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # =============================================================================
    # 配置文件持久化
    # =============================================================================
    # 将环境和智能体配置保存到日志目录，便于实验重现
    # 同时保存YAML（人类可读）和Pickle（Python对象）格式
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # =============================================================================
    # 开始强化学习训练
    # =============================================================================
    # 执行主要的训练循环
    # num_learning_iterations: 总的学习迭代次数
    # init_at_random_ep_len: 是否在随机episode长度处开始训练（有助于提高数据多样性）
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # =============================================================================
    # 资源清理
    # =============================================================================
    # 关闭仿真环境，释放GPU内存和其他资源
    env.close()


if __name__ == "__main__":
    # 运行主训练函数
    main()
    # 关闭Isaac Sim应用
    simulation_app.close()
