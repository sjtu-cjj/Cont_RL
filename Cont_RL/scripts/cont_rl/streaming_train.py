# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train streamingAC agent with RSL-RL."""

import platform
from importlib.metadata import version

if version("rsl-rl-lib") != "2.3.1":
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
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train streamingAC agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Path to pretrained model checkpoint.")
parser.add_argument("--finetune_mode", type=str, default="full", help="Finetune mode for pretrained model.")
# 关节损伤相关参数
parser.add_argument("--enable_joint_damage", action="store_true", default=False, help="Enable joint damage simulation.")
parser.add_argument("--damage_probability", type=float, default=0.1, help="Probability of joint damage per step.")
# parser.add_argument("--damage_duration_min", type=int, default=10, help="Minimum duration of joint damage (steps).")
# parser.add_argument("--damage_duration_max", type=int, default=100, help="Maximum duration of joint damage (steps).")
parser.add_argument("--max_damaged_joints", type=int, default=2, help="Maximum number of joints that can be damaged simultaneously.")
parser.add_argument("--damage_type", type=str, default="zero", choices=["zero", "partial", "random"], 
                   help="Type of joint damage: zero (set to 0), partial (reduce by factor), random (random values).")
parser.add_argument("--damage_severity", type=float, default=1.0, help="Severity of damage (0-1, for partial damage type).")
parser.add_argument("--output_damage_info", action="store_true", default=True, 
                   help="Enable detailed damage information output during training.")
# NOTE: 删除了 --distributed 参数，因为StreamingRunner不支持分布式训练
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import time
from datetime import datetime
import numpy as np
import random

# 导入StreamingRunner而不是OnPolicyRunner
from rsl_rl.runners import StreamingRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Cont_RL.tasks  # noqa: F401

# 导入关节损伤模块
from joint_damage import (
    JointDamageWrapper,
    create_damage_config,
    print_damage_config_info,
    find_damage_wrapper,
    print_damage_statistics
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def format_time(seconds):
    """格式化时间，将秒数转换为易读的格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}小时 {minutes}分钟 {secs}秒"
    elif minutes > 0:
        return f"{minutes}分钟 {secs}秒"
    else:
        return f"{secs}秒"




@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with streamingAC agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # NOTE: 删除了多GPU训练配置，StreamingRunner只支持单GPU训练
    # 确保设备配置正确
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    else:
        agent_cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl_streaming", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging streamingAC experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during streamingAC training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 添加关节损伤包装器（如果启用）
    if args_cli.enable_joint_damage:
        damage_config = create_damage_config(args_cli)
        print_damage_config_info(damage_config)
        env = JointDamageWrapper(env, damage_config)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 验证配置是否适用于StreamingRunner
    if not hasattr(agent_cfg, 'algorithm') or agent_cfg.algorithm.class_name != "streamingAC":
        raise ValueError(
            f"StreamingRunner requires streamingAC algorithm, got {getattr(agent_cfg.algorithm, 'class_name', 'unknown')}"
        )

    # create StreamingRunner from rsl-rl
    print(f"[INFO] Creating StreamingRunner with device: {agent_cfg.device}")
    runner = StreamingRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading streamingAC model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # print training information
    print("\n" + "="*80)
    print(f"🚀 Starting streamingAC training")
    print(f"📁 Log directory: {log_dir}")
    print(f"🏃 Algorithm: {agent_cfg.algorithm.class_name}")
    print(f"🧠 Policy: {agent_cfg.policy.class_name}")
    print(f"🔧 Device: {agent_cfg.device}")
    print(f"🌍 Environments: {env_cfg.scene.num_envs}")
    print(f"📈 Max iterations: {agent_cfg.max_iterations}")
    print(f"💾 Save interval: {agent_cfg.save_interval}")
    print(f"🎯 Steps per env: {agent_cfg.num_steps_per_env}")
    print("="*80 + "\n")

    # 在开始训练前加载预训练模型进行微调
    if args_cli.pretrained_checkpoint:
        print(f"🔄 Loading pretrained model from: {args_cli.pretrained_checkpoint}")
        print(f"🎯 Finetune mode: {args_cli.finetune_mode}")
        
        # 获取StreamingAC算法实例
        alg = runner.alg
        alg.load_pretrained_policy(
            checkpoint_path=args_cli.pretrained_checkpoint,
            finetune_mode=args_cli.finetune_mode,
            reset_optimizer=True
        )
        # 微调时使用较小的学习率
        original_lr = alg.lr
        alg.adjust_learning_rate(alg.lr * 0.1)
        print(f"📉 Learning rate adjusted from {original_lr} to {alg.lr} for finetuning")

    # 记录训练开始时间
    training_start_time = time.time()
    print(f"⏰ Training started at: {datetime.fromtimestamp(training_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # run streaming training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    
    # 记录训练结束时间并计算总时间
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    print("\n" + "="*80)
    print("✅ StreamingAC training completed!")
    print(f"⏰ Total training time: {format_time(total_training_time)}")
    print(f"📁 Results saved in: {log_dir}")
    
    # 打印关节损伤统计信息（如果启用）
    if args_cli.enable_joint_damage:
        damage_wrapper = find_damage_wrapper(env)
        if damage_wrapper:
            print_damage_statistics(damage_wrapper, args_cli.output_damage_info)
    
    print("="*80 + "\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
