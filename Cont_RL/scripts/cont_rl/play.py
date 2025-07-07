# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of StreamingAC agent from RSL-RL."""

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

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained StreamingAC agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# 关节损伤相关参数
parser.add_argument("--enable_joint_damage", action="store_true", default=False, help="Enable joint damage simulation during play.")
parser.add_argument("--damage_probability", type=float, default=0.1, help="Probability of joint damage per step.")
parser.add_argument("--max_damaged_joints", type=int, default=2, help="Maximum number of joints that can be damaged simultaneously.")
parser.add_argument("--damage_type", type=str, default="zero", choices=["zero", "partial", "random"], 
                   help="Type of joint damage: zero (set to 0), partial (reduce by factor), random (random values).")
parser.add_argument("--damage_severity", type=float, default=1.0, help="Severity of damage (0-1, for partial damage type).")
parser.add_argument("--output_damage_info", action="store_true", default=True, 
                   help="Enable detailed damage information output during play.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from typing import Dict

# 导入StreamingRunner而不是OnPolicyRunner
from rsl_rl.runners import StreamingRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import Cont_RL.tasks  # noqa: F401

# 导入关节损伤模块
from joint_damage import (
    JointDamageWrapper,
    create_damage_config,
    print_damage_config_info
)


class PlayJointDamageSimulator:
    """
    用于play场景的关节损伤模拟器
    复用joint_damage.py中的核心逻辑，但不需要环境包装器
    """
    
    def __init__(self, damage_config: Dict, num_envs: int, num_actions: int):
        # 创建一个虚拟的环境对象来初始化JointDamageWrapper
        class MockEnv:
            def __init__(self, num_envs, num_actions):
                self.num_envs = num_envs
                self.action_space = type('ActionSpace', (), {'shape': (num_envs, num_actions)})()
        
        mock_env = MockEnv(num_envs, num_actions)
        
        # 使用原有的JointDamageWrapper，但只使用其损伤逻辑部分
        self.damage_wrapper = JointDamageWrapper(mock_env, damage_config)
        
        print(f"🎮 PlayJointDamageSimulator: Reusing logic from JointDamageWrapper")
        print(f"   📊 Environments: {num_envs}, Actions: {num_actions}")
    
    def apply_joint_damage(self, actions: torch.Tensor) -> torch.Tensor:
        """
        应用关节损伤到动作
        直接调用JointDamageWrapper的apply_joint_damage方法
        """
        return self.damage_wrapper.apply_joint_damage(actions)
    
    def get_damage_statistics(self) -> Dict:
        """获取损伤统计信息"""
        return self.damage_wrapper.get_damage_statistics()
    
    def get_damage_report(self) -> str:
        """获取详细的损伤报告"""
        return self.damage_wrapper.get_damage_report()


def main():
    """Play with trained agent (StreamingAC or PPO)."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 根据任务和算法选择正确的日志目录
    if hasattr(agent_cfg, 'algorithm') and agent_cfg.algorithm.class_name == "streamingAC":
        log_root_path = os.path.join("logs", "rsl_rl_streaming", agent_cfg.experiment_name)
        print(f"[INFO] Using StreamingAC logs from: {log_root_path}")
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        print(f"[INFO] Using PPO logs from: {log_root_path}")
    
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # 根据算法类型选择合适的runner
    if hasattr(agent_cfg, 'algorithm') and agent_cfg.algorithm.class_name == "streamingAC":
        print("[INFO] Using StreamingRunner for StreamingAC algorithm")
        runner = StreamingRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        print("[INFO] Using OnPolicyRunner for PPO algorithm")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # load previously trained model
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    
    # 初始化关节损伤模拟器（如果启用）
    damage_simulator = None
    if args_cli.enable_joint_damage:
        damage_config = create_damage_config(args_cli)
        print_damage_config_info(damage_config)
        
        # 获取动作空间信息
        num_envs = env.num_envs
        num_actions = env.action_space.shape[1]
        
        damage_simulator = PlayJointDamageSimulator(damage_config, num_envs, num_actions)
    
    timestep = 0
    total_reward = 0.0
    episode_count = 0
    
    print("\n" + "="*60)
    print("🎮 Starting trained agent play session")
    print(f"🏃 Algorithm: {getattr(agent_cfg.algorithm, 'class_name', 'PPO')}")
    print(f"🌍 Environments: {env.num_envs}")
    print(f"🎯 Task: {args_cli.task}")
    if args_cli.enable_joint_damage:
        print("⚠️  Joint damage simulation: ENABLED")
    print("="*60 + "\n")
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            
            # 应用关节损伤（如果启用）
            if damage_simulator is not None:
                actions = damage_simulator.apply_joint_damage(actions)
            
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # 打印关节损伤统计信息（如果启用）
    if damage_simulator is not None:
        damage_stats = damage_simulator.get_damage_statistics()
        print("\n" + "="*60)
        print("🤖 JOINT DAMAGE STATISTICS")
        print("="*60)
        print(f"📊 Total damage events: {damage_stats['total_damage_events']}")
        print(f"🎯 Current damaged joints: {damage_stats['current_damaged_joints']}")
        print(f"📈 Damage rate: {damage_stats['damage_rate']:.6f} events/step")
        print(f"⏱️  Total steps: {damage_stats['step_count']}")
        print("="*60)
        
        # 输出详细损伤报告
        if args_cli.output_damage_info:
            print("\n" + damage_simulator.get_damage_report())
            print("="*60)

    print("\n" + "="*60)
    print("🏁 Play session completed!")
    if episode_count > 0:
        print(f"📊 Total episodes: {episode_count}")
        print(f"📈 Final average reward: {total_reward / episode_count:.2f}")
    print("="*60 + "\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
