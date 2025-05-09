# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time
import torch
import numpy as np
import random
import datetime
# 删除pynput导入和相关检查

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
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

# 添加关节屏蔽相关参数
parser.add_argument("--mask_joint_type", type=str, default=None, choices=["hip", "thigh", "calf"], 
                    help="指定要屏蔽的关节类型: hip, thigh 或 calf")
parser.add_argument("--mask_joint_count", type=int, default=1, 
                    help="指定每次要屏蔽的关节数量")
                    
# 添加腿部失效相关参数
parser.add_argument("--leg_type", type=str, default=None, choices=["FL", "FR", "RL", "RR"], 
                    help="指定要失效的腿: 前左(FL), 前右(FR), 后左(RL), 后右(RR)")
                    
# 添加故障模式参数
parser.add_argument("--failure_mode", type=str, default="zero", 
                    choices=["zero", "stuck", "leg_zero", "leg_stuck"], 
                    help="指定关节失效模式: zero(零力矩), stuck(卡死), leg_zero(整条腿零力矩), leg_stuck(整条腿卡死)")

# 添加仿真时间控制参数
parser.add_argument("--max_sim_time", type=float, default=20.0, 
                    help="最大仿真时间（秒），默认为20秒")

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
import numpy as np
import random
# 删除pynput导入和相关检查

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import Cont_RL.tasks  # noqa: F401


# 定义关节掩膜函数
def apply_joint_mask(actions, joint_type=None, mask_count=1, failure_mode="zero", stuck_positions=None, leg_type=None):
    """
    对指定类型的关节应用掩膜
    
    Args:
        actions: 模型输出的动作，形状为[n_envs, 12]
        joint_type: 要屏蔽的关节类型，"hip", "thigh" 或 "calf"
        mask_count: 要屏蔽的关节数量
        failure_mode: 失效模式，"zero"(零力矩), "stuck"(卡死), "leg_zero"(整条腿零力矩), "leg_stuck"(整条腿卡死)
        stuck_positions: 卡死模式下的关节位置，如果为None则使用当前位置卡死
        leg_type: 要失效的腿，"FL", "FR", "RL" 或 "RR"
        
    Returns:
        应用掩膜后的动作，如果是卡死模式，还会返回更新后的stuck_positions
    """
    # 四足机器人有12个关节，每条腿3个关节(hip, thigh, calf)
    # 关节索引: 
    # 腿的顺序: FL (0-2), FR (3-5), RL (6-8), RR (9-11)
    # 每条腿的关节顺序: hip, thigh, calf
    
    # 腿部关节索引映射
    leg_indices = {
        "FL": [0, 1, 2],    # 前左腿: hip, thigh, calf
        "FR": [3, 4, 5],    # 前右腿: hip, thigh, calf
        "RL": [6, 7, 8],    # 后左腿: hip, thigh, calf
        "RR": [9, 10, 11]   # 后右腿: hip, thigh, calf
    }
    
    # 如果是整条腿失效模式，需要leg_type参数
    if failure_mode in ["leg_zero", "leg_stuck"] and leg_type is None:
        raise ValueError(f"在'{failure_mode}'模式下，必须指定--leg_type参数（FL、FR、RL或RR）")
    
    # 关节类型失效模式，需检查joint_type有效性
    if failure_mode in ["zero", "stuck"] and joint_type is None:
        return actions, stuck_positions
    
    # 为每个环境创建掩膜
    n_envs = actions.shape[0]
    
    # 关节类型索引设置
    hip_indices = [0, 3, 6, 9]    # 所有hip关节的索引
    thigh_indices = [1, 4, 7, 10]  # 所有thigh关节的索引
    calf_indices = [2, 5, 8, 11]   # 所有calf关节的索引
    
    # 根据指定类型选择对应的索引列表
    joint_indices = {
        "hip": hip_indices,
        "thigh": thigh_indices,
        "calf": calf_indices
    }.get(joint_type, [])
    
    # 如果是关节类型失效模式，检查joint_type有效性
    if failure_mode in ["zero", "stuck"] and not joint_indices:
        raise ValueError(f"未找到匹配的关节类型: {joint_type}。请指定有效的关节类型: 'hip', 'thigh' 或 'calf'")
    
    # 为卡死模式初始化stuck_positions
    if failure_mode in ["stuck", "leg_stuck"] and stuck_positions is None:
        # 初始化stuck_positions为一个与actions相同形状的张量，但全部填充为None
        stuck_positions = np.full(actions.shape, None, dtype=object)
    
    # 根据失效模式选择要失效的关节索引
    target_indices = []
    
    if failure_mode in ["zero", "stuck"]:
        # 关节类型失效模式：随机选择指定数量的关节
        for env_idx in range(n_envs):
            # 随机选择指定数量的关节进行屏蔽
            mask_indices = random.sample(joint_indices, min(mask_count, len(joint_indices)))
            
            # 应用掩膜
            for idx in mask_indices:
                if failure_mode == "zero":
                    # 零力矩模式：将选定的关节动作设为0
                    actions[env_idx, idx] = 0.0
                elif failure_mode == "stuck":
                    # 卡死模式：如果关节位置尚未初始化，则使用当前动作值作为卡死位置
                    if stuck_positions[env_idx, idx] is None:
                        stuck_positions[env_idx, idx] = actions[env_idx, idx]
                    # 使用卡死位置覆盖当前动作
                    actions[env_idx, idx] = stuck_positions[env_idx, idx]
    
    elif failure_mode in ["leg_zero", "leg_stuck"]:
        # 整条腿失效模式：使用指定腿的所有关节
        leg_joints = leg_indices.get(leg_type, [])
        if not leg_joints:
            raise ValueError(f"未找到匹配的腿类型: {leg_type}。请指定有效的腿类型: 'FL', 'FR', 'RL' 或 'RR'")
        
        # 对每个环境应用整条腿的掩膜
        for env_idx in range(n_envs):
            for idx in leg_joints:
                if failure_mode == "leg_zero":
                    # 整条腿零力矩模式：将选定腿的所有关节动作设为0
                    actions[env_idx, idx] = 0.0
                elif failure_mode == "leg_stuck":
                    # 整条腿卡死模式：如果关节位置尚未初始化，则使用当前动作值作为卡死位置
                    if stuck_positions[env_idx, idx] is None:
                        stuck_positions[env_idx, idx] = actions[env_idx, idx]
                    # 使用卡死位置覆盖当前动作
                    actions[env_idx, idx] = stuck_positions[env_idx, idx]
              
    return actions, stuck_positions


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
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
    
    # 删除环境实例存储相关代码

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # 创建带有日期和时间的目录
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        
        # 创建三级目录结构
        level_1 = date_str  # 年月日
        
        # 第二级目录：根据失效模式和类型参数决定
        level_2 = "no_disable"  # 默认值，未设置关节屏蔽
        if args_cli.failure_mode in ["zero", "stuck"] and args_cli.mask_joint_type is not None:
            level_2 = f"{args_cli.mask_joint_type}_{args_cli.failure_mode}"
        elif args_cli.failure_mode in ["leg_zero", "leg_stuck"] and args_cli.leg_type is not None:
            level_2 = f"{args_cli.leg_type}_{args_cli.failure_mode}"
        
        # 第三级目录：根据失效细节参数决定
        level_3 = "no_joint"  # 默认值
        if args_cli.failure_mode in ["zero", "stuck"] and args_cli.mask_joint_type is not None:
            level_3 = f"{args_cli.mask_joint_type}_{args_cli.failure_mode}_{args_cli.mask_joint_count}_joint"
        elif args_cli.failure_mode in ["leg_zero", "leg_stuck"] and args_cli.leg_type is not None:
            level_3 = f"{args_cli.leg_type}_{args_cli.failure_mode}"
        
        # 设置合适的视频长度，确保约20秒的视频
        # 根据仿真的dt计算合适的步数，这里假设模拟每秒运行约20步
        # 20秒 * 20步/秒 = 400步
        video_length = 400
        
        # 确保目录存在
        video_dir = os.path.join("Cont_RL", "outputs", level_1, level_2, level_3)
        os.makedirs(video_dir, exist_ok=True)
        
        # 使用与第三级文件夹相同的名称作为视频名称前缀
        video_name_prefix = level_3
        
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
            "name_prefix": video_name_prefix,  # 设置视频名称前缀
            # 移除不支持的参数
        }
        print(f"[INFO] Recording videos to: {video_dir}")
        print(f"[INFO] 视频设置为约20秒 ({video_length}步)")
        print(f"[INFO] 视频名称前缀: {video_name_prefix}")
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    
    # 记录开始时间
    start_simulation_time = time.time()
    
    # 最大仿真时间（秒）
    max_sim_time = args_cli.max_sim_time
    
    # 用于存储卡死模式下的关节位置
    stuck_positions = None
    
    # 模拟环境
    while simulation_app.is_running():
        current_time = time.time()
        elapsed_time = current_time - start_simulation_time
        
        # 检查是否已超过最大仿真时间
        if elapsed_time > max_sim_time:
            print(f"[INFO] 达到最大仿真时间 {max_sim_time} 秒，停止仿真")
            break
            
        start_step_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # 使用智能体策略
            actions = policy(obs)
            
            # 应用关节掩膜
            actions, stuck_positions = apply_joint_mask(
                actions, 
                args_cli.mask_joint_type, 
                args_cli.mask_joint_count, 
                args_cli.failure_mode,
                stuck_positions,
                args_cli.leg_type
            )
            
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # 使用与视频录制相同的视频长度
            if timestep >= 400:  # 与上面设置的video_length一致
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_step_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    
    # 计算总仿真时间
    total_simulation_time = time.time() - start_simulation_time
    print(f"[INFO] Total simulation time: {total_simulation_time:.2f} seconds")
    
    # 删除停止监听器代码
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
