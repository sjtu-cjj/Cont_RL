# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
OnPolicyRunner - 在策略强化学习训练器

这个模块实现了用于在策略强化学习算法（如PPO）的完整训练管道。
主要功能包括：
- 支持 PPO 和策略蒸馏训练
- 多GPU分布式训练支持
- 完整的实验管理（日志记录、模型保存、检查点恢复）
- 观测归一化和特权观测处理
- 多种日志系统集成（TensorBoard、Neptune、WandB）
- RND（随机网络蒸馏）内在动机支持
"""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state


class OnPolicyRunner:
    """
    在策略强化学习训练器和评估器
    
    这是RSL-RL库的核心训练组件，支持：
    1. PPO算法训练 - 主流的策略梯度算法
    2. 策略蒸馏训练 - 将教师策略知识转移到学生策略
    3. 多GPU分布式训练 - 大规模并行训练
    4. 完整的实验管理 - 日志、保存、恢复等
    5. 观测处理 - 归一化、特权观测等
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        """
        初始化OnPolicyRunner
        
        Args:
            env: 向量化环境，支持并行仿真
            train_cfg: 训练配置字典，包含算法、策略等配置
            log_dir: 日志保存目录，None则不保存日志
            device: 训练设备（cuda或cpu）
        """
        # =============================================================================
        # 基础配置设置
        # =============================================================================
        self.cfg = train_cfg                      # 完整训练配置
        self.alg_cfg = train_cfg["algorithm"]     # 算法特定配置（如PPO参数）
        self.policy_cfg = train_cfg["policy"]     # 策略网络配置（如隐藏层大小）
        self.device = device                      # 训练设备
        self.env = env                           # 向量化环境实例

        # =============================================================================
        # 多GPU训练配置
        # =============================================================================
        # 检查并配置多GPU分布式训练环境
        self._configure_multi_gpu()

        # =============================================================================
        # 训练类型解析
        # =============================================================================
        # 根据算法类型确定训练模式
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"             # 标准强化学习训练
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"   # 策略蒸馏训练
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # =============================================================================
        # 观测空间解析
        # =============================================================================
        # 从环境获取观测信息，解析观测维度
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]  # 标准观测维度

        # =============================================================================
        # 特权观测类型解析
        # =============================================================================
        # 特权观测是指在训练时可用但部署时不一定可用的额外信息
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                # Critic特权观测：价值函数可以访问更多环境状态信息
                self.privileged_obs_type = "critic"
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                # Teacher特权观测：教师策略的额外观测信息
                self.privileged_obs_type = "teacher"
            else:
                self.privileged_obs_type = None

        # =============================================================================
        # 特权观测维度解析
        # =============================================================================
        # 根据特权观测类型确定其维度
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            # 如果没有特权观测，使用标准观测维度
            num_privileged_obs = num_obs

        # =============================================================================
        # 策略网络创建
        # =============================================================================
        # 动态创建策略网络类（支持多种架构）
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent | StudentTeacher | StudentTeacherRecurrent = policy_class(
            num_obs,                # 标准观测维度
            num_privileged_obs,     # 特权观测维度  
            self.env.num_actions,   # 动作空间维度
            **self.policy_cfg       # 其他策略配置参数
        ).to(self.device)

        # =============================================================================
        # RND（随机网络蒸馏）配置
        # =============================================================================
        # RND用于提供内在动机，鼓励探索新颖状态
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # 检查RND门控状态是否存在
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            
            # 获取RND门控状态维度
            num_rnd_state = rnd_state.shape[1]
            # 添加RND门控状态到配置
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # 根据时间步长缩放RND权重（类似于legged_gym环境中的奖励缩放）
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # =============================================================================
        # 对称性处理配置
        # =============================================================================
        # 如果使用对称性，传递环境配置对象用于处理不同观测项
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # =============================================================================
        # 算法初始化
        # =============================================================================
        # 动态创建算法实例（PPO或Distillation）
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy,                           # 策略网络
            device=self.device,               # 训练设备
            multi_gpu_cfg=self.multi_gpu_cfg, # 多GPU配置
            **self.alg_cfg                    # 算法特定参数
        )

        # =============================================================================
        # 训练参数存储
        # =============================================================================
        self.num_steps_per_env = self.cfg["num_steps_per_env"]     # 每个环境的步数
        self.save_interval = self.cfg["save_interval"]             # 模型保存间隔
        self.empirical_normalization = self.cfg["empirical_normalization"]  # 是否使用经验归一化

        # =============================================================================
        # 观测归一化器设置
        # =============================================================================
        if self.empirical_normalization:
            # 为标准观测和特权观测分别创建归一化器
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            # 不使用归一化，创建恒等变换
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # =============================================================================
        # 算法存储和模型初始化
        # =============================================================================
        # 初始化算法的经验存储缓冲区
        self.alg.init_storage(
            self.training_type,                # 训练类型（rl或distillation）
            self.env.num_envs,                # 环境数量
            self.num_steps_per_env,           # 每个环境的步数
            [num_obs],                        # 观测形状
            [num_privileged_obs],             # 特权观测形状
            [self.env.num_actions],           # 动作形状
        )

        # =============================================================================
        # 日志记录配置
        # =============================================================================
        # 决定是否禁用日志记录
        # 只有rank 0进程（主进程）记录日志
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        
        # 日志记录相关变量初始化
        self.log_dir = log_dir                      # 日志目录
        self.writer = None                          # 日志写入器（稍后初始化）
        self.tot_timesteps = 0                     # 总时间步数
        self.tot_time = 0                          # 总训练时间
        self.current_learning_iteration = 0        # 当前学习迭代次数
        self.git_status_repos = [rsl_rl.__file__]  # Git状态跟踪的仓库列表

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        """
        执行强化学习训练的主循环
        
        Args:
            num_learning_iterations: 学习迭代次数
            init_at_random_ep_len: 是否在随机episode长度处开始训练（提高探索多样性）
        """
        # =============================================================================
        # 日志写入器初始化
        # =============================================================================
        # 只在主进程且有日志目录时初始化写入器
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # 获取日志记录器类型，默认为tensorboard
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            # 根据配置创建相应的日志写入器
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # =============================================================================
        # 训练前检查
        # =============================================================================
        # 检查策略蒸馏训练时是否已加载教师模型
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # =============================================================================
        # 随机化初始episode长度
        # =============================================================================
        # 随机化初始episode长度有助于提高探索的多样性
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # =============================================================================
        # 训练循环准备
        # =============================================================================
        # 获取初始观测
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # 切换到训练模式（如dropout等）

        # =============================================================================
        # 训练统计记录初始化
        # =============================================================================
        # Episode信息记录
        ep_infos = []
        # 使用双端队列记录最近100个episode的统计信息
        rewbuffer = deque(maxlen=100)    # 奖励缓冲区
        lenbuffer = deque(maxlen=100)    # episode长度缓冲区
        # 当前episode的累积统计
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # =============================================================================
        # RND内在奖励统计初始化
        # =============================================================================
        # 如果使用RND，创建额外的缓冲区记录内在和外在奖励
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)  # 外在奖励缓冲区
            irewbuffer = deque(maxlen=100)  # 内在奖励缓冲区
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # =============================================================================
        # 分布式训练参数同步
        # =============================================================================
        # 确保所有进程的参数保持同步
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: 是否需要同步经验归一化器？
            # 目前：否，因为它们应该"渐近地"收敛到相同的值

        # =============================================================================
        # 主训练循环
        # =============================================================================
        # 开始训练循环
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # =========================================================================
            # Rollout阶段 - 数据收集
            # =========================================================================
            # 在推理模式下收集训练数据
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 使用当前策略采样动作
                    actions = self.alg.act(obs, privileged_obs)
                    
                    # 在环境中执行动作
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    
                    # 将数据移动到训练设备
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    
                    # 执行观测归一化
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # 处理环境步骤（存储到经验缓冲区）
                    self.alg.process_env_step(rewards, dones, infos)

                    # 提取内在奖励（仅用于日志记录）
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # =============================================================
                    # 训练统计记录
                    # =============================================================
                    if self.log_dir is not None:
                        # 记录episode信息
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                            
                        # 更新奖励统计
                        if self.alg.rnd:
                            # 分别记录外在和内在奖励
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                            
                        # 更新episode长度
                        cur_episode_length += 1
                        
                        # 清理已完成episode的数据
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # 通用统计
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        
                        # 内在和外在奖励统计
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                # 记录数据收集时间
                stop = time.time()
                collection_time = stop - start
                start = stop

                # =====================================================================
                # 回报计算阶段
                # =====================================================================
                # 计算累积回报（仅在强化学习训练时）
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # =========================================================================
            # 策略更新阶段
            # =========================================================================
            # 使用收集的数据更新策略
            loss_dict = self.alg.update()

            # 记录学习时间
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # =========================================================================
            # 日志记录和模型保存
            # =========================================================================
            if self.log_dir is not None and not self.disable_logs:
                # 记录训练信息
                self.log(locals())
                # 定期保存模型
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清理episode信息
            ep_infos.clear()
            
            # =========================================================================
            # Git状态保存
            # =========================================================================
            # 在第一次迭代时保存代码状态
            if it == start_iter and not self.disable_logs:
                # 获取所有diff文件
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # 如果可能，将它们存储到wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # =============================================================================
        # 训练完成后保存最终模型
        # =============================================================================
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """
        记录训练信息到日志系统和控制台
        
        Args:
            locs: 包含训练统计信息的本地变量字典
            width: 控制台输出宽度
            pad: 文本对齐填充长度
        """
        # =============================================================================
        # 计算训练统计信息
        # =============================================================================
        # 计算收集数据大小（考虑多GPU）
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        
        # 更新总时间步数和训练时间
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # =============================================================================
        # Episode信息记录
        # =============================================================================
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # 处理标量和零维张量信息
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # 记录到日志器和终端
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # =============================================================================
        # 策略和性能指标记录
        # =============================================================================
        # 动作标准差统计
        mean_std = self.alg.policy.action_std.mean()
        # 计算FPS（每秒帧数）
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # 损失函数记录
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # 策略相关指标
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # 性能指标
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # =============================================================================
        # 训练奖励统计记录
        # =============================================================================
        if len(locs["rewbuffer"]) > 0:
            # RND内在和外在奖励分别记录
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            
            # 总体训练统计
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            
            # 时间序列记录（wandb不支持非整数x轴）
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        # =============================================================================
        # 控制台输出格式化
        # =============================================================================
        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            # 有奖励数据时的完整日志
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # 损失函数信息
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # 奖励信息
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # Episode信息
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            # 无奖励数据时的简化日志
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        # 添加episode自定义信息
        log_string += ep_string
        
        # 添加训练总结信息
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                               locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            infos: 额外信息（可选）
        """
        # =============================================================================
        # 构建保存字典
        # =============================================================================
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),        # 策略网络参数
            "optimizer_state_dict": self.alg.optimizer.state_dict(), # 优化器状态
            "iter": self.current_learning_iteration,                 # 当前迭代次数
            "infos": infos,                                          # 额外信息
        }
        
        # =============================================================================
        # 保存RND模型（如果使用）
        # =============================================================================
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
            
        # =============================================================================
        # 保存观测归一化器（如果使用）
        # =============================================================================
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # 保存模型到本地
        torch.save(saved_dict, path)

        # =============================================================================
        # 上传模型到外部日志服务
        # =============================================================================
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        """
        加载模型检查点
        
        Args:
            path: 检查点文件路径
            load_optimizer: 是否加载优化器状态
            
        Returns:
            loaded_dict["infos"]: 检查点中保存的额外信息
        """
        # =============================================================================
        # 加载检查点文件
        # =============================================================================
        loaded_dict = torch.load(path, weights_only=False)
        
        # =============================================================================
        # 加载策略网络参数
        # =============================================================================
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        
        # =============================================================================
        # 加载RND模型（如果使用）
        # =============================================================================
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            
        # =============================================================================
        # 加载观测归一化器（如果使用）
        # =============================================================================
        if self.empirical_normalization:
            if resumed_training:
                # 如果是续训，actor/student归一化器加载用于actor/student
                # critic/teacher归一化器加载用于critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # 如果不是续训但加载了模型，这次运行必须是跟随强化学习训练的蒸馏训练
                # 因此actor归一化器被加载用于teacher模型。student的归一化器不被加载，
                # 因为观测空间可能与之前的强化学习训练不同
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                
        # =============================================================================
        # 加载优化器状态（如果需要）
        # =============================================================================
        if load_optimizer and resumed_training:
            # 算法优化器
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # RND优化器（如果使用）
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
                
        # =============================================================================
        # 加载当前学习迭代次数
        # =============================================================================
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        """
        获取用于推理的策略函数
        
        Args:
            device: 推理设备（可选）
            
        Returns:
            policy: 推理策略函数
        """
        # 切换到评估模式（禁用dropout等）
        self.eval_mode()
        
        # 如果指定了设备，将策略移动到该设备
        if device is not None:
            self.alg.policy.to(device)
            
        # 获取基础推理策略
        policy = self.alg.policy.act_inference
        
        # 如果使用经验归一化，包装归一化处理
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
            
        return policy

    def train_mode(self):
        """切换所有组件到训练模式"""
        # PPO策略网络
        self.alg.policy.train()
        # RND网络（如果使用）
        if self.alg.rnd:
            self.alg.rnd.train()
        # 归一化层（如果使用）
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        """切换所有组件到评估模式"""
        # PPO策略网络
        self.alg.policy.eval()
        # RND网络（如果使用）
        if self.alg.rnd:
            self.alg.rnd.eval()
        # 归一化层（如果使用）
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        """
        添加Git仓库到日志跟踪列表
        
        Args:
            repo_file_path: 仓库文件路径
        """
        self.git_status_repos.append(repo_file_path)

    """
    辅助函数
    """

    def _configure_multi_gpu(self):
        """
        配置多GPU分布式训练环境
        
        这个方法检测分布式训练环境变量，设置相应的rank和world_size，
        并初始化PyTorch分布式进程组用于多GPU通信。
        """
        # =============================================================================
        # 检测分布式训练环境
        # =============================================================================
        # 获取总进程数（等于GPU数量）
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        # 判断是否启用分布式训练
        self.is_distributed = self.gpu_world_size > 1

        # =============================================================================
        # 单GPU训练配置
        # =============================================================================
        # 如果不是分布式训练，设置local和global rank为0并返回
        if not self.is_distributed:
            self.gpu_local_rank = 0      # 本地进程rank
            self.gpu_global_rank = 0     # 全局进程rank
            self.multi_gpu_cfg = None    # 多GPU配置为None
            return

        # =============================================================================
        # 多GPU分布式训练配置
        # =============================================================================
        # 获取rank和world size环境变量
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))   # 当前节点内的GPU编号
        self.gpu_global_rank = int(os.getenv("RANK", "0"))        # 全局进程编号

        # =============================================================================
        # 创建多GPU配置字典
        # =============================================================================
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # 主进程的rank
            "local_rank": self.gpu_local_rank,    # 当前进程的rank
            "world_size": self.gpu_world_size,    # 总进程数
        }

        # =============================================================================
        # 验证设备配置
        # =============================================================================
        # 检查用户指定的设备是否与local rank匹配
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
            
        # =============================================================================
        # 验证多GPU配置参数
        # =============================================================================
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # =============================================================================
        # 初始化PyTorch分布式进程组
        # =============================================================================
        # 使用NCCL后端进行GPU间通信
        torch.distributed.init_process_group(
            backend="nccl",                    # NVIDIA GPU通信库
            rank=self.gpu_global_rank,         # 当前进程的全局rank
            world_size=self.gpu_world_size     # 总进程数
        )
        # 设置当前进程使用的GPU设备
        torch.cuda.set_device(self.gpu_local_rank)
