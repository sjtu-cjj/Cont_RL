# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.algorithms import streamingAC
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
)
from rsl_rl.utils import store_code_state


class StreamingRunner:
    """Streaming runner for online learning algorithms like streamingAC."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env



        # resolve training type - streaming algorithms
        if self.alg_cfg["class_name"] == "streamingAC":
            self.training_type = "streaming"
        else:
            raise ValueError(f"StreamingRunner only supports streaming algorithms, got {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations for streaming AC
        if "critic" in extras["observations"]:
            self.privileged_obs_type = "critic"  # streaming actor-critic
        else:
            self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCritic | ActorCriticRecurrent = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: streamingAC = alg_class(policy, device=self.device, **self.alg_cfg)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # Streaming algorithms don't need storage - they learn online
        # No need to call init_storage()

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Wandb summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

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

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)



        # Start streaming training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # Streaming rollout and update - learn online
            for _ in range(self.num_steps_per_env):
                # Store previous observations for update_params
                prev_obs = obs.clone()
                prev_privileged_obs = privileged_obs.clone()
                
                # Sample actions using streamingAC's sample_action method
                if hasattr(self.alg, 'sample_action'):
                    actions = self.alg.sample_action(obs)
                else:
                    # Fallback to act method if sample_action not available
                    actions = self.alg.act(obs, privileged_obs)
                
                # Step the environment
                obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                # Move to device
                obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                
                # perform normalization
                obs = self.obs_normalizer(obs)
                if self.privileged_obs_type is not None:
                    privileged_obs = self.privileged_obs_normalizer(
                        infos["observations"][self.privileged_obs_type].to(self.device)
                    )
                else:
                    privileged_obs = obs

                # Update parameters for each environment step (streaming learning)
                if hasattr(self.alg, 'update_params'):
                    for env_idx in range(self.env.num_envs):
                        # Extract single environment data
                        s = prev_obs[env_idx]
                        a = actions[env_idx]
                        r = rewards[env_idx]
                        s_prime = obs[env_idx]
                        done = dones[env_idx].bool().item()
                        
                        # Update parameters using streamingAC's update_params
                        loss_info = self.alg.update_params(s, a, r, s_prime, done)
                
                # Book keeping
                if self.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    # Update rewards
                    cur_reward_sum += rewards
                    # Update episode length
                    cur_episode_length += 1
                    # Clear data for completed episodes
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            learn_time = 0.0  # Learning happens during collection for streaming
            
            # Create loss dict for logging compatibility
            loss_dict = {
                "value_function": 0.0,
                "surrogate": 0.0, 
                "entropy": 0.0,
            }
            
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        # Get action std for logging
        if hasattr(self.alg.policy, 'action_std'):
            mean_std = self.alg.policy.action_std.mean()
        else:
            # For streamingAC or other algorithms without action_std
            mean_std = torch.tensor(0.0)
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        
        # Log learning rate if available
        if hasattr(self.alg, 'learning_rate'):
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        elif hasattr(self.alg, 'lr'):
            self.writer.add_scalar("Loss/learning_rate", self.alg.lr, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
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
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        
        # -- Save optimizer(s) for streamingAC (dual optimizers)
        if hasattr(self.alg, 'optimizer_policy') and hasattr(self.alg, 'optimizer_value'):
            saved_dict["optimizer_policy_state_dict"] = self.alg.optimizer_policy.state_dict()
            saved_dict["optimizer_value_state_dict"] = self.alg.optimizer_value.state_dict()
        elif hasattr(self.alg, 'optimizer'):
            # Fallback for single optimizer
            saved_dict["optimizer_state_dict"] = self.alg.optimizer.state_dict()
            
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # streamingAC has separate policy and value optimizers
            if "optimizer_policy_state_dict" in loaded_dict and "optimizer_value_state_dict" in loaded_dict:
                if hasattr(self.alg, 'optimizer_policy') and hasattr(self.alg, 'optimizer_value'):
                    self.alg.optimizer_policy.load_state_dict(loaded_dict["optimizer_policy_state_dict"])
                    self.alg.optimizer_value.load_state_dict(loaded_dict["optimizer_value_state_dict"])
            elif "optimizer_state_dict" in loaded_dict:
                # Fallback for single optimizer
                if hasattr(self.alg, 'optimizer'):
                    self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                    
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- streamingAC
        self.alg.policy.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- streamingAC
        self.alg.policy.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """


"""
使用示例:

```python
from rsl_rl.runners import StreamingRunner
from rsl_rl.env import VecEnv

# 设置训练配置
train_cfg = {
    "algorithm": {
        "class_name": "streamingAC",
        "lr": 1e-3,
        "kappa_policy": 3.0,
        "kappa_value": 2.0,
        # 其他streamingAC参数...
    },
    "policy": {
        "class_name": "ActorCriticStreaming",
        "init_noise_std": 1.0,
        # 其他策略参数...
    },
    "num_steps_per_env": 24,
    "save_interval": 50,
    "empirical_normalization": False,
}

# 创建环境和runner
env = VecEnv(...)  # 你的环境
runner = StreamingRunner(env, train_cfg, log_dir="./logs", device="cuda:0")

# 开始流式学习
runner.learn(num_learning_iterations=1000)

# 保存模型
runner.save("./final_model.pt")

# 获取推理策略
policy = runner.get_inference_policy()
```

注意事项:
1. StreamingRunner专门为在线学习设计，不需要存储rollout数据
2. 参数在每个环境步骤后立即更新
3. 支持双优化器结构（policy和value分离）
4. 支持观测标准化
"""
