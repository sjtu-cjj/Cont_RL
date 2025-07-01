# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
import os

from rsl_rl.modules import ActorCriticStreaming


class ObGD(torch.optim.Optimizer):
    """Online Batch Gradient Descent优化器，来自streaming-drl项目"""
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)
    
    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()


class streamingAC:
    """Streaming AC算法，基于stream_ac_continuous.py的StreamAC类实现"""

    policy: ActorCriticStreaming
    """The actor critic module."""

    def __init__(
        self,
        policy,
        # StreamingAC特有参数
        lr=1.0,
        gamma=0.99,
        lamda=0.8,
        kappa_policy=3.0,
        kappa_value=2.0,
        entropy_coef=0.01,
        device="cpu",
    ):
        # 设备参数
        self.device = device

        # StreamingAC核心参数
        self.gamma = gamma
        self.lamda = lamda
        self.lr = lr
        self.kappa_policy = kappa_policy
        self.kappa_value = kappa_value
        self.entropy_coef = entropy_coef

        # 策略和价值网络
        self.policy = policy
        self.policy.to(self.device)
        
        # 使用ObGD优化器
        self.optimizer_policy = ObGD(
            [p for name, p in self.policy.named_parameters() if 'actor' in name or 'linear_mu' in name or 'linear_std' in name],
            lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy
        )
        self.optimizer_value = ObGD(
            [p for name, p in self.policy.named_parameters() if 'critic' in name],
            lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value
        )

        print(f"StreamingAC initialized with lr={lr}, gamma={gamma}, lamda={lamda}")
        print(f"Policy kappa={kappa_policy}, Value kappa={kappa_value}, entropy_coef={entropy_coef}")

    def pi(self, x):
        """策略网络前向传播"""
        return self.policy.actor_forward(x)

    def v(self, x):
        """价值网络前向传播"""
        return self.policy.critic_forward(x)

    def sample_action(self, obs):
        """采样动作"""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        elif obs.device != self.device:
            obs = obs.to(self.device)
        
        with torch.no_grad():
            mu, std = self.pi(obs)
            dist = Normal(mu, std)
            action = dist.sample()
        
        return action

    # 🔥🔥🔥 重要：网络更新算法起点 🔥🔥🔥
    def update_params(self, s, a, r, s_prime, done, overshooting_info=False):
        """核心更新方法，基于TD误差和eligibility traces进行在线学习"""
        done_mask = 0 if done else 1
        
        # 确保输入为tensor
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if isinstance(a, np.ndarray):
            a = torch.tensor(a, dtype=torch.float32, device=self.device)
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32, device=self.device)
        if isinstance(s_prime, np.ndarray):
            s_prime = torch.tensor(s_prime, dtype=torch.float32, device=self.device)
        if not isinstance(done_mask, torch.Tensor):
            done_mask = torch.tensor(done_mask, dtype=torch.float32, device=self.device)

        # 计算价值函数和TD误差
        v_s = self.v(s)
        v_prime = self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        # 计算策略损失
        mu, std = self.pi(s)
        dist = Normal(mu, std)
        log_prob_pi = -(dist.log_prob(a)).sum()
        
        # 价值函数损失
        value_output = -v_s
        
        # 熵损失（带符号）
        entropy_pi = -self.entropy_coef * dist.entropy().sum() * torch.sign(delta).item()

        # 清零梯度
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        
        # 反向传播
        # critic反向传播，保留计算图
        value_output.backward(retain_graph=True)
        # actor反向传播，计算log_prob_pi和entropy_pi的梯度
        (log_prob_pi + entropy_pi).backward()
        
        # 使用ObGD优化器更新参数，使用delta作为梯度
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        # 超调检测（可选）
        if overshooting_info:
            with torch.no_grad():
                v_s_new = self.v(s)
                v_prime_new = self.v(s_prime)
                td_target_new = r + self.gamma * v_prime_new * done_mask
                delta_bar = td_target_new - v_s_new
                if torch.sign(delta_bar * delta).item() == -1:
                    print("Overshooting Detected!")

        return {
            "value_function": value_output.item(),
            "policy_loss": log_prob_pi.item(),
            "entropy": dist.entropy().sum().item(),
            "td_error": delta.item()
        }
    # 🔥🔥🔥 重要：网络更新算法终点 🔥🔥🔥

    def load_pretrained_policy(self, checkpoint_path, finetune_mode="full", reset_optimizer=True):
        """
        加载预训练模型进行微调
        
        Args:
            checkpoint_path: 预训练模型路径
            finetune_mode: 微调模式
                - "full": 微调整个网络
                - "actor_only": 只微调actor网络，冻结critic
                - "critic_only": 只微调critic网络，冻结actor
                - "partial": 部分微调（可以根据需要自定义）
            reset_optimizer: 是否重置优化器状态（eligibility traces等）
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"预训练模型文件未找到: {checkpoint_path}")
        
        print(f"正在加载预训练模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 打印checkpoint的结构信息
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
        
        # 加载模型权重
        if "model_state_dict" in checkpoint:
            # Isaac Lab标准格式
            model_state_dict = checkpoint["model_state_dict"]
            print("📦 使用model_state_dict格式")
        elif "policy_state_dict" in checkpoint:
            # 可能的其他格式
            model_state_dict = checkpoint["policy_state_dict"]
            print("📦 使用policy_state_dict格式")
        elif "ac_state_dict" in checkpoint:
            # RSL-RL PPO格式
            model_state_dict = checkpoint["ac_state_dict"]
            print("📦 使用ac_state_dict格式（PPO模型）")
        else:
            # 直接的state_dict
            model_state_dict = checkpoint
            print("📦 使用直接state_dict格式")
        
        # 打印预训练模型的网络结构
        print(f"\n🔍 预训练模型参数:")
        for k, v in model_state_dict.items():
            print(f"  {k}: {v.shape}")
        
        # 打印当前网络结构
        current_dict = self.policy.state_dict()
        print(f"\n🔍 当前StreamingAC网络参数:")
        for k, v in current_dict.items():
            print(f"  {k}: {v.shape}")
        
        # 尝试智能参数映射
        mapped_dict = self._smart_parameter_mapping(model_state_dict, current_dict)
        
        # 过滤不匹配的键（如果网络结构有变化）
        policy_dict = self.policy.state_dict()
        filtered_dict = {}
        
        for k, v in mapped_dict.items():
            if k in policy_dict and v.shape == policy_dict[k].shape:
                filtered_dict[k] = v
                print(f"  ✓ 加载参数: {k} {v.shape}")
            else:
                if k in policy_dict:
                    print(f"  ✗ 跳过参数: {k} (形状不匹配: 预训练{v.shape} vs 当前{policy_dict[k].shape})")
                else:
                    print(f"  ✗ 跳过参数: {k} (当前网络中不存在)")
        
        if len(filtered_dict) == 0:
            print("\n⚠️  警告: 没有成功加载任何参数！")
            print("🔧 可能的解决方案:")
            print("   1. 检查预训练模型是否与当前网络结构兼容")
            print("   2. 确认预训练模型格式是否正确")
            print("   3. 考虑调整网络结构使其与预训练模型匹配")
            return False
        
        # 加载匹配的参数
        policy_dict.update(filtered_dict)
        self.policy.load_state_dict(policy_dict)
        
        print(f"\n✅ 成功加载 {len(filtered_dict)}/{len(model_state_dict)} 个参数")
        
        # 根据微调模式设置参数
        self._set_finetune_mode(finetune_mode)
        
        # 重置优化器状态
        if reset_optimizer:
            self.reset_optimizer_states()
            print("已重置优化器状态（eligibility traces）")
        
        print(f"预训练模型加载完成，微调模式: {finetune_mode}")
        return True

    def _smart_parameter_mapping(self, pretrained_dict, current_dict):
        """
        智能参数映射，尝试匹配不同命名格式的参数
        """
        mapped_dict = {}
        
        # 智能映射（PPO -> StreamingAC）- 无论直接匹配结果如何都执行
        print("🔄 尝试智能映射...")
        ppo_to_streaming_mapping = {
            # Actor mappings: PPO(0,2,4,6) -> StreamingAC(0,1,2,3)
            'actor.0.weight': 'actor.0.weight',
            'actor.0.bias': 'actor.0.bias', 
            'actor.2.weight': 'actor.1.weight',  # PPO第2层 -> StreamingAC第1层
            'actor.2.bias': 'actor.1.bias',
            'actor.4.weight': 'actor.2.weight',  # PPO第4层 -> StreamingAC第2层
            'actor.4.bias': 'actor.2.bias',
            'actor.6.weight': 'actor.3.weight',  # PPO第6层 -> StreamingAC第3层（输出层）
            'actor.6.bias': 'actor.3.bias',
            'std': 'std',  # PPO的std参数直接映射
            
            # Critic mappings: PPO(0,2,4,6) -> StreamingAC(0,1,2,3)
            'critic.0.weight': 'critic.0.weight',
            'critic.0.bias': 'critic.0.bias',
            'critic.2.weight': 'critic.1.weight',  # PPO第2层 -> StreamingAC第1层
            'critic.2.bias': 'critic.1.bias',
            'critic.4.weight': 'critic.2.weight',  # PPO第4层 -> StreamingAC第2层
            'critic.4.bias': 'critic.2.bias',
            'critic.6.weight': 'critic.3.weight',  # PPO第6层 -> StreamingAC第3层（输出层）
            'critic.6.bias': 'critic.3.bias'
        }
        
        for pretrained_key, current_key in ppo_to_streaming_mapping.items():
            if pretrained_key in pretrained_dict and current_key in current_dict:
                if current_key not in mapped_dict:  # 避免重复映射
                    mapped_dict[current_key] = pretrained_dict[pretrained_key]
                    print(f"  📌 智能映射: {pretrained_key} -> {current_key}")
        
        
        print(f"📊 映射结果: {len(mapped_dict)}/{len(current_dict)} 个参数成功映射")
        return mapped_dict

    def _set_finetune_mode(self, mode):
        """设置微调模式，决定哪些参数可以训练"""
        if mode == "full":
            # 微调整个网络
            for param in self.policy.parameters():
                param.requires_grad = True
        elif mode == "actor_only":
            # 只微调actor，冻结critic
            for name, param in self.policy.named_parameters():
                if 'actor' in name or 'linear_mu' in name or 'linear_std' in name:
                    param.requires_grad = True
                elif 'critic' in name:
                    param.requires_grad = False
        elif mode == "critic_only":
            # 只微调critic，冻结actor
            for name, param in self.policy.named_parameters():
                if 'critic' in name:
                    param.requires_grad = True
                elif 'actor' in name or 'linear_mu' in name or 'linear_std' in name:
                    param.requires_grad = False
        elif mode == "partial":
            # 示例：只微调最后几层
            for name, param in self.policy.named_parameters():
                # 可以根据需要自定义哪些层参与微调
                if 'output' in name or 'final' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # 重新初始化优化器（只包含需要训练的参数）
        self.optimizer_policy = ObGD(
            [p for name, p in self.policy.named_parameters() 
             if ('actor' in name or 'linear_mu' in name or 'linear_std' in name) and p.requires_grad],
            lr=self.lr, gamma=self.gamma, lamda=self.lamda, kappa=self.kappa_policy
        )
        self.optimizer_value = ObGD(
            [p for name, p in self.policy.named_parameters() 
             if 'critic' in name and p.requires_grad],
            lr=self.lr, gamma=self.gamma, lamda=self.lamda, kappa=self.kappa_value
        )

    def reset_optimizer_states(self):
        """重置优化器状态，清空eligibility traces"""
        for optimizer in [self.optimizer_policy, self.optimizer_value]:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p in optimizer.state:
                        optimizer.state[p]["eligibility_trace"] = torch.zeros_like(p.data)
        print("已重置所有eligibility traces")

    def save_checkpoint(self, save_path, extra_info=None):
        """保存当前模型状态"""
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_policy_state": self.optimizer_policy.state_dict(),
            "optimizer_value_state": self.optimizer_value.state_dict(),
            "algorithm_params": {
                "lr": self.lr,
                "gamma": self.gamma,
                "lamda": self.lamda,
                "kappa_policy": self.kappa_policy,
                "kappa_value": self.kappa_value,
                "entropy_coef": self.entropy_coef
            }
        }
        
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, save_path)
        print(f"模型已保存到: {save_path}")

    def adjust_learning_rate(self, new_lr):
        """调整学习率（微调时可能需要使用更小的学习率）"""
        self.lr = new_lr
        for optimizer in [self.optimizer_policy, self.optimizer_value]:
            for group in optimizer.param_groups:
                group["lr"] = new_lr
        print(f"学习率已调整为: {new_lr}")
