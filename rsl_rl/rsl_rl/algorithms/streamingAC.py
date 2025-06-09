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
        value_output.backward(retain_graph=True)
        (log_prob_pi + entropy_pi).backward()
        
        # 使用ObGD优化器更新参数
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
