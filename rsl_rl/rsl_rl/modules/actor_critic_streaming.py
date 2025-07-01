# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def sparse_init(tensor, sparsity=0.9):
    """Sparse initialization: set a fraction of weights to zero"""
    with torch.no_grad():
        # Initialize with normal distribution first
        nn.init.normal_(tensor, mean=0.0, std=0.1)
        # Create sparse mask
        mask = torch.rand_like(tensor) > sparsity
        tensor.mul_(mask)


def initialize_weights(m):
    """Initialize weights using sparse initialization"""
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


class ActorCriticStreaming(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],  # 改回多层256维隐藏层
        critic_hidden_dims=[256, 256, 256],  # 改回多层256维隐藏层
        activation="elu",  # 更改为elu
        init_noise_std=1.0,
        noise_std_type: str = "scalar",  # 改回scalar类型，表示固定标准差
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        # 不再使用resolve_nn_activation，直接使用leaky_relu
        # activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        
        # Actor Network - 使用Sequential结构，保持PPO命名方式（无LayerNorm）
        actor_layers = []
        # 第0层: 输入层
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        # 第1层: 隐藏层1  
        actor_layers.append(nn.Linear(actor_hidden_dims[0], actor_hidden_dims[1]))
        # 第2层: 隐藏层2
        actor_layers.append(nn.Linear(actor_hidden_dims[1], actor_hidden_dims[2]))
        # 第3层: 输出层
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        
        self.actor = nn.Sequential(*actor_layers)
        
        # 根据noise_std_type决定标准差实现方式
        if noise_std_type == "learned":
            self.linear_std = nn.Linear(actor_hidden_dims[-1], num_actions)
        else:  # scalar type
            self.linear_std = None
            # 固定标准差参数
            self.register_parameter("std", nn.Parameter(torch.ones(num_actions) * init_noise_std))
        
        # Critic Network - 使用Sequential结构，保持PPO命名方式  
        critic_layers = []
        # 第0层: 输入层
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        #critic_layers.append(nn.LayerNorm(critic_hidden_dims[0]))
        # 第1层: 隐藏层1
        critic_layers.append(nn.Linear(critic_hidden_dims[0], critic_hidden_dims[1]))
        #critic_layers.append(nn.LayerNorm(critic_hidden_dims[1]))
        # 第2层: 隐藏层2
        critic_layers.append(nn.Linear(critic_hidden_dims[1], critic_hidden_dims[2]))
        #critic_layers.append(nn.LayerNorm(critic_hidden_dims[2]))
        # 第4层: 输出层
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        
        self.critic = nn.Sequential(*critic_layers)

        # 应用权重初始化
        self.apply(initialize_weights)
        print(f"Actor Critic Streaming Network Initialized")

        print(f"Actor Network (PPO-style): {mlp_input_dim_a} -> {' -> '.join(map(str, actor_hidden_dims))} -> mu({num_actions}) + std({num_actions}, {noise_std_type})")
        print(f"Critic Network (PPO-style): {mlp_input_dim_c} -> {' -> '.join(map(str, critic_hidden_dims))} -> value(1)")

        # 存储噪声类型（兼容性）
        self.noise_std_type = noise_std_type

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def actor_forward(self, x):
        """Actor前向传播，使用Sequential结构（无LayerNorm）"""
        # 前向传播，对所有Linear层（除最后一层）应用elu激活
        for i, layer in enumerate(self.actor):
            x = layer(x)
            # 对Linear层应用elu（除了最后一层）
            if i < len(self.actor) - 1:  # 不是最后一层
                x = F.elu(x)
        
        # 最后一层的输出就是mu
        mu = x
        
        # 根据noise_std_type决定标准差计算方式
        if self.noise_std_type == "learned":
            # 需要重新获取倒数第二层的输出
            x_feature = x  # 这里需要修改，获取特征层输出
            pre_std = self.linear_std(x_feature)
            std = F.softplus(pre_std)  # 使用softplus确保std为正值
        else:  # scalar type
            # 使用固定标准差，扩展为与mu相同的batch维度
            std = self.std.expand_as(mu)
        
        return mu, std

    def critic_forward(self, x):
        """Critic前向传播，使用Sequential结构（无LayerNorm）"""
        # 前向传播，对所有Linear层（除最后一层）应用elu激活
        for i, layer in enumerate(self.critic):
            x = layer(x)
            # 对Linear层应用elu（除了最后一层）
            if i < len(self.critic) - 1:  # 不是最后一层
                x = F.elu(x)
        
        return x

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # 使用新的actor前向传播
        mean, std = self.actor_forward(observations)
        # 创建分布
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """确定性推理，返回动作均值"""
        actions_mean, _ = self.actor_forward(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        """使用新的critic前向传播"""
        value = self.critic_forward(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
