# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


@configclass 
class RslRlStreamingActorCriticCfg:
    """Configuration for streamingAC actor-critic policy."""
    class_name: str = "ActorCriticStreaming"
    
    # network architecture
    init_noise_std: float = 1.0
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "leaky_relu"  # streamingAC使用LeakyReLU
    
    # streaming-specific parameters
    use_layer_norm: bool = True
    sparsity: float = 0.9  # 90% sparsity
    kappa_policy: float = 3.0
    kappa_value: float = 2.0


@configclass
class RslRlStreamingAlgorithmCfg:
    """Configuration for streamingAC algorithm."""
    class_name: str = "streamingAC"
    
    # learning rate (single rate for both policy and value)
    lr: float = 1.0e-3
    
    # ObGD optimizer parameters
    kappa_policy: float = 3.0  # policy network sparsity parameter
    kappa_value: float = 2.0   # value network sparsity parameter
    
    # discount and traces
    gamma: float = 0.99
    lamda: float = 0.95  # eligibility trace decay (注意：参数名是lamda不是lambda_trace)
    
    # entropy regularization
    entropy_coef: float = 0.01
    
    # NOTE: 删除了streamingAC不支持的参数:
    # - actor_lr, critic_lr (只使用单一的lr)
    # - dynamic_entropy (streamingAC内部自动处理)
    # - max_grad_norm (ObGD优化器内部处理)
    # - value_loss_coef (streamingAC内部处理)
    # - reset_traces_on_done (ObGD优化器内部处理)


@configclass
class UnitreeGo2RoughStreamingRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for streamingAC training on rough terrain."""
    
    # runner configuration
    runner_class_name: str = "StreamingRunner"  # 使用StreamingRunner
    num_steps_per_env = 1000
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough_streaming"
    empirical_normalization = False
    
    # policy configuration
    policy = RslRlStreamingActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],  # 保持与PPO相同的网络架构
        critic_hidden_dims=[512, 256, 128],  # 保持与PPO相同的网络架构
        activation="leaky_relu",
        use_layer_norm=True,
        sparsity=0.9,
        kappa_policy=3.0,
        kappa_value=2.0,
    )
    
    # algorithm configuration
    algorithm = RslRlStreamingAlgorithmCfg(
        lr=1.0e-3,
        kappa_policy=3.0,
        kappa_value=2.0,
        gamma=0.99,
        lamda=0.95,
        entropy_coef=0.01,
    )


@configclass
class UnitreeGo2FlatStreamingRunnerCfg(UnitreeGo2RoughStreamingRunnerCfg):
    """Configuration for streamingAC training on flat terrain."""
    
    def __post_init__(self):
        super().__post_init__()

        # 调整平地训练的参数
        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat_streaming"
        
        # 保持网络架构不变，但可以调整其他参数
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
        
        # 平地训练可能需要更低的学习率
        self.algorithm.lr = 5.0e-4
