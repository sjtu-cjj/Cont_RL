# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
关节损伤模拟模块

提供用于机器人关节故障模拟的环境包装器和工具函数。
支持永久性关节损伤，用于研究机器人在故障条件下的适应能力。
"""

import os
import random
import gymnasium as gym
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple


class JointNameMapper:
    """关节名称映射器，处理关节名称到索引的转换"""
    
    def __init__(self):
        # UnitreeGo2标准关节名称到索引的映射
        self.joint_name_to_index = {
            # 前左腿 (FL)
            "FL_hip": 0, "FL_thigh": 1, "FL_calf": 2,
            # 前右腿 (FR) 
            "FR_hip": 3, "FR_thigh": 4, "FR_calf": 5,
            # 后左腿 (RL)
            "RL_hip": 6, "RL_thigh": 7, "RL_calf": 8,
            # 后右腿 (RR)
            "RR_hip": 9, "RR_thigh": 10, "RR_calf": 11
        }
        
        # 支持的完整关节名称映射
        self.full_joint_name_mapping = {
            "FL_hip_joint": "FL_hip", "FL_thigh_joint": "FL_thigh", "FL_calf_joint": "FL_calf",
            "FR_hip_joint": "FR_hip", "FR_thigh_joint": "FR_thigh", "FR_calf_joint": "FR_calf",
            "RL_hip_joint": "RL_hip", "RL_thigh_joint": "RL_thigh", "RL_calf_joint": "RL_calf",
            "RR_hip_joint": "RR_hip", "RR_thigh_joint": "RR_thigh", "RR_calf_joint": "RR_calf"
        }
        
        # 反向映射：索引到关节名称
        self.index_to_joint_name = {v: k for k, v in self.joint_name_to_index.items()}
        
        # 四足机器人通用关节名称（用于fallback）
        self.generic_joint_names = [
            "FL_hip", "FL_thigh", "FL_calf",     # 前左腿
            "FR_hip", "FR_thigh", "FR_calf",     # 前右腿
            "RL_hip", "RL_thigh", "RL_calf",     # 后左腿
            "RR_hip", "RR_thigh", "RR_calf"      # 后右腿
        ]
    
    def parse_joint_specification(self, joint_spec: str) -> Optional[List[int]]:
        """
        解析关节规格字符串，返回关节索引列表
        
        Args:
            joint_spec: 关节规格字符串，支持逗号分隔的多个关节名称
            
        Returns:
            关节索引列表，如果解析失败返回None
        """
        if not joint_spec:
            return None
            
        joint_names = [x.strip() for x in joint_spec.split(',')]
        joint_indices = []
        
        for joint_name in joint_names:
            # 首先检查是否为完整关节名称，如果是则转换为简化名称
            if joint_name in self.full_joint_name_mapping:
                joint_name = self.full_joint_name_mapping[joint_name]
            
            # 查找索引
            if joint_name in self.joint_name_to_index:
                index = self.joint_name_to_index[joint_name]
                joint_indices.append(index)
                print(f"🎯 Mapped joint name '{joint_name}' to index {index}")
            else:
                print(f"⚠️  Unknown joint name: '{joint_name}'. Available names: {list(self.joint_name_to_index.keys())}")
        
        if not joint_indices:
            print(f"⚠️  No valid joints found in specification: {joint_spec}")
            return None
        
        print(f"🎯 Final specified joints to damage: {joint_indices}")
        return joint_indices
    
    def get_joint_names(self, joint_indices: List[int], num_actions: int) -> List[str]:
        """获取关节索引对应的名称列表"""
        # 如果关节数不匹配，使用通用命名
        if num_actions != len(self.generic_joint_names):
            joint_names = [f"joint_{i}" for i in range(num_actions)]
        else:
            joint_names = self.generic_joint_names
        
        return [joint_names[i] if i < len(joint_names) else f"joint_{i}" for i in joint_indices]


class DamageApplier:
    """损伤应用器，负责对动作应用各种类型的损伤"""
    
    @staticmethod
    def apply_zero_damage(damaged_actions: torch.Tensor, env_idx: int, damaged_joints: torch.Tensor) -> None:
        """应用零力矩损伤"""
        damaged_actions[env_idx, damaged_joints] = 0.0
    
    @staticmethod
    def apply_partial_damage(damaged_actions: torch.Tensor, env_idx: int, damaged_joints: torch.Tensor, severity: float) -> None:
        """应用部分损伤"""
        damaged_actions[env_idx, damaged_joints] *= (1.0 - severity)
    
    @staticmethod
    def apply_random_damage(damaged_actions: torch.Tensor, env_idx: int, damaged_joints: torch.Tensor) -> None:
        """应用随机噪声损伤"""
        random_noise = torch.randn_like(damaged_actions[env_idx, damaged_joints]) * 0.1
        damaged_actions[env_idx, damaged_joints] = random_noise


class JointDamageWrapper(gym.Wrapper):
    """
    关节损伤包装器，用于模拟关节故障
    
    支持永久性关节损伤，适合研究机器人在长期故障下的适应能力。
    一旦指定关节损伤，将在整个训练过程中持续损伤（永久性损伤）
    """
    
    def __init__(self, env: gym.Env, damage_config: Dict):
        """
        初始化关节损伤包装器
        
        Args:
            env: 基础环境
            damage_config: 损伤配置字典
        """
        super().__init__(env)
        self.damage_config = damage_config
        self.joint_mapper = JointNameMapper()
        self.damage_applier = DamageApplier()
        
        # 获取动作空间信息
        self.num_actions = env.action_space.shape[1]
        
        # 环境数量将在第一次step时动态推断
        self.num_envs = None
        self.initialized = False
        
        # 损伤状态
        self.damage_masks = None
        self.damage_states = None  # 存储永久性损伤状态：0=正常，1=永久损伤
        
        # 统计信息
        self.total_damage_events = 0
        self.step_count = 0
        
        # 解析指定的损伤关节
        self.specified_joints = self._parse_specified_joints()
        
        # 输出设置
        self.output_damage_info = damage_config.get('output_info', False)
        
        self._print_initialization_info()
    
    def _parse_specified_joints(self) -> Optional[List[int]]:
        """解析环境变量中指定的损伤关节"""
        damaged_joint_env = os.getenv('DAMAGED_JOINT', None)
        if damaged_joint_env:
            return self.joint_mapper.parse_joint_specification(damaged_joint_env)
        return None
    
    def _print_initialization_info(self):
        """打印初始化信息"""
        print(f"🤖 JointDamageWrapper initialized:")
        print(f"   📊 Action space: {self.num_actions} joints")
        print(f"   🌍 Environments: will be determined dynamically")
        print(f"   ⚡ Damage probability: {self.damage_config['probability']}")
        print(f"   ⏰ Damage duration: PERMANENT (entire training)")
        print(f"   🎯 Max damaged joints: {self.damage_config['max_damaged_joints']}")
        print(f"   💥 Damage type: {self.damage_config['type']}")
        if self.specified_joints:
            joint_names = self.joint_mapper.get_joint_names(self.specified_joints, self.num_actions)
            print(f"   🔧 Specified joints: {self.specified_joints} {joint_names}")
        print(f"   📝 Output damage info: {self.output_damage_info}")
    
    def _lazy_init(self, actions: torch.Tensor):
        """基于第一次action的形状来推断环境数量并初始化"""
        if not self.initialized:
            self.num_envs = actions.shape[0] if len(actions.shape) == 2 else 1
            print(f"🌍 JointDamageWrapper: Detected {self.num_envs} environments")
            self.reset_damage_states()
            self.initialized = True
    
    def reset_damage_states(self):
        """重置所有损伤状态"""
        if self.num_envs is None:
            return
            
        # 每个环境的损伤掩膜 [num_envs, num_actions]
        self.damage_masks = torch.ones((self.num_envs, self.num_actions), dtype=torch.float32)
        
        # 每个环境中每个关节的损伤状态 [num_envs, num_actions] (0=未损伤, 1=已损伤)
        self.damage_states = torch.zeros((self.num_envs, self.num_actions), dtype=torch.int32)
        
        print(f"🔄 Damage states reset for {self.num_envs} envs x {self.num_actions} actions")
    
    def apply_joint_damage(self, actions: torch.Tensor) -> torch.Tensor:
        """
        应用关节损伤到动作
        
        Args:
            actions: 原始动作张量 [num_envs, num_actions]
            
        Returns:
            damaged_actions: 应用损伤后的动作张量
        """
        if not self.damage_config['enabled']:
            return actions
            
        # 确保actions是tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
            
        # 懒初始化
        self._lazy_init(actions)
        
        # 随机生成新的损伤（如果还有未损伤的关节）
        if random.random() < self.damage_config['probability']:
            self._generate_new_damage()
        
        # 根据损伤类型应用掩膜
        damaged_actions = self._apply_damage_masks(actions)
        
        # 更新损伤掩膜（永久性损伤，无需清除）
        self._update_damage_masks()
        
        self.step_count += 1
        
        # 输出当前损伤状态
        if self.output_damage_info and self.step_count % 100 == 0:
            self._output_current_damage_status()
        
        return damaged_actions
    
    def _generate_new_damage(self):
        """生成新的关节损伤"""
        for env_idx in range(self.num_envs):
            current_damaged = (self.damage_states[env_idx] > 0).sum().item()
            
            if current_damaged < self.damage_config['max_damaged_joints']:
                available_joints = self._get_available_joints(env_idx)
                
                if len(available_joints) > 0:
                    damaged_joints = self._select_joints_to_damage(available_joints, current_damaged)
                    
                    if len(damaged_joints) > 0:
                        self._apply_permanent_damage(env_idx, damaged_joints)
    
    def _get_available_joints(self, env_idx: int) -> torch.Tensor:
        """获取可用于损伤的关节"""
        if self.specified_joints:
            # 使用指定的关节
            return torch.tensor([j for j in self.specified_joints 
                               if j < self.num_actions and self.damage_states[env_idx, j] == 0])
        else:
            # 随机选择关节（当前未损伤的关节）
            return torch.where(self.damage_states[env_idx] == 0)[0]
    
    def _select_joints_to_damage(self, available_joints: torch.Tensor, current_damaged: int) -> torch.Tensor:
        """选择要损伤的关节"""
        max_new_damage = self.damage_config['max_damaged_joints'] - current_damaged
        
        if self.specified_joints:
            # 对于指定关节，一次性损伤所有可用的
            num_to_damage = min(len(available_joints), max_new_damage)
            return available_joints[:num_to_damage]
        else:
            # 随机选择1-2个关节进行损伤
            num_to_damage = min(random.randint(1, 2), max_new_damage, len(available_joints))
            return available_joints[torch.randperm(len(available_joints))[:num_to_damage]]
    
    def _apply_permanent_damage(self, env_idx: int, damaged_joints: torch.Tensor):
        """应用永久性损伤"""
        for joint_idx in damaged_joints:
            self.damage_states[env_idx, joint_idx] = 1
            
        self.total_damage_events += len(damaged_joints)
        
        # 输出损伤信息
        if self.output_damage_info or self.step_count % 1000 == 0:
            joint_names = self.joint_mapper.get_joint_names(damaged_joints.tolist(), self.num_actions)
            print(f"💥 Env {env_idx}: PERMANENT damage on joints {damaged_joints.tolist()} {joint_names}")
    
    def _apply_damage_masks(self, actions: torch.Tensor) -> torch.Tensor:
        """根据损伤类型应用掩膜"""
        damaged_actions = actions.clone()
        
        for env_idx in range(self.num_envs):
            damaged_joints = self.damage_states[env_idx] > 0
            
            if damaged_joints.any():
                damage_type = self.damage_config['type']
                
                if damage_type == 'zero':
                    self.damage_applier.apply_zero_damage(damaged_actions, env_idx, damaged_joints)
                elif damage_type == 'partial':
                    severity = self.damage_config['severity']
                    self.damage_applier.apply_partial_damage(damaged_actions, env_idx, damaged_joints, severity)
                elif damage_type == 'random':
                    self.damage_applier.apply_random_damage(damaged_actions, env_idx, damaged_joints)
                    
        return damaged_actions
    
    def _update_damage_masks(self):
        """更新损伤掩膜"""
        for env_idx in range(self.num_envs):
            for joint_idx in range(self.num_actions):
                if self.damage_states[env_idx, joint_idx] > 0:
                    damage_type = self.damage_config['type']
                    if damage_type == 'zero':
                        self.damage_masks[env_idx, joint_idx] = 0.0
                    elif damage_type == 'partial':
                        self.damage_masks[env_idx, joint_idx] = 1.0 - self.damage_config['severity']
                    else:  # random
                        self.damage_masks[env_idx, joint_idx] = 0.1
                else:
                    self.damage_masks[env_idx, joint_idx] = 1.0
    
    def _output_current_damage_status(self):
        """输出当前损伤状态"""
        total_damaged = (self.damage_states > 0).sum().item()
        if total_damaged > 0:
            print(f"📊 Step {self.step_count}: {total_damaged} joints currently damaged")
            for env_idx in range(min(3, self.num_envs)):  # 只显示前3个环境
                damaged_joints = torch.where(self.damage_states[env_idx] > 0)[0]
                if len(damaged_joints) > 0:
                    joint_names = self.joint_mapper.get_joint_names(damaged_joints.tolist(), self.num_actions)
                    print(f"   Env {env_idx}: joints {damaged_joints.tolist()} {joint_names}, status: PERMANENTLY DAMAGED")
    
    def _fix_last_action_in_obs(self, obs: Union[Dict, torch.Tensor], damaged_actions: torch.Tensor) -> Union[Dict, torch.Tensor]:
        """
        修正观测值中的last_action部分，确保与实际执行的受损动作一致
        
        Args:
            obs: 环境返回的观测值，可能是dict或tensor
            damaged_actions: 应用损伤后的动作 [num_envs, num_actions]
        
        Returns:
            修正后的观测值
        """
        if isinstance(obs, dict):
            return self._fix_dict_obs(obs, damaged_actions)
        elif hasattr(obs, 'clone'):
            return self._fix_tensor_obs(obs, damaged_actions)
        else:
            if self.output_damage_info and self.step_count % 1000 == 0:
                print(f"⚠️ Warning: Unsupported obs type: {type(obs)}. Cannot fix last_action.")
            return obs
    
    def _fix_dict_obs(self, obs: Dict, damaged_actions: torch.Tensor) -> Dict:
        """修正字典类型的观测值"""
        if 'policy' not in obs:
            if self.output_damage_info and self.step_count % 1000 == 0:
                print(f"⚠️ Warning: obs is dict but no 'policy' key found. Available keys: {list(obs.keys())}")
            return obs
        
        obs_fixed = obs.copy()
        policy_obs = obs['policy']
        
        # 克隆policy观测值
        policy_obs_fixed = policy_obs.clone() if hasattr(policy_obs, 'clone') else policy_obs.copy()
        
        # 假设观测值的最后num_actions维是last_action
        num_actions = damaged_actions.shape[1]
        policy_obs_fixed[:, -num_actions:] = damaged_actions
        obs_fixed['policy'] = policy_obs_fixed
        
        self._log_obs_fix(policy_obs, policy_obs_fixed, damaged_actions, "policy")
        return obs_fixed
    
    def _fix_tensor_obs(self, obs: torch.Tensor, damaged_actions: torch.Tensor) -> torch.Tensor:
        """修正张量类型的观测值"""
        obs_fixed = obs.clone()
        num_actions = damaged_actions.shape[1]
        obs_fixed[:, -num_actions:] = damaged_actions
        
        self._log_obs_fix(obs, obs_fixed, damaged_actions, "tensor")
        return obs_fixed
    
    def _log_obs_fix(self, original_obs: torch.Tensor, fixed_obs: torch.Tensor, 
                     damaged_actions: torch.Tensor, obs_type: str):
        """记录观测值修正信息"""
        if not (self.output_damage_info and self.step_count % 100 == 0):
            return
            
        print(f"🔧 Fixed last_action in {obs_type} observation to match damaged actions")
        num_actions = damaged_actions.shape[1]
        
        for env_idx in range(min(2, self.num_envs)):
            damaged_joints = torch.where(self.damage_states[env_idx] > 0)[0]
            if len(damaged_joints) > 0:
                original_actions = original_obs[:, -num_actions:][env_idx, damaged_joints]
                fixed_actions = fixed_obs[:, -num_actions:][env_idx, damaged_joints]
                print(f"   Env {env_idx} damaged joints {damaged_joints.tolist()}: "
                      f"original={original_actions.cpu().numpy().tolist()}, "
                      f"fixed={fixed_actions.cpu().numpy().tolist()}")
    
    def step(self, actions: torch.Tensor) -> Tuple:
        """环境步进，应用关节损伤"""
        # 应用关节损伤
        damaged_actions = self.apply_joint_damage(actions)
        
        # 执行环境步进
        obs, reward, terminated, truncated, info = self.env.step(damaged_actions)
        
        # 修正观测值中的last_action部分
        if self.initialized and self.damage_config['enabled']:
            obs = self._fix_last_action_in_obs(obs, damaged_actions)
        
        # 在info中添加损伤信息
        if isinstance(info, dict) and self.initialized:
            info['joint_damage'] = self._get_damage_info()
            
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple:
        """重置环境，可选择是否重置损伤状态"""
        obs, info = self.env.reset(**kwargs)
        # 这里选择保持损伤状态，模拟持续的故障
        # 如果想要每次重置都清除损伤，可以调用 self.reset_damage_states()
        return obs, info
    
    def _get_damage_info(self) -> Dict:
        """获取当前损伤信息"""
        return {
            'damage_masks': self.damage_masks.clone(),
            'damage_states': self.damage_states.clone(),
            'total_damage_events': self.total_damage_events,
            'current_damaged_joints': (self.damage_states > 0).sum().item()
        }
    
    def get_damage_statistics(self) -> Dict:
        """获取损伤统计信息"""
        if not self.initialized:
            return {
                'total_damage_events': 0,
                'current_damaged_joints': 0,
                'damage_rate': 0.0,
                'step_count': 0
            }
            
        current_damaged = (self.damage_states > 0).sum().item()
        return {
            'total_damage_events': self.total_damage_events,
            'current_damaged_joints': current_damaged,
            'damage_rate': self.total_damage_events / max(self.step_count, 1),
            'step_count': self.step_count
        }
    
    def get_damage_report(self) -> str:
        """获取详细的损伤报告"""
        if not self.initialized:
            return "Damage wrapper not initialized"
        
        report = []
        report.append(f"🤖 Joint Damage Report (Step {self.step_count}):")
        report.append(f"📊 Total damage events: {self.total_damage_events}")
        report.append(f"📈 Damage rate: {self.total_damage_events / max(self.step_count, 1):.6f} events/step")
        
        # 当前损伤状态
        current_damaged = (self.damage_states > 0).sum().item()
        report.append(f"🔧 Currently damaged joints: {current_damaged}")
        
        if current_damaged > 0:
            report.append("📋 Detailed damage status:")
            for env_idx in range(self.num_envs):
                damaged_joints = torch.where(self.damage_states[env_idx] > 0)[0]
                if len(damaged_joints) > 0:
                    joint_names = self.joint_mapper.get_joint_names(damaged_joints.tolist(), self.num_actions)
                    damage_types = self._get_damage_types_for_joints(env_idx, damaged_joints)
                    
                    report.append(f"   Env {env_idx}:")
                    for joint_idx, name, dtype in zip(damaged_joints, joint_names, damage_types):
                        report.append(f"     - Joint {joint_idx} ({name}): {dtype} damage, status: PERMANENT")
        
        return "\n".join(report)
    
    def _get_damage_types_for_joints(self, env_idx: int, damaged_joints: torch.Tensor) -> List[str]:
        """获取指定关节的损伤类型描述"""
        damage_types = []
        for joint_idx in damaged_joints:
            mask_val = self.damage_masks[env_idx, joint_idx].item()
            if mask_val == 0.0:
                damage_types.append("zero")
            elif mask_val < 1.0:
                damage_types.append(f"partial({1-mask_val:.2f})")
            else:
                damage_types.append("random")
        return damage_types


def create_damage_config(args_cli) -> Dict:
    """
    根据命令行参数创建损伤配置字典
    
    Args:
        args_cli: 命令行参数
        
    Returns:
        损伤配置字典
    """
    return {
        'enabled': args_cli.enable_joint_damage,
        'probability': args_cli.damage_probability,
        'max_damaged_joints': args_cli.max_damaged_joints,
        'type': args_cli.damage_type,
        'severity': args_cli.damage_severity,
        'output_info': args_cli.output_damage_info
    }


def print_damage_config_info(damage_config: Dict):
    """打印损伤配置信息"""
    print("\n" + "="*60)
    print("🤖 JOINT DAMAGE SIMULATION ENABLED")
    print("="*60)
    print(f"💥 Damage Type: {damage_config['type']}")
    print(f"⚡ Probability: {damage_config['probability']:.3f} per step")
    print(f"⏰ Duration: PERMANENT (entire training)")
    print(f"🎯 Max damaged joints: {damage_config['max_damaged_joints']}")
    if damage_config['type'] == 'partial':
        print(f"🔧 Damage severity: {damage_config['severity']:.2f}")
    print("="*60 + "\n")


def find_damage_wrapper(env) -> Optional[JointDamageWrapper]:
    """
    从环境链中查找损伤包装器
    
    Args:
        env: 环境实例
        
    Returns:
        损伤包装器实例，如果未找到返回None
    """
    current_env = env
    while hasattr(current_env, 'env'):
        if isinstance(current_env, JointDamageWrapper):
            return current_env
        current_env = current_env.env
    return None


def print_damage_statistics(damage_wrapper: JointDamageWrapper, output_detailed_report: bool = False):
    """打印损伤统计信息"""
    if damage_wrapper is None:
        return
        
    damage_stats = damage_wrapper.get_damage_statistics()
    print("\n" + "="*60)
    print("🤖 JOINT DAMAGE STATISTICS")
    print("="*60)
    print(f"📊 Total damage events: {damage_stats['total_damage_events']}")
    print(f"🎯 Current damaged joints: {damage_stats['current_damaged_joints']}")
    print(f"📈 Damage rate: {damage_stats['damage_rate']:.6f} events/step")
    print(f"⏱️  Total steps: {damage_stats['step_count']}")
    print("="*60)
    
    # 输出详细损伤报告
    if output_detailed_report:
        print("\n" + damage_wrapper.get_damage_report())
        print("="*60) 