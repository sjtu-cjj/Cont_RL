# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def dof_error_named(
    env: ManagerBasedRLEnv, 
    joint_names: list, 
    scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize position error for specific named joints.
    
    This function computes the error between current joint positions and their default positions
    for a specified list of joints. This encourages the robot to keep these joints close to their
    default positions.
    
    Args:
        env: The environment instance.
        joint_names: List of joint names to include in computation.
        scale: Scaling factor for the error computation.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        torch.Tensor: Joint position error penalty for each environment.
    """
    # Extract the asset
    asset = env.scene[asset_cfg.name]
    
    # Get joint IDs for specified joint names
    joint_ids = []
    for name in joint_names:
        ids = asset.find_joints(name)[0]
        if len(ids) > 0:
            joint_ids.extend(ids)
    
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Get current and default joint positions
    current_dof_pos = asset.data.joint_pos[:, joint_ids]
    default_dof_pos = asset.data.default_joint_pos[:, joint_ids]
    
    # Calculate squared error for each joint and sum across joints
    error = torch.sum(torch.square(current_dof_pos - default_dof_pos), dim=1)
    
    # Apply scaling
    return error * scale


def stand_still(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    lin_threshold: float = 0.1, 
    ang_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize motion when the command is close to zero.
    
    This function penalizes the robot for making unnecessary movements when the command
    is to stand still (i.e., velocity commands are near zero). The penalty is computed as
    the sum of absolute deviations from the default joint positions, and is only applied
    when both linear and angular command components are below their respective thresholds.
    
    Args:
        env: The environment instance.
        command_name: The name of the command to check.
        lin_threshold: Threshold for linear velocity command components (x, y).
        ang_threshold: Threshold for angular velocity command component (yaw).
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        torch.Tensor: Stand still penalty for each environment.
    """
    # Get commands
    commands = env.command_manager.get_command(command_name)
    
    # Get joint positions
    asset = env.scene[asset_cfg.name]
    current_dof_pos = asset.data.joint_pos
    default_dof_pos = asset.data.default_joint_pos
    
    # Calculate joint position deviation
    pos_deviation = torch.sum(torch.abs(current_dof_pos - default_dof_pos), dim=1)
    
    # Check if commands are close to zero
    lin_vel_is_zero = torch.norm(commands[:, :2], dim=1) < lin_threshold
    ang_vel_is_zero = torch.abs(commands[:, 2]) < ang_threshold
    
    # Only apply penalty when both linear and angular velocities are close to zero
    penalty = pos_deviation * lin_vel_is_zero * ang_vel_is_zero
    
    return penalty


def energy_consumption(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    joint_names: list = None, 
    scale: float = 1.0
) -> torch.Tensor:
    """Penalize high energy consumption.
    
    This function computes the energy consumption as the product of joint torques and 
    squared joint velocities. This is a common model for energy consumption in robotics,
    where energy is proportional to the product of force and velocity.
    
    Args:
        env: The environment instance.
        asset_cfg: Configuration for the asset to compute energy for.
        joint_names: Names of joints to include in computation. If None, all joints are used.
        scale: Scaling factor for the energy computation.
        
    Returns:
        torch.Tensor: Energy consumption penalty for each environment.
    """
    # Extract the asset
    asset = env.scene[asset_cfg.name]
    
    # Get joint IDs if joint_names is provided
    if joint_names is not None:
        joint_ids = []
        for name in joint_names:
            ids = asset.find_joints(name)[0]
            if len(ids) > 0:
                joint_ids.extend(ids)
    else:
        # Use all joints
        joint_ids = None
    
    # Get joint torques and velocities
    joint_torques = asset.data.applied_torque
    #joint_torques = asset.data.joint_torques
    joint_vel = asset.data.joint_vel
    
    # Calculate energy as torque * velocity^2
    if joint_ids is not None:
        # Only for specified joints
        energy = torch.sum(torch.abs(joint_torques[:, joint_ids]) * torch.square(joint_vel[:, joint_ids]), dim=1)
    else:
        # For all joints
        energy = torch.sum(torch.abs(joint_torques) * torch.square(joint_vel), dim=1)
    
    # Apply scaling
    return energy * scale


# def joint_adaptation_reward(
#     env: ManagerBasedRLEnv,
#     healthy_joints_bonus: float = 0.1,
#     damaged_joints_penalty: float = 0.2,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward for adapting to joint damage by encouraging use of healthy joints.
    
#     This function encourages the robot to adapt to joint damage by:
#     1. Providing bonus for using healthy joints efficiently
#     2. Penalizing excessive use of damaged joints
    
#     Args:
#         env: The environment instance.
#         healthy_joints_bonus: Bonus weight for using healthy joints.
#         damaged_joints_penalty: Penalty weight for overusing damaged joints.
#         asset_cfg: Configuration for the robot asset.
        
#     Returns:
#         torch.Tensor: Adaptation reward for each environment.
#     """
#     # Extract the asset
#     asset = env.scene[asset_cfg.name]
    
#     # Get joint torques (absolute values for effort calculation)
#     joint_torques = torch.abs(asset.data.applied_torque)
    
#     # For now, assume all joints are healthy (you can modify this based on your damage model)
#     # In practice, you would have a damage mask: damaged_joints_mask = get_damaged_joints_mask(env)
#     num_joints = joint_torques.shape[1]
#     healthy_joints_mask = torch.ones_like(joint_torques, dtype=torch.bool)
    
#     # Calculate torque distribution efficiency
#     total_torque = torch.sum(joint_torques, dim=1, keepdim=True)
#     torque_distribution = joint_torques / (total_torque + 1e-8)  # Avoid division by zero
    
#     # Reward even distribution among healthy joints (avoid overloading single joints)
#     torque_variance = torch.var(torque_distribution, dim=1)
#     adaptation_reward = -torque_variance * healthy_joints_bonus
    
#     return adaptation_reward


# def stability_reward(
#     env: ManagerBasedRLEnv,
#     orientation_threshold: float = 0.1,
#     velocity_threshold: float = 0.05,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward for maintaining stability during adaptation.
    
#     This function rewards the robot for maintaining stable motion patterns
#     even when joints are damaged, emphasizing smooth and controlled movement.
    
#     Args:
#         env: The environment instance.
#         orientation_threshold: Threshold for orientation stability.
#         velocity_threshold: Threshold for velocity smoothness.
#         asset_cfg: Configuration for the robot asset.
        
#     Returns:
#         torch.Tensor: Stability reward for each environment.
#     """
#     # Extract the asset
#     asset = env.scene[asset_cfg.name]
    
#     # Check orientation stability (roll and pitch should be small)
#     root_quat = asset.data.root_quat_w
#     from isaaclab.utils.math import quat_to_euler_xyz
#     roll, pitch, _ = quat_to_euler_xyz(root_quat)
    
#     orientation_penalty = torch.abs(roll) + torch.abs(pitch)
#     orientation_stable = orientation_penalty < orientation_threshold
    
#     # Check velocity smoothness (low angular velocity in roll/pitch)
#     ang_vel = asset.data.root_ang_vel_w
#     velocity_smoothness = torch.abs(ang_vel[:, 0]) + torch.abs(ang_vel[:, 1])  # roll and pitch rates
#     velocity_smooth = velocity_smoothness < velocity_threshold
    
#     # Combine stability metrics
#     stability_score = orientation_stable.float() + velocity_smooth.float()
    
#     return stability_score


# def damaged_joint_compensation(
#     env: ManagerBasedRLEnv,
#     damaged_joints_mask: torch.Tensor = None,
#     compensation_bonus: float = 0.5,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward for compensatory movement patterns when joints are damaged.
    
#     This function encourages the development of alternative movement patterns
#     that compensate for damaged joints by utilizing healthy joints more effectively.
    
#     Args:
#         env: The environment instance.
#         damaged_joints_mask: Boolean mask indicating which joints are damaged.
#         compensation_bonus: Bonus for effective compensation strategies.
#         asset_cfg: Configuration for the robot asset.
        
#     Returns:
#         torch.Tensor: Compensation reward for each environment.
#     """
#     # Extract the asset
#     asset = env.scene[asset_cfg.name]
    
#     # Get joint positions and velocities
#     joint_pos = asset.data.joint_pos
#     joint_vel = asset.data.joint_vel
    
#     if damaged_joints_mask is None:
#         # Default: no damaged joints (you can implement damage detection here)
#         damaged_joints_mask = torch.zeros_like(joint_pos, dtype=torch.bool)
    
#     # Calculate compensation effectiveness
#     # Healthy joints should show more activity when damaged joints are present
#     healthy_joints_mask = ~damaged_joints_mask
    
#     # Measure healthy joint utilization
#     healthy_joint_activity = torch.abs(joint_vel) * healthy_joints_mask.float()
#     damaged_joint_activity = torch.abs(joint_vel) * damaged_joints_mask.float()
    
#     # Reward high healthy joint activity when damaged joints are present
#     if torch.any(damaged_joints_mask):
#         total_healthy_activity = torch.sum(healthy_joint_activity, dim=1)
#         total_damaged_activity = torch.sum(damaged_joint_activity, dim=1)
        
#         # Encourage healthy joints to compensate
#         compensation_ratio = total_healthy_activity / (total_damaged_activity + 1e-8)
#         compensation_reward = torch.clamp(compensation_ratio * compensation_bonus, 0, 1)
#     else:
#         compensation_reward = torch.zeros(env.num_envs, device=env.device)
    
#     return compensation_reward


