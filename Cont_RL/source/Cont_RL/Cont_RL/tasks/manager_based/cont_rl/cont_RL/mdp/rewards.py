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



def base_height_reward(
    env: ManagerBasedRLEnv, 
    target_height: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining base at target height (for stand-up task).
    
    This function rewards the robot for lifting its base to the target height.
    The reward increases linearly with height up to the target, then remains constant.
    
    Args:
        env: The environment instance.
        target_height: Target height for the robot base.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        torch.Tensor: Height reward for each environment.
    """
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    
    # Linear reward up to target height
    height_reward = torch.clamp(current_height / target_height, 0.0, 1.0)
    
    return height_reward


def upright_posture_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining upright posture using exponential kernel.
    
    This function rewards the robot for keeping its body orientation close to upright.
    Uses exponential decay based on roll and pitch angles.
    
    Args:
        env: The environment instance.
        std: Standard deviation for the exponential kernel.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        torch.Tensor: Upright posture reward for each environment.
    """
    asset = env.scene[asset_cfg.name]
    
    # Extract roll and pitch from quaternion
    quat = asset.data.root_quat_w
    # Convert quaternion to euler angles (simplified for roll/pitch)
    # q = [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Calculate roll and pitch
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    
    # Compute orientation error (roll^2 + pitch^2)
    orientation_error = roll**2 + pitch**2
    
    # Exponential reward
    return torch.exp(-orientation_error / std**2)


def stand_up_success_reward(
    env: ManagerBasedRLEnv,
    height_thresh: float = 0.40,
    angle_thresh: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Binary reward for successful standing up.
    
    This function gives a binary reward when the robot successfully stands up,
    defined as having sufficient height and upright orientation.
    
    Args:
        env: The environment instance.
        height_thresh: Minimum height threshold for success.
        angle_thresh: Maximum angle deviation (roll/pitch) for success.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        torch.Tensor: Binary success reward for each environment.
    """
    asset = env.scene[asset_cfg.name]
    
    # Check height condition
    current_height = asset.data.root_pos_w[:, 2]
    height_ok = current_height >= height_thresh
    
    # Check orientation condition
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    
    angle_ok = (torch.abs(roll) < angle_thresh) & (torch.abs(pitch) < angle_thresh)
    
    # Combined success condition
    success = height_ok & angle_ok
    
    return success.float()


def feet_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_contacts: int = 3
) -> torch.Tensor:
    """Reward for having multiple feet in contact with ground.
    
    This function rewards the robot for having multiple feet in contact with the ground,
    which is important for stability during stand-up motion.
    
    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the contact sensor.
        min_contacts: Minimum number of feet that should be in contact.
        
    Returns:
        torch.Tensor: Contact reward for each environment.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check which feet are in contact
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_forces = torch.norm(net_forces, dim=-1).max(dim=1)[0]  # Max over history
    in_contact = contact_forces > 1.0  # Threshold for contact detection
    
    # Count number of feet in contact
    num_contacts = torch.sum(in_contact, dim=1)
    
    # Reward based on number of contacts
    reward = torch.clamp(num_contacts.float() / min_contacts, 0.0, 1.0)
    
    return reward

    return reward


def joint_target_posture_reward(
    env: ManagerBasedRLEnv,
    target_joint_pos: dict,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for moving towards target joint positions (guide to standing posture).
    
    This function provides a gentle guidance reward to encourage the robot to move
    towards the target standing posture. Uses exponential kernel for smooth rewards.
    
    Args:
        env: The environment instance.
        target_joint_pos: Dictionary mapping joint names to target positions.
        std: Standard deviation for the exponential kernel (higher = more lenient).
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        torch.Tensor: Target posture guidance reward for each environment.
    """
    asset = env.scene[asset_cfg.name]
    
    total_error = torch.zeros(env.num_envs, device=env.device)
    num_joints = 0
    
    # Calculate error for each target joint
    for joint_name, target_pos in target_joint_pos.items():
        joint_ids = asset.find_joints(joint_name)[0]
        if len(joint_ids) > 0:
            current_pos = asset.data.joint_pos[:, joint_ids[0]]
            error = (current_pos - target_pos)**2
            total_error += error
            num_joints += 1
    
    if num_joints == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Average error across joints
    avg_error = total_error / num_joints
    
    # Exponential reward (higher std = more lenient)
    reward = torch.exp(-avg_error / std**2)
    
    return reward


