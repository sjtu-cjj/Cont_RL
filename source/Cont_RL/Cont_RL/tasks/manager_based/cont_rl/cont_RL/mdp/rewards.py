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
    joint_torques = asset.data.joint_effort
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