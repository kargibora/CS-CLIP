"""
Modular optimizer and scheduler factory for flexible training configuration.
Supports modern fine-tuning techniques including layer-wise learning rate decay.
"""
import torch
import logging
import re  # Used for layer depth extraction from parameter names
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR, StepLR
import torch.distributed as dist
from utils.dist import get_world_size, is_main_process


def _create_parameter_groups(model, optimizer_config):
    """
    Create parameter groups with different learning rates for different model components.
    
    Args:
        model: The model to optimize
        optimizer_config: Config object with layerwise settings
        
    Returns:
        list: List of parameter group dictionaries
    """
    # Check if layer-wise LR is enabled
    layerwise_config = getattr(optimizer_config, 'layerwise', None)
    if layerwise_config is None:
        # Standard single parameter group
        return [{'params': model.parameters()}]
    
    # Extract layer-wise configuration
    enabled = layerwise_config.get('enabled', False)
    if not enabled:
        return [{'params': model.parameters()}]
    
    # Layer-wise decay configuration
    decay_rate = layerwise_config.get('decay_rate', 0.8)  # Decay factor per layer
    decay_type = layerwise_config.get('decay_type', 'exponential')  # 'exponential' or 'linear'
    
    # Component-specific learning rates
    component_lrs = layerwise_config.get('component_lrs', {})
    
    # Special parameter treatment
    no_decay_patterns = layerwise_config.get('no_decay_patterns', ['bias', 'LayerNorm', 'layer_norm'])
    freeze_patterns = layerwise_config.get('freeze_patterns', [])
    
    base_lr = getattr(optimizer_config, 'learning_rate', 0.01)
    
    # Group parameters by component and layer
    param_groups = []
    frozen_params = set()
    
    # Helper function to get layer depth for a parameter
    def get_layer_depth(name):
        """Extract layer number from parameter name"""
        # For transformer layers: model.transformer.resblocks.0.xxx -> layer 0
        if 'resblocks' in name:
            match = re.search(r'resblocks\.(\d+)', name)
            if match:
                return int(match.group(1))
        
        # For vision transformer: model.visual.transformer.resblocks.0.xxx -> layer 0  
        if 'visual' in name and 'resblocks' in name:
            match = re.search(r'resblocks\.(\d+)', name)
            if match:
                return int(match.group(1))
        
        # For other layer patterns (add more as needed)
        layer_patterns = [
            r'layer\.(\d+)',
            r'layers\.(\d+)', 
            r'block\.(\d+)',
            r'blocks\.(\d+)'
        ]
        
        for pattern in layer_patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        
        return 0  # Default to layer 0 for embeddings, heads, etc.
    
    # Helper function to calculate layer-wise learning rate
    def calculate_layer_lr(layer_depth, total_layers, base_lr, decay_rate, decay_type):
        """Calculate learning rate for a specific layer"""
        if decay_type == 'exponential':
            # Higher layers get lower learning rates
            lr_multiplier = decay_rate ** (total_layers - layer_depth - 1)
        elif decay_type == 'linear':
            # Linear decay from 1.0 to decay_rate
            lr_multiplier = 1.0 - (1.0 - decay_rate) * (total_layers - layer_depth - 1) / max(1, total_layers - 1)
        else:
            lr_multiplier = 1.0
        
        return base_lr * lr_multiplier
    
    # Get total number of layers for normalization
    max_layer_depth = 0
    for name, param in model.named_parameters():
        depth = get_layer_depth(name)
        max_layer_depth = max(max_layer_depth, depth)
    
    total_layers = max_layer_depth + 1
    
    # Group parameters by component and layer
    component_layer_groups = {}
    
    for name, param in model.named_parameters():
        # Check if parameter should be frozen
        should_freeze = any(pattern in name for pattern in freeze_patterns)
        if should_freeze:
            param.requires_grad = False
            frozen_params.add(name)
            continue
        
        # Determine component (visual, text, alignment, etc.)
        component = 'other'
        if 'visual' in name:
            component = 'visual'
        elif any(text_key in name for text_key in ['transformer', 'token_embedding', 'positional_embedding', 'ln_final', 'text_projection']):
            component = 'text'
        elif any(align_key in name for align_key in ['align_head', 'alignment', 'logit_scale']):
            component = 'alignment'
        
        # Get layer depth
        layer_depth = get_layer_depth(name)
        
        # Create group key
        group_key = f"{component}_layer_{layer_depth}"
        
        if group_key not in component_layer_groups:
            component_layer_groups[group_key] = {
                'params': [],
                'component': component,
                'layer_depth': layer_depth,
                'param_names': []
            }
        
        component_layer_groups[group_key]['params'].append(param)
        component_layer_groups[group_key]['param_names'].append(name)
    
    # Create parameter groups with appropriate learning rates
    for group_key, group_info in component_layer_groups.items():
        component = group_info['component']
        layer_depth = group_info['layer_depth']
        
        # Calculate base learning rate for this layer
        layer_lr = calculate_layer_lr(layer_depth, total_layers, base_lr, decay_rate, decay_type)
        
        # Apply component-specific multiplier
        if component in component_lrs:
            layer_lr *= component_lrs[component]
        
        # Split params by weight decay eligibility
        decay_params = []
        no_decay_params = []
        
        for param, name in zip(group_info['params'], group_info['param_names']):
            if any(pattern in name for pattern in no_decay_patterns):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Add parameter groups
        if decay_params:
            param_groups.append({
                'params': decay_params,
                'lr': layer_lr,
                'group_name': f"{group_key}_decay"
            })
        
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params,
                'lr': layer_lr,
                'weight_decay': 0.0,
                'group_name': f"{group_key}_no_decay"
            })
    
    # Log parameter group information
    if is_main_process():
        logging.info(f"Created {len(param_groups)} parameter groups with layer-wise learning rates")
        logging.info(f"Layer-wise decay: {decay_type} with rate {decay_rate}")
        logging.info(f"Total layers detected: {total_layers}")
        
        if frozen_params:
            logging.info(f"Frozen {len(frozen_params)} parameters: {list(frozen_params)[:5]}...")
        
        # Log learning rates for each group
        for group in param_groups[:10]:  # Limit logging to first 10 groups
            group_name = group.get('group_name', 'unnamed')
            lr = group['lr']
            num_params = sum(p.numel() for p in group['params'])
            logging.info(f"  {group_name}: lr={lr:.2e}, params={num_params:,}")
    
    return param_groups


def build_optimizer(model, optimizer_config):
    """
    Build optimizer from configuration with support for layer-wise learning rates.
    
    Args:
        model: The model to optimize
        optimizer_config: Config object with optimizer settings
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer_type = getattr(optimizer_config, 'type', getattr(optimizer_config, 'optimizer_type', 'adam'))
    learning_rate = getattr(optimizer_config, 'learning_rate', 0.01)
    optimizer_kwargs = getattr(optimizer_config, 'optimizer_kwargs', {})
    
    # Scale learning rate for distributed training if specified
    scaled_lr = _scale_learning_rate(learning_rate, optimizer_config)
    
    # Create parameter groups (handles layer-wise LR automatically)
    # Update learning rate in config for parameter group creation
    original_lr = getattr(optimizer_config, 'learning_rate', 0.01)
    optimizer_config.learning_rate = scaled_lr
    param_groups = _create_parameter_groups(model, optimizer_config)
    optimizer_config.learning_rate = original_lr  # Restore original
    
    # Check if layerwise is enabled
    layerwise_enabled = getattr(getattr(optimizer_config, 'layerwise', {}), 'enabled', False)
    
    if layerwise_enabled:
        # Layer-wise mode: exclude per-group parameters
        filtered_kwargs = {k: v for k, v in optimizer_kwargs.items() 
                          if k not in ['weight_decay', 'lr', 'learning_rate']}
        
        # Add default weight_decay only if not specified in any param group
        if 'weight_decay' in optimizer_kwargs and not any('weight_decay' in group for group in param_groups):
            filtered_kwargs['weight_decay'] = optimizer_kwargs['weight_decay']
    else:
        # Standard mode: only exclude lr/learning_rate (they're handled separately)
        filtered_kwargs = {k: v for k, v in optimizer_kwargs.items() 
                          if k not in ['lr', 'learning_rate']}
        
        # Add learning rate for single parameter group
        filtered_kwargs['lr'] = scaled_lr
    
    # Build optimizer based on type
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups, **filtered_kwargs)
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, **filtered_kwargs)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups, **filtered_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    if is_main_process():
        layerwise_enabled = getattr(getattr(optimizer_config, 'layerwise', {}), 'enabled', False)
        mode = "layer-wise" if layerwise_enabled else "standard"
        logging.info(f"Built {optimizer_type} optimizer ({mode}) with base LR={scaled_lr}")
        logging.info(f"Parameter groups: {len(param_groups)}")
        logging.info(f"Optimizer kwargs: {filtered_kwargs}")
    
    return optimizer


def build_scheduler(optimizer, optimizer_config, num_epochs, train_dataloader=None):
    """
    Build learning rate scheduler from configuration.
    
    Args:
        optimizer: The optimizer to schedule
        optimizer_config: Config object with scheduler settings
        num_epochs: Total number of training epochs
        train_dataloader: Training dataloader (for step-based scheduling)
        
    Returns:
        torch.optim.lr_scheduler or None: Configured scheduler
    """
    scheduler_type = getattr(optimizer_config, 'scheduler', 'none')
    
    if scheduler_type.lower() == 'none':
        return None
        
    scheduler_kwargs = getattr(optimizer_config, 'scheduler_kwargs', {})
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    
    # Determine if we're doing epoch-based or step-based scheduling
    step_based = scheduler_kwargs.get('step_based', False)  # Default to epoch-based
    
    if step_based and train_dataloader is not None:
        steps_per_epoch = len(train_dataloader)
        num_training_steps = num_epochs * steps_per_epoch
        unit_name = "steps"
    else:
        num_training_steps = num_epochs
        unit_name = "epochs"
    
    # Calculate warmup duration - support both warmup_epochs and warmup_steps
    warmup_epochs = scheduler_kwargs.get('warmup_epochs', None)
    warmup_steps = scheduler_kwargs.get('warmup_steps', None)
    warmup_ratio = scheduler_kwargs.get('warmup_ratio', 0.05)
    
    if warmup_epochs is not None:
        if step_based and train_dataloader is not None:
            num_warmup_steps = warmup_epochs * steps_per_epoch
        else:
            num_warmup_steps = warmup_epochs
    elif warmup_steps is not None:
        num_warmup_steps = warmup_steps
    else:
        num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    # Create warmup scheduler with proper start LR handling
    if num_warmup_steps > 0:
        warmup_start_lr_ratio = scheduler_kwargs.get('warmup_start_lr_ratio', 0.01)
        
        def warmup_lambda(step):
            if step < num_warmup_steps:
                # Linear warmup from warmup_start_lr_ratio to 1.0
                progress = float(step) / float(max(1, num_warmup_steps))
                return warmup_start_lr_ratio + (1.0 - warmup_start_lr_ratio) * progress
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Create main scheduler based on type
    main_steps = max(1, num_training_steps - num_warmup_steps)
    
    if scheduler_type.lower() == 'linear':
        def linear_decay_lambda(step):
            if step < main_steps:
                return 1.0 - float(step) / float(main_steps)
            return 0.0
        
        main_scheduler = LambdaLR(optimizer, lr_lambda=linear_decay_lambda)
        
    elif scheduler_type.lower() == 'cosine':
        cosine_t_max = scheduler_kwargs.get('cosine_t_max', None)
        # If cosine_t_max is None or not specified, use main_steps
        if cosine_t_max is None:
            cosine_t_max = main_steps
        
        # Handle both absolute and relative minimum LR
        if 'min_lr_ratio' in scheduler_kwargs:
            base_lr = optimizer.param_groups[0]['lr']
            cosine_eta_min = base_lr * scheduler_kwargs['min_lr_ratio']
        else:
            cosine_eta_min = scheduler_kwargs.get('cosine_eta_min', 0)
        
        main_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cosine_t_max, 
            eta_min=cosine_eta_min
        )
        
    elif scheduler_type.lower() == 'step':
        step_size = scheduler_kwargs.get('step_size', 10)
        gamma = scheduler_kwargs.get('gamma', 0.1)
        
        main_scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
    elif scheduler_type.lower() == 'constant':
        # Constant LR after warmup
        main_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    # Combine warmup and main scheduler
    if num_warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps]
        )
        
        if is_main_process():
            logging.info(f"Built {scheduler_type} scheduler with {num_warmup_steps} warmup {unit_name}")
    else:
        scheduler = main_scheduler
        if is_main_process():
            logging.info(f"Built {scheduler_type} scheduler (no warmup)")
    
    if is_main_process():
        logging.info(f"Total training {unit_name}: {num_training_steps}")
        logging.info(f"Scheduler kwargs: {scheduler_kwargs}")
    
    return scheduler


def _scale_learning_rate(base_lr, optimizer_config):
    """
    Scale learning rate for distributed training.
    
    Args:
        base_lr: Base learning rate
        optimizer_config: Config with lr_scaling setting
        
    Returns:
        float: Scaled learning rate
    """
    lr_scaling = getattr(optimizer_config, 'lr_scaling', 'none')
    
    if not (hasattr(optimizer_config, 'distributed') and getattr(optimizer_config, 'distributed', False)):
        return base_lr
        
    if not (dist.is_available() and dist.is_initialized()):
        return base_lr
    
    world_size = get_world_size()
    
    if lr_scaling.lower() == 'linear':
        scaled_lr = base_lr * world_size
    elif lr_scaling.lower() == 'sqrt':
        scaled_lr = base_lr * (world_size ** 0.5)
    else:
        scaled_lr = base_lr
    
    if is_main_process() and lr_scaling != 'none':
        logging.info(f"LR scaling: {lr_scaling}, world_size={world_size}, base_lr={base_lr}, scaled_lr={scaled_lr}")
    
    return scaled_lr


def make_optimizer_and_scheduler(model, config, train_dataloader=None):
    """
    Convenience function to build both optimizer and scheduler.
    
    Args:
        model: The model to optimize
        config: Configuration object (should have optimizer config)
        num_epochs: Number of training epochs
        train_dataloader: Training dataloader
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    num_epochs = config.epochs
    # Handle different config structures
    if hasattr(config, 'optimizer'):
        optimizer_config = config.optimizer
    else:
        # Assume config itself contains optimizer settings
        optimizer_config = config
    
    optimizer = build_optimizer(model, optimizer_config)
    scheduler = build_scheduler(optimizer, optimizer_config, num_epochs, train_dataloader)
    
    return optimizer, scheduler
