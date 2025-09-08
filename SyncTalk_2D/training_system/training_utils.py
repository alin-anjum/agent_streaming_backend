import torch
import os
from .multiscale_discriminator import MultiscaleDiscriminator
from .losses import LSGANLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torch.nn as nn

def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for networks"""
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def save_training_state(save_dir, epoch, optimizer_G, scheduler_G, 
                        state_dict=None,
                        optimizer_D=None, scheduler_D=None, discriminator=None):
    """Saves minimal training state to save disk space."""
    
    # Create checkpoint directory structure
    checkpoint_dir = os.path.join(save_dir, "checkpoints", "training_states")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save only essential information - no optimizer states!
    save_obj = {
        'epoch': epoch,
        'scheduler_last_epoch': scheduler_G.last_epoch,
        'current_lr': scheduler_G.get_last_lr()[0],
    }
    
    # Add any additional state info passed from main loop
    if state_dict is not None:
        save_obj.update(state_dict)
    
    # For Stage 2, save discriminator scheduler info (but not full state)
    if scheduler_D is not None:
        save_obj['scheduler_D_last_epoch'] = scheduler_D.last_epoch
    
    # Save to checkpoint directory with smaller filename
    state_path = os.path.join(checkpoint_dir, f'state_{epoch}.tar')
    
    try:
        torch.save(save_obj, state_path)
        
        # Optional: Keep only recent states to save space
        cleanup_old_states(checkpoint_dir, keep_last_n=3)
        
    except Exception as e:
        print(f"Error saving training state: {e}")
        return None
        
    return state_path

def load_training_state(save_dir, resume_epoch, logger):
    """Load minimal training state from checkpoint"""
    checkpoint_dir = os.path.join(save_dir, "checkpoints", "training_states")
    training_state_path = os.path.join(checkpoint_dir, f'state_{resume_epoch}.tar')
    
    if os.path.exists(training_state_path):
        training_state = torch.load(training_state_path)
        logger.log_message(f"Loaded training state from {training_state_path}")
        
        # Log what we loaded (since we're not loading optimizer states)
        logger.log_message(f"  - Epoch: {training_state.get('epoch', resume_epoch)}")
        logger.log_message(f"  - Last LR: {training_state.get('current_lr', 'unknown')}")
        logger.log_message("  - Note: Optimizer states not saved (using fresh momentum)")
        
        return training_state
    else:
        logger.log_message(f"Warning: Training state file not found at {training_state_path}")
        return None

def cleanup_old_states(checkpoint_dir, keep_last_n=3):
    """Remove old state files, keeping only the last N."""
    import re
    
    state_files = []
    for f in os.listdir(checkpoint_dir):
        match = re.match(r'state_(\d+)\.tar', f)
        if match:
            epoch = int(match.group(1))
            state_files.append((epoch, f))
    
    # Sort by epoch
    state_files.sort(key=lambda x: x[0])
    
    # Keep milestones (every 20 epochs) and last N
    files_to_keep = set()
    for epoch, filename in state_files:
        if epoch % 20 == 0:  # Keep milestone epochs
            files_to_keep.add(filename)
    
    for epoch, filename in state_files[-keep_last_n:]:
        files_to_keep.add(filename)
    
    # Delete others
    for epoch, filename in state_files:
        if filename not in files_to_keep:
            try:
                os.remove(os.path.join(checkpoint_dir, filename))
            except:
                pass

def initialize_stage2_components(lr, current_epoch, stage1_epochs, logger, total_epochs=100):
    """Initialize Stage 2 discriminator and optimizers"""
    logger.log_message("\n" + "="*80)
    logger.log_message("=== Entering Stage 2: High-Frequency Detail Enhancement ===")
    logger.log_message("="*80)
    
    discriminator = MultiscaleDiscriminator(input_nc=3, ndf=64, n_layers=4, num_D=3).cuda()
    criterionGAN = LSGANLoss().cuda()
    
    # Test discriminator
    try:
        with torch.no_grad():
            test_input = torch.randn(1, 3, 328, 328).cuda()
            test_output = discriminator(test_input)
            logger.log_message(f"Discriminator test successful, shapes: {[x.shape for x in test_output]}")
    except Exception as e:
        logger.log_message(f"ERROR: Discriminator test failed: {str(e)}")
    
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr/10, betas=(0.5, 0.999))
    
    # FIX: Always use full epoch range for discriminator scheduler
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=total_epochs, last_epoch=-1)
    
    logger.log_message(f"Discriminator initialized at epoch {current_epoch}")
    logger.log_message(f"Discriminator scheduler T_max={total_epochs}")
    
    return discriminator, criterionGAN, optimizer_D, scheduler_D

def update_discriminator(discriminator, optimizer_D, criterionGAN, labels, preds_float32, scaler, logger, epoch, iteration, config=None):
    """
    Handle discriminator update for a multi-scale discriminator.
    """
    if discriminator is None or optimizer_D is None or criterionGAN is None:
        return 0.0, 0.0, 0.0, 0.0

    set_requires_grad(discriminator, True)
    optimizer_D.zero_grad()
    
    # --- Real Image Pass ---
    pred_real_list = discriminator(labels)
    
    # --- Fake Image Pass ---
    pred_fake_list = discriminator(preds_float32.detach())

    # --- Calculate Loss ---
    loss_D_real = 0
    loss_D_fake = 0
    real_label = 0.9 # Label smoothing

    for pred_real in pred_real_list:
        target_tensor_real = torch.full_like(pred_real, real_label)
        loss_D_real += criterionGAN(pred_real, target_tensor_real)
        
    for pred_fake in pred_fake_list:
        target_tensor_fake = torch.full_like(pred_fake, 0.0)
        loss_D_fake += criterionGAN(pred_fake, target_tensor_fake)

    # FIX: Use global epoch warmup instead of stage-specific
    if config is not None and hasattr(config, 'discriminator_warmup_epochs'):
        warmup_factor = min(epoch / config.discriminator_warmup_epochs, 1.0)
    else:
        warmup_factor = 1.0

    # Combine the losses
    loss_D = (loss_D_real + loss_D_fake) * 0.5 * warmup_factor
    
    if torch.isnan(loss_D):
        logger.log_message(f"WARNING: NaN in discriminator loss at epoch {epoch}, iter {iteration}")
        optimizer_D.zero_grad()
        return 0.0, 0.0, 0.0, 0.0
    
    # Backward pass and optimizer step
    scaler.scale(loss_D).backward()
    scaler.unscale_(optimizer_D)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
    scaler.step(optimizer_D)
    scaler.update()
    
    return loss_D.item(), loss_D_real.item(), loss_D_fake.item(), 0.0

def calculate_adaptive_generator_loss(preds_float32, labels, audio_feat, content_loss, syncnet, 
                                    discriminator, criterionGAN, use_syncnet, in_stage2, 
                                    weight_calculator, target_percentages, logger, indices=None):
    """
    Calculate generator loss with EXACT target percentages as specified in config.
    Now includes temporal regularization loss.
    """
    from .losses import cosine_loss, skin_texture_loss
    
    criterionL1 = torch.nn.L1Loss()
    
    # --- 1. Calculate all raw losses ---
    loss_tensors = {}
    loss_tensors['pixel'] = criterionL1(preds_float32, labels)
    loss_tensors['perceptual'] = content_loss.get_loss(preds_float32, labels)
    loss_tensors['skin_texture'] = skin_texture_loss(preds_float32, labels)
    
    # Add temporal loss if available
    if hasattr(content_loss, 'temporal_loss') and indices is not None:
        loss_tensors['temporal'] = content_loss.temporal_loss.get_loss(preds_float32, indices)
    else:
        loss_tensors['temporal'] = torch.tensor(0.0, device=preds_float32.device)

    if use_syncnet:
        y = torch.ones([preds_float32.shape[0], 1]).float().cuda()
        a, v = syncnet(preds_float32, audio_feat)
        loss_tensors['sync'] = cosine_loss(a, v, y)
    else:
        loss_tensors['sync'] = torch.tensor(0.0, device=preds_float32.device)

    if discriminator is not None and target_percentages.get('gan', 0) > 0:
        pred_fake_list = discriminator(preds_float32)
        loss_G_gan = 0
        for pred_fake in pred_fake_list:
            target_tensor = torch.ones_like(pred_fake)
            loss_G_gan += criterionGAN(pred_fake, target_tensor)
        loss_tensors['gan'] = loss_G_gan
    else:
        loss_tensors['gan'] = torch.tensor(0.0, device=preds_float32.device)

    # --- 2. Calculate weights to achieve EXACT target percentages ---
    # First, get raw loss values
    raw_losses = {name: tensor.detach().item() for name, tensor in loss_tensors.items()}
    
    weights = {}
    
    # Simple approach: Pick one loss as reference (e.g., l1) with weight 1.0
    reference_loss = 'pixel'  # Using pixel/l1 as reference
    reference_percentage = target_percentages.get('l1', target_percentages.get('pixel', 1.0))
    
    if reference_percentage > 0 and raw_losses[reference_loss] > 1e-8:
        # Set reference weight
        weights[reference_loss] = 1.0
        
        # Calculate what the total should be
        # If pixel loss is 0.1 with weight 1.0 and should be 78%, then total should be 0.1/0.78
        reference_contribution = raw_losses[reference_loss] * weights[reference_loss]
        total_target = reference_contribution / (reference_percentage / 100.0)
        
        # Now calculate weights for other losses
        for name, raw_value in raw_losses.items():
            if name != reference_loss:
                # Handle naming consistency (pixel vs l1, temporal vs temporal_regularization)
                lookup_name = name.replace('pixel', 'l1')
                target_pct = target_percentages.get(lookup_name, 0.0)
                
                if target_pct > 0 and raw_value > 1e-8:
                    # What should this loss contribute?
                    target_contribution = total_target * (target_pct / 100.0)
                    # What weight achieves this?
                    weights[name] = target_contribution / raw_value
                else:
                    weights[name] = 0.0
    else:
        # Fallback: equal weights
        weights = {name: 1.0 for name in raw_losses.keys()}

    # --- 3. Apply weights and calculate final loss ---
    total_loss = torch.tensor(0.0, device=preds_float32.device)
    loss_components = {}
    weighted_components = {}

    for name, tensor in loss_tensors.items():
        weight = weights.get(name, 0.0)
        total_loss += tensor * weight
        
        raw_value = tensor.item()
        weighted_value = raw_value * weight
        loss_components[name] = raw_value
        weighted_components[name] = weighted_value

    # Rename for consistency
    if 'pixel' in loss_components:
        loss_components['l1'] = loss_components.pop('pixel')
        weighted_components['l1'] = weighted_components.pop('pixel')
    
    # Verify percentages (for logging)
    total_weighted = sum(weighted_components.values())
    if total_weighted > 0:
        actual_percentages = {k: (v/total_weighted)*100 for k, v in weighted_components.items()}
        # Log if percentages are off by more than 1%
        for k, v in actual_percentages.items():
            target = target_percentages.get(k, 0)
            if abs(v - target) > 1.0 and target > 0:
                logger.log_message(f"WARNING: {k} loss is {v:.1f}% instead of target {target:.1f}%")
    
    return total_loss, loss_components, weighted_components, weights