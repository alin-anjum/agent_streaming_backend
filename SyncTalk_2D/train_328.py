# train_328.py

import argparse
import os
import torch
import time
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import random

# Project imports
from datasetsss_328 import MyDataset
from syncnet_328 import SyncNet_color
from unet_328 import Model
from utils import generate_sample_predictions

# Training system imports
from training_system.traininglogger import TrainingLogger
from training_system.losses import PerceptualLoss
from training_system.training_config import TrainingConfig, get_progressive_targets, AdaptiveWeightCalculator
from training_system.training_utils import (
    calculate_adaptive_generator_loss, set_requires_grad, save_training_state, load_training_state,
    initialize_stage2_components, update_discriminator
)

def get_args():
    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true')
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--resume_from', type=int, default=0)
    return parser.parse_args()

def train_epoch(
        net, train_dataloader, optimizer_G, optimizer_D,
        content_loss, syncnet, discriminator, criterionGAN, scaler, logger,
        epoch, config, use_syncnet, stage_name, target_percentages,
        losses_G, losses_D, 
        scheduler_G
    ):
    """Train for one epoch with adaptive weighting. This function handles PER-BATCH logic."""
    
    net.train()
    if discriminator is not None:
        discriminator.train()
    
    in_stage2 = '2' in stage_name
    weight_calculator = AdaptiveWeightCalculator(config)
    
    # Reset temporal loss buffer at batch start if needed
    if hasattr(content_loss, 'temporal_loss'):
        content_loss.temporal_loss.reset()
    
    # DEBUG: Add this
    d_update_count = 0
    
    with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1} [{stage_name}]', unit='batch') as p:
        for i, batch in enumerate(train_dataloader):
            # Handle different batch formats
            if len(batch) == 3:  # Original format without indices
                imgs, labels, audio_feat = batch
                # Generate indices based on batch position
                batch_size = imgs.shape[0]
                indices = torch.arange(i * batch_size, (i + 1) * batch_size).cuda()
            else:  # New format with indices (if you update your dataset)
                imgs, labels, audio_feat, indices = batch
                indices = indices.cuda()
            
            imgs, labels, audio_feat = imgs.cuda(), labels.cuda(), audio_feat.cuda()
            
            with autocast():
                preds = net(imgs, audio_feat)
            preds_float32 = preds.float()
            
            # --- Discriminator Update ---
            gan_weight = target_percentages.get('gan', 0.0)
            if discriminator is not None and gan_weight > 0 and i % config.discriminator_update_freq == 0:
                # DEBUG: Add this
                d_update_count += 1
                if d_update_count <= 3:  # Log first 3 updates
                    logger.log_message(f"DEBUG: D update #{d_update_count} at iter {i}")
                
                set_requires_grad(discriminator, True)
                
                d_loss_val, d_real, d_fake, d_r1 = update_discriminator(
                    discriminator, optimizer_D, criterionGAN, labels, 
                    preds_float32.detach(), scaler, logger, epoch, i, config  
                )
                
                # DEBUG: Add this
                if d_update_count <= 3:
                    logger.log_message(f"  Returned: d_loss={d_loss_val}, d_real={d_real}, d_fake={d_fake}")
                
                losses_D['real'] += d_real
                losses_D['fake'] += d_fake
                losses_D['r1'] += d_r1
                losses_D['total'] += d_loss_val
                losses_D['count'] += 1

            # --- Generator Update ---
            # FIX: Disable gradients for the discriminator before the generator's update
            if discriminator is not None:
                set_requires_grad(discriminator, False)
            
            optimizer_G.zero_grad()
            
            # Pass indices to the loss calculation
            loss_G, loss_components, weighted_components, current_weights = calculate_adaptive_generator_loss(
                preds_float32, labels, audio_feat, content_loss, syncnet, discriminator, 
                criterionGAN, use_syncnet, in_stage2, weight_calculator, target_percentages, 
                logger, indices  # Now indices is defined
            )
            
            if torch.isnan(loss_G):
                logger.log_message(f"WARNING: NaN in generator loss at epoch {epoch + 1}, iter {i}")
                optimizer_G.zero_grad()
                continue
            
            scaler.scale(loss_G).backward()
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config.gradient_clip_norm)
            scaler.step(optimizer_G)
            scaler.update()
            
            # Accumulate Generator Losses
            losses_G['gan'] += weighted_components.get('gan', 0.0)
            losses_G['l1'] += weighted_components.get('l1', 0.0)
            losses_G['perceptual'] += weighted_components.get('perceptual', 0.0)
            losses_G['sync'] += weighted_components.get('sync', 0.0)
            losses_G['skin_texture'] += weighted_components.get('skin_texture', 0.0)
            losses_G['temporal'] += weighted_components.get('temporal', 0.0)  # Add temporal loss
            losses_G['total'] += loss_G.item()
            losses_G['count'] += 1
            
            # Update Progress Bar
            current_lr = scheduler_G.get_last_lr()[0]
            avg_g_loss_so_far = losses_G['total'] / losses_G['count']
            avg_d_loss_so_far = losses_D['total'] / losses_D['count'] if losses_D['count'] > 0 else 0.0
            p.set_postfix(G_loss=f'{avg_g_loss_so_far:.3f}', D_loss=f'{avg_d_loss_so_far:.3f}', LR=f'{current_lr:.6f}')
            p.update(1)

def main():
    args = get_args()
    config = TrainingConfig() 
    
    # --- Initialization ---
    os.makedirs(args.save_dir, exist_ok=True)
    logger = TrainingLogger(log_dir=os.path.join(args.save_dir, "logs"))
    net = Model(6, mode=args.asr).cuda()
    
    syncnet = None
    if args.use_syncnet:
        if not args.syncnet_checkpoint:
            raise ValueError("Using syncnet requires syncnet_checkpoint")
        syncnet = SyncNet_color(args.asr).eval().cuda()
        syncnet.load_state_dict(torch.load(args.syncnet_checkpoint))
    
    dataset = MyDataset(args.dataset_dir, args.asr)
    train_dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, 
                                 drop_last=True, num_workers=16, pin_memory=True)
    
    # --- Initialize Perceptual and Temporal Losses ---
    from training_system.losses import MultiScaleTemporalLoss
    
    # Base perceptual loss
    base_perceptual_loss = PerceptualLoss(torch.nn.MSELoss())
    
    # Create temporal loss with config
    temporal_loss = MultiScaleTemporalLoss(
        perceptual_loss_fn=base_perceptual_loss,
        window_size=config.temporal_config['window_size'],
        inner_crop_percent=config.temporal_config['inner_crop_percent'],
        inner_weight=config.temporal_config['inner_weight'],
        full_weight=config.temporal_config['full_weight'],
        decay_factor=config.temporal_config['decay_factor']
    )
    
    # Create wrapper to hold both perceptual and temporal losses
    class CombinedPerceptualLoss:
        def __init__(self, perceptual, temporal):
            self.perceptual = perceptual
            self.temporal_loss = temporal
            
        def get_loss(self, fake, real):
            # This method is used for standard perceptual loss
            return self.perceptual.get_loss(fake, real)
    
    content_loss = CombinedPerceptualLoss(base_perceptual_loss, temporal_loss)
    
    # Log temporal loss configuration
    logger.log_message("\nTemporal Loss Configuration:")
    logger.log_message(f"  Window Size: {config.temporal_config['window_size']} frames")
    logger.log_message(f"  Inner Crop: {config.temporal_config['inner_crop_percent']*100:.0f}% (mouth focus)")
    logger.log_message(f"  Inner Weight: {config.temporal_config['inner_weight']}x")
    logger.log_message(f"  Decay Factor: {config.temporal_config['decay_factor']}")
    
    scaler = GradScaler()
    
    # Initialize optimizer
    optimizer_G = optim.Adam(net.parameters(), lr=args.lr)
    
    # Default values
    discriminator, criterionGAN, optimizer_D, scheduler_D = None, None, None, None
    start_epoch = 0
    
    # Resume logic FIRST (before creating schedulers)
    if args.resume_from > 0:
        logger.log_message(f"Resuming training from epoch {args.resume_from}")
        checkpoint_path = os.path.join(args.save_dir, f'{args.resume_from}.pth')
        if os.path.exists(checkpoint_path):
            net.load_state_dict(torch.load(checkpoint_path))
            training_state = load_training_state(args.save_dir, args.resume_from, logger)
            if training_state:
                start_epoch = training_state['epoch']
                # Try to load optimizer state if available
                if 'optimizer_G' in training_state:
                    try:
                        optimizer_G.load_state_dict(training_state['optimizer_G'])
                        logger.log_message("Loaded optimizer state")
                    except:
                        logger.log_message("Could not load optimizer state, using fresh optimizer")
                logger.log_message(f"Successfully resumed from epoch {start_epoch}")
            else:
                logger.log_message(f"Warning: No training state found. Starting with fresh optimizers.")
                start_epoch = args.resume_from
        else:
            logger.log_message(f"Warning: Resume checkpoint not found at {checkpoint_path}. Starting from scratch.")
    
    # Create scheduler - handle the initial_lr issue
    if start_epoch > 0 and not any('initial_lr' in group for group in optimizer_G.param_groups):
        # Manually add initial_lr if resuming with fresh optimizer
        for group in optimizer_G.param_groups:
            group['initial_lr'] = group['lr']
    
    # Now create scheduler
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=args.epochs, last_epoch=start_epoch-1)
    
    # If we couldn't load optimizer state, manually step scheduler to correct epoch
    if start_epoch > 0 and 'optimizer_G' not in locals():
        logger.log_message(f"Fast-forwarding scheduler to epoch {start_epoch}")
        for _ in range(start_epoch):
            scheduler_G.step()

    # Check if GAN is needed in ANY stage
    gan_needed = (
        config.stage1_targets.get('gan', 0) > 0 or
        config.stage2_targets_start.get('gan', 0) > 0 or
        config.stage2_targets_end.get('gan', 0) > 0
    )
    
    if gan_needed:
        logger.log_message("\nGAN loss detected in config - Initializing discriminator...")
        logger.log_message(f"  Stage 1 GAN: {config.stage1_targets.get('gan', 0)}%")
        logger.log_message(f"  Stage 2 Start GAN: {config.stage2_targets_start.get('gan', 0)}%")
        logger.log_message(f"  Stage 2 End GAN: {config.stage2_targets_end.get('gan', 0)}%")
        logger.log_message(f"  Discriminator update frequency: every {config.discriminator_update_freq} iterations\n")
        
        discriminator, criterionGAN, optimizer_D, scheduler_D = initialize_stage2_components(
            args.lr, start_epoch, config.stage1_epochs, logger, args.epochs
        )
        
        if start_epoch > 0:
            logger.log_message(f"Fast-forwarding discriminator scheduler to epoch {start_epoch}")
            for _ in range(start_epoch):
                scheduler_D.step()

        # Fix scheduler for discriminator too
        if scheduler_D is not None:
            scheduler_D.last_epoch = start_epoch - 1

    logger.log_message("\nStarting Two-Stage Training:")
    logger.log_message(f"  Stage 1 (epochs 1-{config.stage1_epochs}): Basic reconstruction")
    logger.log_message(f"  Stage 2 (epochs {config.stage1_epochs+1}-{args.epochs}): Sync-focused training")
    if start_epoch > 0:
        logger.log_message(f"  Resuming from epoch: {start_epoch + 1}")
    
    # --- MAIN TRAINING LOOP ---
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        stage_name = 'Stage 1' if epoch < config.stage1_epochs else 'Stage 2'
        target_percentages = get_progressive_targets(epoch, config.stage1_epochs, config)
        
        target_str = ", ".join([f"{k}: {v:.1f}%" for k, v in target_percentages.items() if v > 0])
        logger.log_message(f"\n--- Starting Epoch {epoch + 1} ({stage_name}) ---")
        logger.log_message(f"Target Percentages: {target_str}")
        
        losses_G, losses_D = logger.reset_losses()
        
        # Reset temporal loss buffer at epoch start
        if hasattr(content_loss, 'temporal_loss'):
            content_loss.temporal_loss.reset()
            logger.log_message("Reset temporal loss buffer for new epoch")
            
        train_epoch(
            net=net, train_dataloader=train_dataloader, 
            optimizer_G=optimizer_G, optimizer_D=optimizer_D, 
            content_loss=content_loss, syncnet=syncnet, 
            discriminator=discriminator, criterionGAN=criterionGAN,
            scaler=scaler, logger=logger, epoch=epoch, config=config, 
            use_syncnet=args.use_syncnet, stage_name=stage_name, 
            target_percentages=target_percentages,
            losses_G=losses_G, losses_D=losses_D, scheduler_G=scheduler_G 
        )
        
        logger.log_epoch(
            epoch=epoch + 1, losses_G=losses_G, losses_D=losses_D,
            scheduler_G=scheduler_G, epoch_start_time=epoch_start_time, stage_name=stage_name
        )
    
        # Step schedulers
        if scheduler_G is not None:
            scheduler_G.step()
        if scheduler_D is not None:
            scheduler_D.step()
        
        # Save checkpoints and state periodically
        # if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
        model_path = os.path.join(args.save_dir, f'{epoch + 1}.pth')
        torch.save(net.state_dict(), model_path)
        
        # Save discriminator if it exists
        if discriminator is not None:
            disc_path = os.path.join(args.save_dir, f'discriminator_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), disc_path)
        
        state_to_save = {
            'epoch': epoch + 1, 
            'stage': stage_name, 
            'target_percentages': target_percentages,
            'has_discriminator': discriminator is not None
        }
        
        state_path = save_training_state(
            save_dir=args.save_dir, epoch=epoch + 1, optimizer_G=optimizer_G, 
            scheduler_G=scheduler_G, state_dict=state_to_save,
            optimizer_D=optimizer_D, scheduler_D=scheduler_D, discriminator=discriminator
        )
        logger.log_message(f"Saved checkpoint to {model_path} and state to {state_path}")
        
        # Generate visual samples if requested
        if args.see_res:
            generate_sample_predictions(
                net=net, dataset=dataset, epoch=epoch + 1, logger=logger,
                num_samples=3, out_dir=os.path.join(args.save_dir, "predictions")
            )

if __name__ == '__main__':
    main()