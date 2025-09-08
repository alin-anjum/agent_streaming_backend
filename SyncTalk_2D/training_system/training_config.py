#training_config.py

from dataclasses import dataclass

from dataclasses import field

@dataclass
class TrainingConfig:
    stage1_epochs: int = 25
    
    # Stage 1: Focus on basic reconstruction and structure
    stage1_targets: dict = field(default_factory=lambda: {
        'l1': 65.0,           # Pixel-wise reconstruction - ensures basic image fidelity and structure
        'perceptual': 15.0,   # Perceptual similarity - maintains visual quality and sharpness
        'sync': 0.0,          # Audio-visual synchronization - lip sync accuracy with audio
        'gan': 10.0,           # Adversarial realism - photorealistic texture and detail enhancement
        'skin_texture': 0.0,  # Fine detail preservation - skin pores, texture, and surface details
        'temporal': 10.0,     # Temporal consistency
    })
    
    # Stage 2 Start: Transition to sync-focused training
    stage2_targets_start: dict = field(default_factory=lambda: {
        'l1': 40.0,           # Pixel-wise reconstruction - maintains structure while allowing flexibility
        'perceptual': 10.0,   # Perceptual similarity - ensures overall visual coherence
        'sync': 30.0,         # Audio-visual synchronization - begins emphasizing lip sync
        'gan': 0.0,           # Adversarial realism - introduces photorealistic refinement
        'skin_texture': 5.0,  # Fine detail preservation - maintains skin quality
        'temporal': 15.0,     # Temporal consistency
    })
    
    # Stage 2 End: Maximum sync emphasis with quality preservation
    stage2_targets_end: dict = field(default_factory=lambda: {
        'l1': 15.0,           # Pixel-wise reconstruction - minimal, allows sync-driven changes
        'perceptual': 8.0,   # Perceptual similarity - basic quality maintenance
        'sync': 60.0,         # Audio-visual synchronization - primary objective for accurate lip sync
        'gan': 0.0,           # Adversarial realism - subtle photorealistic enhancement
        'skin_texture': 5.0,  # Fine detail preservation - consistent skin texture quality
        'temporal': 12.0,     # Temporal consistency
    })
    
    temporal_config: dict = field(default_factory=lambda: {
        'window_size': 5,
        'inner_crop_percent': 0.3,  # 40% center crop for mouth
        'inner_weight': 2.5,        # 2.5x weight on mouth region
        'full_weight': 1.0,
        'decay_factor': 0.85
    })
    
    # Training hyperparameters
    g_lr_ratio: float = 1.0               # Learning rate multiplier for generator
    d_lr_ratio: float = 0.1               # Learning rate multiplier for discriminator (relative to generator)
    discriminator_update_freq: int = 10   # Update discriminator every N generator updates
    gradient_clip_norm: float = 1.0       # Maximum gradient norm for stability
    weight_smoothing: float = 0.9         # EMA factor for smooth weight transitions (0-1, higher = smoother)
    discriminator_warmup_epochs: int = 5  # Gradual discriminator introduction

class AdaptiveWeightCalculator:
    def __init__(self, config):
        self.config = config
        self.logger = None  # Set this from train_epoch if you want logging
        
    def update_weights(self, raw_losses, target_percentages, is_first_batch=False):
        """
        Calculate weights to achieve EXACT target loss percentages.
        No smoothing, no adaptation - just precise math.
        """
        # Filter out losses with 0 target percentage
        active_losses = {k: v for k, v in raw_losses.items() 
                        if target_percentages.get(k, 0) > 0 and v > 1e-8}
        
        if not active_losses:
            return {k: 0.0 for k in raw_losses.keys()}
        
        # Method: Fix one loss weight as reference and solve for others
        # Choose the loss with highest target percentage as reference
        reference_loss = max(target_percentages.items(), key=lambda x: x[1])[0]
        
        if reference_loss not in active_losses:
            # Fallback to first available loss
            reference_loss = list(active_losses.keys())[0]
        
        weights = {}
        
        # Set reference weight to 1.0
        weights[reference_loss] = 1.0
        ref_contribution = active_losses[reference_loss] * 1.0
        ref_percentage = target_percentages[reference_loss] / 100.0
        
        # Calculate total based on reference
        # If ref loss is 0.1 with weight 1.0 and should be 78%, total should be 0.1/0.78
        total_weighted = ref_contribution / ref_percentage
        
        # Calculate weights for other losses
        for loss_name, raw_value in active_losses.items():
            if loss_name != reference_loss:
                target_pct = target_percentages.get(loss_name, 0) / 100.0
                target_contribution = total_weighted * target_pct
                weights[loss_name] = target_contribution / raw_value
        
        # Set weight to 0 for inactive losses
        for loss_name in raw_losses.keys():
            if loss_name not in weights:
                weights[loss_name] = 0.0
        
        # Verify the math (optional - remove in production for speed)
        if self.logger and is_first_batch:
            weighted = {k: raw_losses[k] * weights[k] for k in weights.keys()}
            total = sum(weighted.values())
            actual_pcts = {k: (v/total)*100 for k, v in weighted.items()} if total > 0 else {}
            
            self.logger.log_message("Target vs Actual percentages:")
            for k in target_percentages:
                target = target_percentages.get(k, 0)
                actual = actual_pcts.get(k, 0)
                self.logger.log_message(f"  {k}: target={target:.1f}%, actual={actual:.1f}%")
        
        return weights

def get_progressive_targets(epoch, stage1_epochs, config):
    """Get target percentages for current epoch"""
    if epoch < stage1_epochs:
        return config.stage1_targets
    else:
        # Progressive blend from start to end targets
        progress = min((epoch - stage1_epochs) / 10.0, 1.0)
        
        targets = {}
        for loss_name in config.stage2_targets_start:
            start_val = config.stage2_targets_start[loss_name]
            end_val = config.stage2_targets_end.get(loss_name, start_val)
            targets[loss_name] = start_val + (end_val - start_val) * progress
        
        return targets