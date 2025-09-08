import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lpips
from torchvision.models import AlexNet_Weights
import torch.nn as nn

class LSGANLoss(nn.Module):
    """
    Least Squares GAN loss - more stable than vanilla GAN.
    This version is simplified to accept pre-made target tensors, making it
    compatible with label smoothing and multi-scale discriminators.
    """
    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target):
        """
        Calculates the LSGAN loss.

        Args:
            prediction (Tensor): A single output tensor from a discriminator head.
            target (Tensor): The target tensor (e.g., a tensor of 0.9s for real, 0.0s for fake).

        Returns:
            Tensor: The calculated MSE loss.
        """
        # The logic is now just one line. It expects a single tensor for prediction.
        # The calling function is responsible for looping over multi-scale outputs.
        return self.loss(prediction, target)

class PerceptualLoss():
    def __init__(self, loss):
        """
        Args:
            loss: Loss function (e.g., torch.nn.MSELoss()) - kept for API compatibility
        """
        # Keep the loss parameter for API compatibility, even though LPIPS has its own loss
        self.criterion = loss
        
        # Initialize LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).cuda()
        
        # Freeze LPIPS weights since it's a pretrained perceptual loss
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
            
        # Set to eval mode to ensure consistent behavior
        self.lpips_fn.eval()
        
    def get_loss(self, fakeIm, realIm):
        """
        Calculate perceptual loss between fake and real images.
        
        Args:
            fakeIm: Predicted images tensor [B, C, H, W] in range [0, 1]
            realIm: Ground truth images tensor [B, C, H, W] in range [0, 1]
            
        Returns:
            Scalar tensor containing the perceptual loss
        """
        # LPIPS expects inputs in [-1, 1] range
        fake_scaled = fakeIm * 2 - 1
        real_scaled = realIm * 2 - 1
        
        # LPIPS returns a tensor with shape [batch_size, 1, 1, 1]
        # Take mean to get a scalar loss
        loss = self.lpips_fn(fake_scaled, real_scaled).mean() 
        
        return loss


logloss = nn.BCELoss() # Use the standard BCELoss
def sync_loss_function(audio_embedding, face_embedding):
    """
    Calculates the sync loss for the generator.
    The generator's goal is ALWAYS to be in sync, so the target 'y' is always 1.
    """
    # Calculate cosine similarity
    d = F.cosine_similarity(audio_embedding, face_embedding)
    
    # Scale the similarity from [-1, 1] to [0, 1] for BCELoss
    d_scaled = (d + 1) / 2.0
    
    # Create a target tensor of all ones, with the same batch size and device
    target = torch.ones_like(d_scaled)
    
    # Calculate the loss
    loss = logloss(d_scaled, target)
    return loss

def cosine_loss(a, v, y):
    logloss = nn.BCEWithLogitsLoss()
    d = nn.functional.cosine_similarity(a, v)
    return logloss(d.unsqueeze(1), y)

def skin_texture_loss(pred, target):
    """Specifically designed for skin texture and pores"""
    kernel_size = 5
    gaussian = torch.ones(1, 1, kernel_size, kernel_size).cuda() / (kernel_size**2)
    gaussian = gaussian.repeat(3, 1, 1, 1)
    
    pred_blur = F.conv2d(pred, gaussian, padding=kernel_size//2, groups=3)
    target_blur = F.conv2d(target, gaussian, padding=kernel_size//2, groups=3)
    
    pred_details = pred - pred_blur
    target_details = target - target_blur
    
    pred_sq = F.conv2d(pred**2, gaussian, padding=kernel_size//2, groups=3)
    pred_var = pred_sq - pred_blur**2
    
    target_sq = F.conv2d(target**2, gaussian, padding=kernel_size//2, groups=3)
    target_var = target_sq - target_blur**2
    
    detail_loss = F.l1_loss(pred_details * 3, target_details * 3)
    variance_loss = F.l1_loss(pred_var, target_var)
    
    return (detail_loss + variance_loss) * 10.0

class MultiScaleTemporalLoss:
    """
    Apply temporal loss at multiple crop scales with emphasis on inner region (mouth)
    """
    def __init__(self, perceptual_loss_fn, window_size=5, 
                 inner_crop_percent=0.5, inner_weight=2.0, full_weight=1.0,
                 weight=1.0, decay_factor=0.8):
        """
        Args:
            perceptual_loss_fn: Base perceptual loss (LPIPS, VGG, etc.)
            window_size: Number of previous frames to compare
            inner_crop_percent: Percentage of crop to use for inner region (0.5 = 50% center crop)
            inner_weight: Weight for inner (mouth) region temporal loss
            full_weight: Weight for full face temporal loss
            weight: Overall weight for total temporal loss
            decay_factor: Temporal decay for older frames
        """
        self.perceptual_loss_fn = perceptual_loss_fn
        self.window_size = window_size
        self.inner_crop_percent = inner_crop_percent
        self.inner_weight = inner_weight
        self.full_weight = full_weight
        self.weight = weight
        self.decay_factor = decay_factor
        
        # Frame buffer stores (full_frame, inner_frame, index)
        self.frame_buffer = []
        self.max_buffer_size = window_size + 1
        
        # Precompute temporal weights
        self.temporal_weights = []
        for i in range(window_size):
            self.temporal_weights.append(decay_factor ** i)
        weight_sum = sum(self.temporal_weights)
        self.temporal_weights = [w / weight_sum for w in self.temporal_weights]
        
    def reset(self):
        """Reset frame buffer at epoch start"""
        self.frame_buffer = []
        
    def _get_center_crop(self, frames):
        """Extract center crop based on percentage"""
        b, c, h, w = frames.shape
        
        # Calculate crop dimensions
        crop_h = int(h * self.inner_crop_percent)
        crop_w = int(w * self.inner_crop_percent)
        
        # Calculate center crop coordinates
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        
        return frames[:, :, start_h:end_h, start_w:end_w]
    
    def _compute_temporal_loss_for_region(self, current_frames, prev_frames_list, region_name="full"):
        """Compute temporal loss for a specific region"""
        if not prev_frames_list:
            return torch.tensor(0.0, device=current_frames.device)
        
        total_loss = torch.tensor(0.0, device=current_frames.device)
        
        for (prev_frames, weight, time_diff) in prev_frames_list:
            # Compute perceptual loss between current and previous
            perceptual_diff = self.perceptual_loss_fn.get_loss(current_frames, prev_frames)
            
            # Apply temporal weighting
            weighted_loss = perceptual_diff * weight
            total_loss += weighted_loss
            
        return total_loss
    
    def get_loss(self, preds, indices):
        """
        Calculate multi-scale temporal loss with inner crop focus
        
        Args:
            preds: Current predictions [B, C, H, W]
            indices: Frame indices for tracking temporal order [B]
        """
        device = preds.device
        batch_size = preds.shape[0]
        total_loss = torch.tensor(0.0, device=device)
        loss_count = 0
        
        # Extract inner crops for current frames
        inner_preds = self._get_center_crop(preds)
        
        # Process each sample in batch
        for b in range(batch_size):
            current_full = preds[b:b+1]
            current_inner = inner_preds[b:b+1]
            current_idx = indices[b].item()
            
            # Collect valid previous frames from buffer
            valid_full_comparisons = []
            valid_inner_comparisons = []
            
            for buffer_entry in reversed(self.frame_buffer):
                prev_full, prev_inner, prev_idx = buffer_entry
                
                # Check temporal distance
                time_diff = current_idx - prev_idx
                if 0 < time_diff <= self.window_size:
                    weight_idx = time_diff - 1
                    weight = self.temporal_weights[weight_idx]
                    
                    valid_full_comparisons.append((prev_full, weight, time_diff))
                    valid_inner_comparisons.append((prev_inner, weight, time_diff))
            
            # Calculate losses for both scales
            if valid_full_comparisons:
                # Full face temporal loss
                full_loss = self._compute_temporal_loss_for_region(
                    current_full, valid_full_comparisons, "full"
                )
                
                # Inner region temporal loss
                inner_loss = self._compute_temporal_loss_for_region(
                    current_inner, valid_inner_comparisons, "inner"
                )
                
                # Weighted combination
                sample_loss = (self.full_weight * full_loss + 
                             self.inner_weight * inner_loss)
                
                total_loss += sample_loss
                loss_count += 1
        
        # Update buffer with current frames
        for b in range(batch_size):
            frame_data = (
                preds[b:b+1].detach().clone(),
                inner_preds[b:b+1].detach().clone(),
                indices[b].item()
            )
            self.frame_buffer.append(frame_data)
        
        # Maintain buffer size
        if len(self.frame_buffer) > self.max_buffer_size * batch_size:
            self.frame_buffer = self.frame_buffer[-(self.max_buffer_size * batch_size):]
        
        # Average and apply overall weight
        if loss_count > 0:
            total_loss = total_loss / loss_count
            
        return total_loss * self.weight


# Simplified version if you just want inner crop temporal loss
class InnerCropTemporalLoss:
    """
    Temporal loss focused only on inner crop region
    """
    def __init__(self, perceptual_loss_fn, inner_crop_percent=0.5, 
                 window_size=3, weight=1.0):
        self.perceptual_loss_fn = perceptual_loss_fn
        self.inner_crop_percent = inner_crop_percent
        self.window_size = window_size
        self.weight = weight
        self.prev_inner_crops = []
        
    def reset(self):
        self.prev_inner_crops = []
        
    def _get_center_crop(self, frames):
        """Extract center crop based on percentage"""
        b, c, h, w = frames.shape
        crop_size = int(min(h, w) * self.inner_crop_percent)
        
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        
        return frames[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
    
    def get_loss(self, preds, indices=None):
        """Calculate temporal loss on inner crop only"""
        # Get inner crops
        inner_crops = self._get_center_crop(preds)
        
        loss = torch.tensor(0.0, device=preds.device)
        
        # Compare with previous frames
        if self.prev_inner_crops:
            # Just compare with most recent frame (simpler)
            prev_crop = self.prev_inner_crops[-1]
            loss = self.perceptual_loss_fn.get_loss(inner_crops, prev_crop)
        
        # Update buffer
        self.prev_inner_crops.append(inner_crops.detach().clone())
        if len(self.prev_inner_crops) > self.window_size:
            self.prev_inner_crops = self.prev_inner_crops[-self.window_size:]
        
        return loss * self.weight    