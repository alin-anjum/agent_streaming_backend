import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional, List

class GPUPreprocessor:
    """GPU-accelerated preprocessing for video frames"""
    
    def __init__(self, device: torch.device, target_size: int = 328, crop_size: int = 320):
        self.device = device
        self.target_size = target_size
        self.crop_size = crop_size
        self.border = (target_size - crop_size) // 2
        self.gpu_buffers = {'mask': self._create_mask()}
    
    def _create_mask(self) -> torch.Tensor:
        mask = torch.ones((1, 3, self.crop_size, self.crop_size), device=self.device)
        mask[:, :, 5:-5, 5:-5] = 0
        return mask
    
    def preprocess_batch_gpu(self,
                            imgs: torch.Tensor,
                            bboxes: List[Tuple[int, int, int, int]]
                            ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
        """
        Preprocesses a batch and returns the necessary components for the new pipeline.

        Returns:
            real_imgs: The inner 320x320 crop for the model.
            masked_imgs: The masked inner 320x320 crop.
            original_sizes: List of original (h, w) for final resizing.
            resized_faces_batch: The 328x328 canvas needed for high-quality post-processing.
        """
        batch_size = len(bboxes)
        if batch_size == 0:
            return torch.empty(0), torch.empty(0), [], torch.empty(0)
        
        face_crops_resized = []
        original_sizes = []

        for i, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox
            face_crop = imgs[i:i+1, :, ymin:ymax, xmin:xmax]
            orig_h, orig_w = face_crop.shape[2:]
            original_sizes.append((orig_h, orig_w))
            
            resized_crop = F.interpolate(face_crop, size=(self.target_size, self.target_size), 
                                         mode='bicubic', align_corners=False)
            face_crops_resized.append(resized_crop)
        
        # This is the 328x328 canvas we need to save for post-processing
        resized_faces_batch = torch.cat(face_crops_resized, dim=0)

        # This is the 320x320 inner part for the model
        real_imgs = resized_faces_batch[:, :, self.border:-self.border, self.border:-self.border]
        
        masked_imgs = real_imgs * self.gpu_buffers['mask']
        
        return real_imgs, masked_imgs, original_sizes, resized_faces_batch