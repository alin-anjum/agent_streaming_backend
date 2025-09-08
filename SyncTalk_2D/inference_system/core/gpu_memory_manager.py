# inference_system/core/gpu_memory_manager.py
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union  
import threading

class GPUMemoryManager:
    """Manages pre-allocated GPU memory buffers for inference"""
    
    def __init__(self, device: torch.device, mode: str = "ave"):
        """
        Args:
            device: torch device (cuda)
            mode: Audio mode - "ave", "hubert", or "wenet"
        """
        self.device = device
        self.mode = mode
        self.buffers = {}
        self.buffer_lock = threading.Lock()
        
        # Audio feature dimensions based on mode
        self.audio_shapes = {
            "hubert": (1, 32, 32, 32),
            "wenet": (1, 256, 16, 32),
            "ave": (1, 32, 16, 16)
        }
        
        if mode not in self.audio_shapes:
            raise ValueError(f"Unknown mode: {mode}")
        
        self._allocate_buffers()
    
    def _allocate_buffers(self):
        """Pre-allocate GPU buffers"""
        print(f"[GPUMemoryManager] Pre-allocating GPU buffers for mode: {self.mode}")
        
        # Audio feature buffer
        audio_shape = self.audio_shapes[self.mode]
        self.buffers['audio_feat'] = torch.zeros(
            audio_shape, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Image concat buffer (6 channels: 3 for real image, 3 for masked)
        self.buffers['img_concat'] = torch.zeros(
            (1, 6, 320, 320),  # 320x320 after 4-pixel border crop
            dtype=torch.float32,
            device=self.device
        )
        
        # Output prediction buffer
        self.buffers['pred'] = torch.zeros(
            (1, 3, 320, 320),
            dtype=torch.float32,
            device=self.device
        )
        
        # Intermediate CPU buffers for efficient copying
        self.cpu_buffers = {
            'img_real': np.zeros((3, 320, 320), dtype=np.float32),
            'img_masked': np.zeros((3, 320, 320), dtype=np.float32),
            'audio_feat': np.zeros(audio_shape[1:], dtype=np.float32)
        }
        
        # Print memory usage
        total_gpu_memory = 0
        for name, buffer in self.buffers.items():
            memory_mb = buffer.element_size() * buffer.nelement() / (1024 * 1024)
            total_gpu_memory += memory_mb
            print(f"  {name}: {tuple(buffer.shape)} - {memory_mb:.2f} MB")
        
        print(f"[GPUMemoryManager] Total GPU memory allocated: {total_gpu_memory:.2f} MB")
    
    def prepare_audio_features(self, audio_feat: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare audio features on GPU without new allocation
        
        Args:
            audio_feat: Numpy array or torch tensor of audio features
            
        Returns:
            GPU tensor (view of pre-allocated buffer)
        """
        # Handle both tensor and numpy inputs
        if isinstance(audio_feat, torch.Tensor):
            # Already a tensor - reshape based on mode
            if self.mode == "hubert":
                audio_feat = audio_feat.reshape(32, 32, 32)
            elif self.mode == "wenet":
                audio_feat = audio_feat.reshape(256, 16, 32)
            elif self.mode == "ave":
                audio_feat = audio_feat.reshape(32, 16, 16)
            
            # Copy to pre-allocated GPU buffer
            with torch.no_grad():
                if audio_feat.is_cuda:
                    self.buffers['audio_feat'].copy_(audio_feat.unsqueeze(0))
                else:
                    self.buffers['audio_feat'].copy_(audio_feat.unsqueeze(0).cuda())
        else:
            # Numpy array - use original logic
            if self.mode == "hubert":
                audio_feat_np = audio_feat.reshape(32, 32, 32)
            elif self.mode == "wenet":
                audio_feat_np = audio_feat.reshape(256, 16, 32)
            elif self.mode == "ave":
                audio_feat_np = audio_feat.reshape(32, 16, 16)
            
            # Copy to pre-allocated GPU buffer
            with torch.no_grad():
                self.buffers['audio_feat'].copy_(
                    torch.from_numpy(audio_feat_np).unsqueeze(0)
                )
        
        return self.buffers['audio_feat']
    
    def prepare_image_tensors(self, img_real: np.ndarray, img_masked: np.ndarray) -> torch.Tensor:
        """
        Prepare concatenated image tensors on GPU without new allocation
        
        Args:
            img_real: Real image array (C, H, W) in float32
            img_masked: Masked image array (C, H, W) in float32
            
        Returns:
            GPU tensor (view of pre-allocated buffer)
        """
        # Normalize and copy to GPU buffer
        with torch.no_grad():
            # Copy real image to first 3 channels
            self.buffers['img_concat'][0, :3].copy_(
                torch.from_numpy(img_real / 255.0)
            )
            
            # Copy masked image to last 3 channels
            self.buffers['img_concat'][0, 3:].copy_(
                torch.from_numpy(img_masked / 255.0)
            )
        
        return self.buffers['img_concat']
    
    def get_prediction_buffer(self) -> torch.Tensor:
        """Get pre-allocated prediction buffer"""
        return self.buffers['pred']
    
    def prediction_to_numpy(self) -> np.ndarray:
        """
        Convert prediction buffer to numpy array
        Returns: (H, W, C) uint8 array
        """
        with torch.no_grad():
            # Copy to CPU and convert
            pred_cpu = self.buffers['pred'][0].cpu()
            pred_np = pred_cpu.numpy().transpose(1, 2, 0) * 255
            return np.array(pred_np, dtype=np.uint8)
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        stats = {}
        total_elements = 0
        total_bytes = 0
        
        for name, buffer in self.buffers.items():
            elements = buffer.nelement()
            bytes_used = buffer.element_size() * elements
            total_elements += elements
            total_bytes += bytes_used
            
            stats[name] = {
                'shape': tuple(buffer.shape),
                'dtype': str(buffer.dtype),
                'elements': elements,
                'bytes': bytes_used,
                'mb': bytes_used / (1024 * 1024)
            }
        
        stats['total'] = {
            'elements': total_elements,
            'bytes': total_bytes,
            'mb': total_bytes / (1024 * 1024)
        }
        
        return stats

class BatchGPUMemoryManager(GPUMemoryManager):
    """Extended GPU memory manager with batch processing support"""
    
    def __init__(self, device: torch.device, mode: str = "ave", batch_size: int = 4):
        """
        Args:
            device: torch device (cuda)
            mode: Audio mode
            batch_size: Number of frames to process in parallel
        """
        self.batch_size = batch_size
        super().__init__(device, mode)
    
    def _allocate_buffers(self):
        """Pre-allocate GPU buffers for batch processing"""
        print(f"[BatchGPUMemoryManager] Pre-allocating GPU buffers for batch_size={self.batch_size}, mode={self.mode}")
        
        # Audio feature buffer (batched)
        audio_shape = list(self.audio_shapes[self.mode])
        audio_shape[0] = self.batch_size
        self.buffers['audio_feat'] = torch.zeros(
            tuple(audio_shape), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Image concat buffer (batched)
        self.buffers['img_concat'] = torch.zeros(
            (self.batch_size, 6, 320, 320),
            dtype=torch.float32,
            device=self.device
        )
        
        # Output prediction buffer (batched)
        self.buffers['pred'] = torch.zeros(
            (self.batch_size, 3, 320, 320),
            dtype=torch.float32,
            device=self.device
        )
        
        # Batch accumulation buffers
        self.batch_audio_cpu = []
        self.batch_img_real_cpu = []
        self.batch_img_masked_cpu = []
        self.batch_indices = []
        
        # Print memory usage
        total_gpu_memory = 0
        for name, buffer in self.buffers.items():
            memory_mb = buffer.element_size() * buffer.nelement() / (1024 * 1024)
            total_gpu_memory += memory_mb
            print(f"  {name}: {tuple(buffer.shape)} - {memory_mb:.2f} MB")
        
        print(f"[BatchGPUMemoryManager] Total GPU memory allocated: {total_gpu_memory:.2f} MB")
    
    def add_to_batch(self, audio_feat: Union[np.ndarray, torch.Tensor], img_real: np.ndarray, 
                    img_masked: np.ndarray, frame_idx: int) -> bool:
        """
        Add frame data to batch
        
        Returns:
            True if batch is full and ready for processing
        """
        # Handle tensor input for audio
        if isinstance(audio_feat, torch.Tensor):
            audio_feat_np = audio_feat.cpu().numpy() if audio_feat.is_cuda else audio_feat.numpy()
        else:
            audio_feat_np = audio_feat
        
        # Reshape audio based on mode
        if self.mode == "hubert":
            audio_feat_np = audio_feat_np.reshape(32, 32, 32)
        elif self.mode == "wenet":
            audio_feat_np = audio_feat_np.reshape(256, 16, 32)
        elif self.mode == "ave":
            audio_feat_np = audio_feat_np.reshape(32, 16, 16)
        
        self.batch_audio_cpu.append(audio_feat_np)
        self.batch_img_real_cpu.append(img_real)
        self.batch_img_masked_cpu.append(img_masked)
        self.batch_indices.append(frame_idx)
        
        return len(self.batch_audio_cpu) >= self.batch_size
    
    def prepare_batch(self) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Prepare accumulated batch for inference
        
        Returns:
            audio_tensor, image_tensor, frame_indices
        """
        batch_size = len(self.batch_audio_cpu)
        
        with torch.no_grad():
            # Copy audio features to GPU
            for i in range(batch_size):
                audio_data = self.batch_audio_cpu[i]
                # Ensure it's a numpy array
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy() if audio_data.is_cuda else audio_data.numpy()
                
                self.buffers['audio_feat'][i].copy_(
                    torch.from_numpy(audio_data)
                )
            
            # Copy images to GPU
            for i in range(batch_size):
                self.buffers['img_concat'][i, :3].copy_(
                    torch.from_numpy(self.batch_img_real_cpu[i] / 255.0)
                )
                self.buffers['img_concat'][i, 3:].copy_(
                    torch.from_numpy(self.batch_img_masked_cpu[i] / 255.0)
                )
        
        # Get views for actual batch size
        audio_batch = self.buffers['audio_feat'][:batch_size]
        img_batch = self.buffers['img_concat'][:batch_size]
        indices = self.batch_indices.copy()
        
        # Clear batch
        self.batch_audio_cpu.clear()
        self.batch_img_real_cpu.clear()
        self.batch_img_masked_cpu.clear()
        self.batch_indices.clear()
        
        return audio_batch, img_batch, indices
    
    def get_batch_predictions(self, batch_size: int) -> torch.Tensor:
        """Get prediction buffer for actual batch size"""
        return self.buffers['pred'][:batch_size]