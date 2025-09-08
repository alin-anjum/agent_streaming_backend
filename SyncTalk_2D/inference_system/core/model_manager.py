import torch
import torch.nn as nn
from typing import Dict, Optional
from threading import Lock
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages multiple models in memory with thread-safe access"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, nn.Module] = {}
        self.audio_encoders: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, Dict] = {}
        self._lock = Lock()
        
    def preload_model(self, name: str, checkpoint_path: str, mode: str = 'ave'):
        """Preload a model into memory"""
        with self._lock:
            if name in self.models:
                logger.info(f"Model {name} already loaded")
                return
                
            logger.info(f"Loading model {name} from {checkpoint_path}")
            
            # Load main model
            net = Model(6, mode).to(self.device).eval()
            net.load_state_dict(torch.load(checkpoint_path))
            net = net.half()  # FP16
            self.models[name] = net
            
            # Load audio encoder
            audio_encoder = AudioEncoder().to(self.device).eval().half()
            ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth')
            audio_encoder.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
            self.audio_encoders[name] = audio_encoder
            
            self.model_configs[name] = {
                'mode': mode,
                'checkpoint': checkpoint_path
            }
            
    def get_model(self, name: str) -> Tuple[nn.Module, nn.Module, Dict]:
        """Get model and audio encoder by name"""
        with self._lock:
            if name not in self.models:
                raise ValueError(f"Model {name} not loaded")
            return self.models[name], self.audio_encoders[name], self.model_configs[name]
    
    def list_models(self) -> List[str]:
        """List all loaded models"""
        with self._lock:
            return list(self.models.keys())