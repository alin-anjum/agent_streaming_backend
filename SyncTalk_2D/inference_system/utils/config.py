# inference_system/utils/config.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import os

@dataclass
class ModelConfig:
    """Model-specific configuration"""
    name: str
    checkpoint_path: str
    input_size: int = 328
    use_tensorrt: bool = True
    batch_size: int = 8
    
@dataclass
class SystemConfig:
    """System-wide configuration"""
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    profile_enabled: bool = True
    cache_frames: bool = True
    max_cached_models: int = 10
    
    # Paths
    dataset_root: str = "./dataset"
    checkpoint_root: str = "./checkpoint"
    output_root: str = "./output"
    
    # Video settings
    fps: int = 25
    video_codec: str = "h264_nvenc"  # GPU encoding
    
    @classmethod
    def from_yaml(cls, path: str) -> 'SystemConfig':
        """Load config from YAML file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return cls(**data)
        return cls()
    
    def save_yaml(self, path: str):
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)