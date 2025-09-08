from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict
import numpy as np
from enum import Enum

class ProcessingMode(Enum):
    BATCH = "batch"
    STREAMING = "streaming"

@dataclass
class FrameData:
    """Universal frame data structure for both streaming and batch"""
    frame_idx: int
    img_idx: int
    img: np.ndarray
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    audio_feat: np.ndarray
    stream_id: Optional[str] = None  # For streaming mode
    timestamp: Optional[float] = None  # For synchronization
    parsing: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingRequest:
    """Request structure for processing"""
    request_id: str
    model_name: str
    mode: ProcessingMode
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None