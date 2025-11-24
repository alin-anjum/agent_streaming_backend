#!/usr/bin/env python3
"""
Dynamic Frame Processor for Tab Capture
Handles processing and management of dynamically captured frames
"""

import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

class DynamicFrameProcessor:
    """Process dynamically captured frames for use in Rapido pipeline"""
    
    def __init__(self, frames_directory: str):
        self.frames_directory = Path(frames_directory)
        self.frame_paths = []
        self._load_frame_paths()
        
        logger.info(f"📁 Dynamic frame processor initialized with {len(self.frame_paths)} frames")
    
    def _load_frame_paths(self):
        """Load and sort frame paths"""
        if not self.frames_directory.exists():
            logger.error(f"❌ Frames directory does not exist: {self.frames_directory}")
            return
        
        # Get all PNG files and sort them
        png_files = list(self.frames_directory.glob("frame_*.png"))
        self.frame_paths = sorted(png_files, key=lambda x: int(x.stem.split('_')[1]))
        
        logger.info(f"📊 Loaded {len(self.frame_paths)} frame paths")
    
    def get_frame_count(self) -> int:
        """Get total number of frames"""
        return len(self.frame_paths)
    
    def get_frame_path(self, index: int) -> Optional[str]:
        """Get frame path by index"""
        if 0 <= index < len(self.frame_paths):
            return str(self.frame_paths[index])
        return None
    
    def get_all_frame_paths(self) -> List[str]:
        """Get all frame paths"""
        return [str(path) for path in self.frame_paths]
    
    def get_frame_at_time(self, time_seconds: float, fps: float = 25.0) -> Optional[str]:
        """Get frame at specific time"""
        frame_index = int(time_seconds * fps)
        return self.get_frame_path(frame_index)

