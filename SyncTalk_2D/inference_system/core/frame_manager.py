# inference_system/core/video_frame_manager.py
import cv2
import threading
import time
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
import psutil
import os

@dataclass
class VideoFrameData:
    """Container for video frame data"""
    frame: np.ndarray
    frame_idx: int
    timestamp: float

class VideoFrameManager:
    """Base class for video frame management"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.current_idx = 0
        self.direction = 1  # For pingpong: 1 = forward, -1 = backward
        self.total_frames = 0
        self.fps = 0
        self.video_shape = None
    
    def get_frame(self, frame_idx: int) -> Optional[VideoFrameData]:
        """Get a specific frame by index"""
        raise NotImplementedError
    
    def get_next_frame_pingpong(self) -> Optional[VideoFrameData]:
        """Get next frame in pingpong pattern"""
        raise NotImplementedError

class AllFramesMemory(VideoFrameManager):
    """Load all frames into memory with configurable behavior"""
    
    def __init__(self, video_path: str, enable_async: bool = False, initial_buffer_size: int = 50):
        """
        Args:
            video_path: Path to video file
            enable_async: If True, starts serving frames after initial_buffer_size frames are loaded
            initial_buffer_size: Number of frames to load before serving (only used if enable_async=True)
        """
        super().__init__(video_path)
        self.enable_async = enable_async
        self.initial_buffer_size = initial_buffer_size
        
        self.frames = {}
        self.frames_lock = threading.Lock()
        self.load_complete = False
        self.load_thread = None
        self.stop_loading = threading.Event()
        
        # Load tracking
        self.frames_loaded_count = 0
        self.load_start_time = None
        self.time_to_first_frame = None
        self.total_load_time = None
        
        # Initialize video properties
        self._initialize_video()
    
    def _initialize_video(self):
        """Initialize video properties and load first frame"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Get video shape from first frame
        ret, frame = self.cap.read()
        if ret:
            self.video_shape = frame.shape
            # Store first frame
            self.frames[0] = VideoFrameData(
                frame=frame,
                frame_idx=0,
                timestamp=0.0
            )
            self.frames_loaded_count = 1
        else:
            raise ValueError("Cannot read first frame")
    
    def initialize(self):
        """Initialize frame loading based on configuration"""
        self.load_start_time = time.time()
        
        if self.enable_async:
            # Async mode: start loading in background
            self._start_async_loading()
        else:
            # Sync mode: load all frames before returning
            self._load_all_frames_sync()
    
    def _start_async_loading(self):
        """Start loading frames in background thread"""
        self.load_thread = threading.Thread(target=self._load_frames_worker, name="FrameLoader")
        self.load_thread.daemon = True
        self.load_thread.start()
        
        # Wait for initial buffer
        print(f"[AllFramesMemory] Waiting for initial buffer of {self.initial_buffer_size} frames...")
        while self.frames_loaded_count < min(self.initial_buffer_size, self.total_frames):
            if self.stop_loading.is_set():
                break
            time.sleep(0.001)
        
        self.time_to_first_frame = time.time() - self.load_start_time
        print(f"[AllFramesMemory] Initial buffer ready ({self.frames_loaded_count} frames) in {self.time_to_first_frame:.2f}s")
    
    def _load_all_frames_sync(self):
        """Load all frames synchronously (blocking)"""
        print(f"[AllFramesMemory] Loading all {self.total_frames} frames into memory...")
        
        # Start from frame 1 (frame 0 already loaded)
        for frame_idx in range(1, self.total_frames):
            ret, frame = self.cap.read()
            if ret:
                self.frames[frame_idx] = VideoFrameData(
                    frame=frame,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / self.fps
                )
                self.frames_loaded_count = frame_idx + 1
                
                # Progress update
                if frame_idx % 100 == 0:
                    print(f"[AllFramesMemory] Loaded {frame_idx}/{self.total_frames} frames")
            else:
                print(f"[AllFramesMemory] Failed to read frame {frame_idx}")
                break
        
        self.cap.release()
        self.load_complete = True
        self.time_to_first_frame = time.time() - self.load_start_time
        self.total_load_time = self.time_to_first_frame
        print(f"[AllFramesMemory] Loaded {self.frames_loaded_count} frames in {self.total_load_time:.2f}s")
    
    def _load_frames_worker(self):
        """Worker thread for async frame loading"""
        try:
            # Start from frame 1 (frame 0 already loaded)
            for frame_idx in range(1, self.total_frames):
                if self.stop_loading.is_set():
                    break
                    
                ret, frame = self.cap.read()
                if ret:
                    frame_data = VideoFrameData(
                        frame=frame,
                        frame_idx=frame_idx,
                        timestamp=frame_idx / self.fps
                    )
                    
                    with self.frames_lock:
                        self.frames[frame_idx] = frame_data
                        self.frames_loaded_count = frame_idx + 1
                    
                    # Progress update
                    # if frame_idx % 100 == 0:
                    #     print(f"[FrameLoader] Background loading: {frame_idx}/{self.total_frames} frames")
                else:
                    print(f"[FrameLoader] Failed to read frame {frame_idx}")
                    break
            
            self.load_complete = True
            self.total_load_time = time.time() - self.load_start_time
            # print(f"[FrameLoader] Background loading complete: {self.frames_loaded_count} frames in {self.total_load_time:.2f}s")
            
        except Exception as e:
            print(f"[FrameLoader] Error: {e}")
        finally:
            self.cap.release()
    
    def get_frame(self, frame_idx: int) -> Optional[VideoFrameData]:
        """Get a specific frame - will wait if async loading hasn't reached it yet"""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        # Fast path - frame already loaded
        with self.frames_lock:
            if frame_idx in self.frames:
                return self.frames[frame_idx]
        
        # If async loading is enabled and frame not ready, wait for it
        if self.enable_async and not self.load_complete:
            wait_start = time.time()
            while frame_idx >= self.frames_loaded_count:
                if self.stop_loading.is_set():
                    return None
                if time.time() - wait_start > 5.0:  # 5 second timeout
                    print(f"[AllFramesMemory] Timeout waiting for frame {frame_idx}")
                    return None
                time.sleep(0.001)
            
            # Frame should be loaded now
            with self.frames_lock:
                return self.frames.get(frame_idx)
        
        return None
    
    def get_next_frame_pingpong(self) -> Optional[VideoFrameData]:
        """Get next frame in pingpong pattern"""
        frame = self.get_frame(self.current_idx)
        
        # Update index for pingpong
        self.current_idx += self.direction
        if self.current_idx >= self.total_frames - 1:
            self.current_idx = self.total_frames - 1
            self.direction = -1
        elif self.current_idx <= 0:
            self.current_idx = 0
            self.direction = 1
            
        return frame
    
    def get_stats(self) -> Dict:
        """Get loading statistics"""
        return {
            'frames_loaded': self.frames_loaded_count,
            'total_frames': self.total_frames,
            'load_complete': self.load_complete,
            'time_to_first_frame': self.time_to_first_frame,
            'total_load_time': self.total_load_time,
            'async_enabled': self.enable_async
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_loading.set()
        if self.load_thread and self.load_thread.is_alive():
            self.load_thread.join(timeout=2.0)
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()