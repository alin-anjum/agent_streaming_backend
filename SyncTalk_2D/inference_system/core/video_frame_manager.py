# inference_system/core/video_frame_manager.py
import cv2
import threading
import time
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import psutil
import os
import glob
from tqdm import tqdm

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
    def __init__(self, video_path: str, enable_async: bool = False, initial_buffer_size: int = 50,
                 max_frames_to_load: Optional[int] = None, from_images: bool = False,
                 crop_bbox: Optional[Tuple[int, int, int, int]] = None,
                 frame_range: Optional[Tuple[int, int]] = None,
                 resize_dims: Optional[Tuple[int, int]] = None):
        """
        Args:
            video_path: Path to video file or image directory
            enable_async: If True, starts serving frames after initial_buffer_size frames are loaded
            initial_buffer_size: Number of frames to load before serving (only used if enable_async=True)
            max_frames_to_load: Maximum number of frames to load. If None, loads all frames.
            from_images: If True, load from image sequence instead of video
            crop_bbox: Optional (x1, y1, x2, y2) bounding box for cropping frames
            frame_range: Optional (start_frame, end_frame) to load only a specific range
            resize_dims: Optional (width, height) to resize frames after cropping
        """
        super().__init__(video_path)
        self.enable_async = enable_async
        self.initial_buffer_size = initial_buffer_size
        self.max_frames_to_load = max_frames_to_load
        self.from_images = from_images
        self.crop_bbox = crop_bbox
        self.frame_range = frame_range
        self.resize_dims = resize_dims
        
        if self.resize_dims and self.crop_bbox:
            crop_width = self.crop_bbox[2] - self.crop_bbox[0]
            crop_height = self.crop_bbox[3] - self.crop_bbox[1]
            self.scale_x = self.resize_dims[0] / crop_width
            self.scale_y = self.resize_dims[1] / crop_height
            print(f"[AllFramesMemory] Resize scale factors: x={self.scale_x:.3f}, y={self.scale_y:.3f}")
        elif self.resize_dims:
            # Scale factors will be calculated after loading first frame
            self.scale_x = None
            self.scale_y = None

        # Frame range offset for mapping
        if self.frame_range:
            self.frame_offset = self.frame_range[0]
            print(f"[AllFramesMemory] Using frame range: {self.frame_range[0]} to {self.frame_range[1]}")
        else:
            self.frame_offset = 0
        
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
        
        # Initialize video/image properties
        self._initialize_source()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply crop and resize to frame"""
        original_shape = frame.shape
        
        # First apply crop if specified
        if self.crop_bbox is not None:
            frame = self._crop_frame(frame)
            crop_shape = frame.shape
        else:
            crop_shape = original_shape
        
        # Then apply resize if specified
        if self.resize_dims is not None:
            frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_LINEAR)
            resize_shape = frame.shape
        else:
            resize_shape = crop_shape
        
        # Debug print for first frame
        if not hasattr(self, '_process_logged'):
            print(f"[DEBUG _process_frame] {original_shape} → crop → {crop_shape} → resize → {resize_shape}")
            self._process_logged = True
        
        return frame

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply crop to frame if crop_bbox is set"""
        if self.crop_bbox is None:
            return frame
        
        x1, y1, x2, y2 = self.crop_bbox
        # Ensure crop bounds are within frame
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(x1, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(y1, min(y2, h))
        
        return frame[y1:y2, x1:x2].copy()

    def _initialize_source(self):
        """Initialize video or image sequence properties"""
        if self.from_images:
            self._initialize_images()
        else:
            self._initialize_video()
    
    def _initialize_images(self):
        """Initialize from image sequence"""
        self.image_files = sorted(glob.glob(os.path.join(self.video_path, "*.jpg")) + 
                                 glob.glob(os.path.join(self.video_path, "*.png")))
        if not self.image_files:
            raise ValueError(f"No images found in: {self.video_path}")
        
        # Apply frame range if specified
        if self.frame_range:
            start, end = self.frame_range
            self.image_files = self.image_files[start:end+1]
            self.total_frames = len(self.image_files)
            print(f"[AllFramesMemory] Loading frames {start} to {end} (total: {self.total_frames})")
        else:
            self.total_frames = len(self.image_files)
        
        self.fps = 25  # Default FPS for image sequences
        
        # Limit frames if specified
        if self.max_frames_to_load is not None:
            self.frames_to_load = min(self.max_frames_to_load, self.total_frames)
        else:
            self.frames_to_load = self.total_frames
        
        # Load first frame
        first_frame = cv2.imread(self.image_files[0])
        if first_frame is not None:
            # Calculate scale factors if not already set
            if self.resize_dims and self.scale_x is None:
                orig_h, orig_w = first_frame.shape[:2]
                if self.crop_bbox:
                    # Already calculated in init
                    pass
                else:
                    self.scale_x = self.resize_dims[0] / orig_w
                    self.scale_y = self.resize_dims[1] / orig_h
                print(f"[AllFramesMemory] Resize scale factors: x={self.scale_x:.3f}, y={self.scale_y:.3f}")
            
            # Apply processing
            first_frame = self._process_frame(first_frame)
            self.video_shape = first_frame.shape
            self.frames[0] = VideoFrameData(
                frame=first_frame,
                frame_idx=0,
                timestamp=0.0
            )
            self.frames_loaded_count = 1
        else:
            raise ValueError("Cannot read first image")
    
    def _initialize_video(self):
        """Initialize video properties and load first frame"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
            
        full_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Apply frame range if specified
        if self.frame_range:
            start, end = self.frame_range
            # Validate range
            if start >= full_video_frames or end >= full_video_frames:
                raise ValueError(f"Frame range {self.frame_range} exceeds video length {full_video_frames}")
            
            self.total_frames = end - start + 1
            
            # Seek to start frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            print(f"[AllFramesMemory] Loading frames {start} to {end} from video (total: {self.total_frames})")
        else:
            self.total_frames = full_video_frames
        
        # Limit frames if specified
        if self.max_frames_to_load is not None:
            self.frames_to_load = min(self.max_frames_to_load, self.total_frames)
        else:
            self.frames_to_load = self.total_frames
        
        # Get video shape from first frame
        ret, frame = self.cap.read()
        if ret:
            frame = self._process_frame(frame)
            self.video_shape = frame.shape
            self.frames[0] = VideoFrameData(
                frame=frame,
                frame_idx=0,  # Local index
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
        buffer_target = min(self.initial_buffer_size, self.frames_to_load)
        print(f"[AllFramesMemory] Waiting for initial buffer of {buffer_target} frames...")
        while self.frames_loaded_count < buffer_target:
            if self.stop_loading.is_set():
                break
            time.sleep(0.001)
        
        self.time_to_first_frame = time.time() - self.load_start_time
        print(f"[AllFramesMemory] Initial buffer ready ({self.frames_loaded_count} frames) in {self.time_to_first_frame:.2f}s")
    
    def _load_all_frames_sync(self):
        """Load frames synchronously with processing"""
        if self.from_images:
            # Load from images
            for local_idx in range(1, self.frames_to_load):
                frame = cv2.imread(self.image_files[local_idx])
                if frame is not None:
                    # Apply crop and resize
                    frame = self._process_frame(frame)
                    self.frames[local_idx] = VideoFrameData(
                        frame=frame,
                        frame_idx=local_idx,
                        timestamp=local_idx / self.fps
                    )
                    self.frames_loaded_count = local_idx + 1
                else:
                    print(f"[AllFramesMemory] Failed to read image {local_idx}")
                    break
        else:
            # Load from video
            for local_idx in tqdm(range(1, self.frames_to_load), desc="Loading video frames", unit="frames"):
                ret, frame = self.cap.read()
                if ret:
                    # Apply crop and resize
                    frame = self._process_frame(frame)
                    self.frames[local_idx] = VideoFrameData(
                        frame=frame,
                        frame_idx=local_idx,
                        timestamp=local_idx / self.fps
                    )
                    self.frames_loaded_count = local_idx + 1
                else:
                    print(f"\n[AllFramesMemory] Failed to read frame {local_idx}")
                    break
            
            self.cap.release()
    
    def _load_frames_worker(self):
        """Worker thread for async frame loading"""
        try:
            if self.from_images:
                # Async loading from images
                for frame_idx in range(1, self.frames_to_load):
                    if self.stop_loading.is_set():
                        break
                        
                    frame = cv2.imread(self.image_files[frame_idx])
                    if frame is not None:
                        frame_data = VideoFrameData(
                            frame=frame,
                            frame_idx=frame_idx,
                            timestamp=frame_idx / self.fps
                        )
                        
                        with self.frames_lock:
                            self.frames[frame_idx] = frame_data
                            self.frames_loaded_count = frame_idx + 1
                    else:
                        print(f"[FrameLoader] Failed to read image {frame_idx}")
                        break
            else:
                # Async loading from video
                for frame_idx in range(1, self.frames_to_load):
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
                    else:
                        print(f"[FrameLoader] Failed to read frame {frame_idx}")
                        break
                
                self.cap.release()
            
            self.load_complete = True
            self.total_load_time = time.time() - self.load_start_time
            
        except Exception as e:
            print(f"[FrameLoader] Error: {e}")
        finally:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                self.cap.release()
    
    def get_frame(self, frame_idx: int) -> Optional[VideoFrameData]:
        """Get a specific frame - will wait if async loading hasn't reached it yet"""
        if frame_idx < 0 or frame_idx >= self.frames_to_load:
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
    
    def get_frame_pingpong(self, frame_idx: int) -> Optional[VideoFrameData]:
        """Get frame by index with automatic pingpong wrapping for out-of-range indices"""
        if frame_idx < 0:
            return None
        
        # Map to local index space
        local_idx = frame_idx
        
        # If within range, just return the frame normally
        if local_idx < self.frames_to_load:
            frame_data = self.get_frame(local_idx)
            if frame_data and self.frame_range:
                # Update frame_idx to reflect the original frame number
                original_frame_idx = self.frame_offset + frame_data.frame_idx
                return VideoFrameData(
                    frame=frame_data.frame,
                    frame_idx=original_frame_idx,  # Return original frame index
                    timestamp=frame_data.timestamp
                )
            return frame_data
        
        # Calculate pingpong mapping for out-of-range indices
        cycle_length = 2 * (self.frames_to_load - 1)
        position_in_cycle = local_idx % cycle_length
        
        if position_in_cycle < self.frames_to_load:
            # Forward phase
            mapped_idx = position_in_cycle
        else:
            # Backward phase
            backward_position = position_in_cycle - self.frames_to_load
            mapped_idx = self.frames_to_load - 2 - backward_position
        
        # Get the actual frame
        frame_data = self.get_frame(mapped_idx)
        if frame_data and self.frame_range:
            # Update frame_idx to reflect the original frame number
            original_frame_idx = self.frame_offset + frame_data.frame_idx
            return VideoFrameData(
                frame=frame_data.frame,
                frame_idx=original_frame_idx,  # Return original frame index
                timestamp=frame_data.timestamp
            )
        return frame_data
        
    def get_next_frame_pingpong(self) -> Optional[VideoFrameData]:
        """Get next frame in pingpong pattern"""
        frame = self.get_frame(self.current_idx)
        
        # Update index for pingpong
        self.current_idx += self.direction
        if self.current_idx >= self.frames_to_load - 1:
            self.current_idx = self.frames_to_load - 1
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
            'frames_to_load': self.frames_to_load,
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