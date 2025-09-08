# inference_system/core/landmark_manager.py
import os
import numpy as np
import threading
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class LandmarkData:
    """Container for landmark data"""
    landmarks: np.ndarray  # Shape: (num_points, 2) for x,y coordinates
    frame_idx: int
    bbox: Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax

class LandmarkManager:
    """Manages landmark loading and caching for efficient access"""
    
    def __init__(self, landmark_dir: str, enable_async: bool = False, 
             initial_buffer_size: int = 50, crop_bbox: Optional[Tuple[int, int, int, int]] = None,
             frame_range: Optional[Tuple[int, int]] = None,
             resize_dims: Optional[Tuple[int, int]] = None):
        """
        Args:
            landmark_dir: Directory containing .lms files
            enable_async: If True, starts serving landmarks after initial buffer
            initial_buffer_size: Number of landmarks to load before serving
            crop_bbox: Optional (x1, y1, x2, y2) bounding box for cropping
            frame_range: Optional (start_frame, end_frame) to load only a specific range
            resize_dims: Optional (width, height) to resize frames after cropping
        """
        self.landmark_dir = landmark_dir
        self.enable_async = enable_async
        self.initial_buffer_size = initial_buffer_size
        self.crop_bbox = crop_bbox
        self.frame_range = frame_range
        self.resize_dims = resize_dims
        
        if self.frame_range:
            self.frame_offset = self.frame_range[0]
            print(f"[LandmarkManager] Using frame range: {self.frame_range[0]} to {self.frame_range[1]}")
        else:
            self.frame_offset = 0
            
        # Calculate crop offsets if bbox provided
        if self.crop_bbox:
            self.crop_x_offset = self.crop_bbox[0]
            self.crop_y_offset = self.crop_bbox[1]
            print(f"[LandmarkManager] Using crop bbox: {self.crop_bbox}")
            print(f"[LandmarkManager] Crop offsets: x={self.crop_x_offset}, y={self.crop_y_offset}")
        else:
            self.crop_x_offset = 0
            self.crop_y_offset = 0
        
        # Calculate resize scale factors
        if self.resize_dims:
            if self.crop_bbox:
                # Scale from crop size to resize size
                crop_width = self.crop_bbox[2] - self.crop_bbox[0]
                crop_height = self.crop_bbox[3] - self.crop_bbox[1]
                self.scale_x = self.resize_dims[0] / crop_width
                self.scale_y = self.resize_dims[1] / crop_height
            elif original_video_dims:
                # Scale from original video size to resize size
                self.scale_x = self.resize_dims[0] / original_video_dims[0]
                self.scale_y = self.resize_dims[1] / original_video_dims[1]
            else:
                # Can't calculate scale without knowing original size
                self.scale_x = 1.0
                self.scale_y = 1.0
                print(f"[LandmarkManager] Warning: resize_dims set but no original size provided")
        else:   
            self.scale_x = 1.0
            self.scale_y = 1.0
        
        # Landmark storage
        self.landmarks = {}
        self.landmarks_lock = threading.Lock()
        
        # Loading state
        self.load_complete = False
        self.load_thread = None
        self.stop_loading = threading.Event()
        self.landmarks_loaded_count = 0
        
        # Timing
        self.load_start_time = None
        self.time_to_first_landmark = None
        self.total_load_time = None
        
        # Get list of landmark files
        self._scan_landmark_files()
    
    def _scan_landmark_files(self):
        """Scan directory for .lms files"""
        all_lms_files = sorted([f for f in os.listdir(self.landmark_dir) if f.endswith('.lms')],
                            key=lambda x: int(x.split('.')[0]))
        
        # Apply frame range if specified
        if self.frame_range:
            start, end = self.frame_range
            self.lms_files = all_lms_files[start:end+1]
            self.total_landmarks = len(self.lms_files)
            print(f"[LandmarkManager] Loading landmarks {start} to {end} (total: {self.total_landmarks})")
        else:
            self.lms_files = all_lms_files
            self.total_landmarks = len(self.lms_files)
        
        if self.total_landmarks == 0:
            raise ValueError(f"No .lms files found in range")
        
        print(f"[LandmarkManager] Found {self.total_landmarks} landmark files")
    
    def initialize(self):
        """Initialize landmark loading based on configuration"""
        self.load_start_time = time.time()
        
        if self.enable_async:
            # Async mode: start loading in background
            self._start_async_loading()
        else:
            # Sync mode: load all landmarks before returning
            self._load_all_landmarks_sync()
    
    def _parse_landmark_file(self, lms_path: str) -> np.ndarray:
        """Parse a single landmark file"""
        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        return np.array(lms_list, dtype=np.int32)
    
    def _compute_bbox(self, lms: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute bounding box from landmarks (following original logic)"""
        xmin = int(lms[1][0])
        ymin = int(lms[52][1])
        xmax = int(lms[31][0])
        width = xmax - xmin
        ymax = ymin + width
        return xmin, ymin, xmax, ymax
    
    def _load_landmark(self, frame_idx: int) -> Optional[LandmarkData]:
        """Load a single landmark file"""
        if frame_idx >= len(self.lms_files):
            return None
            
        lms_filename = self.lms_files[frame_idx]
        lms_path = os.path.join(self.landmark_dir, lms_filename)
        
        try:
            # Parse landmark file
            lms = self._parse_landmark_file(lms_path)
            
            # Apply crop adjustment if needed
            if self.crop_bbox:
                lms = lms.astype(np.float32)
                lms[:, 0] -= self.crop_x_offset
                lms[:, 1] -= self.crop_y_offset
            else:
                lms = lms.astype(np.float32)
            
            # Apply resize scaling if needed
            if self.resize_dims:
                lms[:, 0] *= self.scale_x
                lms[:, 1] *= self.scale_y
            
            lms = lms.astype(np.int32)
            
            # Compute bounding box (from adjusted and scaled landmarks)
            xmin, ymin, xmax, ymax = self._compute_bbox(lms)
            
            return LandmarkData(
                landmarks=lms,
                frame_idx=frame_idx,
                bbox=(xmin, ymin, xmax, ymax)
            )
        except Exception as e:
            print(f"[LandmarkManager] Error loading {lms_path}: {e}")
            return None
    
    def _load_all_landmarks_sync(self):
        """Load all landmarks synchronously (blocking)"""
        print(f"[LandmarkManager] Loading all {self.total_landmarks} landmarks into memory...")
        
        for frame_idx in range(self.total_landmarks):
            landmark_data = self._load_landmark(frame_idx)
            
            if landmark_data is not None:
                self.landmarks[frame_idx] = landmark_data
                self.landmarks_loaded_count = frame_idx + 1
                
                if frame_idx % 500 == 0 and frame_idx > 0:
                    elapsed = time.time() - self.load_start_time
                    rate = frame_idx / elapsed
                    # print(f"[LandmarkManager] Loaded {frame_idx}/{self.total_landmarks} landmarks "
                    #       f"({rate:.1f} landmarks/sec)")
        
        self.load_complete = True
        self.time_to_first_landmark = time.time() - self.load_start_time
        self.total_load_time = self.time_to_first_landmark
        print(f"[LandmarkManager] Loaded {self.landmarks_loaded_count} landmarks in {self.total_load_time:.2f}s")
    
    def _start_async_loading(self):
        """Start loading landmarks in background thread"""
        self.load_thread = threading.Thread(target=self._load_landmarks_worker, name="LandmarkLoader")
        self.load_thread.daemon = True
        self.load_thread.start()
        
        # Wait for initial buffer
        print(f"[LandmarkManager] Waiting for initial buffer of {self.initial_buffer_size} landmarks...")
        while self.landmarks_loaded_count < min(self.initial_buffer_size, self.total_landmarks):
            if self.stop_loading.is_set():
                break
            time.sleep(0.001)
        
        self.time_to_first_landmark = time.time() - self.load_start_time
        print(f"[LandmarkManager] Initial buffer ready ({self.landmarks_loaded_count} landmarks) "
              f"in {self.time_to_first_landmark:.2f}s")
    
    def _load_landmarks_worker(self):
        """Worker thread for async landmark loading"""
        try:
            for frame_idx in range(self.total_landmarks):
                if self.stop_loading.is_set():
                    break
                
                landmark_data = self._load_landmark(frame_idx)
                
                if landmark_data is not None:
                    with self.landmarks_lock:
                        self.landmarks[frame_idx] = landmark_data
                        self.landmarks_loaded_count = frame_idx + 1
                    
                    if frame_idx % 500 == 0 and frame_idx > 0:
                        print(f"[LandmarkLoader] Background loading: {frame_idx}/{self.total_landmarks} landmarks")
            
            self.load_complete = True
            self.total_load_time = time.time() - self.load_start_time
            print(f"[LandmarkLoader] Background loading complete: {self.landmarks_loaded_count} landmarks "
                  f"in {self.total_load_time:.2f}s")
            
        except Exception as e:
            print(f"[LandmarkLoader] Error: {e}")
    
    def get_landmark(self, frame_idx: int) -> Optional[LandmarkData]:
        """Get landmark data for a specific frame"""
        # Convert to local index if using frame range
        local_idx = frame_idx
        if self.frame_range:
            # If frame_idx is in original space, convert to local
            if frame_idx >= self.frame_offset:
                local_idx = frame_idx - self.frame_offset
            
            # Check if local index is valid
            if local_idx < 0 or local_idx >= self.total_landmarks:
                return None
        
        # Fast path - landmark already loaded
        with self.landmarks_lock:
            if local_idx in self.landmarks:
                landmark_data = self.landmarks[local_idx]
                # Return with original frame index
                return LandmarkData(
                    landmarks=landmark_data.landmarks,
                    frame_idx=frame_idx,  # Keep original frame index
                    bbox=landmark_data.bbox
                )
        
        # If async loading is enabled and landmark not ready, wait for it
        if self.enable_async and not self.load_complete:
            wait_start = time.time()
            while local_idx >= self.landmarks_loaded_count:
                if self.stop_loading.is_set():
                    return None
                if time.time() - wait_start > 5.0:  # 5 second timeout
                    print(f"[LandmarkManager] Timeout waiting for landmark {local_idx}")
                    return None
                time.sleep(0.001)
            
            # Landmark should be loaded now
            with self.landmarks_lock:
                if local_idx in self.landmarks:
                    landmark_data = self.landmarks[local_idx]
                    return LandmarkData(
                        landmarks=landmark_data.landmarks,
                        frame_idx=frame_idx,  # Keep original frame index
                        bbox=landmark_data.bbox
                    )
        
        return None

    
    def get_stats(self) -> Dict:
        """Get loading statistics"""
        with self.landmarks_lock:
            landmarks_in_memory = len(self.landmarks)
        
        stats = {
            'landmarks_loaded': self.landmarks_loaded_count,
            'landmarks_in_memory': landmarks_in_memory,
            'total_landmarks': self.total_landmarks,
            'load_complete': self.load_complete,
            'time_to_first_landmark': self.time_to_first_landmark,
            'total_load_time': self.total_load_time,
            'async_enabled': self.enable_async
        }
        
        if self.crop_bbox:
            stats['crop_bbox'] = self.crop_bbox
            stats['crop_offset'] = (self.crop_x_offset, self.crop_y_offset)
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_loading.set()
        
        if self.load_thread and self.load_thread.is_alive():
            self.load_thread.join(timeout=2.0)