# inference_system/streaming/stream_wrapper.py

import os
import sys
import torch
import numpy as np
import cv2
import time
from typing import Generator, Optional, Dict, List, Tuple
from dataclasses import dataclass
import threading
import queue
from collections import deque
import time

from torch.cuda.amp import autocast

# Import YOUR EXISTING modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# UNet imports will be done conditionally in __init__
from utils import AudioEncoder, AudDataset, get_audio_features
from inference_system.core.video_frame_manager import AllFramesMemory
from inference_system.core.landmark_manager import LandmarkManager
from inference_system.utils.profiler import PerformanceProfiler

# Import your existing pipeline components
from ..inference_video import FrameData
from .workers import (
    streaming_cpu_pre_processing_worker,
    streaming_gpu_worker,
    streaming_cpu_post_processing_worker
)


class FrameLatencyMonitor:
    """Simple latency monitor for tracking frame generation times"""
    def __init__(self, window_size: int = 1000):
        self.latencies = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
    def add_latency(self, latency_ms: float):
        with self.lock:
            self.latencies.append(latency_ms)
    
    def get_stats(self) -> Dict[str, float]:
        with self.lock:
            if not self.latencies:
                return {}
            latencies = list(self.latencies)
            return {
                'mean': np.mean(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'max': np.max(latencies)
            }



class StreamingInferenceWrapper:
    """
    Wrapper around your existing pipeline to enable streaming.
    This REUSES all your existing worker functions.
    """
    _gpu_stats = {
        'inference_times': [],
        'lock': threading.Lock()
    }

    def __init__(self, model_name: str, checkpoint_path: str, 
             dataset_dir: str, mode: str = "ave", 
             batch_size: int = 4, device: str = 'cuda',
             crop_bbox: Optional[Tuple[int, int, int, int]] = None,
             frame_range: Optional[Tuple[int, int]] = None,
             video_path: Optional[str] = None,
             resize_dims: Optional[Tuple[int, int]] = None,
             use_old_unet: bool = False,
             start_workers_on_init: bool = True):
        
        with self._gpu_stats['lock']:
            self._gpu_stats['inference_times'].clear()

        self.model_name = model_name
        self.dataset_dir = dataset_dir  # Store for later use
        self.mode = mode
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.crop_bbox = crop_bbox
        self.frame_range = frame_range
        self.video_path = video_path
        self.resize_dims = resize_dims
        
        if video_path:
            print(f"  Default video path: {video_path}")
            
        # Load model (conditionally import old or new version)
        if use_old_unet:
            print(f"Loading OLD UNet version (unet_328.py) with mode={mode}")
            from unet_328 import Model
        else:
            print(f"Loading NEW UNet version (unet_328_new.py) with mode={mode}")
            from unet_328_new import Model
            
        print(f"Loading model {model_name} from {checkpoint_path}...")
        self.net = Model(6, mode).to(self.device).eval()
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.net = self.net.half()
        
        # Load audio encoder 
        self.audio_encoder = AudioEncoder().to(self.device).eval().half()
        ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth')
        self.audio_encoder.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
        
        # Preload landmarks for this model
        print(f"Preloading landmarks for model {model_name}...")
        landmark_dir = os.path.join(dataset_dir, "landmarks/")
        self.landmark_manager = LandmarkManager(
            landmark_dir, 
            enable_async=False, 
            crop_bbox=crop_bbox,
            frame_range=frame_range,
            resize_dims=resize_dims
        )
        self.landmark_manager.initialize()
        
        print(f"Landmarks preloaded: {len(self.landmark_manager.landmarks)} landmarks ready")
        if crop_bbox:
            print(f"  Landmarks adjusted for crop bbox: {crop_bbox}")
        if frame_range:
            print(f"  Using frame range: {frame_range[0]} to {frame_range[1]}")

        
        # Preload frames if video_path is provided
        if video_path and os.path.exists(video_path):
            print(f"Preloading frames from video: {video_path}...")
            self.frame_manager = AllFramesMemory(
                video_path,
                from_images=False,
                crop_bbox=crop_bbox,
                frame_range=frame_range,
                resize_dims=resize_dims
            )
            self.frame_manager.initialize()
            print(f"Frames preloaded: {self.frame_manager.frames_loaded_count} frames ready")
            print(f"Frame dimensions: {self.frame_manager.video_shape}")
        else:
            # Try loading from images
            img_dir = os.path.join(dataset_dir, "full_body_img/")
            if os.path.exists(img_dir):
                print(f"Preloading frames from images: {img_dir}...")
                self.frame_manager = AllFramesMemory(
                    img_dir,
                    from_images=True,
                    crop_bbox=crop_bbox,
                    frame_range=frame_range,
                    resize_dims=resize_dims
                )
                self.frame_manager.initialize()
                print(f"Frames preloaded: {self.frame_manager.frames_loaded_count} frames ready")
                print(f"Frame dimensions: {self.frame_manager.video_shape}")
            else:
                print(f"Warning: No video_path provided and no images found in {img_dir}")
                self.frame_manager = None
        
        # Print summary
        print(f"\n[{model_name}] Preload complete:")
        print(f"  Model: ✓")
        print(f"  Landmarks: ✓ ({len(self.landmark_manager.landmarks)} loaded)")
        if self.frame_manager:
            print(f"  Frames: ✓ ({self.frame_manager.frames_loaded_count} loaded, shape: {self.frame_manager.video_shape})")
        else:
            print(f"  Frames: ✗ (will need to be provided at inference time)")
        if crop_bbox:
            print(f"  Crop: {crop_bbox}")
        if resize_dims:
            print(f"  Resize: {resize_dims}")
        if frame_range:
            print(f"  Frame range: {frame_range[0]}-{frame_range[1]}")

        # Initialize queues 
        self.preprocess_queue = queue.Queue(maxsize=batch_size * 20)  # INCREASED 20x
        self.gpu_queue = queue.Queue(maxsize=batch_size * 20)        # INCREASED 20x  
        self.postprocess_queue = queue.Queue(maxsize=batch_size * 20) # INCREASED 20x
        
        # Output buffer for streaming
        self.output_buffer = {}
        self.next_output_idx = 0
        self.output_lock = threading.Lock()
        
        # Performance tracking
        self.profiler = PerformanceProfiler(f"StreamWrapper_{model_name}")
        self.latency_monitor = FrameLatencyMonitor()
        
        self.gpu_inference_times = []
        self.gpu_inference_lock = threading.Lock()

        # Workers
        self.workers = []
        self.running = False
        # Start workers immediately for base functionality, if requested
        if start_workers_on_init:
            self.start_workers()

        self.pipeline_metrics = {
            'preprocess_times': deque(maxlen=1000),
            'gpu_wait_times': deque(maxlen=1000),
            'gpu_compute_times': deque(maxlen=1000),
            'postprocess_times': deque(maxlen=1000),
            'frame_wait_times': deque(maxlen=1000),
            'batch_sizes': deque(maxlen=1000),
            'queue_depths': {
                'preprocess': deque(maxlen=1000),
                'gpu': deque(maxlen=1000),
                'postprocess': deque(maxlen=1000)
            }
        }
        self.metrics_lock = threading.Lock()

    def reset(self):
        """Reset the wrapper to a clean state for a new stream."""
        print(f"[{self.model_name}] Resetting streaming wrapper...")
        
        # Stop existing workers
        self.stop_workers()
        
        # Clear output buffer and reset index
        with self.output_lock:
            self.output_buffer.clear()
            self.next_output_idx = 0
            
        # Re-initialize queues
        self.preprocess_queue = queue.Queue(maxsize=self.batch_size * 20)  # INCREASED 20x
        self.gpu_queue = queue.Queue(maxsize=self.batch_size * 20)        # INCREASED 20x
        self.postprocess_queue = queue.Queue(maxsize=self.batch_size * 20) # INCREASED 20x
        
        # Restart workers
        self.start_workers()
        print(f"[{self.model_name}] Wrapper reset complete")

    def try_get_next_frames(self):
        """Try to get any sequential frames that are ready - non-blocking"""
        frames = []
        
        with self.output_lock:
            if len(self.output_buffer) > 10:
                buffer_min = min(self.output_buffer.keys())
                if buffer_min > self.next_output_idx:
                    gap_size = buffer_min - self.next_output_idx
                    print(f"[ERROR] Frame gap detected! Expected {self.next_output_idx}, "
                        f"but buffer starts at {buffer_min}. Missing {gap_size} frames!")
                    exit(0)

            # Collect sequential frames starting from next expected
            while self.next_output_idx in self.output_buffer:
                frame_data = self.output_buffer.pop(self.next_output_idx)
                frames.append(frame_data)
                self.next_output_idx += 1
                
                if len(frames) >= 10:
                    break
        
        return frames

    def start_output_streaming(self):
        """Thread that immediately releases frames in sequential order"""
        self.expected_frame_idx = 0
        
        def output_stream_worker():
            while self.running:
                ready_batch = []
                
                # Quick check under lock
                with self.output_lock:
                    # Collect any sequential frames available
                    while self.expected_frame_idx in self.output_buffer:
                        frame_data = self.output_buffer.pop(self.expected_frame_idx)
                        ready_batch.append(frame_data)
                        self.expected_frame_idx += 1
                
                # Process outside the lock
                if ready_batch:
                    for frame in ready_batch:
                        try:
                            self.ready_frames_queue.put(frame, timeout=1.0)
                            self.frame_ready_event.set()
                        except queue.Full:
                            print(f"WARNING: Output queue full, dropping frame")
                else:
                    # No frames ready, sleep briefly
                    time.sleep(0.001)  # 1ms
        
        self.output_stream_thread = threading.Thread(
            target=output_stream_worker,
            daemon=True,
            name="OutputStreamer"
        )
        self.output_stream_thread.start()

    def get_next_frame(self, timeout=0.001):
        """Get next frame if ready (non-blocking)"""
        try:
            return self.ready_frames_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _on_frame_ready(self, frame_idx):
        """Callback when a frame is ready - can be used for immediate notification"""
        # This enables true streaming - as soon as a frame is done, it's available
        pass

    def start_workers(self):
        """Start the pipeline workers using YOUR EXISTING worker functions"""
        if self.running:
            return

        self.running = True
        
        self.workers = [
            threading.Thread(
                target=streaming_cpu_pre_processing_worker,
                args=(self.preprocess_queue, self.gpu_queue, self.batch_size, self.profiler),
                daemon=True,
                name="Stream-PreProc"
            ),
            threading.Thread(
                target=streaming_gpu_worker,
                args=(self.gpu_queue, self.postprocess_queue, self.device, 
                    self.net, self.mode, self.profiler, self.batch_size),
                daemon=True,
                name="Stream-GPU"
            ),
            threading.Thread(
                target=streaming_cpu_post_processing_worker,
                args=(self.postprocess_queue, self.output_buffer, self.output_lock, 
                    self.profiler, self._on_frame_ready),
                daemon=True,
                name="Stream-PostProc"
            )
        ]
        
        for worker in self.workers:
            worker.start()
    
    def _streaming_post_processor(self):
        """Modified post-processor that yields frames instead of writing to video"""
        processed_count = 0

        while self.running:
            try:
                item = self.postprocess_queue.get(timeout=0.1)
                if item is None:
                    break
                
                # Process using your existing logic
                p_batch_data, p_pred_batch_np, p_canvases, p_orig_crops = item
                
                # Process entire batch first, storing results
                batch_results = []
                for j in range(len(p_batch_data)):
                    frame_start = time.time()
                    
                    data = p_batch_data[j]
                    pred_np = p_pred_batch_np[j]
                    canvas = p_canvases[j]
                    original_crop = p_orig_crops[j]
                    
                    full_frame = data.img
                    xmin, ymin, xmax, ymax = data.bbox
                    
                    canvas[4:324, 4:324] = np.clip(pred_np, 0, 255).astype(np.uint8)
                    h_crop, w_crop = original_crop.shape[:2]
                    final_face = cv2.resize(canvas, (w_crop, h_crop), interpolation=cv2.INTER_LINEAR)
                    
                    final_frame = full_frame.copy()
                    final_frame[ymin:ymax, xmin:xmax] = final_face
                    
                    # Track latency
                    latency = (time.time() - frame_start) * 1000
                    self.latency_monitor.add_latency(latency)
                    
                    batch_results.append((data.frame_idx, (final_frame, data.img_idx, data.frame_idx)))
                    processed_count += 1
                
                # Sort batch results by frame_idx to ensure order within batch
                batch_results.sort(key=lambda x: x[0])
                
                # Add to output buffer IN ORDER
                with self.output_lock:
                    for frame_idx, frame_data in batch_results:
                        self.output_buffer[frame_idx] = frame_data
                            
            except queue.Empty:
                continue
    
    def get_ready_frames(self) -> List[Tuple]:
        """Get all currently ready frames (non-blocking)"""
        ready_frames = []
        
        with self.output_lock:
            # Debug: Log buffer state on first few calls
            if self.next_output_idx < 5 or self.next_output_idx % 100 == 0:
                buffer_keys = sorted(list(self.output_buffer.keys()))[:10]
                print(f"[get_ready_frames] next_output_idx={self.next_output_idx}, "
                    f"buffer size={len(self.output_buffer)}, "
                    f"buffer keys={buffer_keys}")
            
            # Collect all consecutive frames
            collected = 0
            while self.next_output_idx in self.output_buffer:
                frame_data = self.output_buffer.pop(self.next_output_idx)
                ready_frames.append(frame_data)
                self.next_output_idx += 1
                collected += 1
                
                # Log collection
                if collected <= 5 or collected % 100 == 0:
                    print(f"[get_ready_frames] Collected frame {self.next_output_idx - 1}")
        
        return ready_frames
    
    def stop_workers(self):
        """Stop all workers"""
        if not self.running:
            return
            
        self.running = False
        
        # Send sentinel to each queue to unblock workers
        self.preprocess_queue.put(None)
        self.gpu_queue.put(None)
        self.postprocess_queue.put(None)
        
        # Wait for workers to terminate
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        
        self.workers = []
        
        # Clear queues
        for q in [self.preprocess_queue, self.gpu_queue, self.postprocess_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def __del__(self):
        self.stop_workers()