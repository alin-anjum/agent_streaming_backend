from typing import Generator, Tuple, Optional, List, Dict
import uuid
import queue
import threading
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field

from ..core.pipeline import BasePipeline, OrderedOutputBuffer
from ..core.data_structures import FrameData, ProcessingMode
from ..core.stages import PreProcessingStage, GPUInferenceStage, PostProcessingStage
from .audio_processor import AudioChunkProcessor, AudioChunk

@dataclass
class FrameLatencyTracker:
    """Tracks detailed latency metrics for individual frames"""
    frame_id: int
    start_time: float
    preprocess_start: float = 0
    preprocess_end: float = 0
    gpu_queue_time: float = 0
    gpu_start: float = 0
    gpu_end: float = 0
    postprocess_start: float = 0
    postprocess_end: float = 0
    total_latency: float = 0
    
    def calculate_stage_latencies(self) -> Dict[str, float]:
        """Calculate latency for each pipeline stage"""
        return {
            'preprocess': (self.preprocess_end - self.preprocess_start) * 1000,
            'gpu_wait': (self.gpu_start - self.gpu_queue_time) * 1000,
            'gpu_inference': (self.gpu_end - self.gpu_start) * 1000,
            'postprocess': (self.postprocess_end - self.postprocess_start) * 1000,
            'total': self.total_latency * 1000
        }

class LatencyMonitor:
    """Monitors and reports frame generation latencies"""
    
    def __init__(self, window_size: int = 1000, warning_threshold_ms: float = 30.0):
        self.window_size = window_size
        self.warning_threshold_ms = warning_threshold_ms
        self.latencies = deque(maxlen=window_size)
        self.lock = threading.Lock()
        
        # Track frames exceeding threshold
        self.slow_frames = []
        self.dropped_frames = 0
        
    def add_frame_latency(self, latency_ms: float, frame_id: int):
        """Add a frame latency measurement"""
        with self.lock:
            self.latencies.append(latency_ms)
            
            if latency_ms > self.warning_threshold_ms:
                self.slow_frames.append((frame_id, latency_ms))
                if latency_ms > 40.0:  # Critical threshold
                    self.dropped_frames += 1
                    print(f"⚠️  CRITICAL: Frame {frame_id} took {latency_ms:.2f}ms (>40ms threshold)")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get detailed latency statistics"""
        with self.lock:
            if not self.latencies:
                return {}
            
            latencies = list(self.latencies)
            latencies.sort()
            
            return {
                'count': len(latencies),
                'mean': np.mean(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'p50': np.percentile(latencies, 50),
                'p90': np.percentile(latencies, 90),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'p99_9': np.percentile(latencies, 99.9),
                'dropped_frames': self.dropped_frames,
                'slow_frames_count': len(self.slow_frames)
            }

class StreamProcessor(BasePipeline):
    """
    Enhanced streaming processor with detailed latency tracking and optimization
    """
    
    def __init__(self, model_manager, stream_id: str, model_name: str,
                 frame_manager, landmark_manager, start_frame: int = 0,
                 batch_size: int = 4, target_latency_ms: float = 30.0, **kwargs):
        super().__init__(model_manager, batch_size, **kwargs)
        
        # Stream identification
        self.stream_id = stream_id
        self.model_name = model_name
        
        # Frame management
        self.frame_manager = frame_manager
        self.landmark_manager = landmark_manager
        self.start_frame = start_frame
        self.current_frame_idx = start_frame
        
        # Get model and audio encoder
        self.model, self.audio_encoder, self.model_config = model_manager.get_model(model_name)
        
        # Audio processing
        self.audio_processor = AudioChunkProcessor(
            self.audio_encoder,
            self.model_manager.device,
            self.model_config['mode']
        )
        
        # Output management with priority queue for low-latency frames
        self.output_buffer = OrderedOutputBuffer()
        self.sequence_number = 0
        
        # Latency tracking
        self.target_latency_ms = target_latency_ms
        self.latency_monitor = LatencyMonitor(warning_threshold_ms=target_latency_ms)
        self.frame_trackers = {}  # frame_id -> FrameLatencyTracker
        self.tracker_lock = threading.Lock()
        
        # Dynamic batching based on latency
        self.adaptive_batch_size = batch_size
        self.min_batch_size = 1
        self.max_batch_size = batch_size
        
        # Initialize stages
        self.preprocess_stage = PreProcessingStage()
        self.gpu_stage = GPUInferenceStage(model_manager, self.model_manager.device)
        self.postprocess_stage = PostProcessingStage(num_workers=2)
        
    def _create_frame_tracker(self, frame_id: int) -> FrameLatencyTracker:
        """Create a latency tracker for a frame"""
        tracker = FrameLatencyTracker(frame_id=frame_id, start_time=time.time())
        with self.tracker_lock:
            self.frame_trackers[frame_id] = tracker
        return tracker
    
    def _update_tracker(self, frame_id: int, **kwargs):
        """Update frame tracker with timing information"""
        with self.tracker_lock:
            if frame_id in self.frame_trackers:
                tracker = self.frame_trackers[frame_id]
                for key, value in kwargs.items():
                    setattr(tracker, key, value)
    
    def _finalize_frame_latency(self, frame_id: int):
        """Finalize frame latency tracking and update statistics"""
        with self.tracker_lock:
            if frame_id in self.frame_trackers:
                tracker = self.frame_trackers[frame_id]
                end_time = time.time()
                tracker.total_latency = end_time - tracker.start_time
                
                # Add to monitor
                self.latency_monitor.add_frame_latency(
                    tracker.total_latency * 1000,
                    frame_id
                )
                
                # Adaptive batching based on latency
                if tracker.total_latency * 1000 > self.target_latency_ms:
                    # Reduce batch size if latency is too high
                    self.adaptive_batch_size = max(
                        self.min_batch_size,
                        self.adaptive_batch_size - 1
                    )
                elif tracker.total_latency * 1000 < self.target_latency_ms * 0.7:
                    # Increase batch size if we have headroom
                    self.adaptive_batch_size = min(
                        self.max_batch_size,
                        self.adaptive_batch_size + 1
                    )
                
                # Clean up
                del self.frame_trackers[frame_id]
    
    def _preprocessing_worker(self):
        """Enhanced preprocessing worker with latency tracking"""
        batch = []
        
        while self._running:
            try:
                # Use adaptive batch size
                timeout = 0.005 if len(batch) > 0 else 0.01
                item = self.preprocess_queue.get(timeout=timeout)
                
                if item is None:
                    if batch:
                        self._process_batch(batch)
                    break
                
                # Track preprocessing start
                self._update_tracker(item.frame_idx, preprocess_start=time.time())
                
                batch.append(item)
                
                # Process batch when full or timeout
                if len(batch) >= self.adaptive_batch_size:
                    self._process_batch(batch)
                    batch = []
                    
            except queue.Empty:
                # Process partial batch on timeout to minimize latency
                if batch and len(batch) >= self.min_batch_size:
                    self._process_batch(batch)
                    batch = []
    
    def _process_batch(self, batch: List[FrameData]):
        """Process a batch with latency tracking"""
        # Track preprocessing end for all frames in batch
        for item in batch:
            self._update_tracker(item.frame_idx, preprocess_end=time.time())
        
        # Process batch
        processed = self.preprocess_stage.process_batch(batch)
        
        # Track GPU queue time
        for item in batch:
            self._update_tracker(item.frame_idx, gpu_queue_time=time.time())
        
        self.gpu_queue.put(processed)
    
    def _gpu_worker(self):
        """Enhanced GPU worker with model-aware batching and latency tracking"""
        while self._running:
            try:
                item = self.gpu_queue.get(timeout=0.01)
                if item is None:
                    break
                
                batch_data = item[0]
                
                # Track GPU start
                for frame in batch_data:
                    self._update_tracker(frame.frame_idx, gpu_start=time.time())
                
                # Process on GPU
                results = self.gpu_stage.process_batch(item)
                
                # Track GPU end
                for frame in batch_data:
                    self._update_tracker(frame.frame_idx, gpu_end=time.time())
                
                # Send results for post-processing
                for i, result in enumerate(results):
                    self._update_tracker(
                        batch_data[i].frame_idx,
                        postprocess_start=time.time()
                    )
                    self.postprocess_queue.put(result)
                    
            except queue.Empty:
                continue
    
    def _postprocessing_worker(self):
        """Enhanced post-processing worker with latency tracking"""
        while self._running:
            try:
                item = self.postprocess_queue.get(timeout=0.01)
                if item is None:
                    break
                
                # Process frame
                frame_idx, processed_frame = self.postprocess_stage.process(item)
                
                # Track postprocess end
                self._update_tracker(frame_idx, postprocess_end=time.time())
                
                # Add to output buffer
                self.output_buffer.add(frame_idx, processed_frame)
                
                # Finalize latency tracking
                self._finalize_frame_latency(frame_idx)
                
            except queue.Empty:
                continue
    
    def get_latency_report(self) -> Dict:
        """Get detailed latency report for this stream"""
        stats = self.latency_monitor.get_statistics()
        return {
            'stream_id': self.stream_id,
            'model': self.model_name,
            'adaptive_batch_size': self.adaptive_batch_size,
            'latency_stats': stats
        }