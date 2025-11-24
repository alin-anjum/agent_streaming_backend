"""
Frame Pacer for Consistent Video Delivery
==========================================

This module provides frame pacing functionality to ensure consistent
frame delivery at the target frame rate, regardless of input variations.
"""

import asyncio
import time
import collections
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Metrics for frame pacing performance"""
    target_fps: float
    actual_fps: float
    frames_delivered: int
    frames_dropped: int
    frames_duplicated: int
    average_latency_ms: float
    max_latency_ms: float
    buffer_utilization: float


class FramePacer:
    """
    Ensures frames are delivered at a consistent rate.
    
    Features:
    - Maintains target FPS regardless of input rate
    - Handles both slow and fast input scenarios
    - Provides frame duplication for underruns
    - Implements frame dropping for overruns
    - Tracks detailed performance metrics
    """
    
    def __init__(self, 
                 target_fps: float = 25.0,
                 buffer_size: int = 10,
                 allow_duplication: bool = True,
                 allow_dropping: bool = True):
        """
        Initialize frame pacer.
        
        Args:
            target_fps: Target frame rate for output
            buffer_size: Size of frame buffer
            allow_duplication: Whether to duplicate frames on underrun
            allow_dropping: Whether to drop frames on overrun
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.buffer_size = buffer_size
        self.allow_duplication = allow_duplication
        self.allow_dropping = allow_dropping
        
        # Frame buffer
        self.buffer = collections.deque(maxlen=buffer_size)
        self.last_frame = None  # For duplication
        
        # Timing
        self.next_frame_time = None
        self.start_time = None
        
        # Metrics
        self.frames_delivered = 0
        self.frames_dropped = 0
        self.frames_duplicated = 0
        self.latency_samples = collections.deque(maxlen=100)
        
        # Control
        self.is_running = False
        self.delivery_callback: Optional[Callable] = None
        
    def add_frame(self, frame: Any, timestamp: Optional[float] = None) -> bool:
        """
        Add a frame to the buffer.
        
        Args:
            frame: The frame data
            timestamp: Optional timestamp for latency tracking
            
        Returns:
            True if frame was added, False if dropped
        """
        timestamp = timestamp or time.time()
        
        # Check if buffer is full
        if len(self.buffer) >= self.buffer_size:
            if self.allow_dropping:
                # Drop oldest frame
                dropped = self.buffer.popleft()
                self.frames_dropped += 1
                logger.debug(f"Dropped frame due to buffer overflow (dropped: {self.frames_dropped})")
            else:
                return False
        
        # Add frame to buffer
        self.buffer.append({
            'frame': frame,
            'timestamp': timestamp,
            'added_time': time.time()
        })
        
        # Keep last frame for potential duplication
        self.last_frame = frame
        
        return True
    
    async def start(self, delivery_callback: Callable):
        """
        Start the frame pacer.
        
        Args:
            delivery_callback: Async function to call with each frame
        """
        if self.is_running:
            logger.warning("Frame pacer already running")
            return
            
        self.delivery_callback = delivery_callback
        self.is_running = True
        self.start_time = time.time()
        self.next_frame_time = self.start_time
        
        logger.info(f"🎬 Frame pacer started at {self.target_fps} FPS")
        
        # Start delivery loop
        await self._delivery_loop()
    
    async def _delivery_loop(self):
        """Main delivery loop that maintains target frame rate"""
        consecutive_underruns = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time for next frame
                if current_time >= self.next_frame_time:
                    frame_to_deliver = None
                    latency_ms = 0
                    
                    # Try to get frame from buffer
                    if self.buffer:
                        frame_data = self.buffer.popleft()
                        frame_to_deliver = frame_data['frame']
                        
                        # Calculate latency
                        latency_ms = (current_time - frame_data['added_time']) * 1000
                        self.latency_samples.append(latency_ms)
                        
                        consecutive_underruns = 0
                        
                    elif self.allow_duplication and self.last_frame is not None:
                        # No frames available, duplicate last frame
                        frame_to_deliver = self.last_frame
                        self.frames_duplicated += 1
                        consecutive_underruns += 1
                        
                        if consecutive_underruns % 25 == 0:  # Log every second of underrun
                            logger.warning(
                                f"⚠️ Frame underrun - duplicating frames "
                                f"(duplicated: {self.frames_duplicated})"
                            )
                    
                    # Deliver frame if available
                    if frame_to_deliver is not None and self.delivery_callback:
                        await self.delivery_callback(frame_to_deliver)
                        self.frames_delivered += 1
                        
                        # Log progress
                        if self.frames_delivered % 100 == 0:
                            metrics = self.get_metrics()
                            logger.info(
                                f"📊 Pacer: {metrics.actual_fps:.1f}/{metrics.target_fps:.0f} FPS | "
                                f"Delivered: {metrics.frames_delivered} | "
                                f"Dropped: {metrics.frames_dropped} | "
                                f"Duplicated: {metrics.frames_duplicated} | "
                                f"Latency: {metrics.average_latency_ms:.1f}ms | "
                                f"Buffer: {metrics.buffer_utilization:.0%}"
                            )
                    
                    # Schedule next frame
                    self.next_frame_time += self.frame_interval
                    
                    # Prevent drift - reset if too far behind
                    if current_time - self.next_frame_time > 1.0:
                        logger.warning("Frame pacer fell behind, resetting timing")
                        self.next_frame_time = current_time
                
                # Sleep until next frame time
                sleep_time = max(0, self.next_frame_time - time.time())
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    # Yield control even if no sleep needed
                    await asyncio.sleep(0)
                    
            except Exception as e:
                logger.error(f"Error in frame delivery loop: {e}")
                await asyncio.sleep(self.frame_interval)
    
    def stop(self):
        """Stop the frame pacer"""
        self.is_running = False
        logger.info("🛑 Frame pacer stopped")
    
    def get_metrics(self) -> FrameMetrics:
        """Get current performance metrics"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frames_delivered / elapsed if elapsed > 0 else 0
        
        avg_latency = sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0
        max_latency = max(self.latency_samples) if self.latency_samples else 0
        
        buffer_util = len(self.buffer) / self.buffer_size if self.buffer_size > 0 else 0
        
        return FrameMetrics(
            target_fps=self.target_fps,
            actual_fps=actual_fps,
            frames_delivered=self.frames_delivered,
            frames_dropped=self.frames_dropped,
            frames_duplicated=self.frames_duplicated,
            average_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            buffer_utilization=buffer_util
        )
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.frames_delivered = 0
        self.frames_dropped = 0
        self.frames_duplicated = 0
        self.latency_samples.clear()
        self.start_time = time.time()


class AdaptiveFramePacer(FramePacer):
    """
    Advanced frame pacer that adapts to input patterns.
    
    Additional features:
    - Dynamic buffer sizing based on input rate
    - Predictive frame scheduling
    - Quality-based frame dropping
    """
    
    def __init__(self, 
                 target_fps: float = 25.0,
                 min_buffer_size: int = 5,
                 max_buffer_size: int = 20):
        """
        Initialize adaptive frame pacer.
        
        Args:
            target_fps: Target frame rate
            min_buffer_size: Minimum buffer size
            max_buffer_size: Maximum buffer size
        """
        super().__init__(target_fps, max_buffer_size, True, True)
        
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        
        # Input rate tracking
        self.input_timestamps = collections.deque(maxlen=50)
        self.input_rate = 0.0
        
        # Adaptation parameters
        self.adaptation_interval = 5.0  # Adapt every 5 seconds
        self.last_adaptation_time = 0
        
    def add_frame(self, frame: Any, timestamp: Optional[float] = None) -> bool:
        """Add frame and track input rate"""
        # Track input timing
        current_time = time.time()
        self.input_timestamps.append(current_time)
        
        # Calculate input rate
        if len(self.input_timestamps) >= 2:
            time_span = self.input_timestamps[-1] - self.input_timestamps[0]
            if time_span > 0:
                self.input_rate = len(self.input_timestamps) / time_span
        
        # Adapt buffer size if needed
        if current_time - self.last_adaptation_time > self.adaptation_interval:
            self._adapt_buffer_size()
            self.last_adaptation_time = current_time
        
        return super().add_frame(frame, timestamp)
    
    def _adapt_buffer_size(self):
        """Adapt buffer size based on input/output rate mismatch"""
        if self.input_rate <= 0:
            return
            
        # Calculate rate difference
        rate_ratio = self.input_rate / self.target_fps
        
        # Determine optimal buffer size
        if rate_ratio > 1.2:  # Input faster than output
            # Increase buffer to handle bursts
            new_size = min(self.max_buffer_size, int(self.buffer_size * 1.5))
        elif rate_ratio < 0.8:  # Input slower than output
            # Decrease buffer to reduce latency
            new_size = max(self.min_buffer_size, int(self.buffer_size * 0.75))
        else:
            # Rates are balanced, use medium buffer
            new_size = (self.min_buffer_size + self.max_buffer_size) // 2
        
        if new_size != self.buffer_size:
            logger.info(
                f"📈 Adapting buffer size: {self.buffer_size} → {new_size} "
                f"(input: {self.input_rate:.1f} fps, output: {self.target_fps:.1f} fps)"
            )
            self.buffer_size = new_size
            # Note: deque maxlen can't be changed, so we track separately
            
    def should_drop_frame(self, frame_data: dict) -> bool:
        """
        Determine if a frame should be dropped based on quality metrics.
        
        Args:
            frame_data: Frame information including quality metrics
            
        Returns:
            True if frame should be dropped
        """
        # Drop if buffer is critically full
        if len(self.buffer) >= self.buffer_size * 0.9:
            return True
            
        # Drop if frame is too old
        age = time.time() - frame_data.get('timestamp', 0)
        if age > 0.2:  # 200ms old
            return True
            
        # Keep frame
        return False
