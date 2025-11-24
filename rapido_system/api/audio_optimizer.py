"""
Audio Optimizer for LiveKit Streaming
======================================

This module provides audio optimization features including:
- Jitter buffer for smooth playback
- Adaptive buffering based on network conditions
- Audio level normalization
- Silence detection and suppression
"""

import asyncio
import collections
import logging
import time
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class BufferMode(Enum):
    """Audio buffer modes"""
    LOW_LATENCY = "low_latency"    # 20-40ms buffer
    BALANCED = "balanced"           # 40-80ms buffer  
    HIGH_QUALITY = "high_quality"   # 80-160ms buffer


@dataclass
class AudioStats:
    """Audio processing statistics"""
    chunks_processed: int
    chunks_dropped: int
    buffer_underruns: int
    buffer_overruns: int
    average_level: float
    peak_level: float
    silence_ratio: float
    current_buffer_ms: float
    target_buffer_ms: float


class AudioOptimizer:
    """
    Optimizes audio for LiveKit streaming with jitter buffer and processing.
    
    Features:
    - Adaptive jitter buffer (20-160ms)
    - Dynamic buffer sizing based on network conditions
    - Audio level normalization
    - Silence detection and suppression
    - Smooth buffer management
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 initial_mode: BufferMode = BufferMode.BALANCED):
        """
        Initialize audio optimizer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            initial_mode: Initial buffer mode
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples_per_ms = sample_rate / 1000
        
        # Buffer configuration - ADJUSTED for better flow
        self.buffer_mode = initial_mode
        self.buffer_configs = {
            BufferMode.LOW_LATENCY: {'min': 80, 'target': 120, 'max': 160},
            BufferMode.BALANCED: {'min': 120, 'target': 160, 'max': 200},    # Reduced for better flow
            BufferMode.HIGH_QUALITY: {'min': 160, 'target': 200, 'max': 280}  # Reduced to prevent overrun
        }
        
        # Current buffer settings
        config = self.buffer_configs[self.buffer_mode]
        self.min_buffer_ms = config['min']
        self.target_buffer_ms = config['target']
        self.max_buffer_ms = config['max']
        
        # Audio buffer (stores samples as int16)
        self.buffer = collections.deque()
        self.buffer_lock = asyncio.Lock()
        
        # Statistics
        self.stats = AudioStats(
            chunks_processed=0,
            chunks_dropped=0,
            buffer_underruns=0,
            buffer_overruns=0,
            average_level=0.0,
            peak_level=0.0,
            silence_ratio=0.0,
            current_buffer_ms=0.0,
            target_buffer_ms=self.target_buffer_ms
        )
        
        # Initial buffer requirement - must fill before consuming
        self.initial_buffer_filled = False
        # Set to 80% of target buffer to start playback sooner
        self.initial_buffer_requirement = self.target_buffer_ms * 0.8  # 160ms for HIGH_QUALITY
        
        # Audio processing - DISABLED to prevent crackling
        self.normalization_enabled = False  # Disabled - causes crackling
        self.target_level = 0.7  # Target RMS level (0-1)
        self.silence_threshold = 0.01  # RMS below this is silence
        
        # Smoothing for level changes
        self.current_gain = 1.0
        self.gain_smoothing = 0.95  # Smoothing factor
        
        # Network condition tracking - DELAYED START
        self.network_quality_samples = collections.deque(maxlen=10)
        self.last_adaptation_time = time.time() + 30.0  # Don't adapt for first 30 seconds
        self.adaptation_interval = 10.0  # Adapt every 10 seconds (was 5)
        self.mode_locked = True  # Lock mode initially to prevent switching
        self.mode_lock_duration = 30.0  # Lock for 30 seconds
        self.mode_lock_start = time.time()
        
        # Simple fade-in/out for silence transitions (minimal processing)
        self.fade_samples = 80  # 5ms at 16kHz - short fade to prevent clicks
        
    async def add_audio_chunk(self, audio_chunk: bytes) -> bool:
        """
        Add audio chunk to the buffer with processing.
        
        Args:
            audio_chunk: PCM audio data as bytes
            
        Returns:
            True if chunk was added, False if dropped
        """
        if not audio_chunk:
            return False
            
        async with self.buffer_lock:
            # Convert to samples
            samples = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Check buffer size
            current_buffer_samples = len(self.buffer)
            max_buffer_samples = int(self.max_buffer_ms * self.samples_per_ms)
            
            if current_buffer_samples + len(samples) > max_buffer_samples:
                # Buffer overflow - drop oldest samples
                overflow = (current_buffer_samples + len(samples)) - max_buffer_samples
                for _ in range(overflow):
                    if self.buffer:
                        self.buffer.popleft()
                self.stats.buffer_overruns += 1
                
                if self.stats.buffer_overruns % 10 == 0:
                    logger.warning(f"⚠️ Audio buffer overflow #{self.stats.buffer_overruns}")
            
            # Process audio if enabled
            if self.normalization_enabled:
                samples = self._process_audio(samples)
            
            # Add to buffer
            self.buffer.extend(samples)
            self.stats.chunks_processed += 1
            
            # Update buffer size metric
            self.stats.current_buffer_ms = len(self.buffer) / self.samples_per_ms
            
            return True
    
    async def get_audio_frame(self, duration_ms: float = 40) -> Optional[bytes]:
        """
        Get audio frame of specified duration from buffer.
        
        Args:
            duration_ms: Frame duration in milliseconds
            
        Returns:
            Audio frame as bytes or None if not enough data
        """
        samples_needed = int(duration_ms * self.samples_per_ms)
        
        async with self.buffer_lock:
            current_buffer_ms = len(self.buffer) / self.samples_per_ms
            
            # Check if initial buffer has been filled
            if not self.initial_buffer_filled:
                if current_buffer_ms >= self.initial_buffer_requirement:
                    self.initial_buffer_filled = True
                    logger.info(f"✅ Audio buffer filled to {current_buffer_ms:.0f}ms - starting playback")
                else:
                    # Still filling initial buffer - return silence
                    silence = np.zeros(samples_needed, dtype=np.int16)
                    return silence.tobytes()
            
            # Maintain minimum buffer level to prevent underruns
            # Only generate silence if buffer is CRITICALLY low (below 40ms - one frame)
            min_playback_buffer_ms = 40  # One frame worth
            if current_buffer_ms < min_playback_buffer_ms and current_buffer_ms > 0:
                # Buffer critically low - return silence to let it fill
                self.stats.buffer_underruns += 1
                silence = np.zeros(samples_needed, dtype=np.int16)
                # Apply simple fade to avoid click
                if self.fade_samples > 0:
                    fade_in = np.linspace(0, 1, min(self.fade_samples, len(silence)))
                    silence[:len(fade_in)] = (silence[:len(fade_in)] * fade_in).astype(np.int16)
                return silence.tobytes()
            
            # Check if we have enough for the frame
            if len(self.buffer) < samples_needed:
                # Buffer underrun
                self.stats.buffer_underruns += 1
                
                # Generate silence to maintain stream
                silence = np.zeros(samples_needed, dtype=np.int16)
                return silence.tobytes()
            
            # Extract samples
            samples = []
            for _ in range(samples_needed):
                if self.buffer:
                    samples.append(self.buffer.popleft())
                else:
                    # Pad with silence if needed
                    samples.append(0)
            
            # Update statistics
            self.stats.current_buffer_ms = len(self.buffer) / self.samples_per_ms
            
            # Convert to bytes
            audio_array = np.array(samples, dtype=np.int16)
            return audio_array.tobytes()
    
    def _process_audio(self, samples: np.ndarray) -> np.ndarray:
        """
        Process audio samples (normalization, etc).
        
        Args:
            samples: Audio samples as int16 array
            
        Returns:
            Processed samples
        """
        # Convert to float for processing
        float_samples = samples.astype(np.float32) / 32768.0
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(float_samples ** 2))
        
        # Update statistics
        self.stats.average_level = self.stats.average_level * 0.9 + rms * 0.1
        self.stats.peak_level = max(self.stats.peak_level * 0.999, np.max(np.abs(float_samples)))
        
        # Check for silence
        if rms < self.silence_threshold:
            self.stats.silence_ratio = min(1.0, self.stats.silence_ratio + 0.01)
            # Return original samples for silence (no processing)
            return samples
        else:
            self.stats.silence_ratio = max(0.0, self.stats.silence_ratio - 0.01)
        
        # Calculate target gain for normalization
        if rms > 0:
            target_gain = self.target_level / rms
            # Limit gain to prevent excessive amplification
            target_gain = min(target_gain, 5.0)
        else:
            target_gain = 1.0
        
        # Smooth gain changes to prevent artifacts
        self.current_gain = self.current_gain * self.gain_smoothing + target_gain * (1 - self.gain_smoothing)
        
        # Apply gain
        float_samples *= self.current_gain
        
        # Soft clipping to prevent harsh distortion
        float_samples = np.tanh(float_samples * 0.8) * 1.25
        
        # Convert back to int16
        processed = (float_samples * 32767).clip(-32768, 32767).astype(np.int16)
        
        return processed
    
    async def adapt_to_network(self, packet_loss: float, rtt_ms: float):
        """
        Adapt buffer size based on network conditions.
        
        Args:
            packet_loss: Packet loss percentage (0-100)
            rtt_ms: Round trip time in milliseconds
        """
        current_time = time.time()
        
        # Check if mode is locked
        if self.mode_locked:
            if current_time - self.mode_lock_start < self.mode_lock_duration:
                # Still locked, don't adapt
                return
            else:
                # Unlock after duration
                self.mode_locked = False
                logger.info(f"🔓 Audio buffer mode unlocked after {self.mode_lock_duration}s")
        
        # Track network quality
        quality_score = 1.0
        if packet_loss > 5:
            quality_score *= 0.5
        elif packet_loss > 2:
            quality_score *= 0.75
            
        if rtt_ms > 300:
            quality_score *= 0.5
        elif rtt_ms > 150:
            quality_score *= 0.75
            
        self.network_quality_samples.append(quality_score)
        
        # Adapt periodically
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return
            
        self.last_adaptation_time = current_time
        
        # Calculate average quality
        avg_quality = sum(self.network_quality_samples) / len(self.network_quality_samples) if self.network_quality_samples else 0.5
        
        # Determine appropriate buffer mode - VERY CONSERVATIVE
        # Stay in HIGH_QUALITY unless network is perfect
        if avg_quality > 0.98:  # Near perfect conditions required
            new_mode = BufferMode.BALANCED  # Never go to LOW_LATENCY
        else:  # Any issues = stay in HIGH_QUALITY
            new_mode = BufferMode.HIGH_QUALITY
        
        # Apply new mode if changed
        if new_mode != self.buffer_mode:
            await self.set_buffer_mode(new_mode)
    
    async def set_buffer_mode(self, mode: BufferMode):
        """
        Change buffer mode.
        
        Args:
            mode: Target buffer mode
        """
        # Don't allow mode changes if locked
        if self.mode_locked:
            current_time = time.time()
            remaining = self.mode_lock_duration - (current_time - self.mode_lock_start)
            if remaining > 0:
                logger.debug(f"🔒 Audio buffer mode change blocked (locked for {remaining:.1f}s more)")
                return
        
        if mode == self.buffer_mode:
            return
            
        self.buffer_mode = mode
        config = self.buffer_configs[mode]
        
        async with self.buffer_lock:
            self.min_buffer_ms = config['min']
            self.target_buffer_ms = config['target']
            self.max_buffer_ms = config['max']
            self.stats.target_buffer_ms = self.target_buffer_ms
            
            logger.info(
                f"🎚️ Audio buffer mode changed to {mode.value}: "
                f"{self.min_buffer_ms}-{self.target_buffer_ms}-{self.max_buffer_ms}ms"
            )
    
    def get_stats(self) -> AudioStats:
        """Get current statistics"""
        return AudioStats(
            chunks_processed=self.stats.chunks_processed,
            chunks_dropped=self.stats.chunks_dropped,
            buffer_underruns=self.stats.buffer_underruns,
            buffer_overruns=self.stats.buffer_overruns,
            average_level=self.stats.average_level,
            peak_level=self.stats.peak_level,
            silence_ratio=self.stats.silence_ratio,
            current_buffer_ms=self.stats.current_buffer_ms,
            target_buffer_ms=self.stats.target_buffer_ms
        )
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats.chunks_processed = 0
        self.stats.chunks_dropped = 0
        self.stats.buffer_underruns = 0
        self.stats.buffer_overruns = 0
        self.stats.average_level = 0.0
        self.stats.peak_level = 0.0
        self.stats.silence_ratio = 0.0


class AudioStreamSynchronizer:
    """
    Synchronizes audio with video frames for perfect lip sync.
    
    Features:
    - Frame-accurate audio/video alignment
    - Automatic drift correction
    - Timestamp-based synchronization
    """
    
    def __init__(self,
                 video_fps: float = 25.0,
                 audio_sample_rate: int = 16000):
        """
        Initialize audio/video synchronizer.
        
        Args:
            video_fps: Video frame rate
            audio_sample_rate: Audio sample rate
        """
        self.video_fps = video_fps
        self.audio_sample_rate = audio_sample_rate
        self.ms_per_frame = 1000.0 / video_fps
        self.samples_per_frame = int(audio_sample_rate / video_fps)
        
        # Synchronization state
        self.video_frame_count = 0
        self.audio_sample_count = 0
        self.sync_offset_ms = 0.0
        
        # Drift detection
        self.drift_samples = collections.deque(maxlen=100)
        self.max_drift_ms = 40.0  # Maximum acceptable drift
        
    def calculate_audio_for_frame(self, frame_index: int) -> Tuple[int, int]:
        """
        Calculate audio sample range for a video frame.
        
        Args:
            frame_index: Video frame index
            
        Returns:
            Tuple of (start_sample, end_sample)
        """
        start_sample = int(frame_index * self.samples_per_frame)
        end_sample = int((frame_index + 1) * self.samples_per_frame)
        
        # Apply sync offset
        offset_samples = int(self.sync_offset_ms * self.audio_sample_rate / 1000)
        start_sample += offset_samples
        end_sample += offset_samples
        
        return (start_sample, end_sample)
    
    def update_sync(self, video_frames: int, audio_samples: int):
        """
        Update synchronization based on current counts.
        
        Args:
            video_frames: Number of video frames processed
            audio_samples: Number of audio samples processed
        """
        self.video_frame_count = video_frames
        self.audio_sample_count = audio_samples
        
        # Calculate expected vs actual
        expected_samples = video_frames * self.samples_per_frame
        drift_samples = audio_samples - expected_samples
        drift_ms = (drift_samples / self.audio_sample_rate) * 1000
        
        self.drift_samples.append(drift_ms)
        
        # Calculate average drift
        if len(self.drift_samples) >= 10:
            avg_drift = sum(self.drift_samples) / len(self.drift_samples)
            
            # Apply correction if drift is significant
            if abs(avg_drift) > self.max_drift_ms:
                logger.warning(f"⚠️ A/V drift detected: {avg_drift:.1f}ms, applying correction")
                self.sync_offset_ms = -avg_drift
                self.drift_samples.clear()
    
    def get_sync_status(self) -> dict:
        """Get current synchronization status"""
        expected_samples = self.video_frame_count * self.samples_per_frame
        drift_samples = self.audio_sample_count - expected_samples
        drift_ms = (drift_samples / self.audio_sample_rate) * 1000 if self.audio_sample_rate > 0 else 0
        
        return {
            'video_frames': self.video_frame_count,
            'audio_samples': self.audio_sample_count,
            'drift_ms': drift_ms,
            'sync_offset_ms': self.sync_offset_ms,
            'in_sync': abs(drift_ms) < self.max_drift_ms
        }
