"""
Audio Smoother for SyncTalk audio chunks
Fixes crackling by smoothing boundaries between 40ms chunks
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AudioSmoother:
    """
    Smooths audio chunk boundaries to prevent crackling.
    SyncTalk sends 40ms chunks (640 samples at 16kHz) with hard boundaries.
    This class applies crossfading to smooth the transitions.
    """
    
    def __init__(self, sample_rate: int = 16000, crossfade_ms: float = 2.0):
        """
        Initialize the audio smoother.
        
        Args:
            sample_rate: Audio sample rate (16000 for SyncTalk)
            crossfade_ms: Crossfade duration in milliseconds (2ms default)
        """
        self.sample_rate = sample_rate
        self.crossfade_samples = int(sample_rate * crossfade_ms / 1000)
        
        # Store the tail of the previous chunk for crossfading
        self.previous_tail = None
        self.chunk_count = 0
        
        logger.info(f"🔊 Audio smoother initialized: {crossfade_ms}ms crossfade ({self.crossfade_samples} samples)")
    
    def smooth_chunk(self, audio_bytes: bytes) -> bytes:
        """
        Smooth an audio chunk by applying crossfade with the previous chunk.
        
        Args:
            audio_bytes: Raw PCM audio bytes from SyncTalk (int16)
            
        Returns:
            Smoothed audio bytes (SAME SIZE as input)
        """
        if not audio_bytes:
            return audio_bytes
        
        # Store original length to ensure we return same size
        original_length = len(audio_bytes)
        
        # Convert bytes to numpy array
        audio_samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        if len(audio_samples) == 0:
            return audio_bytes
        
        # Apply crossfade if we have a previous chunk
        if self.previous_tail is not None and len(self.previous_tail) > 0:
            fade_length = min(self.crossfade_samples, len(audio_samples), len(self.previous_tail))
            
            if fade_length > 0:
                # Create smooth fade curves
                fade_out = np.linspace(1.0, 0.0, fade_length)
                fade_in = np.linspace(0.0, 1.0, fade_length)
                
                # Crossfade the beginning of this chunk with the tail of the previous
                crossfaded = (
                    self.previous_tail[-fade_length:] * fade_out +
                    audio_samples[:fade_length] * fade_in
                )
                
                # Replace the beginning of the current chunk with crossfaded audio
                audio_samples[:fade_length] = crossfaded
        
        # Store the tail of this chunk for next crossfade
        tail_length = min(self.crossfade_samples * 2, len(audio_samples))  # Store a bit extra
        self.previous_tail = audio_samples[-tail_length:].copy()
        
        # Convert back to int16 bytes
        smoothed_bytes = audio_samples.astype(np.int16).tobytes()
        
        # CRITICAL: Ensure we return exactly the same size
        if len(smoothed_bytes) != original_length:
            logger.error(f"Audio smoother size mismatch! Input: {original_length}, Output: {len(smoothed_bytes)}")
        
        self.chunk_count += 1
        if self.chunk_count % 100 == 0:
            logger.debug(f"Smoothed {self.chunk_count} audio chunks")
        
        return smoothed_bytes
    
    def reset(self):
        """Reset the smoother state (for new streams)"""
        self.previous_tail = None
        self.chunk_count = 0
        logger.debug("Audio smoother reset")