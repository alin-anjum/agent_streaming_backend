# audio_stream_generator.py
"""
Audio stream generator for real-time audio processing and resampling.

This module handles audio streaming with resampling capabilities,
supporting both synchronous and asynchronous audio chunk processing.
"""

import torch
import torchaudio.transforms as T
import asyncio
import numpy as np
import queue
import threading
import time
import logging

# Get logger
logger = logging.getLogger(__name__)

# Define supported stream markers
START_MARKERS = {b'start_of_stream'}
END_MARKERS = {b'end_of_stream'}
ALL_MARKERS = START_MARKERS | END_MARKERS

# Logging counter for audio processing
audio_none_log_counter = 0

class AudioStreamGenerator:
    """Generator that yields audio chunks as they arrive from websocket"""
    
    def __init__(self, input_sample_rate: int = 24000, pause_threshold: int = 10):
        self.audio_queue = queue.Queue()
        self.is_active = True
        self._lock = threading.Lock()
        
        self.input_sample_rate = input_sample_rate  # Original sample rate from client
        self.target_sample_rate = 16000  # Target sample rate for processing
        self.bytes_per_sample = 2  # 16-bit
        self.frame_duration_ms = 40  # 25 FPS = 40ms per frame
        
        if self.input_sample_rate != self.target_sample_rate:
            self.resampler = T.Resample(
                orig_freq=self.input_sample_rate, 
                new_freq=self.target_sample_rate
            )
        else:
            self.resampler = None
        
        logger.info(f"AudioStreamGenerator: {input_sample_rate}Hz → {self.target_sample_rate}Hz")
        self.pause_threshold = pause_threshold
        
        # Silence generation rate limiting
        self.last_silence_time = 0
        self.silence_chunk_duration = 0.15  # 150ms chunks for better responsiveness
        self.silence_backlog = 0.0  # Track how much silence we owe
        
    async def add_audio_chunk_async(self, audio_chunk: bytes):
        """Add new audio chunk from websocket (with async resampling if needed)"""
        with self._lock:
            if self.is_active:
                # Handle special stream markers
                if audio_chunk in ALL_MARKERS:
                    self.audio_queue.put(audio_chunk)
                    return
                
                # Resample audio asynchronously if necessary
                if self.resampler:
                    resampled_chunk = await asyncio.get_event_loop().run_in_executor(
                        None, self._resample_audio_chunk, audio_chunk
                    )
                    self.audio_queue.put(resampled_chunk)
                else:
                    self.audio_queue.put(audio_chunk)
    
    def add_audio_chunk(self, audio_chunk: bytes):
        """Synchronous version for backward compatibility"""
        with self._lock:
            if self.is_active:
                # Handle special stream markers
                if audio_chunk in ALL_MARKERS:
                    self.audio_queue.put(audio_chunk)
                    return
                
                # Resample audio if necessary (blocking version)
                if self.resampler:
                    resampled_chunk = self._resample_audio_chunk(audio_chunk)
                    self.audio_queue.put(resampled_chunk)
                else:
                    self.audio_queue.put(audio_chunk)
    
    def _resample_audio_chunk(self, audio_chunk: bytes) -> bytes:
        """Resample audio chunk from input_sample_rate to target_sample_rate using torchaudio"""
        if not self.resampler:
            return audio_chunk

        try:
            # Convert bytes to numpy array of int16
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Skip resampling if no data
            if len(audio_int16) == 0:
                return audio_chunk
            
            # Convert to float32 tensor and normalize
            audio_float32 = torch.from_numpy(audio_int16.astype(np.float32) / 32768.0)
            
            # Resample
            resampled_audio = self.resampler(audio_float32)

            # Denormalize, clamp, and convert back to int16
            resampled_int16 = (resampled_audio * 32768.0).clamp(-32768, 32767).to(torch.int16)
            
            result = resampled_int16.numpy().tobytes()
            
            logger.debug(f"torchaudio resampled {len(audio_chunk)} bytes "
                         f"→ {len(result)} bytes")
            
            return result
        except Exception as e:
            logger.error(f"Error resampling audio with torchaudio: {e}")
            # Return original chunk on error as a simple fallback
            return audio_chunk
    
    def _generate_rate_limited_silence(self):
        """Generate silence at real-time rate to prevent backlog buildup"""
        current_time = time.time()
        
        # Initialize timing on first call
        if self.last_silence_time == 0:
            self.last_silence_time = current_time
            self.silence_backlog = self.silence_chunk_duration
        
        # Calculate how much silence we should have generated by now
        time_elapsed = current_time - self.last_silence_time
        silence_owed = time_elapsed + self.silence_backlog
        
        # Generate silence if we owe time
        min_threshold = self.silence_chunk_duration
        
        if silence_owed >= min_threshold:
            # Generate one chunk of silence (150ms)
            num_samples = int(self.target_sample_rate * self.silence_chunk_duration)  # 2400 samples for 150ms
            silent_chunk = (b'\x00\x00') * num_samples  # 2 bytes per sample (16-bit)
            
            # Update timing
            self.silence_backlog = silence_owed - self.silence_chunk_duration
            self.last_silence_time = current_time
            
            logger.debug(f"Yielding {self.silence_chunk_duration*1000:.0f}ms silence, backlog: {self.silence_backlog*1000:.0f}ms")
            return silent_chunk
        else:
            # Not time for another silence chunk yet - return empty to yield control
            # This prevents tight looping and allows other audio to arrive
            return b''
    
    def close(self):
        """Signal end of stream"""
        with self._lock:
            self.is_active = False
            self.audio_queue.put(None)  # Sentinel to stop generator
        
    def __iter__(self):
        return self
        
    def __next__(self):
        global audio_none_log_counter
        
        try:
            # A smaller timeout to be more responsive in a streaming context.
            chunk = self.audio_queue.get(block=False)
            if chunk is None:  # Sentinel value
                raise StopIteration
            
            logger.info(f"AudioStreamGenerator: Got audio chunk of {len(chunk)} bytes")
            # Reset silence tracking when we get real audio
            self.silence_backlog = 0.0
            self.last_silence_time = time.time()
            return chunk
        except queue.Empty:
            with self._lock:
                if not self.is_active:
                    raise StopIteration
            
            # Log "returning None" only every 100 times
            audio_none_log_counter += 1
            # if audio_none_log_counter % 100 == 0:
            #     logger.info(f"AudioStreamGenerator: Queue is empty, hence returning None [COUNT: {audio_none_log_counter}]")
            
            return None
