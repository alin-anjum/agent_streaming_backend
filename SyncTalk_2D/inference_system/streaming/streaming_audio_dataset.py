# inference_system/streaming/streaming_audio_dataset.py

import numpy as np
import torch
from typing import Optional, List, Tuple
from collections import deque
import librosa

from utils import load_wav, preemphasis, melspectrogram
from inference_system.utils.profiler import PerformanceProfiler

class StreamingAudDataset:
    """
    Streaming version of AudDataset that processes PCM chunks without file I/O
    V2: Stores all audio and uses exact same processing as batch version
    """
    
    def __init__(self, audio_encoder, profiler: Optional[PerformanceProfiler] = None):
        # START WITH EMPTY ARRAY, NOT PRE-ALLOCATED!
        self.all_audio = np.array([], dtype=np.float32)  # Empty array
        self.mel_frames = []
        self.samples_processed = 0
        self.profiler = profiler if profiler else PerformanceProfiler("StreamingAudDataset")
        
        # Store audio encoder reference
        self.audio_encoder = audio_encoder
        
        # Constants matching the original
        self.sample_rate = 16000
        self.hop_length = 200
        self.n_fft = 800
        
        # Audio feature caches
        self.audio_feature_buffer = {}
        self.mel_window_cache = {}
        self.highest_frame_processed = -1
        self.circuit_breaker = False
        self.frames_returned_as_ready = set()
        
        # Keep these for tracking
        self._chunk_count = 0
        self._last_mel_update_samples = 0

        self._pending_audio_buffer = []  # Buffer chunks before processing
        self._pending_audio_samples = 0   # Track buffered samples
        self._MIN_PROCESS_SAMPLES = 32000  # Only 2 second buffer
        self._last_mel_computed_samples = 0
        self._mel_computation_overlap = self.n_fft - self.hop_length  # 600 samples overlap
        
        # NEW: Track mel computation more carefully
        self._last_audio_length_for_mel = 0  # Length of audio when we last computed mel
        self.silent_frame_indices = set()
    
    @property
    def highest_frame_processed(self):
        return self._highest_frame_processed

    @highest_frame_processed.setter
    def highest_frame_processed(self, value):
        self._highest_frame_processed = value

    def get_audio_for_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get the raw audio samples that correspond to a specific video frame.
        This uses the same frame-to-audio mapping as the mel spectrogram generation.
        
        For 25 FPS video and 16kHz audio:
        - Each video frame = 40ms = 640 audio samples
        
        Args:
            frame_idx: Video frame index
            
        Returns:
            numpy array of audio samples (float32) or None if not available
        """
        # Same calculation as in AudDataset
        samples_per_frame = int(self.sample_rate / 25.0)  # 640 for 16kHz/25fps
        start_sample = frame_idx * samples_per_frame
        end_sample = start_sample + samples_per_frame
        
        # Check if we have enough audio
        if start_sample >= len(self.all_audio):
            return np.zeros(samples_per_frame, dtype=np.float32)
        
        if end_sample > len(self.all_audio):
            # Partial frame at the end - pad with zeros
            partial_audio = self.all_audio[start_sample:]
            padding = np.zeros(samples_per_frame - len(partial_audio), dtype=np.float32)
            return np.concatenate([partial_audio, padding])
        
        return self.all_audio[start_sample:end_sample].copy()

    def get_pending_frames_needing_silence(self, submitted_frames: set) -> tuple[List[int], float]:
        """
        Find frames that are waiting for audio and calculate silence needed.
        
        Args:
            submitted_frames: Set of frames already submitted for processing
            
        Returns:
            (list of pending frame indices, seconds of silence needed)
        """
        pending_frames = []
        max_silence_needed = 0.0
        
        # Find the highest frame that could theoretically be processed
        # based on current audio
        max_possible_frame = int((len(self.mel_frames) - 16) / 80. * 25.) if len(self.mel_frames) >= 16 else -1
        
        # Check frames that could be processed but haven't been
        for frame_idx in range(max_possible_frame + 1, max_possible_frame + 10):  # Check next 10 frames
            if frame_idx in submitted_frames:
                continue
                
            # Check if this frame needs audio to complete its window
            mel_center = int(80. * (frame_idx / 25.))
            mel_end_needed = mel_center + 8
            
            if mel_end_needed > len(self.mel_frames):
                pending_frames.append(frame_idx)
                mel_deficit = mel_end_needed - len(self.mel_frames)
                silence_needed = mel_deficit / 80.0
                max_silence_needed = max(max_silence_needed, silence_needed)
        
        return pending_frames, max_silence_needed

    def add_silence(self, duration_seconds: float):
        """Add silent audio to the buffer efficiently."""
        num_samples = int(duration_seconds * self.sample_rate)
        
        # Calculate frame indices for this silence duration
        samples_per_frame = self.sample_rate / 25.0
        start_sample_idx = len(self.all_audio)
        
        # Calculate the frame index where this silence begins
        start_frame = int(start_sample_idx / samples_per_frame)
        num_frames = int(num_samples / samples_per_frame)
        
        for i in range(num_frames):
            self.silent_frame_indices.add(start_frame + i)
            
        silence = np.zeros(num_samples, dtype=np.float32)
        
        # Add silence to the audio buffer
        self.all_audio = np.concatenate([self.all_audio, silence])
        self.samples_processed = len(self.all_audio)
        
        # Compute mel incrementally
        self._compute_mel_incremental()

    def is_silent_frame(self, frame_idx: int) -> bool:
        """Check if a frame index corresponds to generated silence."""
        return frame_idx in self.silent_frame_indices
        
    def get_current_frame_count(self) -> int:
        """Get the current number of video frames that can be generated"""
        return self.get_video_frame_count()
        
    def add_audio_chunk(self, audio_chunk: bytes):
        """Add PCM audio chunk with optimized batch processing"""
        if self.circuit_breaker:
            return

        # Convert PCM to float
        pcm_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0
        
        # Add to pending buffer
        self._pending_audio_buffer.append(pcm_float)
        self._pending_audio_samples += len(pcm_float)
        
        # Process when we have enough for several frames (REVERTED - smaller chunks were worse)
        # 32000 samples = 2 seconds = ~50 frames (BACK TO ORIGINAL)
        MIN_SAMPLES_TO_PROCESS = 32000  # REVERTED to original value
        
        should_process = (
            self._pending_audio_samples >= MIN_SAMPLES_TO_PROCESS or
            # Also process on smaller chunks if it's been too long
            (self._pending_audio_samples >= 8000 and len(self._pending_audio_buffer) >= 4)  # REVERTED to original
        )
        
        if should_process:
            self._process_pending_audio()

    def is_buffer_drained(self, submitted_frames: set) -> bool:
        """
        Check if all available audio has been processed and no new frames can be generated.
        
        Args:
            submitted_frames: Set of frame indices already submitted for processing
            
        Returns:
            True if buffer is drained and no new frames can be generated
        """
        # First check if we have any pending audio
        if self._pending_audio_samples > 0:
            return False
        
        # Check if we can generate any new frames
        max_frame = self.get_max_encodable_frame()
        
        # If we can't encode any frames, buffer is drained
        if max_frame < 0:
            return True
        
        # Check if there are any unsubmitted frames
        for frame_idx in range(max_frame + 1):
            if frame_idx not in submitted_frames:
                return False  # Still have frames to process
        
        # All possible frames have been submitted
        return True
                    
    def _process_pending_audio(self):
        """Process pending audio samples into mel frames"""
        if not self._pending_audio_buffer or self.circuit_breaker:
            return
        
        # Concatenate all pending audio
        new_audio = np.concatenate(self._pending_audio_buffer)
        self._pending_audio_buffer = []
        self._pending_audio_samples = 0
        
        # Add to main audio buffer
        self.all_audio = np.concatenate([self.all_audio, new_audio])
        self.samples_processed = len(self.all_audio)
        
        # SMART TRUNCATION - keep recent audio but maintain alignment
        MAX_AUDIO_SECONDS = 5.0  # Increased for safety
        max_samples = int(MAX_AUDIO_SECONDS * self.sample_rate)
        
        if len(self.all_audio) > max_samples * 2:  # Only truncate when really needed
            # Calculate how much to keep based on current processing needs
            # We need to keep enough for the highest frame we're processing
            if self.audio_feature_buffer:
                highest_frame = max(self.audio_feature_buffer.keys())
                # Calculate samples needed for this frame
                samples_needed = int((highest_frame + 10) * self.sample_rate / 25.0)  # +10 frames buffer
                keep_samples = max(max_samples, samples_needed)
            else:
                keep_samples = max_samples
            
            # Truncate if beneficial
            if len(self.all_audio) > keep_samples * 1.5:
                samples_to_remove = len(self.all_audio) - keep_samples
                self.all_audio = self.all_audio[samples_to_remove:]
                # DON'T update samples_processed - keep it as total processed
                
                print(f"[Truncate] Removed {samples_to_remove/16000:.1f}s, kept {keep_samples/16000:.1f}s")
        
        # Compute mel incrementally
        self._compute_mel_incremental()
        
        # Encode audio features
        self._encode_all_pending_audio_features()

    def _compute_mel_incremental(self):
        """Compute mel spectrogram only for new audio data"""
        current_audio_length = len(self.all_audio)
        
        # Check if we have new audio to process
        if current_audio_length <= self._last_audio_length_for_mel:
            return
        
        with self.profiler.timer("Mel Spectrogram Generation (Incremental)"):
            # How much new audio do we have?
            new_audio_samples = current_audio_length - self._last_audio_length_for_mel
            
            # First time or reset - compute everything
            if self._last_audio_length_for_mel == 0:
                mel_full = melspectrogram(self.all_audio).T
                self.mel_frames = list(mel_full)
                self._last_audio_length_for_mel = current_audio_length
                return
            
            # For very large chunks (>2 seconds), recompute for safety
            if new_audio_samples > 32000:  # 2 seconds at 16kHz
                mel_full = melspectrogram(self.all_audio).T
                self.mel_frames = list(mel_full)
                self._last_audio_length_for_mel = current_audio_length
                return
            
            # TRUE INCREMENTAL COMPUTATION
            # We need overlap for the FFT window
            overlap_samples = self.n_fft - self.hop_length  # 600 samples (800-200)
            
            # Start from where we left off, minus overlap
            start_pos = max(0, self._last_audio_length_for_mel - overlap_samples)
            
            # Get the segment to process
            segment = self.all_audio[start_pos:]
            
            if len(segment) < self.n_fft:
                # Not enough samples for a full FFT window
                return
            
            # Compute mel for the segment
            segment_mel = melspectrogram(segment).T
            
            # Calculate how many mel frames we need to skip due to overlap
            # Each mel frame represents hop_length (200) samples
            if start_pos < self._last_audio_length_for_mel:
                # We have overlap
                overlap_in_samples = self._last_audio_length_for_mel - start_pos
                skip_frames = overlap_in_samples // self.hop_length
                
                # Append only the new frames
                if len(segment_mel) > skip_frames:
                    new_frames = segment_mel[skip_frames:]
                    self.mel_frames.extend(new_frames)
            else:
                # No overlap (shouldn't happen with our logic)
                self.mel_frames.extend(segment_mel)
            
            # Update our tracking
            self._last_audio_length_for_mel = current_audio_length
            
            # Safety check - if mel_frames gets too long relative to audio, recompute
            expected_mel_frames = (current_audio_length - self.n_fft + self.hop_length) // self.hop_length + 1
            if abs(len(self.mel_frames) - expected_mel_frames) > 10:  # Allow small discrepancy
                print(f"[Warning] Mel frame count mismatch. Expected: {expected_mel_frames}, Got: {len(self.mel_frames)}. Recomputing...")
                mel_full = melspectrogram(self.all_audio).T
                self.mel_frames = list(mel_full)

    def flush(self):
        """Process any remaining buffered audio"""
        if self._pending_audio_buffer:
            self._process_pending_audio()
        
        # Final mel computation if needed
        self._compute_mel_incremental()
        self._encode_all_pending_audio_features()

    def get_max_encodable_frame(self) -> int:
        """Calculate the maximum frame index we can encode based on available mel frames"""
        if len(self.mel_frames) < 16:
            return -1
        
        # Add a small buffer to avoid boundary issues
        # This ensures we always have enough mel frames for the audio window
        LOOKAHEAD_BUFFER = 2
        
        # Calculate theoretical max frame
        theoretical_max = int((len(self.mel_frames) - 16) / 80.0 * 25.0) + 2 - 1
        
        # Subtract buffer to ensure we have enough mel frames
        safe_max = max(-1, theoretical_max - LOOKAHEAD_BUFFER)
        
        return safe_max
    
    def _get_mel_window_for_frame(self, frame_idx: int) -> Optional[torch.Tensor]:
        """Get 16-frame mel window for a video frame with caching"""
        # Check cache first
        if frame_idx in self.mel_window_cache:
            return self.mel_window_cache[frame_idx]
        
        # Calculate mel indices - exact same as batch version
        start_idx = int(80.0 * (frame_idx / 25.0))
        end_idx = start_idx + 16
        
        if end_idx > len(self.mel_frames):
            return None
        
        # Extract window
        mel_window = np.array(self.mel_frames[start_idx:end_idx])
        
        if mel_window.shape[0] != 16:
            return None
        
        # Create tensor - shape [80, 16]
        mel_tensor = torch.FloatTensor(mel_window.T)
        
        # Cache it
        self.mel_window_cache[frame_idx] = mel_tensor
        
        # Limit cache size to prevent memory issues
        if len(self.mel_window_cache) > 1000:
            # Remove oldest entries
            min_key = min(self.mel_window_cache.keys())
            for i in range(min_key, min_key + 100):
                self.mel_window_cache.pop(i, None)
        
        return mel_tensor
    
    def _encode_all_pending_audio_features(self):
        """Encode audio features in optimal batches for maximum throughput"""
        if self.circuit_breaker:
            return
        
        # Find which frames need encoding
        max_frame = self.get_max_encodable_frame()
        frames_to_encode = []
        
        # Check ALL frames that haven't been encoded yet
        # This ensures we revisit any previously unprocessable frames
        for frame_idx in range(0, max_frame + 1):
            if frame_idx not in self.audio_feature_buffer:
                mel_window = self._get_mel_window_for_frame(frame_idx)
                if mel_window is not None:
                    frames_to_encode.append((frame_idx, mel_window))
        
        if not frames_to_encode:
            return
        
        # Process in optimal batch sizes for GPU efficiency
        OPTIMAL_BATCH_SIZE = 16
        total_frames = len(frames_to_encode)
        
        # Single profiler call for entire operation
        with self.profiler.timer(f"Audio Feature Encoding ({total_frames} frames total)", gpu_sync=True):
            for i in range(0, total_frames, OPTIMAL_BATCH_SIZE):
                batch = frames_to_encode[i:i + OPTIMAL_BATCH_SIZE]
                indices = [f[0] for f in batch]
                mel_tensors = [f[1] for f in batch]
                
                # Stack all tensors at once
                mel_batch = torch.stack(mel_tensors, dim=0).unsqueeze(1)
                
                # Single GPU operation for entire batch
                with torch.no_grad():
                    mel_batch_gpu = mel_batch.cuda().half()
                    audio_features = self.audio_encoder(mel_batch_gpu)
                    features_np = audio_features.cpu().numpy()
                
                # Store results
                for j, idx in enumerate(indices):
                    self.audio_feature_buffer[idx] = features_np[j]
    
    def build_audio_window(self, frame_idx: int) -> np.ndarray:
        """Build 16-frame audio feature window from cached features"""
        window_features = []
        
        for i in range(frame_idx - 8, frame_idx + 8):
            if i < 0:
                # Pad beginning - use first available frame
                if 0 in self.audio_feature_buffer:
                    feat = self.audio_feature_buffer[0].copy()
                else:
                    # Find first available
                    min_key = min(self.audio_feature_buffer.keys()) if self.audio_feature_buffer else 0
                    if min_key in self.audio_feature_buffer:
                        feat = self.audio_feature_buffer[min_key].copy()
                    else:
                        feat = np.zeros(512, dtype=np.float32)
            elif i in self.audio_feature_buffer:
                feat = self.audio_feature_buffer[i].copy()
            else:
                # Pad end or missing frame - use nearest available
                available_keys = [k for k in self.audio_feature_buffer.keys() if k < i]
                if available_keys:
                    last_available = max(available_keys)
                    feat = self.audio_feature_buffer[last_available].copy()
                else:
                    feat = np.zeros(512, dtype=np.float32)
            
            window_features.append(feat)
        
        result = np.stack(window_features)  # Shape: [16, 512]
        
        return result
    
    def get_ready_video_frames(self) -> List[int]:
        """Return list of NEW video frame indices that have complete audio windows"""
        ready_frames = []
    
        # First, encode any pending features in batches
        self._encode_all_pending_audio_features()
        
        # Check what's actually encodable based on mel frames
        max_encodable = self.get_max_encodable_frame()

        # Check all frames up to max_encodable
        for frame_idx in range(0, max_encodable + 1):
            # Skip if already returned
            if frame_idx in self.frames_returned_as_ready:
                continue
            
            # Check if we have the audio features
            if frame_idx not in self.audio_feature_buffer:
                continue
            
            # Frame is ready
            ready_frames.append(frame_idx)
            self.frames_returned_as_ready.add(frame_idx)

            self.highest_frame_processed = max(self.highest_frame_processed, frame_idx)
            
        return ready_frames
    
    def truncate_old_audio(self, keep_seconds: float = 5.0):
        """Truncate old audio data while maintaining frame alignment"""
        keep_samples = int(keep_seconds * self.sample_rate)
        
        if len(self.all_audio) <= keep_samples * 1.5:
            return
        
        samples_to_remove = len(self.all_audio) - keep_samples
        
        # Truncate audio
        self.all_audio = self.all_audio[-keep_samples:]
        
        # Reset mel computation tracking
        self._last_audio_length_for_mel = 0
        
        # Recompute mel for the truncated audio
        mel_full = melspectrogram(self.all_audio).T
        self.mel_frames = list(mel_full)
        self._last_audio_length_for_mel = len(self.all_audio)
        
        # Clear caches - they're all invalid now
        self.audio_feature_buffer.clear()
        self.mel_window_cache.clear()
        self.highest_frame_processed = -1
        
        print(f"[Truncate] Removed {samples_to_remove/16000:.1f}s, kept {keep_samples/16000:.1f}s")
    
    def get_video_frame_count(self) -> int:
        """Calculate how many video frames we can generate (matching batch version)"""
        if len(self.mel_frames) < 16:
            return 0
        
        # Use exact same formula as batch version (from AudDataset)
        return int((len(self.mel_frames) - 16) / 80. * float(25)) + 2
    
    def stop(self):
        """Circuit breaker for immediate stop"""
        self.circuit_breaker = True

    def diagnose_frame_mismatch(self):
        """Diagnose why frame counts don't match"""
        print("\nFrame Count Diagnosis:")
        print(f"  Total audio samples: {len(self.all_audio)}")
        print(f"  Total mel frames: {len(self.mel_frames)}")
        print(f"  Hop length: {self.hop_length}")
        print(f"  N_FFT: {self.n_fft}")
        
        # Check mel frame calculation
        expected_mel = (len(self.all_audio) - self.n_fft + self.hop_length) // self.hop_length + 1
        print(f"  Expected mel frames: {expected_mel}")
        print(f"  Actual mel frames: {len(self.mel_frames)}")
        
        # Check video frame calculation
        if len(self.mel_frames) >= 16:
            video_frames = int((len(self.mel_frames) - 16) / 80.0 * 25.0) + 2
            print(f"  Calculated video frames: {video_frames}")
            
            # Reverse calculate to see what's expected
            expected_mel_for_4642_frames = int((4642 - 2) * 80 / 25 + 16)
            print(f"  Mel frames needed for 4642 video frames: {expected_mel_for_4642_frames}")