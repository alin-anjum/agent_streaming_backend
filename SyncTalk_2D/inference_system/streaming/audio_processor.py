# inference_system/streaming/audio_processor.py

import numpy as np
import torch
from typing import List, Optional, Tuple
from collections import deque
import librosa
from utils import melspectrogram, AudioEncoder

class StreamingAudioProcessor:
    """
    Processes audio chunks for true streaming without file I/O
    """
    
    def __init__(self, audio_encoder: AudioEncoder, device: torch.device, mode: str = 'ave'):
        self.audio_encoder = audio_encoder
        self.device = device
        self.mode = mode
        
        # Audio processing state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.mel_frames = deque()  # All mel frames generated so far
        self.processed_features = []  # All audio encoder outputs
        self.last_processed_idx = 0  # Last mel frame index we processed
        
        # Constants
        self.sample_rate = 16000
        self.hop_length = 200  # Each mel frame = 200 samples = 12.5ms
        self.n_fft = 800
        self.mel_window_size = 16  # Audio encoder expects 16 mel frames
        
    def process_chunk(self, audio_chunk: bytes) -> List[Tuple[int, np.ndarray]]:
        """
        Process an audio chunk and return list of (frame_index, feature) tuples.
        This properly handles the 16-frame windowing requirement.
        """
        # Convert PCM bytes to float
        pcm_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, pcm_float])
        
        # Need at least n_fft samples to generate mel frames
        if len(self.audio_buffer) < self.n_fft:
            return []
        
        # Generate mel spectrogram for available audio
        # Keep some overlap for continuity
        processable_samples = len(self.audio_buffer) - self.n_fft + self.hop_length
        processable_samples = (processable_samples // self.hop_length) * self.hop_length
        
        if processable_samples > 0:
            # Process this segment
            audio_segment = self.audio_buffer[:processable_samples + self.n_fft - self.hop_length]
            mel_segment = melspectrogram(audio_segment).T  # Shape: [time_frames, 80]
            
            # Add new mel frames to our buffer
            for mel_frame in mel_segment:
                self.mel_frames.append(mel_frame)
            
            # Remove processed audio (keep overlap)
            self.audio_buffer = self.audio_buffer[processable_samples:]
        
        # Now process mel frames through audio encoder
        new_features = []
        
        # We need 16 mel frames for each audio encoder pass
        while len(self.mel_frames) >= self.mel_window_size + self.last_processed_idx:
            # Get 16 frames starting from last_processed_idx
            mel_window = []
            for i in range(self.last_processed_idx, self.last_processed_idx + self.mel_window_size):
                mel_window.append(self.mel_frames[i])
            
            # Convert to tensor and process
            mel_array = np.array(mel_window)  # [16, 80]
            mel_tensor = torch.FloatTensor(mel_array.T).unsqueeze(0).unsqueeze(0)  # [1, 1, 80, 16]
            mel_tensor = mel_tensor.to(self.device).half()
            
            with torch.no_grad():
                feature = self.audio_encoder(mel_tensor)  # [1, 512]
            
            feature_np = feature.float().cpu().numpy()[0]  # [512]
            
            # This feature corresponds to frame at the center of the window
            frame_idx = self.last_processed_idx + 8  # Center of 16-frame window
            new_features.append((frame_idx, feature_np))
            
            # Move window forward by 1 frame (sliding window)
            self.last_processed_idx += 1
        
        return new_features
    
    def get_features_for_frame_window(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get audio features for use with get_audio_features.
        Returns a numpy array suitable for the 16-frame windowing in get_audio_features.
        """
        # We need to provide enough context for get_audio_features
        # It expects to extract a 16-frame window centered at frame_idx
        if len(self.processed_features) < 16:
            return None
            
        # Return the most recent features
        return np.array(self.processed_features[-16:])