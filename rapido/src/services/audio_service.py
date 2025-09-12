"""
Audio processing services for Rapido system
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, AsyncGenerator
import time

from ..core.interfaces import IAudioProcessor, ITTSClient, AudioChunk
from ..core.exceptions import AudioProcessingError, TTSError
from ..core.logging_manager import get_logging_manager
from ..core.metrics import get_metrics_collector


class AudioProcessorService(IAudioProcessor):
    """Service for processing audio chunks with optimization"""
    
    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.logger = get_logging_manager().get_logger("audio_processor")
        self.metrics = get_metrics_collector()
        self._processing_stats = {
            "total_chunks": 0,
            "total_duration": 0.0,
            "error_count": 0
        }
    
    async def process_audio(self, audio_chunk: AudioChunk) -> AudioChunk:
        """Process an audio chunk with optimization and validation"""
        start_time = time.time()
        
        try:
            # Validate input
            if audio_chunk.data is None or len(audio_chunk.data) == 0:
                raise AudioProcessingError("Empty audio data provided")
            
            # Record metrics
            self.metrics.get_fps_counter("audio_processing").record_frame()
            
            # Process audio (normalization, noise reduction, etc.)
            processed_data = await self._optimize_audio(audio_chunk.data)
            
            # Create processed chunk
            processed_chunk = AudioChunk(
                data=processed_data,
                sample_rate=audio_chunk.sample_rate,
                timestamp=audio_chunk.timestamp,
                duration=audio_chunk.duration,
                chunk_id=audio_chunk.chunk_id
            )
            
            # Update stats
            self._processing_stats["total_chunks"] += 1
            self._processing_stats["total_duration"] += audio_chunk.duration
            
            # Log processing
            processing_time = time.time() - start_time
            self.logger.info(
                f"Processed audio chunk: {audio_chunk.chunk_id}",
                extra={
                    "audio_chunk_id": audio_chunk.chunk_id,
                    "event_type": "audio_processed",
                    "performance_data": {
                        "processing_time": processing_time,
                        "chunk_duration": audio_chunk.duration,
                        "sample_rate": audio_chunk.sample_rate
                    }
                }
            )
            
            return processed_chunk
            
        except Exception as e:
            self._processing_stats["error_count"] += 1
            self.logger.error(
                f"Audio processing failed for chunk {audio_chunk.chunk_id}",
                extra={
                    "audio_chunk_id": audio_chunk.chunk_id,
                    "event_type": "audio_processing_error",
                    "error_details": {"error": str(e)}
                }
            )
            raise AudioProcessingError(f"Failed to process audio chunk: {e}")
    
    async def _optimize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply audio optimization algorithms"""
        try:
            # Normalize audio levels
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            # Simple noise gate (remove very quiet segments)
            noise_threshold = 0.01
            audio_data = np.where(np.abs(audio_data) < noise_threshold, 0, audio_data)
            
            # Apply gentle compression
            audio_data = np.sign(audio_data) * np.sqrt(np.abs(audio_data))
            
            return audio_data
            
        except Exception as e:
            raise AudioProcessingError(f"Audio optimization failed: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get audio processing metrics"""
        fps_metrics = self.metrics.get_fps_counter("audio_processing").get_stats()
        
        return {
            "processing_stats": self._processing_stats.copy(),
            "fps_metrics": fps_metrics,
            "timestamp": time.time()
        }


class TTSService(ITTSClient):
    """Text-to-Speech service implementation"""
    
    def __init__(self, api_key: str, voice_id: str = "pNInz6obpgDQGcFmaJgB"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.logger = get_logging_manager().get_logger("tts_service")
        self.metrics = get_metrics_collector()
        self._synthesis_stats = {
            "total_requests": 0,
            "total_characters": 0,
            "total_audio_duration": 0.0,
            "error_count": 0
        }
    
    async def synthesize_speech(self, text: str) -> AudioChunk:
        """Convert text to speech using ElevenLabs API"""
        start_time = time.time()
        
        try:
            if not text or not text.strip():
                raise TTSError("Empty text provided for synthesis")
            
            # Import ElevenLabs client
            from elevenlabs import generate, Voice
            
            # Generate speech
            audio_data = generate(
                text=text,
                voice=Voice(voice_id=self.voice_id),
                api_key=self.api_key
            )
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate duration
            sample_rate = 22050  # ElevenLabs default
            duration = len(audio_array) / sample_rate
            
            # Create audio chunk
            chunk_id = f"tts_{int(time.time() * 1000)}"
            audio_chunk = AudioChunk(
                data=audio_array,
                sample_rate=sample_rate,
                timestamp=time.time(),
                duration=duration,
                chunk_id=chunk_id
            )
            
            # Update stats
            self._synthesis_stats["total_requests"] += 1
            self._synthesis_stats["total_characters"] += len(text)
            self._synthesis_stats["total_audio_duration"] += duration
            
            # Log synthesis
            synthesis_time = time.time() - start_time
            self.logger.info(
                f"TTS synthesis completed: {chunk_id}",
                extra={
                    "audio_chunk_id": chunk_id,
                    "event_type": "tts_synthesis",
                    "performance_data": {
                        "synthesis_time": synthesis_time,
                        "text_length": len(text),
                        "audio_duration": duration,
                        "characters_per_second": len(text) / synthesis_time
                    }
                }
            )
            
            return audio_chunk
            
        except Exception as e:
            self._synthesis_stats["error_count"] += 1
            self.logger.error(
                f"TTS synthesis failed for text length {len(text)}",
                extra={
                    "event_type": "tts_synthesis_error",
                    "error_details": {"error": str(e), "text_length": len(text)}
                }
            )
            raise TTSError(f"Speech synthesis failed: {e}")
    
    async def get_synthesis_metrics(self) -> Dict[str, Any]:
        """Get TTS synthesis metrics"""
        return {
            "synthesis_stats": self._synthesis_stats.copy(),
            "timestamp": time.time()
        }


class AudioOptimizerService:
    """Advanced audio optimization and synchronization service"""
    
    def __init__(self, buffer_size: int = 1024):
        self.buffer_size = buffer_size
        self.logger = get_logging_manager().get_logger("audio_optimizer")
        self.metrics = get_metrics_collector()
        self._audio_buffer = []
        self._optimization_stats = {
            "chunks_optimized": 0,
            "total_latency_reduction": 0.0,
            "buffer_underruns": 0,
            "buffer_overruns": 0
        }
    
    async def optimize_stream(self, audio_chunks: AsyncGenerator[AudioChunk, None]) -> AsyncGenerator[AudioChunk, None]:
        """Optimize streaming audio for consistent delivery"""
        try:
            async for chunk in audio_chunks:
                # Add to buffer
                self._audio_buffer.append(chunk)
                
                # Process buffer when it reaches optimal size
                if len(self._audio_buffer) >= self.buffer_size:
                    optimized_chunks = await self._process_buffer()
                    for optimized_chunk in optimized_chunks:
                        yield optimized_chunk
            
            # Process remaining buffer
            if self._audio_buffer:
                optimized_chunks = await self._process_buffer()
                for optimized_chunk in optimized_chunks:
                    yield optimized_chunk
                    
        except Exception as e:
            self.logger.error(
                f"Audio stream optimization failed: {e}",
                extra={
                    "event_type": "audio_optimization_error",
                    "error_details": {"error": str(e)}
                }
            )
            raise AudioProcessingError(f"Stream optimization failed: {e}")
    
    async def _process_buffer(self) -> list[AudioChunk]:
        """Process buffered audio chunks for optimization"""
        if not self._audio_buffer:
            return []
        
        start_time = time.time()
        
        try:
            # Sort by timestamp to ensure correct order
            self._audio_buffer.sort(key=lambda x: x.timestamp)
            
            # Apply jitter buffer smoothing
            optimized_chunks = []
            for chunk in self._audio_buffer:
                optimized_chunk = await self._apply_jitter_correction(chunk)
                optimized_chunks.append(optimized_chunk)
            
            # Update stats
            self._optimization_stats["chunks_optimized"] += len(optimized_chunks)
            processing_time = time.time() - start_time
            self._optimization_stats["total_latency_reduction"] += processing_time
            
            # Clear buffer
            self._audio_buffer.clear()
            
            return optimized_chunks
            
        except Exception as e:
            raise AudioProcessingError(f"Buffer processing failed: {e}")
    
    async def _apply_jitter_correction(self, chunk: AudioChunk) -> AudioChunk:
        """Apply jitter correction to audio chunk"""
        # Simple jitter correction - in production this would be more sophisticated
        return chunk
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get audio optimization metrics"""
        return {
            "optimization_stats": self._optimization_stats.copy(),
            "buffer_size": len(self._audio_buffer),
            "timestamp": time.time()
        }
