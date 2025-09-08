import asyncio
import logging
import os
from typing import AsyncGenerator, Optional
import aiohttp
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import io

logger = logging.getLogger(__name__)

class ElevenLabsTTSClient:
    """Client for streaming TTS from ElevenLabs API."""
    
    def __init__(self, api_key: str, voice_id: str = "pNInz6obpgDQGcFmaJgB"):
        self.api_key = api_key
        self.voice_id = voice_id
        self.client = ElevenLabs(api_key=api_key)
        
        # TTS settings
        self.model = "eleven_multilingual_v2"
        self.output_format = "mp3_44100_128"
        
    async def stream_tts(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio data from ElevenLabs.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            bytes: Audio data chunks
        """
        try:
            logger.info(f"Starting TTS stream for text: {text[:50]}...")
            
            # Use the new ElevenLabs API in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def generate_streaming_audio():
                return self.client.text_to_speech.stream(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model
                )
            
            # Run the streaming TTS generation in a thread pool
            audio_stream = await loop.run_in_executor(None, generate_streaming_audio)
            
            # Stream the real-time audio chunks from ElevenLabs
            def process_stream():
                chunks = []
                for chunk in audio_stream:
                    if isinstance(chunk, bytes):
                        chunks.append(chunk)
                return chunks
            
            # Get chunks as they come in
            audio_chunks = await loop.run_in_executor(None, process_stream)
            
            # Yield the actual streaming chunks
            for chunk in audio_chunks:
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error in TTS streaming: {e}")
            raise
    
    async def stream_audio_real_time(self, text: str, chunk_callback):
        """
        Stream audio in real-time using ElevenLabs streaming API with 40ms chunks.
        Handles long text by splitting into manageable chunks.
        
        Args:
            text: Text to convert to speech
            chunk_callback: Async function to call with each audio chunk
        """
        try:
            logger.info(f"Starting OPTIMIZED real-time audio streaming for text: {len(text)} characters")
            logger.info(f"Text preview: {text[:100]}...")
            
            # Split long text into chunks to avoid ElevenLabs limits
            max_chunk_length = 1000  # 1000 characters per chunk
            text_chunks = []
            
            if len(text) > max_chunk_length:
                # Split at sentence boundaries
                sentences = text.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence + '. ') <= max_chunk_length:
                        current_chunk += sentence + '. '
                    else:
                        if current_chunk:
                            text_chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
                
                if current_chunk:
                    text_chunks.append(current_chunk.strip())
                    
                logger.info(f"Split text into {len(text_chunks)} chunks for better streaming")
            else:
                text_chunks = [text]
            
            loop = asyncio.get_event_loop()
            total_chunk_count = 0
            
            # Process each text chunk
            for chunk_idx, text_chunk in enumerate(text_chunks):
                logger.info(f"Processing text chunk {chunk_idx + 1}/{len(text_chunks)} ({len(text_chunk)} chars)")
                
                await self._stream_single_text_chunk(text_chunk, chunk_callback, loop)
                
            logger.info(f"All text chunks processed successfully")
            
        except Exception as e:
            logger.error(f"Failed to stream audio: {e}")
            raise
    
    async def _stream_single_text_chunk(self, text: str, chunk_callback, loop):
        """Stream a single text chunk"""
        def generate_stream():
            return self.client.text_to_speech.stream(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model,
                output_format="pcm_16000"  # Raw PCM at 16kHz
            )
        
        audio_stream = await loop.run_in_executor(None, generate_stream)
        
        # Buffer for accumulating larger chunks for better SyncTalk performance
        audio_buffer = b''
        # Try 160ms chunks (4 frames) = 640 * 4 = 2,560 samples = 5,120 bytes
        target_chunk_size = 5120  # 160ms of 16kHz audio (4 video frames)
        chunk_count = 0
        
        # Process chunks as they arrive and repackage into 160ms chunks
        for raw_chunk in audio_stream:
            if isinstance(raw_chunk, bytes) and len(raw_chunk) > 0:
                audio_buffer += raw_chunk
                
                # Send 160ms chunks when we have enough data
                while len(audio_buffer) >= target_chunk_size:
                    chunk_160ms = audio_buffer[:target_chunk_size]
                    audio_buffer = audio_buffer[target_chunk_size:]
                    
                    chunk_count += 1
                    logger.debug(f"160ms chunk {chunk_count}: {len(chunk_160ms)} bytes")
                    await chunk_callback(chunk_160ms)
        
        # Send remaining data as final chunk
        if audio_buffer:
            chunk_count += 1
            logger.debug(f"Final chunk {chunk_count}: {len(audio_buffer)} bytes")
            await chunk_callback(audio_buffer)
        
        logger.info(f"Text chunk completed - {chunk_count} 160ms chunks")

    async def generate_full_audio(self, text: str) -> bytes:
        """
        Generate complete audio for the given text.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bytes: Complete audio data
        """
        try:
            logger.info(f"Generating full audio for text: {text[:50]}...")
            
            loop = asyncio.get_event_loop()
            
            def generate_audio():
                return self.client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model,
                    output_format=self.output_format
                )
            
            audio_generator = await loop.run_in_executor(None, generate_audio)
            
            # Consume the generator to get all audio data
            def consume_generator():
                audio_chunks = []
                for chunk in audio_generator:
                    audio_chunks.append(chunk)
                return b''.join(audio_chunks)
            
            audio_data = await loop.run_in_executor(None, consume_generator)
            logger.info("Audio generation completed")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise
    
    async def get_available_voices(self) -> list:
        """Get list of available voices from ElevenLabs."""
        try:
            loop = asyncio.get_event_loop()
            voices = await loop.run_in_executor(None, self.client.voices.get_all)
            return [(voice.voice_id, voice.name) for voice in voices.voices]
        except Exception as e:
            logger.error(f"Error fetching voices: {e}")
            return []
    
    async def stream_tts_with_timing(self, text: str, timing_callback=None) -> AsyncGenerator[tuple, None]:
        """
        Stream TTS with timing information.
        
        Args:
            text: Text to convert to speech
            timing_callback: Optional callback for timing events
            
        Yields:
            tuple: (audio_chunk, timestamp_ms)
        """
        try:
            start_time = asyncio.get_event_loop().time()
            chunk_count = 0
            
            async for chunk in self.stream_tts(text):
                current_time = asyncio.get_event_loop().time()
                timestamp_ms = int((current_time - start_time) * 1000)
                
                if timing_callback:
                    await timing_callback(chunk_count, timestamp_ms, len(chunk))
                
                yield (chunk, timestamp_ms)
                chunk_count += 1
                
        except Exception as e:
            logger.error(f"Error in timed TTS streaming: {e}")
            raise

class TTSAudioBuffer:
    """Buffer for managing streaming TTS audio data."""
    
    def __init__(self, max_buffer_size: int = 1024 * 1024):  # 1MB default
        self.buffer = io.BytesIO()
        self.max_buffer_size = max_buffer_size
        self.total_bytes = 0
        self._lock = asyncio.Lock()
    
    async def write(self, data: bytes) -> None:
        """Write audio data to buffer."""
        async with self._lock:
            if self.total_bytes + len(data) > self.max_buffer_size:
                # Clear buffer if it gets too large
                self.buffer.seek(0)
                self.buffer.truncate(0)
                self.total_bytes = 0
                logger.warning("Audio buffer overflow, cleared buffer")
            
            self.buffer.write(data)
            self.total_bytes += len(data)
    
    async def read(self, size: int = -1) -> bytes:
        """Read audio data from buffer."""
        async with self._lock:
            self.buffer.seek(0)
            data = self.buffer.read(size)
            
            # Remove read data from buffer
            remaining_data = self.buffer.read()
            self.buffer.seek(0)
            self.buffer.truncate(0)
            if remaining_data:
                self.buffer.write(remaining_data)
                self.total_bytes = len(remaining_data)
            else:
                self.total_bytes = 0
            
            return data
    
    async def get_all_data(self) -> bytes:
        """Get all buffered audio data."""
        async with self._lock:
            self.buffer.seek(0)
            return self.buffer.read()
    
    def get_buffer_size(self) -> int:
        """Get current buffer size in bytes."""
        return self.total_bytes
