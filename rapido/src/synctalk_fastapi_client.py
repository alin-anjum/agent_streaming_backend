import asyncio
import json
import logging
import websockets
import pickle
import numpy as np
from typing import AsyncGenerator, Callable, Optional, Dict, Any
import base64
from PIL import Image
import io
import wave
import struct

logger = logging.getLogger(__name__)

class SyncTalkFastAPIClient:
    """WebSocket client specifically designed for SyncTalk-FastAPI server."""
    
    def __init__(self, server_url: str, model_name: str = "enrique_torres"):
        self.server_url = server_url
        self.model_name = model_name
        self.websocket = None
        self.is_connected = False
        self.frame_callback = None
        self.session_active = False
        
        # Audio configuration for SyncTalk
        self.sample_rate = 24000  # SyncTalk expects 24kHz
        self.channels = 1  # Mono
        self.sample_width = 2  # 16-bit
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to SyncTalk server."""
        try:
            logger.info(f"Connecting to SyncTalk FastAPI server: {self.server_url}")
            
            # First, load the model via HTTP API
            await self._load_model()
            
            # Then connect to WebSocket
            ws_url = self.server_url.replace('http://', 'ws://').replace('https://', 'wss://')
            if not ws_url.endswith('/ws/audio_to_video'):
                ws_url = ws_url.rstrip('/') + '/ws/audio_to_video'
            
            logger.info(f"Connecting to WebSocket: {ws_url}")
            self.websocket = await websockets.connect(ws_url)
            self.is_connected = True
            self.session_active = True
            logger.info("Successfully connected to SyncTalk FastAPI server")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SyncTalk FastAPI server: {e}")
            self.is_connected = False
            return False
    
    async def _load_model(self):
        """Load the model via HTTP API before starting WebSocket session."""
        import aiohttp
        
        http_url = self.server_url.rstrip('/')
        load_model_url = f"{http_url}/load_model"
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model_name": self.model_name,
                    "config_type": "default"
                }
                
                async with session.post(load_model_url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Model loaded: {result['message']}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to load model: {response.status} - {error_text}")
                        raise Exception(f"Model loading failed: {error_text}")
                        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.session_active = False
            logger.info("Disconnected from SyncTalk FastAPI server")
    
    def set_frame_callback(self, callback: Callable):
        """Set callback function for receiving video frames."""
        self.frame_callback = callback
    
    async def stream_audio_chunks(self, audio_generator: AsyncGenerator[bytes, None]):
        """
        Stream audio chunks to SyncTalk server and receive video frames.
        
        Args:
            audio_generator: Async generator yielding audio chunks as bytes
        """
        if not self.is_connected or not self.websocket:
            raise ConnectionError("Not connected to SyncTalk server")
        
        try:
            # Start listening for video frames in background
            listen_task = asyncio.create_task(self._listen_for_video_frames())
            
            # Stream audio chunks
            chunk_count = 0
            async for audio_chunk in audio_generator:
                # Convert audio chunk to PCM format expected by SyncTalk
                pcm_data = self._convert_to_pcm(audio_chunk)
                
                # Send PCM data to SyncTalk
                await self.websocket.send(pcm_data)
                chunk_count += 1
                
                logger.debug(f"Sent audio chunk {chunk_count}, size: {len(pcm_data)} bytes")
                
                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.01)
            
            # Keep listening for remaining frames for a bit
            await asyncio.sleep(2.0)
            
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            raise
        finally:
            listen_task.cancel()
            try:
                await listen_task
            except asyncio.CancelledError:
                pass
    
    def _convert_to_pcm(self, audio_data: bytes) -> bytes:
        """
        Convert audio data to PCM format expected by SyncTalk.
        SyncTalk expects: 24kHz, 16-bit, mono PCM data
        """
        # For now, assume the input is already in the correct format
        # In a real implementation, you might need to resample/convert
        
        # Ensure we have at least 400ms of audio (19200 bytes for 24kHz 16-bit mono)
        min_size = 19200
        if len(audio_data) < min_size:
            # Pad with zeros if too short
            padding = b'\x00' * (min_size - len(audio_data))
            audio_data = audio_data + padding
        
        return audio_data
    
    async def _listen_for_video_frames(self):
        """Listen for incoming video frames from SyncTalk server."""
        try:
            while self.session_active and self.websocket:
                try:
                    # Receive pickled data from server
                    pickled_data = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=1.0
                    )
                    
                    # Unpickle the data
                    data = pickle.loads(pickled_data)
                    
                    # Extract video and audio data
                    video_bytes = data.get('video_bytes')
                    audio_bytes = data.get('audio_bytes', b'')
                    
                    if video_bytes:
                        # Convert bytes back to numpy array and then to PIL Image
                        video_array = np.frombuffer(video_bytes, dtype=np.uint8)
                        
                        # Reshape assuming standard video frame dimensions
                        # SyncTalk typically outputs 512x512 RGB frames
                        height, width = 512, 512
                        if len(video_array) == height * width * 3:
                            video_array = video_array.reshape((height, width, 3))
                            frame_image = Image.fromarray(video_array, 'RGB')
                            
                            if self.frame_callback:
                                # Call the callback with frame and timestamp
                                timestamp = asyncio.get_event_loop().time() * 1000
                                await self.frame_callback(frame_image, 0, timestamp)
                            
                            logger.debug("Processed video frame from SyncTalk")
                        else:
                            logger.warning(f"Unexpected video data size: {len(video_array)}")
                    
                except asyncio.TimeoutError:
                    # Timeout is normal, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing video frame: {e}")
                    break
                    
        except asyncio.CancelledError:
            logger.info("Video frame listening cancelled")
        except Exception as e:
            logger.error(f"Error in video frame listener: {e}")

class AudioConverter:
    """Utility class for audio format conversion."""
    
    @staticmethod
    def mp3_to_pcm(mp3_data: bytes, target_sample_rate: int = 24000) -> bytes:
        """Convert MP3 data to PCM format."""
        try:
            import librosa
            import soundfile as sf
            import io
            
            # Load MP3 data
            audio, sr = librosa.load(io.BytesIO(mp3_data), sr=None)
            
            # Resample to target sample rate
            if sr != target_sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
            
            # Convert to 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            logger.error(f"Error converting MP3 to PCM: {e}")
            raise
    
    @staticmethod
    def chunk_audio(audio_data: bytes, chunk_duration_ms: int = 400, sample_rate: int = 24000) -> list:
        """
        Split audio data into chunks.
        
        Args:
            audio_data: PCM audio data as bytes
            chunk_duration_ms: Duration of each chunk in milliseconds
            sample_rate: Sample rate of the audio
            
        Returns:
            List of audio chunks as bytes
        """
        # Calculate samples per chunk
        samples_per_chunk = int((chunk_duration_ms / 1000) * sample_rate)
        bytes_per_sample = 2  # 16-bit = 2 bytes
        bytes_per_chunk = samples_per_chunk * bytes_per_sample
        
        chunks = []
        for i in range(0, len(audio_data), bytes_per_chunk):
            chunk = audio_data[i:i + bytes_per_chunk]
            
            # Pad the last chunk if it's too short
            if len(chunk) < bytes_per_chunk:
                padding = b'\x00' * (bytes_per_chunk - len(chunk))
                chunk = chunk + padding
            
            chunks.append(chunk)
        
        return chunks

# Async generator wrapper for easier integration
async def audio_chunks_from_tts(tts_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    """
    Convert TTS audio stream to PCM chunks suitable for SyncTalk.
    
    Args:
        tts_generator: Async generator yielding TTS audio data
        
    Yields:
        PCM audio chunks as bytes
    """
    audio_buffer = b''
    
    async for audio_chunk in tts_generator:
        # Convert MP3 chunk to PCM if needed
        try:
            pcm_chunk = AudioConverter.mp3_to_pcm(audio_chunk)
            audio_buffer += pcm_chunk
            
            # Split into 400ms chunks
            chunks = AudioConverter.chunk_audio(audio_buffer, chunk_duration_ms=400)
            
            # Yield all complete chunks
            for i, chunk in enumerate(chunks[:-1]):  # All except the last
                yield chunk
            
            # Keep the last (potentially incomplete) chunk in buffer
            if chunks:
                audio_buffer = chunks[-1] if len(chunks[-1]) < 19200 else b''
            
        except Exception as e:
            logger.error(f"Error processing TTS chunk: {e}")
            continue
    
    # Yield any remaining data
    if audio_buffer:
        yield audio_buffer
