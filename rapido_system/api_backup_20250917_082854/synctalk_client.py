import asyncio
import json
import logging
import websockets
from typing import AsyncGenerator, Callable, Optional, Dict, Any
import base64
from PIL import Image
import io
import numpy as np

logger = logging.getLogger(__name__)

class SyncTalkWebSocketClient:
    """WebSocket client for bidirectional communication with SyncTalk server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        self.message_handlers = {}
        self.frame_callback = None
        self.audio_callback = None
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to SyncTalk server."""
        try:
            logger.info(f"Connecting to SyncTalk server: {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            logger.info("Successfully connected to SyncTalk server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SyncTalk server: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from SyncTalk server")
    
    async def send_message(self, message_type: str, data: Dict[Any, Any]):
        """Send a message to the SyncTalk server."""
        if not self.is_connected or not self.websocket:
            raise ConnectionError("Not connected to SyncTalk server")
        
        message = {
            "type": message_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent message: {message_type}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    async def send_audio_stream(self, audio_data: bytes, audio_format: str = "mp3"):
        """Send audio data to SyncTalk server for avatar generation."""
        # Encode audio data as base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        await self.send_message("audio_stream", {
            "audio_data": audio_b64,
            "format": audio_format,
            "sample_rate": 22050,  # Default sample rate
            "channels": 1
        })
    
    async def send_audio_chunk(self, chunk: bytes, chunk_index: int, is_final: bool = False):
        """Send streaming audio chunk to SyncTalk server."""
        chunk_b64 = base64.b64encode(chunk).decode('utf-8')
        
        await self.send_message("audio_chunk", {
            "chunk_data": chunk_b64,
            "chunk_index": chunk_index,
            "is_final": is_final,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def request_avatar_generation(self, config: Dict[str, Any]):
        """Request avatar generation with specific configuration."""
        await self.send_message("avatar_request", {
            "config": config,
            "request_id": f"req_{int(asyncio.get_event_loop().time())}"
        })
    
    async def listen_for_messages(self):
        """Listen for incoming messages from SyncTalk server."""
        if not self.is_connected or not self.websocket:
            raise ConnectionError("Not connected to SyncTalk server")
        
        try:
            async for message in self.websocket:
                await self._handle_incoming_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed by server")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")
            self.is_connected = False
    
    async def _handle_incoming_message(self, raw_message: str):
        """Handle incoming messages from SyncTalk server."""
        try:
            message = json.loads(raw_message)
            message_type = message.get("type")
            data = message.get("data", {})
            
            logger.debug(f"Received message: {message_type}")
            
            if message_type == "avatar_frame":
                await self._handle_avatar_frame(data)
            elif message_type == "avatar_status":
                await self._handle_avatar_status(data)
            elif message_type == "error":
                await self._handle_error(data)
            elif message_type in self.message_handlers:
                await self.message_handlers[message_type](data)
            else:
                logger.warning(f"Unhandled message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_avatar_frame(self, data: Dict[str, Any]):
        """Handle incoming avatar frame data."""
        try:
            frame_data = data.get("frame_data")
            frame_index = data.get("frame_index", 0)
            timestamp = data.get("timestamp", 0)
            
            if frame_data:
                # Decode base64 frame data
                frame_bytes = base64.b64decode(frame_data)
                
                # Convert to PIL Image
                frame_image = Image.open(io.BytesIO(frame_bytes))
                
                if self.frame_callback:
                    await self.frame_callback(frame_image, frame_index, timestamp)
                
                logger.debug(f"Processed avatar frame {frame_index}")
            
        except Exception as e:
            logger.error(f"Error processing avatar frame: {e}")
    
    async def _handle_avatar_status(self, data: Dict[str, Any]):
        """Handle avatar generation status updates."""
        status = data.get("status")
        message = data.get("message", "")
        
        logger.info(f"Avatar status: {status} - {message}")
        
        if status == "generation_complete":
            logger.info("Avatar generation completed")
        elif status == "generation_failed":
            logger.error(f"Avatar generation failed: {message}")
    
    async def _handle_error(self, data: Dict[str, Any]):
        """Handle error messages from server."""
        error_code = data.get("error_code")
        error_message = data.get("message", "Unknown error")
        
        logger.error(f"Server error [{error_code}]: {error_message}")
    
    def set_frame_callback(self, callback: Callable):
        """Set callback function for receiving avatar frames."""
        self.frame_callback = callback
    
    def set_message_handler(self, message_type: str, handler: Callable):
        """Set custom message handler for specific message types."""
        self.message_handlers[message_type] = handler
    
    async def stream_audio_and_receive_frames(self, audio_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[Image.Image, None]:
        """
        Stream audio to server and receive avatar frames.
        
        Args:
            audio_generator: Async generator yielding audio chunks
            
        Yields:
            PIL.Image: Avatar frames received from server
        """
        frame_queue = asyncio.Queue()
        
        async def frame_handler(frame: Image.Image, frame_index: int, timestamp: float):
            await frame_queue.put((frame, frame_index, timestamp))
        
        # Set frame callback
        original_callback = self.frame_callback
        self.set_frame_callback(frame_handler)
        
        try:
            # Start listening for messages in background
            listen_task = asyncio.create_task(self.listen_for_messages())
            
            # Stream audio chunks
            chunk_index = 0
            async for audio_chunk in audio_generator:
                await self.send_audio_chunk(audio_chunk, chunk_index)
                chunk_index += 1
                
                # Yield any received frames
                while not frame_queue.empty():
                    frame, idx, ts = await frame_queue.get()
                    yield frame
            
            # Send final chunk marker
            await self.send_audio_chunk(b"", chunk_index, is_final=True)
            
            # Continue yielding frames until generation is complete
            # (This would need additional logic based on SyncTalk server behavior)
            
        finally:
            # Restore original callback
            self.set_frame_callback(original_callback)
            listen_task.cancel()

class FrameBuffer:
    """Buffer for managing incoming avatar frames."""
    
    def __init__(self, max_frames: int = 100):
        self.frames = {}  # frame_index -> (frame, timestamp)
        self.max_frames = max_frames
        self.current_index = 0
        self._lock = asyncio.Lock()
    
    async def add_frame(self, frame: Image.Image, frame_index: int, timestamp: float):
        """Add frame to buffer."""
        async with self._lock:
            if len(self.frames) >= self.max_frames:
                # Remove oldest frame
                oldest_idx = min(self.frames.keys())
                del self.frames[oldest_idx]
            
            self.frames[frame_index] = (frame, timestamp)
    
    async def get_frame(self, frame_index: int) -> Optional[tuple]:
        """Get specific frame by index."""
        async with self._lock:
            return self.frames.get(frame_index)
    
    async def get_next_frame(self) -> Optional[tuple]:
        """Get next sequential frame."""
        async with self._lock:
            if self.current_index in self.frames:
                frame_data = self.frames[self.current_index]
                self.current_index += 1
                return frame_data
            return None
    
    async def get_all_frames(self) -> Dict[int, tuple]:
        """Get all buffered frames."""
        async with self._lock:
            return self.frames.copy()
    
    def get_frame_count(self) -> int:
        """Get number of buffered frames."""
        return len(self.frames)
