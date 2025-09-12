"""
SyncTalk service implementation for Rapido system
"""

import asyncio
import websockets
import json
import numpy as np
from typing import AsyncGenerator, Dict, Any, Optional
import time
import aiohttp
from urllib.parse import urljoin

from ..core.interfaces import ISyncTalkClient, AudioChunk, VideoFrame
from ..core.exceptions import SyncTalkConnectionError
from ..core.logging_manager import get_logging_manager
from ..core.metrics import get_metrics_collector


class SyncTalkService(ISyncTalkClient):
    """Production SyncTalk service with robust error handling and monitoring"""
    
    def __init__(self, server_url: str, model_name: str = "enrique_torres", 
                 reconnect_attempts: int = 3, timeout: int = 30):
        self.server_url = server_url.rstrip('/')
        self.model_name = model_name
        self.reconnect_attempts = reconnect_attempts
        self.timeout = timeout
        
        self.websocket = None
        self.is_connected = False
        self.session_active = False
        
        self.logger = get_logging_manager().get_logger("synctalk_service")
        self.metrics = get_metrics_collector()
        
        # Connection metrics
        self._connection_stats = {
            "connection_attempts": 0,
            "successful_connections": 0,
            "disconnections": 0,
            "reconnections": 0,
            "audio_chunks_sent": 0,
            "frames_received": 0,
            "total_latency": 0.0,
            "error_count": 0
        }
    
    async def connect(self) -> bool:
        """Establish connection to SyncTalk server with retry logic"""
        self._connection_stats["connection_attempts"] += 1
        
        for attempt in range(self.reconnect_attempts):
            try:
                self.logger.info(
                    f"Attempting to connect to SyncTalk server (attempt {attempt + 1}/{self.reconnect_attempts})",
                    extra={"event_type": "synctalk_connection_attempt", "attempt": attempt + 1}
                )
                
                # Load model first
                if not await self._load_model():
                    continue
                
                # Connect to WebSocket
                ws_url = self.server_url.replace('http://', 'ws://').replace('https://', 'wss://')
                ws_url = urljoin(ws_url + '/', 'ws/audio_to_video')
                
                self.websocket = await asyncio.wait_for(
                    websockets.connect(ws_url), 
                    timeout=self.timeout
                )
                
                self.is_connected = True
                self.session_active = True
                self._connection_stats["successful_connections"] += 1
                
                self.logger.info(
                    "Successfully connected to SyncTalk server",
                    extra={
                        "event_type": "synctalk_connected",
                        "server_url": self.server_url,
                        "model_name": self.model_name
                    }
                )
                
                return True
                
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Connection timeout on attempt {attempt + 1}",
                    extra={"event_type": "synctalk_connection_timeout", "attempt": attempt + 1}
                )
            except Exception as e:
                self.logger.error(
                    f"Connection failed on attempt {attempt + 1}: {e}",
                    extra={
                        "event_type": "synctalk_connection_error",
                        "attempt": attempt + 1,
                        "error_details": {"error": str(e)}
                    }
                )
            
            if attempt < self.reconnect_attempts - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        self._connection_stats["error_count"] += 1
        self.is_connected = False
        raise SyncTalkConnectionError(f"Failed to connect after {self.reconnect_attempts} attempts")
    
    async def _load_model(self) -> bool:
        """Load the SyncTalk model via HTTP API"""
        try:
            load_model_url = urljoin(self.server_url + '/', 'load_model')
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model_name": self.model_name,
                    "config_type": "default"
                }
                
                async with session.post(
                    load_model_url, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(
                            f"Model loaded successfully: {result.get('message', 'Unknown response')}",
                            extra={"event_type": "synctalk_model_loaded", "model_name": self.model_name}
                        )
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(
                            f"Failed to load model: {response.status} - {error_text}",
                            extra={
                                "event_type": "synctalk_model_load_error",
                                "status_code": response.status,
                                "error_details": {"error": error_text}
                            }
                        )
                        return False
                        
        except Exception as e:
            self.logger.error(
                f"Model loading failed: {e}",
                extra={
                    "event_type": "synctalk_model_load_error",
                    "error_details": {"error": str(e)}
                }
            )
            return False
    
    async def send_audio(self, audio_chunk: AudioChunk) -> None:
        """Send audio chunk to SyncTalk server"""
        if not self.is_connected or not self.websocket:
            raise SyncTalkConnectionError("Not connected to SyncTalk server")
        
        try:
            start_time = time.time()
            
            # Convert audio data to format expected by SyncTalk
            if audio_chunk.data.dtype != np.int16:
                # Convert float32 [-1, 1] to int16
                audio_data_int16 = (audio_chunk.data * 32767).astype(np.int16)
            else:
                audio_data_int16 = audio_chunk.data
            
            # Create message
            message = {
                "type": "audio",
                "audio_data": audio_data_int16.tobytes().hex(),
                "sample_rate": audio_chunk.sample_rate,
                "timestamp": audio_chunk.timestamp,
                "chunk_id": audio_chunk.chunk_id
            }
            
            # Send via WebSocket
            await self.websocket.send(json.dumps(message))
            
            # Update metrics
            self._connection_stats["audio_chunks_sent"] += 1
            send_latency = time.time() - start_time
            self._connection_stats["total_latency"] += send_latency
            
            self.logger.info(
                f"Audio chunk sent to SyncTalk: {audio_chunk.chunk_id}",
                extra={
                    "audio_chunk_id": audio_chunk.chunk_id,
                    "event_type": "synctalk_audio_sent",
                    "performance_data": {
                        "send_latency": send_latency,
                        "chunk_duration": audio_chunk.duration,
                        "sample_rate": audio_chunk.sample_rate
                    }
                }
            )
            
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            self.session_active = False
            self._connection_stats["disconnections"] += 1
            raise SyncTalkConnectionError("Connection to SyncTalk server was closed")
        except Exception as e:
            self._connection_stats["error_count"] += 1
            self.logger.error(
                f"Failed to send audio chunk {audio_chunk.chunk_id}: {e}",
                extra={
                    "audio_chunk_id": audio_chunk.chunk_id,
                    "event_type": "synctalk_audio_send_error",
                    "error_details": {"error": str(e)}
                }
            )
            raise SyncTalkConnectionError(f"Failed to send audio: {e}")
    
    async def receive_frames(self) -> AsyncGenerator[VideoFrame, None]:
        """Receive video frames from SyncTalk server"""
        if not self.is_connected or not self.websocket:
            raise SyncTalkConnectionError("Not connected to SyncTalk server")
        
        frame_counter = 0
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "video_frame":
                        # Record frame reception
                        self.metrics.get_fps_counter("synctalk_output").record_frame()
                        
                        # Decode frame data
                        frame_data_bytes = bytes.fromhex(data["frame_data"])
                        frame_array = np.frombuffer(frame_data_bytes, dtype=np.uint8)
                        
                        # Decode image
                        import cv2
                        frame_image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        if frame_image is None:
                            raise ValueError("Failed to decode frame image")
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
                        
                        # Create VideoFrame
                        video_frame = VideoFrame(
                            data=frame_rgb,
                            timestamp=data.get("timestamp", time.time()),
                            frame_number=frame_counter,
                            width=frame_rgb.shape[1],
                            height=frame_rgb.shape[0],
                            fps=data.get("fps", 25.0)
                        )
                        
                        frame_counter += 1
                        self._connection_stats["frames_received"] += 1
                        
                        # Get current SyncTalk output FPS
                        synctalk_fps = self.metrics.get_fps_counter("synctalk_output").get_fps()
                        
                        self.logger.info(
                            f"Received frame from SyncTalk: {frame_counter}",
                            extra={
                                "event_type": "synctalk_frame_received",
                                "synctalk_fps": synctalk_fps,
                                "performance_data": {
                                    "frame_number": frame_counter,
                                    "frame_size": f"{video_frame.width}x{video_frame.height}",
                                    "fps": video_frame.fps
                                }
                            }
                        )
                        
                        yield video_frame
                        
                    elif data.get("type") == "error":
                        error_msg = data.get("message", "Unknown error from SyncTalk")
                        self.logger.error(
                            f"SyncTalk server error: {error_msg}",
                            extra={
                                "event_type": "synctalk_server_error",
                                "error_details": {"error": error_msg}
                            }
                        )
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(
                        f"Invalid JSON received from SyncTalk: {e}",
                        extra={
                            "event_type": "synctalk_invalid_json",
                            "error_details": {"error": str(e)}
                        }
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error processing frame from SyncTalk: {e}",
                        extra={
                            "event_type": "synctalk_frame_processing_error",
                            "error_details": {"error": str(e)}
                        }
                    )
                    
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            self.session_active = False
            self._connection_stats["disconnections"] += 1
            self.logger.warning(
                "SyncTalk connection closed during frame reception",
                extra={"event_type": "synctalk_connection_closed"}
            )
        except Exception as e:
            self._connection_stats["error_count"] += 1
            self.logger.error(
                f"Error receiving frames from SyncTalk: {e}",
                extra={
                    "event_type": "synctalk_frame_receive_error",
                    "error_details": {"error": str(e)}
                }
            )
            raise SyncTalkConnectionError(f"Failed to receive frames: {e}")
    
    async def disconnect(self) -> None:
        """Close connection to SyncTalk server"""
        if self.websocket:
            try:
                await self.websocket.close()
                self.logger.info(
                    "Disconnected from SyncTalk server",
                    extra={"event_type": "synctalk_disconnected"}
                )
            except Exception as e:
                self.logger.error(
                    f"Error during disconnect: {e}",
                    extra={
                        "event_type": "synctalk_disconnect_error",
                        "error_details": {"error": str(e)}
                    }
                )
            finally:
                self.websocket = None
                self.is_connected = False
                self.session_active = False
    
    async def get_connection_metrics(self) -> Dict[str, Any]:
        """Get connection performance metrics"""
        synctalk_fps = self.metrics.get_fps_counter("synctalk_output").get_stats()
        
        avg_latency = (
            self._connection_stats["total_latency"] / max(1, self._connection_stats["audio_chunks_sent"])
        )
        
        return {
            "connection_stats": self._connection_stats.copy(),
            "synctalk_fps": synctalk_fps["fps"],
            "synctalk_fps_stats": synctalk_fps,
            "average_latency": avg_latency,
            "is_connected": self.is_connected,
            "session_active": self.session_active,
            "timestamp": time.time()
        }
