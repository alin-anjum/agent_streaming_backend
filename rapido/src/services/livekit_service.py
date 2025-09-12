"""
LiveKit service implementation for Rapido system
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional
import time
from dataclasses import dataclass

from ..core.interfaces import ILiveKitPublisher, VideoFrame
from ..core.exceptions import LiveKitConnectionError
from ..core.logging_manager import get_logging_manager
from ..core.metrics import get_metrics_collector

# LiveKit imports with error handling
try:
    import livekit.api as lk_api
    import livekit.rtc as rtc
    from livekit.rtc import VideoSource, VideoFrame as LKVideoFrame
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


@dataclass
class LiveKitConfig:
    """LiveKit configuration"""
    url: str
    api_key: str
    api_secret: str
    room_name: str = "avatar_room"
    participant_name: str = "avatar_bot"
    video_width: int = 854
    video_height: int = 480
    video_fps: float = 25.0
    max_bitrate: int = 1_500_000


class LiveKitService(ILiveKitPublisher):
    """Production LiveKit service with robust streaming capabilities"""
    
    def __init__(self, config: LiveKitConfig):
        if not LIVEKIT_AVAILABLE:
            raise LiveKitConnectionError("LiveKit SDK not available. Install with: pip install livekit")
        
        self.config = config
        self.room = None
        self.video_source = None
        self.video_track = None
        self.is_connected = False
        self.is_publishing = False
        
        self.logger = get_logging_manager().get_logger("livekit_service")
        self.metrics = get_metrics_collector()
        
        # Publishing metrics
        self._publishing_stats = {
            "connection_attempts": 0,
            "successful_connections": 0,
            "frames_published": 0,
            "publish_failures": 0,
            "total_publish_time": 0.0,
            "disconnections": 0,
            "reconnections": 0
        }
    
    async def connect(self, room_name: str = None) -> bool:
        """Connect to LiveKit room"""
        if room_name:
            self.config.room_name = room_name
            
        self._publishing_stats["connection_attempts"] += 1
        
        try:
            self.logger.info(
                f"Connecting to LiveKit room: {self.config.room_name}",
                extra={
                    "event_type": "livekit_connection_attempt",
                    "room_name": self.config.room_name,
                    "participant_name": self.config.participant_name
                }
            )
            
            # Create room instance
            self.room = rtc.Room(loop=asyncio.get_event_loop())
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Generate access token
            token = await self._generate_access_token()
            
            # Connect to room
            await self.room.connect(self.config.url, token)
            
            # Create video source and track
            self.video_source = rtc.VideoSource(
                self.config.video_width,
                self.config.video_height
            )
            
            self.video_track = rtc.LocalVideoTrack.create_video_track(
                "avatar_video",
                self.video_source
            )
            
            # Publish video track
            video_options = rtc.TrackPublishOptions()
            video_options.video_encoding = rtc.VideoEncoding(
                max_bitrate=self.config.max_bitrate,
                max_framerate=self.config.video_fps
            )
            
            await self.room.local_participant.publish_track(
                self.video_track,
                video_options
            )
            
            self.is_connected = True
            self.is_publishing = True
            self._publishing_stats["successful_connections"] += 1
            
            self.logger.info(
                f"Successfully connected to LiveKit room: {self.config.room_name}",
                extra={
                    "event_type": "livekit_connected",
                    "room_name": self.config.room_name,
                    "video_config": {
                        "width": self.config.video_width,
                        "height": self.config.video_height,
                        "fps": self.config.video_fps,
                        "bitrate": self.config.max_bitrate
                    }
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to connect to LiveKit: {e}",
                extra={
                    "event_type": "livekit_connection_error",
                    "room_name": self.config.room_name,
                    "error_details": {"error": str(e)}
                }
            )
            self.is_connected = False
            self.is_publishing = False
            raise LiveKitConnectionError(f"Failed to connect to LiveKit: {e}")
    
    def _setup_event_handlers(self):
        """Setup LiveKit event handlers"""
        @self.room.on("disconnected")
        def on_disconnected():
            self.is_connected = False
            self.is_publishing = False
            self._publishing_stats["disconnections"] += 1
            self.logger.warning(
                "Disconnected from LiveKit room",
                extra={"event_type": "livekit_disconnected"}
            )
        
        @self.room.on("reconnecting")
        def on_reconnecting():
            self.logger.info(
                "Reconnecting to LiveKit room",
                extra={"event_type": "livekit_reconnecting"}
            )
        
        @self.room.on("reconnected")
        def on_reconnected():
            self.is_connected = True
            self.is_publishing = True
            self._publishing_stats["reconnections"] += 1
            self.logger.info(
                "Reconnected to LiveKit room",
                extra={"event_type": "livekit_reconnected"}
            )
        
        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            self.logger.info(
                f"Participant connected: {participant.identity}",
                extra={
                    "event_type": "livekit_participant_joined",
                    "participant_id": participant.identity,
                    "event_source": "frontend"
                }
            )
        
        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            self.logger.info(
                f"Participant disconnected: {participant.identity}",
                extra={
                    "event_type": "livekit_participant_left",
                    "participant_id": participant.identity,
                    "event_source": "frontend"
                }
            )
    
    async def _generate_access_token(self) -> str:
        """Generate LiveKit access token"""
        try:
            token = lk_api.AccessToken(self.config.api_key, self.config.api_secret) \
                .with_identity(self.config.participant_name) \
                .with_name(self.config.participant_name) \
                .with_grants(lk_api.VideoGrants(
                    room_join=True,
                    room=self.config.room_name,
                    can_publish=True,
                    can_subscribe=True
                )).to_jwt()
            
            return token
            
        except Exception as e:
            raise LiveKitConnectionError(f"Failed to generate access token: {e}")
    
    async def publish_frame(self, frame: VideoFrame, lesson_id: str = None) -> bool:
        """Publish video frame to LiveKit room"""
        if not self.is_connected or not self.is_publishing or not self.video_source:
            raise LiveKitConnectionError("Not connected to LiveKit or not publishing")
        
        start_time = time.time()
        
        try:
            # Record frame publishing
            self.metrics.get_fps_counter("livekit_output").record_frame()
            
            # Convert VideoFrame to LiveKit format
            lk_frame = await self._convert_to_livekit_frame(frame)
            
            # Capture frame to video source
            await self.video_source.capture_frame(lk_frame)
            
            # Update stats
            self._publishing_stats["frames_published"] += 1
            publish_time = time.time() - start_time
            self._publishing_stats["total_publish_time"] += publish_time
            
            # Get current output FPS
            livekit_fps = self.metrics.get_fps_counter("livekit_output").get_fps()
            
            self.logger.info(
                f"Published frame to LiveKit: {frame.frame_number}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "livekit_frame_published",
                    "livekit_fps": livekit_fps,
                    "performance_data": {
                        "frame_number": frame.frame_number,
                        "publish_time": publish_time,
                        "frame_size": f"{frame.width}x{frame.height}",
                        "room_name": self.config.room_name
                    }
                }
            )
            
            return True
            
        except Exception as e:
            self._publishing_stats["publish_failures"] += 1
            self.logger.error(
                f"Failed to publish frame {frame.frame_number}: {e}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "livekit_publish_error",
                    "error_details": {
                        "error": str(e),
                        "frame_number": frame.frame_number
                    }
                }
            )
            return False
    
    async def _convert_to_livekit_frame(self, frame: VideoFrame) -> LKVideoFrame:
        """Convert VideoFrame to LiveKit VideoFrame format"""
        try:
            # Ensure frame dimensions match expected output
            if frame.width != self.config.video_width or frame.height != self.config.video_height:
                import cv2
                resized_data = cv2.resize(
                    frame.data, 
                    (self.config.video_width, self.config.video_height)
                )
            else:
                resized_data = frame.data
            
            # Ensure RGB format
            if len(resized_data.shape) == 3 and resized_data.shape[2] == 3:
                # Convert RGB to YUV420 for LiveKit
                import cv2
                yuv_data = cv2.cvtColor(resized_data, cv2.COLOR_RGB2YUV_I420)
                
                # Create LiveKit VideoFrame
                lk_frame = LKVideoFrame(
                    width=self.config.video_width,
                    height=self.config.video_height,
                    type=rtc.VideoBufferType.I420,
                    data=yuv_data.tobytes()
                )
                
                return lk_frame
            else:
                raise ValueError(f"Unsupported frame format: {resized_data.shape}")
                
        except Exception as e:
            raise LiveKitConnectionError(f"Failed to convert frame format: {e}")
    
    async def get_output_metrics(self) -> Dict[str, Any]:
        """Get LiveKit output metrics"""
        livekit_fps = self.metrics.get_fps_counter("livekit_output").get_stats()
        
        avg_publish_time = (
            self._publishing_stats["total_publish_time"] / 
            max(1, self._publishing_stats["frames_published"])
        )
        
        # Get room statistics if available
        room_stats = {}
        if self.room and self.is_connected:
            try:
                # Get participant count
                room_stats["participant_count"] = len(self.room.remote_participants)
                room_stats["room_name"] = self.config.room_name
            except Exception:
                pass
        
        return {
            "publishing_stats": self._publishing_stats.copy(),
            "livekit_fps": livekit_fps["fps"],
            "livekit_fps_stats": livekit_fps,
            "average_publish_time": avg_publish_time,
            "is_connected": self.is_connected,
            "is_publishing": self.is_publishing,
            "room_stats": room_stats,
            "timestamp": time.time()
        }
    
    async def disconnect(self) -> None:
        """Disconnect from LiveKit room"""
        if self.room:
            try:
                await self.room.disconnect()
                self.logger.info(
                    "Disconnected from LiveKit room",
                    extra={"event_type": "livekit_manual_disconnect"}
                )
            except Exception as e:
                self.logger.error(
                    f"Error during LiveKit disconnect: {e}",
                    extra={
                        "event_type": "livekit_disconnect_error",
                        "error_details": {"error": str(e)}
                    }
                )
            finally:
                self.room = None
                self.video_source = None
                self.video_track = None
                self.is_connected = False
                self.is_publishing = False
