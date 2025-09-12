"""
Main orchestrator service for coordinating all Rapido components
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, AsyncGenerator
import time
from pathlib import Path
import cv2

from ..core.interfaces import AudioChunk, VideoFrame, SlideData
from ..core.exceptions import (
    RapidoException, AudioProcessingError, VideoProcessingError,
    SyncTalkConnectionError, LiveKitConnectionError, DataParsingError
)
from ..core.logging_manager import get_logging_manager
from ..core.metrics import get_metrics_collector
from ..core.security import SecurityManager

from .audio_service import AudioProcessorService, TTSService
from .video_service import VideoProcessorService, FrameComposerService
from .data_service import DataParsingService, SlideFrameManager
from .synctalk_service import SyncTalkService
from .livekit_service import LiveKitService, LiveKitConfig


class RapidoOrchestratorService:
    """
    Main orchestrator service that coordinates all Rapido components
    for complete presentation processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logging_manager().get_logger("orchestrator")
        self.metrics = get_metrics_collector()
        
        # Initialize security manager
        self.security_manager = SecurityManager()
        
        # Initialize services
        self._initialize_services()
        
        # Processing state
        self.current_lesson_id: Optional[str] = None
        self.is_processing = False
        self.processing_stats = {
            "lessons_processed": 0,
            "total_slides_processed": 0,
            "total_audio_chunks": 0,
            "total_frames_composed": 0,
            "total_processing_time": 0.0,
            "errors_encountered": 0
        }
    
    def _initialize_services(self):
        """Initialize all service components"""
        try:
            # Audio services
            self.audio_processor = AudioProcessorService(
                sample_rate=self.config.get("audio", {}).get("sample_rate", 22050),
                channels=self.config.get("audio", {}).get("channels", 1)
            )
            
            # TTS service (if API key provided)
            elevenlabs_key = self.config.get("tts", {}).get("api_key")
            if elevenlabs_key:
                self.tts_service = TTSService(
                    api_key=elevenlabs_key,
                    voice_id=self.config.get("tts", {}).get("voice_id", "pNInz6obpgDQGcFmaJgB")
                )
            else:
                self.tts_service = None
                self.logger.warning("TTS service not initialized - no API key provided")
            
            # Video services
            self.video_processor = VideoProcessorService(
                target_width=self.config.get("video", {}).get("width", 854),
                target_height=self.config.get("video", {}).get("height", 480)
            )
            
            self.frame_composer = FrameComposerService(
                overlay_position=self.config.get("video", {}).get("overlay_position", "bottom-right"),
                overlay_scale=self.config.get("video", {}).get("overlay_scale", 0.5)
            )
            
            # Data services
            self.data_parser = DataParsingService(self.security_manager)
            self.frame_manager = SlideFrameManager(
                self.config.get("paths", {}).get("slide_frames", "./presentation_frames")
            )
            
            # Communication services (initialized on demand)
            self.synctalk_service: Optional[SyncTalkService] = None
            self.livekit_service: Optional[LiveKitService] = None
            
            self.logger.info("All services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise RapidoException(f"Service initialization failed: {e}")
    
    async def initialize_communication_services(self, lesson_id: str) -> bool:
        """Initialize SyncTalk and LiveKit services for a lesson"""
        try:
            # Initialize SyncTalk service
            synctalk_config = self.config.get("synctalk", {})
            if synctalk_config.get("server_url"):
                self.synctalk_service = SyncTalkService(
                    server_url=synctalk_config["server_url"],
                    model_name=synctalk_config.get("model_name", "enrique_torres")
                )
                
                if not await self.synctalk_service.connect():
                    raise SyncTalkConnectionError("Failed to connect to SyncTalk service")
                
                self.logger.info("SyncTalk service connected successfully")
            
            # Initialize LiveKit service
            livekit_config = self.config.get("livekit", {})
            if all(key in livekit_config for key in ["url", "api_key", "api_secret"]):
                config = LiveKitConfig(
                    url=livekit_config["url"],
                    api_key=livekit_config["api_key"],
                    api_secret=livekit_config["api_secret"],
                    room_name=lesson_id,  # Use lesson ID as room name
                    participant_name="rapido_avatar_bot"
                )
                
                self.livekit_service = LiveKitService(config)
                
                if not await self.livekit_service.connect(lesson_id):
                    raise LiveKitConnectionError("Failed to connect to LiveKit service")
                
                self.logger.info(f"LiveKit service connected to room: {lesson_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize communication services: {e}")
            return False
    
    async def process_lesson(self, lesson_id: str, slide_data_path: str) -> Dict[str, Any]:
        """
        Process a complete lesson from slide data to live streaming
        
        Args:
            lesson_id: Unique identifier for the lesson
            slide_data_path: Path to the slide data JSON file
            
        Returns:
            Dict containing processing results and metrics
        """
        if self.is_processing:
            raise RapidoException("Another lesson is currently being processed")
        
        start_time = time.time()
        self.current_lesson_id = lesson_id
        self.is_processing = True
        
        try:
            # Validate inputs
            self.security_manager.validate_request(
                lesson_id=lesson_id,
                file_path=slide_data_path
            )
            
            # Log lesson start
            self.logger.log_lesson_start(lesson_id)
            
            # Parse slide data
            with self.metrics.time_operation("parse_slide_data"):
                slide_data = await self.data_parser.parse_slide_data(slide_data_path)
            
            # Get slide frames
            with self.metrics.time_operation("load_slide_frames"):
                frame_paths = await self.frame_manager.get_slide_frames(lesson_id)
                if not frame_paths:
                    raise DataParsingError(f"No slide frames found for lesson: {lesson_id}")
            
            # Initialize communication services
            await self.initialize_communication_services(lesson_id)
            
            # Process the lesson
            result = await self._process_lesson_content(lesson_id, slide_data, frame_paths)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["lessons_processed"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            
            self.logger.info(
                f"Lesson processing completed: {lesson_id}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "lesson_completed",
                    "performance_data": {
                        "total_processing_time": processing_time,
                        "slides_processed": len(frame_paths),
                        "audio_segments": len(slide_data.timing_segments)
                    }
                }
            )
            
            return {
                "success": True,
                "lesson_id": lesson_id,
                "processing_time": processing_time,
                "metrics": result.get("metrics", {}),
                "slides_processed": len(frame_paths),
                "audio_segments": len(slide_data.timing_segments)
            }
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            self.logger.error(
                f"Lesson processing failed: {lesson_id}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "lesson_processing_error",
                    "error_details": {"error": str(e)}
                }
            )
            
            return {
                "success": False,
                "lesson_id": lesson_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
        finally:
            # Cleanup
            await self._cleanup_services()
            self.current_lesson_id = None
            self.is_processing = False
    
    async def _process_lesson_content(self, lesson_id: str, slide_data: SlideData, 
                                    frame_paths: List[str]) -> Dict[str, Any]:
        """Process the actual lesson content"""
        
        # Create audio stream from narration
        audio_stream = await self._create_audio_stream(lesson_id, slide_data)
        
        # Create video frame stream from slides
        video_stream = await self._create_video_stream(lesson_id, frame_paths, slide_data.duration)
        
        # Process streams concurrently
        if self.synctalk_service:
            # With SyncTalk: audio -> avatar -> composition -> LiveKit
            return await self._process_with_synctalk(lesson_id, audio_stream, video_stream)
        else:
            # Without SyncTalk: direct composition -> LiveKit
            return await self._process_without_synctalk(lesson_id, video_stream)
    
    async def _create_audio_stream(self, lesson_id: str, slide_data: SlideData) -> AsyncGenerator[AudioChunk, None]:
        """Create audio stream from slide narration"""
        
        if self.tts_service:
            # Use TTS to generate audio from text
            audio_chunk = await self.tts_service.synthesize_speech(slide_data.narration_text)
            processed_chunk = await self.audio_processor.process_audio(audio_chunk)
            
            self.logger.log_audio_chunk(
                lesson_id=lesson_id,
                audio_chunk_id=processed_chunk.chunk_id,
                chunk_duration=processed_chunk.duration
            )
            
            yield processed_chunk
        else:
            # Create silent audio chunks for timing segments
            for i, segment in enumerate(slide_data.timing_segments):
                duration = segment["end"] - segment["start"]
                samples = int(duration * 22050)
                
                # Create silent audio
                audio_data = np.zeros(samples, dtype=np.float32)
                
                chunk = AudioChunk(
                    data=audio_data,
                    sample_rate=22050,
                    timestamp=segment["start"],
                    duration=duration,
                    chunk_id=f"{lesson_id}_segment_{i:03d}"
                )
                
                processed_chunk = await self.audio_processor.process_audio(chunk)
                
                self.logger.log_audio_chunk(
                    lesson_id=lesson_id,
                    audio_chunk_id=processed_chunk.chunk_id,
                    chunk_duration=processed_chunk.duration
                )
                
                yield processed_chunk
    
    async def _create_video_stream(self, lesson_id: str, frame_paths: List[str], 
                                 total_duration: float) -> AsyncGenerator[VideoFrame, None]:
        """Create video stream from slide frames"""
        
        if not frame_paths:
            return
        
        # Calculate timing for each frame
        frame_duration = total_duration / len(frame_paths)
        
        for i, frame_path in enumerate(frame_paths):
            try:
                # Load frame data
                frame_data_bytes = await self.frame_manager.load_frame(frame_path)
                if not frame_data_bytes:
                    continue
                
                # Decode image
                image_array = np.frombuffer(frame_data_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create VideoFrame
                video_frame = VideoFrame(
                    data=image_rgb,
                    timestamp=i * frame_duration,
                    frame_number=i,
                    width=image_rgb.shape[1],
                    height=image_rgb.shape[0],
                    fps=25.0
                )
                
                # Process frame
                processed_frame = await self.video_processor.process_frame(video_frame)
                
                # Record slide frame metrics
                self.metrics.get_fps_counter("slide_frames").record_frame()
                
                yield processed_frame
                
            except Exception as e:
                self.logger.error(
                    f"Failed to process slide frame {i}: {e}",
                    extra={
                        "lesson_id": lesson_id,
                        "event_type": "slide_frame_error",
                        "error_details": {"error": str(e), "frame_index": i}
                    }
                )
                continue
    
    async def _process_with_synctalk(self, lesson_id: str, 
                                   audio_stream: AsyncGenerator[AudioChunk, None],
                                   video_stream: AsyncGenerator[VideoFrame, None]) -> Dict[str, Any]:
        """Process lesson with SyncTalk avatar generation"""
        
        frames_composed = 0
        
        try:
            # Start audio processing task
            audio_task = asyncio.create_task(self._send_audio_to_synctalk(lesson_id, audio_stream))
            
            # Process avatar frames from SyncTalk and compose with slides
            async for slide_frame in video_stream:
                # Get avatar frame from SyncTalk
                async for avatar_frame in self.synctalk_service.receive_frames():
                    try:
                        # Compose slide with avatar
                        composed_frame = await self.frame_composer.compose_frame(
                            slide_frame,
                            avatar_frame,
                            lesson_id
                        )
                        
                        # Send to LiveKit if available
                        if self.livekit_service:
                            await self.livekit_service.publish_frame(composed_frame, lesson_id)
                        
                        frames_composed += 1
                        self.processing_stats["total_frames_composed"] += 1
                        
                        # Log FPS metrics periodically
                        if frames_composed % 25 == 0:  # Every second at 25fps
                            await self._log_fps_metrics(lesson_id)
                        
                        break  # Move to next slide frame
                        
                    except Exception as e:
                        self.logger.error(
                            f"Frame composition error: {e}",
                            extra={
                                "lesson_id": lesson_id,
                                "event_type": "frame_composition_error",
                                "error_details": {"error": str(e)}
                            }
                        )
            
            # Wait for audio processing to complete
            await audio_task
            
            return {
                "frames_composed": frames_composed,
                "metrics": await self._collect_final_metrics()
            }
            
        except Exception as e:
            self.logger.error(
                f"SyncTalk processing failed: {e}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "synctalk_processing_error",
                    "error_details": {"error": str(e)}
                }
            )
            raise
    
    async def _process_without_synctalk(self, lesson_id: str,
                                      video_stream: AsyncGenerator[VideoFrame, None]) -> Dict[str, Any]:
        """Process lesson without SyncTalk (slides only)"""
        
        frames_processed = 0
        
        try:
            async for slide_frame in video_stream:
                # Send slide directly to LiveKit if available
                if self.livekit_service:
                    await self.livekit_service.publish_frame(slide_frame, lesson_id)
                
                frames_processed += 1
                
                # Log FPS metrics periodically
                if frames_processed % 25 == 0:
                    await self._log_fps_metrics(lesson_id)
            
            return {
                "frames_processed": frames_processed,
                "metrics": await self._collect_final_metrics()
            }
            
        except Exception as e:
            self.logger.error(
                f"Direct processing failed: {e}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "direct_processing_error",
                    "error_details": {"error": str(e)}
                }
            )
            raise
    
    async def _send_audio_to_synctalk(self, lesson_id: str, 
                                    audio_stream: AsyncGenerator[AudioChunk, None]):
        """Send audio chunks to SyncTalk service"""
        if not self.synctalk_service:
            return
        
        try:
            async for audio_chunk in audio_stream:
                await self.synctalk_service.send_audio(audio_chunk)
                self.processing_stats["total_audio_chunks"] += 1
                
        except Exception as e:
            self.logger.error(
                f"Audio streaming to SyncTalk failed: {e}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "synctalk_audio_error",
                    "error_details": {"error": str(e)}
                }
            )
            raise
    
    async def _log_fps_metrics(self, lesson_id: str):
        """Log current FPS metrics"""
        fps_metrics = self.metrics.get_fps_metrics()
        
        # Get service-specific metrics
        synctalk_fps = 0.0
        livekit_fps = 0.0
        
        if self.synctalk_service:
            synctalk_metrics = await self.synctalk_service.get_connection_metrics()
            synctalk_fps = synctalk_metrics.get("synctalk_fps", 0.0)
        
        if self.livekit_service:
            livekit_metrics = await self.livekit_service.get_output_metrics()
            livekit_fps = livekit_metrics.get("livekit_fps", 0.0)
        
        self.logger.log_fps_metrics(
            lesson_id=lesson_id,
            slide_fps=fps_metrics.get("slide_frames", 0.0),
            synctalk_fps=synctalk_fps,
            composer_fps=fps_metrics.get("composer", 0.0),
            livekit_fps=livekit_fps
        )
    
    async def _collect_final_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive final metrics"""
        metrics = self.metrics.get_all_metrics()
        
        # Add service-specific metrics
        service_metrics = {}
        
        if hasattr(self, 'audio_processor'):
            service_metrics["audio"] = await self.audio_processor.get_metrics()
        
        if hasattr(self, 'video_processor'):
            service_metrics["video"] = await self.video_processor.get_processing_stats()
        
        if hasattr(self, 'frame_composer'):
            service_metrics["composer"] = await self.frame_composer.get_composition_metrics()
        
        if self.synctalk_service:
            service_metrics["synctalk"] = await self.synctalk_service.get_connection_metrics()
        
        if self.livekit_service:
            service_metrics["livekit"] = await self.livekit_service.get_output_metrics()
        
        return {
            "system_metrics": metrics,
            "service_metrics": service_metrics,
            "processing_stats": self.processing_stats.copy()
        }
    
    async def _cleanup_services(self):
        """Cleanup communication services"""
        try:
            if self.synctalk_service:
                await self.synctalk_service.disconnect()
                self.synctalk_service = None
            
            if self.livekit_service:
                await self.livekit_service.disconnect()
                self.livekit_service = None
                
        except Exception as e:
            self.logger.error(f"Service cleanup error: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        return {
            "is_processing": self.is_processing,
            "current_lesson_id": self.current_lesson_id,
            "processing_stats": self.processing_stats.copy(),
            "services_initialized": {
                "audio_processor": hasattr(self, 'audio_processor'),
                "video_processor": hasattr(self, 'video_processor'),
                "frame_composer": hasattr(self, 'frame_composer'),
                "data_parser": hasattr(self, 'data_parser'),
                "frame_manager": hasattr(self, 'frame_manager'),
                "tts_service": self.tts_service is not None,
                "synctalk_service": self.synctalk_service is not None,
                "livekit_service": self.livekit_service is not None
            },
            "timestamp": time.time()
        }
    
    async def stop_processing(self) -> bool:
        """Stop current processing"""
        if not self.is_processing:
            return True
        
        try:
            await self._cleanup_services()
            self.is_processing = False
            self.current_lesson_id = None
            
            self.logger.info(
                "Processing stopped by request",
                extra={"event_type": "processing_stopped"}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop processing: {e}")
            return False
