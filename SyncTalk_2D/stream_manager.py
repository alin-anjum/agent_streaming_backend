# stream_manager.py
"""
Multi-stream management system for SyncTalk FastAPI

This module provides support for multiple concurrent inference streams,
allowing multiple clients to connect simultaneously with different avatars
and configurations.
"""

import uuid
import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import WebSocket
from fastapi.concurrency import run_in_threadpool

from audio_stream_generator import AudioStreamGenerator
from chroma_key import create_chroma_key_processor

logger = logging.getLogger(__name__)

@dataclass
class StreamState:
    """State for a single inference stream"""
    stream_id: str
    websocket: WebSocket
    avatar_name: str
    avatar_config: Dict[str, Any]
    model_name: str  # folder_name for inference API
    background_url: Optional[str]
    chroma_key_processor: Optional[Any]
    audio_generator: Optional[AudioStreamGenerator]
    generator_task: Optional[asyncio.Task]
    start_time: datetime
    last_activity_time: datetime
    sample_rate: int
    client_info: Dict[str, Any]
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity_time = datetime.now()


class StreamManager:
    """
    Manages multiple concurrent inference streams
    
    This class provides:
    - Concurrent stream management with unique stream IDs
    - Per-stream state isolation (avatar configs, chroma key, etc.)
    - Resource cleanup and lifecycle management
    - Stream monitoring and statistics
    """
    
    def __init__(self, inference_api, max_concurrent_streams: int = 5):
        self.inference_api = inference_api
        self.max_concurrent_streams = max_concurrent_streams
        
        # Active streams: stream_id -> StreamState
        self.active_streams: Dict[str, StreamState] = {}
        self.streams_lock = threading.RLock()
        
        # Stream statistics
        self.total_streams_created = 0
        self.total_streams_completed = 0
        
        logger.info(f"StreamManager initialized with max {max_concurrent_streams} concurrent streams")
    
    def get_active_stream_count(self) -> int:
        """Get number of currently active streams"""
        with self.streams_lock:
            return len(self.active_streams)
    
    def get_stream_info(self) -> Dict[str, Any]:
        """Get information about all active streams"""
        with self.streams_lock:
            stream_info = {}
            for stream_id, stream_state in self.active_streams.items():
                stream_info[stream_id] = {
                    'avatar_name': stream_state.avatar_name,
                    'model_name': stream_state.model_name,
                    'start_time': stream_state.start_time.isoformat(),
                    'last_activity': stream_state.last_activity_time.isoformat(),
                    'duration_seconds': (datetime.now() - stream_state.start_time).total_seconds(),
                    'sample_rate': stream_state.sample_rate,
                    'background_url': stream_state.background_url,
                    'client_info': stream_state.client_info
                }
            return {
                'active_streams': stream_info,
                'active_count': len(self.active_streams),
                'max_concurrent': self.max_concurrent_streams,
                'total_created': self.total_streams_created,
                'total_completed': self.total_streams_completed
            }
    
    def can_accept_new_stream(self) -> bool:
        """Check if we can accept a new stream"""
        with self.streams_lock:
            return len(self.active_streams) < self.max_concurrent_streams
    
    def get_active_models(self) -> Set[str]:
        """Get set of currently active model names"""
        with self.streams_lock:
            return {stream.model_name for stream in self.active_streams.values()}
    
    async def create_stream(self, websocket: WebSocket, avatar_name: str, 
                           avatar_config: Dict[str, Any], sample_rate: int,
                           background_url: Optional[str] = None,
                           client_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new inference stream
        
        Args:
            websocket: WebSocket connection
            avatar_name: Name of the avatar to use
            avatar_config: Avatar configuration dictionary
            sample_rate: Audio sample rate
            background_url: Optional background URL for chroma key
            client_info: Optional client information
            
        Returns:
            stream_id: Unique identifier for the created stream
            
        Raises:
            ValueError: If maximum concurrent streams exceeded or model not available
        """
        if not self.can_accept_new_stream():
            raise ValueError(f"Maximum concurrent streams ({self.max_concurrent_streams}) exceeded")
        
        # Generate unique stream ID
        stream_id = str(uuid.uuid4())
        
        # Get model name (folder_name) for inference API
        model_name = avatar_config.get("folder_name", avatar_name)
        
        # Verify model is loaded
        if model_name not in self.inference_api.models:
            raise ValueError(f"Model '{model_name}' not loaded in inference API")
        
        # Create stream state
        stream_state = StreamState(
            stream_id=stream_id,
            websocket=websocket,
            avatar_name=avatar_name,
            avatar_config=avatar_config.copy(),
            model_name=model_name,
            background_url=background_url,
            chroma_key_processor=None,
            audio_generator=None,
            generator_task=None,
            start_time=datetime.now(),
            last_activity_time=datetime.now(),
            sample_rate=sample_rate,
            client_info=client_info or {}
        )
        
        # Initialize chroma key processor if needed
        try:
            await self._init_chroma_key_for_stream(stream_state)
        except Exception as e:
            logger.error(f"Failed to initialize chroma key for stream {stream_id}: {e}")
            # Continue without chroma key
        
        # Register stream
        with self.streams_lock:
            self.active_streams[stream_id] = stream_state
            self.total_streams_created += 1
        
        logger.info(f"Created stream {stream_id} for avatar '{avatar_name}' (model: {model_name}, "
                   f"sample_rate: {sample_rate}Hz, active_streams: {len(self.active_streams)})")
        
        return stream_id
    
    async def _init_chroma_key_for_stream(self, stream_state: StreamState):
        """Initialize chroma key processor for a specific stream"""
        avatar_config = stream_state.avatar_config
        background_url = stream_state.background_url
        
        # Check if chroma key is enabled
        ck_cfg = avatar_config.get("chroma_key", {})
        if not ck_cfg.get("enabled", False):
            return
        
        # Determine dimensions
        resize_dims = avatar_config.get("resize_dims")
        if resize_dims and len(resize_dims) == 2:
            width, height = resize_dims
            logger.info(f"Stream {stream_state.stream_id}: Using resize_dims for chroma key: {width}x{height}")
        else:
            # Fallback to crop_bbox approach
            crop_bbox = avatar_config.get("crop_bbox")
            if crop_bbox and len(crop_bbox) == 4:
                x1, y1, x2, y2 = crop_bbox
                width, height = max(0, x2 - x1), max(0, y2 - y1)
                logger.info(f"Stream {stream_state.stream_id}: Using crop_bbox for chroma key: {width}x{height}")
            else:
                # Default to 1080p
                width, height = 1920, 1080
                logger.info(f"Stream {stream_state.stream_id}: Using default 1080p for chroma key")
        
        # Use provided background_url or fall back to config default
        effective_background_url = background_url
        if not effective_background_url:
            effective_background_url = ck_cfg.get("default_background_url")
        
        if not effective_background_url:
            logger.info(f"Stream {stream_state.stream_id}: Chroma key enabled but no background URL available")
            return
        
        # Get chroma key settings
        target_color = ck_cfg.get("target_color", "#00FF00")
        color_threshold = int(ck_cfg.get("color_threshold", 35))
        edge_blur = float(ck_cfg.get("edge_blur", 0.08))
        despill_factor = float(ck_cfg.get("despill_factor", 0.5))
        use_gpu = True
        
        # Create processor in threadpool
        try:
            chroma_key_processor = await run_in_threadpool(
                create_chroma_key_processor,
                width=width,
                height=height,
                background_image_url=effective_background_url,
                target_color=target_color,
                color_threshold=color_threshold,
                edge_blur=edge_blur,
                despill_factor=despill_factor,
                use_gpu=use_gpu,
            )
            
            if chroma_key_processor:
                stream_state.chroma_key_processor = chroma_key_processor
                logger.info(f"Stream {stream_state.stream_id}: Chroma key processor initialized")
            else:
                logger.error(f"Stream {stream_state.stream_id}: Failed to create chroma key processor")
                
        except Exception as e:
            logger.error(f"Stream {stream_state.stream_id}: Error creating chroma key processor: {e}")
    
    def get_stream(self, stream_id: str) -> Optional[StreamState]:
        """Get stream state by ID"""
        with self.streams_lock:
            return self.active_streams.get(stream_id)
    
    def update_stream_activity(self, stream_id: str):
        """Update last activity time for a stream"""
        with self.streams_lock:
            if stream_id in self.active_streams:
                self.active_streams[stream_id].update_activity()
    
    def set_audio_generator(self, stream_id: str, audio_generator: AudioStreamGenerator):
        """Set audio generator for a stream"""
        with self.streams_lock:
            if stream_id in self.active_streams:
                self.active_streams[stream_id].audio_generator = audio_generator
    
    def set_generator_task(self, stream_id: str, task: asyncio.Task):
        """Set generator task for a stream"""
        with self.streams_lock:
            if stream_id in self.active_streams:
                self.active_streams[stream_id].generator_task = task
    
    async def cleanup_stream(self, stream_id: str, reason: str = "normal"):
        """
        Clean up a stream and its resources
        
        Args:
            stream_id: Stream ID to cleanup
            reason: Reason for cleanup (for logging)
        """
        stream_state = None
        
        # Remove from active streams
        with self.streams_lock:
            stream_state = self.active_streams.pop(stream_id, None)
            if stream_state:
                self.total_streams_completed += 1
        
        if not stream_state:
            logger.warning(f"Attempted to cleanup non-existent stream {stream_id}")
            return
        
        logger.info(f"Cleaning up stream {stream_id} (reason: {reason}, "
                   f"duration: {(datetime.now() - stream_state.start_time).total_seconds():.1f}s)")
        
        try:
            # Stop and cleanup audio generator
            if stream_state.audio_generator:
                try:
                    stream_state.audio_generator.close()
                    logger.info(f"Stream {stream_id}: Audio generator closed")
                except Exception as e:
                    logger.warning(f"Stream {stream_id}: Error closing audio generator: {e}")
            
            # Cancel generator task
            if stream_state.generator_task and not stream_state.generator_task.done():
                try:
                    stream_state.generator_task.cancel()
                    await asyncio.wait_for(stream_state.generator_task, timeout=2.0)
                    logger.info(f"Stream {stream_id}: Generator task cancelled")
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info(f"Stream {stream_id}: Generator task cleanup completed")
                except Exception as e:
                    logger.warning(f"Stream {stream_id}: Error cancelling generator task: {e}")
            
            # Stop inference for this specific stream
            try:
                model_name = stream_state.model_name
                active_model_streams = [s for s in self.active_streams.values() 
                                       if s.model_name == model_name]
                
                # Stop the specific stream's inference
                logger.info(f"Stream {stream_id}: Stopping inference for stream (model: {model_name})")
                self.inference_api.stop_stream(model_name, stream_id)
                
                if not active_model_streams:
                    logger.info(f"Stream {stream_id}: No other active streams for model {model_name}")
                else:
                    logger.info(f"Stream {stream_id}: {len(active_model_streams)} other streams still active for model {model_name}")
                    
            except Exception as e:
                logger.error(f"Stream {stream_id}: Error stopping inference: {e}")
            
            # Cleanup chroma key processor (no explicit cleanup needed for GPU resources)
            if stream_state.chroma_key_processor:
                stream_state.chroma_key_processor = None
                logger.info(f"Stream {stream_id}: Chroma key processor cleaned up")
            
            logger.info(f"Stream {stream_id}: Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Stream {stream_id}: Error during cleanup: {e}")
            import traceback
            traceback.print_exc()
    
    async def cleanup_all_streams(self, reason: str = "shutdown"):
        """Clean up all active streams"""
        stream_ids = list(self.active_streams.keys())
        logger.info(f"Cleaning up {len(stream_ids)} active streams (reason: {reason})")
        
        # Cleanup all streams concurrently
        cleanup_tasks = [self.cleanup_stream(stream_id, reason) for stream_id in stream_ids]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("All streams cleaned up")
    
    @asynccontextmanager
    async def stream_context(self, websocket: WebSocket, avatar_name: str, 
                           avatar_config: Dict[str, Any], sample_rate: int,
                           background_url: Optional[str] = None,
                           client_info: Optional[Dict[str, Any]] = None):
        """
        Context manager for stream lifecycle management
        
        Usage:
            async with stream_manager.stream_context(...) as stream_id:
                # Use stream_id for operations
                pass
            # Stream is automatically cleaned up
        """
        stream_id = None
        try:
            stream_id = await self.create_stream(
                websocket, avatar_name, avatar_config, sample_rate, 
                background_url, client_info
            )
            yield stream_id
        except Exception as e:
            logger.error(f"Error in stream context: {e}")
            raise
        finally:
            if stream_id:
                await self.cleanup_stream(stream_id, "context_exit")

# Global stream manager instance (will be initialized in lifespan)
stream_manager: Optional[StreamManager] = None
