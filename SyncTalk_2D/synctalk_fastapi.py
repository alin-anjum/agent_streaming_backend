import torch
import torchaudio.transforms as T
import asyncio
import os
import configparser
import logging
import traceback
import numpy as np
import threading
from contextlib import asynccontextmanager

from datetime import datetime
from frame_message_pb2 import FrameMessage 
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv
from config_loader import (
    load_avatar_configs,
    default_avatar_configs,
)


from inference_system.api import InferenceAPI
from audio_stream_generator import AudioStreamGenerator, ALL_MARKERS, START_MARKERS, END_MARKERS

import uvicorn

# Configure logging
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler
file_handler = logging.FileHandler(os.path.join(log_dir, 'synctalk_api.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Configure root logger to avoid duplicate messages
logging.getLogger().handlers = []

# Log that logging is configured
logger.info("Logging configured - writing to console and file: logs/synctalk_api.log")

# Load environment variables from .env if present
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global inference_api, stream_manager
    
    logger.info("Initializing SyncTalk FastAPI server with multi-stream support")
    
    # Basic CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        logger.info(f"CUDA available - GPU: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available")
    
    # Initialize inference API
    inference_api = InferenceAPI(dataset_base_path="./model")
    logger.info("Inference API initialized")

    # Initialize stream manager
    max_concurrent_streams = int(os.getenv("MAX_CONCURRENT_STREAMS", "5"))
    stream_manager = StreamManager(inference_api, max_concurrent_streams)
    logger.info(f"Stream manager initialized (max concurrent: {max_concurrent_streams})")

    # Start background loading of all avatar models (non-blocking)
    global background_loading_task
    background_loading_task = asyncio.create_task(background_load_all_models())
    logger.info("Background model loading started - FastAPI ready to serve requests")
    
    yield
    
    # Shutdown - cleanup background task
    logger.info("Shutting down SyncTalk FastAPI server")
    
    # Cleanup all active streams
    if stream_manager:
        try:
            await stream_manager.cleanup_all_streams("server_shutdown")
            logger.info("All streams cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up streams: {e}")
    
    # Cancel background tasks
    if background_loading_task and not background_loading_task.done():
        background_loading_task.cancel()
        try:
            await background_loading_task
        except asyncio.CancelledError:
            pass

app = FastAPI(lifespan=lifespan)

# Global variables
inference_api = None
is_model_loading = False
config = configparser.ConfigParser()
avatar_configs: Dict[str, Any] = {}
last_inference_time: Optional[datetime] = None  # Global tracking of last inference activity

def update_global_inference_time():
    """Update the global last inference time to current time"""
    global last_inference_time
    last_inference_time = datetime.now()

# Import stream manager
from stream_manager import StreamManager

# Stream manager instance (initialized in lifespan)
stream_manager: Optional[StreamManager] = None

# Model cache tracking - stores config for each loaded model by folder_name
loaded_model_configs: Dict[str, Dict[str, Any]] = {}

# Background loading task reference
background_loading_task: Optional[asyncio.Task] = None
timestamp_update_task: Optional[asyncio.Task] = None

# Stream markers are now imported from audio_stream_generator

# Logging counters
websocket_send_log_counter = 0
# audio_none_log_counter moved to audio_stream_generator module

# Load config if exists
try:
    config.read("config.cfg", encoding="utf-8")
except:
    logger.warning("No config.cfg found, using defaults")

class ModelRequest(BaseModel):
    model_name: str
    config_type: str = "default"

# -----------------------------
# Avatar configuration loading
# -----------------------------

async def load_or_refresh_avatar_configs() -> Dict[str, Any]:
    """Wrapper to load avatar configs synchronously in a threadpool for non-blocking IO."""
    global avatar_configs
    configs = await run_in_threadpool(load_avatar_configs, logger)
    avatar_configs = configs
    return configs

def get_crop_size_from_bbox(crop_bbox: Optional[list]) -> Optional[tuple]:
    if not crop_bbox or len(crop_bbox) != 4:
        return None
    x1, y1, x2, y2 = crop_bbox
    return max(0, x2 - x1), max(0, y2 - y1)

def is_model_already_loaded(new_config: Dict[str, Any]) -> bool:
    """
    Check if model is already loaded with identical configuration.
    Returns True if configurations match AND model exists in inference API.
    """
    if inference_api is None:
        logger.info("Inference API not initialized")
        return False
    
    # Check if model exists in inference API
    folder_name = new_config.get('folder_name')
    available_models = list(inference_api.models.keys())
    if folder_name not in inference_api.models:
        logger.info(f"Model '{folder_name}' not found in inference API models. Available: {available_models}")
        return False
    else:
        logger.info(f"Model '{folder_name}' found in inference API. Available: {available_models}")
    
    # Check if we have a cached config for this model
    if folder_name not in loaded_model_configs:
        logger.info(f"No cached configuration found for model '{folder_name}'")
        return False
    
    loaded_config = loaded_model_configs[folder_name]
    
    # Compare all relevant configuration parameters
    config_keys = [
        'avatar_name', 'checkpoint', 'video_path', 'crop_bbox', 
        'resize_dims', 'is_old_unet', 'mode', 'batch_size', 'folder_name'
    ]
    
    for key in config_keys:
        if new_config.get(key) != loaded_config.get(key):
            logger.info(f"Model config difference detected in '{key}': "
                       f"new='{new_config.get(key)}' vs loaded='{loaded_config.get(key)}'")
            return False
    
    logger.info(f"Model '{folder_name}' is already loaded with identical configuration")
    return True

async def background_load_all_models():
    """
    Background task to preload all avatar models without blocking FastAPI startup.
    Simple approach: just call preload_models and cache configs for successful loads.
    """
    global loaded_model_configs
    
    try:
        logger.info("Starting background loading of all avatar models...")
        
        # Get avatar configurations
        avatars = await load_or_refresh_avatar_configs()
        
        if not avatars:
            logger.warning("No avatar configurations found for background loading")
            return
        
        # Build model configurations
        all_model_configs = {}
        preload_cache_configs = {}
        
        for avatar_name, avatar_config in avatars.items():
            try:
                # Check for preload flag - default to False if not present
                if not avatar_config.get("preload", False):
                    logger.info(f"Skipping preload for {avatar_name} (preload flag is not set to true)")
                    continue
                    
                folder_name = avatar_config.get("folder_name", avatar_name)
                
                checkpoint = avatar_config.get("checkpoint")
                video_path = avatar_config.get("video_path")
                crop_bbox_list = avatar_config.get("crop_bbox")
                crop_bbox = None
                if crop_bbox_list is not None and len(crop_bbox_list) == 4:
                    crop_bbox = tuple(crop_bbox_list)
                
                # Get resize dimensions if specified
                resize_dims_list = avatar_config.get("resize_dims")
                resize_dims = None
                if resize_dims_list is not None and len(resize_dims_list) == 2:
                    resize_dims = tuple(resize_dims_list)
                
                # Check if this avatar should use the old UNet version
                is_old_unet = avatar_config.get("is_old_unet", False)

                # Use code defaults for mode and batch_size
                mode = "ave"
                batch_size = 4

                # Skip if checkpoint doesn't exist
                if not checkpoint or not os.path.exists(checkpoint):
                    logger.warning(f"Checkpoint not found for {avatar_name}: {checkpoint}, skipping")
                    continue
                
                # Skip if video doesn't exist
                if not video_path or not os.path.exists(video_path):
                    logger.warning(f"Video not found for {avatar_name}: {video_path}, skipping")
                    continue

                # Configure model
                model_config = {
                    "checkpoint": checkpoint,
                    "mode": mode,
                    "batch_size": batch_size,
                    "video_path": video_path,
                    "use_old_unet": is_old_unet,
                }
                
                # Get frame_range if specified
                frame_range_list = avatar_config.get("frame_range")
                if frame_range_list is not None and len(frame_range_list) == 2:
                    model_config["frame_range"] = tuple(frame_range_list)
                
                # Only add crop_bbox if it's provided in config
                if crop_bbox is not None:
                    model_config["crop_bbox"] = crop_bbox
                    
                # Only add resize_dims if it's provided in config
                if resize_dims is not None:
                    model_config["resize_dims"] = resize_dims
                
                all_model_configs[folder_name] = model_config
                
                # Store full config for caching (used for comparison in load_model)
                cache_config = {
                    'avatar_name': avatar_name,
                    'folder_name': folder_name,
                    'checkpoint': checkpoint,
                    'video_path': video_path,
                    'crop_bbox': crop_bbox,
                    'resize_dims': resize_dims,
                    'is_old_unet': is_old_unet,
                    'mode': mode,
                    'batch_size': batch_size,
                    'frame_range': tuple(frame_range_list) if frame_range_list else None
                }
                preload_cache_configs[folder_name] = cache_config
                
                logger.info(f"Prepared config for {avatar_name} (folder: {folder_name})")
                
            except Exception as e:
                logger.error(f"Failed to prepare config for {avatar_name}: {e}")
                continue
        
        if not all_model_configs:
            logger.warning("No valid model configurations found for background loading")
            return
        
        logger.info(f"Background loading {len(all_model_configs)} models: {list(all_model_configs.keys())}")
        
        # Load all models at once
        await run_in_threadpool(inference_api.preload_models, all_model_configs)
        
        # Cache configurations and warm up successfully loaded models
        for avatar_name, avatar_config in avatars.items():
            # Only check models that were marked for preload
            if not avatar_config.get("preload", False):
                continue

            folder_name = avatar_config.get("folder_name", avatar_name)

            # Check if the model was actually loaded successfully
            if folder_name in inference_api.models:
                loaded_model_configs[folder_name] = preload_cache_configs[folder_name]
                logger.info(f"Background loaded and cached: {folder_name}")

                # Warm up the model if warmup_on_load is true
                if avatar_config.get("warmup_on_load", False):
                    logger.info(f"🔥 Warming up model {folder_name}...")
                    await run_in_threadpool(inference_api.warmup_model, folder_name, warmup_batches=5)
                    logger.info(f"Model {folder_name} warmed up")
            # Check if it was supposed to be loaded but failed
            elif folder_name in all_model_configs:
                logger.warning(f"Background load failed: {folder_name}")
        
        completed_models = [f for f in all_model_configs.keys() if f in inference_api.models]
        logger.info(f"Background loading complete! Loaded {len(completed_models)}/{len(all_model_configs)} models: {completed_models}")
        
    except Exception as e:
        logger.error(f"Background loading failed: {e}")
        logger.error(traceback.format_exc())

# Legacy functions removed - functionality moved to StreamManager
# AudioStreamGenerator moved to separate module to avoid circular imports

@app.get("/models")
async def list_models():
    """List available models from avatar config (remote/local)."""
    try:
        avatars = await load_or_refresh_avatar_configs()
        return {"models": sorted(list(avatars.keys()))}
    except Exception:
        # Fallback to default
        return {"models": sorted(list(default_avatar_configs().keys()))}

@app.post("/load_model")
async def load_model(request: ModelRequest):
    """Load a model using the new inference API based on avatar config (remote/local)."""
    global is_model_loading, loaded_model_configs
    
    if is_model_loading:
        raise HTTPException(status_code=409, detail="Another model is currently being loaded")
    
    try:
        is_model_loading = True
        
        # Refresh avatar configs
        avatars = await load_or_refresh_avatar_configs()
        avatar_name = request.model_name
        if avatar_name not in avatars:
            raise HTTPException(status_code=404, detail=f"Avatar '{avatar_name}' not found in config")

        avatar_config = avatars[avatar_name]

        # Use folder_name for internal functions, fallback to avatar_name if not specified
        folder_name = avatar_config.get("folder_name", avatar_name)
        
        checkpoint = avatar_config.get("checkpoint")
        video_path = avatar_config.get("video_path")
        crop_bbox_list = avatar_config.get("crop_bbox")
        crop_bbox = None
        if crop_bbox_list is not None and len(crop_bbox_list) == 4:
            crop_bbox = tuple(crop_bbox_list)
        
        # Get resize dimensions if specified
        resize_dims_list = avatar_config.get("resize_dims")
        resize_dims = None
        if resize_dims_list is not None and len(resize_dims_list) == 2:
            resize_dims = tuple(resize_dims_list)
        
        # Check if this avatar should use the old UNet version
        is_old_unet = avatar_config.get("is_old_unet", False)

        # Use code defaults for mode and batch_size
        mode = "ave"
        batch_size = 4
        
        # Get frames to load for comparison
        frame_range_list = avatar_config.get("frame_range")

        # Build configuration for comparison
        new_model_config = {
            'avatar_name': avatar_name,
            'folder_name': folder_name,
            'checkpoint': checkpoint,
            'video_path': video_path,
            'crop_bbox': crop_bbox,
            'resize_dims': resize_dims,
            'is_old_unet': is_old_unet,
            'mode': mode,
            'batch_size': batch_size,
            'frame_range': tuple(frame_range_list) if frame_range_list else None
        }

        # Config will be stored per-stream when streams are created

        # Check if this exact model configuration is already loaded and exists in inference API
        if is_model_already_loaded(new_model_config):
            logger.info(f"Model {avatar_name} (folder: {folder_name}) is already loaded with identical configuration")
            return {"message": f"Model {avatar_name} already loaded"}


        # Configure model (using folder_name for internal model key)
        model_config = {
            folder_name: {
                "checkpoint": checkpoint if checkpoint and os.path.exists(checkpoint) else None,
                "mode": mode,
                "batch_size": batch_size,
                "video_path": video_path,
                "use_old_unet": is_old_unet,
            }
        }
        
        # Get frame_range if specified
        if frame_range_list is not None and len(frame_range_list) == 2:
            model_config[folder_name]["frame_range"] = tuple(frame_range_list)
            
        # Only add crop_bbox if it's provided in config
        if crop_bbox is not None:
            model_config[folder_name]["crop_bbox"] = crop_bbox
            
        # Only add resize_dims if it's provided in config
        if resize_dims is not None:
            model_config[folder_name]["resize_dims"] = resize_dims
        
        logger.info(f"⚙️ Model {avatar_name} (folder: {folder_name}) not found in preloaded models or config changed, loading fresh...")
        logger.info(f"Loading model {avatar_name} with config: {model_config}")
        
        # Load model using new API
        await run_in_threadpool(inference_api.preload_models, model_config)
        
        # Warm up the newly loaded model if warmup_on_load is true
        if avatar_config.get("warmup_on_load", False):
            logger.info(f"🔥 Warming up model {folder_name}...")
            await run_in_threadpool(inference_api.warmup_model, folder_name, warmup_batches=5)
            logger.info(f"Model {folder_name} warmed up")
        
        # Video frames and chroma key will be initialized per-stream
        logger.info("Model loaded - video frames and chroma key will be initialized per-stream")

        # Store the successfully loaded configuration for future comparison
        loaded_model_configs[folder_name] = new_model_config.copy()
        
        logger.info(f"Model {avatar_name} loaded successfully and cached")
        return {"message": f"Model {avatar_name} loaded successfully"}
        
    except Exception as e:
        logger.error(f"Error loading model {request.model_name}: {str(e)}")
        logger.error(traceback.format_exc())
        # Clear cached config on error to force reload next time
        if 'folder_name' in locals():
            loaded_model_configs.pop(folder_name, None)
            logger.info(f"Cleared cached config for {folder_name} due to loading error")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    finally:
        is_model_loading = False

@app.get("/status")
async def get_service_status():
    """Get current service status including multi-stream information"""
    global last_inference_time
    if not stream_manager:
        return {"error": "Stream manager not initialized"}
    
    stream_info = stream_manager.get_stream_info()
    active_count = stream_info['active_count']
    max_concurrent = stream_info['max_concurrent']
    
    return {
        "available": active_count < max_concurrent,
        "status": "ready" if active_count < max_concurrent else "busy",
        "is_model_loaded": inference_api is not None and len(inference_api.models) > 0,
        "loaded_models": list(inference_api.models.keys()) if inference_api else [],
        "last_inference_time": last_inference_time.timestamp() if last_inference_time else None,
        "multi_stream": stream_info,
        "capacity": {
            "current": active_count,
            "maximum": max_concurrent,
            "available": max_concurrent - active_count
        }
    }

@app.get("/streams")
async def get_streams():
    """Get information about active streams"""
    if not stream_manager:
        return {"error": "Stream manager not initialized"}
    
    return stream_manager.get_stream_info()

@app.get("/streams/{stream_id}")
async def get_stream_details(stream_id: str):
    """Get detailed information about a specific stream"""
    if not stream_manager:
        return {"error": "Stream manager not initialized"}
    
    stream_state = stream_manager.get_stream(stream_id)
    if not stream_state:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    return {
        "stream_id": stream_state.stream_id,
        "avatar_name": stream_state.avatar_name,
        "model_name": stream_state.model_name,
        "start_time": stream_state.start_time.isoformat(),
        "last_activity": stream_state.last_activity_time.isoformat(),
        "duration_seconds": (datetime.now() - stream_state.start_time).total_seconds(),
        "sample_rate": stream_state.sample_rate,
        "background_url": stream_state.background_url,
        "has_chroma_key": stream_state.chroma_key_processor is not None,
        "client_info": stream_state.client_info,
        "websocket_state": stream_state.websocket.client_state.name if hasattr(stream_state.websocket.client_state, 'name') else str(stream_state.websocket.client_state.value)
    }

@app.get("/debug/workers")
async def debug_workers():
    """Debug endpoint to check worker status and concurrent stream safety"""
    if inference_api is None:
        return {"error": "No inference API available"}
    
    debug_info = {}
    concurrent_streams = 0
    
    for model_name, wrapper in inference_api.models.items():
        # Count streams using this model
        model_streams = []
        if stream_manager:
            model_streams = [s.stream_id for s in stream_manager.active_streams.values() 
                           if s.model_name == model_name]
        
        concurrent_streams += len(model_streams)
        
        # Check for per-stream workers wrapper
        if hasattr(wrapper, 'stream_workers'):
            worker_info = {
                "architecture": "per-stream-workers",
                "gpu_running": getattr(wrapper, 'gpu_running', False),
                "stream_workers": len(wrapper.stream_workers),
                "active_streams": len(model_streams),
                "stream_ids": model_streams,
                "concurrent_safe": True,  # Per-stream workers are always safe
                "has_shared_resources": {
                    "frame_manager": hasattr(wrapper, 'frame_manager') and wrapper.frame_manager is not None,
                    "landmark_manager": hasattr(wrapper, 'landmark_manager') and wrapper.landmark_manager is not None,
                    "gpu_worker": getattr(wrapper, 'gpu_running', False)
                }
            }
        else:
            worker_info = {
                "architecture": "legacy",
                "worker_status": wrapper.debug_worker_status() if hasattr(wrapper, 'debug_worker_status') else "unknown",
                "workers_running": getattr(wrapper, 'running', False),
                "active_streams": len(model_streams),
                "stream_ids": model_streams,
                "concurrent_safe": len(model_streams) <= 1 or getattr(wrapper, 'running', False),
                "has_shared_resources": {
                    "frame_manager": hasattr(wrapper, 'frame_manager') and wrapper.frame_manager is not None,
                    "landmark_manager": hasattr(wrapper, 'landmark_manager') and wrapper.landmark_manager is not None
                }
            }
        
        debug_info[model_name] = worker_info
    
    return {
        "total_concurrent_streams": concurrent_streams,
        "concurrent_safety_status": "All models safe for concurrent streams" if all(
            info["concurrent_safe"] for info in debug_info.values()
        ) else "⚠️ Some models may have concurrent issues",
        "per_model_debug": debug_info,
        "architecture_status": "Using per-stream worker architecture for optimal isolation"
    }

@app.get("/debug/resources")
async def get_resource_usage():
    """Get resource usage information showing memory efficiency across streams"""
    if not stream_manager or not inference_api:
        return {"error": "Services not initialized"}
    
    resource_info = {}
    
    # Analyze resource sharing per model
    for model_name, wrapper in inference_api.models.items():
        model_streams = [s for s in stream_manager.active_streams.values() 
                        if s.model_name == model_name]
        
        resource_info[model_name] = {
            "active_streams": len(model_streams),
            "stream_ids": [s.stream_id for s in model_streams],
            "frame_manager": {
                "is_preloaded": hasattr(wrapper, 'frame_manager') and wrapper.frame_manager is not None,
                "frames_loaded": wrapper.frame_manager.frames_loaded_count if hasattr(wrapper, 'frame_manager') and wrapper.frame_manager else 0,
                "video_shape": wrapper.frame_manager.video_shape if hasattr(wrapper, 'frame_manager') and wrapper.frame_manager else None,
                "shared_across_streams": len(model_streams) > 1
            },
            "landmark_manager": {
                "is_preloaded": hasattr(wrapper, 'landmark_manager') and wrapper.landmark_manager is not None,
                "landmarks_loaded": len(wrapper.landmark_manager.landmarks) if hasattr(wrapper, 'landmark_manager') and wrapper.landmark_manager else 0,
                "shared_across_streams": len(model_streams) > 1
            },
            "memory_efficiency": "optimal" if len(model_streams) > 1 else "single_stream"
        }
    
    # Calculate overall efficiency
    total_streams = len(stream_manager.active_streams)
    total_models = len(resource_info)
    shared_models = sum(1 for info in resource_info.values() if info["frame_manager"]["shared_across_streams"])
    
    return {
        "summary": {
            "total_active_streams": total_streams,
            "unique_models_loaded": total_models,
            "models_with_shared_resources": shared_models,
            "memory_efficiency_score": f"{shared_models}/{total_models}" if total_models > 0 else "N/A"
        },
        "per_model_resources": resource_info,
        "optimization_status": "Resources are being reused efficiently" if shared_models > 0 else "No resource sharing (single streams per model)"
    }




def is_audio_silent(audio_bytes):
    """Check if audio data is completely silent (all zeros)"""
    if audio_bytes is None:
        return True
    return all(b == 0 for b in audio_bytes)

async def send_frame_to_websocket(websocket: WebSocket, video_data, audio_chunk, current_frame, end_of_stream, start_of_stream):
    """Send a single frame directly to websocket with non-blocking serialization"""
    global websocket_send_log_counter
    
    try:
        # Move blocking operations to thread pool
        def prepare_data():
            # Create protobuf message
            frame_msg = FrameMessage()
            
            if video_data is not None:
                frame_msg.video_bytes = video_data.tobytes()
            
            if audio_chunk is not None:
                frame_msg.audio_bytes = audio_chunk
            
            if start_of_stream:
                frame_msg.start_speech = True
            
            if end_of_stream:
                frame_msg.end_speech = True
            
            # Serialize to bytes (much faster than pickle)
            return frame_msg.SerializeToString()
        
        # Always log important stream events
        if start_of_stream:
            logger.info("Detected start of stream")
        if end_of_stream:
            logger.info("Detected end of stream")
        
        # Log websocket send details only every 100 frames or for special events
        websocket_send_log_counter += 1
        should_log = False
        
        if should_log:
            video_size = video_data.size * video_data.itemsize if video_data is not None else 0
            audio_size = len(audio_chunk) if audio_chunk is not None else 0
            
            log_msg = f"WEBSOCKET_SEND: frame={current_frame}, video_bytes={video_size}, audio_bytes={audio_size}"
            if start_of_stream:
                log_msg += " [START_STREAM]"
            if end_of_stream:
                log_msg += " [END_STREAM]"
            if websocket_send_log_counter % 100 == 0:
                log_msg += f" [LOG_COUNT: {websocket_send_log_counter}]"
            
            logger.info(log_msg)
        
        # Execute blocking operations in thread pool
        protobuf_bytes = await asyncio.get_event_loop().run_in_executor(
            None, prepare_data
        )
        
        # Send non-blocking
        await websocket.send_bytes(protobuf_bytes)
        
    except Exception as e:
        logger.error(f"Error sending frame to websocket: {str(e)}")
        raise

async def video_generator_task(stream_id: str, last_index: int = 0):
    """Stream-specific video generation task.
    Generates video from audio stream and sends to websocket for a specific stream.
    """
    stream_state = stream_manager.get_stream(stream_id)
    if not stream_state:
        logger.error(f"Stream {stream_id} not found")
        return

    websocket = stream_state.websocket
    model_name = stream_state.model_name
    audio_stream = stream_state.audio_generator
    chroma_key_processor = stream_state.chroma_key_processor

    logger.info(f"Starting video generation for stream {stream_id} (model: {model_name})")

    loop = asyncio.get_running_loop()
    # Increased queue size to 13 (1/2 second of frames) to provide a better
    # buffer and absorb transient delays in the consumer (e.g., chroma keying).
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=13)

    # Sentinels to signal end and error from the producer thread
    DONE = object()
    ERROR = object()
    
    # Flag to signal producer to stop
    producer_stop_event = threading.Event()

    def producer() -> None:
        try:
            logger.info(f"Producer thread starting for stream {stream_id}, model: {model_name}")
            for frame, img_idx, current_frame, frame_audio in inference_api.generate_video_stream(
                model_name=model_name,
                audio_chunks=audio_stream,
                is_silent=False,
                last_index=last_index,
                continue_silent=True,
                frame_generation_rate=25.0,
                stream_id=stream_id,  # Pass stream_id for tracking
            ):
                # Check if we should stop
                if producer_stop_event.is_set():
                    logger.info(f"Producer thread stopping for stream {stream_id} (stop event set)")
                    break
                    
                # Block when the queue is full to apply backpressure
                asyncio.run_coroutine_threadsafe(
                    frame_queue.put((frame, img_idx, current_frame, frame_audio)),
                    loop
                ).result()
            logger.info(f"Producer thread finished normally for stream {stream_id}")
        except Exception as e:
            logger.error(f"Error in generation producer for stream {stream_id}: {e}")
            logger.error(traceback.format_exc())
            asyncio.run_coroutine_threadsafe(
                frame_queue.put((ERROR, e, None, None)),
                loop
            ).result()
        finally:
            logger.info(f"Producer thread exiting for stream {stream_id}")
            asyncio.run_coroutine_threadsafe(
                frame_queue.put((DONE, None, None, None)),
                loop
            ).result()

    # Start producer on a worker thread
    producer_task = asyncio.create_task(asyncio.to_thread(producer))

    frame_count = 0
    start_time = datetime.now()
    last_processed_frame_index = last_index

    try:
        while True:
            item = await frame_queue.get()
            tag = item[0]
            if tag is DONE:
                break
            if tag is ERROR:
                break

            frame, img_idx, current_frame, frame_audio = item

            # Handle stream markers
            if isinstance(frame, bytes) and frame in ALL_MARKERS:
                start_of_stream = frame in START_MARKERS
                end_of_stream = frame in END_MARKERS
                await send_frame_to_websocket(websocket, None, None, last_processed_frame_index, end_of_stream, start_of_stream)
                logger.info(f"Stream {stream_id}: Received stream marker: {frame.decode()}")
                continue

            if frame is None:
                logger.info(f"Stream {stream_id}: Received None frame.")
                break

            # Convert BGR to RGB if frame has 3 channels
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = frame[:, :, ::-1]

            # Apply chroma key processing if enabled (offload to executor)
            processed_frame = frame
            if chroma_key_processor is not None:
                try:
                    processed_frame = await loop.run_in_executor(None, chroma_key_processor.process, frame)
                except Exception as e:
                    logger.error(f"Stream {stream_id}: Error in chroma key processing: {e}")
                    # Fall back to original frame on error
                    processed_frame = frame

            # Convert float32 audio to int16
            frame_audio_int16 = (frame_audio * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()

            # Check websocket state before sending frame
            if websocket.client_state.value in [3, 4]:  # CLOSED or CLOSING
                logger.info(f"Stream {stream_id}: Websocket disconnected during video generation, stopping...")
                producer_stop_event.set()  # Signal producer to stop
                break
            
            # Send processed frame
            try:
                await send_frame_to_websocket(websocket, processed_frame, frame_audio_int16, current_frame, False, False)
                # Update stream activity and global inference time
                stream_manager.update_stream_activity(stream_id)
                update_global_inference_time()
            except Exception as e:
                # If sending fails, websocket is likely disconnected
                error_msg = str(e)
                if "websocket" in error_msg.lower() or "connection" in error_msg.lower():
                    logger.info(f"Stream {stream_id}: Websocket disconnected during frame send: {error_msg}")
                    break
                else:
                    logger.error(f"Stream {stream_id}: Error sending frame: {error_msg}")
                    raise
                    
            frame_count += 1
            last_processed_frame_index = current_frame
            
            # Log progress and FPS periodically
            if frame_count % 25 == 0:  # Log every 25 frames for more frequent updates
                elapsed = (datetime.now() - start_time).total_seconds()
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Stream {stream_id}: Generation FPS: {fps:.2f} | Frames: {frame_count} | Current: {current_frame}")
        
        # Final FPS summary
        elapsed = (datetime.now() - start_time).total_seconds()
        final_fps = frame_count / elapsed if elapsed > 0 else 0
        logger.info(f"Stream {stream_id}: Video generation task completed. Total frames: {frame_count}, Final FPS: {final_fps:.2f}, Duration: {elapsed:.2f}s")

    except Exception as e:
        logger.error(f"Stream {stream_id}: Error in video generator task: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Signal end of video generation (only if websocket is still connected)
        try:
            if websocket.client_state.value not in [3, 4]:  # Not CLOSED or CLOSING
                await send_frame_to_websocket(websocket, None, None, last_processed_frame_index, True, False)
        except Exception as e:
            logger.warning(f"Stream {stream_id}: Could not send final frame (websocket likely disconnected): {e}")
            
        # Signal producer to stop
        producer_stop_event.set()
            
        # Cleanup producer task
        try:
            await asyncio.wait_for(producer_task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.warning(f"Stream {stream_id}: Producer task did not complete in time")
        except Exception as e:
            logger.warning(f"Stream {stream_id}: Error waiting for producer task: {e}")
        
        logger.info(f"Stream {stream_id}: Video generation task cleanup complete")

@app.websocket("/ws/audio_to_video")
async def websocket_endpoint(
    websocket: WebSocket, 
    sample_rate: int = Query(16000),
    background_url: Optional[str] = Query(None),
    avatar_name: Optional[str] = Query(None)
):
    """WebSocket endpoint for streaming audio-to-video conversion with multi-stream support"""
    global websocket_send_log_counter
    websocket_send_log_counter = 0
    logger.info(f"New websocket connection attempt (avatar: {avatar_name}, sample_rate: {sample_rate}Hz)")

    if not stream_manager:
        logger.error("Stream manager not initialized")
        await websocket.close(code=1008, reason="Stream manager not initialized")
        return

    if not stream_manager.can_accept_new_stream():
        active_count = stream_manager.get_active_stream_count()
        max_streams = stream_manager.max_concurrent_streams
        logger.warning(f"Rejected websocket connection - max concurrent streams reached ({active_count}/{max_streams})")
        await websocket.close(code=1008, reason=f"Maximum concurrent streams ({max_streams}) exceeded")
        return
    
    if inference_api is None or len(inference_api.models) == 0:
        logger.error("No model loaded - rejecting websocket connection")
        await websocket.close(code=1008, reason="No model loaded. Please load a model first.")
        return
    
    if not avatar_name:
        available_models = list(inference_api.models.keys())
        if not available_models:
            logger.error("No loaded models available")
            await websocket.close(code=1008, reason="No loaded models available")
            return
        avatars = await load_or_refresh_avatar_configs()
        for name, config in avatars.items():
            folder_name = config.get("folder_name", name)
            if folder_name in available_models:
                avatar_name = name
                break
        if not avatar_name:
            logger.error("No avatar configuration found for available models")
            await websocket.close(code=1008, reason="No avatar configuration found for available models")
            return
        logger.info(f"Auto-selected avatar: {avatar_name}")

    avatars = await load_or_refresh_avatar_configs()
    if avatar_name not in avatars:
        logger.error(f"Avatar '{avatar_name}' not found in configurations")
        await websocket.close(code=1008, reason=f"Avatar '{avatar_name}' not found")
        return

    avatar_config = avatars[avatar_name]
    model_name = avatar_config.get("folder_name", avatar_name)

    if model_name not in inference_api.models:
        logger.error(f"Model '{model_name}' not loaded")
        await websocket.close(code=1008, reason=f"Model '{model_name}' not loaded. Please load the model first.")
        return

    await websocket.accept()
    logger.info(f"Websocket connection accepted for avatar '{avatar_name}' (model: {model_name})")

    client_info = {
        'user_agent': websocket.headers.get('user-agent', 'unknown'),
        'remote_addr': getattr(websocket.client, 'host', 'unknown'),
        'avatar_requested': avatar_name
    }

    stream_id = None
    try:
        async with stream_manager.stream_context(
            websocket=websocket,
            avatar_name=avatar_name,
            avatar_config=avatar_config,
            sample_rate=sample_rate,
            background_url=background_url,
            client_info=client_info
        ) as stream_id:
            logger.info(f"Stream {stream_id} created successfully")
            
            audio_stream = AudioStreamGenerator(input_sample_rate=sample_rate)
            stream_manager.set_audio_generator(stream_id, audio_stream)
            
            generator_task = asyncio.create_task(video_generator_task(stream_id, last_index=0))
            stream_manager.set_generator_task(stream_id, generator_task)

            while True:
                try:
                    audio_chunk = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)  # INCREASED 10x
                    
                    if audio_chunk in END_MARKERS:
                        logger.info(f"Stream {stream_id}: end_of_stream marker, bytes={len(audio_chunk)}")
                    elif audio_chunk in START_MARKERS:
                        logger.info(f"Stream {stream_id}: start_of_stream marker, bytes={len(audio_chunk)}")
                    else:
                        logger.info(f"Stream {stream_id}: Added audio chunk of {len(audio_chunk)} bytes @ {sample_rate}Hz to stream")

                    await audio_stream.add_audio_chunk_async(audio_chunk)
                    stream_manager.update_stream_activity(stream_id)
                    update_global_inference_time()
                    
                except asyncio.TimeoutError:
                    if websocket.client_state.value in [3, 4]:  # CLOSED or CLOSING
                        logger.info(f"Stream {stream_id}: Websocket disconnected (detected during timeout)")
                        break
                    continue
                        
                except Exception as e:
                    error_msg = str(e)
                    if "websocket" in error_msg.lower() or "connection" in error_msg.lower():
                        logger.info(f"Stream {stream_id}: Websocket disconnected: {error_msg}")
                    else:
                        logger.error(f"Stream {stream_id}: Error in websocket loop: {error_msg}")
                        logger.error(traceback.format_exc())
                    break

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error during stream lifecycle for stream {stream_id or 'unassigned'}: {error_msg}")
        logger.error(traceback.format_exc())
        if websocket.client_state.value not in [3, 4]:
            await websocket.close(code=1011, reason=f"Server error: {error_msg}")
    
    finally:
        logger.info(f"Websocket endpoint for stream {stream_id or 'unassigned'} is now fully exited.")

# Drop frames endpoint removed - no longer needed with direct websocket publishing

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8001, ws_ping_interval=300, ws_ping_timeout=300)
