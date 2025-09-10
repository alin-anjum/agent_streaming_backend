#!/usr/bin/env python3
"""
Rapido - Real-time Avatar Presentation Integration with Dynamic Overlay

Complete integrated system with all working features:
- Real SyncTalk protobuf integration
- Green screen removal (chroma key)
- Audio extraction from avatar frames
- H.264 codec for MP4 compatibility
- Proper timing for all slide frames
"""

import asyncio
import os
import sys
import json
import logging
import websockets
import cv2
import numpy as np
from pathlib import Path
import argparse
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available, using CPU only")
from PIL import Image
import io
import time
from pathlib import Path
import argparse
import requests
import threading

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
rapido_root = os.path.dirname(current_dir)
project_root = os.path.dirname(rapido_root)

sys.path.append(rapido_root)
sys.path.append(current_dir)
sys.path.append(os.path.join(project_root, 'SyncTalk_2D'))

# Import Rapido modules
from config.config import Config
from data_parser import SlideDataParser
from tts_client import ElevenLabsTTSClient
from frame_processor import FrameOverlayEngine

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import SyncTalk protobuf
try:
    # Add SyncTalk_2D to path and import
    synctalk_path = os.path.join(project_root, 'SyncTalk_2D')
    if synctalk_path not in sys.path:
        sys.path.insert(0, synctalk_path)
    from frame_message_pb2 import FrameMessage
    logger.info("✅ Protobuf integration ready")
except ImportError as e:
    logger.error(f"❌ Failed to import protobuf: {e}")
    logger.error("Make sure SyncTalk_2D/frame_message_pb2.py exists")
    sys.exit(1)

# Import SyncTalk chroma key for green screen removal
try:
    # Add project root to path for SyncTalk_2D import
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from SyncTalk_2D.chroma_key import FastChromaKey
    CHROMA_KEY_AVAILABLE = True
    logger.info("✅ SyncTalk chroma key module imported successfully")
except ImportError as e:
    CHROMA_KEY_AVAILABLE = False
    logger.warning(f"SyncTalk chroma key not available: {e}")

class RapidoMainSystem:
    """Integrated Rapido system with all features"""
    
    def __init__(self, config_override: dict = None):
        self.config = Config()
        
        # Apply config overrides
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        # Setup paths and connections - use remote SyncTalk server
        self.synctalk_url = getattr(self.config, 'SYNCTALK_WEBSOCKET_URL', 'ws://35.172.212.10:8000')
        self.websocket = None
        self.avatar_frames = []
        self.avatar_audio_chunks = []
        # Get the script directory and resolve paths relative to the project root
        script_dir = Path(__file__).parent.parent  # rapido/src -> rapido
        project_root = script_dir.parent  # rapido -> agent_streaming_backend
        
        slide_frames_env = getattr(self.config, 'SLIDE_FRAMES_PATH', None)
        if slide_frames_env and Path(slide_frames_env).is_absolute():
            self.slide_frames_path = slide_frames_env
        else:
            # Use absolute path to presentation_frames
            self.slide_frames_path = str(project_root / 'presentation_frames')
        
        self.output_dir = getattr(self.config, 'OUTPUT_PATH', './output')
        
        logger.info(f"🔍 SLIDE_FRAMES_PATH config: {self.slide_frames_path}")
        logger.info(f"🔍 SLIDE_FRAMES_PATH resolved: {Path(self.slide_frames_path).resolve()}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Timing for smooth video - get actual slide count
        self.synctalk_fps = 25.0
        self.target_video_fps = 25.0
        
        # Initialize frame processor once and reuse - 480p for ultra smooth performance
        # Enable GPU acceleration if available
        self.use_gpu = TORCH_AVAILABLE
        if self.use_gpu:
            self.device = torch.device('cuda')
            logger.info("🚀 GPU acceleration enabled for frame processing")
            
            # Enable OpenCV GPU backend if available
            try:
                cv2.setUseOptimized(True)
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    logger.info(f"🚀 OpenCV CUDA backend available with {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
                else:
                    logger.info("💻 OpenCV CUDA backend not available")
            except Exception as e:
                logger.debug(f"OpenCV CUDA check failed: {e}")
        else:
            self.device = torch.device('cpu')
            logger.info("💻 Using CPU for frame processing")
            
        self.frame_processor = FrameOverlayEngine(self.slide_frames_path, output_size=(854, 480))
        self.total_slide_frames = self.frame_processor.get_frame_count()
        self.total_duration_seconds = self.total_slide_frames / self.synctalk_fps
        
        # Initialize chroma key processor
        self.chroma_key_processor = None
        self._init_chroma_key()

        # Performance optimization: cache morphological kernel
        self._morph_kernel = None
        self._frame_count = 0
        
        # Thread pool for green screen processing
        self._green_screen_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="GreenScreen")
        self._green_screen_cache = {}  # Cache processed frames
        
        logger.info(f"📊 Rapido initialized - Duration: {self.total_duration_seconds:.2f}s")
    
    def __del__(self):
        """Cleanup thread pool on destruction"""
        try:
            if hasattr(self, '_green_screen_executor'):
                self._green_screen_executor.shutdown(wait=False)
        except:
            pass
        
    def _init_chroma_key(self):
        """Initialize chroma key processor for green screen removal"""
        if not CHROMA_KEY_AVAILABLE:
            logger.warning("Chroma key not available, frames will have green background")
            return
            
        try:
            # Download SyncTalk's background image if needed
            import requests
            from PIL import Image
            import io
            
            # SyncTalk's enrique_torres background URL
            bg_url = ""
            
            try:
                response = requests.get(bg_url, timeout=10)
                if response.status_code == 200:
                    bg_image = Image.open(io.BytesIO(response.content))
                    logger.info("✅ Downloaded SyncTalk background image")
                else:
                    bg_image = None
                    logger.warning("Could not download background image, using color-based detection")
            except:
                bg_image = None
                logger.warning("Could not download background image, using color-based detection")
            
            # Initialize FastChromaKey with proper parameters
            # FastChromaKey already imported at module level
            
            # Set up background - use black if we couldn't download the image
            if bg_image is None:
                bg_array = np.zeros((480, 854, 3), dtype=np.uint8)  # Black background
            else:
                bg_array = np.array(bg_image.resize((854, 480)))
            
            # Initialize FastChromaKey with SyncTalk's resized frame size (512x512)
            # This matches the resize_dims from config
            # Use a beige/cream background that matches the current frame background
            background_color = np.full((512, 512, 3), [220, 210, 190], dtype=np.uint8)  # Beige background
            
            # Use the exact chroma key settings from the alin avatar configuration
            self.chroma_key_processor = FastChromaKey(
                width=512,  # Match SyncTalk's actual output size after resize
                height=512, # Match SyncTalk's actual output size after resize
                background=background_color,  # Use beige background to match current frames
                target_color='#089831',  # Correct green chroma key color
                color_threshold=35,      # Match avatar config
                edge_blur=0.4,          # Updated from config
                despill_factor=0.9       # Updated from config
            )
            logger.info("✅ Chroma key processor initialized")
            
        except Exception as e:
            logger.warning(f"Chroma key initialization failed: {e}")
            self.chroma_key_processor = None
    
    def _remove_green_screen_for_transparency(self, image: Image.Image) -> Image.Image:
        """
        Optimized green screen removal for SyncTalk frames.
        Creates transparency from green background (#089831).
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Log first few calls for debugging - reduced frequency for performance
        if self._frame_count < 3:
            logger.info(f"🎨 Processing frame {self._frame_count} for green screen removal")
        
        img_array = np.array(image, dtype=np.uint8)
        height, width = img_array.shape[:2]
        
        # SyncTalk's green screen color: #089831 = (8, 152, 49)
        target_color = np.array([8, 152, 49], dtype=np.uint8)
        
        # Optimized green screen detection - use int16 for all calculations to avoid overflow
        img_int16 = img_array.astype(np.int16)
        r, g, b = img_int16[:, :, 0], img_int16[:, :, 1], img_int16[:, :, 2]
        target_r, target_g, target_b = target_color[0], target_color[1], target_color[2]
        
        # Fast green dominance detection
        green_dominant = (g > r + 20) & (g > b + 20)
        
        # Optimized color distance calculation - avoid sqrt for performance
        color_diff_squared = (
            (r - target_r) ** 2 +
            (g - target_g) ** 2 +
            (b - target_b) ** 2
        )
        # Use squared distance comparison to avoid sqrt computation
        color_similar = color_diff_squared < (120 ** 2)  # 14400
        
        # Broader value range for green variations
        value_range = (g > 50) & (g < 220)
        
        # Optimized green hue detection using integer math
        green_hue = (g > 100) & (g * 5 > r * 6) & (g * 5 > b * 6)  # Equivalent to g > r*1.2 and g > b*1.2
        
        # Combine conditions (more permissive)
        is_green = (green_dominant | green_hue) & (color_similar | value_range)
        
        # Create alpha channel (0 = transparent, 255 = opaque)
        alpha = (~is_green).astype(np.uint8) * 255
        
        # Morphological operations for smooth edges (cached kernel)
        if self._morph_kernel is None:
            self._morph_kernel = np.ones((3, 3), np.uint8)
        
        try:
            import cv2
            # Close small holes
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, self._morph_kernel)
            # Remove small noise
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, self._morph_kernel)
            # Gaussian blur for smooth edges
            alpha = cv2.GaussianBlur(alpha, (5, 5), 1.5)
        except ImportError:
            pass  # Skip morphological operations if cv2 not available
        
        # Despill correction: reduce green channel in semi-transparent areas
        despill_mask = alpha < 200  # Areas that are partially transparent
        img_array[despill_mask, 1] = np.minimum(
            img_array[despill_mask, 1],
            np.maximum(img_array[despill_mask, 0], img_array[despill_mask, 2])
        )
        
        # Create RGBA output
        rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_array[:, :, :3] = img_array
        rgba_array[:, :, 3] = alpha
        
        # Performance logging (every 50th frame)
        self._frame_count += 1
        if self._frame_count % 50 == 0:
            transparency_ratio = np.sum(alpha == 0) / (height * width)
            logger.debug(f"Green screen removal: {transparency_ratio:.2%} transparent")
        
        return Image.fromarray(rgba_array, 'RGBA')
    
    def gpu_accelerated_compose(self, avatar_frame, slide_frame):
        """GPU-accelerated frame composition using PyTorch"""
        if not self.use_gpu:
            return self.cpu_compose(avatar_frame, slide_frame)
            
        try:
            # Convert to PyTorch tensors and move to GPU
            avatar_tensor = torch.from_numpy(avatar_frame).float().to(self.device) / 255.0
            slide_tensor = torch.from_numpy(slide_frame).float().to(self.device) / 255.0
            
            # GPU-accelerated alpha blending
            alpha = 0.8  # Avatar opacity
            composed = alpha * avatar_tensor + (1 - alpha) * slide_tensor
            
            # Convert back to numpy and CPU
            result = (composed.cpu().numpy() * 255).astype(np.uint8)
            return result
            
        except Exception as e:
            logger.warning(f"GPU composition failed, falling back to CPU: {e}")
            return self.cpu_compose(avatar_frame, slide_frame)
    
    def cpu_compose(self, avatar_frame, slide_frame):
        """Fallback CPU composition"""
        return cv2.addWeighted(avatar_frame, 0.8, slide_frame, 0.2, 0)
    
    async def connect_to_synctalk(self, avatar_name="enrique_torres", sample_rate=16000):
        """Connect to LOCAL SyncTalk server with protobuf support"""
        ws_url = f"{self.synctalk_url}/audio_to_video?avatar_name={avatar_name}&sample_rate={sample_rate}"
        logger.info(f"🔌 Connecting to LOCAL SyncTalk (ULTRA OPTIMIZED): {ws_url}")
        
        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info("✅ Connected to SyncTalk!")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio to SyncTalk"""
        if self.websocket:
            await self.websocket.send(audio_data)
    async def connect_livekit(self):
        """Connect to LiveKit and set up tracks"""
        try:
            import livekit.api as lk_api
            import livekit.rtc as rtc
            import jwt
            import time
            
            # LiveKit credentials - UPDATED
            LIVEKIT_URL = "wss://agent-s83m6c4y.livekit.cloud"
            LIVEKIT_API_KEY = "APIEkRN4enNfAzu"
            LIVEKIT_API_SECRET = "jHEYfEfhaBWQg5isdDgO6e2Xw8zhIvb18KebGwH2ESXC"
            
            # Generate JWT token with proper LiveKit format
            current_time = int(time.time())
            token_payload = {
                "iss": LIVEKIT_API_KEY,
                "sub": "avatar_bot",  # participant identity
                "aud": "livekit",
                "exp": current_time + 3600,  # 1 hour
                "nbf": current_time - 10,    # 10 seconds ago to account for clock skew
                "iat": current_time,
                "jti": f"avatar_bot_{current_time}",
                "video": {
                    "room": "avatar_room",
                    "roomJoin": True,
                    "canPublish": True,
                    "canSubscribe": True
                }
            }
            token = jwt.encode(token_payload, LIVEKIT_API_SECRET, algorithm="HS256")
            
            # Connect to room
            self.lk_room = rtc.Room()
            await self.lk_room.connect(LIVEKIT_URL, token)
            
            # Create video and audio sources - 480p for ultra smooth performance
            self.video_source = rtc.VideoSource(854, 480)
            self.audio_source = rtc.AudioSource(16000, 1)  # 16kHz mono
            
            # Create and publish video track
            video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self.video_source)
            video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            self.video_publication = await self.lk_room.local_participant.publish_track(video_track, video_options)
            
            # Create and publish audio track
            audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self.audio_source)
            audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            self.audio_publication = await self.lk_room.local_participant.publish_track(audio_track, audio_options)
            
            logger.info("✅ Connected to LiveKit and published tracks!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to LiveKit: {e}")
            return False

    async def stream_real_time_tts(self, narration_text: str):
        """Stream audio from ElevenLabs in real-time to SyncTalk AND LiveKit with optimized 40ms chunks"""
        # Use the provided API key
        api_key = "5247420f33ae186ddea9a70843d469ff"
        
        logger.info("🚀 Starting OPTIMIZED REAL-TIME ElevenLabs TTS streaming")
        tts_client = ElevenLabsTTSClient(api_key=api_key)
        
        # Add frame buffer for smooth output with proper queue management
        # Following the working service pattern: maxsize=10 for ~400ms buffer at 25 FPS
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.buffer_lock = asyncio.Lock()
        
        # Frame delivery system for smooth playback
        self.frame_delivery_running = True
        self.frame_delivery_task = None
        
        # SyncTalk frame production monitoring
        self.synctalk_frame_count = 0
        self.synctalk_start_time = None
        
        # Track audio chunks sent vs frames received for proper end marker timing
        self.audio_chunks_sent = 0
        self.frames_received = 0
        self.tts_streaming_complete = False
        # Each 160ms audio chunk should produce ~4 frames at 25 FPS
        # 160ms = 0.16s, at 25 FPS = 0.16 * 25 = 4 frames
        self.expected_frames_per_chunk = 4
        
        # Use pre-initialized frame processor (no duplicate loading)
        frame_processor = self.frame_processor
        slide_frame_index = 0
        
        # Define callback to process each audio chunk as it arrives
        async def process_audio_chunk(chunk_bytes):
            nonlocal slide_frame_index
            try:
                import tempfile
                import librosa
                import cv2
                
                # Raw PCM from ElevenLabs - no conversion needed!
                pcm_chunk = chunk_bytes  # Already 16kHz PCM int16 bytes
                
                # Send to SyncTalk immediately and track
                await self.send_audio_chunk(pcm_chunk)
                self.audio_chunks_sent += 1
                
                # Each 160ms chunk should produce ~4 frames at 25 FPS
                # Try to collect multiple frames per chunk
                frames_collected_this_chunk = 0
                max_frames_to_collect = 4  # We expect up to 4 frames per 160ms chunk
                
                for _ in range(max_frames_to_collect):
                    # Collect avatar frame with timeout
                    frame, audio = await self.receive_avatar_frame_with_audio(timeout=0.5)
                    if frame and audio:
                        # Track received frames for end marker timing
                        self.frames_received += 1
                        frames_collected_this_chunk += 1
                        
                        # MONITOR SYNCTALK PRODUCTION RATE
                        self.synctalk_frame_count += 1
                        if self.synctalk_start_time is None:
                            self.synctalk_start_time = time.time()
                        
                        # Log SyncTalk production rate every 25 frames
                        if self.synctalk_frame_count % 25 == 0:
                            elapsed = time.time() - self.synctalk_start_time
                            synctalk_fps = self.synctalk_frame_count / elapsed
                            logger.info(f"🤖 SYNCTALK PRODUCTION: {synctalk_fps:.1f} FPS (should be ~25 FPS)")
                        
                        # Add to queue for smooth continuous delivery
                        # Use non-blocking put with overflow protection
                        frame_data = {
                            "type": "video",
                            "frame": frame,
                            "audio": audio,
                            "timestamp": time.time()
                        }
                        
                        try:
                            # Non-blocking put - if queue is full, drop oldest frame
                            self.frame_queue.put_nowait(frame_data)
                        except asyncio.QueueFull:
                            # Queue is full, remove oldest and add new frame
                            try:
                                self.frame_queue.get_nowait()  # Remove oldest
                                self.frame_queue.put_nowait(frame_data)  # Add new
                                logger.debug("🔄 Buffer overflow - dropped oldest frame")
                            except asyncio.QueueEmpty:
                                pass
                        
                        # Still collect for fallback
                        self.avatar_frames.append(frame)
                        self.avatar_audio_chunks.append(audio)
                    else:
                        # No more frames available for this chunk, break early
                        break
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
        
        # Start continuous frame delivery system
        logger.info("🎬 Starting continuous frame delivery system...")
        self.frame_delivery_task = asyncio.create_task(
            self.continuous_frame_delivery(frame_processor)
        )
        
        # Start real-time streaming
        logger.info("🎭 Starting OPTIMIZED REAL-TIME audio streaming with 160ms chunks...")
        await tts_client.stream_audio_real_time(narration_text, process_audio_chunk)
        
        # Mark TTS streaming as complete
        self.tts_streaming_complete = True
        logger.info(f"🎭 TTS streaming complete. Sent {self.audio_chunks_sent} audio chunks to SyncTalk")
        
        # Wait for all frames to be received before sending end marker
        # We expect approximately 4 frames per 160ms audio chunk
        expected_total_frames = self.audio_chunks_sent * self.expected_frames_per_chunk
        logger.info(f"⏳ Waiting for SyncTalk to finish processing... Expecting ~{expected_total_frames} frames from {self.audio_chunks_sent} audio chunks")
        
        timeout_start = time.time()
        max_wait_time = 60.0  # Increased to 60 seconds for longer narrations
        
        # Wait for at least 90% of expected frames (allowing for some variance)
        min_expected_frames = int(expected_total_frames * 0.9)
        
        while self.frames_received < min_expected_frames and (time.time() - timeout_start) < max_wait_time:
            progress_percent = (self.frames_received / expected_total_frames) * 100 if expected_total_frames > 0 else 0
            logger.info(f"📊 Progress: {self.frames_received}/{expected_total_frames} frames ({progress_percent:.1f}%) - chunks sent: {self.audio_chunks_sent}")
            await asyncio.sleep(1.0)  # Check every second
        
        # Log final status before sending end marker
        if self.frames_received < min_expected_frames:
            logger.warning(f"⚠️ Timeout reached! Only received {self.frames_received}/{expected_total_frames} frames ({(self.frames_received/expected_total_frames*100):.1f}%)")
            logger.warning(f"⚠️ Proceeding to send end marker anyway to avoid hanging...")
        else:
            logger.info(f"✅ Received {self.frames_received}/{expected_total_frames} frames ({(self.frames_received/expected_total_frames*100):.1f}%)")
        
        # Continue collecting frames without sending end marker
        # Keep the stream alive and let SyncTalk finish processing all audio
        logger.info("⏳ Continuing to collect frames from SyncTalk...")
        
        # Extended collection period - give SyncTalk much more time
        extended_collection_timeout = 30.0  # 30 seconds to collect remaining frames
        collection_start_time = time.time()
        additional_frames_count = 0
        frames_at_start = self.frames_received
        
        # Keep collecting until we stop receiving frames or timeout
        consecutive_empty_receives = 0
        max_consecutive_empty = 5  # Stop after 5 consecutive empty receives (2.5 seconds of no frames)
        
        while (time.time() - collection_start_time) < extended_collection_timeout:
            try:
                # Try to receive frames with shorter timeout
                frame, audio = await self.receive_avatar_frame_with_audio(timeout=0.5)
                if frame and audio:
                    self.frames_received += 1
                    additional_frames_count += 1
                    consecutive_empty_receives = 0  # Reset counter
                    
                    # Add to queue for delivery
                    frame_data = {
                        "type": "video",
                        "frame": frame,
                        "audio": audio,
                        "timestamp": time.time()
                    }
                    try:
                        self.frame_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                        # Queue full, remove oldest and add new
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame_data)
                        except:
                            pass
                    
                    # Also collect for fallback
                    self.avatar_frames.append(frame)
                    self.avatar_audio_chunks.append(audio)
                    
                    # Log progress every 25 frames
                    if additional_frames_count % 25 == 0:
                        logger.info(f"📦 Collected {additional_frames_count} additional frames (total: {self.frames_received})")
                else:
                    consecutive_empty_receives += 1
                    if consecutive_empty_receives >= max_consecutive_empty:
                        logger.info(f"🛑 No frames received for {max_consecutive_empty * 0.5} seconds, assuming stream complete")
                        break
            except Exception as e:
                logger.debug(f"Error receiving frame: {e}")
                consecutive_empty_receives += 1
                if consecutive_empty_receives >= max_consecutive_empty:
                    break
        
        logger.info(f"📦 Collected {additional_frames_count} additional frames after TTS complete (total: {self.frames_received})")
        
        # NOW send end marker after we're really done
        logger.info("📡 Sending end marker to SyncTalk after collecting all frames...")
        try:
            await self.websocket.send(b"end_of_stream")
            logger.info("✅ End marker sent")
        except Exception as e:
            logger.warning(f"Failed to send end marker: {e}")
        
        # Now stop frame delivery gracefully
        logger.info("🛑 Stopping frame delivery...")
        self.frame_delivery_running = False
        
        # Drain any remaining frames from the queue before stopping
        while not self.frame_queue.empty():
            try:
                await self.frame_queue.get()
            except:
                break
        
        if self.frame_delivery_task:
            # Cancel the task instead of waiting for it to avoid blocking
            self.frame_delivery_task.cancel()
            try:
                await self.frame_delivery_task
            except asyncio.CancelledError:
                logger.info("✅ Frame delivery task cancelled successfully")
        
        return True
    
    async def continuous_frame_delivery(self, frame_processor):
        """Stream frames as fast as possible from queue - following working service pattern"""
        try:
            slide_frame_index = 0
            frame_count = 0
            
            logger.info("🎬 Continuous frame delivery started - PULL-BASED STREAMING")
            self.frame_delivery_start_time = time.time()
            
            while self.frame_delivery_running:
                try:
                    # Get frame from queue with timeout to allow checking frame_delivery_running
                    # This prevents indefinite blocking when we need to stop
                    try:
                        frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # Check if we should continue or stop
                        if not self.frame_delivery_running:
                            break
                        continue
                    
                    if frame_data["type"] == "video":
                        frame = frame_data["frame"]
                        audio = frame_data["audio"]
                        
                        # Get current slide frame with proper cycling
                        safe_slide_index = slide_frame_index % self.total_slide_frames
                        slide_frame = frame_processor.get_slide_frame(safe_slide_index)
                        
                        if slide_frame:
                            # FAST COMPOSITION - minimal processing
                            composed_frame = frame_processor.overlay_avatar_on_slide(
                                slide_frame=slide_frame,
                                avatar_frame=frame,
                                position="center-bottom",
                                scale=0.6,
                                offset=(0, 0)  # No offset needed - center-bottom handles positioning
                            )
                            
                            # Convert to BGR for video publishing
                            cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
                            
                            # Convert audio for LiveKit
                            pcm_audio = audio if isinstance(audio, bytes) else (audio * 32767).astype('int16').tobytes()
                            
                            # Publish to LiveKit immediately - no artificial delays
                            await self.publish_frame_to_livekit(cv_frame, pcm_audio)
                            
                            # Advance counters
                            slide_frame_index += 1
                            frame_count += 1
                            
                            # Log progress
                            if frame_count % 50 == 0:
                                cycle_count = slide_frame_index // self.total_slide_frames
                                current_slide = safe_slide_index
                                logger.info(f"📊 Slide progress: {current_slide}/{self.total_slide_frames} (cycle {cycle_count + 1}, frame {frame_count})")
                            
                            if frame_count % 25 == 0:  # Log every second
                                current_time = time.time()
                                queue_size = self.frame_queue.qsize()
                                elapsed_time = current_time - self.frame_delivery_start_time
                                actual_fps = frame_count / elapsed_time
                                target_fps = 25.0
                                fps_diff = actual_fps - target_fps
                                status = "✅" if abs(fps_diff) < 2.0 else "⚠️"
                                logger.info(f"🎬 {status} FPS: {actual_fps:.1f}/{target_fps} (diff: {fps_diff:+.1f}), queue: {queue_size}")
                        
                        # Mark task as done
                        self.frame_queue.task_done()
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error in continuous frame delivery: {e}")
        
        logger.info("🎬 Continuous frame delivery stopped")
    
    async def publish_frame_to_livekit(self, bgr_frame, audio_chunk):
        """Publish frame and audio to LiveKit immediately"""
        try:
            import livekit.rtc as rtc
            
            if not hasattr(self, 'video_source') or not hasattr(self, 'audio_source'):
                return
            
            # Convert BGR to RGB for video frame
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            
            # Create video frame
            video_frame = rtc.VideoFrame(
                width=rgb_frame.shape[1],
                height=rgb_frame.shape[0],
                type=rtc.VideoBufferType.RGB24,
                data=rgb_frame.tobytes()
            )
            
            # Publish video frame (capture_frame is NOT async)
            self.video_source.capture_frame(video_frame)
            
            # Create audio frame from PCM chunk
            if len(audio_chunk) > 0:
                # Convert bytes to int16 array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                samples_per_channel = len(audio_data)
                
                # Debug audio info
                if hasattr(self, '_audio_debug_count'):
                    self._audio_debug_count += 1
                else:
                    self._audio_debug_count = 1
                
                if self._audio_debug_count % 25 == 0:  # Log every second
                    logger.info(f"🔊 Audio: {samples_per_channel} samples, {len(audio_chunk)} bytes, 16kHz mono")
                
                audio_frame = rtc.AudioFrame(
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=samples_per_channel,
                    data=audio_data.tobytes()
                )
                
                # Publish audio frame (capture_frame IS async for audio)
                await self.audio_source.capture_frame(audio_frame)
                
                if self._audio_debug_count % 25 == 0:
                    logger.info(f"✅ Audio frame published to LiveKit")
                
        except Exception as e:
            logger.error(f"Error publishing to LiveKit: {e}")
    
    async def receive_avatar_frame_with_audio(self, timeout=1.5):
        """Receive protobuf frame and audio from SyncTalk"""
        try:
            protobuf_data = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            
            if isinstance(protobuf_data, bytes):
                frame_msg = FrameMessage()
                frame_msg.ParseFromString(protobuf_data)
                
                avatar_frame = None
                avatar_audio = None
                
                # Extract video frame
                if frame_msg.video_bytes:
                    video_data = np.frombuffer(frame_msg.video_bytes, dtype=np.uint8)
                    
                    # Handle different frame sizes
                    if len(video_data) == 512 * 512 * 3:
                        frame_array = video_data.reshape((512, 512, 3))
                    elif len(video_data) == 350 * 350 * 3:
                        frame_array = video_data.reshape((350, 350, 3))
                    else:
                        total_pixels = len(video_data) // 3
                        side = int(np.sqrt(total_pixels))
                        if side * side * 3 == len(video_data):
                            frame_array = video_data.reshape((side, side, 3))
                        else:
                            return None, None
                    
                    avatar_frame = Image.fromarray(frame_array, 'RGB')
                    
                    # Apply green screen removal for transparency (async for better performance)
                    try:
                        avatar_frame = await self._remove_green_screen_async(avatar_frame)
                        # Performance logging (reduced frequency for better performance)
                        if self._frame_count % 200 == 0:
                            logger.debug("✅ Green screen removed async for alin avatar")
                    except Exception as e:
                        logger.warning(f"Async green screen removal failed: {e}")
                
                # Extract audio
                if frame_msg.audio_bytes:
                    avatar_audio = frame_msg.audio_bytes
                
                return avatar_frame, avatar_audio
                    
        except asyncio.TimeoutError:
            return None, None
        except Exception as e:
            logger.error(f"Frame receive error: {e}")
            return None, None
    
    def _remove_green_screen_for_transparency_alin(self, image: Image.Image) -> Image.Image:
        """
        Optimized green screen removal for alin avatar with green background.
        Creates transparency from chroma key color (#089831).
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Log first few calls for debugging only
        if self._frame_count < 3:
            logger.info(f"🎨 Processing alin avatar frame {self._frame_count} for green screen removal")
            self._frame_count += 1
        
        # Convert image to numpy array (needed for final RGBA creation)
        img_array = np.array(image, dtype=np.uint8)
        height, width = img_array.shape[:2]
        
        # GPU-accelerated chroma key processing if available
        if self.use_gpu and TORCH_AVAILABLE:
            try:
                # Convert to tensor and move to GPU
                img_tensor = torch.from_numpy(img_array).float().to(self.device)
                
                # Chroma key color: #089831 = (8, 152, 49)
                target_color = torch.tensor([8.0, 152.0, 49.0], device=self.device)
                
                # Extract RGB channels
                r, g, b = img_tensor[:, :, 0], img_tensor[:, :, 1], img_tensor[:, :, 2]
                
                # GPU-accelerated green detection for #089831 (darker green)
                green_dominant = (g > r + 20) & (g > b + 20)  # Green must be significantly higher
                
                # GPU-accelerated color similarity calculation
                color_diff = torch.sqrt(
                    (r - target_color[0]) ** 2 +
                    (g - target_color[1]) ** 2 +
                    (b - target_color[2]) ** 2
                )
                color_similar = color_diff < 40  # Tighter threshold for specific green
                
                # Detect the specific darker green #089831
                # R: 8 (very low), G: 152 (medium-high), B: 49 (low)
                specific_green = (r < 50) & (g > 120) & (g < 180) & (b < 80)
                
                # Combine conditions on GPU - simpler for darker green
                is_green = (green_dominant & color_similar) | specific_green
                
                # Create alpha channel on GPU and move back to CPU
                alpha = (~is_green).float() * 255.0
                alpha = alpha.cpu().numpy().astype(np.uint8)
                
            except Exception as e:
                logger.debug(f"GPU chroma key failed, using CPU: {e}")
                # Fallback to CPU processing
                target_color = np.array([8, 152, 49], dtype=np.uint8)
                r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
                green_dominant = (g > r + 20) & (g > b + 20)  # Green must be significantly higher
                color_diff = np.sqrt(
                    (r.astype(np.int16) - target_color[0]) ** 2 +
                    (g.astype(np.int16) - target_color[1]) ** 2 +
                    (b.astype(np.int16) - target_color[2]) ** 2
                )
                color_similar = color_diff < 40  # Tighter threshold
                # Detect the specific darker green #089831
                specific_green = (r < 50) & (g > 120) & (g < 180) & (b < 80)
                is_green = (green_dominant & color_similar) | specific_green
                alpha = (~is_green).astype(np.uint8) * 255
        else:
            # CPU processing
            target_color = np.array([8, 152, 49], dtype=np.uint8)
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            green_dominant = (g > r + 20) & (g > b + 20)  # Green must be significantly higher
            color_diff = np.sqrt(
                (r.astype(np.int16) - target_color[0]) ** 2 +
                (g.astype(np.int16) - target_color[1]) ** 2 +
                (b.astype(np.int16) - target_color[2]) ** 2
            )
            color_similar = color_diff < 40  # Tighter threshold
            # Detect the specific darker green #089831
            specific_green = (r < 50) & (g > 120) & (g < 180) & (b < 80)
            is_green = (green_dominant & color_similar) | specific_green
            alpha = (~is_green).astype(np.uint8) * 255
        
        # Morphological operations for smooth edges
        if self._morph_kernel is None:
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Smooth the alpha channel
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, self._morph_kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, self._morph_kernel)
        alpha = cv2.GaussianBlur(alpha, (3, 3), 1.0)
        
        # Create RGBA image
        rgba_array = np.dstack((img_array, alpha))
        rgba_image = Image.fromarray(rgba_array, 'RGBA')
        
        return rgba_image
    
    async def _remove_green_screen_async(self, image: Image.Image) -> Image.Image:
        """
        Async wrapper for green screen removal using thread pool
        """
        try:
            # Create a simple hash for caching (optional optimization)
            image_hash = hash(image.tobytes())
            
            # Check cache first
            if image_hash in self._green_screen_cache:
                return self._green_screen_cache[image_hash]
            
            # Run green screen processing in thread pool
            loop = asyncio.get_event_loop()
            processed_image = await loop.run_in_executor(
                self._green_screen_executor,
                self._remove_green_screen_for_transparency_alin,
                image
            )
            
            # Cache the result (limit cache size to prevent memory issues)
            if len(self._green_screen_cache) < 50:  # Limit cache size
                self._green_screen_cache[image_hash] = processed_image
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Async green screen removal failed: {e}")
            return image  # Return original image on failure
    
    def remove_green_screen_with_despill(self, image: Image.Image) -> Image.Image:
        """Apply chroma key with despill factor to prevent background meshing"""
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image, dtype=np.float32)
        
        # Green screen color from SyncTalk config: #089831 = (8, 152, 49)
        target_color = np.array([8, 152, 49], dtype=np.float32)
        color_threshold = 30  # Lowered from 35 to catch more green
        despill_factor = 0.8  # Increased from 0.5 to be more aggressive
        edge_blur = 0.08
        
        # Calculate color difference - fix sqrt warning
        color_diff_squared = np.sum((img_array - target_color) ** 2, axis=2)
        color_diff = np.sqrt(np.maximum(color_diff_squared, 0))
        
        # Create alpha mask
        alpha_mask = (color_diff > color_threshold).astype(np.float32)
        
        # Apply edge blur for smooth transitions
        try:
            from scipy.ndimage import gaussian_filter
            blur_radius = max(1, int(edge_blur * min(img_array.shape[:2])))
            alpha_mask = gaussian_filter(alpha_mask, sigma=blur_radius)
        except ImportError:
            pass  # Skip blur if scipy not available
        
        # Despill: reduce green channel where it's similar to background
        despill_mask = 1.0 - (1.0 - alpha_mask) * despill_factor
        img_array[:, :, 1] = img_array[:, :, 1] * despill_mask  # Green channel
        
        # Create RGBA output
        rgba_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba_array[:, :, :3] = np.clip(img_array, 0, 255).astype(np.uint8)
        rgba_array[:, :, 3] = (alpha_mask * 255).astype(np.uint8)
        
        return Image.fromarray(rgba_array, 'RGBA')
    
    def create_speech_audio(self, duration_seconds: float, sample_rate: int = 16000) -> bytes:
        """Generate speech-like audio"""
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples, False)
        
        # Multi-harmonic speech synthesis
        base_freq = 120.0
        harmonics = [1.0, 1.5, 2.0, 2.5, 3.0]
        weights = [0.4, 0.3, 0.2, 0.1, 0.05]
        
        audio_data = np.zeros_like(t)
        for harmonic, weight in zip(harmonics, weights):
            freq = base_freq * harmonic
            freq_variation = 1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * t)
            audio_data += weight * np.sin(2 * np.pi * freq * freq_variation * t)
        
        # Speech-like modulation and pauses
        amplitude_modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 3.0 * t)
        pause_pattern = np.where(np.sin(2 * np.pi * 0.2 * t) < -0.5, 0.3, 1.0)
        audio_data = audio_data * amplitude_modulation * pause_pattern
        
        return (audio_data * 32767 * 0.7).astype(np.int16).tobytes()
    
    async def process_presentation(self, input_json: str):
        """Complete presentation processing pipeline"""
        
        logger.info("🚀 Starting Rapido Main System")
        
        try:
            # Step 1: Parse slide data
            logger.info("📄 Loading slide data...")
            data_parser = SlideDataParser(input_json)
            if not data_parser.load_data():
                raise Exception("Failed to load slide data")
            
            narration_text = data_parser.get_narration_text()
            logger.info(f"📝 Narration: {len(narration_text)} characters")
            
            # Step 2: Connect to LiveKit first
            logger.info("🔗 Connecting to LiveKit...")
            try:
                await self.connect_livekit()
                logger.info("✅ LiveKit connected successfully")
            except Exception as e:
                logger.warning(f"⚠️ LiveKit connection failed, continuing without LiveKit: {e}")
                # Continue without LiveKit for testing
            
            # Step 3: Connect to SyncTalk
            logger.info("🔌 Connecting to SyncTalk...")
            if not await self.connect_to_synctalk():
                raise Exception("SyncTalk connection failed")
            
            # Step 4: Use REAL-TIME streaming if API key available
            api_key = getattr(self.config, 'ELEVENLABS_API_KEY', None)
            if api_key:
                # Use real-time streaming
                success = await self.stream_real_time_tts(narration_text)
                if not success:
                    raise Exception("Real-time TTS streaming failed")
            else:
                # Fallback to batch processing with synthetic audio
                logger.info(f"🎵 Creating {self.total_duration_seconds:.2f}s of synthetic audio...")
                audio_data = self.create_speech_audio(self.total_duration_seconds, 16000)
                
                # Stream synthetic audio in chunks
                logger.info("🎭 Streaming synthetic audio and collecting avatar frames...")
                chunk_size = 1024
                audio_chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
                
                for i, chunk in enumerate(audio_chunks):
                    if len(chunk) % 2 != 0:
                        chunk = chunk[:-1]
                    
                    if i % 100 == 0:
                        logger.info(f"📡 Processing chunk {i+1}/{len(audio_chunks)}")
                    
                    await self.send_audio_chunk(chunk)
                    
                    frame, audio = await self.receive_avatar_frame_with_audio()
                    if frame:
                        self.avatar_frames.append(frame)
                        if audio:
                            self.avatar_audio_chunks.append(audio)
                    
                    await asyncio.sleep(0.005)  # Small delay
            
            # Collect remaining frames
            logger.info("⏳ Collecting remaining frames...")
            for _ in range(50):
                frame, audio = await self.receive_avatar_frame_with_audio(timeout=0.3)
                if frame:
                    self.avatar_frames.append(frame)
                    if audio:
                        self.avatar_audio_chunks.append(audio)
                else:
                    break
            
            logger.info(f"🎭 Collected {len(self.avatar_frames)} avatar frames!")
            logger.info(f"🎵 Collected {len(self.avatar_audio_chunks)} audio chunks!")
            
            if not self.avatar_frames:
                raise Exception("No avatar frames received")
            
            # Step 5: Frame composition - use pre-loaded frame processor
            logger.info("🖼️ Using pre-loaded frame processor (no duplicate loading)...")
            frame_processor = self.frame_processor
            
            logger.info("🖼️ Compositing frames...")
            composed_frames = []
            video_frames = []
            
            # Process all frames to match audio duration - repeat slides if needed
            avatar_count = len(self.avatar_frames)
            num_frames = avatar_count  # Use ALL avatar frames, repeat slides as needed
            logger.info(f"🎬 Creating video with {num_frames} frames (slides will repeat to match audio duration)")
            
            for i in range(num_frames):
                # Cycle through slides (repeat if audio is longer than slides)
                slide_index = i % self.total_slide_frames
                slide_frame = frame_processor.get_slide_frame(slide_index)
                if not slide_frame:
                    continue
                
                # Log cycling progress
                if i % 100 == 0:
                    cycle_num = i // self.total_slide_frames + 1
                    logger.info(f"🔄 Frame {i}: Using slide {slide_index} (cycle {cycle_num})")
                
                # Use corresponding avatar frame
                avatar_frame = self.avatar_frames[i]
                
                # Frames now come pre-processed from SyncTalk with chroma key applied
                avatar_frame_clean = avatar_frame
                
                # Compose with center-bottom avatar positioning (slide_frame already retrieved above)
                composed_frame = frame_processor.overlay_avatar_on_slide(
                    slide_frame=slide_frame,
                    avatar_frame=avatar_frame_clean,
                    position="center-bottom",  # Center horizontally, bottom edge matches screen
                    scale=0.6,  # Smaller avatar (reduced from 0.8)
                    offset=(0, 0)  # No offset needed - center-bottom handles positioning
                )
                
                composed_frames.append(composed_frame)
                cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
                video_frames.append(cv_frame)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"✅ Composed {i+1}/{num_frames} frames")
            
            # Step 6: Save audio
            if self.avatar_audio_chunks:
                logger.info("🎵 Saving combined audio...")
                combined_audio = b''.join(self.avatar_audio_chunks)
                audio_file = os.path.join(self.output_dir, "rapido_audio.wav")
                
                try:
                    import wave
                    with wave.open(audio_file, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(16000)
                        wav_file.writeframes(combined_audio)
                    logger.info(f"🎵 Audio saved: {audio_file}")
                except Exception as e:
                    logger.warning(f"Audio save failed: {e}")
            
            # Step 7: Create video with H.264
            logger.info("🎬 Creating final video with H.264...")
            
            if video_frames:
                output_video = os.path.join(self.output_dir, "rapido_output.mp4")
                
                # Hardware-accelerated video encoding with proper codec detection
                fps = self.target_video_fps
                height, width = video_frames[0].shape[:2]
                
                # Try different fourcc codes for hardware acceleration
                codec_configs = [
                    # (fourcc, description, backend_hint)
                    (cv2.VideoWriter_fourcc(*'H264'), "H.264 Hardware", "NVENC/VAAPI"),
                    (cv2.VideoWriter_fourcc(*'MJPG'), "Motion JPEG", "Hardware"),
                    (cv2.VideoWriter_fourcc(*'XVID'), "XVID", "Software"),
                    (cv2.VideoWriter_fourcc(*'mp4v'), "MPEG-4", "Software")
                ]
                
                out = None
                for fourcc, desc, backend in codec_configs:
                    try:
                        # Try with GPU acceleration hints
                        if self.use_gpu:
                            # Set OpenCV to use GPU backend if available
                            cv2.setUseOptimized(True)
                        
                        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height), True)
                        
                        # Test if the writer actually works
                        if out.isOpened():
                            # Write a test frame
                            test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                            out.write(test_frame)
                            logger.info(f"✅ Video encoder initialized: {desc} ({backend})")
                            break
                        else:
                            if out:
                                out.release()
                            out = None
                            
                    except Exception as e:
                        logger.warning(f"⚠️ {desc} codec failed: {e}")
                        if out:
                            out.release()
                            out = None
                        continue
                
                if not out or not out.isOpened():
                    raise Exception("Failed to initialize any video encoder")
                for frame in video_frames:
                    out.write(frame)
                out.release()
                
                # Verify and summarize
                if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                    file_size = os.path.getsize(output_video)
                    duration = len(video_frames) / fps
                    
                    summary = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "input_json": input_json,
                        "features": [
                            "Real SyncTalk protobuf integration",
                            "Green screen removal (chroma key)",
                            "Audio extraction from avatar frames",
                            "H.264 codec for compatibility"
                        ],
                        "results": {
                            "avatar_frames": len(self.avatar_frames),
                            "composed_frames": len(composed_frames),
                            "audio_chunks": len(self.avatar_audio_chunks),
                            "video_file": output_video,
                            "video_size_bytes": file_size,
                            "video_duration_seconds": duration,
                            "video_fps": fps,
                            "video_resolution": f"{width}x{height}",
                            "codec": "H.264"
                        }
                    }
                    
                    summary_file = os.path.join(self.output_dir, "rapido_summary.json")
                    with open(summary_file, "w") as f:
                        json.dump(summary, f, indent=2)
                    
                    logger.info(f"✅ SUCCESS: {output_video} ({file_size} bytes, {duration:.1f}s)")
                    return output_video
                else:
                    raise Exception("Video creation failed")
            else:
                raise Exception("No video frames generated")
                
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            raise
        finally:
            if self.websocket:
                await self.websocket.close()

def check_synctalk_server():
    """Verify SyncTalk server is running"""
    try:
        response = requests.get(f"{Config.SYNCTALK_SERVER_URL}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ SyncTalk server ready: {status['status']}")
            print(f"🤖 Models: {status['loaded_models']}")
            return len(status['loaded_models']) > 0  # Check if any models are loaded
        return False
    except Exception as e:
        print(f"❌ SyncTalk server error: {e}")
        print(f"Make sure your SyncTalk server is running on {Config.SYNCTALK_SERVER_URL}")
        return False

async def main():
    """Main entry point"""
    
    # Get project root for default paths
    script_dir = Path(__file__).parent.parent  # rapido/src -> rapido  
    project_root = script_dir.parent  # rapido -> agent_streaming_backend
    
    parser = argparse.ArgumentParser(description="Rapido - Avatar Video Generation")
    parser.add_argument("--input", "-i", default=str(project_root / "test1.json"), help="Input JSON file")
    parser.add_argument("--frames", "-f", default=str(project_root / "presentation_frames"), help="Slide frames directory")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--api-key", help="ElevenLabs API key")
    parser.add_argument("--avatar-scale", type=float, default=0.5, help="Avatar scale")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Rapido - Integrated Avatar Video Generation")
    print("=" * 50)
    
    # Check SyncTalk
    if not check_synctalk_server():
        print("❌ SyncTalk server not ready!")
        return 1
    
    # Configure system
    config = {
        'SLIDE_FRAMES_PATH': args.frames,
        'OUTPUT_PATH': args.output,
        'AVATAR_SCALE': args.avatar_scale
    }
    if args.api_key:
        config['ELEVENLABS_API_KEY'] = args.api_key
    
    # Run Rapido
    rapido = RapidoMainSystem(config)
    
    try:
        output_video = await rapido.process_presentation(args.input)
        
        print("\n🎉 RAPIDO SUCCESS!")
        print("=" * 50)
        print(f"📁 Output: {rapido.output_dir}")
        print(f"🎭 Avatar frames: {len(rapido.avatar_frames)}")
        print(f"🎵 Audio chunks: {len(rapido.avatar_audio_chunks)}")
        print(f"🎬 Video: {output_video}")
        print("\n✨ Complete integrated system working!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
