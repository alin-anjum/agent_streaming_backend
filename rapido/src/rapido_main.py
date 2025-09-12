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
import aiohttp
from urllib.parse import urlencode
import cv2
import numpy as np
from pathlib import Path
import argparse
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import tempfile
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

# Import optimized modules (will check after logger is set up)
OPTIMIZATIONS_AVAILABLE = False

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

# Now try to import optimized modules
try:
    from livekit_optimized import LiveKitOptimizedPublisher, QualityLevel
    from frame_pacer import FramePacer, AdaptiveFramePacer
    from audio_optimizer import AudioOptimizer, BufferMode, AudioStreamSynchronizer
    from audio_smoother import AudioSmoother
    OPTIMIZATIONS_AVAILABLE = True
    logger.info("✅ Optimization modules loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Optimized modules not available: {e}")
    OPTIMIZATIONS_AVAILABLE = False

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
        self.synctalk_url = getattr(self.config, 'SYNCTALK_WEBSOCKET_URL', 'ws://35.172.212.10:8000/ws')
        self.websocket = None
        self.aiohttp_session = None
        self.audio_send_queue = None
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
            
        # Dynamic frame capture configuration
        self.use_dynamic_capture = getattr(self.config, 'USE_DYNAMIC_CAPTURE', False)
        self.capture_url = getattr(self.config, 'CAPTURE_URL', 'https://test.creatium.com/presentation')

        # Real-time slide frame streaming (tab-capture) additions - EXACT SYNCTALK PATTERN
        # Mirror the exact SyncTalk pattern: asyncio.Queue with producer task
        # that watches the capture directory and feeds frames to a queue
        # for non-blocking consumption by the compositor.
        self.slide_frame_queue: Optional[asyncio.Queue] = None  # Created after capture starts
        self._slide_producer_task: Optional[asyncio.Task] = None
        self.slide_frames_cache = {}  # Cache loaded slide frames by index
        self.slide_frame_count = 0    # Current number of available slide frames
        self.max_cached_frames = 1000  # Limit cache size to prevent memory issues
        self._cache_lock = asyncio.Lock()  # Thread safety for cache operations
        self.dynamic_frame_processor = None
        self.dynamic_frames_dir = None  # Set after capture starts

        # Initialize frame processor based on capture mode
        if self.use_dynamic_capture:
            # For dynamic capture, create minimal processor for overlay operations only
            logger.info("🎬 Dynamic capture mode - creating overlay-only processor")
            # Create a dummy empty directory to avoid loading any static frames
            import tempfile
            temp_dir = tempfile.mkdtemp()
            self.frame_processor = FrameOverlayEngine(temp_dir, output_size=(854, 480), dynamic_mode=True)
            self.total_slide_frames = 0
            self.total_duration_seconds = 0
        else:
            # For static frames, load the presentation frames
            logger.info("📁 Static frame mode - loading presentation frames")
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
        
        # Initialize optimization components
        if OPTIMIZATIONS_AVAILABLE:
            # Optimized LiveKit publisher
            self.livekit_publisher = None  # Will be initialized in connect_livekit
            
            # Frame pacer for consistent delivery
            self.frame_pacer = None  # Will be initialized in connect_livekit
            
            # Audio optimizer
            self.audio_optimizer = None  # Will be initialized in connect_livekit
            
            # A/V synchronizer
            self.av_synchronizer = None  # Will be initialized in connect_livekit
            
            # Audio smoother for SyncTalk chunks (fixes crackling)
            # TEMPORARILY DISABLED to test if it's causing issues
            self.audio_smoother = None  # AudioSmoother(sample_rate=16000, crossfade_ms=2.0)
            logger.info("⚠️ Audio smoother DISABLED for testing")
            
            logger.info("✨ Optimization modules available and ready")
        else:
            self.audio_smoother = None
            logger.warning("⚠️ Running without optimization modules")
        
        # Initialize buffering strategy configuration  
        # REALITY-BASED: SyncTalk produces ~23.1 FPS, Frame pacer delivers ~17.7 FPS
        self.buffer_config = {
            'min_buffer_threshold': 75,    # Smaller buffer - faster startup
            'min_buffer_level': 5,         # Very low threshold - we'll duplicate frames instead of pausing
            'refill_target': 10,           # Not used - we don't refill during playback
            'target_fps': 23.0,            # REALISTIC: Match SyncTalk's actual output (not 25!)
            'max_timing_drift': 0.08,      # Max timing drift before reset (2 frame intervals)
            'duplicate_when_empty': True,  # Duplicate last frame instead of pausing - SMOOTH PLAYBACK!
            'max_composition_time_ms': 30, # Max acceptable composition time
            'max_publish_time_ms': 40,     # Max acceptable publish time
            'bypass_frame_pacer': False    # Fixed: Configure frame pacer properly instead of bypassing
        }
        
        logger.info(f"📊 Rapido initialized - Duration: {self.total_duration_seconds:.2f}s")
        logger.info(f"📦 Buffer config: threshold={self.buffer_config['min_buffer_threshold']}, level={self.buffer_config['min_buffer_level']}, fps={self.buffer_config['target_fps']}")
        logger.info(f"🎯 REALITY-BASED optimization: SyncTalk ~23 FPS, Frame pacer properly configured for 23 FPS")
    
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
                target_color='#bcfeb6',  # Match SyncTalk server green chroma key color
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
        Creates transparency from green background (#bcfeb6).
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Log first few calls for debugging - reduced frequency for performance
        if self._frame_count < 3:
            logger.info(f"🎨 Processing frame {self._frame_count} for green screen removal")
        
        img_array = np.array(image, dtype=np.uint8)
        height, width = img_array.shape[:2]
        
        # SyncTalk server's green screen color: #bcfeb6 = (188, 254, 182)
        target_color = np.array([188, 254, 182], dtype=np.uint8)
        
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
        """Connect to SyncTalk server using DECOUPLED architecture for maximum performance"""
        # Build optimized WebSocket URL with correct endpoint
        # Replace /ws with /ws/audio_to_video to get the correct endpoint
        ws_url = self.synctalk_url.replace('/ws', '/ws/audio_to_video')
        params = {
            "avatar_name": avatar_name,
            "sample_rate": str(sample_rate)
        }
        full_url = f"{ws_url}?{urlencode(params)}"
        
        logger.info(f"🔌 Connecting to SyncTalk (DECOUPLED ARCHITECTURE): {full_url}")
        
        try:
            # Use aiohttp for better WebSocket performance
            self.aiohttp_session = aiohttp.ClientSession()
            self.websocket = await self.aiohttp_session.ws_connect(
                full_url,
                max_msg_size=10 * 1024 * 1024,  # 10MB max message size for performance
                heartbeat=30  # Keep connection alive
            )
            
            # Initialize audio queue for decoupled sending
            self.audio_send_queue = asyncio.Queue(maxsize=1000)  # Much larger queue
            
            # Start separate audio processor for non-blocking sends
            self.audio_processor_task = asyncio.create_task(self._audio_processor_loop())
            
            logger.info("✅ Connected to SyncTalk with DECOUPLED architecture!")
            logger.info("🚀 Audio processor started - audio sending now non-blocking!")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            if self.aiohttp_session:
                await self.aiohttp_session.close()
            return False
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Queue audio for non-blocking send to SyncTalk"""
        if self.audio_send_queue:
            try:
                # Queue audio for sending - with better error handling
                await self.audio_send_queue.put({
                    "type": "audio", 
                    "data": audio_data
                })
            except Exception as e:
                logger.error(f"Failed to queue audio: {e}")
                # Fallback to direct send if queue fails
                if self.websocket:
                    try:
                        await self.websocket.send_bytes(audio_data)
                    except Exception as send_error:
                        logger.error(f"Direct audio send also failed: {send_error}")
    
    async def _audio_processor_loop(self):
        """Separate audio processor loop - sends audio as fast as possible without blocking"""
        audio_chunks_sent = 0
        last_log_time = time.time()
        
        logger.info("🎵 Audio processor loop started - decoupled from frame processing")
        
        while True:
            try:
                # Get audio data from queue with timeout
                try:
                    audio_item = await asyncio.wait_for(self.audio_send_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if we should continue
                    if hasattr(self, 'frame_delivery_running') and not self.frame_delivery_running:
                        break
                    continue
                
                if audio_item["type"] == "audio":
                    # Send audio immediately to SyncTalk using aiohttp WebSocket
                    await self.websocket.send_bytes(audio_item["data"])
                    audio_chunks_sent += 1
                    
                    # Debug log for first few sends
                    if audio_chunks_sent <= 3:
                        logger.info(f"🎵 Audio chunk {audio_chunks_sent} sent to SyncTalk ({len(audio_item['data'])} bytes)")
                    
                    # Log audio sending rate periodically
                    current_time = time.time()
                    if current_time - last_log_time > 3.0:
                        elapsed = current_time - last_log_time
                        audio_rate = audio_chunks_sent / elapsed
                        queue_size = self.audio_send_queue.qsize()
                        logger.info(f"🎵 AUDIO PROCESSOR: {audio_rate:.1f} chunks/sec sent | Queue: {queue_size}")
                        audio_chunks_sent = 0
                        last_log_time = current_time
                        
                elif audio_item["type"] == "end_stream":
                    # Send end of stream marker
                    await self.websocket.send_bytes(b"end_of_stream")
                    logger.info("📡 End of stream marker sent via audio processor")
                    break
                    
            except Exception as e:
                logger.error(f"CRITICAL: Audio processor loop error: {e}")
                logger.error(f"Audio item type: {audio_item.get('type', 'unknown') if 'audio_item' in locals() else 'no item'}")
                logger.error(f"WebSocket state: {self.websocket.closed if self.websocket else 'no websocket'}")
                await asyncio.sleep(0.1)
                continue
                
        logger.info("🎵 Audio processor loop stopped")
    async def connect_livekit(self):
        """Connect to LiveKit with optimized settings"""
        # Check if optimizations are available
        if OPTIMIZATIONS_AVAILABLE:
            logger.info("Attempting optimized LiveKit connection...")
            success = await self._connect_livekit_optimized()
            if not success:
                logger.warning("Optimized connection failed, falling back to legacy")
                return await self._connect_livekit_legacy()
            return success
        else:
            logger.info("Optimizations not available, using legacy connection")
            return await self._connect_livekit_legacy()
    
    async def _connect_livekit_optimized(self):
        """Connect to LiveKit with all optimizations enabled"""
        try:
            # First connect using legacy method to ensure compatibility
            import livekit.api as lk_api
            import livekit.rtc as rtc
            import jwt
            import time
            
            # LiveKit credentials
            LIVEKIT_URL = "wss://rapido-pme0lo9d.livekit.cloud"
            LIVEKIT_API_KEY = "APImuXsSp8NH5jY"
            LIVEKIT_API_SECRET = "6k9Swe5O6NxeI0WvVTCTrs2k1Ec25byeM4NlnTCKn5GB"
            
            # Generate JWT token
            current_time = int(time.time())
            token_payload = {
                "iss": LIVEKIT_API_KEY,
                "sub": "avatar_bot",
                "aud": "livekit",
                "exp": current_time + 3600,
                "nbf": current_time - 10,
                "iat": current_time,
                "jti": f"avatar_bot_{current_time}",
                "video": {
                    "room": room_name,  # FIXED: Use actual room name from config
                    "roomJoin": True,
                    "canPublish": True,
                    "canSubscribe": True
                }
            }
            token = jwt.encode(token_payload, LIVEKIT_API_SECRET, algorithm="HS256")
            
            # Connect to room using standard method  
            self.lk_room = rtc.Room()
            await self.lk_room.connect(LIVEKIT_URL, token)
            logger.info(f"✅ Connected to LiveKit room: {room_name}")
            
            # Log room connection details
            logger.info(f"🎬 Publishing video track to room: {room_name}")
            logger.info(f"🎵 Publishing audio track to room: {room_name}")
            
            # Create sources
            self.video_source = rtc.VideoSource(854, 480)
            self.audio_source = rtc.AudioSource(16000, 1)
            
            # Publish tracks
            video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self.video_source)
            video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            self.video_publication = await self.lk_room.local_participant.publish_track(video_track, video_options)
            
            audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self.audio_source)
            audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            self.audio_publication = await self.lk_room.local_participant.publish_track(audio_track, audio_options)
            
            logger.info("✅ Connected to LiveKit (legacy method)")
            
            # Now initialize optimization components on top of the basic connection
            # Initialize frame pacer for consistent delivery - CONFIGURED FOR 23 FPS REALITY
            self.frame_pacer = AdaptiveFramePacer(
                target_fps=23.0,        # FIXED: Match SyncTalk's actual output, not wishful 25 FPS
                min_buffer_size=5,      # Smaller buffer for lower latency
                max_buffer_size=25      # Optimized for 23 FPS (1 second buffer)
            )
            
            # Initialize audio optimizer with HIGH_QUALITY mode for stability
            self.audio_optimizer = AudioOptimizer(
                sample_rate=16000,
                channels=1,
                initial_mode=BufferMode.HIGH_QUALITY  # Start with large buffer
            )
            
            # Initialize A/V synchronizer
            self.av_synchronizer = AudioStreamSynchronizer(
                video_fps=25.0,
                audio_sample_rate=16000
            )
            
            # Create a simple wrapper for frame delivery
            async def deliver_frame_wrapper(frame):
                """Deliver frame to LiveKit using legacy video source"""
                if hasattr(self, 'video_source') and self.video_source:
                    try:
                        # Ensure frame is RGB numpy array
                        if isinstance(frame, Image.Image):
                            frame = np.array(frame)
                        
                        # Create video frame
                        video_frame = rtc.VideoFrame(
                            width=frame.shape[1],
                            height=frame.shape[0],
                            type=rtc.VideoBufferType.RGB24,
                            data=frame.tobytes()
                        )
                        
                        # Publish video frame
                        self.video_source.capture_frame(video_frame)
                    except Exception as e:
                        logger.error(f"Error delivering frame: {e}")
            
            # Start frame pacer with the wrapper
            self.frame_pacer_task = asyncio.create_task(self.frame_pacer.start(deliver_frame_wrapper))
            
            # Start audio processing loop (modified for direct audio source)
            async def audio_loop_wrapper():
                """Process audio through optimizer and send to LiveKit"""
                audio_sent_count = 0
                last_log_time = time.time()
                
                while True:
                    try:
                        if self.audio_optimizer:
                            # Try to get audio frame - will return None if buffer underrun
                            audio_frame_bytes = await self.audio_optimizer.get_audio_frame(duration_ms=40)
                            
                            if audio_frame_bytes and hasattr(self, 'audio_source'):
                                # Convert to audio frame
                                audio_data = np.frombuffer(audio_frame_bytes, dtype=np.int16)
                                audio_frame = rtc.AudioFrame(
                                    sample_rate=16000,
                                    num_channels=1,
                                    samples_per_channel=len(audio_data),
                                    data=audio_data.tobytes()
                                )
                                await self.audio_source.capture_frame(audio_frame)
                                audio_sent_count += 1
                            else:
                                # No audio available - generate silence to maintain stream
                                silence_samples = int(16000 * 0.04)  # 40ms of silence
                                silence_data = np.zeros(silence_samples, dtype=np.int16)
                                audio_frame = rtc.AudioFrame(
                                    sample_rate=16000,
                                    num_channels=1,
                                    samples_per_channel=silence_samples,
                                    data=silence_data.tobytes()
                                )
                                await self.audio_source.capture_frame(audio_frame)
                            
                            # Log audio rate periodically
                            current_time = time.time()
                            if current_time - last_log_time > 5.0:
                                audio_rate = audio_sent_count / (current_time - last_log_time)
                                logger.debug(f"Audio output rate: {audio_rate:.1f} chunks/sec (target: 25)")
                                audio_sent_count = 0
                                last_log_time = current_time
                        
                        await asyncio.sleep(0.04)  # 40ms = 25 fps
                    except Exception as e:
                        logger.error(f"Error in audio loop: {e}")
                        await asyncio.sleep(0.04)
            
            self.audio_loop_task = asyncio.create_task(audio_loop_wrapper())
            
            # Give tasks a moment to start
            await asyncio.sleep(0.1)
            
            logger.info("✅ Connected to LiveKit with optimizations enabled!")
            logger.info(f"Frame pacer running: {self.frame_pacer.is_running}")
            logger.info(f"Audio optimizer initialized: {self.audio_optimizer is not None}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to LiveKit (optimized): {e}")
            return False
    
    async def _connect_livekit_legacy(self):
        """Legacy LiveKit connection without optimizations"""
        try:
            import livekit.api as lk_api
            import livekit.rtc as rtc
            import jwt
            import time
            
            # LiveKit credentials
            LIVEKIT_URL = "wss://rapido-pme0lo9d.livekit.cloud"
            LIVEKIT_API_KEY = "APImuXsSp8NH5jY"
            LIVEKIT_API_SECRET = "6k9Swe5O6NxeI0WvVTCTrs2k1Ec25byeM4NlnTCKn5GB"
            
            # Get room name from config
            room_name = getattr(self.config, 'LIVEKIT_ROOM', 'avatar-room2')
            logger.info(f"🏠 Legacy LiveKit connecting to room: {room_name}")
            
            # Generate JWT token
            current_time = int(time.time())
            token_payload = {
                "iss": LIVEKIT_API_KEY,
                "sub": "avatar_bot",
                "aud": "livekit",
                "exp": current_time + 3600,
                "nbf": current_time - 10,
                "iat": current_time,
                "jti": f"avatar_bot_{current_time}",
                "video": {
                    "room": room_name,  # FIXED: Use actual room name from config
                    "roomJoin": True,
                    "canPublish": True,
                    "canSubscribe": True
                }
            }
            token = jwt.encode(token_payload, LIVEKIT_API_SECRET, algorithm="HS256")
            
            # Connect to room
            self.lk_room = rtc.Room()
            await self.lk_room.connect(LIVEKIT_URL, token)
            logger.info(f"✅ Connected to LiveKit room: {room_name}")
            
            # Create sources
            self.video_source = rtc.VideoSource(854, 480)
            self.audio_source = rtc.AudioSource(16000, 1)
            
            # Publish tracks
            video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self.video_source)
            video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            self.video_publication = await self.lk_room.local_participant.publish_track(video_track, video_options)
            
            audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self.audio_source)
            audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            self.audio_publication = await self.lk_room.local_participant.publish_track(audio_track, audio_options)
            
            logger.info("✅ Connected to LiveKit (legacy mode)!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to LiveKit (legacy): {e}")
            return False
    
    # Removed unused methods - frame delivery and audio processing are now handled inline in _connect_livekit_optimized

    async def stream_real_time_tts(self, narration_text: str):
        """Stream audio from ElevenLabs in real-time to SyncTalk AND LiveKit with optimized 40ms chunks"""
        # Use the provided API key
        api_key = "5247420f33ae186ddea9a70843d469ff"
        
        logger.info("🚀 Starting OPTIMIZED REAL-TIME ElevenLabs TTS streaming")
        tts_client = ElevenLabsTTSClient(api_key=api_key)
        
        # Single queue buffering system optimized for 23fps reality
        # Fast collection from SyncTalk, realistic timing control for output
        self.intake_queue = asyncio.Queue(maxsize=200)  # Optimized buffer size for 23fps
        self.buffer_lock = asyncio.Lock()
        self.last_frame_data = None  # Keep last frame for duplication when buffer empty
        
        # Frame delivery system for smooth playback
        self.frame_delivery_running = True
        self.frame_delivery_task = None
        
        # SyncTalk frame production monitoring - SEPARATE FROM COMPOSITION
        self.synctalk_frame_count = 0
        self.synctalk_start_time = None
        self.synctalk_last_log_time = None
        self.synctalk_last_frame_count = 0

        # Composition/delivery monitoring - SEPARATE FROM SYNCTALK
        self.composition_frame_count = 0
        self.composition_start_time = None
        
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
                
                # DECOUPLED: Queue audio immediately (non-blocking!)
                await self.send_audio_chunk(pcm_chunk)
                self.audio_chunks_sent += 1
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
        
        # Start DECOUPLED frame collection system
        logger.info("🎬 Starting DECOUPLED frame collection and controlled output...")
        
        # Start independent frame collector (no blocking audio!)
        self.frame_collector_task = asyncio.create_task(self._frame_collector_loop())
        
        # Start controlled output delivery
        self.frame_delivery_task = asyncio.create_task(
            self.continuous_frame_delivery(frame_processor)
        )
        
        # Start slide frame producer for dynamic capture - EXACT SYNCTALK PATTERN
        if self.use_dynamic_capture:
            logger.info("🎬 Starting slide frame producer (background)...")
            self._slide_producer_task = asyncio.create_task(self._produce_slide_frames())
        
        # Start real-time streaming
        logger.info("🎭 Starting OPTIMIZED REAL-TIME audio streaming with 160ms chunks...")
        if self.audio_optimizer:
            initial_req = getattr(self.audio_optimizer, 'initial_buffer_requirement', 360)
            target = getattr(self.audio_optimizer, 'target_buffer_ms', 320)
            logger.info(f"📦 Audio buffering: {initial_req:.0f}ms initial fill, {target*.5:.0f}ms minimum playback")
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
                        self.intake_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                        # Intake full, remove oldest and add new
                        try:
                            self.intake_queue.get_nowait()
                            self.intake_queue.put_nowait(frame_data)
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
        
        # NOW send end marker after we're really done (via audio processor queue)
        logger.info("📡 Sending end marker to SyncTalk after collecting all frames...")
        try:
            await self.audio_send_queue.put({"type": "end_stream"})
            logger.info("✅ End marker queued for audio processor")
        except Exception as e:
            logger.warning(f"Failed to queue end marker: {e}")
        
        # Now stop frame delivery gracefully
        logger.info("🛑 Stopping frame delivery system...")
        self.frame_delivery_running = False
        
        # Drain any remaining frames from intake queue before stopping            
        while not self.intake_queue.empty():
            try:
                await self.intake_queue.get() 
            except:
                break
        
        # Cancel all tasks
        tasks_to_cancel = []
        if hasattr(self, 'frame_collector_task') and self.frame_collector_task:
            tasks_to_cancel.append(('Frame collector', self.frame_collector_task))
        if hasattr(self, 'audio_processor_task') and self.audio_processor_task:
            tasks_to_cancel.append(('Audio processor', self.audio_processor_task))
        if self.frame_delivery_task:
            tasks_to_cancel.append(('Frame delivery', self.frame_delivery_task))
        
        for task_name, task in tasks_to_cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"✅ {task_name} task cancelled successfully")
        
        # Close aiohttp session
        if self.aiohttp_session:
            await self.aiohttp_session.close()
            logger.info("✅ aiohttp session closed")
        
        return True
    
    async def continuous_frame_delivery(self, frame_processor):
        """Stream frames with buffering strategy to maintain steady 25fps output"""
        try:
            slide_frame_index = 0
            frame_count = 0
            
            # Buffering configuration - use configurable values
            min_buffer_threshold = self.buffer_config['min_buffer_threshold']
            min_buffer_level = self.buffer_config['min_buffer_level']
            refill_target = self.buffer_config['refill_target']
            target_fps = self.buffer_config['target_fps']
            max_timing_drift = self.buffer_config['max_timing_drift']
            frame_interval = 1.0 / target_fps  # ~0.043 seconds = 43.5ms per frame at 23fps
            
            logger.info(f"🎬 Output delivery started - waiting for {min_buffer_threshold} frame intake buffer")
            logger.info(f"⏱️ Performance targets: Composition <{self.buffer_config['max_composition_time_ms']}ms | Publish <{self.buffer_config['max_publish_time_ms']}ms")
            logger.info(f"🎯 REALITY-BASED: Target {target_fps} FPS to match SyncTalk actual performance")
            self.frame_delivery_start_time = time.time()
            
            # Phase 1: Wait for buffer to fill above threshold
            buffer_filled = False
            buffer_wait_start = time.time()
            
            while self.frame_delivery_running and not buffer_filled:
                intake_size = self.intake_queue.qsize()
                
                if intake_size >= min_buffer_threshold:
                    buffer_filled = True
                    buffer_wait_time = time.time() - buffer_wait_start
                    logger.info(f"🎬 ✅ Intake buffer filled! {intake_size} frames ready after {buffer_wait_time:.1f}s - starting steady {target_fps}fps output")
                    break
                
                # Log buffer filling progress every 2 seconds
                if int(time.time() - buffer_wait_start) % 2 == 0:
                    elapsed = time.time() - buffer_wait_start
                    if elapsed > 1:  # Only log after first second
                        logger.info(f"🎬 📦 Intake buffer: {intake_size}/{min_buffer_threshold} frames ({elapsed:.1f}s elapsed)")
                
                # Wait a bit before checking again
                await asyncio.sleep(0.5)
            
            if not buffer_filled:
                logger.warning("🎬 ⚠️ Frame delivery stopped before buffer was filled")
                return
            
            # Phase 2: Steady 25fps output with buffer monitoring
            # Use high-precision timing for exactly 25fps
            next_frame_time = time.time()
            frames_output = 0
            last_timing_log = time.time()
            
            # Reset frame delivery start time AFTER buffer fills for accurate FPS measurement
            self.frame_delivery_start_time = time.time()
            frame_count = 0  # Reset counter for accurate sustained FPS
            
            # Detailed output rate tracking
            output_rate_start = time.time()
            output_rate_count = 0
            duplicated_count = 0
            real_frame_count = 0
            
            while self.frame_delivery_running:
                try:
                    # PRECISE 25fps timing - wait until exactly the right time
                    current_time = time.time()
                    if current_time < next_frame_time:
                        sleep_time = next_frame_time - current_time
                        # Use precise sleep for timing accuracy
                        if sleep_time > 0.001:  # Only sleep if > 1ms
                            await asyncio.sleep(sleep_time)
                    
                    # Get frame DIRECTLY from intake (single timing control)
                    intake_size = self.intake_queue.qsize()
                    is_duplicated_frame = False
                    
                    try:
                        frame_data = self.intake_queue.get_nowait()
                        self.last_frame_data = frame_data  # Save for potential duplication
                        real_frame_count += 1
                    except asyncio.QueueEmpty:
                        # Intake empty - duplicate last frame for smooth playback (NO PAUSES!)
                        if self.last_frame_data is not None:
                            frame_data = self.last_frame_data.copy()  # Duplicate last frame
                            is_duplicated_frame = True
                            duplicated_count += 1
                            if intake_size == 0:  # Only log first time intake goes empty
                                logger.info(f"🎬 🔄 Intake empty - duplicating last frame for smooth playback")
                        else:
                            # No previous frame available, skip this cycle
                            logger.warning("🎬 ⚠️ No frames available and no previous frame to duplicate")
                            next_frame_time += frame_interval
                            continue
                    
                    if frame_data["type"] == "video":
                        frame = frame_data["frame"]
                        audio = frame_data["audio"]
                        
                        # Get slide frame - FAST CACHED ACCESS for dynamic capture
                        if self.use_dynamic_capture and self.slide_frame_count > 0:
                            # Use fast cached access - no disk I/O blocking
                            safe_slide_index = slide_frame_index % self.slide_frame_count
                            # TODO: change this to use queue system
                            slide_frame = self.get_cached_slide_frame(safe_slide_index)
                            self.total_slide_frames = self.slide_frame_count  # Update for progress logging
                            if slide_frame_index % 25 == 0:  # Debug every second
                                logger.info(f"🎬 DYNAMIC: Using cached frame {safe_slide_index}/{self.slide_frame_count}")
                        else:
                            # Fallback to frame processor for static frames
                            current_frame_count = frame_processor.get_frame_count()
                            if current_frame_count > 0:
                                safe_slide_index = slide_frame_index % current_frame_count
                                self.total_slide_frames = current_frame_count
                            else:
                                safe_slide_index = 0
                            slide_frame = frame_processor.get_slide_frame(safe_slide_index)
                            if slide_frame_index % 25 == 0:  # Debug every second
                                logger.info(f"🎬 STATIC: Using frame processor {safe_slide_index}/{current_frame_count}, dynamic_count={self.slide_frame_count}")
                        
                        if slide_frame:
                            # FAST COMPOSITION with performance tracking
                            composition_start = time.time()
                            composed_frame = frame_processor.overlay_avatar_on_slide(
                                slide_frame=slide_frame,
                                avatar_frame=frame,
                                position="center-bottom",
                                scale=0.6,
                                offset=(0, 0)  # No offset needed - center-bottom handles positioning
                            )
                            composition_duration = (time.time() - composition_start) * 1000
                            
                            # Convert to BGR for video publishing
                            conversion_start = time.time()
                            cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
                            conversion_duration = (time.time() - conversion_start) * 1000
                            
                            # Convert audio for LiveKit
                            pcm_audio = audio if isinstance(audio, bytes) else (audio * 32767).astype('int16').tobytes()
                            
                            # Track slow operations
                            max_composition_ms = self.buffer_config['max_composition_time_ms']
                            if composition_duration > max_composition_ms:
                                logger.warning(f"🐌 Slow composition: {composition_duration:.1f}ms (target: <{max_composition_ms}ms)")
                            if conversion_duration > 10:
                                logger.warning(f"🐌 Slow BGR conversion: {conversion_duration:.1f}ms")
                            
                            # Publish to LiveKit at precise timing with performance measurement
                            publish_start = time.time()
                            await self.publish_frame_to_livekit(cv_frame, pcm_audio)
                            publish_duration = (time.time() - publish_start) * 1000  # Convert to milliseconds
                            
                            # Track publishing performance
                            if not hasattr(self, '_publish_times'):
                                self._publish_times = []
                            self._publish_times.append(publish_duration)
                            
                            # Log slow publishing
                            max_publish_ms = self.buffer_config['max_publish_time_ms']
                            if publish_duration > max_publish_ms:
                                logger.warning(f"🐌 Slow publish: {publish_duration:.1f}ms (target: <{max_publish_ms}ms for 25fps)")
                            
                            # Advance counters
                            slide_frame_index += 1
                            frame_count += 1
                            frames_output += 1
                            output_rate_count += 1
                            
                            # Set next frame time for precise 25fps - CRITICAL timing
                            next_frame_time += frame_interval
                            
                            # Handle timing drift - if we're more than max_timing_drift behind, reset
                            current_time = time.time()
                            if next_frame_time < current_time - max_timing_drift:
                                drift_amount = current_time - next_frame_time
                                next_frame_time = current_time + frame_interval
                                logger.debug(f"🎬 🔄 Reset frame timing due to {drift_amount*1000:.1f}ms drift")
                            
                            # Log detailed output rate every 3 seconds
                            if current_time - last_timing_log > 3.0:
                                time_elapsed = current_time - last_timing_log
                                actual_output_fps = frames_output / time_elapsed
                                timing_accuracy = (actual_output_fps / target_fps) * 100
                                status_icon = "✅" if abs(actual_output_fps - target_fps) < 0.5 else "⚠️"
                                
                                # Calculate output rate metrics
                                total_output_elapsed = current_time - output_rate_start
                                total_output_fps = output_rate_count / total_output_elapsed if total_output_elapsed > 0 else 0
                                real_fps = real_frame_count / total_output_elapsed if total_output_elapsed > 0 else 0
                                duplicate_fps = duplicated_count / total_output_elapsed if total_output_elapsed > 0 else 0
                                duplicate_ratio = (duplicated_count / output_rate_count * 100) if output_rate_count > 0 else 0
                                
                                # Calculate pipeline performance
                                if hasattr(self, '_publish_times') and self._publish_times:
                                    avg_publish_time = sum(self._publish_times) / len(self._publish_times)
                                    max_publish_time = max(self._publish_times)
                                    slow_publishes = len([t for t in self._publish_times if t > 40])
                                    slow_ratio = (slow_publishes / len(self._publish_times)) * 100
                                    self._publish_times = []  # Reset for next period
                                else:
                                    avg_publish_time = 0
                                    max_publish_time = 0
                                    slow_ratio = 0
                                
                                logger.info(f"🎬 {status_icon} PRECISE OUTPUT: {actual_output_fps:.2f}/{target_fps} FPS ({timing_accuracy:.1f}% accuracy)")
                                logger.info(f"📊 OUTPUT BREAKDOWN: Total: {total_output_fps:.1f} FPS | Real: {real_fps:.1f} | Duplicated: {duplicate_fps:.1f} ({duplicate_ratio:.1f}%)")
                                logger.info(f"⏱️ PIPELINE TIMING: Avg: {avg_publish_time:.1f}ms | Max: {max_publish_time:.1f}ms | Slow: {slow_ratio:.1f}% (target: <{self.buffer_config['max_publish_time_ms']}ms)")
                                
                                # Log frame pacer configuration status
                                if hasattr(self, 'frame_pacer') and self.frame_pacer:
                                    pacer_fps = self.frame_pacer.target_fps
                                    logger.info(f"🔄 FRAME PACER: Configured for {pacer_fps} FPS (matching SyncTalk reality)")
                                
                                frames_output = 0
                                last_timing_log = current_time
                            
                            # Log progress and performance
                            if frame_count % 50 == 0:
                                cycle_count = slide_frame_index // self.total_slide_frames if self.total_slide_frames > 0 else 0
                                current_slide = safe_slide_index
                                cache_indicator = " - CACHED" if self.use_dynamic_capture else ""
                                logger.info(f"📊 Slide progress: {current_slide}/{self.total_slide_frames} (cycle {cycle_count + 1}, frame {frame_count}){cache_indicator}")
                            
                            if frame_count % 25 == 0:  # Log every second  
                                elapsed_time = current_time - self.frame_delivery_start_time
                                sustained_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                                fps_diff = sustained_fps - target_fps
                                status = "✅" if abs(fps_diff) < 1.0 else "⚠️"
                                
                                # New color coding for intake buffer status
                                if intake_size > 30:
                                    buffer_status = "🟢"  # Plenty of buffer
                                elif intake_size > 10:
                                    buffer_status = "🟡"  # Getting low but still safe
                                elif intake_size > 0:
                                    buffer_status = "🟠"  # Very low but still have frames
                                else:
                                    buffer_status = "🔄"  # Duplicating frames
                                logger.info(f"🎬 {status} SUSTAINED: {sustained_fps:.1f}/{target_fps} FPS (diff: {fps_diff:+.1f}) {buffer_status} intake: {intake_size} frames")
                        
                        # Mark task as done - frame successfully processed and output
                        # Only mark as done if we actually got a frame from intake (not duplicated)
                        if not is_duplicated_frame:
                            self.intake_queue.task_done()
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Error in buffered frame delivery: {e}")
        
        logger.info("🎬 Buffered frame delivery stopped")
    
    async def _frame_collector_loop(self):
        """Independent frame collector - runs continuously without blocking audio"""
        logger.info("🎬 Frame collector started - collecting frames independently from audio")
        
        frame_collection_count = 0
        collection_start_time = time.time()
        
        while self.frame_delivery_running:
            try:
                # Collect frames as fast as SyncTalk produces them
                frame, audio = await self.receive_avatar_frame_with_audio(timeout=0.1)
                
                if frame and audio:
                    # Track received frames for monitoring
                    self.frames_received += 1
                    frame_collection_count += 1
                    
                    # MONITOR SYNCTALK PRODUCTION RATE
                    self.synctalk_frame_count += 1
                    if self.synctalk_start_time is None:
                        self.synctalk_start_time = time.time()
                    
                    # Log SyncTalk production rate every 25 frames
                    if self.synctalk_frame_count % 25 == 0:
                        elapsed = time.time() - self.synctalk_start_time
                        synctalk_fps = self.synctalk_frame_count / elapsed
                        fps_status = "✅" if synctalk_fps >= 24.0 else "⚠️"
                        logger.info(f"🤖 SYNCTALK DECOUPLED: {fps_status} {synctalk_fps:.1f} FPS (no backpressure!)")
                        
                        # Calculate decoupling benefit
                        current_time = time.time()
                        collection_elapsed = current_time - collection_start_time
                        collection_rate = frame_collection_count / collection_elapsed if collection_elapsed > 0 else 0
                        if collection_rate > synctalk_fps * 1.1:
                            logger.info(f"🚀 DECOUPLING BENEFIT: Collecting at {collection_rate:.1f} FPS vs SyncTalk {synctalk_fps:.1f} FPS")
                    
                    # Add to queue for smooth continuous delivery
                    frame_data = {
                        "type": "video",
                        "frame": frame,
                        "audio": audio,
                        "timestamp": time.time()
                    }
                    
                    try:
                        # Add to fast intake queue
                        self.intake_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                        # Intake full, drop oldest frame
                        try:
                            self.intake_queue.get_nowait()  # Remove oldest
                            self.intake_queue.put_nowait(frame_data)  # Add new
                            logger.debug("🔄 Intake overflow - dropped oldest frame")
                        except asyncio.QueueEmpty:
                            pass
                    
                    # Still collect for fallback
                    self.avatar_frames.append(frame)
                    self.avatar_audio_chunks.append(audio)
                    
            except asyncio.TimeoutError:
                # No frames available right now - continue collecting
                continue
            except Exception as e:
                logger.error(f"Error in frame collector: {e}")
                await asyncio.sleep(0.1)
                continue
        
        logger.info("🎬 Frame collector stopped")
    
    def configure_buffer_strategy(self, **kwargs):
        """
        Configure buffering strategy parameters.
        
        Args:
            min_buffer_threshold (int): Minimum frames to wait before starting output (default: 30)
            min_buffer_level (int): Minimum frames to maintain during playback (default: 10)
            refill_target (int): Target frames when refilling buffer (default: 20)
            target_fps (float): Target output frame rate (default: 25.0)
            max_timing_drift (float): Max timing drift before reset in seconds (default: 0.08)
        """
        valid_keys = {
            'min_buffer_threshold', 'min_buffer_level', 'refill_target', 
            'target_fps', 'max_timing_drift'
        }
        
        for key, value in kwargs.items():
            if key in valid_keys:
                old_value = self.buffer_config.get(key, 'N/A')
                self.buffer_config[key] = value
                logger.info(f"📦 Buffer config updated: {key}={old_value} → {value}")
            else:
                logger.warning(f"⚠️ Invalid buffer config key: {key}")
        
        # Validate configuration
        if self.buffer_config['min_buffer_level'] >= self.buffer_config['min_buffer_threshold']:
            logger.warning("⚠️ min_buffer_level should be less than min_buffer_threshold")
        
        if self.buffer_config['refill_target'] > self.buffer_config['min_buffer_threshold']:
            logger.warning("⚠️ refill_target should not exceed min_buffer_threshold")
        
        logger.info(f"📦 Current buffer config: {self.buffer_config}")
    
    async def publish_frame_to_livekit(self, bgr_frame, audio_chunk):
        """Publish frame and audio to LiveKit with properly configured optimizations"""
        # Check if optimizations are available and properly configured
        use_optimized = (OPTIMIZATIONS_AVAILABLE and 
                        hasattr(self, 'frame_pacer') and self.frame_pacer and
                        hasattr(self, 'audio_optimizer') and self.audio_optimizer)
        
        if use_optimized:
            # Convert BGR to RGB for optimized path with FIXED frame pacer
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            return await self.publish_frame_to_livekit_optimized(rgb_frame, audio_chunk)
        else:
            # Log once why we're using legacy mode
            if not hasattr(self, '_logged_legacy_mode'):
                self._logged_legacy_mode = True
                reasons = []
                if not OPTIMIZATIONS_AVAILABLE:
                    reasons.append("optimizations not available")
                if not hasattr(self, 'frame_pacer') or not self.frame_pacer:
                    reasons.append("frame_pacer not initialized")
                if not hasattr(self, 'audio_optimizer') or not self.audio_optimizer:
                    reasons.append("audio_optimizer not initialized")
                logger.info(f"Using legacy LiveKit publishing ({', '.join(reasons)})")
            
            # Use legacy publishing
            return await self._publish_frame_to_livekit_legacy(bgr_frame, audio_chunk)
    
    async def publish_frame_to_livekit_optimized(self, composed_frame, audio_chunk):
        """Optimized frame and audio publishing with buffering"""
        try:
            # Add frame to pacer (it should be RGB numpy array)
            if self.frame_pacer:
                # Ensure it's a numpy array
                if isinstance(composed_frame, Image.Image):
                    frame_array = np.array(composed_frame)
                else:
                    frame_array = composed_frame
                
                # Add frame with current timestamp
                success = self.frame_pacer.add_frame(frame_array)
                if not success:
                    logger.debug("Frame dropped by pacer")
            else:
                logger.warning("Frame pacer not initialized!")
            
            # Add audio to optimizer
            if audio_chunk and len(audio_chunk) > 0 and self.audio_optimizer:
                # Add audio chunk to buffer
                success = await self.audio_optimizer.add_audio_chunk(audio_chunk)
                if not success:
                    logger.debug("Audio chunk dropped (buffer full)")
                
                # Don't adapt too frequently - let buffer stabilize
                # Only adapt every 10 seconds
                if not hasattr(self, '_last_audio_adapt_time'):
                    self._last_audio_adapt_time = 0
                
                current_time = time.time()
                if current_time - self._last_audio_adapt_time > 10.0:
                    # Use conservative network estimates to keep larger buffers
                    await self.audio_optimizer.adapt_to_network(
                        packet_loss=0.5,  # Assume slight packet loss to keep buffer large
                        rtt_ms=100.0      # Assume moderate RTT
                    )
                    self._last_audio_adapt_time = current_time
            
            # Log performance metrics periodically
            if not hasattr(self, '_publish_count'):
                self._publish_count = 0
            self._publish_count += 1
                
            if self._publish_count % 100 == 0:  # Every 4 seconds at 25fps
                self._log_optimization_metrics()
                
        except Exception as e:
            logger.error(f"Error in optimized publishing: {e}")
    
    async def _publish_frame_to_livekit_legacy(self, bgr_frame, audio_chunk):
        """Legacy frame and audio publishing"""
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
            
            # Publish video frame
            self.video_source.capture_frame(video_frame)
            
            # Create audio frame from PCM chunk
            if len(audio_chunk) > 0:
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                samples_per_channel = len(audio_data)
                
                audio_frame = rtc.AudioFrame(
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=samples_per_channel,
                    data=audio_data.tobytes()
                )
                
                await self.audio_source.capture_frame(audio_frame)
                
        except Exception as e:
            logger.error(f"Error in legacy publishing: {e}")
    
    def _log_optimization_metrics(self):
        """Log comprehensive optimization metrics"""
        try:
            if self.frame_pacer:
                metrics = self.frame_pacer.get_metrics()
                logger.info(
                    f"📊 Frame Pacer: {metrics.actual_fps:.1f}/{metrics.target_fps:.0f} FPS | "
                    f"Delivered: {metrics.frames_delivered} | "
                    f"Dropped: {metrics.frames_dropped} | "
                    f"Duplicated: {metrics.frames_duplicated} | "
                    f"Latency: {metrics.average_latency_ms:.1f}ms | "
                    f"Buffer: {metrics.buffer_utilization:.0%}"
                )
            
            if self.audio_optimizer:
                audio_stats = self.audio_optimizer.get_stats()
                buffer_mode = getattr(self.audio_optimizer, 'buffer_mode', None)
                mode_str = buffer_mode.value if buffer_mode else "unknown"
                logger.info(
                    f"🎵 Audio [{mode_str}]: Buffer {audio_stats.current_buffer_ms:.0f}/{audio_stats.target_buffer_ms:.0f}ms | "
                    f"Underruns: {audio_stats.buffer_underruns} | "
                    f"Overruns: {audio_stats.buffer_overruns} | "
                    f"Level: {audio_stats.average_level:.2f} | "
                    f"Silence: {audio_stats.silence_ratio:.0%}"
                )
            
            # LiveKit stats logging (simplified without publisher)
            if hasattr(self, 'lk_room') and self.lk_room:
                logger.info(
                    f"📡 LiveKit: Connected and publishing | "
                    f"Frame pacer: {self.frame_pacer.is_running if self.frame_pacer else 'N/A'} | "
                    f"Audio optimizer: {'Active' if self.audio_optimizer else 'N/A'}"
                )
                
            if self.av_synchronizer:
                sync_status = self.av_synchronizer.get_sync_status()
                status_icon = "✅" if sync_status['in_sync'] else "⚠️"
                logger.info(
                    f"{status_icon} A/V Sync: Drift {sync_status['drift_ms']:.1f}ms | "
                    f"Offset: {sync_status['sync_offset_ms']:.1f}ms | "
                    f"Frames: {sync_status['video_frames']} | "
                    f"Samples: {sync_status['audio_samples']}"
                )
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    async def receive_avatar_frame_with_audio(self, timeout=1.5):
        """Receive protobuf frame and audio from SyncTalk using aiohttp WebSocket"""
        try:
            # Use aiohttp WebSocket receive
            msg = await asyncio.wait_for(self.websocket.receive(), timeout=timeout)
            
            if msg.type == aiohttp.WSMsgType.BINARY:
                protobuf_data = msg.data
            else:
                return None, None
            
            if isinstance(protobuf_data, bytes):
                frame_msg = FrameMessage()
                frame_msg.ParseFromString(protobuf_data)
                
                avatar_frame = None
                avatar_audio = None
                
                # Extract video frame
                if frame_msg.video_bytes:
                    video_data = np.frombuffer(frame_msg.video_bytes, dtype=np.uint8)
                    
                    # Handle different frame sizes - optimized for performance
                    if len(video_data) == 512 * 512 * 3:
                        frame_array = video_data.reshape((512, 512, 3))
                    elif len(video_data) == 350 * 350 * 3:
                        frame_array = video_data.reshape((350, 350, 3))
                    elif len(video_data) == 256 * 256 * 3:
                        frame_array = video_data.reshape((256, 256, 3))
                    elif len(video_data) == 128 * 128 * 3:
                        frame_array = video_data.reshape((128, 128, 3))
                    elif len(video_data) == 64 * 64 * 3:
                        frame_array = video_data.reshape((64, 64, 3))
                    else:
                        total_pixels = len(video_data) // 3
                        side = int(np.sqrt(total_pixels))
                        if side * side * 3 == len(video_data):
                            frame_array = video_data.reshape((side, side, 3))
                        else:
                            logger.warning(f"🔍 Unexpected frame size: {len(video_data)} bytes ({total_pixels} pixels)")
                            return None, None
                    
                    avatar_frame = Image.fromarray(frame_array, 'RGB')
                    
                    # Upscale ALL smaller frames to 512x512 for consistent processing and quality
                    original_size = frame_array.shape[:2]
                    if original_size != (512, 512):
                        avatar_frame = avatar_frame.resize((512, 512), Image.LANCZOS)
                        if self._frame_count < 3:
                            logger.info(f"🔍 Upscaling {original_size[0]}x{original_size[1]} → 512x512 using LANCZOS interpolation")
                    elif self._frame_count < 3:
                        logger.info(f"🔍 Frame already 512x512, no upscaling needed")
                    
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
                    # Apply audio smoothing to fix crackling from 40ms chunk boundaries
                    if self.audio_smoother:
                        avatar_audio = self.audio_smoother.smooth_chunk(frame_msg.audio_bytes)
                        # Log periodically
                        if self._frame_count % 250 == 0:
                            logger.debug(f"🔊 Audio smoother applied to {self.audio_smoother.chunk_count} chunks")
                    else:
                        avatar_audio = frame_msg.audio_bytes
                else:
                    avatar_audio = None
                
                return avatar_frame, avatar_audio
                    
        except asyncio.TimeoutError:
            return None, None
        except Exception as e:
            logger.error(f"Frame receive error: {e}")
            return None, None
    
    def _remove_green_screen_for_transparency_alin(self, image: Image.Image) -> Image.Image:
        """
        Optimized green screen removal for alin avatar with green background.
        Creates transparency from chroma key color (#bcfeb6).
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Log first few calls for debugging only
        if self._frame_count < 3:
            logger.info(f"🎨 Processing alin avatar frame {self._frame_count} for green screen removal (targeting #C2FFB6)")
            self._frame_count += 1
        
        # Convert image to numpy array (needed for final RGBA creation)
        img_array = np.array(image, dtype=np.uint8)
        height, width = img_array.shape[:2]
        
        # GPU-accelerated chroma key processing if available
        if self.use_gpu and TORCH_AVAILABLE:
            try:
                # Convert to tensor and move to GPU
                img_tensor = torch.from_numpy(img_array).float().to(self.device)
                
                # Chroma key color: #bcfeb6 = (188, 254, 182)
                target_color = torch.tensor([188.0, 254.0, 182.0], device=self.device)
                
                # Extract RGB channels
                r, g, b = img_tensor[:, :, 0], img_tensor[:, :, 1], img_tensor[:, :, 2]
                
                # GPU-accelerated green detection for #bcfeb6 (lighter green)
                green_dominant = (g > r + 20) & (g > b + 20)  # Green must be significantly higher
                
                # GPU-accelerated color similarity calculation
                color_diff = torch.sqrt(
                    (r - target_color[0]) ** 2 +
                    (g - target_color[1]) ** 2 +
                    (b - target_color[2]) ** 2
                )
                # Much tighter threshold to avoid removing avatar parts
                color_similar = color_diff < 25  # Reduced from 50 to 25
                
                # More specific detection for the exact light green background
                # #C2FFB6: R=194, G=255, B=182 - very specific ranges
                exact_green = (g > 240) & (g < 260) & (r > 180) & (r < 210) & (b > 170) & (b < 195)
                
                # Only remove pixels that are very close to the exact background color
                # AND have green dominance (G channel significantly higher than R and B)
                green_dominant = (g > r + 30) & (g > b + 40)  # Green must be much higher
                
                # Protect skin tones and avatar parts - exclude pixels that look like skin/clothing
                likely_skin = (r > 100) & (r < 255) & (g > 80) & (g < 200) & (b > 60) & (b < 180) & (r > g - 20)
                likely_clothing = (r < 100) & (g < 100) & (b < 100)  # Dark colors like suits
                
                # Combine conditions - much more restrictive AND avoid avatar parts
                is_green = color_similar & (exact_green | green_dominant) & (~likely_skin) & (~likely_clothing)
                
                # Create alpha channel on GPU and move back to CPU
                alpha = (~is_green).float() * 255.0
                alpha = alpha.cpu().numpy().astype(np.uint8)
                
            except Exception as e:
                logger.debug(f"GPU chroma key failed, using CPU: {e}")
                # Fallback to CPU processing
                target_color = np.array([188, 254, 182], dtype=np.uint8)
                r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
                
                color_diff = np.sqrt(
                    (r.astype(np.int16) - target_color[0]) ** 2 +
                    (g.astype(np.int16) - target_color[1]) ** 2 +
                    (b.astype(np.int16) - target_color[2]) ** 2
                )
                color_similar = color_diff < 25  # Much tighter threshold
                
                # More specific detection for exact background color
                exact_green = (g > 240) & (g < 260) & (r > 180) & (r < 210) & (b > 170) & (b < 195)
                green_dominant = (g > r + 30) & (g > b + 40)  # Green must be much higher
                
                # Protect skin tones and avatar parts
                likely_skin = (r > 100) & (r < 255) & (g > 80) & (g < 200) & (b > 60) & (b < 180) & (r > g - 20)
                likely_clothing = (r < 100) & (g < 100) & (b < 100)  # Dark colors like suits
                
                is_green = color_similar & (exact_green | green_dominant) & (~likely_skin) & (~likely_clothing)
                alpha = (~is_green).astype(np.uint8) * 255
        else:
            # CPU processing
            target_color = np.array([194, 255, 182], dtype=np.uint8)  # #C2FFB6
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            
            color_diff = np.sqrt(
                (r.astype(np.int16) - target_color[0]) ** 2 +
                (g.astype(np.int16) - target_color[1]) ** 2 +
                (b.astype(np.int16) - target_color[2]) ** 2
            )
            color_similar = color_diff < 25  # Much tighter threshold
            
            # More specific detection for exact background color
            exact_green = (g > 240) & (g < 260) & (r > 180) & (r < 210) & (b > 170) & (b < 195)
            green_dominant = (g > r + 30) & (g > b + 40)  # Green must be much higher
            
            # Protect skin tones and avatar parts
            likely_skin = (r > 100) & (r < 255) & (g > 80) & (g < 200) & (b > 60) & (b < 180) & (r > g - 20)
            likely_clothing = (r < 100) & (g < 100) & (b < 100)  # Dark colors like suits
            
            is_green = color_similar & (exact_green | green_dominant) & (~likely_skin) & (~likely_clothing)
            alpha = (~is_green).astype(np.uint8) * 255
        
        # Morphological operations for smooth edges
        if self._morph_kernel is None:
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Smooth the alpha channel
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, self._morph_kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, self._morph_kernel)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 1.5)  # Slightly more blur for smoother edges
        
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
    
    async def setup_dynamic_frame_capture(self):
        """Setup dynamic frame capture system"""
        if not self.use_dynamic_capture:
            logger.info("📁 Using static frames")
            return
            
        logger.info("🎬 Setting up dynamic frame capture...")
        logger.info(f"🌐 Capturing from URL: {self.capture_url}")
        logger.info("⏱️ Capture will end automatically when presentation finishes")
        
        try:
            # Start dynamic capture (non-blocking)
            from tab_capture import capture_presentation_frames, DynamicFrameProcessor
            self.dynamic_frames_dir = await capture_presentation_frames(
                capture_url=self.capture_url
            )
            
            if self.dynamic_frames_dir is None:
                raise Exception("Failed to start dynamic frame capture - no directory returned")
                
            logger.info("✅ Dynamic frame capture started in background")
            logger.info(f"📁 Frames will be saved to: {self.dynamic_frames_dir}")
            
            # Wait for first few frames to be captured before proceeding
            logger.info("⏳ Waiting for first slide frames to be captured...")
            await self._wait_for_initial_frames()
            
            # For dynamic capture, we use the cache system instead of frame processor reloading
            logger.info(f"🔄 Dynamic capture ready - using cache system for: {self.dynamic_frames_dir}")
            # Keep original frame processor for overlay operations only
            logger.info(f"📊 Dynamic capture setup completed - cache system active")
            
            logger.info("✅ Dynamic frame capture setup completed (non-blocking)")
            
        except Exception as e:
            logger.error(f"❌ Dynamic frame capture setup failed: {e}")
            raise
    
    async def _wait_for_initial_frames(self):
        """Wait for initial frames to be captured before starting SyncTalk"""
        frames_dir = Path(self.dynamic_frames_dir)
        min_frames_needed = 3
        max_wait_time = 30  # seconds
        
        start_time = time.time()
        while (time.time() - start_time) < max_wait_time:
            if frames_dir.exists():
                frame_files = list(frames_dir.glob("frame_*.png"))
                if len(frame_files) >= min_frames_needed:
                    logger.info(f"✅ Found {len(frame_files)} slide frames, ready to start SyncTalk!")
                    logger.info(f"📸 First frames: {[f.name for f in sorted(frame_files)[:3]]}")
                    return
                else:
                    logger.info(f"⏳ Found {len(frame_files)}/{min_frames_needed} frames, waiting for more...")
            else:
                logger.info("⏳ Waiting for first slide frame to be captured...")
            
            await asyncio.sleep(1.0)
        
        raise Exception(f"Timeout waiting for initial frames after {max_wait_time}s")
    
    async def _produce_slide_frames(self):
        """
        Producer task that watches dynamic capture directory and loads new frames into queue.
        Mirrors the exact SyncTalk pattern for non-blocking frame delivery.
        """
        if not self.dynamic_frames_dir:
            logger.error("❌ No dynamic frames directory set")
            return
            
        logger.info(f"🎬 Starting slide frame producer for: {self.dynamic_frames_dir}")
        frames_dir = Path(self.dynamic_frames_dir)
        
        processed_frames = set()  # Track which frames we've already processed
        frame_index_counter = 0  # Sequential counter for proper indexing
        
        while self.frame_delivery_running:
            try:
                # Find all PNG files in the directory
                if frames_dir.exists():
                    frame_files = sorted(list(frames_dir.glob("frame_*.png")))
                    
                    # Process new frames only
                    for frame_file in frame_files:
                        if frame_file.name not in processed_frames:
                            try:
                                # Check for duplicate frames by file size (basic deduplication)
                                file_size = frame_file.stat().st_size
                                if file_size < 1000:  # Skip very small files (likely corrupted)
                                    logger.debug(f"⚠️ Skipping small file: {frame_file.name} ({file_size} bytes)")
                                    processed_frames.add(frame_file.name)  # Mark as processed to avoid retry
                                    continue
                                
                                # Load frame with better error handling
                                frame_image = Image.open(frame_file).resize((854, 480))
                                
                                # Thread-safe cache operations
                                async with self._cache_lock:
                                    # Cache the frame with proper sequential index
                                    self.slide_frames_cache[frame_index_counter] = frame_image
                                    self.slide_frame_count = frame_index_counter + 1
                                    
                                    # Memory management: remove old frames if cache gets too large
                                    if len(self.slide_frames_cache) > self.max_cached_frames:
                                        # Remove oldest 100 frames to free memory
                                        frames_to_remove = 100
                                        oldest_keys = sorted(self.slide_frames_cache.keys())[:frames_to_remove]
                                        for old_key in oldest_keys:
                                            del self.slide_frames_cache[old_key]
                                        logger.info(f"🧹 Memory cleanup: removed {frames_to_remove} old frames, cache size: {len(self.slide_frames_cache)}")
                                
                                # Add to processed set
                                processed_frames.add(frame_file.name)
                                frame_index_counter += 1
                                
                                # Log progress every 50 frames
                                if frame_index_counter % 50 == 0:
                                    logger.info(f"📈 Slide producer: {frame_index_counter} frames cached")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error loading slide frame {frame_file}: {e}")
                                # Mark as processed to avoid infinite retry, but don't increment counter
                                processed_frames.add(frame_file.name)
                                # Add retry logic for network/IO issues
                                if "Permission denied" in str(e) or "being used by another process" in str(e):
                                    logger.info(f"🔄 Will retry {frame_file.name} in next cycle")
                                    processed_frames.discard(frame_file.name)  # Allow retry
                
                # Check every 100ms for new frames
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error in slide frame producer: {e}")
                await asyncio.sleep(1.0)  # Wait longer on error
                
        logger.info("🛑 Slide frame producer stopped")

    def get_cached_slide_frame(self, frame_index: int) -> Optional[Image.Image]:
        """Get slide frame from cache - non-blocking like SyncTalk"""
        # Create a snapshot of cache to avoid race conditions
        cache_snapshot = dict(self.slide_frames_cache)
        frame_count_snapshot = self.slide_frame_count
        
        if frame_index in cache_snapshot:
            return cache_snapshot[frame_index]
        
        # If frame doesn't exist, cycle through available frames
        if frame_count_snapshot > 0:
            safe_index = frame_index % frame_count_snapshot
            return cache_snapshot.get(safe_index)
            
        return None
    
    def remove_green_screen_with_despill(self, image: Image.Image) -> Image.Image:
        """Apply chroma key with despill factor to prevent background meshing"""
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image, dtype=np.float32)
        
        # Green screen color from SyncTalk server: #bcfeb6 = (188, 254, 182)
        target_color = np.array([188, 254, 182], dtype=np.float32)
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
            
            # Step 2.5: Setup frame capture (static or dynamic)
            logger.info("🎬 Setting up frame capture...")
            await self.setup_dynamic_frame_capture()
            
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
            # Clean up slide producer task
            if hasattr(self, '_slide_producer_task') and self._slide_producer_task:
                logger.info("🛑 Stopping slide frame producer task...")
                self._slide_producer_task.cancel()
                try:
                    await self._slide_producer_task
                except asyncio.CancelledError:
                    logger.info("✅ Slide frame producer task stopped")
                except Exception as e:
                    logger.warning(f"⚠️ Error stopping slide producer task: {e}")
            
            # Cleanup connections properly
            if hasattr(self, 'audio_processor_task') and self.audio_processor_task:
                self.audio_processor_task.cancel()
            if hasattr(self, 'frame_collector_task') and self.frame_collector_task:
                self.frame_collector_task.cancel()
            if self.websocket:
                await self.websocket.close()
            if self.aiohttp_session:
                await self.aiohttp_session.close()

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
    parser.add_argument("--frames", "-f", default=r"C:\Work\agent-backend\custom_capture\new_frame", help="Slide frames directory")
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
