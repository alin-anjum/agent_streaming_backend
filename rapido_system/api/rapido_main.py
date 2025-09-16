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
import uuid
from pathlib import Path
import argparse
import requests
import threading

# Import optimized modules (will check after logger is set up)
OPTIMIZATIONS_AVAILABLE = False

# Compute project roots once
current_dir = os.path.dirname(os.path.abspath(__file__))
api_dir = current_dir
rapido_system_root = os.path.dirname(api_dir)
project_root = os.path.dirname(rapido_system_root)

# Add project root to Python path for imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add SyncTalk_2D path for protobuf imports
sync_talk_path = os.path.join(project_root, 'SyncTalk_2D')
if sync_talk_path not in sys.path:
    sys.path.append(sync_talk_path)

# Now import with correct paths
from rapido_system.core.config.config import Config
from rapido_system.api.data_parser import SlideDataParser
from rapido_system.api.tts_client import ElevenLabsTTSClient
from rapido_system.api.frame_processor import FrameOverlayEngine

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

# Import SyncTalk protobuf (path already added above)
try:
    from frame_message_pb2 import FrameMessage
    logger.info("✅ Protobuf integration ready")
except ImportError as e:
    logger.error(f"❌ Failed to import protobuf: {e}")
    logger.error("Make sure SyncTalk_2D/frame_message_pb2.py exists")
    sys.exit(1)

# Import SyncTalk chroma key for green screen removal
try:
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
        self.end_of_stream_sent = False  # Prevent any audio sends after EOS
        self._video_writer = None        # OpenCV VideoWriter for final MP4
        self._video_output_path = None   # Path to saved video
        self._composed_frames_dir = None # Directory to save composed frames
        self._composed_frame_index = 0   # Incremental index for saved frames
        self._audio_pcm_path = None      # Path to raw PCM during composition
        self._audio_wav_path = None      # Final WAV path (converted from PCM)
        self._final_mp4_path = None      # Final stitched video output
        self.avatar_frames = []
        self.avatar_audio_chunks = []
        # Get the script directory and resolve paths relative to the project root
        script_dir = Path(__file__).parent.parent  # rapido_system/api -> rapido_system
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
        self.video_job_id = getattr(self.config, 'VIDEO_JOB_ID', None)
        logger.info(f"🔍 Dynamic capture config: USE_DYNAMIC_CAPTURE={self.use_dynamic_capture}, CAPTURE_URL={self.capture_url}")
        if self.video_job_id:
            logger.info(f"🆔 Video Job ID for document capture: {self.video_job_id}")

        # Real-time slide frame streaming (tab-capture) additions - EXACT SYNCTALK PATTERN
        # Mirror the exact SyncTalk pattern: asyncio.Queue with producer task
        # that watches the capture directory and feeds frames to a queue
        # for non-blocking consumption by the compositor.
        self.slide_frame_queue: Optional[asyncio.Queue] = None  # Created after capture starts
        self.slide_control_queue: Optional[asyncio.Queue] = None  # Signals advance to next slide
        self._slide_producer_task: Optional[asyncio.Task] = None
        self.slide_frames_cache = {}  # Cache loaded slide frames by index
        self.slide_frame_count = 0    # Current number of available slide frames
        self.max_cached_frames = 1000  # Limit cache size to prevent memory issues
        self._cache_lock = asyncio.Lock()  # Thread safety for cache operations
        self.dynamic_frame_processor = None
        self.dynamic_frames_dir = None  # Set after capture starts
        self.presenter_map = {}
        self.doc_canvas_size = (1920, 1080)
        # Slide timing offset: negative advances earlier (frames at 25fps). Default -30 (~1.2s)
        try:
            self.slide_timing_offset_frames = int(getattr(self.config, 'SLIDE_TIMING_OFFSET_FRAMES', -30))
        except Exception:
            self.slide_timing_offset_frames = -30
        # Clamp offset to a safe range
        self.slide_timing_offset_frames = max(-60, min(0, self.slide_timing_offset_frames))

        # Initialize frame processor based on capture mode
        if self.use_dynamic_capture:
            # For dynamic capture, create minimal processor for overlay operations only
            logger.info("🎬 Dynamic capture mode - creating overlay-only processor")
            # Create a dummy empty directory to avoid loading any static frames
            import tempfile
            temp_dir = tempfile.mkdtemp()
            self.frame_processor = FrameOverlayEngine(temp_dir, output_size=(1280, 720), dynamic_mode=True)
            self.total_slide_frames = 0
            self.total_duration_seconds = 0
        else:
            # For static frames, load the presentation frames
            logger.info("📁 Static frame mode - loading presentation frames")
            self.frame_processor = FrameOverlayEngine(self.slide_frames_path, output_size=(1280, 720))
            self.total_slide_frames = self.frame_processor.get_frame_count()
        self.total_duration_seconds = self.total_slide_frames / self.synctalk_fps
        # Try to load presenter positions from captured document

        # Embed/pause control state
        self.paused_for_embed = False
        self.timeline_slides = []
        self._timeline_index_by_canvas = {}
        self._deferred_advance = False
        self._load_presentation_timeline()
        self._initial_embed_checked = False

        # LiveKit track publish gating: only publish after buffer prefill
        self._tracks_published = False

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    async def _publish_control_event(self, payload: dict):
        """Send a control message over LiveKit data channel (topic 'control')."""
        try:
            import livekit.rtc as rtc
            if getattr(self, 'lk_room', None) and self.lk_room.local_participant:
                data = json.dumps(payload).encode('utf-8')
                await self.lk_room.local_participant.publish_data(
                    data,
                    rtc.DataPacketKind.RELIABLE,
                    topic='control'
                )
        except Exception as e:
            logger.warning(f"Control publish failed: {e}")

    async def send_show_embed(self, slide_id: str):
        await self._publish_control_event({
            'id': str(uuid.uuid4()),
            'message': 'showEmbed',
            'timestamp': self._now_ms(),
            'slideId': slide_id,
        })

    async def send_resume_stream(self):
        await self._publish_control_event({
            'id': str(uuid.uuid4()),
            'message': 'resumeStream',
            'timestamp': self._now_ms(),
        })

    async def send_stream_event_to_frontend(self, event_type: str, **kwargs):
        """Map to StreamEndEvent on completion to notify frontend UI."""
        if event_type == 'stream_ended':
            await self._publish_control_event({
                'id': str(uuid.uuid4()),
                'message': 'streamEnd',
                'timestamp': self._now_ms(),
            })

    async def _pause_for_embed(self, embed_id: str):
        """Pause LiveKit publishing and instruct frontend to show embed."""
        if self.paused_for_embed:
            return
        self.paused_for_embed = True
        logger.info(f"⏸️ Pausing publishing for embed: {embed_id}")
        # Drain any buffered frames to stop pending publishing
        try:
            await self._drain_queues_for_pause()
        except Exception as _:
            pass
        await self.send_show_embed(embed_id)

    async def _resume_from_embed(self):
        """Resume LiveKit publishing and advance slides if a defer was pending."""
        if not self.paused_for_embed:
            return
        self.paused_for_embed = False
        logger.info("▶️ Resuming publishing after embed")
        await self.send_resume_stream()
        # If there are consecutive embeds after current canvas, immediately pause again
        try:
            embed_next = self._next_embed_after_current()
        except Exception:
            embed_next = None
        if embed_next:
            await self._pause_for_embed(embed_next)
            # keep _deferred_advance True for when embed chain ends
            return
        # Otherwise, if we deferred a slide advance while pausing, trigger it now
        if self._deferred_advance and self.slide_control_queue is not None:
            try:
                self.slide_control_queue.put_nowait("advance")
            except Exception:
                pass
            self._deferred_advance = False

    def _next_embed_after_current(self) -> Optional[str]:
        """If the next timeline item after current canvas is an embed, return its id."""
        try:
            current_id = getattr(self, '_current_slide_id', None)
            if not current_id or not self.timeline_slides:
                return None
            idx = self._timeline_index_by_canvas.get(current_id)
            if idx is None:
                return None
            if idx + 1 < len(self.timeline_slides):
                nxt = self.timeline_slides[idx + 1]
                if nxt.get('type') == 'embed':
                    return nxt.get('id')
            return None
        except Exception:
            return None

    async def _drain_queues_for_pause(self):
        """Drop any queued frames so no stale frames get published after pause.
        Also prevents memory growth while waiting on embed UI."""
        drained_avatar = 0
        drained_slide = 0
        try:
            # Drain avatar frame intake queue
            if hasattr(self, 'intake_queue') and self.intake_queue is not None:
                while True:
                    try:
                        _ = self.intake_queue.get_nowait()
                        drained_avatar += 1
                    except asyncio.QueueEmpty:
                        break
                # Avoid duplicating stale last frame during pause
                self.last_frame_data = None
        except Exception:
            pass
        try:
            # Drain slide frames if any
            if self.slide_frame_queue is not None:
                while True:
                    try:
                        _ = self.slide_frame_queue.get_nowait()
                        drained_slide += 1
                    except asyncio.QueueEmpty:
                        break
        except Exception:
            pass
        if drained_avatar or drained_slide:
            logger.info(f"🧺 Drained queues on pause - avatar={drained_avatar}, slide={drained_slide}")

    def _load_presentation_timeline(self):
        """Build ordered timeline from parsed_slideData with canvas and embed entries."""
        try:
            if not self.video_job_id:
                return
            parsed_path = Path(f"./rapido_system/data/parsed_slideData/{self.video_job_id}.json")
            if not parsed_path.exists():
                logger.info(f"No parsed_slideData found at {parsed_path}; timeline will be empty")
                return
            with parsed_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            timeline = []
            for entry in (data.get("slides") or []):
                info = entry.get("slideInfo") or {}
                slide_type = info.get("slideType")
                if slide_type == 1 and info.get("slideId"):
                    timeline.append({"type": "canvas", "id": info["slideId"]})
                elif slide_type == 2 and info.get("embedId"):
                    timeline.append({"type": "embed", "id": info["embedId"]})
            self.timeline_slides = timeline
            self._timeline_index_by_canvas = {
                it["id"]: i for i, it in enumerate(timeline) if it.get("type") == "canvas"
            }
            if timeline:
                logger.info(f"📜 Loaded timeline with {len(timeline)} items (canvas+embed)")
        except Exception as e:
            logger.warning(f"Failed to load presentation timeline: {e}")
        try:
            self._load_presenter_positions()
        except Exception as e:
            logger.warning(f"Presenter position map not loaded: {e}")
        
        # Initialize chroma key processor
        self.chroma_key_processor = None
        self._init_chroma_key()

        # Performance optimization: cache morphological kernel
        self._morph_kernel = None
        self._frame_count = 0
        
        # Precompute composed output targets
        try:
            safe_suffix = self.video_job_id if self.video_job_id else str(int(time.time()))
            os.makedirs(self.output_dir, exist_ok=True)
            self._composed_frames_dir = os.path.join(self.output_dir, f"composed_frames_{safe_suffix}")
            self._audio_pcm_path = os.path.join(self.output_dir, f"audio_{safe_suffix}.pcm")
            self._audio_wav_path = os.path.join(self.output_dir, f"audio_{safe_suffix}.wav")
            self._final_mp4_path = os.path.join(self.output_dir, f"final_{safe_suffix}.mp4")
            # Cleanup existing audio files
            try:
                if os.path.exists(self._audio_pcm_path):
                    os.remove(self._audio_pcm_path)
                if os.path.exists(self._audio_wav_path):
                    os.remove(self._audio_wav_path)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Failed to initialize composed output targets: {e}")
        
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
            'min_buffer_threshold': 75,    # Wait for 75 frames (~3s at 25fps)
            'min_buffer_level': 40,        # Keep at least ~1.6s buffered for smoothness
            'refill_target': 75,           # Align with threshold
            'target_fps': 25.0,            # Target 25 FPS output
            'max_timing_drift': 0.08,      # Max timing drift before reset (2 frame intervals)
            'duplicate_when_empty': True,  # Duplicate last frame instead of pausing - SMOOTH PLAYBACK!
            'max_composition_time_ms': 30, # Max acceptable composition time
            'max_publish_time_ms': 40,     # Max acceptable publish time
            'bypass_frame_pacer': False    # Fixed: Configure frame pacer properly instead of bypassing
        }
        
        logger.info(f"📊 Rapido initialized - Duration: {self.total_duration_seconds:.2f}s")
        logger.info(f"📦 Buffer config: threshold={self.buffer_config['min_buffer_threshold']}, level={self.buffer_config['min_buffer_level']}, fps={self.buffer_config['target_fps']}")
        logger.info(f"🎯 Output target set to {self.buffer_config['target_fps']} FPS (duplication on underrun enabled)")
    
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
            
            # Initialize FastChromaKey with reduced avatar frame size (400x400) for faster compose
            # This must match the processing resize below
            # Use a beige/cream background that matches the current frame background
            background_color = np.full((400, 400, 3), [220, 210, 190], dtype=np.uint8)  # Beige background
            
            # Use the exact chroma key settings from the alin avatar configuration
            self.chroma_key_processor = FastChromaKey(
                width=400,
                height=400,
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

    def _load_presenter_positions(self):
        """Load presenterConfig map {slide_id: config} from captured document for this job id."""
        if not self.video_job_id:
            return
        try:
            from pathlib import Path
            import json
            document_dir = Path("./captured_documents")
            pattern = f"document_{self.video_job_id}_*.json"
            files = list(document_dir.glob(pattern))
            if not files:
                return
            latest = max(files, key=lambda f: f.stat().st_mtime)
            with latest.open('r', encoding='utf-8') as f:
                data = json.load(f)
            # Default canvas size
            self.doc_canvas_size = (1920, 1080)
            # Build presenter map
            slides = data.get('slides', [])
            for slide in slides:
                if not isinstance(slide, dict):
                    continue
                if slide.get('contentType') != 'canvas':
                    continue
                sid = slide.get('id')
                presenter = slide.get('presenterConfig') or slide.get('defaultPresenterConfig')
                if sid and presenter and isinstance(presenter, dict):
                    self.presenter_map[sid] = presenter
                # Try find canvas dims from background image natural size if present
                objs = slide.get('objects', [])
                for obj in objs:
                    if isinstance(obj, dict) and 'width' in obj and 'height' in obj:
                        try:
                            w = int(round(float(obj['width'])))
                            h = int(round(float(obj['height'])))
                            if w > 0 and h > 0:
                                self.doc_canvas_size = (w, h)
                                break
                        except Exception:
                            pass
            logger.info(f"📍 Presenter map loaded for {len(self.presenter_map)} slides; canvas={self.doc_canvas_size}")
        except Exception as e:
            logger.warning(f"Failed loading presenter positions: {e}")
    
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
        if self.end_of_stream_sent:
            logger.debug("🔇 Ignoring audio chunk after EOS")
            return
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
        """Connect to LiveKit - FORCE LEGACY MODE for stability"""
        # FORCE LEGACY MODE - Optimized mode was causing InvalidState errors
        logger.info("🔄 Using LEGACY LiveKit connection for maximum stability")
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
            LIVEKIT_URL = "wss://agent-staging-2y5e52au.livekit.cloud"
            LIVEKIT_API_KEY = "APIwYq3kxYAwbqP"
            LIVEKIT_API_SECRET = "Jgp3uNValdBdnJefeAt8qk2ZGRsBFGNxk97NfelT9gKC"
            
            # Get room name from config
            room_name = getattr(self.config, 'LIVEKIT_ROOM', 'avatar-room2')
            logger.info(f"🏠 Optimized LiveKit connecting to room: {room_name}")
            
            # Generate JWT token
            current_time = int(time.time())
            token_payload = {
                "iss": LIVEKIT_API_KEY,
                "sub": f"avatar_bot_{room_name}",
                "aud": "livekit",
                "exp": current_time + 3600,
                "nbf": current_time - 10,
                "iat": current_time,
                "jti": f"avatar_bot_{room_name}_{current_time}",
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
            
            # Create sources (720p)
            self.video_source = rtc.VideoSource(1280, 720)
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
                        # Ensure frame is proper RGB numpy array with correct memory layout
                        if isinstance(frame, Image.Image):
                            rgb_frame = np.array(frame, dtype=np.uint8)
                        else:
                            rgb_frame = np.ascontiguousarray(frame, dtype=np.uint8)
                        
                        # Handle RGBA frames (convert to RGB)
                        if len(rgb_frame.shape) == 3:
                            if rgb_frame.shape[2] == 4:
                                # RGBA to RGB conversion - blend with white background
                                alpha = rgb_frame[:, :, 3:4] / 255.0
                                rgb_frame = rgb_frame[:, :, :3] * alpha + (1 - alpha) * 255
                                rgb_frame = rgb_frame.astype(np.uint8)
                            elif rgb_frame.shape[2] != 3:
                                logger.error(f"Frame pacer: Invalid frame format: {rgb_frame.shape}")
                                return
                        else:
                            logger.error(f"Frame pacer: Invalid frame format: {rgb_frame.shape}")
                            return
                        
                        height, width, channels = rgb_frame.shape
                        
                        # Ensure contiguous memory layout
                        if not rgb_frame.flags['C_CONTIGUOUS']:
                            rgb_frame = np.ascontiguousarray(rgb_frame)
                        
                        # Create video frame
                        video_frame = rtc.VideoFrame(
                            width=width,
                            height=height,
                            type=rtc.VideoBufferType.RGB24,
                            data=rgb_frame.tobytes()
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
            import uuid
            
            # LiveKit credentials
            LIVEKIT_URL = "wss://agent-staging-2y5e52au.livekit.cloud"
            LIVEKIT_API_KEY = "APIwYq3kxYAwbqP"
            LIVEKIT_API_SECRET = "Jgp3uNValdBdnJefeAt8qk2ZGRsBFGNxk97NfelT9gKC"
            
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
            
            # Create sources (720p)
            self.video_source = rtc.VideoSource(1280, 720)
            self.audio_source = rtc.AudioSource(16000, 1)
            
            # Publish tracks
            video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self.video_source)
            video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            self.video_publication = await self.lk_room.local_participant.publish_track(video_track, video_options)
            
            audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self.audio_source)
            audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            self.audio_publication = await self.lk_room.local_participant.publish_track(audio_track, audio_options)
            
            logger.info("✅ Connected to LiveKit (legacy mode)!")
            # Bind control data reception if available
            try:
                @self.lk_room.on("data_received")
                def _on_data_received(data, participant, kind, topic: str):
                    try:
                        if topic != "control":
                            return
                        raw = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
                        evt = json.loads(raw)
                        msg = (evt or {}).get("message")
                        if msg in ("resumeStream", "goLive"):
                            asyncio.create_task(self._resume_from_embed())
                    except Exception as ee:
                        logger.debug(f"control data parsing failed: {ee}")
            except Exception as e:
                logger.warning(f"LiveKit data event binding not available: {e}")
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
        
        # Avatar frame intake queue - allow large backlog since SyncTalk can burst >50 FPS
        # This prevents fast-forwarding when output is paced at 25 FPS.
        self.intake_queue = asyncio.Queue(maxsize=5000)
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
        
        # Note: For dynamic capture, frames are now fed directly to queue by browser service
        # No separate slide frame producer task needed since browser feeds queue directly
        
        # Note: stream_started event will be sent when buffer fills (see continuous_frame_delivery)
        
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

        # Send end_stream message to LiveKit frontend clients FIRST
        await self.send_stream_event_to_frontend("stream_ending",
            message="Avatar presentation is ending",
            total_audio_chunks=self.audio_chunks_sent,
            duration_seconds=time.time() - getattr(self, 'stream_start_time', time.time())
        )

        # Immediately signal end-of-stream to SyncTalk so it can stop generating new frames
        # and flush remaining frames naturally.
        logger.info("📡 Sending end marker to SyncTalk immediately after TTS completion...")
        try:
            self.end_of_stream_sent = True
            await self.audio_send_queue.put({"type": "end_stream"})
            logger.info("✅ End marker queued for audio processor")
        except Exception as e:
            logger.warning(f"Failed to queue end marker: {e}")

        # ENHANCED: More responsive frame draining with shorter timeouts
        quiet_required_seconds = 0.5  # Reduced from 1.0 for faster response
        empty_required_seconds = 0.5  # Reduced from 1.0 for faster response
        check_interval = 0.2  # Reduced from 0.5 for more responsive checking
        max_drain_time = 10.0  # Maximum time to wait for draining
        
        last_received_count = self.frames_received
        quiet_elapsed = 0.0
        empty_elapsed = 0.0
        drain_start_time = time.time()
        
        logger.info("⏳ Enhanced frame draining - shorter timeouts, more responsive...")
        
        while True:
            await asyncio.sleep(check_interval)
            
            # Check for overall timeout
            if time.time() - drain_start_time > max_drain_time:
                logger.warning(f"⚠️ Frame draining timeout after {max_drain_time}s - forcing completion")
                break
            
            # Check quiet period on incoming frames
            current_received = self.frames_received
            if current_received == last_received_count:
                quiet_elapsed += check_interval
            else:
                quiet_elapsed = 0.0
                last_received_count = current_received
            
            # Check output queue emptiness
            qsize = self.intake_queue.qsize()
            if qsize == 0:
                empty_elapsed += check_interval
            else:
                empty_elapsed = 0.0
            
            # More frequent status updates for better visibility
            if int(time.time() * 2) % 5 == 0:  # Every 2.5 seconds
                logger.info(f"🧺 Drain status: queue={qsize}, quiet={quiet_elapsed:.1f}s, empty={empty_elapsed:.1f}s, received={current_received}")
            
            # Exit when both conditions satisfied
            if quiet_elapsed >= quiet_required_seconds and empty_elapsed >= empty_required_seconds:
                logger.info(
                    f"✅ Enhanced drain complete: queue empty ({empty_elapsed:.1f}s) and quiet ({quiet_elapsed:.1f}s). "
                    f"Total frames received: {current_received}"
                )
                break
        
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
        
        # Close video writer if open and summarize
        if self._video_writer is not None:
            try:
                self._video_writer.release()
                logger.info(f"🎬 Final video saved: {self._video_output_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to finalize video writer: {e}")
            self._video_writer = None

        # Cleanup dynamic capture browser if running
        try:
            if hasattr(self, '_browser_service') and self._browser_service is not None:
                await self._browser_service.cleanup()
                logger.info("🧹 Dynamic capture browser cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup dynamic capture browser: {e}")
        
        # Close SyncTalk websocket then aiohttp session
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("✅ SyncTalk websocket closed")
            except Exception as e:
                logger.warning(f"⚠️ Websocket close failed: {e}")
            self.websocket = None
        if self.aiohttp_session:
            await self.aiohttp_session.close()
            logger.info("✅ aiohttp session closed")

        # Disconnect LiveKit room if connected
        try:
            if hasattr(self, 'lk_room') and self.lk_room:
                await self.lk_room.disconnect()
                logger.info("✅ LiveKit room disconnected")
        except Exception as e:
            logger.warning(f"⚠️ LiveKit disconnect failed: {e}")
        
        # Finalize WAV and stitch frames+audio into MP4
        try:
            self._finalize_audio_wav(sample_rate=16000)
            self._stitch_final_video(fps=self.target_video_fps)
        except Exception as e:
            logger.error(f"Final stitching failed: {e}")

        return True
    
    async def continuous_frame_delivery(self, frame_processor):
        """Stream frames with buffering strategy to maintain steady 25fps output"""
        try:
            slide_frame_index = 0
            frame_count = 0
            # If timeline starts with embeds (or multiple embeds), pause immediately before first output
            if not getattr(self, '_initial_embed_checked', False):
                try:
                    # When no current slide yet, we look at the first timeline item
                    if self.timeline_slides:
                        # If first item(s) are embeds, pause and show the first embed
                        first = self.timeline_slides[0]
                        if first.get('type') == 'embed':
                            await self._pause_for_embed(first.get('id'))
                            self._deferred_advance = True
                except Exception:
                    pass
                self._initial_embed_checked = True
            
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
                
                # Ensure threshold is not higher than the queue size
                effective_threshold = min(min_buffer_threshold, self.intake_queue.maxsize)
                # If audio ended, skip prefill and start output immediately
                if getattr(self, 'tts_streaming_complete', False) and intake_size > 0:
                    buffer_filled = True
                    buffer_wait_time = time.time() - buffer_wait_start
                    logger.info(f"🎬 Audio ended - starting with current buffer {intake_size} after {buffer_wait_time:.1f}s")
                    break
                if intake_size >= effective_threshold:
                    buffer_filled = True
                    buffer_wait_time = time.time() - buffer_wait_start
                    logger.info(f"🎬 ✅ Intake buffer filled! {intake_size} frames ready after {buffer_wait_time:.1f}s - starting steady {target_fps}fps output")
                    
                    # Send stream started event to frontend
                    await self.send_stream_event_to_frontend(
                        "stream_started", 
                        message="Avatar presentation stream has started",
                        target_fps=target_fps,
                        buffer_size=intake_size
                    )
                    break
                
                # Log buffer filling progress every 2 seconds
                if int(time.time() - buffer_wait_start) % 2 == 0:
                    elapsed = time.time() - buffer_wait_start
                    if elapsed > 1:  # Only log after first second
                        logger.info(
                            f"🎬 📦 Intake buffer: {intake_size}/{effective_threshold} frames "
                            f"(queue max: {self.intake_queue.maxsize}, {elapsed:.1f}s elapsed)"
                        )
                
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
                        # After audio end, DO NOT duplicate; end when intake empties
                        if getattr(self, 'tts_streaming_complete', False):
                            logger.info("🎬 Intake empty and audio ended - stopping output")
                            break
                        # During audio, duplicate last frame for smooth playback
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
                        
                        # Get slide frame from slide_frame_queue with same buffer logic as avatar frames
                        slide_frame = None
                        slide_frame_available = False
                        
                        if self.slide_frame_queue and not self.slide_frame_queue.empty():
                            try:
                                # Get slide frame from queue (non-blocking)
                                queue_item = self.slide_frame_queue.get_nowait()
                                # Support both 2-tuple (index, image) and 3-tuple (index, image, meta)
                                if isinstance(queue_item, tuple) and len(queue_item) >= 2:
                                    slide_frame_index_from_queue = queue_item[0]
                                    slide_frame = queue_item[1]
                                    slide_meta = queue_item[2] if len(queue_item) >= 3 else None
                                else:
                                    slide_frame_index_from_queue = 0
                                    slide_frame = queue_item
                                    slide_meta = None
                                slide_frame_available = True
                                self.last_slide_frame = slide_frame  # Save for potential duplication
                                self.last_slide_frame_index = slide_frame_index_from_queue
                                
                                # Handle per-slide loop counter on transition
                                if slide_meta and isinstance(slide_meta, dict):
                                    incoming_slide_id = slide_meta.get("slide_id")
                                    incoming_slide_idx = slide_meta.get("slide_index")
                                    expected_loop = slide_meta.get("expected_loop_frames")
                                    
                                    # Initialize tracking state if missing
                                    if not hasattr(self, '_current_slide_id'):
                                        self._current_slide_id = None
                                        self._current_slide_index = None
                                        self._current_slide_frame_loops = 0
                                        self._advance_signal_time = None
                                        self._last_nav_delay_frames = None
                                    
                                    # If slide changed, log previous and reset counter
                                    if self._current_slide_id is not None and incoming_slide_id != self._current_slide_id:
                                        logger.info(f"🧮 Slide {self._current_slide_index} ({self._current_slide_id}) looped for {self._current_slide_frame_loops} frames")
                                        if hasattr(self, '_current_slide_expected') and self._current_slide_expected is not None:
                                            logger.info(f"🎯 Expected ~{self._current_slide_expected} frames at 25fps")
                                        self._current_slide_frame_loops = 0
                                    
                                    # Update current slide tracking
                                    self._current_slide_id = incoming_slide_id
                                    self._current_slide_index = incoming_slide_idx
                                    # Minimal margin; subtract measured navigation delay and apply early-advance offset
                                    margin_frames = 0
                                    lead_frames = 0
                                    try:
                                        if hasattr(self, '_advance_signal_time') and self._advance_signal_time:
                                            # Measure time from last advance signal to now (first frame of new slide)
                                            nav_delay_sec = max(0.0, time.time() - self._advance_signal_time)
                                            measured_lead = int(round(nav_delay_sec * 25.0))
                                            self._last_nav_delay_frames = measured_lead
                                            # Clamp lead between 2 and 120 frames
                                            lead_frames = min(120, max(2, measured_lead))
                                            logger.info(f"⏱️ Measured nav delay: {nav_delay_sec*1000:.0f}ms (~{measured_lead} frames); applying lead={lead_frames}")
                                        elif hasattr(self, '_last_nav_delay_frames') and self._last_nav_delay_frames:
                                            lead_frames = min(120, max(2, int(self._last_nav_delay_frames)))
                                            logger.info(f"⏱️ Using last nav delay lead={lead_frames} frames")
                                    except Exception:
                                        lead_frames = 0
                                    # Compute threshold with guard and timing offset
                                    base_expected = (
                                        int(expected_loop or 0)
                                        + int(self.slide_timing_offset_frames)
                                        + margin_frames
                                        - int(lead_frames)
                                    )
                                    self._current_slide_expected = max(1, base_expected)
                                    logger.info(
                                        f"🎯 Slide timing: expected_loop={expected_loop}, offset={self.slide_timing_offset_frames}, "
                                        f"margin={margin_frames}, lead={lead_frames} → threshold={self._current_slide_expected} frames"
                                    )
                                    # Reset signal time so we don't reuse it
                                    self._advance_signal_time = None
                                
                                # Always count one frame consumption for current slide
                                if hasattr(self, '_current_slide_frame_loops'):
                                    self._current_slide_frame_loops += 1
                                    # If we've reached expected frames, signal the capture to advance
                                    try:
                                        if hasattr(self, '_current_slide_expected') and isinstance(self._current_slide_expected, int) and self._current_slide_expected > 0:
                                            if self._current_slide_frame_loops >= self._current_slide_expected:
                                                if self.slide_control_queue is not None:
                                                    try:
                                                        embed_id = self._next_embed_after_current()
                                                    except Exception:
                                                        embed_id = None
                                                    if embed_id:
                                                        # Pause and show embed; defer advance until resume
                                                        await self._pause_for_embed(embed_id)
                                                        self._deferred_advance = True
                                                        self._current_slide_expected = None
                                                    else:
                                                        # Record signal time to measure navigation latency
                                                        self._advance_signal_time = time.time()
                                                        self.slide_control_queue.put_nowait("advance")
                                                        logger.info(f"➡️ Signaled advance after {self._current_slide_frame_loops} frames (threshold {self._current_slide_expected}) for slide {self._current_slide_index} ({self._current_slide_id})")
                                                        # Set expected to None to avoid repeated signals for this slide
                                                        self._current_slide_expected = None
                                    except Exception:
                                        pass
                                
                                if slide_frame_index % 25 == 0:  # Debug every second
                                    queue_size = self.slide_frame_queue.qsize()
                                    logger.info(f"🎬 QUEUE: Using slide frame {slide_frame_index_from_queue} (queue: {queue_size} frames)")
                                    
                            except asyncio.QueueEmpty:
                                slide_frame_available = False
                        
                        # If no frame available from queue, use duplication logic like avatar frames
                        if not slide_frame_available:
                            if hasattr(self, 'last_slide_frame') and self.last_slide_frame is not None:
                                # Duplicate last slide frame for smooth playback
                                slide_frame = self.last_slide_frame.copy()
                                # Count duplication towards current slide loops as well
                                if hasattr(self, '_current_slide_frame_loops'):
                                    self._current_slide_frame_loops += 1
                                    # Check pacing condition on duplicates too
                                    try:
                                        if hasattr(self, '_current_slide_expected') and isinstance(self._current_slide_expected, int) and self._current_slide_expected > 0:
                                            if self._current_slide_frame_loops >= self._current_slide_expected:
                                                if self.slide_control_queue is not None:
                                                    try:
                                                        embed_id = self._next_embed_after_current()
                                                    except Exception:
                                                        embed_id = None
                                                    if embed_id:
                                                        await self._pause_for_embed(embed_id)
                                                        self._deferred_advance = True
                                                        self._current_slide_expected = None
                                                    else:
                                                        self._advance_signal_time = time.time()
                                                        self.slide_control_queue.put_nowait("advance")
                                                        logger.info(f"➡️ Signaled advance after {self._current_slide_frame_loops} frames (threshold {self._current_slide_expected}) for slide {self._current_slide_index} ({self._current_slide_id}) [duplicate]")
                                                        self._current_slide_expected = None
                                    except Exception:
                                        pass
                                if slide_frame_index % 25 == 0:  # Debug every second
                                    queue_size = self.slide_frame_queue.qsize() if self.slide_frame_queue else 0
                                    logger.info(f"🎬 🔄 SLIDE QUEUE EMPTY - duplicating last slide frame (queue: {queue_size})")
                            else:
                                # Fallback to frame processor for initial frames or when no queue available
                                current_frame_count = frame_processor.get_frame_count()
                                if current_frame_count > 0:
                                    safe_slide_index = slide_frame_index % current_frame_count
                                    self.total_slide_frames = current_frame_count
                                else:
                                    safe_slide_index = 0
                                slide_frame = frame_processor.get_slide_frame(safe_slide_index)
                                if slide_frame_index % 25 == 0:  # Debug every second
                                    logger.info(f"🎬 FALLBACK: Using frame processor {safe_slide_index}/{current_frame_count} (no queue frames available)")
                        
                        if slide_frame is not None:
                            # FAST COMPOSITION with performance tracking
                            composition_start = time.time()
                            # Determine presenter-based absolute position and fixed size if available
                            abs_pos = None
                            fixed_sz = None
                            try:
                                current_slide_id = getattr(self, '_current_slide_id', None)
                                presenter = self.presenter_map.get(current_slide_id) if current_slide_id else None
                                # Output size from slide_frame
                                if isinstance(slide_frame, Image.Image):
                                    out_w, out_h = slide_frame.size
                                else:
                                    out_w, out_h = (1280, 720)
                                canvas_w, canvas_h = self.doc_canvas_size
                                # Default: 400x400 at 1080p → scale with output height relative to canvas
                                base_avatar_px = 400
                                scale_ratio = (out_h / float(canvas_h)) if canvas_h else (720.0 / 1080.0)
                                target_edge = max(50, int(round(base_avatar_px * scale_ratio)))
                                fixed_sz = (target_edge, target_edge)
                                if presenter:
                                    left = float(presenter.get('left', 0))
                                    top = float(presenter.get('top', 0))
                                    # Map to output coordinates
                                    x_px = int(left / max(1.0, canvas_w) * out_w)
                                    y_px = int(top / max(1.0, canvas_h) * out_h)
                                    # Nudge downward slightly (scaled to output height)
                                    y_nudge = int(round(24 * (out_h / 720.0)))
                                    y_px = y_px + y_nudge
                                    abs_pos = (x_px, y_px)
                                else:
                                    abs_pos = None
                            except Exception:
                                abs_pos = None
                                fixed_sz = None

                            # Clamp absolute position to keep avatar within bounds if we have both abs_pos and fixed size
                            if abs_pos and fixed_sz and isinstance(slide_frame, Image.Image):
                                out_w, out_h = slide_frame.size
                                ax, ay = abs_pos
                                fw, fh = fixed_sz
                                ax = max(0, min(ax, max(0, out_w - fw)))
                                ay = max(0, min(ay, max(0, out_h - fh)))
                                abs_pos = (ax, ay)

                            composed_frame = frame_processor.overlay_avatar_on_slide(
                                slide_frame=slide_frame,
                                avatar_frame=frame,
                                position="center-bottom",
                                scale=0.85,
                                offset=(0, 0),
                                fixed_size=fixed_sz,
                                absolute_position=abs_pos
                            )
                            composition_duration = (time.time() - composition_start) * 1000
                            
                            # Convert audio for LiveKit and append to PCM archive
                            pcm_audio = audio if isinstance(audio, bytes) else (audio * 32767).astype('int16').tobytes()
                            try:
                                if self._audio_pcm_path:
                                    with open(self._audio_pcm_path, 'ab') as f_pcm:
                                        f_pcm.write(pcm_audio)
                            except Exception as e:
                                logger.debug(f"Failed to append PCM audio: {e}")
                            
                            # Track slow operations
                            max_composition_ms = self.buffer_config['max_composition_time_ms']
                            if composition_duration > max_composition_ms:
                                logger.warning(f"🐌 Slow composition: {composition_duration:.1f}ms (target: <{max_composition_ms}ms)")
                            
                            # Convert composed PIL image to BGR for saving/publishing
                            cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)

                            # Ensure composed frames directory exists
                            try:
                                if self._composed_frames_dir and not os.path.exists(self._composed_frames_dir):
                                    os.makedirs(self._composed_frames_dir, exist_ok=True)
                                    logger.info(f"💾 Composed frames directory: {self._composed_frames_dir}")
                            except Exception as e:
                                logger.warning(f"⚠️ Failed to create composed frames directory: {e}")

                            # Save composed frame to disk
                            try:
                                if self._composed_frames_dir:
                                    frame_path = os.path.join(
                                        self._composed_frames_dir,
                                        f"frame_{self._composed_frame_index:06d}.jpg"
                                    )
                                    # Save as JPEG to reduce size; cv_frame is BGR
                                    cv2.imwrite(frame_path, cv_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                                    self._composed_frame_index += 1
                            except Exception as e:
                                logger.debug(f"Failed to save composed frame: {e}")

                            # Publish to LiveKit at precise timing with performance measurement
                            publish_start = time.time()
                            # Initialize video writer on first frame
                            if self._video_writer is None:
                                try:
                                    os.makedirs(self.output_dir, exist_ok=True)
                                except Exception:
                                    pass
                                ts = int(time.time())
                                self._video_output_path = os.path.join(self.output_dir, f"rapido_stream_{ts}.mp4")
                                height_v, width_v = cv_frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                self._video_writer = cv2.VideoWriter(self._video_output_path, fourcc, target_fps, (width_v, height_v))
                                if self._video_writer and self._video_writer.isOpened():
                                    logger.info(f"🎬 Video writer opened: {self._video_output_path} @ {target_fps}fps, {width_v}x{height_v}")
                                else:
                                    logger.warning("⚠️ Failed to open video writer - final MP4 will not be saved")

                            # Write frame to MP4 if writer is available
                            try:
                                if self._video_writer and self._video_writer.isOpened():
                                    self._video_writer.write(cv_frame)
                            except Exception as e:
                                logger.warning(f"⚠️ Video write failed: {e}")
                            # Skip publishing during embed pause; also discard any intake/slide frames during pause
                            if getattr(self, 'paused_for_embed', False):
                                # While paused, do not publish and do not retain last_frame_data
                                self.last_frame_data = None
                                # Best-effort drain slide queue quickly to prevent growth
                                try:
                                    if self.slide_frame_queue is not None:
                                        drained = 0
                                        while True:
                                            try:
                                                _ = self.slide_frame_queue.get_nowait()
                                                drained += 1
                                            except asyncio.QueueEmpty:
                                                break
                                        if drained:
                                            logger.debug(f"Slide queue drained during pause: {drained}")
                                except Exception:
                                    pass
                            else:
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
                                # Show slide frame queue status instead of cache-based progress
                                slide_queue_size = self.slide_frame_queue.qsize() if self.slide_frame_queue else 0
                                current_slide_info = f"{getattr(self, 'last_slide_frame_index', 'N/A')}" if hasattr(self, 'last_slide_frame_index') else "N/A"
                                queue_indicator = " - QUEUE" if slide_frame_available else " - DUPLICATED" if hasattr(self, 'last_slide_frame') else " - FALLBACK"
                                logger.info(f"📊 Slide progress: Frame {current_slide_info} (queue: {slide_queue_size}, output frame {frame_count}){queue_indicator}")
                            
                            if frame_count % 25 == 0:  # Log every second  
                                elapsed_time = current_time - self.frame_delivery_start_time
                                sustained_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                                fps_diff = sustained_fps - target_fps
                                status = "✅" if abs(fps_diff) < 1.0 else "⚠️"
                                
                                # Buffer status for both avatar and slide frames
                                if intake_size > 30:
                                    avatar_buffer_status = "🟢"  # Plenty of buffer
                                elif intake_size > 10:
                                    avatar_buffer_status = "🟡"  # Getting low but still safe
                                elif intake_size > 0:
                                    avatar_buffer_status = "🟠"  # Very low but still have frames
                                else:
                                    avatar_buffer_status = "🔄"  # Duplicating frames
                                
                                # Slide frame buffer status
                                slide_queue_size = self.slide_frame_queue.qsize() if self.slide_frame_queue else 0
                                if slide_queue_size > 30:
                                    slide_buffer_status = "🟢"
                                elif slide_queue_size > 10:
                                    slide_buffer_status = "🟡"
                                elif slide_queue_size > 0:
                                    slide_buffer_status = "🟠"
                                else:
                                    slide_buffer_status = "🔄"
                                
                                logger.info(f"🎬 {status} SUSTAINED: {sustained_fps:.1f}/{target_fps} FPS (diff: {fps_diff:+.1f}) {avatar_buffer_status} avatar: {intake_size} | {slide_buffer_status} slide: {slide_queue_size}")
                        
                        # Mark tasks as done - frames successfully processed and output
                        # Only mark as done if we actually got frames from queues (not duplicated)
                        if not is_duplicated_frame:
                            self.intake_queue.task_done()
                        
                        # Mark slide frame task as done if we got it from queue
                        if slide_frame_available and self.slide_frame_queue:
                            self.slide_frame_queue.task_done()
                        
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
    
    async def publish_frame_to_livekit(self, frame, audio_chunk):
        """Publish frame and audio to LiveKit - FORCE LEGACY MODE for maximum performance"""
        # FORCE LEGACY MODE - Disable all optimizations for smooth streaming
        # The optimized path was causing InvalidState errors and slow performance
        return await self._publish_frame_to_livekit_legacy(frame, audio_chunk)

    def _finalize_audio_wav(self, sample_rate: int = 16000):
        """Convert concatenated PCM to WAV."""
        try:
            import wave, os
            if not self._audio_pcm_path or not os.path.exists(self._audio_pcm_path):
                return
            with open(self._audio_pcm_path, 'rb') as f_in:
                pcm_bytes = f_in.read()
            with wave.open(self._audio_wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_bytes)
            logger.info(f"🎵 Final WAV written: {self._audio_wav_path}")
        except Exception as e:
            logger.error(f"Failed to finalize WAV: {e}")

    def _stitch_final_video(self, fps: float):
        """Use ffmpeg to stitch frames and WAV into MP4."""
        try:
            import subprocess, os
            if not self._composed_frames_dir or not os.path.isdir(self._composed_frames_dir):
                logger.warning("No composed frames to stitch")
                return
            if not self._audio_wav_path or not os.path.exists(self._audio_wav_path):
                logger.warning("No WAV audio to stitch")
                return
            pattern = os.path.join(self._composed_frames_dir, 'frame_%06d.jpg')
            cmd = [
                'ffmpeg', '-y',
                '-framerate', f'{fps}',
                '-i', pattern,
                '-i', self._audio_wav_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest', self._final_mp4_path
            ]
            logger.info(f"🎞️ ffmpeg stitching → {self._final_mp4_path}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"✅ Final MP4 ready: {self._final_mp4_path}")
        except FileNotFoundError:
            logger.error("ffmpeg not found. Install ffmpeg to enable stitching.")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e}")
    
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
        """Legacy frame and audio publishing - FAST DIRECT MODE"""
        try:
            import livekit.rtc as rtc
            
            if not hasattr(self, 'video_source') or not hasattr(self, 'audio_source'):
                return
            
            # Log once that we're using legacy mode
            if not hasattr(self, '_legacy_mode_logged'):
                logger.info("🔄 Using LEGACY LiveKit publishing mode for maximum performance")
                self._legacy_mode_logged = True
            
            # Convert BGR to RGB for video frame (simple and fast)
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            
            # Create video frame with explicit dimensions
            video_frame = rtc.VideoFrame(
                width=width,
                height=height,
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
                    
                    # Upscale ALL smaller frames to 400x400 for consistent processing and quality
                    original_size = frame_array.shape[:2]
                    if original_size != (400, 400):
                        avatar_frame = avatar_frame.resize((400, 400), Image.LANCZOS)
                        if self._frame_count < 3:
                            logger.info(f"🔍 Upscaling {original_size[0]}x{original_size[1]} → 400x400 using LANCZOS interpolation")
                    elif self._frame_count < 3:
                        logger.info(f"🔍 Frame already 400x400, no upscaling needed")
                    
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
        
        # Initialize slide frame queue for real-time frame capture
        if self.slide_frame_queue is None:
            self.slide_frame_queue = asyncio.Queue(maxsize=200)  # Buffer up to 100 slide frames
            logger.info("📥 Slide frame queue initialized (maxsize: 100)")
        if self.slide_control_queue is None:
            self.slide_control_queue = asyncio.Queue()  # Unbounded control signals
            logger.info("🎚️ Slide control queue initialized for frame-based pacing")
        
        try:
            # Start dynamic capture (direct to queue - no file system)
            from tab_capture.capture_api import capture_presentation_frames_to_queue
            browser_service = await capture_presentation_frames_to_queue(
                capture_url=self.capture_url,
                frame_queue=self.slide_frame_queue,
                video_job_id=self.video_job_id,
                slide_advance_queue=self.slide_control_queue
            )
            
            if not browser_service:
                raise Exception("Failed to start dynamic frame capture - could not initialize browser")
                
            logger.info("✅ Dynamic frame capture started in background")
            logger.info("📥 Frames will be fed directly to queue (no file system)")
            
            # No need for file watcher since frames are fed directly to queue
            # The browser service will populate slide_frame_queue directly
            
            # Wait for first few frames to be queued before proceeding
            logger.info("⏳ Waiting for first slide frames to be queued...")
            await self._wait_for_initial_frames()
            
            # For dynamic capture, we use the queue system with direct feeding
            logger.info("🔄 Dynamic capture ready - using direct queue feeding system")
            
            logger.info("✅ Dynamic frame capture setup completed (non-blocking)")
            # Keep reference for cleanup at shutdown
            self._browser_service = browser_service
            
        except Exception as e:
            logger.error(f"❌ Dynamic frame capture setup failed: {e}")
            raise
    
    async def _wait_for_initial_frames(self):
        """Wait for initial frames to be populated in slide_frame_queue before starting SyncTalk"""
        min_frames_needed = 1
        max_wait_time = 8  # seconds
        
        if not self.slide_frame_queue:
            raise Exception("slide_frame_queue not initialized - cannot wait for frames")
        
        logger.info(f"⏳ Waiting for {min_frames_needed} frames to be queued...")
        
        start_time = time.time()
        while (time.time() - start_time) < max_wait_time:
            queue_size = self.slide_frame_queue.qsize()
            
            if queue_size >= min_frames_needed:
                logger.info(f"✅ Found {queue_size} frames in queue, ready to start SyncTalk!")
                return
            
            elapsed = time.time() - start_time
            logger.info(f"⏳ Queue has {queue_size}/{min_frames_needed} frames - {elapsed:.1f}s elapsed")
            await asyncio.sleep(1.0)
        
        # Timeout - check final count
        final_queue_size = self.slide_frame_queue.qsize()
        raise Exception(f"Timeout waiting for initial frames after {max_wait_time}s - only got {final_queue_size}/{min_frames_needed}")
    
    async def _watch_and_queue_frames(self):
        """
        DEPRECATED: Real-time file watcher that immediately queues frames as they are created in the capture directory.
        This method is no longer used since we now feed frames directly from browser to queue.
        Uses polling with minimal delay to catch frames as soon as they're written.
        """
        logger.warning("⚠️ _watch_and_queue_frames is deprecated - using direct queue feeding instead")
        return
        
        if not self.dynamic_frames_dir:
            logger.error("❌ No dynamic frames directory set")
            return
            
        logger.info(f"🎬 Starting real-time frame watcher for: {self.dynamic_frames_dir}")
        frames_dir = Path(self.dynamic_frames_dir)
        
        processed_frames = set()  # Track which frames we've already processed
        frame_index_counter = 0  # Sequential counter for proper indexing
        last_frame_count = 0  # Track directory changes
        
        while self.frame_delivery_running:
            try:
                # Find all PNG files in the directory with fast detection
                if frames_dir.exists():
                    frame_files = sorted(list(frames_dir.glob("frame_*.png")))
                    current_frame_count = len(frame_files)
                    
                    # Only process if we have new frames (performance optimization)
                    if current_frame_count > last_frame_count:
                        # Process only the new frames
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

                                    if frame_image is not None:                                    
                                        # Put frame in slide_frame_queue instead of cache
                                        try:
                                            self.slide_frame_queue.put_nowait((frame_index_counter, frame_image))
                                            self.slide_frame_count = frame_index_counter + 1
                                            
                                        except asyncio.QueueFull:
                                            # If queue is full, remove oldest frame and add new one
                                            try:
                                                self.slide_frame_queue.get_nowait()  # Remove oldest
                                                self.slide_frame_queue.put_nowait((frame_index_counter, frame_image))
                                            except asyncio.QueueEmpty:
                                                pass
                                        
                                        # Add to processed set
                                        processed_frames.add(frame_file.name)
                                        frame_index_counter += 1
                                        
                                        # Log progress every 25 frames
                                        if frame_index_counter % 25 == 0:
                                            queue_size = self.slide_frame_queue.qsize()
                                            logger.info(f"📈 Real-time watcher: {frame_index_counter} frames queued (queue size: {queue_size})")
                                    else:
                                        logger.warning(f"⚠️ Could not load frame: {frame_file}")
                                        processed_frames.add(frame_file.name)
                                    
                                except Exception as e:
                                    logger.error(f"❌ Error loading slide frame {frame_file}: {e}")
                                    # Mark as processed to avoid infinite retry, but don't increment counter
                                    processed_frames.add(frame_file.name)
                                    # Add retry logic for network/IO issues
                                    if "Permission denied" in str(e) or "being used by another process" in str(e):
                                        logger.info(f"🔄 Will retry {frame_file.name} in next cycle")
                                        processed_frames.discard(frame_file.name)  # Allow retry
                        
                        # Update last frame count after processing
                        last_frame_count = current_frame_count
                
                # Fast polling for real-time detection (25ms = 40fps detection rate)
                await asyncio.sleep(0.025)
                
            except Exception as e:
                logger.error(f"❌ Error in slide frame producer: {e}")
                await asyncio.sleep(1.0)  # Wait longer on error
                
        logger.info("🛑 Real-time frame watcher stopped")

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
    
    async def send_stream_event_to_frontend(self, event_type: str, **event_data):
        """Send stream events to frontend via LiveKit data channel"""
        try:
            if not hasattr(self, 'lk_room') or not self.lk_room:
                logger.warning("⚠️ Cannot send event - LiveKit room not available")
                return
            
            # Create event in same format as frontend expects
            event_message = {
                "id": f"avatar_event_{int(time.time() * 1000)}",
                "event": event_type,
                "timestamp": int(time.time() * 1000),
                **event_data
            }
            
            # Encode same way as frontend  
            import json
            data = json.dumps(event_message).encode('utf-8')
            
            # Send to all participants via data channel on "control" topic (same as frontend listens)
            await self.lk_room.local_participant.publish_data(data, topic="control")
            
            logger.info(f"📤 Sent event to frontend: {event_type} (id: {event_message['id']})")
            
        except Exception as e:
            logger.error(f"Failed to send stream event: {e}")
    
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
            # Note: No slide producer task cleanup needed since browser service handles frame feeding directly
            
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
    script_dir = Path(__file__).parent.parent  # rapido_system/api -> rapido_system  
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
