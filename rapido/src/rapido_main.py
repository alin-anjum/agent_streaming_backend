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
    from chroma_key import FastChromaKey
    CHROMA_KEY_AVAILABLE = True
    logger.info("✅ SyncTalk chroma key module imported successfully")
except ImportError as e:
    CHROMA_KEY_AVAILABLE = False
    logger.warning(f"❌ SyncTalk chroma key not available: {e}")

class RapidoMainSystem:
    """Integrated Rapido system with all features"""
    
    def __init__(self, config_override: dict = None):
        self.config = Config()
        
        # Apply config overrides
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        # Setup paths and connections - use remote SyncTalk server
        self.synctalk_url = getattr(self.config, 'SYNCTALK_WEBSOCKET_URL', 'ws://34.172.49.60:8000')
        self.websocket = None
        self.avatar_frames = []
        self.avatar_audio_chunks = []
        self.slide_frames_path = getattr(self.config, 'SLIDE_FRAMES_PATH', '../frames')
        self.output_dir = getattr(self.config, 'OUTPUT_PATH', './output')
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Timing for smooth video - get actual slide count
        self.synctalk_fps = 25.0
        self.target_video_fps = 25.0
        
        # Initialize frame processor once and reuse - 480p for ultra smooth performance
        self.frame_processor = FrameOverlayEngine('../presentation_frames', output_size=(854, 480))
        self.total_slide_frames = self.frame_processor.get_frame_count()
        self.total_duration_seconds = self.total_slide_frames / self.synctalk_fps
        
        # Initialize chroma key processor
        self.chroma_key_processor = None
        self._init_chroma_key()

        # Performance optimization: cache morphological kernel
        self._morph_kernel = None
        self._frame_count = 0
        
        logger.info(f"📊 Rapido initialized - Duration: {self.total_duration_seconds:.2f}s")
        
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
            bg_url = "https://raw.githubusercontent.com/vinthony/SyncTalk/main/data/enrique_torres/background.jpg"
            
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
            
            # Initialize FastChromaKey with SyncTalk's settings
            self.chroma_key_processor = FastChromaKey(
                background_image=bg_image,
                target_color=(8, 152, 49),  # SyncTalk's #089831 green
                color_threshold=80,
                edge_blur=5
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
        
        img_array = np.array(image, dtype=np.uint8)
        height, width = img_array.shape[:2]
        
        # SyncTalk's green screen color: #089831 = (8, 152, 49)
        target_color = np.array([8, 152, 49], dtype=np.uint8)
        
        # Extract RGB channels
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Green dominance detection (vectorized)
        green_dominant = (g > r + 40) & (g > b + 40)
        
        # Color similarity to target green
        color_diff = np.sqrt(
            (r.astype(np.int16) - target_color[0]) ** 2 +
            (g.astype(np.int16) - target_color[1]) ** 2 +
            (b.astype(np.int16) - target_color[2]) ** 2
        )
        color_similar = color_diff < 80
        
        # Value range check
        value_range = (g > 100) & (g < 200)
        
        # Combine conditions
        is_green = green_dominant & color_similar & value_range
        
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
        
        # Add frame buffer for smooth output with continuous delivery
        self.frame_buffer = []
        self.audio_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        # Frame delivery system for smooth playback
        self.frame_delivery_running = True
        self.frame_delivery_task = None
        
        # SyncTalk frame production monitoring
        self.synctalk_frame_count = 0
        self.synctalk_start_time = None
        
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
                
                # Send to SyncTalk immediately
                await self.send_audio_chunk(pcm_chunk)
                
                # Collect avatar frame with longer timeout (40ms chunk = 1 frame)
                frame, audio = await self.receive_avatar_frame_with_audio(timeout=2.0)
                if frame and audio:
                    # MONITOR SYNCTALK PRODUCTION RATE
                    self.synctalk_frame_count += 1
                    if self.synctalk_start_time is None:
                        self.synctalk_start_time = time.time()
                    
                    # Log SyncTalk production rate every 25 frames
                    if self.synctalk_frame_count % 25 == 0:
                        elapsed = time.time() - self.synctalk_start_time
                        synctalk_fps = self.synctalk_frame_count / elapsed
                        logger.info(f"🤖 SYNCTALK PRODUCTION: {synctalk_fps:.1f} FPS (should be ~25 FPS)")
                    
                    # Add to buffer for smooth continuous delivery
                    async with self.buffer_lock:
                        self.frame_buffer.append(frame)
                        self.audio_buffer.append(audio)
                    
                    # Still collect for fallback
                    self.avatar_frames.append(frame)
                    self.avatar_audio_chunks.append(audio)
                    
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
        
        # Send end marker to SyncTalk to process remaining buffers
        logger.info("📡 Sending end marker to SyncTalk...")
        try:
            await self.websocket.send(b"end_of_stream")
            logger.info("✅ End marker sent")
        except Exception as e:
            logger.warning(f"Failed to send end marker: {e}")
        
        # Wait longer for SyncTalk to process remaining buffers
        logger.info("⏳ Waiting for SyncTalk to process remaining audio buffers...")
        await asyncio.sleep(10.0)  # Wait 10 seconds for remaining processing
        
        # Stop frame delivery and process remaining frames
        self.frame_delivery_running = False
        if self.frame_delivery_task:
            await self.frame_delivery_task
        
        return True
    
    async def continuous_frame_delivery(self, frame_processor):
        """Continuously deliver frames at exactly 25 FPS for smooth playback"""
        try:
            frame_interval = 1.0 / 25.0  # 40ms per frame
            slide_frame_index = 0
            frame_count = 0
            last_frame_time = time.time()
            
            logger.info("🎬 Continuous frame delivery started - 25 FPS TARGET")
            self.frame_delivery_start_time = time.time()
            
            while self.frame_delivery_running:
                current_time = time.time()
                
                # Check if it's time for next frame (every 40ms)
                if current_time - last_frame_time >= frame_interval:
                    # Get frame from buffer
                    frame = None
                    audio = None
                    
                    async with self.buffer_lock:
                        if self.frame_buffer and self.audio_buffer:
                            frame = self.frame_buffer.pop(0)
                            audio = self.audio_buffer.pop(0)
                    
                    if frame and audio:
                        # Get current slide frame with proper cycling
                        safe_slide_index = slide_frame_index % self.total_slide_frames
                        slide_frame = frame_processor.get_slide_frame(safe_slide_index)
                        
                        if slide_frame:
                            # FAST COMPOSITION - minimal processing with smaller avatar
                            slide_width = slide_frame.width
                            avatar_width = int(frame.width * 0.6)  # Reduced from 0.8 to 0.6
                            center_x_offset = (slide_width - avatar_width) // 2
                            
                            # Compose frame with smaller avatar
                            composed_frame = frame_processor.overlay_avatar_on_slide(
                                slide_frame=slide_frame,
                                avatar_frame=frame,
                                position="bottom-left",
                                scale=0.6,  # Reduced from 0.8 to 0.6
                                offset=(center_x_offset, 0)
                            )
                            
                            # Convert to BGR for video publishing
                            cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
                            
                            # Convert audio for LiveKit
                            pcm_audio = audio if isinstance(audio, bytes) else (audio * 32767).astype('int16').tobytes()
                            
                            # Publish to LiveKit
                            await self.publish_frame_to_livekit(cv_frame, pcm_audio)
                            
                            # Advance counters - increment without cycling (cycling handled in safe_slide_index)
                            slide_frame_index = slide_frame_index + 1
                            frame_count += 1
                            last_frame_time = current_time
                            
                            # Log slide progress with cycling info
                            if frame_count % 50 == 0:
                                cycle_count = slide_frame_index // self.total_slide_frames
                                current_slide = safe_slide_index
                                logger.info(f"📊 Slide progress: {current_slide}/{self.total_slide_frames} (cycle {cycle_count + 1}, frame {frame_count})")
                            
                            if frame_count % 25 == 0:  # Log every second
                                buffer_size = len(self.frame_buffer)
                                elapsed_time = current_time - self.frame_delivery_start_time
                                actual_fps = frame_count / elapsed_time
                                target_fps = 25.0
                                fps_diff = actual_fps - target_fps
                                status = "✅" if abs(fps_diff) < 1.0 else "⚠️"
                                logger.info(f"🎬 {status} FPS: {actual_fps:.1f}/{target_fps} (diff: {fps_diff:+.1f}), buffer: {buffer_size}")
                
                # Adaptive sleep based on buffer status
                buffer_size = len(self.frame_buffer)
                if buffer_size == 0:
                    await asyncio.sleep(0.01)  # 10ms when empty - wait for frames
                elif buffer_size > 50:
                    await asyncio.sleep(0.005)  # 5ms when full - can process faster
                else:
                    await asyncio.sleep(0.001)  # 1ms normal operation
                
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
                    
                    # Apply green screen removal for transparency
                    if self.chroma_key_processor is not None or CHROMA_KEY_AVAILABLE:
                        try:
                            avatar_frame = self._remove_green_screen_for_transparency(avatar_frame)
                            # Performance logging (minimal)
                            if self._frame_count % 100 == 0:
                                logger.debug("✅ Green screen removed, transparency applied")
                        except Exception as e:
                            logger.warning(f"Green screen removal failed, using original frame: {e}")
                
                # Extract audio
                if frame_msg.audio_bytes:
                    avatar_audio = frame_msg.audio_bytes
                
                return avatar_frame, avatar_audio
                    
        except asyncio.TimeoutError:
            return None, None
        except Exception as e:
            logger.error(f"Frame receive error: {e}")
            return None, None
    
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
        
        # Calculate color difference
        color_diff = np.sqrt(np.sum((img_array - target_color) ** 2, axis=2))
        
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
            if not await self.connect_livekit():
                raise Exception("LiveKit connection failed")
            
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
                
                # Calculate center offset for bottom positioning with smaller avatar
                slide_width = slide_frame.width
                avatar_width = int(avatar_frame_clean.width * 0.6)  # Reduced from 0.8 to 0.6
                center_x_offset = (slide_width - avatar_width) // 2
                
                composed_frame = frame_processor.overlay_avatar_on_slide(
                    slide_frame=slide_frame,
                    avatar_frame=avatar_frame_clean,
                    position="bottom-left",  # Bottom left, but we'll offset to center
                    scale=0.6,  # Smaller avatar (reduced from 0.8)
                    offset=(center_x_offset, 0)  # Center horizontally, no bottom offset
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
                
                fourcc = cv2.VideoWriter_fourcc(*'H264')  # Better compatibility
                fps = self.target_video_fps
                height, width = video_frames[0].shape[:2]
                
                out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
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
    """Verify LOCAL SyncTalk server is running"""
    try:
        response = requests.get("http://localhost:8001/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ LOCAL SyncTalk ready (ULTRA OPTIMIZED): {status['status']}")
            print(f"🤖 Models: {status['loaded_models']}")
            return "Alin-cc-dataset" in status['loaded_models']
        return False
    except Exception as e:
        print(f"❌ LOCAL SyncTalk server error: {e}")
        print("Make sure your LOCAL SyncTalk server is running on localhost:8001")
        return False

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Rapido - Avatar Video Generation")
    parser.add_argument("--input", "-i", default="../test1.json", help="Input JSON file")
    parser.add_argument("--frames", "-f", default="../frames", help="Slide frames directory")
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
