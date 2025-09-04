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

class RapidoMainSystem:
    """Integrated Rapido system with all features"""
    
    def __init__(self, config_override: dict = None):
        self.config = Config()
        
        # Apply config overrides
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        # Setup paths and connections
        self.synctalk_url = getattr(self.config, 'SYNCTALK_WEBSOCKET_URL', 'ws://localhost:8001')
        self.websocket = None
        self.avatar_frames = []
        self.avatar_audio_chunks = []
        self.slide_frames_path = getattr(self.config, 'SLIDE_FRAMES_PATH', '../frames')
        self.output_dir = getattr(self.config, 'OUTPUT_PATH', './output')
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Timing for smooth video
        self.total_slide_frames = 306
        self.synctalk_fps = 25.0
        self.target_video_fps = 25.0
        self.total_duration_seconds = self.total_slide_frames / self.synctalk_fps
        
        logger.info(f"📊 Rapido initialized - Duration: {self.total_duration_seconds:.2f}s")
        
    async def connect_to_synctalk(self, avatar_name="enrique_torres", sample_rate=16000):
        """Connect to SyncTalk with protobuf support"""
        ws_url = f"{self.synctalk_url}/audio_to_video?avatar_name={avatar_name}&sample_rate={sample_rate}"
        logger.info(f"🔌 Connecting to SyncTalk: {ws_url}")
        
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
            
            # Step 2: Generate audio
            api_key = getattr(self.config, 'ELEVENLABS_API_KEY', None)
            if api_key:
                logger.info("🎵 Using ElevenLabs TTS")
                tts_client = ElevenLabsTTSClient(api_key=api_key)
                real_audio = await tts_client.generate_full_audio(narration_text)
                logger.info(f"Generated {len(real_audio)} bytes of real TTS")
                
                # Save MP3 and convert to PCM using librosa
                try:
                    import tempfile
                    import librosa
                    import soundfile as sf
                    
                    # Save MP3 to temp file
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                        temp_mp3.write(real_audio)
                        temp_mp3_path = temp_mp3.name
                    
                    # Load and resample to 16kHz
                    audio_array, sr = librosa.load(temp_mp3_path, sr=16000, mono=True)
                    audio_data = (audio_array * 32767).astype('int16').tobytes()
                    
                    # Cleanup
                    import os
                    os.unlink(temp_mp3_path)
                    
                    logger.info(f"🎵 Using real TTS audio: {len(audio_data)} bytes PCM")
                except Exception as e:
                    logger.warning(f"Audio conversion failed: {e}, using synthetic audio")
                    audio_data = self.create_speech_audio(self.total_duration_seconds, 16000)
            else:
                logger.info(f"🎵 Creating {self.total_duration_seconds:.2f}s of synthetic audio...")
                audio_data = self.create_speech_audio(self.total_duration_seconds, 16000)
            
            # Step 3: Connect to SyncTalk
            logger.info("🔌 Connecting to SyncTalk...")
            if not await self.connect_to_synctalk():
                raise Exception("SyncTalk connection failed")
            
            # Step 4: Stream audio and collect frames
            logger.info("🎭 Streaming audio and collecting avatar frames...")
            
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
            
            # Step 5: Frame composition
            logger.info("🖼️ Initializing frame composition...")
            frame_processor = FrameOverlayEngine(self.slide_frames_path, output_size=(1280, 720))
            
            logger.info("🖼️ Compositing frames...")
            composed_frames = []
            video_frames = []
            
            # Process all frames to match audio duration
            avatar_count = len(self.avatar_frames)
            num_frames = min(self.total_slide_frames, avatar_count)  # Use all available frames
            logger.info(f"🎬 Creating video with {num_frames} frames for proper audio sync")
            
            for i in range(num_frames):
                slide_frame = frame_processor.get_slide_frame(i)
                if not slide_frame:
                    continue
                
                avatar_index = i % avatar_count if avatar_count > 0 else 0
                avatar_frame = self.avatar_frames[avatar_index]
                
                # Apply chroma key with despill to prevent background meshing
                avatar_frame_clean = self.remove_green_screen_with_despill(avatar_frame)
                
                # Calculate center offset for bottom positioning
                slide_width = slide_frame.width
                avatar_width = int(avatar_frame_clean.width * 0.8)
                center_x_offset = (slide_width - avatar_width) // 2
                
                composed_frame = frame_processor.overlay_avatar_on_slide(
                    slide_frame=slide_frame,
                    avatar_frame=avatar_frame_clean,
                    position="bottom-left",  # Bottom left, but we'll offset to center
                    scale=0.8,  # Bigger avatar
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
    """Verify SyncTalk server is running"""
    try:
        response = requests.get("http://localhost:8001/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ SyncTalk ready: {status['status']}")
            print(f"🤖 Models: {status['loaded_models']}")
            return "Alin-cc-dataset" in status['loaded_models']
        return False
    except Exception as e:
        print(f"❌ SyncTalk server error: {e}")
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
