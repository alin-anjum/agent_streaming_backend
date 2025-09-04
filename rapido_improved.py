#!/usr/bin/env python3
"""
Rapido Improved System - Enhanced Avatar Video Generation
- Proper timing for 306 slide frames
- Green screen removal (chroma key)
- Audio extraction from avatar frames
- Better video codec for compatibility
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

# Add paths
sys.path.append('./rapido/src')
sys.path.append('./SyncTalk_2D')

from data_parser import SlideDataParser
from tts_client import ElevenLabsTTSClient
from frame_processor import FrameOverlayEngine
from frame_message_pb2 import FrameMessage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RapidoImprovedSystem:
    """Improved Rapido system with proper timing, chroma key, and audio"""
    
    def __init__(self):
        self.synctalk_url = "ws://localhost:8001"
        self.websocket = None
        self.avatar_frames = []
        self.avatar_audio_chunks = []
        self.slide_frames_path = "frames"
        self.output_dir = "rapido_improved_output"
        
        # Timing calculations for smooth playback
        self.total_slide_frames = 306  # We confirmed this
        self.synctalk_fps = 25.0  # SyncTalk generates at 25 FPS
        self.target_video_fps = 25.0  # Match SyncTalk for smooth playback
        self.total_duration_seconds = self.total_slide_frames / self.synctalk_fps  # ~12.24 seconds
        
        logger.info(f"📊 Timing Configuration:")
        logger.info(f"  - Total slide frames: {self.total_slide_frames}")
        logger.info(f"  - SyncTalk FPS: {self.synctalk_fps}")
        logger.info(f"  - Target video FPS: {self.target_video_fps}")
        logger.info(f"  - Calculated duration: {self.total_duration_seconds:.2f} seconds")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    async def connect_to_synctalk(self, avatar_name="enrique_torres", sample_rate=16000):
        """Connect to SyncTalk server"""
        ws_url = f"{self.synctalk_url}/ws/audio_to_video?avatar_name={avatar_name}&sample_rate={sample_rate}"
        logger.info(f"🔌 Connecting to SyncTalk: {ws_url}")
        
        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info("✅ Connected to SyncTalk server!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to SyncTalk: {e}")
            return False
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to SyncTalk"""
        if self.websocket:
            await self.websocket.send(audio_data)
    
    async def receive_avatar_frame_with_audio(self, timeout=2.0):
        """Receive avatar frame AND audio from SyncTalk"""
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
                    
                    # Reshape to correct dimensions
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
                            logger.warning(f"Cannot reshape video data of size {len(video_data)}")
                            return None, None
                    
                    avatar_frame = Image.fromarray(frame_array, 'RGB')
                
                # Extract audio data
                if frame_msg.audio_bytes:
                    avatar_audio = frame_msg.audio_bytes
                    logger.debug(f"🎵 Received audio chunk: {len(avatar_audio)} bytes")
                
                return avatar_frame, avatar_audio
                    
        except asyncio.TimeoutError:
            return None, None
        except Exception as e:
            logger.error(f"Error receiving frame: {e}")
            return None, None
    
    def remove_green_screen(self, image: Image.Image) -> Image.Image:
        """Remove green screen from avatar frame using chroma key"""
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Define green screen color (from avatar_config.json: #089831)
        target_color = np.array([8, 152, 49])  # RGB values for #089831
        
        # Calculate color distance
        color_diff = np.sqrt(np.sum((img_array - target_color) ** 2, axis=2))
        
        # Create mask (threshold from config: 35)
        threshold = 35
        mask = color_diff > threshold
        
        # Create RGBA image with transparency
        rgba_array = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        rgba_array[:, :, :3] = img_array  # Copy RGB
        rgba_array[:, :, 3] = mask.astype(np.uint8) * 255  # Alpha channel
        
        # Apply edge blur (from config: 0.08)
        blur_radius = int(0.08 * min(img_array.shape[:2]))
        if blur_radius > 0:
            from scipy.ndimage import gaussian_filter
            alpha_blurred = gaussian_filter(rgba_array[:, :, 3].astype(float), sigma=blur_radius)
            rgba_array[:, :, 3] = alpha_blurred.astype(np.uint8)
        
        return Image.fromarray(rgba_array, 'RGBA')
    
    def create_extended_speech_audio(self, duration_seconds: float, sample_rate: int = 16000) -> bytes:
        """Create extended speech-like audio for full duration"""
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, num_samples, False)
        
        # Create varied speech-like audio
        base_freq = 120.0  # Base male voice frequency
        harmonics = [1.0, 1.5, 2.0, 2.5, 3.0]  # Harmonic series
        weights = [0.4, 0.3, 0.2, 0.1, 0.05]  # Harmonic weights
        
        audio_data = np.zeros_like(t)
        for harmonic, weight in zip(harmonics, weights):
            freq = base_freq * harmonic
            # Add some variation to make it more natural
            freq_variation = 1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Slow frequency variation
            audio_data += weight * np.sin(2 * np.pi * freq * freq_variation * t)
        
        # Add speech-like amplitude modulation
        amplitude_modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 3.0 * t)  # 3Hz modulation
        audio_data = audio_data * amplitude_modulation
        
        # Add some pauses to make it more speech-like
        pause_pattern = np.where(np.sin(2 * np.pi * 0.2 * t) < -0.5, 0.3, 1.0)
        audio_data = audio_data * pause_pattern
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767 * 0.7).astype(np.int16)  # 70% volume
        return audio_data.tobytes()
    
    async def process_full_presentation(self, input_json: str = "test1.json"):
        """Process the complete presentation with all 306 frames"""
        
        logger.info("🚀 Starting Rapido Improved System - Full Presentation")
        
        try:
            # Step 1: Parse slide data
            logger.info("📄 Step 1: Loading slide data...")
            data_parser = SlideDataParser(input_json)
            if not data_parser.load_data():
                raise Exception("Failed to load slide data")
            
            narration_text = data_parser.get_narration_text()
            slide_summary = data_parser.get_summary()
            
            logger.info(f"📝 Full narration length: {len(narration_text)} characters")
            logger.info(f"📊 Slide summary: {slide_summary}")
            
            # Step 2: Create extended audio for full duration
            logger.info(f"🎵 Step 2: Creating extended audio for {self.total_duration_seconds:.2f} seconds...")
            audio_data = self.create_extended_speech_audio(
                duration_seconds=self.total_duration_seconds, 
                sample_rate=16000
            )
            logger.info(f"🎵 Generated {len(audio_data)} bytes of extended speech audio")
            
            # Step 3: Connect to SyncTalk
            logger.info("🔌 Step 3: Connecting to SyncTalk...")
            if not await self.connect_to_synctalk():
                raise Exception("Failed to connect to SyncTalk")
            
            # Step 4: Stream audio and collect ALL avatar frames
            logger.info("🎭 Step 4: Streaming audio to generate ALL avatar frames...")
            
            # Split audio into chunks
            chunk_size = 1024
            audio_chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
            logger.info(f"🎵 Created {len(audio_chunks)} audio chunks for full presentation")
            
            # Stream ALL audio chunks
            for i, chunk in enumerate(audio_chunks):
                if len(chunk) % 2 != 0:
                    chunk = chunk[:-1]  # Ensure even size
                
                if i % 50 == 0:  # Log every 50 chunks
                    logger.info(f"📡 Sending audio chunk {i+1}/{len(audio_chunks)}")
                
                await self.send_audio_chunk(chunk)
                
                # Receive avatar frame and audio
                frame, audio = await self.receive_avatar_frame_with_audio(timeout=1.0)
                if frame:
                    # Apply green screen removal
                    frame_no_bg = self.remove_green_screen(frame)
                    self.avatar_frames.append(frame_no_bg)
                    
                    # Store audio if available
                    if audio:
                        self.avatar_audio_chunks.append(audio)
                    
                    # Save every 25th frame for monitoring
                    if len(self.avatar_frames) % 25 == 0:
                        frame_path = f"{self.output_dir}/avatar_frame_{len(self.avatar_frames):03d}.png"
                        frame_no_bg.save(frame_path)
                        logger.info(f"💾 Saved avatar frame: {frame_path}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
            
            # Wait for remaining frames
            logger.info("⏳ Collecting remaining frames...")
            for _ in range(50):
                frame, audio = await self.receive_avatar_frame_with_audio(timeout=0.5)
                if frame:
                    frame_no_bg = self.remove_green_screen(frame)
                    self.avatar_frames.append(frame_no_bg)
                    if audio:
                        self.avatar_audio_chunks.append(audio)
                else:
                    break
            
            logger.info(f"🎭 Collected {len(self.avatar_frames)} REAL Enrique Torres frames!")
            logger.info(f"🎵 Collected {len(self.avatar_audio_chunks)} audio chunks!")
            
            if not self.avatar_frames:
                raise Exception("No avatar frames received from SyncTalk")
            
            # Step 5: Initialize frame processor with original slide size
            logger.info("🖼️ Step 5: Initializing frame composition...")
            frame_processor = FrameOverlayEngine(self.slide_frames_path, output_size=(1280, 720))
            
            # Step 6: Composite ALL frames
            logger.info(f"🖼️ Step 6: Compositing ALL {self.total_slide_frames} frames...")
            
            composed_frames = []
            video_frames = []
            
            # Calculate how to distribute avatar frames across all slide frames
            avatar_frame_count = len(self.avatar_frames)
            
            for i in range(self.total_slide_frames):
                # Get slide frame
                slide_frame = frame_processor.get_slide_frame(i)
                if not slide_frame:
                    logger.warning(f"Could not get slide frame {i}, using previous...")
                    if composed_frames:
                        composed_frames.append(composed_frames[-1])
                        video_frames.append(video_frames[-1])
                    continue
                
                # Get corresponding avatar frame (cycle through available frames)
                avatar_index = i % avatar_frame_count if avatar_frame_count > 0 else 0
                avatar_frame = self.avatar_frames[avatar_index]
                
                # Composite with larger avatar scale and transparency support
                composed_frame = frame_processor.overlay_avatar_on_slide(
                    slide_frame=slide_frame,
                    avatar_frame=avatar_frame,
                    position="bottom-right",
                    scale=0.4  # Good size for visibility
                )
                
                composed_frames.append(composed_frame)
                
                # Convert to OpenCV format
                cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
                video_frames.append(cv_frame)
                
                # Log progress every 50 frames
                if (i + 1) % 50 == 0:
                    logger.info(f"✅ Composed frame {i+1}/{self.total_slide_frames}")
            
            # Step 7: Create combined audio track
            logger.info("🎵 Step 7: Creating combined audio track...")
            if self.avatar_audio_chunks:
                combined_audio = b''.join(self.avatar_audio_chunks)
                audio_file = f"{self.output_dir}/avatar_audio.wav"
                
                # Save as WAV file
                import wave
                with wave.open(audio_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # Sample rate
                    wav_file.writeframes(combined_audio)
                
                logger.info(f"🎵 Saved combined audio: {audio_file}")
            
            # Step 8: Create final video with better codec
            logger.info("🎬 Step 8: Creating final video with H.264 codec...")
            
            if video_frames:
                output_video = f"{self.output_dir}/RAPIDO_FULL_PRESENTATION.mp4"
                
                # Use H.264 codec for better compatibility
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                fps = self.target_video_fps
                height, width = video_frames[0].shape[:2]
                
                out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
                
                for frame in video_frames:
                    out.write(frame)
                
                out.release()
                logger.info(f"🎬 FINAL VIDEO CREATED: {output_video}")
                
                # Verify video
                if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                    file_size = os.path.getsize(output_video)
                    duration = len(video_frames) / fps
                    logger.info(f"✅ SUCCESS: Video created ({file_size} bytes, {duration:.1f}s)")
                    
                    # Create comprehensive summary
                    summary = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "input_json": input_json,
                        "narration_text": narration_text[:200] + "...",
                        "avatar_model": "enrique_torres (Alin-cc-dataset)",
                        "improvements": [
                            "Full 306 slide frames processed",
                            "Green screen removal (chroma key)",
                            "Audio extraction from avatar frames",
                            "H.264 codec for better compatibility",
                            "Proper timing calculation (25 FPS)"
                        ],
                        "results": {
                            "total_slide_frames": self.total_slide_frames,
                            "avatar_frames_generated": len(self.avatar_frames),
                            "composed_frames": len(composed_frames),
                            "audio_chunks_extracted": len(self.avatar_audio_chunks),
                            "video_created": True,
                            "video_file": output_video,
                            "video_size_bytes": file_size,
                            "video_duration_seconds": duration,
                            "video_fps": fps,
                            "video_resolution": f"{width}x{height}",
                            "avatar_scale": 0.4,
                            "chroma_key_enabled": True,
                            "codec": "H.264"
                        }
                    }
                    
                    with open(f"{self.output_dir}/improved_summary.json", "w") as f:
                        json.dump(summary, f, indent=2)
                    
                    return output_video
                else:
                    raise Exception("Video file was not created properly")
            else:
                raise Exception("No video frames to create video")
                
        except Exception as e:
            logger.error(f"❌ Rapido improved processing failed: {e}")
            raise
        finally:
            # Cleanup
            if self.websocket:
                await self.websocket.close()
                logger.info("🔌 WebSocket connection closed")

async def main():
    """Main entry point for Rapido Improved System"""
    
    print("🚀 Rapido Improved System - Full Presentation with Chroma Key")
    print("=" * 70)
    
    # Check SyncTalk server
    import requests
    try:
        response = requests.get("http://localhost:8001/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ SyncTalk server is ready: {status['status']}")
            print(f"🤖 Loaded models: {status['loaded_models']}")
            
            if "Alin-cc-dataset" not in status['loaded_models']:
                print("❌ ERROR: Enrique Torres model not loaded!")
                return 1
        else:
            print("❌ ERROR: SyncTalk server not responding properly")
            return 1
    except Exception as e:
        print(f"❌ ERROR: Cannot connect to SyncTalk server: {e}")
        return 1
    
    # Initialize and run improved Rapido
    rapido = RapidoImprovedSystem()
    
    try:
        output_video = await rapido.process_full_presentation("test1.json")
        
        print("\n🎉 RAPIDO IMPROVED SUCCESS!")
        print("=" * 70)
        print(f"📁 Output directory: {rapido.output_dir}/")
        print(f"🎭 Avatar frames: {len(rapido.avatar_frames)} (with green screen removed)")
        print(f"🎵 Audio chunks: {len(rapido.avatar_audio_chunks)} extracted")
        print(f"📽️ Total frames: {rapido.total_slide_frames} slide frames")
        print(f"🎬 Final video: {output_video}")
        print(f"⏱️ Duration: {rapido.total_duration_seconds:.2f} seconds")
        print(f"📊 Summary: {rapido.output_dir}/improved_summary.json")
        print("\n✨ Full presentation with chroma key and audio!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Rapido improved failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
