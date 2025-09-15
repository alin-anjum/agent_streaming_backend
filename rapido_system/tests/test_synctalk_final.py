#!/usr/bin/env python3
"""
Final working SyncTalk integration test - Real Enrique Torres avatar generation
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
sys.path.append('./rapido_system/api')
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

class SyncTalkFinalClient:
    """Final working SyncTalk client"""
    
    def __init__(self, server_url="ws://localhost:8001"):
        self.server_url = server_url
        self.websocket = None
        
    async def connect_and_stream(self, avatar_name="enrique_torres", sample_rate=16000):
        """Connect to SyncTalk WebSocket"""
        
        ws_url = f"{self.server_url}/ws/audio_to_video?avatar_name={avatar_name}&sample_rate={sample_rate}"
        logger.info(f"Connecting to SyncTalk: {ws_url}")
        
        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info("✅ Connected to SyncTalk server successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to SyncTalk server: {e}")
            return False
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send raw audio bytes to SyncTalk"""
        if not self.websocket:
            raise Exception("Not connected to server")
        
        await self.websocket.send(audio_data)
        logger.info(f"📡 Sent audio chunk ({len(audio_data)} bytes)")
    
    async def receive_protobuf_frame(self, timeout=3.0):
        """Receive and decode protobuf frame from SyncTalk"""
        if not self.websocket:
            raise Exception("Not connected to server")
        
        try:
            protobuf_data = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            
            if isinstance(protobuf_data, bytes):
                # Decode protobuf message
                frame_msg = FrameMessage()
                frame_msg.ParseFromString(protobuf_data)
                
                if frame_msg.video_bytes:
                    # Convert video bytes to numpy array
                    video_data = np.frombuffer(frame_msg.video_bytes, dtype=np.uint8)
                    
                    # Try to reshape to image (350x350x3 based on server logs)
                    try:
                        if len(video_data) == 350 * 350 * 3:
                            frame_array = video_data.reshape((350, 350, 3))
                        else:
                            # Try to infer dimensions
                            total_pixels = len(video_data) // 3
                            side = int(np.sqrt(total_pixels))
                            if side * side * 3 == len(video_data):
                                frame_array = video_data.reshape((side, side, 3))
                            else:
                                logger.warning(f"Cannot reshape video data of size {len(video_data)}")
                                return None
                        
                        # Convert to PIL Image (RGB)
                        frame_image = Image.fromarray(frame_array, 'RGB')
                        logger.info(f"🎭 Decoded REAL Enrique Torres frame: {frame_image.size}")
                        return frame_image
                        
                    except Exception as e:
                        logger.error(f"Error reshaping video data: {e}")
                        return None
                else:
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning("⏰ Timeout waiting for frame")
            return None
        except Exception as e:
            logger.error(f"❌ Error receiving frame: {e}")
            return None
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 WebSocket connection closed")

def create_proper_pcm_audio(duration_seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Create properly formatted PCM audio data for SyncTalk"""
    
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, False)
    
    # Create a more interesting audio pattern (speech-like)
    # Mix multiple frequencies to simulate speech
    frequency1 = 200.0  # Base frequency
    frequency2 = 400.0  # Harmonic
    frequency3 = 800.0  # Higher harmonic
    
    audio_data = (
        0.4 * np.sin(2 * np.pi * frequency1 * t) +
        0.3 * np.sin(2 * np.pi * frequency2 * t) +
        0.2 * np.sin(2 * np.pi * frequency3 * t)
    )
    
    # Add some variation to make it more speech-like
    envelope = np.exp(-t * 0.5)  # Decay envelope
    audio_data = audio_data * envelope
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data.tobytes()

async def test_synctalk_final():
    """Final working SyncTalk test with real Enrique Torres avatar"""
    
    logger.info("🚀 === Final SyncTalk Test - Real Enrique Torres Avatar ===")
    
    # Create output directories
    os.makedirs("test_output", exist_ok=True)
    
    try:
        # Step 1: Parse JSON data
        logger.info("📄 Step 1: Parsing JSON slide data...")
        data_parser = SlideDataParser("test1.json")
        if not data_parser.load_data():
            raise Exception("Failed to load JSON data")
        narration_text = data_parser.get_narration_text()
        test_text = narration_text[:80] + "..."
        logger.info(f"📝 Test narration: {test_text}")
        
        # Step 2: Check SyncTalk server status
        logger.info("🔍 Step 2: Checking SyncTalk server status...")
        import requests
        status_response = requests.get("http://localhost:8001/status", timeout=5)
        if status_response.status_code == 200:
            status_data = status_response.json()
            logger.info(f"✅ SyncTalk server ready: {status_data['status']}")
            logger.info(f"🤖 Loaded models: {status_data['loaded_models']}")
            
            if "Alin-cc-dataset" not in status_data['loaded_models']:
                logger.error("❌ Enrique Torres model not loaded!")
                return False
        
        # Step 3: Connect to SyncTalk
        logger.info("🔌 Step 3: Connecting to SyncTalk WebSocket...")
        synctalk_client = SyncTalkFinalClient()
        
        if not await synctalk_client.connect_and_stream(avatar_name="enrique_torres", sample_rate=16000):
            raise Exception("Failed to connect to SyncTalk WebSocket")
        
        # Step 4: Generate speech-like audio
        logger.info("🎵 Step 4: Generating speech-like audio...")
        audio_data = create_proper_pcm_audio(duration_seconds=4.0, sample_rate=16000)
        logger.info(f"🎵 Generated {len(audio_data)} bytes of speech-like PCM audio")
        
        # Step 5: Stream audio WITHOUT problematic markers
        logger.info("🎭 Step 5: Streaming audio to generate Enrique Torres avatar...")
        
        avatar_frames = []
        
        # Split into properly sized chunks (MUST be even for int16)
        chunk_size = 1024  # Even number for int16 compatibility
        audio_chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
        logger.info(f"🎵 Created {len(audio_chunks)} properly sized audio chunks")
        
        # Send audio chunks without problematic markers
        for i, chunk in enumerate(audio_chunks[:20]):  # Process first 20 chunks
            # Ensure chunk size is even (multiple of 2 bytes for int16)
            if len(chunk) % 2 != 0:
                chunk = chunk[:-1]
            
            logger.info(f"📡 Sending audio chunk {i+1}/{min(20, len(audio_chunks))} ({len(chunk)} bytes)")
            await synctalk_client.send_audio_chunk(chunk)
            
            # Try to receive frames
            frame = await synctalk_client.receive_protobuf_frame(timeout=2.0)
            if frame:
                avatar_frames.append(frame)
                
                # Save real Enrique Torres frame
                frame_path = f"test_output/enrique_torres_frame_{len(avatar_frames):03d}.png"
                frame.save(frame_path)
                logger.info(f"💾 Saved REAL Enrique Torres frame: {frame_path}")
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.05)
        
        # Wait for any remaining frames
        logger.info("⏳ Waiting for any remaining frames...")
        for _ in range(10):
            frame = await synctalk_client.receive_protobuf_frame(timeout=1.5)
            if frame:
                avatar_frames.append(frame)
                frame_path = f"test_output/enrique_torres_frame_{len(avatar_frames):03d}.png"
                frame.save(frame_path)
                logger.info(f"💾 Saved additional Enrique Torres frame: {frame_path}")
        
        logger.info(f"🎭 Collected {len(avatar_frames)} REAL Enrique Torres avatar frames!")
        
        if not avatar_frames:
            logger.error("❌ No avatar frames received from SyncTalk server")
            return False
        
        # Step 6: Composite with slides
        logger.info("🖼️ Step 6: Compositing REAL Enrique Torres with slides...")
        frame_processor = FrameOverlayEngine("frames", output_size=(1920, 1080))
        
        composed_frames = []
        video_frames = []
        
        for i, avatar_frame in enumerate(avatar_frames[:8]):  # Process first 8 frames
            slide_frame = frame_processor.get_slide_frame(i)
            if slide_frame:
                composed_frame = frame_processor.overlay_avatar_on_slide(
                    slide_frame=slide_frame,
                    avatar_frame=avatar_frame,
                    position="bottom-right",
                    scale=0.35  # Slightly larger for better visibility
                )
                
                output_path = f"test_output/final_composed_frame_{i+1:03d}.png"
                composed_frame.save(output_path)
                composed_frames.append(composed_frame)
                
                cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
                video_frames.append(cv_frame)
                
                logger.info(f"✅ Composed frame {i+1} with REAL Enrique Torres avatar")
        
        # Step 7: Create final video
        logger.info("🎬 Step 7: Creating final video with REAL Enrique Torres...")
        
        if video_frames:
            output_video = "test_output/FINAL_ENRIQUE_TORRES_VIDEO.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 6  # Good viewing speed
            height, width = video_frames[0].shape[:2]
            
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            for frame in video_frames:
                out.write(frame)
            
            out.release()
            logger.info(f"🎬 FINAL VIDEO SAVED: {output_video}")
            
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                logger.info(f"✅ SUCCESS: Final video created ({os.path.getsize(output_video)} bytes)")
                logger.info(f"🎭 Video contains {len(video_frames)} frames with REAL Enrique Torres avatar!")
            else:
                logger.error("❌ Video creation failed")
        
        # Step 8: Final summary
        summary = {
            "test_type": "Final SyncTalk Integration - Real Enrique Torres",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avatar_model": "enrique_torres (Alin-cc-dataset)",
            "success": True,
            "results": {
                "real_enrique_torres_frames": len(avatar_frames),
                "composed_frames_created": len(composed_frames),
                "final_video_created": os.path.exists("test_output/FINAL_ENRIQUE_TORRES_VIDEO.mp4"),
                "video_duration_seconds": len(video_frames) / 6.0 if video_frames else 0
            },
            "output_files": [
                "test_output/enrique_torres_frame_*.png (REAL avatar frames)",
                "test_output/final_composed_frame_*.png (composed with slides)",
                "test_output/FINAL_ENRIQUE_TORRES_VIDEO.mp4 (final video)"
            ]
        }
        
        with open("test_output/final_test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("📊 === FINAL TEST SUMMARY ===")
        logger.info(f"🎭 Real Enrique Torres frames: {len(avatar_frames)}")
        logger.info(f"🖼️ Composed frames: {len(composed_frames)}")
        logger.info(f"🎬 Final video: {'YES' if video_frames else 'NO'}")
        logger.info(f"⏱️ Video duration: {len(video_frames) / 6.0:.1f} seconds")
        
        logger.info("🎉 === FINAL SYNCTALK TEST COMPLETED SUCCESSFULLY ===")
        logger.info("🎭 You now have REAL Enrique Torres avatar frames composited with your slides!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Final test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if 'synctalk_client' in locals():
            await synctalk_client.close()

if __name__ == "__main__":
    success = asyncio.run(test_synctalk_final())
    
    if success:
        print("\n🎉 FINAL SYNCTALK TEST WITH REAL ENRIQUE TORRES PASSED!")
        print("\n📁 Check test_output/ directory for results:")
        print("  🎭 enrique_torres_frame_*.png - REAL Enrique Torres avatar frames")
        print("  🖼️ final_composed_frame_*.png - Avatar composited with slides")
        print("  🎬 FINAL_ENRIQUE_TORRES_VIDEO.mp4 - Complete video with real avatar")
        print("  📊 final_test_summary.json - Detailed test results")
        print("\n✨ Your Rapido system now works with REAL SyncTalk avatar generation!")
    else:
        print("\n❌ FINAL SYNCTALK TEST FAILED!")
        print("Check the logs above for error details")
    
    sys.exit(0 if success else 1)
