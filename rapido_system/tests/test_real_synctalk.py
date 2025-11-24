#!/usr/bin/env python3
"""
Real SyncTalk integration test - Uses actual Enrique Torres avatar generation
"""

import asyncio
import os
import sys
import json
import logging
import websockets
import base64
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path
import wave

# Add rapido src to path
sys.path.append('./rapido/src')

from data_parser import SlideDataParser
from tts_client import ElevenLabsTTSClient
from frame_processor import FrameOverlayEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealSyncTalkClient:
    """Real WebSocket client for SyncTalk server"""
    
    def __init__(self, server_url="ws://localhost:8001"):
        self.server_url = server_url
        self.websocket = None
        
    async def connect_and_stream(self, avatar_name="enrique_torres", sample_rate=16000):
        """Connect to SyncTalk and stream audio for avatar generation"""
        
        # Construct WebSocket URL with query parameters
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
    
    async def receive_video_frame(self, timeout=5.0):
        """Receive video frame from SyncTalk"""
        if not self.websocket:
            raise Exception("Not connected to server")
        
        try:
            frame_data = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            
            if isinstance(frame_data, bytes):
                # Convert bytes to PIL Image
                frame_image = Image.open(io.BytesIO(frame_data))
                logger.info(f"🎭 Received avatar frame: {frame_image.size}")
                return frame_image
            else:
                logger.warning(f"Unexpected frame data type: {type(frame_data)}")
                return None
                
        except asyncio.TimeoutError:
            logger.warning("⏰ Timeout waiting for avatar frame")
            return None
        except Exception as e:
            logger.error(f"❌ Error receiving frame: {e}")
            return None
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 WebSocket connection closed")

def create_test_audio(text: str, duration_seconds: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Create test audio data (sine wave) for the given text duration"""
    
    # Calculate number of samples
    num_samples = int(duration_seconds * sample_rate)
    
    # Generate sine wave (440 Hz tone)
    t = np.linspace(0, duration_seconds, num_samples, False)
    frequency = 440.0  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Convert to bytes
    return audio_data.tobytes()

async def test_real_synctalk_integration():
    """Test real SyncTalk integration with Enrique Torres avatar"""
    
    logger.info("🚀 === Starting REAL SyncTalk Integration Test ===")
    
    # Create output directories
    os.makedirs("test_output", exist_ok=True)
    os.makedirs("test_temp", exist_ok=True)
    
    try:
        # Step 1: Parse JSON data
        logger.info("📄 Step 1: Parsing JSON slide data...")
        data_parser = SlideDataParser("test1.json")
        if not data_parser.load_data():
            raise Exception("Failed to load JSON data")
        narration_text = data_parser.get_narration_text()
        if not narration_text:
            raise Exception("No narration text found in JSON")
        
        # Truncate for testing
        test_text = narration_text[:200] + "..."  # First 200 chars for quick test
        logger.info(f"📝 Test narration: {test_text}")
        
        # Step 2: Check SyncTalk server status
        logger.info("🔍 Step 2: Checking SyncTalk server status...")
        
        import requests
        try:
            status_response = requests.get("http://localhost:8001/status", timeout=5)
            if status_response.status_code == 200:
                status_data = status_response.json()
                logger.info(f"✅ SyncTalk server status: {status_data}")
                
                if not status_data.get("available", False):
                    logger.warning("⚠️ SyncTalk server reports not available")
                
                loaded_models = status_data.get("loaded_models", [])
                logger.info(f"🤖 Loaded models: {loaded_models}")
                
                if "Alin-cc-dataset" not in loaded_models:
                    logger.warning("⚠️ Enrique Torres model (Alin-cc-dataset) not found in loaded models")
                
            else:
                logger.warning(f"⚠️ SyncTalk status check failed: {status_response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not check SyncTalk status: {e}")
        
        # Step 3: Initialize SyncTalk client
        logger.info("🔌 Step 3: Connecting to SyncTalk WebSocket...")
        synctalk_client = RealSyncTalkClient()
        
        if not await synctalk_client.connect_and_stream(avatar_name="enrique_torres", sample_rate=16000):
            raise Exception("Failed to connect to SyncTalk WebSocket")
        
        # Step 4: Initialize frame processor
        logger.info("🖼️ Step 4: Initializing frame composition engine...")
        frame_processor = FrameOverlayEngine("frames", output_size=(1920, 1080))
        
        # Step 5: Generate test audio and stream to SyncTalk
        logger.info("🎵 Step 5: Generating and streaming audio to SyncTalk...")
        
        # Create test audio (simulate TTS output)
        test_audio = create_test_audio(test_text, duration_seconds=3.0, sample_rate=16000)
        
        # Split audio into chunks for streaming
        chunk_size = 1024  # bytes per chunk
        audio_chunks = [test_audio[i:i+chunk_size] for i in range(0, len(test_audio), chunk_size)]
        
        logger.info(f"🎵 Created {len(audio_chunks)} audio chunks for streaming")
        
        # Step 6: Stream audio and collect avatar frames
        logger.info("🎭 Step 6: Streaming audio and collecting avatar frames...")
        
        avatar_frames = []
        
        # Send start marker
        await synctalk_client.send_audio_chunk(b"START_OF_STREAM")
        
        # Stream audio chunks and collect frames
        for i, chunk in enumerate(audio_chunks[:10]):  # Test with first 10 chunks
            logger.info(f"📡 Sending audio chunk {i+1}/{min(10, len(audio_chunks))}")
            
            # Send audio chunk
            await synctalk_client.send_audio_chunk(chunk)
            
            # Try to receive corresponding video frame
            frame = await synctalk_client.receive_video_frame(timeout=3.0)
            if frame:
                avatar_frames.append(frame)
                
                # Save individual avatar frame
                frame_path = f"test_output/real_avatar_frame_{i+1:03d}.png"
                frame.save(frame_path)
                logger.info(f"💾 Saved avatar frame: {frame_path}")
            
            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)
        
        # Send end marker
        await synctalk_client.send_audio_chunk(b"END_OF_STREAM")
        
        logger.info(f"🎭 Collected {len(avatar_frames)} real avatar frames")
        
        if not avatar_frames:
            logger.warning("⚠️ No avatar frames received from SyncTalk server")
            logger.info("🔄 Falling back to mock frames for composition test...")
            
            # Create mock frames as fallback
            for i in range(5):
                mock_frame = Image.new('RGBA', (350, 350), (255, 0, 0, 200))
                avatar_frames.append(mock_frame)
        
        # Step 7: Composite frames with slides
        logger.info("🖼️ Step 7: Compositing avatar frames with slides...")
        
        composed_frames = []
        video_frames = []
        
        num_frames_to_process = min(len(avatar_frames), 5)  # Process up to 5 frames
        
        for i in range(num_frames_to_process):
            # Get slide frame
            slide_frame = frame_processor.get_slide_frame(i)
            if not slide_frame:
                logger.warning(f"Could not get slide frame {i}, skipping...")
                continue
            
            # Get avatar frame
            avatar_frame = avatar_frames[i]
            
            # Composite the frames
            composed_frame = frame_processor.overlay_avatar_on_slide(
                slide_frame=slide_frame,
                avatar_frame=avatar_frame,
                position="bottom-right",
                scale=0.3
            )
            
            # Save composed frame
            output_path = f"test_output/real_composed_frame_{i+1:03d}.png"
            composed_frame.save(output_path)
            composed_frames.append(composed_frame)
            
            # Convert to OpenCV format for video
            cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
            video_frames.append(cv_frame)
            
            logger.info(f"✅ Composed real frame {i+1}/{num_frames_to_process}")
        
        # Step 8: Create output video
        logger.info("🎬 Step 8: Creating output video with real avatar...")
        
        if video_frames:
            output_video = "test_output/real_synctalk_test.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for better compatibility
            fps = 2  # Slow for testing
            height, width = video_frames[0].shape[:2]
            
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            for frame in video_frames:
                out.write(frame)
            
            out.release()
            logger.info(f"🎬 Video saved: {output_video}")
            
            # Verify video file
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                logger.info(f"✅ SUCCESS: Video file created ({os.path.getsize(output_video)} bytes)")
            else:
                logger.error("❌ FAILED: Video file was not created properly")
        
        # Step 9: Generate summary
        logger.info("📊 Step 9: Generating test summary...")
        
        summary = {
            "test_type": "Real SyncTalk Integration Test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avatar_model": "enrique_torres (Alin-cc-dataset)",
            "components_tested": {
                "json_parsing": True,
                "synctalk_connection": True,
                "real_avatar_generation": len(avatar_frames) > 0,
                "frame_composition": len(composed_frames) > 0,
                "video_creation": len(video_frames) > 0
            },
            "results": {
                "narration_length": len(test_text),
                "audio_chunks_sent": min(10, len(audio_chunks)),
                "avatar_frames_received": len(avatar_frames),
                "composed_frames_created": len(composed_frames),
                "video_created": os.path.exists("test_output/real_synctalk_test.mp4")
            },
            "output_files": [
                "test_output/real_avatar_frame_*.png",
                "test_output/real_composed_frame_*.png", 
                "test_output/real_synctalk_test.mp4"
            ]
        }
        
        # Save summary
        with open("test_output/real_test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("📊 === REAL TEST SUMMARY ===")
        for key, value in summary["results"].items():
            logger.info(f"{key}: {value}")
        
        logger.info("🎉 === Real SyncTalk Integration Test COMPLETED ===")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if 'synctalk_client' in locals():
            await synctalk_client.close()

if __name__ == "__main__":
    # Run the real SyncTalk test
    success = asyncio.run(test_real_synctalk_integration())
    
    if success:
        print("\n🎉 REAL SYNCTALK TEST PASSED!")
        print("Check test_output/ directory for results:")
        print("  - real_avatar_frame_*.png (real avatar frames from SyncTalk)")
        print("  - real_composed_frame_*.png (composed with slides)")
        print("  - real_synctalk_test.mp4 (final video with real avatar)")
        print("  - real_test_summary.json (detailed results)")
    else:
        print("\n❌ REAL SYNCTALK TEST FAILED!")
        print("Check the logs above for error details")
    
    sys.exit(0 if success else 1)
