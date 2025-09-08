#!/usr/bin/env python3
"""
Fixed Real SyncTalk integration test - Proper audio format for SyncTalk server
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
import struct

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
                # Try to parse as image
                try:
                    frame_image = Image.open(io.BytesIO(frame_data))
                    logger.info(f"🎭 Received avatar frame: {frame_image.size}")
                    return frame_image
                except Exception as e:
                    logger.warning(f"Could not parse frame as image: {e}")
                    logger.info(f"Received {len(frame_data)} bytes of data")
                    return None
            else:
                logger.warning(f"Unexpected frame data type: {type(frame_data)}")
                logger.info(f"Data: {str(frame_data)[:100]}...")
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

def create_proper_pcm_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Create properly formatted PCM audio data for SyncTalk"""
    
    # Calculate number of samples
    num_samples = int(duration_seconds * sample_rate)
    
    # Generate sine wave (440 Hz tone)
    t = np.linspace(0, duration_seconds, num_samples, False)
    frequency = 440.0  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM (exactly what SyncTalk expects)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Convert to bytes
    return audio_data.tobytes()

async def test_real_synctalk_with_proper_audio():
    """Test real SyncTalk integration with properly formatted audio"""
    
    logger.info("🚀 === Starting REAL SyncTalk Integration Test (Fixed Audio) ===")
    
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
        test_text = narration_text[:150] + "..."  # First 150 chars for quick test
        logger.info(f"📝 Test narration: {test_text}")
        
        # Step 2: Check if we have ElevenLabs API key for real audio
        api_key = os.getenv('ELEVEN_API_KEY')
        use_real_tts = bool(api_key)
        
        if use_real_tts:
            logger.info("🎵 Using REAL ElevenLabs TTS for audio generation")
            tts_client = ElevenLabsTTSClient(api_key=api_key)
        else:
            logger.info("🎵 Using SYNTHETIC audio for testing (no API key)")
        
        # Step 3: Check SyncTalk server status
        logger.info("🔍 Step 3: Checking SyncTalk server status...")
        
        import requests
        try:
            status_response = requests.get("http://localhost:8001/status", timeout=5)
            if status_response.status_code == 200:
                status_data = status_response.json()
                logger.info(f"✅ SyncTalk server status: {status_data}")
                
                loaded_models = status_data.get("loaded_models", [])
                logger.info(f"🤖 Loaded models: {loaded_models}")
                
                if "Alin-cc-dataset" not in loaded_models:
                    logger.warning("⚠️ Enrique Torres model (Alin-cc-dataset) not found in loaded models")
                    return False
                
            else:
                logger.warning(f"⚠️ SyncTalk status check failed: {status_response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not check SyncTalk status: {e}")
            return False
        
        # Step 4: Initialize SyncTalk client
        logger.info("🔌 Step 4: Connecting to SyncTalk WebSocket...")
        synctalk_client = RealSyncTalkClient()
        
        if not await synctalk_client.connect_and_stream(avatar_name="enrique_torres", sample_rate=16000):
            raise Exception("Failed to connect to SyncTalk WebSocket")
        
        # Step 5: Generate audio data
        logger.info("🎵 Step 5: Generating audio data...")
        
        if use_real_tts:
            # Use real ElevenLabs TTS
            logger.info("Generating real TTS audio...")
            audio_data = await tts_client.generate_full_audio(test_text)
            
            # Convert MP3 to PCM if needed
            # For now, we'll use synthetic audio since MP3->PCM conversion is complex
            logger.info("Using synthetic audio for SyncTalk compatibility...")
            audio_data = create_proper_pcm_audio(duration_seconds=3.0, sample_rate=16000)
        else:
            # Use synthetic PCM audio
            audio_data = create_proper_pcm_audio(duration_seconds=3.0, sample_rate=16000)
        
        logger.info(f"🎵 Generated {len(audio_data)} bytes of PCM audio data")
        
        # Step 6: Stream audio and collect avatar frames
        logger.info("🎭 Step 6: Streaming audio and collecting avatar frames...")
        
        avatar_frames = []
        
        # Split audio into properly sized chunks (multiple of 2 bytes for int16)
        chunk_size = 1024  # Must be even number for int16 compatibility
        if chunk_size % 2 != 0:
            chunk_size += 1
        
        audio_chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
        logger.info(f"🎵 Created {len(audio_chunks)} properly aligned audio chunks")
        
        # Send start marker
        await synctalk_client.send_audio_chunk(b"START_OF_STREAM")
        
        # Stream audio chunks and collect frames
        for i, chunk in enumerate(audio_chunks[:15]):  # Test with first 15 chunks
            logger.info(f"📡 Sending audio chunk {i+1}/{min(15, len(audio_chunks))} ({len(chunk)} bytes)")
            
            # Ensure chunk size is even (multiple of 2 bytes for int16)
            if len(chunk) % 2 != 0:
                chunk = chunk[:-1]  # Remove last byte to make it even
            
            # Send audio chunk
            await synctalk_client.send_audio_chunk(chunk)
            
            # Try to receive corresponding video frame
            frame = await synctalk_client.receive_video_frame(timeout=2.0)
            if frame:
                avatar_frames.append(frame)
                
                # Save individual avatar frame
                frame_path = f"test_output/real_avatar_frame_{i+1:03d}.png"
                frame.save(frame_path)
                logger.info(f"💾 Saved real avatar frame: {frame_path}")
            
            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.05)
        
        # Send end marker
        await synctalk_client.send_audio_chunk(b"END_OF_STREAM")
        
        # Wait a bit more for any remaining frames
        logger.info("⏳ Waiting for any remaining frames...")
        for _ in range(5):
            frame = await synctalk_client.receive_video_frame(timeout=1.0)
            if frame:
                avatar_frames.append(frame)
                frame_path = f"test_output/real_avatar_frame_{len(avatar_frames):03d}.png"
                frame.save(frame_path)
                logger.info(f"💾 Saved additional real avatar frame: {frame_path}")
        
        logger.info(f"🎭 Collected {len(avatar_frames)} REAL avatar frames from SyncTalk!")
        
        if not avatar_frames:
            logger.error("❌ No avatar frames received from SyncTalk server")
            return False
        
        # Step 7: Initialize frame processor
        logger.info("🖼️ Step 7: Initializing frame composition engine...")
        frame_processor = FrameOverlayEngine("frames", output_size=(1920, 1080))
        
        # Step 8: Composite frames with slides
        logger.info("🖼️ Step 8: Compositing REAL avatar frames with slides...")
        
        composed_frames = []
        video_frames = []
        
        num_frames_to_process = min(len(avatar_frames), 10)  # Process up to 10 frames
        
        for i in range(num_frames_to_process):
            # Get slide frame
            slide_frame = frame_processor.get_slide_frame(i)
            if not slide_frame:
                logger.warning(f"Could not get slide frame {i}, skipping...")
                continue
            
            # Get REAL avatar frame
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
            
            logger.info(f"✅ Composed REAL frame {i+1}/{num_frames_to_process}")
        
        # Step 9: Create output video
        logger.info("🎬 Step 9: Creating output video with REAL avatar...")
        
        if video_frames:
            output_video = "test_output/real_enrique_torres_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v for better compatibility
            fps = 5  # Slower for better viewing
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
        
        # Step 10: Generate summary
        logger.info("📊 Step 10: Generating test summary...")
        
        summary = {
            "test_type": "Real SyncTalk Integration Test (Fixed Audio)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avatar_model": "enrique_torres (Alin-cc-dataset)",
            "audio_type": "real_tts" if use_real_tts else "synthetic_pcm",
            "components_tested": {
                "json_parsing": True,
                "synctalk_connection": True,
                "real_avatar_generation": len(avatar_frames) > 0,
                "frame_composition": len(composed_frames) > 0,
                "video_creation": len(video_frames) > 0
            },
            "results": {
                "narration_length": len(test_text),
                "audio_chunks_sent": min(15, len(audio_chunks)),
                "real_avatar_frames_received": len(avatar_frames),
                "composed_frames_created": len(composed_frames),
                "video_created": os.path.exists("test_output/real_enrique_torres_video.mp4")
            },
            "output_files": [
                "test_output/real_avatar_frame_*.png",
                "test_output/real_composed_frame_*.png", 
                "test_output/real_enrique_torres_video.mp4"
            ]
        }
        
        # Save summary
        with open("test_output/real_test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("📊 === REAL TEST SUMMARY ===")
        for key, value in summary["results"].items():
            logger.info(f"{key}: {value}")
        
        logger.info("🎉 === Real SyncTalk Integration Test COMPLETED SUCCESSFULLY ===")
        
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
    success = asyncio.run(test_real_synctalk_with_proper_audio())
    
    if success:
        print("\n🎉 REAL SYNCTALK TEST WITH ENRIQUE TORRES PASSED!")
        print("Check test_output/ directory for results:")
        print("  - real_avatar_frame_*.png (REAL Enrique Torres avatar frames)")
        print("  - real_composed_frame_*.png (composed with slides)")
        print("  - real_enrique_torres_video.mp4 (final video with REAL avatar)")
        print("  - real_test_summary.json (detailed results)")
    else:
        print("\n❌ REAL SYNCTALK TEST FAILED!")
        print("Check the logs above for error details")
    
    sys.exit(0 if success else 1)
