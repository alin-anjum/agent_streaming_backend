#!/usr/bin/env python3
"""
Real SyncTalk integration test with proper protobuf decoding
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

# Add rapido src to path
sys.path.append('./rapido/src')
# Add SyncTalk_2D to path for protobuf
sys.path.append('./SyncTalk_2D')

from data_parser import SlideDataParser
from tts_client import ElevenLabsTTSClient
from frame_processor import FrameOverlayEngine

# Import protobuf message
try:
    from frame_message_pb2 import FrameMessage
    logger = logging.getLogger(__name__)
    logger.info("✅ Successfully imported protobuf FrameMessage")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"❌ Failed to import protobuf: {e}")
    logger.info("Generating protobuf files...")
    
    # Generate protobuf files if they don't exist
    import subprocess
    try:
        result = subprocess.run([
            "python", "-m", "grpc_tools.protoc", 
            "--proto_path=SyncTalk_2D", 
            "--python_out=SyncTalk_2D", 
            "SyncTalk_2D/frame_message.proto"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            logger.info("✅ Generated protobuf files successfully")
            from frame_message_pb2 import FrameMessage
        else:
            logger.error(f"❌ Failed to generate protobuf: {result.stderr}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error generating protobuf: {e}")
        sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SyncTalkProtobufClient:
    """WebSocket client that properly handles SyncTalk protobuf messages"""
    
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
            # Receive protobuf bytes
            protobuf_data = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            
            if isinstance(protobuf_data, bytes):
                # Decode protobuf message
                frame_msg = FrameMessage()
                frame_msg.ParseFromString(protobuf_data)
                
                logger.info(f"📦 Received protobuf message:")
                logger.info(f"  - Has video: {len(frame_msg.video_bytes) > 0 if frame_msg.video_bytes else False}")
                logger.info(f"  - Has audio: {len(frame_msg.audio_bytes) > 0 if frame_msg.audio_bytes else False}")
                logger.info(f"  - Start speech: {frame_msg.start_speech}")
                logger.info(f"  - End speech: {frame_msg.end_speech}")
                
                if frame_msg.video_bytes:
                    # Convert video bytes to numpy array
                    video_data = np.frombuffer(frame_msg.video_bytes, dtype=np.uint8)
                    
                    # Reshape to image (assuming 350x350x3 based on server logs)
                    try:
                        # Try common image shapes for SyncTalk
                        if len(video_data) == 350 * 350 * 3:
                            frame_array = video_data.reshape((350, 350, 3))
                        elif len(video_data) == 256 * 256 * 3:
                            frame_array = video_data.reshape((256, 256, 3))
                        elif len(video_data) == 512 * 512 * 3:
                            frame_array = video_data.reshape((512, 512, 3))
                        else:
                            logger.warning(f"Unknown video data size: {len(video_data)}")
                            # Try to infer square dimensions
                            total_pixels = len(video_data) // 3
                            side = int(np.sqrt(total_pixels))
                            if side * side * 3 == len(video_data):
                                frame_array = video_data.reshape((side, side, 3))
                                logger.info(f"Inferred dimensions: {side}x{side}")
                            else:
                                logger.error(f"Cannot reshape video data of size {len(video_data)}")
                                return None
                        
                        # Convert to PIL Image (RGB)
                        frame_image = Image.fromarray(frame_array, 'RGB')
                        logger.info(f"🎭 Decoded avatar frame: {frame_image.size}")
                        return frame_image
                        
                    except Exception as e:
                        logger.error(f"Error reshaping video data: {e}")
                        logger.info(f"Video data size: {len(video_data)} bytes")
                        return None
                else:
                    logger.info("📦 Protobuf message contains no video data")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning("⏰ Timeout waiting for protobuf frame")
            return None
        except Exception as e:
            logger.error(f"❌ Error receiving protobuf frame: {e}")
            return None
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("🔌 WebSocket connection closed")

def create_proper_pcm_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Create properly formatted PCM audio data for SyncTalk"""
    
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, False)
    frequency = 440.0  # A4 note
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data.tobytes()

async def test_synctalk_with_protobuf():
    """Test SyncTalk with proper protobuf decoding"""
    
    logger.info("🚀 === Starting SyncTalk Protobuf Integration Test ===")
    
    # Create output directories
    os.makedirs("test_output", exist_ok=True)
    
    try:
        # Step 1: Parse JSON data
        logger.info("📄 Step 1: Parsing JSON slide data...")
        data_parser = SlideDataParser("test1.json")
        if not data_parser.load_data():
            raise Exception("Failed to load JSON data")
        narration_text = data_parser.get_narration_text()
        test_text = narration_text[:100] + "..."
        logger.info(f"📝 Test narration: {test_text}")
        
        # Step 2: Check SyncTalk server status
        logger.info("🔍 Step 2: Checking SyncTalk server status...")
        import requests
        status_response = requests.get("http://localhost:8001/status", timeout=5)
        if status_response.status_code == 200:
            status_data = status_response.json()
            logger.info(f"✅ SyncTalk server ready: {status_data['status']}")
            logger.info(f"🤖 Loaded models: {status_data['loaded_models']}")
        
        # Step 3: Connect to SyncTalk
        logger.info("🔌 Step 3: Connecting to SyncTalk WebSocket...")
        synctalk_client = SyncTalkProtobufClient()
        
        if not await synctalk_client.connect_and_stream(avatar_name="enrique_torres", sample_rate=16000):
            raise Exception("Failed to connect to SyncTalk WebSocket")
        
        # Step 4: Generate and stream audio
        logger.info("🎵 Step 4: Generating and streaming audio...")
        
        # Create test audio
        audio_data = create_proper_pcm_audio(duration_seconds=2.0, sample_rate=16000)
        logger.info(f"🎵 Generated {len(audio_data)} bytes of PCM audio")
        
        # Split into chunks
        chunk_size = 1024
        audio_chunks = [audio_data[i:i+chunk_size] for i in range(0, len(audio_data), chunk_size)]
        logger.info(f"🎵 Created {len(audio_chunks)} audio chunks")
        
        # Step 5: Stream audio and collect frames
        logger.info("🎭 Step 5: Streaming audio and collecting avatar frames...")
        
        avatar_frames = []
        
        # Send start marker
        await synctalk_client.send_audio_chunk(b"START_OF_STREAM")
        
        # Stream audio chunks
        for i, chunk in enumerate(audio_chunks[:10]):  # Test with first 10 chunks
            if len(chunk) % 2 != 0:
                chunk = chunk[:-1]  # Ensure even size for int16
            
            logger.info(f"📡 Sending audio chunk {i+1}/{min(10, len(audio_chunks))}")
            await synctalk_client.send_audio_chunk(chunk)
            
            # Try to receive frame
            frame = await synctalk_client.receive_protobuf_frame(timeout=3.0)
            if frame:
                avatar_frames.append(frame)
                
                # Save frame
                frame_path = f"test_output/protobuf_avatar_frame_{i+1:03d}.png"
                frame.save(frame_path)
                logger.info(f"💾 Saved REAL Enrique Torres frame: {frame_path}")
            
            await asyncio.sleep(0.1)
        
        # Send end marker
        await synctalk_client.send_audio_chunk(b"END_OF_STREAM")
        
        # Wait for any remaining frames
        logger.info("⏳ Waiting for remaining frames...")
        for _ in range(5):
            frame = await synctalk_client.receive_protobuf_frame(timeout=2.0)
            if frame:
                avatar_frames.append(frame)
                frame_path = f"test_output/protobuf_avatar_frame_{len(avatar_frames):03d}.png"
                frame.save(frame_path)
                logger.info(f"💾 Saved additional frame: {frame_path}")
        
        logger.info(f"🎭 Collected {len(avatar_frames)} REAL Enrique Torres avatar frames!")
        
        if not avatar_frames:
            logger.error("❌ No avatar frames received")
            return False
        
        # Step 6: Composite with slides
        logger.info("🖼️ Step 6: Compositing with slide frames...")
        frame_processor = FrameOverlayEngine("frames", output_size=(1920, 1080))
        
        composed_frames = []
        video_frames = []
        
        for i, avatar_frame in enumerate(avatar_frames[:5]):  # Process first 5 frames
            slide_frame = frame_processor.get_slide_frame(i)
            if slide_frame:
                composed_frame = frame_processor.overlay_avatar_on_slide(
                    slide_frame=slide_frame,
                    avatar_frame=avatar_frame,
                    position="bottom-right",
                    scale=0.3
                )
                
                output_path = f"test_output/protobuf_composed_frame_{i+1:03d}.png"
                composed_frame.save(output_path)
                composed_frames.append(composed_frame)
                
                cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
                video_frames.append(cv_frame)
                
                logger.info(f"✅ Composed frame {i+1} with REAL Enrique Torres avatar")
        
        # Step 7: Create video
        logger.info("🎬 Step 7: Creating video with REAL avatar...")
        
        if video_frames:
            output_video = "test_output/enrique_torres_real_avatar.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 5
            height, width = video_frames[0].shape[:2]
            
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            for frame in video_frames:
                out.write(frame)
            
            out.release()
            logger.info(f"🎬 Video saved: {output_video}")
            
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                logger.info(f"✅ SUCCESS: Video created ({os.path.getsize(output_video)} bytes)")
            else:
                logger.error("❌ Video creation failed")
        
        # Step 8: Summary
        summary = {
            "test_type": "SyncTalk Protobuf Integration Test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avatar_model": "enrique_torres (Alin-cc-dataset)",
            "results": {
                "real_avatar_frames_received": len(avatar_frames),
                "composed_frames_created": len(composed_frames),
                "video_created": os.path.exists("test_output/enrique_torres_real_avatar.mp4")
            }
        }
        
        with open("test_output/protobuf_test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("📊 === TEST SUMMARY ===")
        for key, value in summary["results"].items():
            logger.info(f"{key}: {value}")
        
        logger.info("🎉 === SyncTalk Protobuf Test COMPLETED ===")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if 'synctalk_client' in locals():
            await synctalk_client.close()

if __name__ == "__main__":
    success = asyncio.run(test_synctalk_with_protobuf())
    
    if success:
        print("\n🎉 REAL ENRIQUE TORRES AVATAR TEST PASSED!")
        print("Check test_output/ directory for results:")
        print("  - protobuf_avatar_frame_*.png (REAL Enrique Torres frames)")
        print("  - protobuf_composed_frame_*.png (composed with slides)")
        print("  - enrique_torres_real_avatar.mp4 (final video)")
    else:
        print("\n❌ PROTOBUF TEST FAILED!")
    
    sys.exit(0 if success else 1)
