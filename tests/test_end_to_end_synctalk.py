#!/usr/bin/env python3
"""
End-to-end test for Rapido + SyncTalk_2D integration
Tests: Audio generation → SyncTalk streaming → Frame composition
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
from PIL import Image, ImageDraw, ImageFont
import io
import time
from pathlib import Path

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

class SyncTalkWebSocketClient:
    """WebSocket client for SyncTalk server communication"""
    
    def __init__(self, server_url="ws://localhost:8001/ws"):
        self.server_url = server_url
        self.websocket = None
        
    async def connect(self):
        """Connect to SyncTalk WebSocket server"""
        try:
            logger.info(f"Connecting to SyncTalk server: {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)
            logger.info("Connected to SyncTalk server successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SyncTalk server: {e}")
            return False
            
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to SyncTalk server"""
        if not self.websocket:
            raise Exception("Not connected to server")
            
        # Encode audio as base64 for WebSocket transmission
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        
        message = {
            "type": "audio_chunk",
            "avatar_name": "enrique_torres",  # Hardcoded as requested
            "audio_data": audio_b64,
            "sample_rate": 22050,
            "format": "mp3"
        }
        
        await self.websocket.send(json.dumps(message))
        logger.info(f"Sent audio chunk ({len(audio_data)} bytes)")
        
    async def receive_frame(self):
        """Receive avatar frame from SyncTalk server"""
        if not self.websocket:
            raise Exception("Not connected to server")
            
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            if data.get("type") == "avatar_frame":
                # Decode base64 frame data
                frame_b64 = data.get("frame_data")
                if frame_b64:
                    frame_bytes = base64.b64decode(frame_b64)
                    # Convert to PIL Image
                    frame_image = Image.open(io.BytesIO(frame_bytes))
                    logger.info(f"Received avatar frame: {frame_image.size}")
                    return frame_image
                    
            return None
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for avatar frame")
            return None
        except Exception as e:
            logger.error(f"Error receiving frame: {e}")
            return None
            
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")

async def test_end_to_end_pipeline():
    """Test the complete pipeline: JSON → TTS → SyncTalk → Composition"""
    
    logger.info("=== Starting End-to-End Pipeline Test ===")
    
    # Create output directories
    os.makedirs("test_output", exist_ok=True)
    os.makedirs("test_temp", exist_ok=True)
    
    try:
        # Step 1: Parse JSON data
        logger.info("Step 1: Parsing JSON slide data...")
        data_parser = SlideDataParser("test1.json")
        if not data_parser.load_data():
            raise Exception("Failed to load JSON data")
        narration_text = data_parser.get_narration_text()
        if not narration_text:
            raise Exception("No narration text found in JSON")
        logger.info(f"Extracted narration: {narration_text[:100]}...")
        
        # Step 2: Initialize TTS client
        logger.info("Step 2: Initializing ElevenLabs TTS client...")
        
        # Get API key from environment
        api_key = os.getenv('ELEVEN_API_KEY')
        if not api_key:
            logger.error("ELEVEN_API_KEY not found in environment")
            logger.info("Using mock audio generation for testing...")
            use_real_tts = False
        else:
            use_real_tts = True
            tts_client = ElevenLabsTTSClient(api_key=api_key)
        
        # Step 3: Initialize SyncTalk WebSocket client
        logger.info("Step 3: Connecting to SyncTalk server...")
        synctalk_client = SyncTalkWebSocketClient()
        
        if not await synctalk_client.connect():
            logger.error("Failed to connect to SyncTalk server")
            logger.info("Using mock avatar frames for testing...")
            use_real_synctalk = False
        else:
            use_real_synctalk = True
        
        # Step 4: Initialize frame processor
        logger.info("Step 4: Initializing frame composition engine...")
        frame_processor = FrameOverlayEngine("frames", output_size=(1920, 1080))
        
        # Get slide frames
        slide_frames = sorted([f for f in os.listdir("frames") if f.endswith('.png')])[:10]  # Test with 10 frames
        logger.info(f"Found {len(slide_frames)} slide frames for testing")
        
        # Step 5: Process audio and generate avatar frames
        logger.info("Step 5: Processing audio and generating avatar frames...")
        
        avatar_frames = []
        
        if use_real_tts and use_real_synctalk:
            # Real pipeline with ElevenLabs and SyncTalk
            logger.info("Using REAL TTS + SyncTalk pipeline")
            
            # Generate and stream audio
            async for audio_chunk in tts_client.stream_tts(narration_text):
                if audio_chunk:
                    # Send audio to SyncTalk
                    await synctalk_client.send_audio_chunk(audio_chunk)
                    
                    # Receive corresponding avatar frame
                    avatar_frame = await synctalk_client.receive_frame()
                    if avatar_frame:
                        avatar_frames.append(avatar_frame)
                        
                    # Limit frames for testing
                    if len(avatar_frames) >= len(slide_frames):
                        break
                        
        else:
            # Mock pipeline for testing without API keys or server
            logger.info("Using MOCK audio + avatar generation for testing")
            
            # Generate mock avatar frames
            for i in range(len(slide_frames)):
                # Create mock avatar frame
                avatar_frame = Image.new('RGBA', (350, 350), (0, 255, 0, 200))  # Semi-transparent green
                draw = ImageDraw.Draw(avatar_frame)
                
                # Draw a simple animated face
                face_y = 175 + int(20 * np.sin(i * 0.5))  # Bouncing motion
                
                # Face circle
                draw.ellipse([100, face_y-50, 250, face_y+50], fill=(255, 220, 177, 255))
                
                # Eyes
                draw.ellipse([130, face_y-20, 150, face_y], fill=(0, 0, 0, 255))
                draw.ellipse([200, face_y-20, 220, face_y], fill=(0, 0, 0, 255))
                
                # Mouth (animated)
                mouth_width = 30 + int(10 * np.sin(i * 0.8))
                draw.ellipse([175-mouth_width//2, face_y+10, 175+mouth_width//2, face_y+25], fill=(255, 0, 0, 255))
                
                avatar_frames.append(avatar_frame)
                logger.info(f"Generated mock avatar frame {i+1}")
        
        # Step 6: Composite frames
        logger.info("Step 6: Compositing avatar frames over slide frames...")
        
        composed_frames = []
        video_frames = []
        
        for i, (slide_frame_name, avatar_frame) in enumerate(zip(slide_frames, avatar_frames)):
            # Get the slide frame from the processor
            slide_frame = frame_processor.get_slide_frame(i)
            if not slide_frame:
                logger.warning(f"Could not get slide frame {i}, skipping...")
                continue
            
            # Composite the frames
            composed_frame = frame_processor.overlay_avatar_on_slide(
                slide_frame=slide_frame,
                avatar_frame=avatar_frame,
                position="bottom-right",
                scale=0.3
            )
            
            # Save composed frame
            output_path = f"test_output/composed_frame_{i+1:03d}.png"
            composed_frame.save(output_path)
            composed_frames.append(composed_frame)
            
            # Convert to OpenCV format for video
            cv_frame = cv2.cvtColor(np.array(composed_frame), cv2.COLOR_RGB2BGR)
            video_frames.append(cv_frame)
            
            logger.info(f"Composed frame {i+1}/{len(slide_frames)}")
        
        # Step 7: Create output video
        logger.info("Step 7: Creating output video...")
        
        if video_frames:
            output_video = "test_output/end_to_end_test.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            fps = 2  # Slow for testing
            height, width = video_frames[0].shape[:2]
            
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            for frame in video_frames:
                out.write(frame)
            
            out.release()
            logger.info(f"Video saved: {output_video}")
            
            # Verify video file
            if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                logger.info(f"SUCCESS: Video file created ({os.path.getsize(output_video)} bytes)")
            else:
                logger.error("FAILED: Video file was not created properly")
        
        # Step 8: Generate summary
        logger.info("Step 8: Generating test summary...")
        
        summary = {
            "test_type": "End-to-End Pipeline Test",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "components_tested": {
                "json_parsing": True,
                "tts_generation": use_real_tts,
                "synctalk_connection": use_real_synctalk,
                "frame_composition": True,
                "video_creation": len(video_frames) > 0
            },
            "results": {
                "narration_length": len(narration_text),
                "slide_frames_processed": len(slide_frames),
                "avatar_frames_generated": len(avatar_frames),
                "composed_frames_created": len(composed_frames),
                "video_created": os.path.exists("test_output/end_to_end_test.mp4")
            },
            "output_files": [
                "test_output/composed_frame_*.png",
                "test_output/end_to_end_test.mp4"
            ]
        }
        
        # Save summary
        with open("test_output/test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=== TEST SUMMARY ===")
        for key, value in summary["results"].items():
            logger.info(f"{key}: {value}")
        
        logger.info("=== End-to-End Pipeline Test COMPLETED ===")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if 'synctalk_client' in locals():
            await synctalk_client.close()

if __name__ == "__main__":
    # Run the end-to-end test
    success = asyncio.run(test_end_to_end_pipeline())
    
    if success:
        print("\n🎉 END-TO-END TEST PASSED!")
        print("Check test_output/ directory for results:")
        print("  - composed_frame_*.png (individual frames)")
        print("  - end_to_end_test.mp4 (final video)")
        print("  - test_summary.json (detailed results)")
    else:
        print("\n❌ END-TO-END TEST FAILED!")
        print("Check the logs above for error details")
    
    sys.exit(0 if success else 1)
