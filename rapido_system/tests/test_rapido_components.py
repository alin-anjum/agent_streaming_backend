#!/usr/bin/env python3
"""
Test script for Rapido components (excluding SyncTalk inference).
Tests:
1. Audio generation and streaming from ElevenLabs
2. Audio processing and format conversion
3. Frame composition and overlay
"""

import asyncio
import os
import sys
import json
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import tempfile
import wave

# Add rapido to path
sys.path.append(str(Path(__file__).parent / "rapido" / "src"))

from data_parser import SlideDataParser
from tts_client import ElevenLabsTTSClient
from frame_processor import FrameOverlayEngine
from synctalk_fastapi_client import audio_chunks_from_tts

# Test configuration
TEST_CONFIG = {
    'ELEVENLABS_API_KEY': os.getenv('ELEVEN_API_KEY') or os.getenv('ELEVENLABS_API_KEY'),
    'ELEVENLABS_VOICE_ID': 'pNInz6obpgDQGcFmaJgB',  # Adam voice
    'SLIDE_FRAMES_PATH': './frames',
    'OUTPUT_PATH': './test_output',
    'TEMP_PATH': './test_temp'
}

class MockInferenceFrame:
    """Mock inference frame generator for testing frame composition."""
    
    def __init__(self, width=350, height=350):
        self.width = width
        self.height = height
        self.frame_count = 0
    
    def generate_mock_frame(self) -> np.ndarray:
        """Generate a mock avatar frame with moving elements."""
        # Create a frame with a moving circle to simulate avatar motion
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Background color (simulating green screen)
        frame[:, :] = [8, 152, 49]  # Green background
        
        # Add a moving "face" circle
        center_x = self.width // 2
        center_y = self.height // 2 + int(10 * np.sin(self.frame_count * 0.1))
        radius = 80
        
        # Draw face circle
        cv2.circle(frame, (center_x, center_y), radius, (200, 180, 160), -1)
        
        # Add "eyes"
        eye_offset = 25
        cv2.circle(frame, (center_x - eye_offset, center_y - 15), 8, (0, 0, 0), -1)
        cv2.circle(frame, (center_x + eye_offset, center_y - 15), 8, (0, 0, 0), -1)
        
        # Add "mouth" - changes with frame count to simulate talking
        mouth_height = int(5 + 10 * abs(np.sin(self.frame_count * 0.3)))
        cv2.ellipse(frame, (center_x, center_y + 20), (20, mouth_height), 0, 0, 180, (0, 0, 0), -1)
        
        self.frame_count += 1
        return frame

async def test_1_audio_generation():
    """Test 1: Audio generation and streaming from ElevenLabs"""
    print("\n" + "="*60)
    print("TEST 1: Audio Generation and Streaming")
    print("="*60)
    
    if not TEST_CONFIG['ELEVENLABS_API_KEY']:
        print("❌ SKIPPED: No ElevenLabs API key found")
        return False
    
    try:
        # Parse test data
        print("📋 Parsing test1.json...")
        data_parser = SlideDataParser('test1.json')
        data_parser.load_data()
        narration_text = data_parser.get_narration_text()
        print(f"   Narration text length: {len(narration_text)} characters")
        print(f"   First 100 chars: {narration_text[:100]}...")
        
        # Initialize TTS client
        print("🎤 Initializing ElevenLabs TTS client...")
        tts_client = ElevenLabsTTSClient(
            api_key=TEST_CONFIG['ELEVENLABS_API_KEY'],
            voice_id=TEST_CONFIG['ELEVENLABS_VOICE_ID']
        )
        
        # Test streaming TTS
        print("🎵 Starting TTS streaming...")
        audio_chunks = []
        chunk_count = 0
        total_size = 0
        
        async for audio_chunk in tts_client.stream_tts(narration_text):
            audio_chunks.append(audio_chunk)
            chunk_count += 1
            total_size += len(audio_chunk)
            
            if chunk_count <= 5:  # Show first 5 chunks
                print(f"   Chunk {chunk_count}: {len(audio_chunk)} bytes")
            elif chunk_count == 6:
                print("   ... (more chunks)")
            
            # Limit test to prevent long execution
            if chunk_count >= 10:
                break
        
        print(f"✅ TTS streaming successful!")
        print(f"   Total chunks received: {chunk_count}")
        print(f"   Total audio data: {total_size:,} bytes")
        
        # Save a sample for verification
        if audio_chunks:
            os.makedirs(TEST_CONFIG['TEMP_PATH'], exist_ok=True)
            with open(f"{TEST_CONFIG['TEMP_PATH']}/sample_audio.mp3", 'wb') as f:
                for chunk in audio_chunks[:5]:  # Save first 5 chunks
                    f.write(chunk)
            print(f"   Sample saved to: {TEST_CONFIG['TEMP_PATH']}/sample_audio.mp3")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_2_audio_processing():
    """Test 2: Audio processing and format conversion"""
    print("\n" + "="*60)
    print("TEST 2: Audio Processing and Format Conversion")
    print("="*60)
    
    if not TEST_CONFIG['ELEVENLABS_API_KEY']:
        print("❌ SKIPPED: No ElevenLabs API key found")
        return False
    
    try:
        # Get TTS stream
        print("🎤 Getting TTS stream...")
        tts_client = ElevenLabsTTSClient(
            api_key=TEST_CONFIG['ELEVENLABS_API_KEY'],
            voice_id=TEST_CONFIG['ELEVENLABS_VOICE_ID']
        )
        
        # Test short text for faster processing
        test_text = "Hello, this is a test of audio processing and format conversion."
        tts_stream = tts_client.stream_tts(test_text)
        
        # Test audio format conversion
        print("🔄 Testing audio format conversion...")
        pcm_chunks = []
        chunk_count = 0
        
        async for pcm_chunk in audio_chunks_from_tts(tts_stream):
            pcm_chunks.append(pcm_chunk)
            chunk_count += 1
            
            if chunk_count <= 3:
                print(f"   PCM chunk {chunk_count}: {len(pcm_chunk)} bytes")
            elif chunk_count == 4:
                print("   ... (more PCM chunks)")
            
            # Limit for testing
            if chunk_count >= 5:
                break
        
        print(f"✅ Audio processing successful!")
        print(f"   Total PCM chunks: {chunk_count}")
        print(f"   PCM format: 24kHz, 16-bit, mono")
        
        # Save PCM sample
        if pcm_chunks:
            os.makedirs(TEST_CONFIG['TEMP_PATH'], exist_ok=True)
            pcm_file = f"{TEST_CONFIG['TEMP_PATH']}/sample_audio.pcm"
            with open(pcm_file, 'wb') as f:
                for chunk in pcm_chunks:
                    f.write(chunk)
            print(f"   PCM sample saved to: {pcm_file}")
            
            # Also save as WAV for easier verification
            wav_file = f"{TEST_CONFIG['TEMP_PATH']}/sample_audio.wav"
            with wave.open(wav_file, 'wb') as wav:
                wav.setnchannels(1)  # mono
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(24000)  # 24kHz
                for chunk in pcm_chunks:
                    wav.writeframes(chunk)
            print(f"   WAV sample saved to: {wav_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_3_frame_composition():
    """Test 3: Frame composition and overlay"""
    print("\n" + "="*60)
    print("TEST 3: Frame Composition and Overlay")
    print("="*60)
    
    try:
        # Check if slide frames exist
        frames_path = Path(TEST_CONFIG['SLIDE_FRAMES_PATH'])
        if not frames_path.exists():
            print(f"❌ SKIPPED: Frames directory not found: {frames_path}")
            return False
        
        frame_files = list(frames_path.glob('*.png'))
        if not frame_files:
            print(f"❌ SKIPPED: No PNG frames found in {frames_path}")
            return False
        
        print(f"📁 Found {len(frame_files)} slide frames")
        
        # Initialize frame processor
        print("🖼️ Initializing frame processor...")
        frame_processor = FrameOverlayEngine(
            slide_frames_path=str(frames_path),
            output_size=(1920, 1080)
        )
        
        # Initialize mock inference frame generator
        mock_generator = MockInferenceFrame()
        
        # Test frame composition
        print("🎨 Testing frame composition...")
        os.makedirs(TEST_CONFIG['OUTPUT_PATH'], exist_ok=True)
        
        # Process first few frames
        for i, frame_file in enumerate(frame_files[:5]):  # Test with first 5 frames
            print(f"   Processing frame {i+1}: {frame_file.name}")
            
            # Load slide frame using PIL
            slide_frame = Image.open(str(frame_file))
            if slide_frame is None:
                print(f"   ⚠️ Could not load frame: {frame_file}")
                continue
            
            # Generate mock inference frame and convert to PIL
            inference_frame_cv = mock_generator.generate_mock_frame()
            inference_frame = Image.fromarray(cv2.cvtColor(inference_frame_cv, cv2.COLOR_BGR2RGB))
            
            # Compose frames
            composed_frame = frame_processor.overlay_avatar_on_slide(
                slide_frame, 
                inference_frame,
                position='bottom-right',
                scale=0.3
            )
            
            # Save composed frame
            output_file = f"{TEST_CONFIG['OUTPUT_PATH']}/composed_frame_{i+1:03d}.png"
            composed_frame.save(output_file)
            print(f"   ✅ Saved: {output_file}")
        
        print(f"✅ Frame composition successful!")
        print(f"   Composed frames saved to: {TEST_CONFIG['OUTPUT_PATH']}")
        
        # Create a simple video from composed frames for verification
        print("🎬 Creating test video...")
        video_file = f"{TEST_CONFIG['OUTPUT_PATH']}/test_composition.mp4"
        
        # Get frame dimensions from the first composed frame
        test_frame_path = f"{TEST_CONFIG['OUTPUT_PATH']}/composed_frame_001.png"
        if os.path.exists(test_frame_path):
            test_frame = cv2.imread(test_frame_path)
            height, width = test_frame.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_file, fourcc, 2.0, (width, height))
            
            # Add frames to video
            for i in range(1, 6):
                frame_file = f"{TEST_CONFIG['OUTPUT_PATH']}/composed_frame_{i:03d}.png"
                if os.path.exists(frame_file):
                    frame = cv2.imread(frame_file)
                    video_writer.write(frame)
            
            video_writer.release()
            print(f"   Test video saved to: {video_file}")
        else:
            print("   ⚠️ Skipping video creation - no composed frames found")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all component tests"""
    print("🚀 RAPIDO COMPONENT TESTING")
    print("=" * 60)
    print("Testing Rapido components (excluding SyncTalk inference)")
    print()
    
    # Create necessary directories
    for path in [TEST_CONFIG['OUTPUT_PATH'], TEST_CONFIG['TEMP_PATH']]:
        os.makedirs(path, exist_ok=True)
    
    # Run tests
    results = {}
    
    results['audio_generation'] = await test_1_audio_generation()
    results['audio_processing'] = await test_2_audio_processing()
    results['frame_composition'] = await test_3_frame_composition()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All available components working correctly!")
        print("\nNext steps:")
        print("1. Set up SyncTalk model files for full inference testing")
        print("2. Start SyncTalk server")
        print("3. Run end-to-end Rapido workflow")
    else:
        print("⚠️ Some components need attention before proceeding")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
