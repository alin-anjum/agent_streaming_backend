#!/usr/bin/env python3
"""
Test script to verify Rapido setup and configuration
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_environment():
    """Test environment configuration."""
    print("🔧 Testing Environment Configuration...")
    
    # Check for .env file
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found - using environment variables")
    
    # Check required environment variables
    required_vars = ['ELEVEN_API_KEY']
    optional_vars = ['SYNCTALK_SERVER_URL', 'ELEVENLABS_VOICE_ID']
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is missing (required)")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set (optional)")

def test_input_data():
    """Test input data availability."""
    print("\n📄 Testing Input Data...")
    
    # Check for test1.json
    test_file = Path(__file__).parent.parent / 'test1.json'
    if test_file.exists():
        print("✅ test1.json found")
        
        try:
            with open(test_file, 'r') as f:
                data = json.load(f)
            
            if 'slide_data' in data:
                print("✅ slide_data structure found")
                
                if 'narrationData' in data['slide_data']:
                    narration = data['slide_data']['narrationData']
                    text = narration.get('text', '')
                    duration = narration.get('totalDuration', 0)
                    tokens = narration.get('tokens', [])
                    
                    print(f"✅ Narration text: {len(text)} characters")
                    print(f"✅ Duration: {duration}ms ({duration/1000:.1f}s)")
                    print(f"✅ Timing tokens: {len(tokens)}")
                else:
                    print("❌ narrationData not found in slide_data")
            else:
                print("❌ slide_data not found in JSON")
                
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print("❌ test1.json not found")

def test_slide_frames():
    """Test slide frames availability."""
    print("\n🖼️  Testing Slide Frames...")
    
    frames_dir = Path(__file__).parent.parent / 'frames'
    if frames_dir.exists():
        print("✅ frames directory found")
        
        png_files = list(frames_dir.glob('frame_*.png'))
        if png_files:
            print(f"✅ Found {len(png_files)} PNG frame files")
            
            # Check naming convention
            frame_numbers = []
            for png_file in png_files:
                try:
                    # Extract frame number from filename
                    name = png_file.stem  # frame_00001
                    if name.startswith('frame_'):
                        num_str = name[6:]  # 00001
                        frame_num = int(num_str)
                        frame_numbers.append(frame_num)
                except ValueError:
                    pass
            
            if frame_numbers:
                frame_numbers.sort()
                print(f"✅ Frame range: {min(frame_numbers)} to {max(frame_numbers)}")
                
                # Check for gaps
                expected_frames = set(range(min(frame_numbers), max(frame_numbers) + 1))
                actual_frames = set(frame_numbers)
                missing = expected_frames - actual_frames
                
                if missing:
                    print(f"⚠️  Missing frame numbers: {sorted(list(missing))[:10]}...")
                else:
                    print("✅ No gaps in frame sequence")
            else:
                print("⚠️  No properly named frame files found")
        else:
            print("❌ No PNG files found in frames directory")
    else:
        print("❌ frames directory not found")

def test_dependencies():
    """Test Python dependencies."""
    print("\n📦 Testing Python Dependencies...")
    
    dependencies = [
        'aiohttp',
        'websockets',
        'cv2',
        'PIL',
        'numpy',
        'elevenlabs',
        'ffmpeg'
    ]
    
    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif dep == 'PIL':
                from PIL import Image
                print("✅ Pillow (PIL)")
            elif dep == 'ffmpeg':
                import ffmpeg
                print("✅ ffmpeg-python")
            else:
                module = __import__(dep)
                if hasattr(module, '__version__'):
                    print(f"✅ {dep}: {module.__version__}")
                else:
                    print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} not found")

def test_output_directory():
    """Test output directory."""
    print("\n📁 Testing Output Directory...")
    
    output_dir = Path(__file__).parent / 'output'
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Created output directory")
        except Exception as e:
            print(f"❌ Cannot create output directory: {e}")
    else:
        print("✅ Output directory exists")
    
    # Test write permissions
    try:
        test_file = output_dir / 'test_write.tmp'
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Output directory is writable")
    except Exception as e:
        print(f"❌ Cannot write to output directory: {e}")

def test_rapido_modules():
    """Test Rapido module imports."""
    print("\n🐍 Testing Rapido Modules...")
    
    modules_to_test = [
        ('config.config', 'Config'),
        ('src.data_parser', 'SlideDataParser'),
        ('src.tts_client', 'ElevenLabsTTSClient'),
        ('src.synctalk_client', 'SyncTalkWebSocketClient'),
        ('src.frame_processor', 'FrameOverlayEngine'),
        ('src.timing_sync', 'TimingSynchronizer'),
        ('src.video_generator', 'VideoGenerator'),
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"❌ Cannot import {module_name}: {e}")
        except AttributeError as e:
            print(f"❌ {class_name} not found in {module_name}: {e}")

def main():
    """Run all tests."""
    print("🚀 Rapido Setup Test")
    print("=" * 50)
    
    test_environment()
    test_input_data()
    test_slide_frames()
    test_dependencies()
    test_output_directory()
    test_rapido_modules()
    
    print("\n" + "=" * 50)
    print("🏁 Setup test completed!")
    print("\nIf you see any ❌ errors above, please fix them before running Rapido.")
    print("⚠️  warnings are optional but recommended to fix.")

if __name__ == "__main__":
    main()
