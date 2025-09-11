import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # 11Labs Configuration
    ELEVENLABS_API_KEY = os.getenv('ELEVEN_API_KEY') or os.getenv('ELEVENLABS_API_KEY')
    ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', 'pNInz6obpgDQGcFmaJgB')  # Default to Adam voice
    
    # SyncTalk Server Configuration - LOCAL SERVER
    SYNCTALK_WEBSOCKET_URL = os.getenv('SYNCTALK_WEBSOCKET_URL', 'ws://35.172.212.10:8000/ws')
    SYNCTALK_SERVER_URL = os.getenv('SYNCTALK_SERVER_URL', 'http://35.172.212.10:8000')
    
    # Frame Configuration
    FRAME_RATE = int(os.getenv('FRAME_RATE', '30'))  # FPS for output video
    AVATAR_OVERLAY_POSITION = os.getenv('AVATAR_OVERLAY_POSITION', 'bottom-right')  # Position for avatar overlay
    AVATAR_SCALE = float(os.getenv('AVATAR_SCALE', '0.5'))  # Scale factor for avatar frames (increased)
    
    # File Paths
    INPUT_DATA_PATH = os.getenv('INPUT_DATA_PATH', '../test1.json')
    SLIDE_FRAMES_PATH = os.getenv('SLIDE_FRAMES_PATH', '../presentation_frames')
    OUTPUT_PATH = os.getenv('OUTPUT_PATH', './output')
    TEMP_PATH = os.getenv('TEMP_PATH', './temp')
    
    # Audio Configuration
    AUDIO_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', '22050'))
    AUDIO_FORMAT = os.getenv('AUDIO_FORMAT', 'mp3')
    
    # Video Configuration
    VIDEO_CODEC = os.getenv('VIDEO_CODEC', 'libx264')
    VIDEO_QUALITY = os.getenv('VIDEO_QUALITY', 'medium')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'rapido.log')

    # Dynamic frame capture settings
    USE_DYNAMIC_CAPTURE = os.getenv('USE_DYNAMIC_CAPTURE', 'True').lower() == 'true'
    CAPTURE_URL = os.getenv('CAPTURE_URL', 'http://localhost:5173/video-capture/81eceadf-2503-4915-a2bf-12eb252329e4')
