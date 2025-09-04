# Rapido - Real-time Avatar Presentation Integration with Dynamic Overlay

Rapido is a comprehensive system that creates engaging video presentations by combining slide content with AI-generated avatar narration. It processes slide data, generates speech using ElevenLabs TTS, streams audio to a SyncTalk server for avatar generation, and composites the results into a final video.

## Features

- **Slide Data Processing**: Parses JSON slide data with timing information
- **TTS Integration**: Streams text-to-speech using ElevenLabs API
- **Real-time Avatar Generation**: Communicates with SyncTalk server via WebSocket
- **Frame Composition**: Overlays avatar frames onto slide frames
- **Timing Synchronization**: Ensures perfect sync between audio, avatar, and slides
- **Video Generation**: Creates final MP4 output with synchronized audio/video

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Slide Data    │───▶│   Data Parser    │───▶│  Timing Sync    │
│   (JSON)        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ElevenLabs    │───▶│   TTS Client     │───▶│  Audio Stream   │
│   TTS API       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SyncTalk      │◀───│   WebSocket      │◀───│  Audio Chunks   │
│   Server        │    │   Client         │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Avatar Frames  │───▶│  Frame Overlay   │───▶│  Final Video    │
│                 │    │  Engine          │    │  Generator      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Slide Frames   │
                       │  (PNG files)    │
                       └─────────────────┘
```

## Installation

1. **Clone or create the Rapido directory structure**

2. **Install Python dependencies**:
   ```bash
   cd rapido
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required for video generation):
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Configuration

### Required Environment Variables

- `ELEVEN_API_KEY`: Your ElevenLabs API key
- `SYNCTALK_SERVER_URL`: WebSocket URL for SyncTalk server (e.g., `ws://localhost:8000`)

### Optional Configuration

- `ELEVENLABS_VOICE_ID`: Voice ID for TTS (default: Adam voice)
- `FRAME_RATE`: Output video frame rate (default: 30)
- `AVATAR_OVERLAY_POSITION`: Avatar position on slide (default: "bottom-right")
- `AVATAR_SCALE`: Avatar scale factor (default: 0.3)
- `VIDEO_CODEC`: Video codec for output (default: "libx264")

## Usage

### Basic Usage

```bash
# Using the simple runner
python run_rapido.py

# Or using the main script directly
python src/rapido_main.py
```

### Advanced Usage

```bash
python src/rapido_main.py \
  --input ../test1.json \
  --frames ../frames \
  --output ./output \
  --synctalk-url ws://localhost:8000 \
  --api-key your_elevenlabs_key \
  --frame-rate 30 \
  --verbose
```

### Command Line Options

- `--input, -i`: Input JSON file path
- `--frames, -f`: Slide frames directory
- `--output, -o`: Output directory
- `--synctalk-url`: SyncTalk server URL
- `--api-key`: ElevenLabs API key
- `--voice-id`: ElevenLabs voice ID
- `--frame-rate`: Output frame rate
- `--verbose, -v`: Enable verbose logging

## Input Data Format

Rapido expects slide data in the following JSON format:

```json
{
  "slide_data": {
    "id": "slide-id",
    "narrationData": {
      "text": "Your narration text here...",
      "totalDuration": 45662,
      "tokens": [
        {
          "id": "word-id",
          "text": "word",
          "startTime": 1000,
          "endTime": 1500,
          "duration": 500,
          "type": "word"
        }
      ]
    },
    "objects": [
      // Slide objects with animation triggers
    ]
  }
}
```

## Slide Frames

Place your slide frames in the frames directory as PNG files:
- `frame_00000.png`
- `frame_00001.png`
- `frame_00002.png`
- etc.

## SyncTalk Server

Rapido requires a SyncTalk server running to generate avatar frames. The server should:

1. Accept WebSocket connections
2. Receive audio chunks via WebSocket
3. Generate avatar frames based on audio
4. Stream avatar frames back via WebSocket

### Expected WebSocket Messages

**Outgoing (to SyncTalk):**
```json
{
  "type": "audio_chunk",
  "data": {
    "chunk_data": "base64_encoded_audio",
    "chunk_index": 0,
    "is_final": false,
    "timestamp": 1234567890.123
  }
}
```

**Incoming (from SyncTalk):**
```json
{
  "type": "avatar_frame",
  "data": {
    "frame_data": "base64_encoded_image",
    "frame_index": 0,
    "timestamp": 1234567890.123
  }
}
```

## Output

Rapido generates:

1. **Final Video**: `output/rapido_output.mp4`
   - Synchronized audio and video
   - Avatar overlaid on slide frames
   - Proper timing alignment

2. **Logs**: `rapido.log`
   - Detailed processing information
   - Error messages and debugging info

## Troubleshooting

### Common Issues

1. **"Failed to connect to SyncTalk server"**
   - Ensure SyncTalk server is running
   - Check WebSocket URL in configuration
   - Verify network connectivity

2. **"No audio data generated"**
   - Check ElevenLabs API key
   - Verify narration text is not empty
   - Check internet connectivity

3. **"Error loading slide frames"**
   - Ensure frames directory exists
   - Check PNG file naming convention
   - Verify file permissions

4. **FFmpeg errors during video generation**
   - Install FFmpeg system-wide
   - Check video codec compatibility
   - Verify output directory permissions

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python src/rapido_main.py --verbose
```

## Development

### Project Structure

```
rapido/
├── config/
│   └── config.py          # Configuration management
├── src/
│   ├── data_parser.py     # JSON slide data parser
│   ├── tts_client.py      # ElevenLabs TTS client
│   ├── synctalk_client.py # WebSocket client for SyncTalk
│   ├── frame_processor.py # Frame overlay engine
│   ├── timing_sync.py     # Timing synchronization
│   ├── video_generator.py # Video output generation
│   └── rapido_main.py     # Main orchestrator
├── output/                # Generated videos
├── temp/                  # Temporary files
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
└── run_rapido.py         # Simple runner script
```

### Key Components

1. **SlideDataParser**: Extracts narration text and timing data
2. **ElevenLabsTTSClient**: Streams TTS audio from ElevenLabs
3. **SyncTalkWebSocketClient**: Manages WebSocket communication
4. **FrameOverlayEngine**: Composites avatar onto slide frames
5. **TimingSynchronizer**: Coordinates timing between components
6. **VideoGenerator**: Creates final MP4 output

## License

This project is for educational and development purposes. Ensure you have proper licenses for:
- ElevenLabs TTS API usage
- SyncTalk server usage
- Any slide content and media assets

## Contributing

When contributing to Rapido:

1. Follow Python PEP 8 style guidelines
2. Add logging for important operations
3. Include error handling for external APIs
4. Update documentation for new features
5. Test with various slide data formats
