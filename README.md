# Agent Streaming Backend

A comprehensive system for generating AI-powered avatar videos with real-time audio streaming and frame composition.

## Overview

This project combines multiple technologies to create a complete pipeline for generating avatar-based video content:

- **Rapido Client**: Orchestrates the entire workflow
- **SyncTalk_2D Server**: Provides AI-powered avatar generation using NeRF technology
- **ElevenLabs Integration**: Real-time text-to-speech streaming
- **Frame Composition**: Overlays avatar frames onto slide backgrounds

## System Architecture

```
JSON Input → ElevenLabs TTS → Audio Stream → SyncTalk Server → Avatar Frames → Frame Compositor → Final Video
```

## Features

- Real-time audio streaming from ElevenLabs
- AI-powered avatar generation with SyncTalk
- Automatic frame composition and overlay
- Support for multiple avatar models
- CUDA acceleration for GPU processing
- WebSocket-based communication
- Comprehensive error handling and logging

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alin-anjum/agent_streaming_backend.git
cd agent_streaming_backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy and edit the environment variables
export ELEVEN_API_KEY="your_elevenlabs_api_key"
export ELEVENLABS_VOICE_ID="pNInz6obpgDQGcFmaJgB"
```

### Running the System

1. Start the SyncTalk server:
```bash
cd SyncTalk_2D
python synctalk_fastapi.py
```

2. Test the components:
```bash
python test_rapido_components.py
```

3. Run the full pipeline:
```bash
cd rapido
python src/rapido_main.py
```

## Project Structure

```
agent_streaming_backend/
├── rapido/                     # Main client application
│   ├── config/                # Configuration files
│   ├── src/                   # Source code
│   │   ├── data_parser.py     # JSON data parsing
│   │   ├── tts_client.py      # ElevenLabs integration
│   │   ├── frame_processor.py # Frame composition
│   │   └── rapido_main.py     # Main orchestrator
│   └── requirements.txt       # Python dependencies
├── SyncTalk_2D/               # Avatar generation server
│   ├── synctalk_fastapi.py    # FastAPI server
│   ├── avatar_config.json     # Avatar configurations
│   └── inference_system/      # AI inference logic
├── frames/                    # Input slide frames
├── test1.json                # Sample input data
├── test_rapido_components.py  # Component tests
└── requirements.txt          # Combined dependencies
```

## Configuration

### Avatar Models

The system supports multiple avatar models configured in `SyncTalk_2D/avatar_config.json`:

- **enrique_torres**: Default model with Alin-cc dataset
- Additional models can be added with proper checkpoint and video files

### Audio Settings

- **Sample Rate**: 22,050 Hz (configurable)
- **Format**: MP3 (default)
- **Voice**: ElevenLabs voice ID (configurable)

### Video Output

- **Frame Rate**: 30 FPS (configurable)
- **Resolution**: 1920x1080 (configurable)
- **Codec**: H.264 (recommended for compatibility)

## API Endpoints

### SyncTalk Server (Port 8001)

- `GET /`: Server status
- `WebSocket /ws`: Real-time avatar generation
- Avatar inference with audio input streaming

## Testing

The project includes comprehensive tests:

1. **Audio Generation**: Tests ElevenLabs TTS integration
2. **Frame Composition**: Tests avatar overlay on slide frames
3. **End-to-End**: Complete workflow testing

Run tests with:
```bash
python test_rapido_components.py
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA compatibility issues:

1. Check your GPU architecture:
```bash
nvidia-smi
```

2. Install compatible PyTorch:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Port Conflicts

If port 8001 is in use:

1. Find the process:
```bash
netstat -ano | findstr :8001
```

2. Kill the process:
```bash
taskkill /F /PID <process_id>
```

### Unicode Errors (Windows)

The system handles Unicode logging errors automatically by removing emojis from log messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- SyncTalk for avatar generation technology
- ElevenLabs for text-to-speech services
- FastAPI for the web framework
- PyTorch for deep learning capabilities
