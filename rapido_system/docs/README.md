# AI Avatar Streaming Backend

A comprehensive real-time AI avatar presentation system with LiveKit streaming, combining ElevenLabs TTS, SyncTalk_2D avatar generation, and dynamic frame composition.

## 🚀 Overview

This project creates a complete pipeline for generating and streaming AI-powered avatar videos in real-time:

- **Rapido Client**: Orchestrates the entire real-time pipeline
- **SyncTalk_2D Server**: AI-powered avatar generation using NeRF technology
- **ElevenLabs Integration**: Real-time text-to-speech streaming
- **LiveKit Streaming**: Real-time video/audio broadcasting
- **Frame Composition**: Dynamic overlay of avatar frames onto slide backgrounds
- **React Frontend**: Standalone LiveKit viewer with Creatium-style UI

## 🏗️ System Architecture

```
JSON Input → ElevenLabs TTS Stream → SyncTalk Server → Avatar Frames → Frame Compositor → LiveKit Stream → React Frontend
```

## ✨ Features

- **Real-time streaming** with ElevenLabs TTS and SyncTalk inference
- **LiveKit integration** for low-latency video/audio streaming
- **Chroma keying** (green screen removal) with configurable parameters
- **Dynamic frame composition** with slide looping
- **Standalone React frontend** with modern UI
- **JWT-based authentication** for secure streaming
- **CUDA acceleration** for GPU processing
- **WebSocket communication** for real-time data flow
- **Comprehensive error handling** and logging

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 16+ (for frontend)
- CUDA-compatible GPU (recommended)
- LiveKit account and credentials

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/alin-anjum/agent_streaming_backend.git
cd agent_streaming_backend
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
cd rapido && pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
export ELEVEN_API_KEY="your_elevenlabs_api_key"
export ELEVENLABS_VOICE_ID="pNInz6obpgDQGcFmaJgB"
export LIVEKIT_URL="wss://your-livekit-url"
export LIVEKIT_API_KEY="your_api_key"
export LIVEKIT_API_SECRET="your_api_secret"
```

4. **Install frontend dependencies:**
```bash
cd avatar-frontend
npm install
```

### Running the System

#### Option 1: Automated Startup (Recommended)
```bash
cd rapido
python start_system.py
```
This will automatically start SyncTalk server, wait for it to be ready, then launch Rapido.

#### Option 2: Manual Startup

1. **Start SyncTalk server:**
```bash
cd SyncTalk_2D
python synctalk_fastapi.py
```

2. **Start Rapido client:**
```bash
cd rapido
python run_rapido.py
```

3. **Start frontend (separate terminal):**
```bash
cd avatar-frontend
node token-server.js &  # Start JWT token server
npm start              # Start React app
```

### Testing the Setup

```bash
# Test environment and dependencies
cd tests
python test_setup.py

# Run component tests
python test_rapido_components.py
```

## 📁 Project Structure

```
agent_streaming_backend/
├── rapido/                          # 🚀 Main AI avatar pipeline
│   ├── src/
│   │   ├── rapido_main.py          # Main orchestrator with LiveKit
│   │   ├── tts_client.py           # ElevenLabs real-time streaming
│   │   ├── synctalk_client.py      # WebSocket client for SyncTalk
│   │   ├── frame_processor.py      # Frame composition engine
│   │   └── ...
│   ├── run_rapido.py               # Simple launcher
│   ├── start_system.py             # Full system orchestrator
│   └── requirements.txt            # Python dependencies
├── SyncTalk_2D/                     # 🤖 Avatar inference server
│   ├── synctalk_fastapi.py         # FastAPI WebSocket server
│   ├── avatar_config.json          # Avatar & chroma key config
│   ├── chroma_key.py              # Green screen processing
│   └── inference_system/           # AI inference pipeline
├── avatar-frontend/                 # 🎨 Standalone React viewer
│   ├── src/
│   │   ├── App.js                  # LiveKit viewer component
│   │   └── App.css                 # Creatium-style UI
│   ├── token-server.js             # JWT token generator
│   └── package.json                # React dependencies
├── tests/                          # 🧪 All test files
│   ├── test_setup.py               # Environment validation
│   ├── test_rapido_components.py   # Component tests
│   └── test_*.py                   # Functional tests
├── frames/                         # Input slide frames (PNG)
├── test1.json                      # Sample presentation data
└── README.md
```

## ⚙️ Configuration

### Avatar Models

Configure in `SyncTalk_2D/avatar_config.json`:

```json
{
  "enrique_torres": {
    "model_path": "model/checkpoints/enrique_torres.pth",
    "video_path": "model/Alin-cc-dataset/enrique_torres.mp4",
    "chroma_key": {
      "color_threshold": 35,
      "edge_blur": 0.4,
      "despill_factor": 0.9
    }
  }
}
```

### LiveKit Settings

```bash
LIVEKIT_URL="wss://your-project.livekit.cloud"
LIVEKIT_API_KEY="your_api_key"
LIVEKIT_API_SECRET="your_secret"
```

### Audio/Video Settings

- **Audio**: 16kHz PCM (converted from ElevenLabs MP3)
- **Video**: 1920x1080, 30 FPS
- **Streaming**: Real-time frame-by-frame publishing

## 🎛️ API Endpoints

### SyncTalk Server (Port 8000)
- `GET /status`: Server health check
- `WebSocket /audio_to_video`: Real-time avatar inference
- Parameters: `avatar_name`, `sample_rate`

### Frontend Token Server (Port 3001)
- `GET /token`: Generate LiveKit JWT tokens
- Parameters: `room`, `identity`

### React Frontend (Port 3000)
- LiveKit room viewer with audio/video tracks
- Automatic token fetching and connection
- Modern Creatium-style interface

## 🧪 Testing

Comprehensive test suite organized in `tests/` folder:

```bash
# Environment validation
python tests/test_setup.py

# Component testing
python tests/test_rapido_components.py

# End-to-end testing
python tests/test_end_to_end_synctalk.py
```

## 🔧 Advanced Usage

### Custom Chroma Key Settings

Update `SyncTalk_2D/chroma_key.py` or `avatar_config.json`:
- `color_threshold`: 35 (green detection sensitivity)
- `edge_blur`: 0.4 (edge smoothing)
- `despill_factor`: 0.9 (green spill removal)

### Slide Frame Management

- Place PNG frames in `frames/` directory
- Naming: `frame_00001.png`, `frame_00002.png`, etc.
- Automatic looping if fewer slides than video duration

### LiveKit Room Management

```javascript
// Frontend automatically connects to room
const LIVEKIT_URL = 'wss://your-project.livekit.cloud';
const room = 'avatar-presentation';
```

## 🐛 Troubleshooting

### CUDA Issues
```bash
# Check GPU compatibility
nvidia-smi

# Install compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### LiveKit Connection Issues
- Verify API credentials in environment variables
- Check JWT token generation in browser dev tools
- Ensure WebSocket connection to LiveKit cloud

### Port Conflicts
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <process_id>

# Linux/Mac
lsof -i :8000
kill -9 <process_id>
```

### Frontend Issues
```bash
# Clear React cache
cd avatar-frontend
rm -rf node_modules package-lock.json
npm install
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python tests/test_setup.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SyncTalk** for avatar generation technology
- **ElevenLabs** for real-time text-to-speech services
- **LiveKit** for real-time streaming infrastructure
- **FastAPI** for the WebSocket server framework
- **React** for the frontend framework
- **PyTorch** for deep learning capabilities

## 🔗 Links

- [ElevenLabs API Documentation](https://elevenlabs.io/docs)
- [LiveKit Documentation](https://docs.livekit.io/)
- [SyncTalk Research Paper](https://github.com/ZiqiaoPeng/SyncTalk)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Built with ❤️ for real-time AI avatar streaming**