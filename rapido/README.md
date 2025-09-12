# Rapido - Real-time Avatar Presentation System

## Refactored Production System v2.0

Rapido is a comprehensive real-time avatar presentation system that creates dynamic, interactive video presentations by combining slide content with AI-generated avatars. This refactored version features a modular architecture, comprehensive logging, security features, and production-ready components.

## 🚀 Features

### Core Capabilities
- **Real-time Avatar Generation**: Integration with SyncTalk for AI-powered talking avatars
- **Dynamic Video Composition**: Advanced frame composition with chroma key support
- **Live Streaming**: LiveKit integration for real-time video streaming
- **Text-to-Speech**: ElevenLabs integration for high-quality narration
- **Slide Processing**: Automated slide frame processing and timing

### Production Features
- **Modular Architecture**: Clean separation of concerns with interfaces and services
- **Comprehensive Logging**: Date-based log rotation with structured JSON logging
- **Security**: JWT authentication, input validation, rate limiting
- **Monitoring**: Real-time FPS metrics, performance tracking, error monitoring
- **Testing**: Full unit and integration test suites
- **Error Handling**: Robust error handling with detailed logging

### Logging Capabilities
The system provides detailed logging of:
1. **Lesson ID** for each processing session
2. **Starting timestamps** and processing events
3. **FPS metrics** from slide frames, SyncTalk, composer, and LiveKit
4. **Audio chunks** streamed to SyncTalk with timing information
5. **Event tracking** from both backend and frontend components
6. **Performance data** with timing and resource usage
7. **Error details** with context and stack traces

## 🏗️ Architecture

### Modular Components

```
src/
├── core/                    # Core interfaces and shared components
│   ├── interfaces.py        # Service interfaces and data models
│   ├── exceptions.py        # Custom exception classes
│   ├── logging_manager.py   # Advanced logging with date rotation
│   ├── metrics.py          # Performance metrics and FPS counters
│   └── security.py         # Authentication and security utilities
│
├── services/               # Business logic services
│   ├── audio_service.py    # Audio processing and TTS
│   ├── video_service.py    # Video processing and composition
│   ├── synctalk_service.py # SyncTalk integration
│   ├── livekit_service.py  # LiveKit streaming
│   ├── data_service.py     # Data parsing and frame management
│   └── orchestrator_service.py # Main coordination service
│
├── rapido_refactored_main.py  # Main application entry point
└── rapido_api_refactored.py   # FastAPI web service
```

### Key Interfaces

- **IAudioProcessor**: Audio processing and optimization
- **IVideoProcessor**: Video frame processing
- **ISyncTalkClient**: SyncTalk communication
- **ILiveKitPublisher**: LiveKit streaming
- **IDataParser**: Slide data parsing
- **ITTSClient**: Text-to-speech synthesis

## 📋 Requirements

### Dependencies
```bash
# Core requirements
fastapi>=0.116.1
uvicorn>=0.35.0
numpy>=1.26.0
opencv-python>=4.11.0
pillow>=11.3.0
asyncio

# Audio/Video processing
librosa>=0.11.0
soundfile>=0.13.1
ffmpeg-python>=0.2.0

# AI Services
elevenlabs>=2.13.0
transformers>=4.36.0

# Communication
livekit>=1.0.12
websockets>=15.0.1
aiohttp>=3.12.15

# Security
pyjwt>=2.10.1
cryptography>=45.0.7
bcrypt>=4.3.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### Environment Variables
```bash
# Required
JWT_SECRET=your_jwt_secret_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
LIVEKIT_URL=your_livekit_server_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Optional
SYNCTALK_SERVER_URL=http://35.172.212.10:8000
ELEVENLABS_VOICE_ID=pNInz6obpgDQGcFmaJgB
SLIDE_FRAMES_PATH=./presentation_frames
RAPIDO_LOG_DIR=./logs
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd rapido

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Configuration
```bash
# Copy configuration template
cp config/development.json config/local.json

# Edit configuration for your environment
vim config/local.json
```

### 3. Running the CLI Application
```bash
# Process a lesson
python src/rapido_refactored_main.py --lesson-id "lesson_001" --slide-data "./data/lesson_001.json"

# Check status
python src/rapido_refactored_main.py --lesson-id "lesson_001" --status-only

# Use custom configuration
python src/rapido_refactored_main.py --config config/local.json --lesson-id "lesson_001"
```

### 4. Running the API Server
```bash
# Start the API server
python src/rapido_api_refactored.py --host 0.0.0.0 --port 8080

# With auto-reload for development
python src/rapido_api_refactored.py --host 0.0.0.0 --port 8080 --reload
```

## 🔧 API Usage

### Process a Lesson
```bash
curl -X POST "http://localhost:8080/process_lesson" \
  -H "Content-Type: application/json" \
  -d '{
    "lesson_id": "lesson_001",
    "slide_data_path": "./data/lesson_001.json",
    "enable_tts": true,
    "enable_synctalk": true,
    "enable_livekit": true
  }'
```

### Check System Status
```bash
curl "http://localhost:8080/status"
```

### WebSocket for Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/lesson/lesson_001');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Update:', data);
};
```

## 📊 Monitoring and Logging

### Log Files
The system creates date-based log files in the configured log directory:
- `rapido_YYYY-MM-DD.jsonl`: Main application logs
- `rapido_performance_YYYY-MM-DD.jsonl`: Performance metrics

### Log Format
Each log entry contains structured JSON data:
```json
{
  "timestamp": "2025-01-10T14:30:25.123456",
  "level": "INFO",
  "logger": "rapido.orchestrator",
  "message": "Processing lesson started",
  "lesson_id": "lesson_001",
  "event_type": "lesson_start",
  "event_source": "backend",
  "performance_data": {
    "processing_time": 0.123,
    "frame_count": 25
  }
}
```

### Metrics Dashboard
The system tracks comprehensive metrics:
- **FPS Metrics**: Real-time frame rates for all components
- **Processing Stats**: Timing and throughput statistics
- **Error Rates**: Error frequency and types
- **Resource Usage**: Memory and CPU utilization

## 🧪 Testing

### Run Unit Tests
```bash
# Run all unit tests
pytest rapido/tests/unit/ -v

# Run specific test file
pytest rapido/tests/unit/test_core_components.py -v

# Run with coverage
pytest rapido/tests/unit/ --cov=src --cov-report=html
```

### Run Integration Tests
```bash
# Run integration tests
pytest rapido/tests/integration/ -v

# Run end-to-end tests
pytest rapido/tests/integration/test_system_integration.py::TestEndToEndWorkflow -v
```

### Test Configuration
```bash
# Set test environment variables
export RAPIDO_TEST_MODE=true
export JWT_SECRET=test_secret_key
```

## 🔒 Security Features

### Authentication
- JWT-based session management
- Configurable token expiration
- Secure token validation

### Input Validation
- Lesson ID format validation
- File path traversal prevention
- Audio data format validation
- JSON structure validation

### Rate Limiting
- Per-client request rate limiting
- Configurable limits per minute
- Automatic client blocking

### Data Security
- Input sanitization
- Secure filename handling
- Error message sanitization

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "src/rapido_api_refactored.py", "--host", "0.0.0.0", "--port", "8080"]
```

### Production Configuration
```bash
# Use production config
export RAPIDO_CONFIG=config/production.json

# Set production environment variables
export JWT_SECRET=secure_production_secret
export RAPIDO_LOG_DIR=/var/log/rapido

# Start with production settings
python src/rapido_api_refactored.py --host 0.0.0.0 --port 8080
```

### Systemd Service
```ini
[Unit]
Description=Rapido Avatar Presentation System
After=network.target

[Service]
Type=simple
User=rapido
WorkingDirectory=/opt/rapido
Environment=RAPIDO_CONFIG=config/production.json
ExecStart=/opt/rapido/venv/bin/python src/rapido_api_refactored.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## 📈 Performance Optimization

### Recommended Settings
- **CPU**: 4+ cores for concurrent processing
- **Memory**: 8GB+ RAM for video processing
- **Storage**: SSD for frame caching
- **Network**: High bandwidth for streaming

### Optimization Tips
1. **Frame Caching**: Enable frame caching for repeated lessons
2. **Buffer Sizes**: Tune buffer sizes based on available memory
3. **Concurrent Processing**: Limit concurrent lessons based on resources
4. **Log Rotation**: Configure appropriate log rotation intervals
5. **Metrics Collection**: Adjust metrics collection frequency

## 🐛 Troubleshooting

### Common Issues

#### SyncTalk Connection Failed
```bash
# Check SyncTalk server status
curl http://35.172.212.10:8000/status

# Verify network connectivity
ping 35.172.212.10
```

#### LiveKit Connection Issues
```bash
# Verify LiveKit credentials
export LIVEKIT_URL=your_url
export LIVEKIT_API_KEY=your_key
export LIVEKIT_API_SECRET=your_secret

# Test connection
python -c "from src.services.livekit_service import LiveKitService; print('LiveKit available')"
```

#### Permission Issues
```bash
# Fix log directory permissions
sudo mkdir -p /var/log/rapido
sudo chown rapido:rapido /var/log/rapido

# Fix frame directory permissions
sudo chown -R rapido:rapido ./presentation_frames
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python src/rapido_refactored_main.py --lesson-id "debug_lesson" -v
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/
```

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features
- Update documentation for changes

## 📄 License

[Specify license here]

## 📞 Support

For questions, issues, or feature requests:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation
- Examine log files for detailed error information