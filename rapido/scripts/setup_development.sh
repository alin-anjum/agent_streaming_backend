#!/bin/bash
# Development environment setup script for Rapido

set -e

echo "🛠️  Setting up Rapido Development Environment"
echo "=============================================="

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
echo "📍 Python version: $python_version"

if [[ "$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')" < "3.9" ]]; then
    echo "❌ Python 3.9+ is required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install \
    pytest>=7.0.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.0.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0 \
    pre-commit>=3.0.0

# Create directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p output
mkdir -p temp
mkdir -p presentation_frames
mkdir -p data/lessons

# Create environment file
echo "📝 Creating environment file..."
cat > .env << EOF
# Development environment variables
JWT_SECRET=development_secret_key_change_in_production
RAPIDO_ENV=development

# API Keys (replace with your actual keys)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=pNInz6obpgDQGcFmaJgB

# LiveKit (optional for development)
LIVEKIT_URL=your_livekit_server_url
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# SyncTalk
SYNCTALK_SERVER_URL=http://localhost:8000

# Paths
RAPIDO_LOG_DIR=./logs
SLIDE_FRAMES_PATH=./presentation_frames
INPUT_DATA_PATH=./test1.json
OUTPUT_PATH=./output

# Development settings
LOG_LEVEL=DEBUG
RAPIDO_WORKERS=1
EOF

# Create sample test data
echo "📝 Creating sample test data..."
cat > test1.json << EOF
{
  "slide_data": {
    "slideId": "development_lesson_001",
    "narrationData": {
      "text": "Welcome to the Rapido development environment. This is a sample narration for testing the system.",
      "timing": [
        {"start": 0.0, "end": 3.0, "text": "Welcome to the Rapido development environment."},
        {"start": 3.0, "end": 6.0, "text": "This is a sample narration for testing the system."}
      ]
    }
  }
}
EOF

# Create sample presentation frames
echo "🖼️  Creating sample presentation frames..."
mkdir -p presentation_frames/development_lesson_001

# Create simple colored images for testing
python3 << EOF
import numpy as np
from PIL import Image
import os

os.makedirs('presentation_frames/development_lesson_001', exist_ok=True)

for i in range(5):
    # Create a simple colored rectangle
    color = (50 + i * 40, 100 + i * 30, 150 + i * 20)
    image = Image.new('RGB', (854, 480), color)
    
    # Add some text
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text = f"Development Slide {i+1}"
        draw.text((50, 50), text, fill=(255, 255, 255), font=font)
    except ImportError:
        pass  # Skip text if PIL fonts not available
    
    image.save(f'presentation_frames/development_lesson_001/slide_{i+1:03d}.png')

print("Sample presentation frames created")
EOF

# Setup git hooks (if git is available and this is a git repo)
if [ -d ".git" ] && command -v git &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
fi

# Create development scripts
echo "📝 Creating development scripts..."
cat > scripts/dev_run_api.sh << EOF
#!/bin/bash
# Development API server runner

source venv/bin/activate
export RAPIDO_CONFIG=config/development.json
python src/rapido_api_refactored.py --host 127.0.0.1 --port 8080 --reload
EOF

cat > scripts/dev_run_cli.sh << EOF
#!/bin/bash
# Development CLI runner

source venv/bin/activate
export RAPIDO_CONFIG=config/development.json
python src/rapido_refactored_main.py --lesson-id development_lesson_001 --slide-data test1.json
EOF

chmod +x scripts/dev_run_api.sh scripts/dev_run_cli.sh

# Test the setup
echo "🧪 Testing the setup..."
source venv/bin/activate

# Quick import test
python3 -c "
import sys
sys.path.append('src')
try:
    from core.logging_manager import LoggingManager
    from core.metrics import MetricsCollector
    from services.audio_service import AudioProcessorService
    print('✅ Core modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run the API server: ./scripts/dev_run_api.sh"
echo "3. Or run CLI: ./scripts/dev_run_cli.sh"
echo "4. Run tests: ./scripts/run_tests.sh"
echo ""
echo "📚 Useful commands:"
echo "   source venv/bin/activate  # Activate virtual environment"
echo "   python src/rapido_api_refactored.py --help  # API help"
echo "   python src/rapido_refactored_main.py --help  # CLI help"
echo ""
echo "🔧 Development tools:"
echo "   black src/  # Format code"
echo "   isort src/  # Sort imports"
echo "   flake8 src/  # Lint code"
echo "   pytest tests/  # Run tests"
