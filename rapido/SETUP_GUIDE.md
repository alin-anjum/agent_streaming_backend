# Complete Setup Guide: Rapido + SyncTalk System

This guide will help you set up the complete system with both Rapido (the client) and SyncTalk-FastAPI (the server) working together.

## 📁 Directory Structure

After following this guide, you'll have:
```
C:\TEST_STREAM\
├── test1.json              # Your slide data
├── frames/                 # Your slide frames (frame_00000.png, etc.)
├── rapido/                 # Rapido client (this project)
│   ├── src/               # All Rapido source code
│   ├── config/            # Configuration
│   ├── output/            # Generated videos
│   └── ...
└── SyncTalk-FastAPI/      # SyncTalk server
    ├── main.py            # SyncTalk core
    ├── synctalk_fastapi.py # FastAPI server
    ├── data/              # Model data
    ├── model/             # Trained models
    └── ...
```

## 🚀 Quick Start (Recommended)

### Step 1: Set Up SyncTalk Server

1. **Navigate to SyncTalk directory**:
   ```bash
   cd ../SyncTalk-FastAPI
   ```

2. **Install SyncTalk dependencies**:
   ```bash
   # Create conda environment
   conda create -n synctalk python==3.8.8
   conda activate synctalk
   
   # Install PyTorch (adjust CUDA version as needed)
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
   
   # Install other dependencies
   pip install -r requirements.txt
   
   # Install custom modules
   pip install ./freqencoder
   pip install ./shencoder
   pip install ./gridencoder
   pip install ./raymarching
   ```

3. **Download pre-trained models**:
   - Download [May.zip](https://drive.google.com/file/d/18Q2H612CAReFxBd9kxr-i1dD8U1AUfsV/view?usp=sharing) to `data/` folder
   - Download [trial_may.zip](https://drive.google.com/file/d/1C2639qi9jvhRygYHwPZDGs8pun3po3W7/view?usp=sharing) to `model/` folder
   - Unzip both files

4. **Configure SyncTalk**:
   ```bash
   # Edit config.cfg to set your model paths
   cp config.cfg config.cfg.backup  # Backup original
   ```

5. **Start SyncTalk FastAPI Server**:
   ```bash
   python synctalk_fastapi.py
   ```
   
   The server will start on `http://localhost:8000`

### Step 2: Set Up Rapido Client

1. **Navigate back to Rapido**:
   ```bash
   cd ../rapido
   ```

2. **Install Rapido dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   # Copy your .env file (already created with your API keys)
   # Make sure it includes:
   # ELEVEN_API_KEY=5247420f33ae186ddea9a70843d469ff
   # SYNCTALK_SERVER_URL=http://localhost:8000
   ```

4. **Test the setup**:
   ```bash
   python test_setup.py
   ```

5. **Run Rapido**:
   ```bash
   python quick_start.py
   ```

## 🔧 Detailed Configuration

### SyncTalk Configuration

Edit `../SyncTalk-FastAPI/config.cfg`:

```ini
[BASE]
model_path = model/trial_may
data_path = data/May
agent_config_file = https://your-agent-config-url.json

[MODELS]
model_may = model/trial_may
data_may = data/May
coordinates_may = 256,256
```

### Rapido Configuration

Your `.env` file should contain:

```bash
# ElevenLabs Configuration
ELEVEN_API_KEY=5247420f33ae186ddea9a70843d469ff
ELEVENLABS_VOICE_ID=pNInz6obpgDQGcFmaJgB

# SyncTalk Server Configuration  
SYNCTALK_SERVER_URL=http://localhost:8000

# Frame Configuration
FRAME_RATE=30
AVATAR_OVERLAY_POSITION=bottom-right
AVATAR_SCALE=0.3

# File Paths (relative to rapido directory)
INPUT_DATA_PATH=../test1.json
SLIDE_FRAMES_PATH=../frames
OUTPUT_PATH=./output
TEMP_PATH=./temp
```

## 🎬 How It Works

1. **Rapido** reads your slide data (`test1.json`) and extracts narration text
2. **Rapido** streams the text to ElevenLabs TTS API to generate speech
3. **Rapido** converts the audio to PCM format and streams it to **SyncTalk server**
4. **SyncTalk server** generates avatar video frames synchronized with the audio
5. **SyncTalk server** streams the video frames back to **Rapido**
6. **Rapido** overlays the avatar frames onto your slide frames
7. **Rapido** generates the final video with synchronized audio and visuals

## 🐛 Troubleshooting

### SyncTalk Issues

**"CUDA out of memory"**:
- Reduce batch size in SyncTalk configuration
- Close other GPU-intensive applications
- Use a GPU with more VRAM

**"Model not found"**:
- Ensure model files are downloaded and unzipped correctly
- Check paths in `config.cfg`
- Verify model names match configuration

**"Port 8000 already in use"**:
```bash
# Kill process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
python synctalk_fastapi.py --port 8001
```

### Rapido Issues

**"Failed to connect to SyncTalk server"**:
- Ensure SyncTalk server is running on `http://localhost:8000`
- Check firewall settings
- Verify server URL in `.env` file

**"ElevenLabs API error"**:
- Verify API key is correct
- Check internet connection
- Ensure you have API credits

**"No slide frames found"**:
- Verify `frames/` directory exists with PNG files
- Check file naming: `frame_00000.png`, `frame_00001.png`, etc.
- Ensure paths in `.env` are correct

## 📊 Performance Tips

### For Better Quality
- Use higher resolution slide frames (1920x1080)
- Ensure good quality narration text
- Use appropriate avatar scale (0.2-0.4 works well)

### For Better Performance
- Use SSD storage for faster file I/O
- Ensure adequate GPU memory (8GB+ recommended)
- Close unnecessary applications
- Use batch processing for multiple presentations

## 🔄 Workflow Example

```bash
# Terminal 1: Start SyncTalk Server
cd SyncTalk-FastAPI
conda activate synctalk
python synctalk_fastapi.py

# Terminal 2: Run Rapido Client
cd rapido
python quick_start.py

# Follow the interactive prompts
# Output video will be saved in rapido/output/
```

## 📝 Advanced Usage

### Custom Models
To use your own trained SyncTalk model:
1. Train model following SyncTalk documentation
2. Add model configuration to `config.cfg`
3. Update model name in Rapido configuration

### Custom Avatar Positioning
Modify avatar overlay settings in `.env`:
```bash
AVATAR_OVERLAY_POSITION=top-left    # top-left, top-right, bottom-left, bottom-right, center
AVATAR_SCALE=0.25                   # 0.1 to 1.0
```

### Batch Processing
Process multiple presentations by modifying input paths:
```bash
python src/rapido_main.py --input presentation1.json --output ./output/video1/
python src/rapido_main.py --input presentation2.json --output ./output/video2/
```

## 🆘 Getting Help

1. **Check logs**: Both Rapido and SyncTalk generate detailed logs
2. **Run tests**: Use `python test_setup.py` to verify configuration
3. **Verify models**: Ensure SyncTalk models are properly downloaded
4. **Check resources**: Monitor GPU memory and disk space
5. **Update dependencies**: Ensure all packages are up to date

## 🎯 Success Indicators

When everything is working correctly, you should see:

**SyncTalk Server**:
```
INFO - Starting SyncTalk FastAPI server
INFO - Model may loaded successfully
INFO - Server running on http://localhost:8000
```

**Rapido Client**:
```
INFO - Successfully connected to SyncTalk FastAPI server
INFO - Starting TTS audio generation and streaming to SyncTalk...
INFO - Processed avatar frame 0
INFO - Video generation completed: ./output/rapido_output.mp4
```

Your final video will be saved as `rapido/output/rapido_output.mp4` with synchronized audio, avatar, and slide content!
