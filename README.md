# 🎭 Rapido System - Real-time Avatar Streaming Backend

**Clean, organized, production-ready avatar streaming system with FastAPI and LiveKit integration.**

## 🏗️ Clean Project Structure

```
agent_streaming_backend/
├── rapido_system/               # 🎯 Main Rapido system
│   ├── api/                    # 🌐 FastAPI server & endpoints  
│   │   ├── rapido_api.py      # Main API server
│   │   ├── rapido_main.py     # Core Rapido system
│   │   ├── tab_capture/       # Browser automation
│   │   └── ...                # Other API components
│   ├── core/                   # ⚙️ Core system components
│   │   ├── config/            # Configuration files
│   │   └── chrome_extension/   # Chrome extension for capture
│   ├── data/                   # 📁 All data files organized
│   │   ├── input/             # Input files (test data)
│   │   ├── output/            # Generated outputs  
│   │   ├── cache/             # Cache files (6GB moved here!)
│   │   ├── frames/            # Frame data
│   │   └── logs/              # Log files
│   ├── docs/                   # 📚 All documentation
│   ├── scripts/               # 🔧 Utility scripts
│   ├── tests/                 # 🧪 Test files
│   └── requirements.txt       # Python dependencies
├── SyncTalk_2D/               # 🎨 SyncTalk avatar system
├── venv/                      # Python virtual environment
├── start_rapido.sh           # 🚀 Easy startup script
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Start the System
```bash
# Simple startup (recommended)
./start_rapido.sh

# Manual startup
source venv/bin/activate
xvfb-run -a -s "-screen 0 1920x1080x24" python3 rapido_system/api/rapido_api.py
```

### 2. Test the API
```bash
curl http://localhost:8080/health
# Response: {"status":"healthy", ...}
```

### 3. Use with Frontend
- **Local**: `http://localhost:8080`
- **Production**: Deploy with Azure proxy (see separate repo)

## 🎯 Key Features

✅ **Clean Architecture** - Organized into logical components  
✅ **FastAPI Server** - Modern async Python API  
✅ **LiveKit Integration** - Real-time video streaming  
✅ **SyncTalk Avatar** - AI-powered avatar generation  
✅ **Browser Automation** - Chrome tab capture  
✅ **Production Ready** - Proper error handling & logging  

## 🔧 Development

### Install Dependencies
```bash
source venv/bin/activate
pip install -r rapido_system/requirements.txt
```

### Key Commands
```bash
# Restart server only (keeps same URL if using proxy)
pkill -f rapido_api.py
./start_rapido.sh

# View logs
tail -f rapido_system/data/logs/rapido_api.log

# Run tests
python -m pytest rapido_system/tests/
```

## 🌐 Production Deployment

For production, use the separate **Azure Proxy Repository** that provides:
- ✅ Stable HTTPS URLs that never change
- ✅ Professional SSL certificates  
- ✅ CORS handling for frontend integration
- ✅ Auto-deployment from GitHub

**Repository**: `../rapido-azure-proxy/` (moved outside this project)

## 📊 What Changed

### ✅ Improvements
- **Organized Structure** - Everything in logical folders
- **Single Requirements** - One requirements.txt file
- **Data Management** - All data files in `data/` folder  
- **Documentation** - All docs in `docs/` folder
- **Clean Root** - No more scattered files
- **6GB Cache** - Moved from root to `data/cache/`
- **Startup Script** - Easy `./start_rapido.sh` command

### 🗑️ Cleaned Up
- ❌ Multiple requirements.txt files
- ❌ Scattered .md files at root
- ❌ 6GB cache file polluting root
- ❌ Browser data cache (6GB saved!)
- ❌ Temporary patch files
- ❌ Azure proxy files (moved to separate repo)

## 🎭 How It Works

1. **FastAPI Server** (`rapido_system/api/rapido_api.py`) handles HTTP requests
2. **Rapido Main** processes avatar generation with SyncTalk integration  
3. **Browser Capture** automates Chrome for dynamic content capture
4. **LiveKit Streaming** broadcasts real-time video to frontend
5. **Azure Proxy** (separate repo) provides production HTTPS endpoints

## 📞 Support

- **Issues**: Create GitHub issue
- **Logs**: Check `rapido_system/data/logs/`
- **Health Check**: `http://localhost:8080/health`
