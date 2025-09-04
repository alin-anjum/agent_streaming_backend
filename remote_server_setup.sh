#!/bin/bash
# Setup script for SyncTalk-FastAPI server on remote machine

echo "Setting up SyncTalk-FastAPI server..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip if not already installed
sudo apt install -y python3 python3-pip python3-venv git

# Install CUDA and PyTorch dependencies (if GPU available)
# Note: This assumes CUDA is already installed on the server
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone the SyncTalk-FastAPI repository
if [ ! -d "SyncTalk-FastAPI" ]; then
    git clone https://github.com/ProfJim-Inc/SyncTalk-FastAPI.git
fi

cd SyncTalk-FastAPI

# Switch to the fix/fix-frame-freezing branch
git checkout fix/fix-frame-freezing

# Install Python dependencies
pip3 install -r requirements.txt

# Install additional FastAPI dependencies
pip3 install fastapi uvicorn websockets python-multipart

echo "Setup complete!"
echo ""
echo "To start the server, run:"
echo "cd SyncTalk-FastAPI"
echo "python3 synctalk_fastapi.py"
echo ""
echo "Or with uvicorn:"
echo "uvicorn synctalk_fastapi:app --host 0.0.0.0 --port 8000"


