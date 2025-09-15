#!/bin/bash
# Script to start SyncTalk-FastAPI server

cd SyncTalk-FastAPI

echo "Starting SyncTalk-FastAPI server on port 8000..."
echo "Server will be accessible at: http://0.0.0.0:8000"
echo "WebSocket endpoint: ws://0.0.0.0:8000/ws/audio_to_video"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server with uvicorn for better production performance
python3 -m uvicorn synctalk_fastapi:app --host 0.0.0.0 --port 8000 --reload


