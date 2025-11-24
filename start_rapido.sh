#!/bin/bash
# Rapido System Startup Script

echo "🚀 Starting Rapido System"
echo "========================="

# Activate virtual environment
source venv/bin/activate

# Start the Rapido API server
echo "Starting Rapido API server on port 8080..."
xvfb-run -a -s "-screen 0 1920x1080x24" python3 rapido_system/api/rapido_api.py

echo "✅ Rapido API server started"
echo "Access at: http://localhost:8080/health"
