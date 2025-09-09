#!/usr/bin/env python3
"""
Simple runner script for Rapido
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rapido_main import main

if __name__ == "__main__":
    # Set default paths relative to this script
    script_dir = Path(__file__).parent
    
    # Set environment variables if not already set
    if not os.getenv('INPUT_DATA_PATH'):
        os.environ['INPUT_DATA_PATH'] = str(script_dir.parent / 'test1.json')
    
    if not os.getenv('SLIDE_FRAMES_PATH'):
        os.environ['SLIDE_FRAMES_PATH'] = str(script_dir.parent / 'presentation_frames')
    
    if not os.getenv('OUTPUT_PATH'):
        os.environ['OUTPUT_PATH'] = str(script_dir / 'output')
    
    # Run the main function
    sys.exit(asyncio.run(main()))
