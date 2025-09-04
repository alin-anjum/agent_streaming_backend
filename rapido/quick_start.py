#!/usr/bin/env python3
"""
Quick start script for Rapido - demonstrates basic usage
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rapido_main import RapidoOrchestrator

async def quick_demo():
    """Run a quick demonstration of Rapido."""
    print("🚀 Rapido Quick Start Demo")
    print("=" * 40)
    
    # Configuration for demo
    config_override = {
        'INPUT_DATA_PATH': '../test1.json',
        'SLIDE_FRAMES_PATH': '../frames',
        'OUTPUT_PATH': './output',
        'FRAME_RATE': 30,
        'AVATAR_OVERLAY_POSITION': 'bottom-right',
        'AVATAR_SCALE': 0.3
    }
    
    # Check if API key is available
    if not os.getenv('ELEVEN_API_KEY'):
        print("❌ ELEVEN_API_KEY not found in environment variables")
        print("Please set your ElevenLabs API key:")
        print("export ELEVEN_API_KEY=your_api_key_here")
        return
    
    # Check if SyncTalk server URL is available
    synctalk_url = os.getenv('SYNCTALK_SERVER_URL')
    if not synctalk_url:
        print("⚠️  SYNCTALK_SERVER_URL not found, using default: ws://localhost:8000")
        config_override['SYNCTALK_SERVER_URL'] = 'ws://localhost:8000'
    
    print(f"📡 SyncTalk Server: {synctalk_url or 'ws://localhost:8000'}")
    print(f"🎤 ElevenLabs API: {'✅ Configured' if os.getenv('ELEVEN_API_KEY') else '❌ Missing'}")
    
    try:
        # Initialize orchestrator
        print("\n🔧 Initializing Rapido...")
        orchestrator = RapidoOrchestrator(config_override)
        
        if not await orchestrator.initialize():
            print("❌ Failed to initialize Rapido")
            return
        
        print("✅ Rapido initialized successfully!")
        
        # Show slide data summary
        stats = orchestrator.get_processing_stats()
        slide_data = stats.get('slide_data', {})
        
        print(f"\n📄 Slide Data Summary:")
        print(f"   - Slide ID: {slide_data.get('slide_id', 'unknown')}")
        print(f"   - Narration: {slide_data.get('narration_text_length', 0)} characters")
        print(f"   - Duration: {slide_data.get('total_duration_seconds', 0):.1f} seconds")
        print(f"   - Timing tokens: {slide_data.get('token_count', 0)}")
        print(f"   - Animation triggers: {slide_data.get('animation_triggers_count', 0)}")
        
        # Show frame processor info
        if orchestrator.frame_processor:
            frame_count = orchestrator.frame_processor.get_frame_count()
            print(f"   - Slide frames: {frame_count}")
        
        print(f"\n🎬 Ready to process presentation!")
        print(f"📁 Output will be saved to: {config_override['OUTPUT_PATH']}")
        
        # Ask user if they want to proceed
        try:
            response = input("\n▶️  Start processing? (y/N): ").strip().lower()
            if response != 'y':
                print("⏹️  Processing cancelled by user")
                return
        except KeyboardInterrupt:
            print("\n⏹️  Processing cancelled by user")
            return
        
        # Process the presentation
        print("\n🎬 Starting presentation processing...")
        print("This may take a few minutes depending on the presentation length...")
        
        output_path = await orchestrator.process_presentation()
        
        if output_path:
            print(f"\n🎉 Success! Generated video: {output_path}")
            
            # Show final statistics
            final_stats = orchestrator.get_processing_stats()
            print(f"\n📊 Final Statistics:")
            print(f"   - Avatar frames received: {final_stats.get('avatar_frames_received', 0)}")
            print(f"   - Final frames processed: {final_stats.get('final_frames_processed', 0)}")
            print(f"   - Audio buffer size: {final_stats.get('audio_buffer_size', 0)} bytes")
        else:
            print("❌ Failed to generate video")
            
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("Check the logs for more details")

def main():
    """Main entry point."""
    # Check if test script should be run first
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🧪 Running setup test first...")
        import test_setup
        test_setup.main()
        print("\n" + "=" * 40)
    
    # Run the demo
    asyncio.run(quick_demo())

if __name__ == "__main__":
    main()
