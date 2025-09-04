#!/usr/bin/env python3
"""
Rapido - Real-time Avatar Presentation Integration with Dynamic Overlay

Main orchestrator that coordinates:
1. JSON slide data parsing
2. 11Labs TTS streaming
3. SyncTalk WebSocket communication
4. Avatar frame processing
5. Frame overlay composition
6. Video generation with synchronized audio
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config
from src.data_parser import SlideDataParser
from src.tts_client import ElevenLabsTTSClient, TTSAudioBuffer
from src.synctalk_client import SyncTalkWebSocketClient, FrameBuffer
from src.synctalk_fastapi_client import SyncTalkFastAPIClient, audio_chunks_from_tts
from src.frame_processor import FrameOverlayEngine
from src.timing_sync import TimingSynchronizer, SyncEventType, AudioVideoSynchronizer
from src.video_generator import VideoGenerator, FrameSequenceGenerator

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class RapidoOrchestrator:
    """Main orchestrator for the Rapido pipeline."""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize the Rapido orchestrator."""
        self.config = Config()
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        # Initialize components
        self.data_parser = None
        self.tts_client = None
        self.synctalk_client = None
        self.frame_processor = None
        self.timing_sync = None
        self.video_generator = None
        
        # Data storage
        self.slide_data = None
        self.narration_text = ""
        self.audio_buffer = TTSAudioBuffer()
        self.avatar_frame_buffer = FrameBuffer()
        self.final_frames = {}
        
        # State tracking
        self.is_processing = False
        self.processing_stats = {}
        
    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing Rapido components...")
            
            # Initialize data parser
            self.data_parser = SlideDataParser(self.config.INPUT_DATA_PATH)
            if not self.data_parser.load_data():
                logger.error("Failed to load slide data")
                return False
            
            self.slide_data = self.data_parser.get_summary()
            self.narration_text = self.data_parser.get_narration_text()
            
            if not self.narration_text:
                logger.error("No narration text found in slide data")
                return False
            
            logger.info(f"Loaded slide data: {self.slide_data}")
            
            # Initialize TTS client
            if not self.config.ELEVENLABS_API_KEY:
                logger.error("ElevenLabs API key not configured")
                return False
            
            self.tts_client = ElevenLabsTTSClient(
                api_key=self.config.ELEVENLABS_API_KEY,
                voice_id=getattr(self.config, 'ELEVENLABS_VOICE_ID', 'pNInz6obpgDQGcFmaJgB')
            )
            
            # Initialize SyncTalk client (use FastAPI client for better compatibility)
            synctalk_url = getattr(self.config, 'SYNCTALK_SERVER_URL', 'ws://localhost:8000')
            # Convert WebSocket URL to HTTP URL for FastAPI client
            http_url = synctalk_url.replace('ws://', 'http://').replace('wss://', 'https://')
            self.synctalk_client = SyncTalkFastAPIClient(http_url, model_name="enrique_torres")
            
            # Set up frame callback
            self.synctalk_client.set_frame_callback(self._handle_avatar_frame)
            
            # Initialize frame processor
            self.frame_processor = FrameOverlayEngine(
                self.config.SLIDE_FRAMES_PATH,
                output_size=(1920, 1080)  # HD output
            )
            
            # Initialize timing synchronizer
            self.timing_sync = TimingSynchronizer(frame_rate=self.config.FRAME_RATE)
            
            # Set up timing tokens from slide data
            tokens = self.data_parser.get_tokens() or []
            self.timing_sync.set_timing_tokens(tokens)
            
            # Set up animation triggers
            triggers = self.data_parser.extract_animation_triggers()
            self.timing_sync.set_animation_triggers(triggers)
            
            # Initialize video generator
            output_file = os.path.join(self.config.OUTPUT_PATH, "rapido_output.mp4")
            self.video_generator = VideoGenerator(
                output_path=output_file,
                frame_rate=self.config.FRAME_RATE,
                resolution=(1920, 1080),
                video_codec=self.config.VIDEO_CODEC
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    async def _handle_avatar_frame(self, frame, frame_index: int, timestamp: float):
        """Handle incoming avatar frames from SyncTalk."""
        try:
            await self.avatar_frame_buffer.add_frame(frame, frame_index, timestamp)
            logger.debug(f"Received avatar frame {frame_index} at {timestamp}ms")
            
            # Process frame immediately if we have corresponding slide frame
            slide_frame = self.frame_processor.get_slide_frame(frame_index)
            if slide_frame:
                # Apply overlay
                overlay_config = {
                    "position": getattr(self.config, 'AVATAR_OVERLAY_POSITION', 'bottom-right'),
                    "scale": getattr(self.config, 'AVATAR_SCALE', 0.3),
                    "offset": (50, 50),
                    "blend_mode": "normal"
                }
                
                final_frame = self.frame_processor.overlay_avatar_on_slide(
                    slide_frame,
                    frame,
                    **overlay_config
                )
                
                self.final_frames[frame_index] = final_frame
                logger.debug(f"Processed final frame {frame_index}")
            
        except Exception as e:
            logger.error(f"Error handling avatar frame: {e}")
    
    async def process_presentation(self) -> str:
        """Process the complete presentation pipeline."""
        if self.is_processing:
            logger.warning("Processing already in progress")
            return ""
        
        self.is_processing = True
        
        try:
            logger.info("Starting Rapido presentation processing...")
            
            # Step 1: Connect to SyncTalk server
            logger.info("Connecting to SyncTalk server...")
            if not await self.synctalk_client.connect():
                raise RuntimeError("Failed to connect to SyncTalk server")
            
            # Step 2: Start timing synchronization
            self.timing_sync.start_synchronization()
            
            # Step 3: Start streaming TTS and receiving avatar frames
            logger.info("Starting TTS streaming and avatar generation...")
            
            # Create tasks for concurrent processing
            tasks = [
                asyncio.create_task(self._stream_audio_to_synctalk()),
                asyncio.create_task(self._listen_for_avatar_frames()),
                asyncio.create_task(self._synchronization_loop())
            ]
            
            # Wait for audio streaming to complete
            await tasks[0]
            
            # Give some time for remaining avatar frames to arrive
            await asyncio.sleep(2.0)
            
            # Cancel remaining tasks
            for task in tasks[1:]:
                task.cancel()
            
            # Step 4: Generate final video
            logger.info("Generating final video...")
            
            # Get complete audio data
            complete_audio = await self.audio_buffer.get_all_data()
            
            if not complete_audio:
                raise RuntimeError("No audio data generated")
            
            # Calculate total duration
            total_duration_ms = self.slide_data.get('total_duration_ms', 45000)
            
            # Generate video with progress tracking
            output_path = await self.video_generator.generate_video_with_progress(
                self.final_frames,
                complete_audio,
                total_duration_ms,
                self._progress_callback
            )
            
            logger.info(f"Presentation processing completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing presentation: {e}")
            raise
        finally:
            self.is_processing = False
            # Cleanup
            if self.synctalk_client:
                await self.synctalk_client.disconnect()
    
    async def _stream_audio_to_synctalk(self):
        """Stream TTS audio to SyncTalk server."""
        try:
            logger.info("Starting TTS audio generation and streaming to SyncTalk...")
            
            # Create TTS audio stream
            tts_stream = self.tts_client.stream_tts(self.narration_text)
            
            # Buffer audio for final video while streaming to SyncTalk
            async def buffered_tts_stream():
                async for audio_chunk in tts_stream:
                    # Buffer audio for final video
                    await self.audio_buffer.write(audio_chunk)
                    yield audio_chunk
            
            # Convert TTS stream to PCM chunks for SyncTalk
            pcm_stream = audio_chunks_from_tts(buffered_tts_stream())
            
            # Stream to SyncTalk FastAPI server
            await self.synctalk_client.stream_audio_chunks(pcm_stream)
            
            logger.info("Audio streaming to SyncTalk completed")
            
        except Exception as e:
            logger.error(f"Error streaming audio to SyncTalk: {e}")
            raise
    
    async def _listen_for_avatar_frames(self):
        """Listen for incoming avatar frames."""
        try:
            # The FastAPI client handles frame listening internally
            # We just need to keep this task alive while streaming
            while self.synctalk_client.session_active:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Avatar frame listening cancelled")
        except Exception as e:
            logger.error(f"Error listening for avatar frames: {e}")
    
    async def _synchronization_loop(self):
        """Run the timing synchronization loop."""
        try:
            total_duration = self.slide_data.get('total_duration_ms', 45000)
            await self.timing_sync.synchronization_loop(total_duration)
        except asyncio.CancelledError:
            logger.info("Synchronization loop cancelled")
        except Exception as e:
            logger.error(f"Error in synchronization loop: {e}")
    
    async def _progress_callback(self, event_type: str, data: Dict[str, Any]):
        """Handle progress updates."""
        logger.info(f"Progress: {event_type} - {data}")
        self.processing_stats[event_type] = data
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = {
            "is_processing": self.is_processing,
            "slide_data": self.slide_data,
            "narration_length": len(self.narration_text),
            "avatar_frames_received": self.avatar_frame_buffer.get_frame_count(),
            "final_frames_processed": len(self.final_frames),
            "audio_buffer_size": self.audio_buffer.get_buffer_size(),
            "sync_stats": self.timing_sync.get_synchronization_stats() if self.timing_sync else {}
        }
        stats.update(self.processing_stats)
        return stats

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rapido - Real-time Avatar Presentation Integration")
    parser.add_argument("--input", "-i", help="Input JSON file path", default="../test1.json")
    parser.add_argument("--frames", "-f", help="Slide frames directory", default="../frames")
    parser.add_argument("--output", "-o", help="Output directory", default="./output")
    parser.add_argument("--synctalk-url", help="SyncTalk server URL")
    parser.add_argument("--api-key", help="ElevenLabs API key")
    parser.add_argument("--voice-id", help="ElevenLabs voice ID")
    parser.add_argument("--frame-rate", type=int, help="Output frame rate", default=30)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare configuration overrides
    config_override = {}
    if args.input:
        config_override['INPUT_DATA_PATH'] = args.input
    if args.frames:
        config_override['SLIDE_FRAMES_PATH'] = args.frames
    if args.output:
        config_override['OUTPUT_PATH'] = args.output
    if args.synctalk_url:
        config_override['SYNCTALK_SERVER_URL'] = args.synctalk_url
    if args.api_key:
        config_override['ELEVENLABS_API_KEY'] = args.api_key
    if args.voice_id:
        config_override['ELEVENLABS_VOICE_ID'] = args.voice_id
    if args.frame_rate:
        config_override['FRAME_RATE'] = args.frame_rate
    
    # Initialize and run orchestrator
    orchestrator = RapidoOrchestrator(config_override)
    
    try:
        # Initialize components
        if not await orchestrator.initialize():
            logger.error("Failed to initialize Rapido")
            return 1
        
        # Process presentation
        output_path = await orchestrator.process_presentation()
        
        if output_path:
            print(f"\nSuccess! Generated video: {output_path}")
            
            # Print final statistics
            stats = orchestrator.get_processing_stats()
            print("\nProcessing Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            return 0
        else:
            logger.error("Failed to generate video")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
