"""
Refactored Rapido Main Application
Using the new modular architecture with comprehensive logging and monitoring
"""

import asyncio
import os
import sys
import json
import argparse
from typing import Dict, Any
from pathlib import Path

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Core imports
from core.logging_manager import setup_logging, get_logging_manager
from core.metrics import get_metrics_collector
from core.exceptions import RapidoException, ConfigurationError
from core.security import SecurityManager, SecurityConfig

# Service imports
from services.orchestrator_service import RapidoOrchestratorService


class RapidoApplication:
    """
    Main Rapido application class using modular architecture
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_configuration(config_path)
        
        # Setup logging first
        log_dir = self.config.get("logging", {}).get("log_dir", 
                                                   os.path.join(os.path.dirname(__file__), "..", "logs"))
        self.logging_manager = setup_logging(log_dir)
        self.logger = self.logging_manager.get_logger("main")
        
        # Initialize metrics
        self.metrics = get_metrics_collector()
        
        # Initialize orchestrator
        self.orchestrator = None
        
        self.logger.info("Rapido application initialized")
    
    def _load_configuration(self, config_path: str = None) -> Dict[str, Any]:
        """Load application configuration"""
        # Default configuration
        config = {
            "logging": {
                "log_dir": os.getenv("RAPIDO_LOG_DIR", "./logs")
            },
            "security": {
                "jwt_secret": os.getenv("JWT_SECRET", "rapido_default_secret_change_in_production")
            },
            "audio": {
                "sample_rate": int(os.getenv("AUDIO_SAMPLE_RATE", "22050")),
                "channels": int(os.getenv("AUDIO_CHANNELS", "1"))
            },
            "video": {
                "width": int(os.getenv("VIDEO_WIDTH", "854")),
                "height": int(os.getenv("VIDEO_HEIGHT", "480")),
                "overlay_position": os.getenv("OVERLAY_POSITION", "bottom-right"),
                "overlay_scale": float(os.getenv("OVERLAY_SCALE", "0.5"))
            },
            "tts": {
                "api_key": os.getenv("ELEVENLABS_API_KEY"),
                "voice_id": os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
            },
            "synctalk": {
                "server_url": os.getenv("SYNCTALK_SERVER_URL", "http://35.172.212.10:8000"),
                "model_name": os.getenv("SYNCTALK_MODEL_NAME", "enrique_torres")
            },
            "livekit": {
                "url": os.getenv("LIVEKIT_URL"),
                "api_key": os.getenv("LIVEKIT_API_KEY"),
                "api_secret": os.getenv("LIVEKIT_API_SECRET")
            },
            "paths": {
                "slide_frames": os.getenv("SLIDE_FRAMES_PATH", "./presentation_frames"),
                "input_data": os.getenv("INPUT_DATA_PATH", "./test1.json"),
                "output": os.getenv("OUTPUT_PATH", "./output")
            }
        }
        
        # Load custom configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    config.update(custom_config)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_path}: {e}")
        
        return config
    
    async def initialize(self) -> bool:
        """Initialize the application components"""
        try:
            self.logger.info("Initializing Rapido application components")
            
            # Validate configuration
            await self._validate_configuration()
            
            # Initialize orchestrator
            self.orchestrator = RapidoOrchestratorService(self.config)
            
            self.logger.info("Rapido application initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Rapido application: {e}")
            return False
    
    async def _validate_configuration(self):
        """Validate application configuration"""
        
        # Check required paths exist
        paths = self.config.get("paths", {})
        slide_frames_path = paths.get("slide_frames")
        
        if slide_frames_path and not os.path.exists(slide_frames_path):
            self.logger.warning(f"Slide frames path does not exist: {slide_frames_path}")
        
        # Validate LiveKit configuration
        livekit_config = self.config.get("livekit", {})
        if not all(livekit_config.get(key) for key in ["url", "api_key", "api_secret"]):
            self.logger.warning("LiveKit configuration incomplete - LiveKit streaming will be disabled")
        
        # Validate TTS configuration
        tts_config = self.config.get("tts", {})
        if not tts_config.get("api_key"):
            self.logger.warning("TTS API key not provided - TTS will be disabled")
        
        # Validate security configuration
        security_config = self.config.get("security", {})
        if not security_config.get("jwt_secret") or security_config["jwt_secret"] == "rapido_default_secret_change_in_production":
            self.logger.warning("Using default JWT secret - change for production use")
    
    async def process_lesson(self, lesson_id: str, slide_data_path: str = None) -> Dict[str, Any]:
        """
        Process a lesson using the orchestrator
        
        Args:
            lesson_id: Unique identifier for the lesson
            slide_data_path: Path to slide data file (optional, uses config default)
            
        Returns:
            Processing results
        """
        if not self.orchestrator:
            raise RapidoException("Application not initialized")
        
        if not slide_data_path:
            slide_data_path = self.config["paths"]["input_data"]
        
        self.logger.info(f"Starting lesson processing: {lesson_id}")
        
        try:
            result = await self.orchestrator.process_lesson(lesson_id, slide_data_path)
            
            if result["success"]:
                self.logger.info(f"Lesson processing completed successfully: {lesson_id}")
            else:
                self.logger.error(f"Lesson processing failed: {lesson_id} - {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error during lesson processing: {e}")
            return {
                "success": False,
                "lesson_id": lesson_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get application status"""
        status = {
            "application_initialized": self.orchestrator is not None,
            "configuration": {
                "tts_enabled": bool(self.config.get("tts", {}).get("api_key")),
                "synctalk_configured": bool(self.config.get("synctalk", {}).get("server_url")),
                "livekit_configured": all(
                    self.config.get("livekit", {}).get(key) 
                    for key in ["url", "api_key", "api_secret"]
                )
            }
        }
        
        if self.orchestrator:
            orchestrator_status = await self.orchestrator.get_status()
            status.update(orchestrator_status)
        
        return status
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Rapido application")
        
        try:
            if self.orchestrator:
                await self.orchestrator.stop_processing()
            
            # Reset metrics
            self.metrics.reset_metrics()
            
            self.logger.info("Rapido application shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# CLI Interface
async def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(description="Rapido - Real-time Avatar Presentation System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--lesson-id", required=True, help="Lesson identifier")
    parser.add_argument("--slide-data", help="Path to slide data JSON file")
    parser.add_argument("--status-only", action="store_true", help="Show status only")
    
    args = parser.parse_args()
    
    # Create application
    app = RapidoApplication(config_path=args.config)
    
    # Initialize
    if not await app.initialize():
        print("❌ Failed to initialize Rapido application")
        return 1
    
    try:
        if args.status_only:
            # Show status only
            status = await app.get_status()
            print("📊 Rapido Application Status")
            print("=" * 50)
            print(json.dumps(status, indent=2))
            return 0
        
        # Process lesson
        print(f"🚀 Starting lesson processing: {args.lesson_id}")
        result = await app.process_lesson(args.lesson_id, args.slide_data)
        
        if result["success"]:
            print(f"✅ Lesson processing completed successfully!")
            print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"📄 Slides processed: {result.get('slides_processed', 0)}")
            print(f"🎵 Audio segments: {result.get('audio_segments', 0)}")
            return 0
        else:
            print(f"❌ Lesson processing failed: {result.get('error', 'Unknown error')}")
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 130  # Standard exit code for Ctrl+C
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    finally:
        await app.shutdown()


# API Interface (for use with FastAPI)
class RapidoAPI:
    """API wrapper for Rapido application"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.app = RapidoApplication()
        if config:
            self.app.config.update(config)
    
    async def initialize(self) -> bool:
        """Initialize API"""
        return await self.app.initialize()
    
    async def process_lesson_async(self, lesson_id: str, slide_data_path: str) -> Dict[str, Any]:
        """Process lesson asynchronously for API use"""
        return await self.app.process_lesson(lesson_id, slide_data_path)
    
    async def get_status_async(self) -> Dict[str, Any]:
        """Get status for API use"""
        return await self.app.get_status()
    
    async def shutdown_async(self):
        """Shutdown for API use"""
        await self.app.shutdown()


if __name__ == "__main__":
    # Run the main CLI application
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
