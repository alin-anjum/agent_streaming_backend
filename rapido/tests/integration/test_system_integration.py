"""
Integration tests for Rapido system components
"""

import pytest
import asyncio
import numpy as np
import tempfile
import json
from pathlib import Path
import time
from unittest.mock import Mock, AsyncMock, patch

from src.core.interfaces import AudioChunk, VideoFrame, SlideData
from src.core.logging_manager import setup_logging
from src.core.metrics import get_metrics_collector
from src.core.security import SecurityManager, SecurityConfig
from src.services.audio_service import AudioProcessorService, TTSService
from src.services.video_service import VideoProcessorService, FrameComposerService
from src.services.data_service import DataParsingService, SlideFrameManager


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for integration tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        
        # Create directory structure
        logs_dir = workspace / "logs"
        data_dir = workspace / "data"
        frames_dir = workspace / "frames"
        
        logs_dir.mkdir()
        data_dir.mkdir()
        frames_dir.mkdir()
        
        # Create test lesson data
        lesson_dir = frames_dir / "integration_lesson_001"
        lesson_dir.mkdir()
        
        # Create dummy frame files
        for i in range(5):
            frame_file = lesson_dir / f"slide_{i:03d}.png"
            # Create minimal PNG data
            frame_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        
        # Create test slide data file
        slide_data = {
            "slide_data": {
                "slideId": "integration_lesson_001",
                "narrationData": {
                    "text": "This is an integration test narration for testing the complete system.",
                    "timing": [
                        {"start": 0.0, "end": 2.0, "text": "This is an integration test"},
                        {"start": 2.0, "end": 4.5, "text": "narration for testing"},
                        {"start": 4.5, "end": 6.0, "text": "the complete system."}
                    ]
                }
            }
        }
        
        slide_file = data_dir / "integration_lesson_001.json"
        with open(slide_file, 'w') as f:
            json.dump(slide_data, f)
        
        yield {
            "workspace": workspace,
            "logs_dir": logs_dir,
            "data_dir": data_dir,
            "frames_dir": frames_dir,
            "slide_file": slide_file,
            "lesson_id": "integration_lesson_001"
        }


@pytest.fixture
def system_config(temp_workspace):
    """Create system configuration for integration tests"""
    return {
        "logging": {
            "log_dir": str(temp_workspace["logs_dir"])
        },
        "security": SecurityConfig(
            jwt_secret="integration_test_secret_key_12345"
        ),
        "audio": {
            "sample_rate": 22050,
            "channels": 1
        },
        "video": {
            "target_width": 854,
            "target_height": 480,
            "overlay_position": "bottom-right",
            "overlay_scale": 0.5
        }
    }


@pytest.fixture
async def integrated_services(system_config, temp_workspace):
    """Create integrated service instances"""
    # Setup logging
    logging_manager = setup_logging(system_config["logging"]["log_dir"])
    
    # Setup security
    security_manager = SecurityManager(system_config["security"])
    
    # Create services
    services = {
        "audio_processor": AudioProcessorService(
            sample_rate=system_config["audio"]["sample_rate"],
            channels=system_config["audio"]["channels"]
        ),
        "video_processor": VideoProcessorService(
            target_width=system_config["video"]["target_width"],
            target_height=system_config["video"]["target_height"]
        ),
        "frame_composer": FrameComposerService(
            overlay_position=system_config["video"]["overlay_position"],
            overlay_scale=system_config["video"]["overlay_scale"]
        ),
        "data_parser": DataParsingService(security_manager),
        "frame_manager": SlideFrameManager(str(temp_workspace["frames_dir"])),
        "security_manager": security_manager,
        "logging_manager": logging_manager
    }
    
    return services


class TestAudioVideoIntegration:
    """Test audio and video processing integration"""
    
    @pytest.mark.asyncio
    async def test_audio_processing_chain(self, integrated_services):
        """Test complete audio processing chain"""
        audio_service = integrated_services["audio_processor"]
        
        # Create test audio chunk
        audio_data = np.random.random(1024).astype(np.float32)
        original_chunk = AudioChunk(
            data=audio_data,
            sample_rate=22050,
            timestamp=time.time(),
            duration=1.0,
            chunk_id="integration_test_001"
        )
        
        # Process audio
        processed_chunk = await audio_service.process_audio(original_chunk)
        
        # Verify processing
        assert processed_chunk is not None
        assert processed_chunk.chunk_id == original_chunk.chunk_id
        assert processed_chunk.data is not None
        assert len(processed_chunk.data) == len(original_chunk.data)
        
        # Check metrics were recorded
        metrics = await audio_service.get_metrics()
        assert metrics["processing_stats"]["total_chunks"] >= 1
    
    @pytest.mark.asyncio
    async def test_video_processing_chain(self, integrated_services):
        """Test complete video processing chain"""
        video_service = integrated_services["video_processor"]
        composer_service = integrated_services["frame_composer"]
        
        # Create test frames
        base_frame_data = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        base_frame = VideoFrame(
            data=base_frame_data,
            timestamp=time.time(),
            frame_number=1,
            width=640,
            height=360,
            fps=30.0
        )
        
        overlay_frame_data = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
        # Add green screen area
        overlay_frame_data[50:100, 50:100] = [0, 255, 0]
        overlay_frame = VideoFrame(
            data=overlay_frame_data,
            timestamp=time.time(),
            frame_number=1,
            width=320,
            height=180,
            fps=30.0
        )
        
        # Process base frame
        processed_base = await video_service.process_frame(base_frame)
        assert processed_base.width == 854
        assert processed_base.height == 480
        
        # Process overlay frame
        processed_overlay = await video_service.process_frame(overlay_frame)
        
        # Compose frames
        composed_frame = await composer_service.compose_frame(
            processed_base,
            processed_overlay,
            lesson_id="integration_test"
        )
        
        # Verify composition
        assert composed_frame is not None
        assert composed_frame.width == 854
        assert composed_frame.height == 480
        
        # Check metrics were recorded
        video_metrics = await video_service.get_processing_stats()
        composer_metrics = await composer_service.get_composition_metrics()
        
        assert video_metrics["processing_stats"]["total_frames"] >= 2
        assert composer_metrics["composition_stats"]["total_compositions"] >= 1


class TestDataProcessingIntegration:
    """Test data parsing and frame management integration"""
    
    @pytest.mark.asyncio
    async def test_data_parsing_integration(self, integrated_services, temp_workspace):
        """Test complete data parsing workflow"""
        data_parser = integrated_services["data_parser"]
        frame_manager = integrated_services["frame_manager"]
        
        # Parse slide data
        slide_data = await data_parser.parse_slide_data(str(temp_workspace["slide_file"]))
        
        # Verify parsed data
        assert slide_data.slide_id == "integration_lesson_001"
        assert len(slide_data.narration_text) > 0
        assert slide_data.duration > 0
        assert len(slide_data.timing_segments) == 3
        
        # Get slide frames
        frames = await frame_manager.get_slide_frames(temp_workspace["lesson_id"])
        assert len(frames) == 5
        
        # Load a frame
        frame_data = await frame_manager.load_frame(frames[0])
        assert frame_data is not None
        assert len(frame_data) > 0
        
        # Check metrics
        parsing_metrics = await data_parser.get_parsing_metrics()
        frame_stats = await frame_manager.get_frame_stats()
        
        assert parsing_metrics["parsing_stats"]["successful_parses"] >= 1
        assert frame_stats["frame_stats"]["frames_loaded"] >= 1


class TestSecurityIntegration:
    """Test security integration across services"""
    
    @pytest.mark.asyncio
    async def test_security_validation_integration(self, integrated_services, temp_workspace):
        """Test security validation across different services"""
        security_manager = integrated_services["security_manager"]
        data_parser = integrated_services["data_parser"]
        
        # Test valid security validation
        lesson_id = temp_workspace["lesson_id"]
        file_path = str(temp_workspace["slide_file"])
        
        # This should pass security validation
        security_manager.validate_request(
            lesson_id=lesson_id,
            file_path=file_path.replace(str(temp_workspace["workspace"]) + "/", ""),
            client_ip="127.0.0.1"
        )
        
        # Generate and validate token
        token = security_manager.generate_session_token(lesson_id, "test_user")
        payload = security_manager.validate_session_token(token)
        
        assert payload["lesson_id"] == lesson_id
        assert payload["user_id"] == "test_user"
        
        # Test file path validation in data parser
        slide_data = await data_parser.parse_slide_data(file_path)
        assert slide_data is not None


class TestLoggingAndMetricsIntegration:
    """Test logging and metrics integration"""
    
    @pytest.mark.asyncio
    async def test_logging_integration(self, integrated_services, temp_workspace):
        """Test logging integration across services"""
        logging_manager = integrated_services["logging_manager"]
        audio_service = integrated_services["audio_processor"]
        
        lesson_id = temp_workspace["lesson_id"]
        
        # Log lesson start
        logging_manager.log_lesson_start(lesson_id)
        
        # Process audio (which logs events)
        audio_data = np.random.random(512).astype(np.float32)
        audio_chunk = AudioChunk(
            data=audio_data,
            sample_rate=22050,
            timestamp=time.time(),
            duration=0.5,
            chunk_id="integration_audio_001"
        )
        
        await audio_service.process_audio(audio_chunk)
        
        # Log FPS metrics
        metrics_collector = get_metrics_collector()
        fps_metrics = metrics_collector.get_fps_metrics()
        
        logging_manager.log_fps_metrics(
            lesson_id=lesson_id,
            slide_fps=25.0,
            synctalk_fps=fps_metrics.get("synctalk_output", 0.0),
            composer_fps=fps_metrics.get("composer", 0.0),
            livekit_fps=fps_metrics.get("livekit_output", 0.0)
        )
        
        # Log an event
        logging_manager.log_event(
            event_type="integration_test",
            message="Integration test completed successfully",
            lesson_id=lesson_id,
            event_source="backend"
        )
        
        # Verify log files were created
        log_files = list(temp_workspace["logs_dir"].glob("rapido_*.jsonl"))
        assert len(log_files) > 0
        
        # Read and verify log content
        with open(log_files[0], 'r') as f:
            log_lines = f.readlines()
            assert len(log_lines) > 0
            
            # Parse first log line
            log_entry = json.loads(log_lines[0])
            assert "timestamp" in log_entry
            assert "level" in log_entry
            assert "message" in log_entry
    
    @pytest.mark.asyncio
    async def test_metrics_integration(self, integrated_services):
        """Test metrics collection integration"""
        audio_service = integrated_services["audio_processor"]
        video_service = integrated_services["video_processor"]
        metrics_collector = get_metrics_collector()
        
        # Process some audio and video to generate metrics
        audio_data = np.random.random(256).astype(np.float32)
        audio_chunk = AudioChunk(
            data=audio_data,
            sample_rate=22050,
            timestamp=time.time(),
            duration=0.25,
            chunk_id="metrics_test_audio"
        )
        
        video_data = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        video_frame = VideoFrame(
            data=video_data,
            timestamp=time.time(),
            frame_number=1,
            width=320,
            height=240,
            fps=30.0
        )
        
        # Process multiple times to generate metrics
        for i in range(3):
            await audio_service.process_audio(audio_chunk)
            await video_service.process_frame(video_frame)
        
        # Get comprehensive metrics
        all_metrics = metrics_collector.get_all_metrics()
        
        assert "fps_metrics" in all_metrics
        assert "performance_metrics" in all_metrics
        assert "timestamp" in all_metrics
        
        # Check specific service metrics
        audio_metrics = await audio_service.get_metrics()
        video_metrics = await video_service.get_processing_stats()
        
        assert audio_metrics["processing_stats"]["total_chunks"] >= 3
        assert video_metrics["processing_stats"]["total_frames"] >= 3


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_lesson_processing_workflow(self, integrated_services, temp_workspace):
        """Test complete lesson processing from data to composed frames"""
        data_parser = integrated_services["data_parser"]
        frame_manager = integrated_services["frame_manager"]
        audio_service = integrated_services["audio_processor"]
        video_service = integrated_services["video_processor"]
        composer_service = integrated_services["frame_composer"]
        logging_manager = integrated_services["logging_manager"]
        
        lesson_id = temp_workspace["lesson_id"]
        
        # 1. Log lesson start
        logging_manager.log_lesson_start(lesson_id)
        
        # 2. Parse slide data
        slide_data = await data_parser.parse_slide_data(str(temp_workspace["slide_file"]))
        assert slide_data is not None
        
        # 3. Get slide frames
        frame_paths = await frame_manager.get_slide_frames(lesson_id)
        assert len(frame_paths) > 0
        
        # 4. Simulate audio processing
        for i, timing_segment in enumerate(slide_data.timing_segments):
            # Create audio chunk for timing segment
            audio_duration = timing_segment["end"] - timing_segment["start"]
            samples = int(audio_duration * 22050)
            audio_data = np.random.random(samples).astype(np.float32)
            
            audio_chunk = AudioChunk(
                data=audio_data,
                sample_rate=22050,
                timestamp=timing_segment["start"],
                duration=audio_duration,
                chunk_id=f"{lesson_id}_audio_{i:03d}"
            )
            
            # Process audio
            processed_audio = await audio_service.process_audio(audio_chunk)
            assert processed_audio is not None
            
            # Log audio chunk
            logging_manager.log_audio_chunk(
                lesson_id=lesson_id,
                audio_chunk_id=processed_audio.chunk_id,
                chunk_duration=processed_audio.duration
            )
        
        # 5. Simulate video frame processing and composition
        for i, frame_path in enumerate(frame_paths[:3]):  # Process first 3 frames
            # Load and create base frame
            frame_data_bytes = await frame_manager.load_frame(frame_path)
            assert frame_data_bytes is not None
            
            # Create mock frame data (in real system this would be decoded image)
            base_frame_data = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            base_frame = VideoFrame(
                data=base_frame_data,
                timestamp=time.time(),
                frame_number=i,
                width=640,
                height=360,
                fps=25.0
            )
            
            # Create overlay frame (simulating avatar)
            overlay_frame_data = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
            # Add green screen
            overlay_frame_data[50:100, 50:100] = [0, 255, 0]
            overlay_frame = VideoFrame(
                data=overlay_frame_data,
                timestamp=time.time(),
                frame_number=i,
                width=320,
                height=180,
                fps=25.0
            )
            
            # Process frames
            processed_base = await video_service.process_frame(base_frame)
            processed_overlay = await video_service.process_frame(overlay_frame)
            
            # Compose frames
            composed_frame = await composer_service.compose_frame(
                processed_base,
                processed_overlay,
                lesson_id=lesson_id
            )
            
            assert composed_frame is not None
            assert composed_frame.width == 854
            assert composed_frame.height == 480
        
        # 6. Get and log final metrics
        metrics_collector = get_metrics_collector()
        fps_metrics = metrics_collector.get_fps_metrics()
        
        logging_manager.log_fps_metrics(
            lesson_id=lesson_id,
            slide_fps=fps_metrics.get("slide_frames", 25.0),
            synctalk_fps=fps_metrics.get("synctalk_output", 0.0),
            composer_fps=fps_metrics.get("composer", 0.0),
            livekit_fps=fps_metrics.get("livekit_output", 0.0)
        )
        
        # 7. Verify final state
        parsing_metrics = await data_parser.get_parsing_metrics()
        audio_metrics = await audio_service.get_metrics()
        video_metrics = await video_service.get_processing_stats()
        composer_metrics = await composer_service.get_composition_metrics()
        
        assert parsing_metrics["parsing_stats"]["successful_parses"] >= 1
        assert audio_metrics["processing_stats"]["total_chunks"] >= 3
        assert video_metrics["processing_stats"]["total_frames"] >= 6  # base + overlay for 3 frames
        assert composer_metrics["composition_stats"]["total_compositions"] >= 3
        
        # 8. Verify log files contain expected events
        log_files = list(temp_workspace["logs_dir"].glob("rapido_*.jsonl"))
        assert len(log_files) > 0
        
        # Count different event types in logs
        event_types = set()
        with open(log_files[0], 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if "event_type" in log_entry:
                        event_types.add(log_entry["event_type"])
                except json.JSONDecodeError:
                    continue
        
        # Should have various event types
        expected_events = {
            "lesson_start",
            "audio_processed",
            "frame_processed",
            "frame_composed",
            "fps_update"
        }
        
        # At least some of these events should be present
        assert len(event_types.intersection(expected_events)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
