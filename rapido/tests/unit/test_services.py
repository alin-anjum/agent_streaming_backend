"""
Unit tests for Rapido services
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import time
import aiohttp
from aiohttp import web

from src.core.interfaces import AudioChunk, VideoFrame, SlideData
from src.core.exceptions import (
    AudioProcessingError, VideoProcessingError, DataParsingError,
    SyncTalkConnectionError, LiveKitConnectionError
)
from src.services.audio_service import AudioProcessorService, TTSService, AudioOptimizerService
from src.services.video_service import VideoProcessorService, FrameComposerService
from src.services.data_service import DataParsingService, SlideFrameManager


class TestAudioProcessorService:
    """Test AudioProcessorService"""
    
    @pytest.fixture
    def audio_service(self):
        """Create AudioProcessorService instance"""
        return AudioProcessorService(sample_rate=22050, channels=1)
    
    @pytest.fixture
    def sample_audio_chunk(self):
        """Create sample AudioChunk"""
        data = np.random.random(1024).astype(np.float32)
        return AudioChunk(
            data=data,
            sample_rate=22050,
            timestamp=time.time(),
            duration=1.0,
            chunk_id="test_chunk_001"
        )
    
    @pytest.mark.asyncio
    async def test_process_audio_success(self, audio_service, sample_audio_chunk):
        """Test successful audio processing"""
        result = await audio_service.process_audio(sample_audio_chunk)
        
        assert isinstance(result, AudioChunk)
        assert result.chunk_id == sample_audio_chunk.chunk_id
        assert result.sample_rate == sample_audio_chunk.sample_rate
        assert result.data is not None
        assert len(result.data) == len(sample_audio_chunk.data)
    
    @pytest.mark.asyncio
    async def test_process_audio_empty_data(self, audio_service):
        """Test audio processing with empty data"""
        empty_chunk = AudioChunk(
            data=np.array([]),
            sample_rate=22050,
            timestamp=time.time(),
            duration=0.0,
            chunk_id="empty_chunk"
        )
        
        with pytest.raises(AudioProcessingError):
            await audio_service.process_audio(empty_chunk)
    
    @pytest.mark.asyncio
    async def test_process_audio_none_data(self, audio_service):
        """Test audio processing with None data"""
        none_chunk = AudioChunk(
            data=None,
            sample_rate=22050,
            timestamp=time.time(),
            duration=0.0,
            chunk_id="none_chunk"
        )
        
        with pytest.raises(AudioProcessingError):
            await audio_service.process_audio(none_chunk)
    
    @pytest.mark.asyncio
    async def test_audio_optimization(self, audio_service):
        """Test audio optimization algorithms"""
        # Create audio with known characteristics
        original_data = np.array([0.5, -0.3, 0.1, -0.8, 0.9], dtype=np.float32)
        optimized_data = await audio_service._optimize_audio(original_data)
        
        assert len(optimized_data) == len(original_data)
        assert optimized_data.dtype == np.float32
        
        # Check normalization (values should be in [-1, 1])
        assert np.all(optimized_data >= -1.0)
        assert np.all(optimized_data <= 1.0)
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, audio_service, sample_audio_chunk):
        """Test metrics collection"""
        # Process some audio to generate metrics
        await audio_service.process_audio(sample_audio_chunk)
        
        metrics = await audio_service.get_metrics()
        
        assert "processing_stats" in metrics
        assert "fps_metrics" in metrics
        assert "timestamp" in metrics
        
        stats = metrics["processing_stats"]
        assert stats["total_chunks"] == 1
        assert stats["error_count"] == 0


class TestTTSService:
    """Test TTSService"""
    
    @pytest.fixture
    def tts_service(self):
        """Create TTSService instance with mock API key"""
        return TTSService(api_key="test_api_key", voice_id="test_voice")
    
    @pytest.mark.asyncio
    async def test_synthesize_speech_empty_text(self, tts_service):
        """Test TTS with empty text"""
        with pytest.raises(Exception):  # Could be TTSError or other exception
            await tts_service.synthesize_speech("")
    
    @pytest.mark.asyncio
    async def test_get_synthesis_metrics(self, tts_service):
        """Test TTS metrics retrieval"""
        metrics = await tts_service.get_synthesis_metrics()
        
        assert "synthesis_stats" in metrics
        assert "timestamp" in metrics
        
        stats = metrics["synthesis_stats"]
        assert "total_requests" in stats
        assert "error_count" in stats


class TestVideoProcessorService:
    """Test VideoProcessorService"""
    
    @pytest.fixture
    def video_service(self):
        """Create VideoProcessorService instance"""
        return VideoProcessorService(target_width=854, target_height=480)
    
    @pytest.fixture
    def sample_video_frame(self):
        """Create sample VideoFrame"""
        data = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        return VideoFrame(
            data=data,
            timestamp=time.time(),
            frame_number=1,
            width=640,
            height=360,
            fps=30.0
        )
    
    @pytest.mark.asyncio
    async def test_process_frame_success(self, video_service, sample_video_frame):
        """Test successful frame processing"""
        result = await video_service.process_frame(sample_video_frame)
        
        assert isinstance(result, VideoFrame)
        assert result.frame_number == sample_video_frame.frame_number
        assert result.width == 854  # Target width
        assert result.height == 480  # Target height
        assert result.data.shape == (480, 854, 3)
    
    @pytest.mark.asyncio
    async def test_process_frame_empty_data(self, video_service):
        """Test frame processing with empty data"""
        empty_frame = VideoFrame(
            data=np.array([]),
            timestamp=time.time(),
            frame_number=1,
            width=0,
            height=0,
            fps=30.0
        )
        
        with pytest.raises(VideoProcessingError):
            await video_service.process_frame(empty_frame)
    
    @pytest.mark.asyncio
    async def test_frame_optimization(self, video_service):
        """Test frame optimization"""
        # Create frame with known size
        original_data = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        optimized_data = await video_service._optimize_frame(original_data)
        
        # Should be resized to target dimensions
        assert optimized_data.shape == (480, 854, 3)
    
    @pytest.mark.asyncio
    async def test_color_correction(self, video_service):
        """Test color correction"""
        original_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        corrected_data = await video_service._apply_color_correction(original_data)
        
        assert corrected_data.shape == original_data.shape
        assert corrected_data.dtype == original_data.dtype
    
    @pytest.mark.asyncio
    async def test_get_fps_metrics(self, video_service, sample_video_frame):
        """Test FPS metrics"""
        # Process some frames
        await video_service.process_frame(sample_video_frame)
        
        metrics = await video_service.get_fps_metrics()
        
        assert "current_fps" in metrics
        assert "frame_count" in metrics
        assert metrics["current_fps"] >= 0


class TestFrameComposerService:
    """Test FrameComposerService"""
    
    @pytest.fixture
    def composer_service(self):
        """Create FrameComposerService instance"""
        return FrameComposerService(
            overlay_position="bottom-right",
            overlay_scale=0.5
        )
    
    @pytest.fixture
    def base_frame(self):
        """Create base VideoFrame"""
        data = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
        return VideoFrame(
            data=data,
            timestamp=time.time(),
            frame_number=1,
            width=854,
            height=480,
            fps=30.0
        )
    
    @pytest.fixture
    def overlay_frame(self):
        """Create overlay VideoFrame"""
        # Create frame with some green pixels for chroma key testing
        data = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        # Add some green pixels
        data[50:100, 50:100] = [0, 255, 0]  # Pure green
        return VideoFrame(
            data=data,
            timestamp=time.time(),
            frame_number=1,
            width=320,
            height=240,
            fps=30.0
        )
    
    @pytest.mark.asyncio
    async def test_compose_frame_success(self, composer_service, base_frame, overlay_frame):
        """Test successful frame composition"""
        result = await composer_service.compose_frame(
            base_frame, 
            overlay_frame, 
            lesson_id="test_lesson"
        )
        
        assert isinstance(result, VideoFrame)
        assert result.width == base_frame.width
        assert result.height == base_frame.height
        assert result.data.shape == base_frame.data.shape
    
    @pytest.mark.asyncio
    async def test_chroma_key_application(self, composer_service, overlay_frame):
        """Test chroma key (green screen removal)"""
        processed_overlay = await composer_service._apply_chroma_key(overlay_frame.data)
        
        # Should have alpha channel after chroma key
        assert processed_overlay.shape[2] == 4  # RGBA
    
    @pytest.mark.asyncio
    async def test_overlay_scaling(self, composer_service, overlay_frame):
        """Test overlay scaling"""
        scaled_overlay = await composer_service._scale_overlay(overlay_frame.data)
        
        # Should be scaled by the scale factor (0.5)
        expected_height = int(overlay_frame.height * 0.5)
        expected_width = int(overlay_frame.width * 0.5)
        
        assert scaled_overlay.shape[0] == expected_height
        assert scaled_overlay.shape[1] == expected_width
    
    @pytest.mark.asyncio
    async def test_frame_composition(self, composer_service, base_frame, overlay_frame):
        """Test actual frame composition"""
        # Create RGBA overlay
        overlay_rgba = np.dstack([
            overlay_frame.data,
            np.full((overlay_frame.height, overlay_frame.width), 255, dtype=np.uint8)
        ])
        
        composed = await composer_service._composite_frames(
            base_frame.data,
            overlay_rgba
        )
        
        assert composed.shape == base_frame.data.shape
    
    @pytest.mark.asyncio
    async def test_get_composition_metrics(self, composer_service, base_frame, overlay_frame):
        """Test composition metrics"""
        # Compose some frames
        await composer_service.compose_frame(base_frame, overlay_frame)
        
        metrics = await composer_service.get_composition_metrics()
        
        assert "composition_stats" in metrics
        assert "composer_fps" in metrics
        assert "timestamp" in metrics
        
        stats = metrics["composition_stats"]
        assert stats["total_compositions"] == 1


class TestDataParsingService:
    """Test DataParsingService"""
    
    @pytest.fixture
    def data_service(self):
        """Create DataParsingService instance"""
        return DataParsingService()
    
    @pytest.fixture
    def temp_json_file(self):
        """Create temporary JSON file with valid slide data"""
        data = {
            "slide_data": {
                "slideId": "test_slide_001",
                "narrationData": {
                    "text": "This is a test narration for the slide.",
                    "timing": [
                        {"start": 0.0, "end": 2.5, "text": "This is a test"},
                        {"start": 2.5, "end": 5.0, "text": "narration for the slide."}
                    ]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            return f.name
    
    @pytest.fixture
    def invalid_json_file(self):
        """Create temporary file with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            return f.name
    
    @pytest.mark.asyncio
    async def test_parse_slide_data_success(self, data_service, temp_json_file):
        """Test successful slide data parsing"""
        slide_data = await data_service.parse_slide_data(temp_json_file)
        
        assert isinstance(slide_data, SlideData)
        assert slide_data.slide_id == "test_slide_001"
        assert "test narration" in slide_data.narration_text.lower()
        assert slide_data.duration == 5.0  # Based on timing data
        assert len(slide_data.timing_segments) == 2
        
        # Cleanup
        Path(temp_json_file).unlink()
    
    @pytest.mark.asyncio
    async def test_parse_slide_data_file_not_found(self, data_service):
        """Test parsing non-existent file"""
        with pytest.raises(DataParsingError):
            await data_service.parse_slide_data("non_existent_file.json")
    
    @pytest.mark.asyncio
    async def test_parse_slide_data_invalid_json(self, data_service, invalid_json_file):
        """Test parsing invalid JSON"""
        with pytest.raises(DataParsingError):
            await data_service.parse_slide_data(invalid_json_file)
        
        # Cleanup
        Path(invalid_json_file).unlink()
    
    @pytest.mark.asyncio
    async def test_validate_data_valid_structure(self, data_service):
        """Test data validation with valid structure"""
        valid_data = {
            "slide_data": {
                "narrationData": {
                    "text": "Test narration",
                    "timing": [
                        {"start": 0.0, "end": 2.0, "text": "Test"}
                    ]
                }
            }
        }
        
        is_valid = await data_service.validate_data(valid_data)
        assert is_valid
    
    @pytest.mark.asyncio
    async def test_validate_data_invalid_structure(self, data_service):
        """Test data validation with invalid structure"""
        invalid_data = {
            "slide_data": {
                # Missing narrationData
            }
        }
        
        is_valid = await data_service.validate_data(invalid_data)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_validate_data_empty_text(self, data_service):
        """Test data validation with empty text"""
        invalid_data = {
            "slide_data": {
                "narrationData": {
                    "text": "",  # Empty text
                }
            }
        }
        
        is_valid = await data_service.validate_data(invalid_data)
        assert not is_valid
    
    @pytest.mark.asyncio
    async def test_get_parsing_metrics(self, data_service, temp_json_file):
        """Test parsing metrics collection"""
        # Parse a file to generate metrics
        try:
            await data_service.parse_slide_data(temp_json_file)
        except:
            pass  # Ignore errors for metrics test
        
        metrics = await data_service.get_parsing_metrics()
        
        assert "parsing_stats" in metrics
        assert "success_rate" in metrics
        assert "timestamp" in metrics
        
        # Cleanup
        Path(temp_json_file).unlink()


class TestSlideFrameManager:
    """Test SlideFrameManager"""
    
    @pytest.fixture
    def temp_frames_dir(self):
        """Create temporary frames directory structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = Path(tmpdir)
            
            # Create lesson directory with frame files
            lesson_dir = frames_dir / "test_lesson_123"
            lesson_dir.mkdir()
            
            # Create dummy frame files
            for i in range(3):
                frame_file = lesson_dir / f"frame_{i:03d}.png"
                frame_file.write_bytes(b"dummy_image_data")
            
            yield str(frames_dir)
    
    @pytest.fixture
    def frame_manager(self, temp_frames_dir):
        """Create SlideFrameManager instance"""
        return SlideFrameManager(temp_frames_dir)
    
    @pytest.mark.asyncio
    async def test_get_slide_frames_success(self, frame_manager):
        """Test successful frame listing"""
        frames = await frame_manager.get_slide_frames("test_lesson_123")
        
        assert len(frames) == 3
        assert all("frame_" in frame for frame in frames)
        assert all(".png" in frame for frame in frames)
    
    @pytest.mark.asyncio
    async def test_get_slide_frames_invalid_lesson(self, frame_manager):
        """Test frame listing with invalid lesson ID"""
        frames = await frame_manager.get_slide_frames("../invalid_lesson")
        assert len(frames) == 0
    
    @pytest.mark.asyncio
    async def test_get_slide_frames_nonexistent_lesson(self, frame_manager):
        """Test frame listing for non-existent lesson"""
        frames = await frame_manager.get_slide_frames("nonexistent_lesson")
        assert len(frames) == 0
    
    @pytest.mark.asyncio
    async def test_load_frame_success(self, frame_manager, temp_frames_dir):
        """Test successful frame loading"""
        # Create a test frame file
        lesson_dir = Path(temp_frames_dir) / "test_lesson_123"
        test_frame = lesson_dir / "test_frame.png"
        test_data = b"test_image_data"
        test_frame.write_bytes(test_data)
        
        # Load the frame
        frame_data = await frame_manager.load_frame(str(test_frame))
        assert frame_data == test_data
    
    @pytest.mark.asyncio
    async def test_load_frame_nonexistent(self, frame_manager):
        """Test loading non-existent frame"""
        frame_data = await frame_manager.load_frame("nonexistent_frame.png")
        assert frame_data is None
    
    @pytest.mark.asyncio
    async def test_get_frame_stats(self, frame_manager):
        """Test frame statistics"""
        stats = await frame_manager.get_frame_stats()
        
        assert "frame_stats" in stats
        assert "cache_hit_rate" in stats
        assert "cached_frames" in stats
        assert "timestamp" in stats


if __name__ == "__main__":
    pytest.main([__file__])
