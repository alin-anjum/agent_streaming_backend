"""
Unit tests for core Rapido components
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import json
from pathlib import Path
import time

from src.core.interfaces import AudioChunk, VideoFrame, SlideData
from src.core.exceptions import (
    RapidoException, AudioProcessingError, VideoProcessingError,
    DataParsingError, SecurityError
)
from src.core.logging_manager import LoggingManager, get_logging_manager
from src.core.metrics import MetricsCollector, FPSCounter, PerformanceTimer
from src.core.security import (
    SecurityManager, InputValidator, AuthenticationManager,
    RateLimiter, SecurityConfig
)


class TestAudioChunk:
    """Test AudioChunk dataclass"""
    
    def test_audio_chunk_creation(self):
        """Test creating AudioChunk"""
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        chunk = AudioChunk(
            data=data,
            sample_rate=22050,
            timestamp=time.time(),
            duration=1.5,
            chunk_id="test_chunk"
        )
        
        assert chunk.sample_rate == 22050
        assert chunk.duration == 1.5
        assert chunk.chunk_id == "test_chunk"
        assert np.array_equal(chunk.data, data)


class TestVideoFrame:
    """Test VideoFrame dataclass"""
    
    def test_video_frame_creation(self):
        """Test creating VideoFrame"""
        data = np.zeros((480, 854, 3), dtype=np.uint8)
        frame = VideoFrame(
            data=data,
            timestamp=time.time(),
            frame_number=42,
            width=854,
            height=480,
            fps=30.0
        )
        
        assert frame.width == 854
        assert frame.height == 480
        assert frame.frame_number == 42
        assert frame.fps == 30.0
        assert frame.data.shape == (480, 854, 3)


class TestLoggingManager:
    """Test LoggingManager functionality"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary log directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_logging_manager_creation(self, temp_log_dir):
        """Test creating LoggingManager"""
        manager = LoggingManager(log_dir=temp_log_dir)
        assert manager.log_dir == Path(temp_log_dir)
        
        # Test logger creation
        logger = manager.get_logger("test_component")
        assert logger.name == "rapido.test_component"
    
    def test_lesson_start_logging(self, temp_log_dir):
        """Test lesson start logging"""
        manager = LoggingManager(log_dir=temp_log_dir)
        manager.log_lesson_start("test_lesson_123")
        
        # Verify log file was created
        log_files = list(Path(temp_log_dir).glob("rapido_*.jsonl"))
        assert len(log_files) > 0
    
    def test_fps_metrics_logging(self, temp_log_dir):
        """Test FPS metrics logging"""
        manager = LoggingManager(log_dir=temp_log_dir)
        manager.log_fps_metrics(
            lesson_id="test_lesson_123",
            slide_fps=25.0,
            synctalk_fps=24.5,
            composer_fps=25.2,
            livekit_fps=25.1
        )
        
        # Verify metrics were logged
        log_files = list(Path(temp_log_dir).glob("rapido_*.jsonl"))
        assert len(log_files) > 0
    
    def test_error_logging(self, temp_log_dir):
        """Test error logging"""
        manager = LoggingManager(log_dir=temp_log_dir)
        
        test_error = ValueError("Test error message")
        context = {"component": "test", "operation": "test_op"}
        
        manager.log_error(
            error=test_error,
            lesson_id="test_lesson_123",
            context=context
        )
        
        # Verify error was logged
        log_files = list(Path(temp_log_dir).glob("rapido_*.jsonl"))
        assert len(log_files) > 0
    
    def test_performance_logging(self, temp_log_dir):
        """Test performance logging context manager"""
        manager = LoggingManager(log_dir=temp_log_dir)
        
        with manager.log_performance("test_operation", "test_lesson_123"):
            time.sleep(0.01)  # Small delay for measurable duration
        
        # Verify performance was logged
        log_files = list(Path(temp_log_dir).glob("rapido_performance_*.jsonl"))
        assert len(log_files) > 0


class TestMetricsCollector:
    """Test MetricsCollector functionality"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create MetricsCollector instance"""
        return MetricsCollector()
    
    def test_fps_counter_creation(self, metrics_collector):
        """Test FPS counter creation"""
        fps_counter = metrics_collector.get_fps_counter("test_stream")
        assert isinstance(fps_counter, FPSCounter)
        
        # Test frame recording
        fps_counter.record_frame()
        fps_counter.record_frame()
        
        stats = fps_counter.get_stats()
        assert stats["frame_count"] >= 1
    
    def test_performance_metrics(self, metrics_collector):
        """Test performance metrics recording"""
        with metrics_collector.time_operation("test_operation") as timer:
            time.sleep(0.01)  # Small delay
        
        summary = metrics_collector.get_performance_summary("test_operation")
        assert summary["total_operations"] == 1
        assert summary["avg_duration"] > 0
    
    def test_custom_metrics(self, metrics_collector):
        """Test custom metrics recording"""
        metrics_collector.record_custom_metric(
            "test_metric",
            42,
            metadata={"unit": "count"}
        )
        
        all_metrics = metrics_collector.get_all_metrics()
        assert "test_metric" in all_metrics["custom_metrics"]
        assert all_metrics["custom_metrics"]["test_metric"][0]["value"] == 42
    
    def test_fps_metrics(self, metrics_collector):
        """Test FPS metrics collection"""
        # Record some frames
        counter = metrics_collector.get_fps_counter("test_fps")
        for _ in range(5):
            counter.record_frame()
            time.sleep(0.001)  # Small delay
        
        fps_metrics = metrics_collector.get_fps_metrics()
        assert "test_fps" in fps_metrics
        assert fps_metrics["test_fps"] >= 0
    
    def test_metrics_reset(self, metrics_collector):
        """Test metrics reset functionality"""
        # Add some metrics
        metrics_collector.record_custom_metric("test", 1)
        counter = metrics_collector.get_fps_counter("test")
        counter.record_frame()
        
        # Reset and verify
        metrics_collector.reset_metrics()
        
        all_metrics = metrics_collector.get_all_metrics()
        assert len(all_metrics["custom_metrics"]) == 0
        assert all_metrics["fps_metrics"]["test"]["frame_count"] == 0


class TestFPSCounter:
    """Test FPSCounter functionality"""
    
    def test_fps_calculation(self):
        """Test FPS calculation"""
        counter = FPSCounter(window_size=10)
        
        # Record frames with known intervals
        start_time = time.time()
        for i in range(5):
            counter.record_frame()
            time.sleep(0.02)  # 50 FPS target
        
        fps = counter.get_fps()
        assert fps > 0
        # Should be roughly 50 FPS (allowing for timing variations)
        assert 30 < fps < 70
    
    def test_fps_stats(self):
        """Test FPS statistics"""
        counter = FPSCounter(window_size=5)
        
        # Record frames
        for _ in range(10):
            counter.record_frame()
            time.sleep(0.01)
        
        stats = counter.get_stats()
        assert "fps" in stats
        assert "min_fps" in stats
        assert "max_fps" in stats
        assert "std_fps" in stats
        assert "frame_count" in stats
        
        assert stats["frame_count"] > 0
        assert stats["fps"] > 0


class TestSecurityManager:
    """Test SecurityManager and security components"""
    
    @pytest.fixture
    def security_config(self):
        """Create test security config"""
        return SecurityConfig(
            jwt_secret="test_secret_key_12345",
            jwt_algorithm="HS256",
            jwt_expiry_hours=1
        )
    
    @pytest.fixture
    def security_manager(self, security_config):
        """Create SecurityManager instance"""
        return SecurityManager(security_config)
    
    def test_input_validator(self):
        """Test input validation"""
        validator = InputValidator()
        
        # Test lesson ID validation
        assert validator.validate_lesson_id("lesson_123")
        assert validator.validate_lesson_id("test-lesson_456")
        assert not validator.validate_lesson_id("")
        assert not validator.validate_lesson_id("../../../etc/passwd")
        assert not validator.validate_lesson_id("a" * 100)  # Too long
        
        # Test file path validation
        assert validator.validate_file_path("data/test.json")
        assert validator.validate_file_path("slides/lesson1/frame001.png")
        assert not validator.validate_file_path("../../../etc/passwd")
        assert not validator.validate_file_path("/absolute/path")
        
        # Test filename sanitization
        dangerous_name = "test<>file|?.txt"
        safe_name = validator.sanitize_filename(dangerous_name)
        assert "<" not in safe_name
        assert ">" not in safe_name
        assert "|" not in safe_name
        assert "?" not in safe_name
    
    def test_authentication_manager(self, security_config):
        """Test JWT authentication"""
        auth_manager = AuthenticationManager(security_config)
        
        # Test token generation
        token = auth_manager.generate_token("test_lesson_123", "user_456")
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Test token verification
        payload = auth_manager.verify_token(token)
        assert payload["lesson_id"] == "test_lesson_123"
        assert payload["user_id"] == "user_456"
        
        # Test lesson ID extraction
        lesson_id = auth_manager.extract_lesson_id(token)
        assert lesson_id == "test_lesson_123"
    
    def test_rate_limiter(self):
        """Test rate limiting"""
        rate_limiter = RateLimiter(requests_per_minute=5)
        
        # Test within limit
        for i in range(5):
            assert rate_limiter.is_allowed("client_123")
        
        # Test over limit
        assert not rate_limiter.is_allowed("client_123")
        
        # Test different client
        assert rate_limiter.is_allowed("client_456")
    
    def test_security_validation(self, security_manager):
        """Test comprehensive security validation"""
        # Test valid request
        assert security_manager.validate_request(
            lesson_id="test_lesson_123",
            file_path="data/test.json",
            client_ip="192.168.1.1"
        )
        
        # Test invalid lesson ID
        with pytest.raises(SecurityError):
            security_manager.validate_request(
                lesson_id="../../../etc/passwd",
                client_ip="192.168.1.1"
            )
        
        # Test invalid file path
        with pytest.raises(SecurityError):
            security_manager.validate_request(
                lesson_id="test_lesson_123",
                file_path="../../../etc/passwd",
                client_ip="192.168.1.1"
            )
    
    def test_token_operations(self, security_manager):
        """Test token generation and validation"""
        # Generate session token
        token = security_manager.generate_session_token("test_lesson_123", "user_456")
        assert isinstance(token, str)
        
        # Validate session token
        payload = security_manager.validate_session_token(token)
        assert payload["lesson_id"] == "test_lesson_123"
        assert payload["user_id"] == "user_456"


class TestExceptions:
    """Test custom exceptions"""
    
    def test_rapido_exception(self):
        """Test base RapidoException"""
        error = RapidoException(
            "Test error",
            error_code="TEST001",
            details={"component": "test"}
        )
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST001"
        assert error.details["component"] == "test"
    
    def test_specific_exceptions(self):
        """Test specific exception types"""
        audio_error = AudioProcessingError("Audio failed")
        assert isinstance(audio_error, RapidoException)
        
        video_error = VideoProcessingError("Video failed")
        assert isinstance(video_error, RapidoException)
        
        data_error = DataParsingError("Data parsing failed")
        assert isinstance(data_error, RapidoException)
        
        security_error = SecurityError("Security validation failed")
        assert isinstance(security_error, RapidoException)


@pytest.mark.asyncio
class TestAsyncComponents:
    """Test async component functionality"""
    
    async def test_performance_timer_async(self):
        """Test PerformanceTimer in async context"""
        from src.core.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        with collector.time_operation("async_test") as timer:
            await asyncio.sleep(0.01)
        
        assert timer.duration > 0
        assert timer.success
        
        summary = collector.get_performance_summary("async_test")
        assert summary["total_operations"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
