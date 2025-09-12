"""
Advanced logging manager with date-based separation and structured logging
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time

@dataclass
class LogEntry:
    """Structured log entry for Rapido system"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    lesson_id: Optional[str] = None
    slide_fps: Optional[float] = None
    audio_chunk_id: Optional[str] = None
    synctalk_fps: Optional[float] = None
    composer_fps: Optional[float] = None
    livekit_fps: Optional[float] = None
    event_type: Optional[str] = None
    event_source: Optional[str] = None  # 'backend' or 'frontend'
    performance_data: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None


class DateRotatingJSONHandler(logging.handlers.BaseRotatingHandler):
    """Custom handler that creates new log files based on dates with JSON formatting"""
    
    def __init__(self, log_dir: str, filename_prefix: str = "rapido"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.current_date = None
        super().__init__(filename="", mode='a', delay=True)
    
    def get_current_filename(self):
        """Generate filename for current date"""
        current_date = date.today()
        return self.log_dir / f"{self.filename_prefix}_{current_date.isoformat()}.jsonl"
    
    def shouldRollover(self, record):
        """Check if we should rollover to a new file (new day)"""
        current_date = date.today()
        return self.current_date != current_date
    
    def doRollover(self):
        """Rollover to new file for new day"""
        if self.stream:
            self.stream.close()
            self.stream = None
        self.current_date = date.today()
    
    def _open(self):
        """Open the log file for current date"""
        if self.shouldRollover(None):
            self.doRollover()
        
        filename = self.get_current_filename()
        return open(filename, 'a', encoding='utf-8')
    
    def emit(self, record):
        """Emit a record as JSON"""
        try:
            if self.shouldRollover(record):
                self.doRollover()
            
            if self.stream is None:
                self.stream = self._open()
            
            # Convert log record to structured format
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            
            # Add custom fields if present
            for field in ['lesson_id', 'slide_fps', 'audio_chunk_id', 'synctalk_fps', 
                         'composer_fps', 'livekit_fps', 'event_type', 'event_source',
                         'performance_data', 'error_details']:
                if hasattr(record, field):
                    log_data[field] = getattr(record, field)
            
            # Write JSON line
            self.stream.write(json.dumps(log_data) + '\n')
            self.stream.flush()
            
        except Exception:
            self.handleError(record)


class LoggingManager:
    """Centralized logging manager for Rapido system"""
    
    def __init__(self, log_dir: str = "/home/ubuntu/agent_streaming_backend/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create root logger
        root_logger = logging.getLogger("rapido")
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler with structured format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Date-rotating JSON file handler
        json_handler = DateRotatingJSONHandler(
            log_dir=str(self.log_dir),
            filename_prefix="rapido"
        )
        json_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(json_handler)
        
        # Performance metrics logger
        perf_logger = logging.getLogger("rapido.performance")
        perf_handler = DateRotatingJSONHandler(
            log_dir=str(self.log_dir),
            filename_prefix="rapido_performance"
        )
        perf_handler.setLevel(logging.DEBUG)
        perf_logger.addHandler(perf_handler)
        perf_logger.propagate = False  # Don't propagate to root logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific component"""
        return logging.getLogger(f"rapido.{name}")
    
    def log_lesson_start(self, lesson_id: str, logger_name: str = "orchestrator"):
        """Log the start of a lesson"""
        logger = self.get_logger(logger_name)
        logger.info(
            f"Starting lesson: {lesson_id}",
            extra={
                "lesson_id": lesson_id,
                "event_type": "lesson_start",
                "event_source": "backend"
            }
        )
    
    def log_fps_metrics(self, lesson_id: str, slide_fps: float = None, 
                       synctalk_fps: float = None, composer_fps: float = None,
                       livekit_fps: float = None, logger_name: str = "metrics"):
        """Log FPS metrics"""
        logger = self.get_logger(logger_name)
        logger.info(
            "FPS Metrics Update",
            extra={
                "lesson_id": lesson_id,
                "slide_fps": slide_fps,
                "synctalk_fps": synctalk_fps,
                "composer_fps": composer_fps,
                "livekit_fps": livekit_fps,
                "event_type": "fps_update",
                "event_source": "backend"
            }
        )
    
    def log_audio_chunk(self, lesson_id: str, audio_chunk_id: str, 
                       chunk_duration: float = None, logger_name: str = "audio"):
        """Log audio chunk streaming"""
        logger = self.get_logger(logger_name)
        logger.info(
            f"Audio chunk streamed: {audio_chunk_id}",
            extra={
                "lesson_id": lesson_id,
                "audio_chunk_id": audio_chunk_id,
                "event_type": "audio_chunk_streamed",
                "event_source": "backend",
                "performance_data": {"chunk_duration": chunk_duration} if chunk_duration else None
            }
        )
    
    def log_event(self, event_type: str, message: str, lesson_id: str = None,
                 event_source: str = "backend", logger_name: str = "events", **kwargs):
        """Log a general event"""
        logger = self.get_logger(logger_name)
        extra_data = {
            "lesson_id": lesson_id,
            "event_type": event_type,
            "event_source": event_source
        }
        extra_data.update(kwargs)
        
        logger.info(message, extra=extra_data)
    
    def log_error(self, error: Exception, lesson_id: str = None, 
                 context: Dict[str, Any] = None, logger_name: str = "errors"):
        """Log an error with context"""
        logger = self.get_logger(logger_name)
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        logger.error(
            f"Error occurred: {error}",
            extra={
                "lesson_id": lesson_id,
                "event_type": "error",
                "event_source": "backend",
                "error_details": error_details
            }
        )
    
    @contextmanager
    def log_performance(self, operation_name: str, lesson_id: str = None):
        """Context manager for logging performance metrics"""
        start_time = time.time()
        logger = self.get_logger("performance")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            logger.info(
                f"Performance: {operation_name}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "performance",
                    "event_source": "backend",
                    "performance_data": {
                        "operation": operation_name,
                        "duration_seconds": duration
                    }
                }
            )


# Global logging manager instance
_logging_manager = None

def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager

def setup_logging(log_dir: str = None) -> LoggingManager:
    """Setup and return the global logging manager"""
    global _logging_manager
    _logging_manager = LoggingManager(log_dir) if log_dir else LoggingManager()
    return _logging_manager
