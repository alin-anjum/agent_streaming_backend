"""
Data parsing and management services for Rapido system
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from ..core.interfaces import IDataParser, SlideData
from ..core.exceptions import DataParsingError
from ..core.logging_manager import get_logging_manager
from ..core.security import InputValidator, SecurityManager


class DataParsingService(IDataParser):
    """Service for parsing and validating presentation data"""
    
    def __init__(self, security_manager: SecurityManager = None):
        self.security_manager = security_manager or SecurityManager()
        self.validator = InputValidator()
        self.logger = get_logging_manager().get_logger("data_parser")
        
        # Parsing statistics
        self._parsing_stats = {
            "files_parsed": 0,
            "successful_parses": 0,
            "validation_failures": 0,
            "parsing_errors": 0,
            "total_slides_processed": 0
        }
    
    async def parse_slide_data(self, file_path: str) -> SlideData:
        """Parse slide data from JSON file with validation"""
        start_time = time.time()
        
        try:
            # Security validation
            if not self.validator.validate_file_path(file_path):
                raise DataParsingError(f"Invalid file path: {file_path}")
            
            # Check file exists and is readable
            path = Path(file_path)
            if not path.exists():
                raise DataParsingError(f"File not found: {file_path}")
            
            if not path.is_file():
                raise DataParsingError(f"Path is not a file: {file_path}")
            
            self._parsing_stats["files_parsed"] += 1
            
            # Read and parse JSON
            with open(path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
            
            # Validate JSON structure
            if not await self.validate_data(raw_data):
                self._parsing_stats["validation_failures"] += 1
                raise DataParsingError(f"Invalid JSON structure in file: {file_path}")
            
            # Extract slide data
            slide_data = await self._extract_slide_data(raw_data)
            
            # Update statistics
            self._parsing_stats["successful_parses"] += 1
            self._parsing_stats["total_slides_processed"] += 1
            
            # Log successful parsing
            parsing_time = time.time() - start_time
            self.logger.info(
                f"Successfully parsed slide data from: {file_path}",
                extra={
                    "event_type": "slide_data_parsed",
                    "performance_data": {
                        "parsing_time": parsing_time,
                        "file_path": file_path,
                        "slide_id": slide_data.slide_id,
                        "text_length": len(slide_data.narration_text),
                        "duration": slide_data.duration
                    }
                }
            )
            
            return slide_data
            
        except json.JSONDecodeError as e:
            self._parsing_stats["parsing_errors"] += 1
            self.logger.error(
                f"Invalid JSON in file {file_path}: {e}",
                extra={
                    "event_type": "json_parsing_error",
                    "error_details": {"error": str(e), "file_path": file_path}
                }
            )
            raise DataParsingError(f"Invalid JSON format: {e}")
            
        except Exception as e:
            self._parsing_stats["parsing_errors"] += 1
            self.logger.error(
                f"Failed to parse slide data from {file_path}: {e}",
                extra={
                    "event_type": "slide_parsing_error",
                    "error_details": {"error": str(e), "file_path": file_path}
                }
            )
            raise DataParsingError(f"Failed to parse slide data: {e}")
    
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate parsed slide data structure"""
        try:
            # Check required top-level structure
            required_fields = ["slide_data"]
            if not self.validator.validate_json_structure(data, required_fields):
                return False
            
            slide_data = data["slide_data"]
            if not isinstance(slide_data, dict):
                return False
            
            # Check for narration data
            if "narrationData" not in slide_data:
                return False
            
            narration_data = slide_data["narrationData"]
            if not isinstance(narration_data, dict):
                return False
            
            # Validate narration structure
            narration_required = ["text"]
            if not self.validator.validate_json_structure(narration_data, narration_required):
                return False
            
            # Validate text content
            text = narration_data.get("text", "")
            if not isinstance(text, str) or len(text.strip()) == 0:
                return False
            
            # Validate timing data if present
            if "timing" in narration_data:
                timing_data = narration_data["timing"]
                if not isinstance(timing_data, list):
                    return False
                
                # Validate timing segments
                for segment in timing_data:
                    if not isinstance(segment, dict):
                        return False
                    
                    segment_required = ["start", "end", "text"]
                    if not self.validator.validate_json_structure(segment, segment_required):
                        return False
                    
                    # Validate timing values
                    try:
                        start = float(segment["start"])
                        end = float(segment["end"])
                        if start < 0 or end < 0 or start >= end:
                            return False
                    except (ValueError, TypeError):
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Data validation error: {e}",
                extra={
                    "event_type": "data_validation_error",
                    "error_details": {"error": str(e)}
                }
            )
            return False
    
    async def _extract_slide_data(self, raw_data: Dict[str, Any]) -> SlideData:
        """Extract and structure slide data"""
        try:
            slide_data = raw_data["slide_data"]
            narration_data = slide_data["narrationData"]
            
            # Extract basic information
            slide_id = slide_data.get("slideId", f"slide_{int(time.time())}")
            narration_text = narration_data["text"].strip()
            
            # Extract timing segments
            timing_segments = []
            if "timing" in narration_data:
                timing_segments = narration_data["timing"]
            
            # Calculate duration
            duration = 0.0
            if timing_segments:
                # Use last segment end time
                duration = max(float(segment["end"]) for segment in timing_segments)
            else:
                # Estimate duration based on text length (rough estimate)
                # Assume average reading speed of 150 words per minute
                word_count = len(narration_text.split())
                duration = (word_count / 150.0) * 60.0  # Convert to seconds
            
            # Create SlideData object
            return SlideData(
                slide_id=slide_id,
                narration_text=narration_text,
                duration=duration,
                timing_segments=timing_segments
            )
            
        except Exception as e:
            raise DataParsingError(f"Failed to extract slide data: {e}")
    
    async def get_parsing_metrics(self) -> Dict[str, Any]:
        """Get data parsing metrics"""
        success_rate = (
            self._parsing_stats["successful_parses"] / 
            max(1, self._parsing_stats["files_parsed"])
        )
        
        return {
            "parsing_stats": self._parsing_stats.copy(),
            "success_rate": success_rate,
            "timestamp": time.time()
        }


class SlideFrameManager:
    """Manager for slide frame resources"""
    
    def __init__(self, frames_directory: str):
        self.frames_directory = Path(frames_directory)
        self.logger = get_logging_manager().get_logger("frame_manager")
        self.validator = InputValidator()
        
        # Frame management stats
        self._frame_stats = {
            "frames_loaded": 0,
            "frame_cache_hits": 0,
            "frame_cache_misses": 0,
            "loading_errors": 0
        }
        
        # Simple frame cache
        self._frame_cache = {}
    
    async def get_slide_frames(self, lesson_id: str) -> List[str]:
        """Get list of available slide frames for a lesson"""
        try:
            # Validate lesson ID
            if not self.validator.validate_lesson_id(lesson_id):
                raise ValueError(f"Invalid lesson ID: {lesson_id}")
            
            # Find frame files
            lesson_frames_dir = self.frames_directory / lesson_id
            if not lesson_frames_dir.exists():
                self.logger.warning(
                    f"No frames directory found for lesson: {lesson_id}",
                    extra={
                        "lesson_id": lesson_id,
                        "event_type": "frames_directory_missing",
                        "expected_path": str(lesson_frames_dir)
                    }
                )
                return []
            
            # Get frame files (common image extensions)
            frame_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
            frame_files = []
            
            for ext in frame_extensions:
                frame_files.extend(lesson_frames_dir.glob(f'*{ext}'))
            
            # Sort frames by name (assuming sequential naming)
            frame_files.sort()
            
            # Convert to strings and validate paths
            validated_frames = []
            for frame_file in frame_files:
                relative_path = str(frame_file.relative_to(self.frames_directory))
                if self.validator.validate_file_path(relative_path):
                    validated_frames.append(str(frame_file))
            
            self.logger.info(
                f"Found {len(validated_frames)} slide frames for lesson: {lesson_id}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "slide_frames_loaded",
                    "frame_count": len(validated_frames)
                }
            )
            
            return validated_frames
            
        except Exception as e:
            self.logger.error(
                f"Failed to get slide frames for lesson {lesson_id}: {e}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "slide_frames_error",
                    "error_details": {"error": str(e)}
                }
            )
            return []
    
    async def load_frame(self, frame_path: str) -> Optional[bytes]:
        """Load a single frame file"""
        try:
            # Check cache first
            if frame_path in self._frame_cache:
                self._frame_stats["frame_cache_hits"] += 1
                return self._frame_cache[frame_path]
            
            self._frame_stats["frame_cache_misses"] += 1
            
            # Validate path
            if not self.validator.validate_file_path(frame_path):
                raise ValueError(f"Invalid frame path: {frame_path}")
            
            # Load frame
            path = Path(frame_path)
            if not path.exists():
                raise FileNotFoundError(f"Frame file not found: {frame_path}")
            
            with open(path, 'rb') as file:
                frame_data = file.read()
            
            # Cache the frame (simple caching, production would use LRU)
            self._frame_cache[frame_path] = frame_data
            self._frame_stats["frames_loaded"] += 1
            
            return frame_data
            
        except Exception as e:
            self._frame_stats["loading_errors"] += 1
            self.logger.error(
                f"Failed to load frame: {frame_path} - {e}",
                extra={
                    "event_type": "frame_load_error",
                    "error_details": {"error": str(e), "frame_path": frame_path}
                }
            )
            return None
    
    async def get_frame_stats(self) -> Dict[str, Any]:
        """Get frame management statistics"""
        cache_hit_rate = (
            self._frame_stats["frame_cache_hits"] / 
            max(1, self._frame_stats["frame_cache_hits"] + self._frame_stats["frame_cache_misses"])
        )
        
        return {
            "frame_stats": self._frame_stats.copy(),
            "cache_hit_rate": cache_hit_rate,
            "cached_frames": len(self._frame_cache),
            "timestamp": time.time()
        }
