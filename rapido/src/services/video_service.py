"""
Video processing services for Rapido system
"""

import asyncio
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple
import time
from pathlib import Path

from ..core.interfaces import IVideoProcessor, VideoFrame
from ..core.exceptions import VideoProcessingError
from ..core.logging_manager import get_logging_manager
from ..core.metrics import get_metrics_collector


class VideoProcessorService(IVideoProcessor):
    """Service for processing video frames"""
    
    def __init__(self, target_width: int = 854, target_height: int = 480):
        self.target_width = target_width
        self.target_height = target_height
        self.logger = get_logging_manager().get_logger("video_processor")
        self.metrics = get_metrics_collector()
        self._processing_stats = {
            "total_frames": 0,
            "resize_operations": 0,
            "color_corrections": 0,
            "error_count": 0
        }
    
    async def process_frame(self, frame: VideoFrame) -> VideoFrame:
        """Process a video frame with optimization"""
        start_time = time.time()
        
        try:
            if frame.data is None or frame.data.size == 0:
                raise VideoProcessingError("Empty frame data provided")
            
            # Record frame processing
            self.metrics.get_fps_counter("video_processing").record_frame()
            
            # Process frame
            processed_data = await self._optimize_frame(frame.data)
            
            # Create processed frame
            processed_frame = VideoFrame(
                data=processed_data,
                timestamp=frame.timestamp,
                frame_number=frame.frame_number,
                width=processed_data.shape[1],
                height=processed_data.shape[0],
                fps=frame.fps
            )
            
            # Update stats
            self._processing_stats["total_frames"] += 1
            
            # Log processing
            processing_time = time.time() - start_time
            self.logger.info(
                f"Processed video frame: {frame.frame_number}",
                extra={
                    "event_type": "frame_processed",
                    "performance_data": {
                        "processing_time": processing_time,
                        "frame_number": frame.frame_number,
                        "input_size": f"{frame.width}x{frame.height}",
                        "output_size": f"{processed_frame.width}x{processed_frame.height}"
                    }
                }
            )
            
            return processed_frame
            
        except Exception as e:
            self._processing_stats["error_count"] += 1
            self.logger.error(
                f"Video frame processing failed for frame {frame.frame_number}",
                extra={
                    "event_type": "frame_processing_error",
                    "error_details": {"error": str(e), "frame_number": frame.frame_number}
                }
            )
            raise VideoProcessingError(f"Failed to process frame: {e}")
    
    async def _optimize_frame(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply frame optimization"""
        try:
            # Resize if necessary
            current_height, current_width = frame_data.shape[:2]
            if current_width != self.target_width or current_height != self.target_height:
                frame_data = cv2.resize(frame_data, (self.target_width, self.target_height))
                self._processing_stats["resize_operations"] += 1
            
            # Apply color correction
            frame_data = await self._apply_color_correction(frame_data)
            self._processing_stats["color_corrections"] += 1
            
            # Ensure proper format
            if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
                # Convert BGR to RGB if needed
                frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            
            return frame_data
            
        except Exception as e:
            raise VideoProcessingError(f"Frame optimization failed: {e}")
    
    async def _apply_color_correction(self, frame: np.ndarray) -> np.ndarray:
        """Apply color correction and enhancement"""
        try:
            # Convert to LAB color space for better color correction
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            
            # Split LAB channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            
            # Convert back to RGB
            result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            return result
            
        except Exception as e:
            # Fallback: return original frame if color correction fails
            self.logger.warning(f"Color correction failed, using original frame: {e}")
            return frame
    
    async def get_fps_metrics(self) -> Dict[str, float]:
        """Get FPS metrics for video processing"""
        fps_counter = self.metrics.get_fps_counter("video_processing")
        stats = fps_counter.get_stats()
        return {
            "current_fps": stats["fps"],
            "min_fps": stats["min_fps"],
            "max_fps": stats["max_fps"],
            "avg_fps": stats["fps"],
            "frame_count": stats["frame_count"]
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        return {
            "processing_stats": self._processing_stats.copy(),
            "fps_metrics": await self.get_fps_metrics(),
            "timestamp": time.time()
        }


class FrameComposerService:
    """Service for composing frames with overlays"""
    
    def __init__(self, overlay_position: str = "bottom-right", overlay_scale: float = 0.5):
        self.overlay_position = overlay_position
        self.overlay_scale = overlay_scale
        self.logger = get_logging_manager().get_logger("frame_composer")
        self.metrics = get_metrics_collector()
        self._composition_stats = {
            "total_compositions": 0,
            "overlay_operations": 0,
            "chroma_key_operations": 0,
            "error_count": 0
        }
    
    async def compose_frame(self, base_frame: VideoFrame, overlay_frame: VideoFrame, 
                          lesson_id: str = None) -> VideoFrame:
        """Compose base frame with overlay"""
        start_time = time.time()
        
        try:
            # Record composition
            self.metrics.get_fps_counter("composer").record_frame()
            
            # Apply chroma key to overlay (remove green screen)
            overlay_processed = await self._apply_chroma_key(overlay_frame.data)
            
            # Scale overlay
            overlay_scaled = await self._scale_overlay(overlay_processed)
            
            # Compose frames
            composed_data = await self._composite_frames(base_frame.data, overlay_scaled)
            
            # Create composed frame
            composed_frame = VideoFrame(
                data=composed_data,
                timestamp=max(base_frame.timestamp, overlay_frame.timestamp),
                frame_number=base_frame.frame_number,
                width=base_frame.width,
                height=base_frame.height,
                fps=base_frame.fps
            )
            
            # Update stats
            self._composition_stats["total_compositions"] += 1
            self._composition_stats["overlay_operations"] += 1
            self._composition_stats["chroma_key_operations"] += 1
            
            # Log composition
            composition_time = time.time() - start_time
            self.logger.info(
                f"Composed frame: {base_frame.frame_number}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "frame_composed",
                    "composer_fps": self.metrics.get_fps_counter("composer").get_fps(),
                    "performance_data": {
                        "composition_time": composition_time,
                        "frame_number": base_frame.frame_number,
                        "overlay_scale": self.overlay_scale
                    }
                }
            )
            
            return composed_frame
            
        except Exception as e:
            self._composition_stats["error_count"] += 1
            self.logger.error(
                f"Frame composition failed for frame {base_frame.frame_number}",
                extra={
                    "lesson_id": lesson_id,
                    "event_type": "frame_composition_error",
                    "error_details": {"error": str(e), "frame_number": base_frame.frame_number}
                }
            )
            raise VideoProcessingError(f"Failed to compose frame: {e}")
    
    async def _apply_chroma_key(self, frame_data: np.ndarray) -> np.ndarray:
        """Apply chroma key (green screen removal)"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame_data, cv2.COLOR_RGB2HSV)
            
            # Define green screen color range
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([80, 255, 255])
            
            # Create mask for green pixels
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Apply Gaussian blur to mask edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Create alpha channel
            alpha = 255 - mask
            
            # Convert frame to RGBA
            if frame_data.shape[2] == 3:
                rgba_frame = np.dstack([frame_data, alpha])
            else:
                rgba_frame = frame_data.copy()
                rgba_frame[:, :, 3] = alpha
            
            return rgba_frame
            
        except Exception as e:
            self.logger.warning(f"Chroma key failed, using original frame: {e}")
            # Return original frame with full alpha if chroma key fails
            if frame_data.shape[2] == 3:
                alpha = np.full((frame_data.shape[0], frame_data.shape[1], 1), 255, dtype=np.uint8)
                return np.dstack([frame_data, alpha])
            return frame_data
    
    async def _scale_overlay(self, overlay_data: np.ndarray) -> np.ndarray:
        """Scale overlay frame"""
        try:
            if self.overlay_scale == 1.0:
                return overlay_data
            
            height, width = overlay_data.shape[:2]
            new_width = int(width * self.overlay_scale)
            new_height = int(height * self.overlay_scale)
            
            scaled = cv2.resize(overlay_data, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return scaled
            
        except Exception as e:
            raise VideoProcessingError(f"Overlay scaling failed: {e}")
    
    async def _composite_frames(self, base_frame: np.ndarray, overlay_frame: np.ndarray) -> np.ndarray:
        """Composite base frame with overlay"""
        try:
            base_height, base_width = base_frame.shape[:2]
            overlay_height, overlay_width = overlay_frame.shape[:2]
            
            # Calculate overlay position
            if self.overlay_position == "bottom-right":
                x_offset = base_width - overlay_width - 20
                y_offset = base_height - overlay_height - 20
            elif self.overlay_position == "bottom-left":
                x_offset = 20
                y_offset = base_height - overlay_height - 20
            elif self.overlay_position == "top-right":
                x_offset = base_width - overlay_width - 20
                y_offset = 20
            elif self.overlay_position == "top-left":
                x_offset = 20
                y_offset = 20
            else:  # center
                x_offset = (base_width - overlay_width) // 2
                y_offset = (base_height - overlay_height) // 2
            
            # Ensure overlay fits within base frame
            x_offset = max(0, min(x_offset, base_width - overlay_width))
            y_offset = max(0, min(y_offset, base_height - overlay_height))
            
            # Create result frame
            result = base_frame.copy()
            
            # Extract region of interest
            roi = result[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width]
            
            if overlay_frame.shape[2] == 4:  # RGBA overlay
                # Use alpha blending
                alpha = overlay_frame[:, :, 3] / 255.0
                for c in range(3):
                    roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay_frame[:, :, c]
            else:  # RGB overlay
                # Simple replacement
                result[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = overlay_frame
            
            return result
            
        except Exception as e:
            raise VideoProcessingError(f"Frame composition failed: {e}")
    
    async def get_composition_metrics(self) -> Dict[str, Any]:
        """Get frame composition metrics"""
        composer_fps = self.metrics.get_fps_counter("composer").get_stats()
        return {
            "composition_stats": self._composition_stats.copy(),
            "composer_fps": composer_fps["fps"],
            "composer_fps_stats": composer_fps,
            "timestamp": time.time()
        }
