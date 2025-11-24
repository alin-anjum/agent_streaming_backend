import cv2
import numpy as np
from PIL import Image
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import ffmpeg
import asyncio

logger = logging.getLogger(__name__)

class VideoGenerator:
    """Generates final video output with synchronized audio and visual elements."""
    
    def __init__(
        self,
        output_path: str,
        frame_rate: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        video_codec: str = "libx264",
        audio_codec: str = "aac"
    ):
        self.output_path = Path(output_path)
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer
        self.video_writer = None
        self.temp_video_path = None
        self.temp_audio_path = None
        
    def initialize_video_writer(self, temp_dir: str) -> str:
        """Initialize video writer for frame output."""
        # Create temporary video file
        self.temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.temp_video_path,
            fourcc,
            self.frame_rate,
            self.resolution
        )
        
        if not self.video_writer.isOpened():
            raise RuntimeError("Failed to initialize video writer")
        
        logger.info(f"Video writer initialized: {self.temp_video_path}")
        return self.temp_video_path
    
    def write_frame(self, frame: Image.Image):
        """Write a single frame to the video."""
        if not self.video_writer:
            raise RuntimeError("Video writer not initialized")
        
        # Convert PIL Image to OpenCV format
        frame_rgb = frame.convert('RGB')
        frame_array = np.array(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        
        # Resize if necessary
        if frame_bgr.shape[:2][::-1] != self.resolution:
            frame_bgr = cv2.resize(frame_bgr, self.resolution)
        
        self.video_writer.write(frame_bgr)
    
    def finalize_video(self):
        """Finalize and close video writer."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            logger.info("Video writer finalized")
    
    def save_audio(self, audio_data: bytes, temp_dir: str) -> str:
        """Save audio data to temporary file."""
        self.temp_audio_path = os.path.join(temp_dir, "temp_audio.mp3")
        
        with open(self.temp_audio_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Audio saved: {self.temp_audio_path}")
        return self.temp_audio_path
    
    def combine_audio_video(self, video_path: str, audio_path: str) -> str:
        """Combine video and audio using ffmpeg."""
        try:
            logger.info("Combining audio and video...")
            
            # Use ffmpeg-python to combine audio and video
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(audio_path)
            
            output = ffmpeg.output(
                video_input,
                audio_input,
                str(self.output_path),
                vcodec=self.video_codec,
                acodec=self.audio_codec,
                **{'b:v': '2M', 'b:a': '192k'}  # Set bitrates
            )
            
            # Overwrite output file if it exists
            output = ffmpeg.overwrite_output(output)
            
            # Run the ffmpeg command
            ffmpeg.run(output, capture_stdout=True, capture_stderr=True)
            
            logger.info(f"Video generated successfully: {self.output_path}")
            return str(self.output_path)
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"Error combining audio and video: {e}")
            raise
    
    async def generate_video_from_frames(
        self,
        frames: Dict[int, Image.Image],
        audio_data: bytes,
        total_duration_ms: float
    ) -> str:
        """
        Generate video from frame dictionary and audio data.
        
        Args:
            frames: Dictionary of frame_index -> PIL.Image
            audio_data: Audio data as bytes
            total_duration_ms: Total duration in milliseconds
            
        Returns:
            str: Path to generated video file
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Initialize video writer
                temp_video_path = self.initialize_video_writer(temp_dir)
                
                # Calculate total frames needed
                total_frames = int((total_duration_ms / 1000) * self.frame_rate)
                
                logger.info(f"Generating video with {total_frames} frames")
                
                # Write frames to video
                for frame_idx in range(total_frames):
                    if frame_idx in frames:
                        # Use provided frame
                        frame = frames[frame_idx]
                    else:
                        # Use previous frame or create blank frame
                        frame = self._get_fallback_frame(frames, frame_idx)
                    
                    self.write_frame(frame)
                    
                    # Log progress
                    if frame_idx % 30 == 0:  # Log every second
                        progress = (frame_idx / total_frames) * 100
                        logger.info(f"Video generation progress: {progress:.1f}%")
                
                # Finalize video
                self.finalize_video()
                
                # Save audio
                temp_audio_path = self.save_audio(audio_data, temp_dir)
                
                # Combine audio and video
                final_video_path = self.combine_audio_video(temp_video_path, temp_audio_path)
                
                return final_video_path
                
            except Exception as e:
                logger.error(f"Error generating video: {e}")
                raise
    
    def _get_fallback_frame(self, frames: Dict[int, Image.Image], frame_idx: int) -> Image.Image:
        """Get fallback frame when specific frame is not available."""
        # Try to find the most recent frame
        for i in range(frame_idx, -1, -1):
            if i in frames:
                return frames[i]
        
        # If no previous frame, try to find next frame
        for i in range(frame_idx, max(frames.keys()) + 1):
            if i in frames:
                return frames[i]
        
        # Create blank frame as last resort
        return Image.new('RGB', self.resolution, color='black')
    
    async def generate_video_with_progress(
        self,
        frames: Dict[int, Image.Image],
        audio_data: bytes,
        total_duration_ms: float,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Generate video with progress callbacks.
        
        Args:
            frames: Dictionary of frame_index -> PIL.Image
            audio_data: Audio data as bytes
            total_duration_ms: Total duration in milliseconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Path to generated video file
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Initialize video writer
                temp_video_path = self.initialize_video_writer(temp_dir)
                
                # Calculate total frames needed
                total_frames = int((total_duration_ms / 1000) * self.frame_rate)
                
                if progress_callback:
                    await progress_callback("video_generation_started", {
                        "total_frames": total_frames,
                        "duration_ms": total_duration_ms
                    })
                
                # Write frames to video
                for frame_idx in range(total_frames):
                    if frame_idx in frames:
                        frame = frames[frame_idx]
                    else:
                        frame = self._get_fallback_frame(frames, frame_idx)
                    
                    self.write_frame(frame)
                    
                    # Progress callback
                    if progress_callback and frame_idx % 10 == 0:
                        progress = (frame_idx / total_frames) * 100
                        await progress_callback("video_frame_written", {
                            "frame_index": frame_idx,
                            "progress_percent": progress
                        })
                
                # Finalize video
                self.finalize_video()
                
                if progress_callback:
                    await progress_callback("video_frames_complete", {})
                
                # Save audio
                temp_audio_path = self.save_audio(audio_data, temp_dir)
                
                if progress_callback:
                    await progress_callback("audio_saved", {})
                
                # Combine audio and video
                final_video_path = self.combine_audio_video(temp_video_path, temp_audio_path)
                
                if progress_callback:
                    await progress_callback("video_generation_complete", {
                        "output_path": final_video_path
                    })
                
                return final_video_path
                
            except Exception as e:
                if progress_callback:
                    await progress_callback("video_generation_error", {
                        "error": str(e)
                    })
                raise

class FrameSequenceGenerator:
    """Generates frame sequences for video output."""
    
    def __init__(self, frame_rate: int = 30):
        self.frame_rate = frame_rate
        self.frame_duration_ms = 1000 / frame_rate
    
    def interpolate_missing_frames(
        self,
        frames: Dict[int, Image.Image],
        total_frames: int
    ) -> Dict[int, Image.Image]:
        """
        Interpolate missing frames using simple frame holding.
        
        Args:
            frames: Sparse dictionary of available frames
            total_frames: Total number of frames needed
            
        Returns:
            Complete dictionary of frames
        """
        complete_frames = {}
        
        # Sort frame indices
        frame_indices = sorted(frames.keys())
        
        if not frame_indices:
            # No frames available, create blank frames
            blank_frame = Image.new('RGB', (1920, 1080), color='black')
            for i in range(total_frames):
                complete_frames[i] = blank_frame.copy()
            return complete_frames
        
        current_frame = frames[frame_indices[0]]
        
        for frame_idx in range(total_frames):
            if frame_idx in frames:
                # Use actual frame
                current_frame = frames[frame_idx]
                complete_frames[frame_idx] = current_frame
            else:
                # Use last known frame
                complete_frames[frame_idx] = current_frame.copy()
        
        return complete_frames
    
    def create_transition_frames(
        self,
        start_frame: Image.Image,
        end_frame: Image.Image,
        transition_frames: int,
        transition_type: str = "fade"
    ) -> List[Image.Image]:
        """
        Create transition frames between two images.
        
        Args:
            start_frame: Starting frame
            end_frame: Ending frame
            transition_frames: Number of transition frames to create
            transition_type: Type of transition ("fade", "slide", "dissolve")
            
        Returns:
            List of transition frames
        """
        frames = []
        
        if transition_type == "fade":
            for i in range(transition_frames):
                alpha = i / (transition_frames - 1) if transition_frames > 1 else 1
                
                # Blend frames
                blended = Image.blend(start_frame.convert('RGBA'), end_frame.convert('RGBA'), alpha)
                frames.append(blended.convert('RGB'))
        
        elif transition_type == "slide":
            # Simple slide transition (left to right)
            width, height = start_frame.size
            
            for i in range(transition_frames):
                progress = i / (transition_frames - 1) if transition_frames > 1 else 1
                offset = int(width * progress)
                
                # Create composite frame
                composite = Image.new('RGB', (width, height))
                
                # Paste start frame (sliding out)
                if offset < width:
                    start_crop = start_frame.crop((offset, 0, width, height))
                    composite.paste(start_crop, (0, 0))
                
                # Paste end frame (sliding in)
                if offset > 0:
                    end_crop = end_frame.crop((0, 0, offset, height))
                    composite.paste(end_crop, (width - offset, 0))
                
                frames.append(composite)
        
        else:  # dissolve or default
            frames = self.create_transition_frames(start_frame, end_frame, transition_frames, "fade")
        
        return frames
    
    def add_frame_effects(
        self,
        frames: Dict[int, Image.Image],
        effects: Dict[str, Any]
    ) -> Dict[int, Image.Image]:
        """
        Add visual effects to frame sequence.
        
        Args:
            frames: Dictionary of frames
            effects: Effects configuration
            
        Returns:
            Dictionary of processed frames
        """
        processed_frames = {}
        
        for frame_idx, frame in frames.items():
            processed_frame = frame.copy()
            
            # Apply fade in/out
            if "fade_in_frames" in effects and frame_idx < effects["fade_in_frames"]:
                alpha = frame_idx / effects["fade_in_frames"]
                black_frame = Image.new('RGB', frame.size, color='black')
                processed_frame = Image.blend(black_frame, processed_frame, alpha)
            
            total_frames = len(frames)
            if "fade_out_frames" in effects and frame_idx >= total_frames - effects["fade_out_frames"]:
                fade_progress = (total_frames - frame_idx) / effects["fade_out_frames"]
                black_frame = Image.new('RGB', frame.size, color='black')
                processed_frame = Image.blend(black_frame, processed_frame, fade_progress)
            
            processed_frames[frame_idx] = processed_frame
        
        return processed_frames
