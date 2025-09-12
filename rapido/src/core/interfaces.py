"""
Core interfaces for Rapido system components
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass 
class AudioChunk:
    """Represents an audio chunk with metadata"""
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float
    chunk_id: str


@dataclass
class VideoFrame:
    """Represents a video frame with metadata"""
    data: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int
    fps: float


@dataclass
class SlideData:
    """Represents slide presentation data"""
    slide_id: str
    narration_text: str
    duration: float
    timing_segments: List[Dict[str, Any]]


class IAudioProcessor(ABC):
    """Interface for audio processing components"""
    
    @abstractmethod
    async def process_audio(self, audio_chunk: AudioChunk) -> AudioChunk:
        """Process an audio chunk and return the processed version"""
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        pass


class IVideoProcessor(ABC):
    """Interface for video processing components"""
    
    @abstractmethod
    async def process_frame(self, frame: VideoFrame) -> VideoFrame:
        """Process a video frame and return the processed version"""
        pass
    
    @abstractmethod
    async def get_fps_metrics(self) -> Dict[str, float]:
        """Get FPS metrics for logging"""
        pass


class ISyncTalkClient(ABC):
    """Interface for SyncTalk client implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to SyncTalk server"""
        pass
    
    @abstractmethod
    async def send_audio(self, audio_chunk: AudioChunk) -> None:
        """Send audio chunk to SyncTalk"""
        pass
    
    @abstractmethod
    async def receive_frames(self) -> AsyncGenerator[VideoFrame, None]:
        """Receive video frames from SyncTalk"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        pass
    
    @abstractmethod
    async def get_connection_metrics(self) -> Dict[str, Any]:
        """Get connection performance metrics"""
        pass


class ILiveKitPublisher(ABC):
    """Interface for LiveKit publishing"""
    
    @abstractmethod
    async def connect(self, room_name: str) -> bool:
        """Connect to LiveKit room"""
        pass
    
    @abstractmethod
    async def publish_frame(self, frame: VideoFrame) -> bool:
        """Publish video frame to room"""
        pass
    
    @abstractmethod
    async def get_output_metrics(self) -> Dict[str, Any]:
        """Get output FPS and quality metrics"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from room"""
        pass


class IDataParser(ABC):
    """Interface for data parsing components"""
    
    @abstractmethod
    async def parse_slide_data(self, file_path: str) -> SlideData:
        """Parse slide data from file"""
        pass
    
    @abstractmethod
    async def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate parsed data"""
        pass


class ITTSClient(ABC):
    """Interface for Text-to-Speech clients"""
    
    @abstractmethod
    async def synthesize_speech(self, text: str) -> AudioChunk:
        """Convert text to speech"""
        pass
    
    @abstractmethod
    async def get_synthesis_metrics(self) -> Dict[str, Any]:
        """Get TTS performance metrics"""
        pass
