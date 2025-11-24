import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class SyncEventType(Enum):
    AUDIO_START = "audio_start"
    AUDIO_CHUNK = "audio_chunk"
    AVATAR_FRAME = "avatar_frame"
    SLIDE_ANIMATION = "slide_animation"
    WORD_TIMING = "word_timing"

@dataclass
class SyncEvent:
    """Represents a synchronization event."""
    event_type: SyncEventType
    timestamp_ms: float
    data: Any
    frame_index: Optional[int] = None
    processed: bool = False

@dataclass
class TimingToken:
    """Represents a timing token from narration data."""
    id: str
    text: str
    start_time: float  # milliseconds
    end_time: float    # milliseconds
    duration: float    # milliseconds
    token_type: str = "word"

class TimingSynchronizer:
    """Synchronizes audio, avatar frames, and slide animations."""
    
    def __init__(self, frame_rate: int = 30):
        self.frame_rate = frame_rate
        self.frame_duration_ms = 1000 / frame_rate  # milliseconds per frame
        
        self.events = []  # List of SyncEvent
        self.timing_tokens = []  # List of TimingToken
        self.animation_triggers = []  # List of animation trigger events
        
        self.start_time = None
        self.current_time_ms = 0
        self.current_frame = 0
        
        self.event_callbacks = {}  # event_type -> callback function
        
    def set_timing_tokens(self, tokens: List[Dict[str, Any]]):
        """Set timing tokens from narration data."""
        self.timing_tokens = []
        for token in tokens:
            timing_token = TimingToken(
                id=token.get('id', ''),
                text=token.get('text', ''),
                start_time=token.get('startTime', 0),
                end_time=token.get('endTime', 0),
                duration=token.get('duration', 0),
                token_type=token.get('type', 'word')
            )
            self.timing_tokens.append(timing_token)
        
        logger.info(f"Loaded {len(self.timing_tokens)} timing tokens")
    
    def set_animation_triggers(self, triggers: List[Dict[str, Any]]):
        """Set animation triggers from slide data."""
        self.animation_triggers = triggers
        logger.info(f"Loaded {len(self.animation_triggers)} animation triggers")
    
    def add_event(self, event: SyncEvent):
        """Add synchronization event."""
        self.events.append(event)
        # Keep events sorted by timestamp
        self.events.sort(key=lambda e: e.timestamp_ms)
    
    def register_callback(self, event_type: SyncEventType, callback: Callable):
        """Register callback for specific event type."""
        self.event_callbacks[event_type] = callback
    
    def start_synchronization(self):
        """Start the synchronization timer."""
        self.start_time = time.time() * 1000  # Convert to milliseconds
        self.current_time_ms = 0
        self.current_frame = 0
        logger.info("Synchronization started")
    
    def get_current_time_ms(self) -> float:
        """Get current time in milliseconds since synchronization started."""
        if self.start_time is None:
            return 0
        return (time.time() * 1000) - self.start_time
    
    def get_current_frame_index(self) -> int:
        """Get current frame index based on elapsed time."""
        current_time = self.get_current_time_ms()
        return int(current_time / self.frame_duration_ms)
    
    def get_frame_timestamp(self, frame_index: int) -> float:
        """Get timestamp for specific frame index."""
        return frame_index * self.frame_duration_ms
    
    def get_active_tokens_at_time(self, timestamp_ms: float) -> List[TimingToken]:
        """Get timing tokens that are active at given timestamp."""
        active_tokens = []
        for token in self.timing_tokens:
            if token.start_time <= timestamp_ms <= token.end_time:
                active_tokens.append(token)
        return active_tokens
    
    def get_animation_triggers_at_time(self, timestamp_ms: float) -> List[Dict[str, Any]]:
        """Get animation triggers that should fire at given timestamp."""
        triggers = []
        
        # Find tokens that match the current time
        active_tokens = self.get_active_tokens_at_time(timestamp_ms)
        
        for trigger in self.animation_triggers:
            trigger_id = trigger.get('trigger_id')
            
            # Check if any active token matches this trigger
            for token in active_tokens:
                if token.id == trigger_id:
                    # Check if this is the start of the token (for entry animations)
                    if (trigger.get('trigger_type') == 'entry' and 
                        abs(timestamp_ms - token.start_time) < self.frame_duration_ms):
                        triggers.append(trigger)
                    # Check if this is the end of the token (for exit animations)
                    elif (trigger.get('trigger_type') == 'exit' and 
                          abs(timestamp_ms - token.end_time) < self.frame_duration_ms):
                        triggers.append(trigger)
        
        return triggers
    
    async def process_events_at_time(self, timestamp_ms: float):
        """Process all events that should occur at given timestamp."""
        current_time = self.get_current_time_ms()
        
        # Process timing tokens
        active_tokens = self.get_active_tokens_at_time(timestamp_ms)
        if active_tokens and SyncEventType.WORD_TIMING in self.event_callbacks:
            await self.event_callbacks[SyncEventType.WORD_TIMING](active_tokens, timestamp_ms)
        
        # Process animation triggers
        animation_triggers = self.get_animation_triggers_at_time(timestamp_ms)
        if animation_triggers and SyncEventType.SLIDE_ANIMATION in self.event_callbacks:
            await self.event_callbacks[SyncEventType.SLIDE_ANIMATION](animation_triggers, timestamp_ms)
        
        # Process queued events
        events_to_process = [e for e in self.events if not e.processed and e.timestamp_ms <= timestamp_ms]
        
        for event in events_to_process:
            if event.event_type in self.event_callbacks:
                try:
                    await self.event_callbacks[event.event_type](event.data, event.timestamp_ms)
                    event.processed = True
                except Exception as e:
                    logger.error(f"Error processing event {event.event_type}: {e}")
    
    async def synchronization_loop(self, duration_ms: Optional[float] = None):
        """Main synchronization loop."""
        if self.start_time is None:
            self.start_synchronization()
        
        target_end_time = None
        if duration_ms:
            target_end_time = self.start_time + duration_ms
        
        try:
            while True:
                current_time = self.get_current_time_ms()
                
                # Check if we've reached the end time
                if target_end_time and time.time() * 1000 >= target_end_time:
                    break
                
                # Process events for current time
                await self.process_events_at_time(current_time)
                
                # Update current frame
                self.current_frame = self.get_current_frame_index()
                
                # Sleep until next frame
                await asyncio.sleep(self.frame_duration_ms / 1000)
                
        except asyncio.CancelledError:
            logger.info("Synchronization loop cancelled")
        except Exception as e:
            logger.error(f"Error in synchronization loop: {e}")
        finally:
            logger.info("Synchronization loop ended")
    
    def calculate_avatar_frame_timing(self, audio_duration_ms: float) -> List[Dict[str, Any]]:
        """
        Calculate expected avatar frame timing based on audio duration.
        
        Args:
            audio_duration_ms: Total audio duration in milliseconds
            
        Returns:
            List of frame timing info
        """
        frame_timings = []
        total_frames = int(audio_duration_ms / self.frame_duration_ms)
        
        for frame_idx in range(total_frames):
            timestamp = frame_idx * self.frame_duration_ms
            frame_timings.append({
                'frame_index': frame_idx,
                'timestamp_ms': timestamp,
                'expected_time': timestamp / 1000  # seconds
            })
        
        return frame_timings
    
    def synchronize_avatar_frames(self, avatar_frames: Dict[int, Any], audio_start_time: float) -> Dict[int, Any]:
        """
        Synchronize avatar frames with audio timing.
        
        Args:
            avatar_frames: Dictionary of frame_index -> frame_data
            audio_start_time: Audio start timestamp in milliseconds
            
        Returns:
            Dictionary of synchronized frames with timing info
        """
        synchronized_frames = {}
        
        for frame_idx, frame_data in avatar_frames.items():
            # Calculate when this frame should be displayed
            display_time = audio_start_time + (frame_idx * self.frame_duration_ms)
            
            # Find corresponding slide frame index
            slide_frame_idx = int(display_time / self.frame_duration_ms)
            
            synchronized_frames[slide_frame_idx] = {
                'avatar_frame': frame_data,
                'display_time': display_time,
                'slide_frame_index': slide_frame_idx,
                'avatar_frame_index': frame_idx
            }
        
        return synchronized_frames
    
    def get_synchronization_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        current_time = self.get_current_time_ms()
        
        return {
            'current_time_ms': current_time,
            'current_frame': self.current_frame,
            'total_events': len(self.events),
            'processed_events': sum(1 for e in self.events if e.processed),
            'timing_tokens_count': len(self.timing_tokens),
            'animation_triggers_count': len(self.animation_triggers),
            'frame_rate': self.frame_rate,
            'frame_duration_ms': self.frame_duration_ms
        }

class AudioVideoSynchronizer:
    """Specialized synchronizer for audio-video alignment."""
    
    def __init__(self, frame_rate: int = 30, audio_sample_rate: int = 22050):
        self.frame_rate = frame_rate
        self.audio_sample_rate = audio_sample_rate
        self.samples_per_frame = audio_sample_rate / frame_rate
        
    def audio_sample_to_frame(self, sample_index: int) -> int:
        """Convert audio sample index to video frame index."""
        return int(sample_index / self.samples_per_frame)
    
    def frame_to_audio_sample(self, frame_index: int) -> int:
        """Convert video frame index to audio sample index."""
        return int(frame_index * self.samples_per_frame)
    
    def timestamp_to_frame(self, timestamp_ms: float) -> int:
        """Convert timestamp to frame index."""
        return int((timestamp_ms / 1000) * self.frame_rate)
    
    def frame_to_timestamp(self, frame_index: int) -> float:
        """Convert frame index to timestamp in milliseconds."""
        return (frame_index / self.frame_rate) * 1000
    
    def calculate_sync_offset(self, audio_start_time: float, video_start_time: float) -> float:
        """Calculate synchronization offset between audio and video."""
        return audio_start_time - video_start_time
    
    def apply_sync_correction(self, frames: Dict[int, Any], offset_ms: float) -> Dict[int, Any]:
        """Apply synchronization correction to frames."""
        corrected_frames = {}
        offset_frames = self.timestamp_to_frame(offset_ms)
        
        for frame_idx, frame_data in frames.items():
            new_frame_idx = frame_idx + offset_frames
            if new_frame_idx >= 0:  # Don't include negative frame indices
                corrected_frames[new_frame_idx] = frame_data
        
        return corrected_frames
