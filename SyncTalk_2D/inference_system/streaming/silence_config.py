"""
Configuration for optimizing silence generation with multiple streams.
This module provides dynamic configuration based on the number of active streams
to prevent FPS drops during silence periods.
"""

class SilenceOptimizationConfig:
    """Configuration for multi-stream silence optimization"""
    
    # Buffer sizes during silence (seconds per stream count)
    SILENCE_BUFFER_SIZES = {
        1: 0.5,   # Single stream: 0.5s buffer (low latency)
        2: 1.0,   # Two streams: slightly larger to reduce contention
        3: 1.2,   # Three streams: 1s buffer
        4: 1.8,   # Four+ streams: larger buffer for efficiency
    }
    
    # Sleep durations during silence (milliseconds)
    SILENCE_SLEEP_DURATIONS = {
        'generating': {
            1: 2,    # Single stream: 5ms
            2: 1,   # Two streams: 10ms to reduce contention
            3: 8,   # Three streams: 15ms
            4: 12,   # Four+ streams: 20ms
        },
        'buffer_full': {
            1: 25,   # Single stream: 40ms
            2: 12,   # Two streams: 60ms
            3: 40,   # Three streams: 80ms
            4: 60,  # Four+ streams: 100ms
        },
        'normal': {
            1: 4,   # Single stream: 10ms
            2: 2,   # Two streams: 20ms
            3: 15,   # Three streams: 30ms
            4: 25,   # Four+ streams: 40ms
        }
    }
    
    # Batch sizes for silence generation (frames)
    SILENCE_BATCH_SIZES = {
        1: 12,   # Single stream: 12 frames (0.48s)
        2: 12,   # Two streams: 20 frames (0.8s) - larger batches
        3: 20,   # Three streams: 25 frames (1s)
        4: 25,   # Four+ streams: 30 frames (1.2s)
    }
    
    # Refill thresholds during silence (percentage of buffer)
    SILENCE_REFILL_THRESHOLDS = {
        1: 0.5,   # Single stream: refill at 50%
        2: 0.5,   # Two streams: refill at 80% to batch better
        3: 0.4,   # Three streams: refill at 30%
        4: 0.3,  # Four+ streams: refill at 25%
    }
    
    @classmethod
    def get_buffer_size(cls, stream_count: int, base_size: float = 0.5) -> float:
        """Get optimal buffer size for given stream count during silence"""
        return cls.SILENCE_BUFFER_SIZES.get(min(stream_count, 4), 1.2)
    
    @classmethod
    def get_sleep_duration(cls, stream_count: int, state: str) -> float:
        """Get optimal sleep duration in seconds"""
        ms = cls.SILENCE_SLEEP_DURATIONS.get(state, {}).get(min(stream_count, 4), 10)
        return ms / 1000.0
    
    @classmethod
    def get_batch_size(cls, stream_count: int) -> int:
        """Get optimal batch size for silence generation"""
        return cls.SILENCE_BATCH_SIZES.get(min(stream_count, 4), 30)
    
    @classmethod
    def get_refill_threshold(cls, stream_count: int) -> float:
        """Get optimal refill threshold as percentage"""
        return cls.SILENCE_REFILL_THRESHOLDS.get(min(stream_count, 4), 0.25)
    
    @classmethod
    def should_stagger_streams(cls, stream_count: int) -> bool:
        """Whether to use staggered timing for streams"""
        return stream_count > 1
    
    @classmethod
    def get_stream_offset(cls, stream_id: str, max_offset_ms: int = 10) -> float:
        """Get a consistent offset for a stream to prevent simultaneous processing"""
        # Use hash to get consistent but different offset per stream
        offset_ms = (hash(stream_id) % max_offset_ms)
        return offset_ms / 1000.0
