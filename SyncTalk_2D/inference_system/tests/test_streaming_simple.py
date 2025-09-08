# inference_system/tests/test_streaming_simple.py

import os
import sys
import cv2
import time
import wave
import numpy as np
import subprocess
from collections import Counter, deque
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import psutil
import json

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from inference_system.api import InferenceAPI
from inference_system.core.video_frame_manager import AllFramesMemory
from inference_system.core.landmark_manager import LandmarkManager
from inference_system.utils.profiler import PerformanceProfiler

from inference_system.tests.elevenlabs_tts_generator import create_synthetic_tts_generator, create_elevenlabs_tts_generator


@dataclass
class FrameMetrics:
    """Track detailed metrics for each frame"""
    frame_id: int
    submit_time: float
    receive_time: float
    write_time: float
    latency_ms: float = 0.0
    
class StreamingMetrics:
    """Track overall streaming performance metrics"""
    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.target_frame_time_ms = 1000.0 / target_fps
        
        # Latency tracking
        self.frame_latencies = deque(maxlen=1000)
        self.frame_metrics = {}
        self.lock = threading.Lock()
        
        # Timing tracking
        self.first_frame_time = None
        self.last_frame_time = None
        self.frame_times = deque(maxlen=100)
        
        # Statistics
        self.frames_submitted = 0
        self.frames_received = 0
        self.frames_written = 0
        self.dropped_frames = 0
        self.late_frames = 0
        
    def submit_frame(self, frame_id: int):
        """Record when a frame is submitted for processing"""
        with self.lock:
            self.frame_metrics[frame_id] = FrameMetrics(
                frame_id=frame_id,
                submit_time=time.perf_counter(),
                receive_time=0,
                write_time=0
            )
            self.frames_submitted += 1
    
    def receive_frame(self, frame_id: int):
        """Record when a frame is received from processing"""
        current_time = time.perf_counter()
        with self.lock:
            if frame_id in self.frame_metrics:
                metrics = self.frame_metrics[frame_id]
                metrics.receive_time = current_time
                metrics.latency_ms = (current_time - metrics.submit_time) * 1000
                
                self.frame_latencies.append(metrics.latency_ms)
                self.frames_received += 1
                
                # Track if frame is late
                if metrics.latency_ms > self.target_frame_time_ms:
                    self.late_frames += 1
                
                # Track frame timing
                if self.last_frame_time:
                    frame_delta = (current_time - self.last_frame_time) * 1000
                    self.frame_times.append(frame_delta)
                
                self.last_frame_time = current_time
                
                if self.first_frame_time is None:
                    self.first_frame_time = current_time
    
    def write_frame(self, frame_id: int):
        """Record when a frame is written"""
        with self.lock:
            if frame_id in self.frame_metrics:
                self.frame_metrics[frame_id].write_time = time.perf_counter()
                self.frames_written += 1
    
    def get_statistics(self) -> Dict:
        """Get current performance statistics"""
        with self.lock:
            if not self.frame_latencies:
                return {}
            
            latencies = list(self.frame_latencies)
            frame_deltas = list(self.frame_times) if self.frame_times else [0]
            
            return {
                'frames_submitted': self.frames_submitted,
                'frames_received': self.frames_received,
                'frames_written': self.frames_written,
                'dropped_frames': self.dropped_frames,
                'late_frames': self.late_frames,
                'latency_ms': {
                    'mean': np.mean(latencies),
                    'p50': np.percentile(latencies, 50),
                    'p90': np.percentile(latencies, 90),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                    'max': np.max(latencies),
                    'min': np.min(latencies)
                },
                'frame_timing_ms': {
                    'mean': np.mean(frame_deltas),
                    'std': np.std(frame_deltas),
                    'max': np.max(frame_deltas),
                    'min': np.min(frame_deltas)
                },
                'effective_fps': 1000.0 / np.mean(frame_deltas) if np.mean(frame_deltas) > 0 else 0,
                'late_frame_percentage': (self.late_frames / self.frames_received * 100) if self.frames_received > 0 else 0
            }

def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds"""
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def create_audio_with_silences_chunk_generator(audio_path: str, 
                                                     chunk_duration_ms: int = 500,
                                                     tts_duration_seconds: float = 10.0,
                                                     pause_duration_seconds: float = 10.0,
                                                     network_latency_ms: float = 50.0):
    """
    Create a generator that simulates a real-time streaming TTS system.
    Starts with pause (yielding None), then alternates to audio chunks.
    Adds network latency simulation and adjusts pause duration to maintain consistent timing.
    
    Args:
        audio_path: Path to WAV file (must be 16kHz mono 16-bit)
        chunk_duration_ms: Duration of each audio chunk in milliseconds
        tts_duration_seconds: Target duration of audio to stream
        pause_duration_seconds: Base pause duration (will be adjusted based on actual streaming time)
        network_latency_ms: Simulated network latency in milliseconds when yielding audio chunks
    """
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        
        if sample_rate != 16000 or channels != 1 or sample_width != 2:
            raise ValueError(f"Expected 16kHz mono 16-bit audio, got {sample_rate}Hz, "
                           f"{channels} channels, {sample_width*8}-bit")
        
        chunk_frames = int(sample_rate * chunk_duration_ms / 1000)
        network_latency_seconds = network_latency_ms / 1000.0
        
        # State tracking
        segment_count = 0
        total_audio_time = 0.0
        last_status_time = time.time()
        
        # START WITH PAUSE STATE
        is_paused = True
        pause_start_time = time.time()
        actual_pause_duration = pause_duration_seconds  # For first pause, use base duration
        
        segment_start_time = None
        audio_streaming_duration = 0.0  # Track actual time spent streaming audio
        samples_sent_in_segment = 0
        samples_per_segment = int(sample_rate * tts_duration_seconds)
        
        # Track None yields
        none_yield_count = 0
        last_none_print_time = time.time()
        none_print_interval = 0.5  # Print None yield status every 0.5 seconds
        
        print(f"[TTS Simulator] Starting: {pause_duration_seconds}s base pause, {tts_duration_seconds}s audio cycles")
        print(f"[TTS Simulator] Network latency simulation: {network_latency_ms}ms per chunk")
        print(f"[TTS Simulator] Status: STARTING WITH PAUSE - will yield None continuously...")
        
        while True:
            current_time = time.time()
            
            # Handle pause state
            if is_paused:
                # Yield None continuously during pause
                yield None
                none_yield_count += 1
                
                # Print None yield status periodically
                if current_time - last_none_print_time >= none_print_interval:
                    elapsed_pause = current_time - pause_start_time
                    remaining_pause = actual_pause_duration - elapsed_pause
                    if segment_count == 0:
                        print(f"[TTS Simulator] INITIAL PAUSE: Yielding None ({none_yield_count} times so far) - "
                              f"{elapsed_pause:.1f}s elapsed, {remaining_pause:.1f}s remaining")
                    else:
                        print(f"[TTS Simulator] PAUSED (after segment {segment_count}): "
                              f"Yielding None ({none_yield_count} times so far) - "
                              f"{elapsed_pause:.1f}s elapsed, {remaining_pause:.1f}s remaining "
                              f"(adjusted pause: {actual_pause_duration:.1f}s)")
                    last_none_print_time = current_time
                
                # Check if pause is over
                if current_time - pause_start_time >= actual_pause_duration:
                    # End of pause
                    print(f"[TTS Simulator] Pause complete. Yielded None {none_yield_count} times during {actual_pause_duration:.1f}s pause.")
                    none_yield_count = 0
                    is_paused = False
                    segment_count += 1
                    segment_start_time = current_time
                    samples_sent_in_segment = 0
                    print(f"[TTS Simulator] Starting audio - Beginning segment {segment_count}")
                    print(f"[TTS Simulator] Status: STREAMING AUDIO (with {network_latency_ms}ms latency per chunk)")
                    last_status_time = current_time
                else:
                    # Continue pause
                    time.sleep(0.01)  # Small sleep to avoid busy loop
                    continue
            
            # Print audio streaming status every second
            if current_time - last_status_time >= 1.0:
                elapsed_segment = current_time - segment_start_time
                audio_progress = (samples_sent_in_segment / samples_per_segment) * 100
                chunks_sent = samples_sent_in_segment // chunk_frames
                print(f"[TTS Simulator] STREAMING: Segment {segment_count}, "
                      f"{elapsed_segment:.1f}s elapsed, {audio_progress:.1f}% complete, "
                      f"{chunks_sent} chunks sent")
                last_status_time = current_time
            
            # Read audio chunk
            chunk = wav_file.readframes(chunk_frames)
            if not chunk:
                print(f"\n[TTS Simulator] Audio file exhausted. Total time streamed: {total_audio_time:.1f}s")
                return
            
            # Simulate network latency before yielding
            time.sleep(network_latency_seconds)
            yield chunk
            
            # Update counters
            chunk_samples = len(chunk) // 2  # 16-bit audio = 2 bytes per sample
            samples_sent_in_segment += chunk_samples
            total_audio_time = (total_audio_time * sample_rate + chunk_samples) / sample_rate
            
            # Check if we need to start a pause
            if samples_sent_in_segment >= samples_per_segment:
                # Calculate how long it actually took to stream the audio
                audio_streaming_duration = current_time - segment_start_time
                
                # Calculate adjusted pause duration
                # If audio was streamed faster than tts_duration_seconds, pause needs to be longer
                time_saved = tts_duration_seconds - audio_streaming_duration
                actual_pause_duration = pause_duration_seconds + time_saved
                
                print(f"\n[TTS Simulator] Completed segment {segment_count}. Total audio sent: {total_audio_time:.1f}s")
                print(f"[TTS Simulator] Audio streaming took {audio_streaming_duration:.1f}s "
                      f"(target was {tts_duration_seconds:.1f}s)")
                print(f"[TTS Simulator] Adjusting pause duration: {pause_duration_seconds:.1f}s + {time_saved:.1f}s = {actual_pause_duration:.1f}s")
                print(f"[TTS Simulator] Starting adjusted pause - will yield None continuously...")
                
                is_paused = True
                pause_start_time = current_time
                last_none_print_time = current_time

def create_simple_audio_chunk_generator(audio_path: str, chunk_duration_ms: int = 500):
    """
    Create a generator that yields audio chunks from a WAV file.
    
    Args:
        audio_path: Path to WAV file
        chunk_duration_ms: Duration of each chunk in milliseconds
    
    Yields:
        bytes: PCM audio chunks
    """
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        
        # Verify format
        if sample_rate != 16000 or channels != 1 or sample_width != 2:
            raise ValueError(f"Expected 16kHz mono 16-bit audio, got {sample_rate}Hz, "
                           f"{channels} channels, {sample_width*8}-bit")
        
        # Calculate chunk size in frames
        chunk_frames = int(sample_rate * chunk_duration_ms / 1000)
        
        while True:
            chunk = wav_file.readframes(chunk_frames)
            if not chunk:
                break
            yield chunk

def create_repeating_dropout_pattern(base_pattern: List[Tuple[float, float]], 
                                   repeat_interval: float,
                                   total_duration: float) -> List[Tuple[float, float]]:
    """
    Create a repeating dropout pattern for testing sync drift accumulation.
    
    Args:
        base_pattern: List of (offset, duration) relative to start of each cycle
        repeat_interval: How often to repeat the pattern (seconds)
        total_duration: Total audio duration (seconds)
    
    Returns:
        Complete dropout pattern covering the entire duration
    """
    full_pattern = []
    current_cycle_start = 0.0
    
    while current_cycle_start < total_duration:
        for offset, duration in base_pattern:
            dropout_start = current_cycle_start + offset
            
            # Only add if it's within the total duration
            if dropout_start < total_duration:
                # Clip duration if it extends beyond total duration
                actual_duration = min(duration, total_duration - dropout_start)
                full_pattern.append((dropout_start, actual_duration))
        
        current_cycle_start += repeat_interval
    
    return full_pattern

def create_choppy_audio_generator(audio_path: str, 
                                 dropout_config: dict,
                                 chunk_duration_ms: int = 100):
    """
    Generator with properly repeating dropout patterns.
    """
    # Get audio duration first
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        total_duration = frames / float(rate)
    
    # Generate repeating pattern
    dropout_pattern = create_repeating_dropout_pattern(
        dropout_config['base_pattern'],
        dropout_config['repeat_interval'],
        total_duration
    )
    
    print(f"[Dropout Pattern] Generated {len(dropout_pattern)} dropout events")
    for i, (start, dur) in enumerate(dropout_pattern[:5]):
        print(f"  Dropout {i+1}: {start:.3f}s for {dur*1000:.0f}ms")
    
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        chunk_frames = int(sample_rate * chunk_duration_ms / 1000)
        
        samples_read = 0
        dropout_index = 0
        
        while True:
            current_time = samples_read / float(sample_rate)
            
            # Check all dropouts to find if we're in one
            in_dropout = False
            for idx in range(dropout_index, len(dropout_pattern)):
                start, duration = dropout_pattern[idx]
                if current_time >= start and current_time < start + duration:
                    in_dropout = True
                    if idx > dropout_index:
                        print(f"[Dropout] Entering dropout at {current_time:.3f}s")
                        dropout_index = idx
                    break
                elif current_time >= start + duration and idx == dropout_index:
                    # We've passed this dropout
                    dropout_index += 1
            
            if in_dropout:
                yield None
                # Advance time by chunk duration during dropout
                samples_read += chunk_frames
            else:
                chunk = wav_file.readframes(chunk_frames)
                if not chunk:
                    break
                
                yield chunk
                samples_read += len(chunk) // 2  # 16-bit audio


# Test 1: Small frequent dropouts (simulates jittery network)
test_config_1 = {
    'base_pattern': [
        (0.5, 0.02),   # 20ms dropout at 0.5s
        (1.2, 0.03),   # 30ms dropout at 1.2s  
        (1.8, 0.02),   # 20ms dropout at 1.8s
    ],
    'repeat_interval': 2.0,  # Repeat every 2 seconds
    'enabled': True
}

# Test 2: Larger periodic dropouts (simulates buffering)
test_config_2 = {
    'base_pattern': [
        (0.8, 0.1),    # 100ms dropout at 0.8s
    ],
    'repeat_interval': 1.5,  # Repeat every 1.5 seconds
    'enabled': True
}

# Test 3: Increasing dropout pattern (stress test)
test_config_3 = {
    'base_pattern': [
        (0.3, 0.01),   # 10ms
        (0.6, 0.02),   # 20ms
        (0.9, 0.04),   # 40ms
        (1.2, 0.08),   # 80ms
    ],
    'repeat_interval': 2.0,
    'enabled': True
}


def performance_monitor_thread(api, metrics, writer_stats, stop_event):
    """
    Monitor performance metrics in real-time
    """
    while not stop_event.is_set():
        time.sleep(1.0)  # Report every second
        
        stats = metrics.get_statistics()
        if stats and stats['frames_received'] > 0:
            print(f"\n[Monitor] Performance Stats:")
            print(f"  Frames: Submitted={stats['frames_submitted']}, "
                  f"Received={stats['frames_received']}, "
                  f"Written={stats['frames_written']}")
            print(f"  Latency (ms): mean={stats['latency_ms']['mean']:.1f}, "
                  f"p95={stats['latency_ms']['p95']:.1f}, "
                  f"p99={stats['latency_ms']['p99']:.1f}, "
                  f"max={stats['latency_ms']['max']:.1f}")
            print(f"  Frame timing: {stats['frame_timing_ms']['mean']:.1f}ms ± "
                  f"{stats['frame_timing_ms']['std']:.1f}ms")
            print(f"  Effective FPS: {stats['effective_fps']:.1f}")
            print(f"  Late frames: {stats['late_frame_percentage']:.1f}%")
            
            # Get pipeline queue sizes if available
            for model_name, wrapper in api.models.items():
                preproc_size = wrapper.preprocess_queue.qsize()
                gpu_size = wrapper.gpu_queue.qsize()
                postproc_size = wrapper.postprocess_queue.qsize()
                output_buffer_size = len(wrapper.output_buffer)
                print(f"  Pipeline queues: Pre={preproc_size}, GPU={gpu_size}, "
                      f"Post={postproc_size}, Output={output_buffer_size}")
            
            # System resources
            cpu_percent = psutil.cpu_percent(interval=0)
            memory = psutil.virtual_memory()
            print(f"  System: CPU={cpu_percent:.1f}%, Memory={memory.percent:.1f}%")

def video_writer_thread(write_queue, temp_path, fps, frame_shape, stats_dict, metrics):
    """
    Separate thread for writing video frames to disk.
    Also collects frame audio and saves it as a WAV file.
    
    Args:
        write_queue: Queue containing (frame, index, audio) tuples
        temp_path: Output video path
        fps: Video framerate
        frame_shape: (height, width) for video initialization
        stats_dict: Shared dict for statistics
        metrics: StreamingMetrics instance
    """
    writer = None
    frames_written = 0
    frame_indices = []
    frame_index_counts = Counter()
    
    # Audio collection
    audio_chunks = []
    audio_sample_rate = 16000
    
    try:
        while True:
            try:
                # Get frame with timeout
                item = write_queue.get(timeout=1.0)
                
                if item is None:  # Sentinel value to stop
                    break
                
                frame, index, audio_chunk = item
                
                # Initialize writer on first frame
                if writer is None:
                    h, w = frame_shape if frame_shape else frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(temp_path, fourcc, fps, (w, h))
                    print(f"[Writer Thread] Initialized video writer: {w}x{h} @ {fps}fps")
                
                # Write frame
                writer.write(frame)
                frames_written += 1
                
                # Collect audio chunk
                if audio_chunk is not None:
                    audio_chunks.append((index, audio_chunk))
                
                # Track metrics
                metrics.write_frame(index)
                
                # Track statistics
                frame_indices.append(index)
                frame_index_counts[index] += 1
                
                # Update shared stats
                stats_dict['frames_written'] = frames_written
                stats_dict['queue_size'] = write_queue.qsize()
                
                # Progress update
                if frames_written % 50 == 0:
                    print(f"[Writer Thread] Written {frames_written} frames, queue size: {write_queue.qsize()}")
                    
            except queue.Empty:
                continue
                
    except Exception as e:
        print(f"[Writer Thread] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if writer:
            writer.release()
            print(f"[Writer Thread] Released video writer. Total frames written: {frames_written}")
        
        # Save reconstructed audio as WAV file
        if audio_chunks:
            # Sort by frame index to ensure correct order
            audio_chunks.sort(key=lambda x: x[0])
            
            # Concatenate all audio chunks
            all_audio = np.concatenate([chunk[1] for chunk in audio_chunks])
            
            # Convert float32 [-1, 1] to int16 for WAV file
            audio_int16 = (all_audio * 32767).astype(np.int16)
            
            # Save as WAV file
            reconstructed_audio_path = temp_path.replace('_temp.mp4', '_reconstructed.wav')
            with wave.open(reconstructed_audio_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(audio_sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            print(f"[Writer Thread] Saved reconstructed audio: {reconstructed_audio_path}")
            print(f"  Total audio samples: {len(all_audio)}")
            print(f"  Duration: {len(all_audio)/audio_sample_rate:.2f}s")
            print(f"  From {len(audio_chunks)} frames")
            
            stats_dict['reconstructed_audio_path'] = reconstructed_audio_path
            stats_dict['audio_duration'] = len(all_audio) / audio_sample_rate
        
        # Store final statistics
        stats_dict['final_frames_written'] = frames_written
        stats_dict['frame_indices'] = frame_indices
        stats_dict['frame_index_counts'] = dict(frame_index_counts)

def test_streaming_simple():
    """Test basic streaming without real-time constraints"""
    
    print("=== Simple Streaming Test ===\n")
    
    # Create profiler
    profiler = PerformanceProfiler("SimpleStreaming")
    
    # Initialize API
    api = InferenceAPI(dataset_base_path="./dataset")
    
    # Define a crop region (x1, y1, x2, y2)
    crop_bbox = (596, 10, 1196, 610)
    frame_range = (200, 900)
    # crop_bbox = (0, 0, 1919, 1079)

    # Load models
    print("Loading model...")
    api.preload_models({
        # "Kate-cc": {
        #     "checkpoint": "./checkpoint/Kate-cc/100.pth",
        #     "mode": "ave",
        #     "batch_size": 4  # Smaller batch for lower latency
        # }
        "alin": {
            "checkpoint": "./checkpoint/alin/100.pth",
            "mode": "ave",
            "batch_size": 4,
            "crop_bbox": crop_bbox,
            "frame_range": frame_range,
            "resize_dims": (350, 350),
            "video_path": "dataset/alin/alin.mp4" 
        }
    })
    
    # Pre-load video frames
    # print("Pre-loading video frames...")
    # frame_manager = AllFramesMemory("dataset/alin/alin.mp4", from_images=False, crop_bbox=crop_bbox, frame_range=(100, 900))
    # frame_manager.initialize()
    
    # landmark_dir = os.path.join("dataset/Kate-cc/", "landmarks/")
    # landmark_manager = LandmarkManager(
    #     landmark_dir, 
    #     enable_async=False, 
    #     crop_bbox=crop_bbox,
    #     frame_range=(100, 900)  
    # )
    # landmark_manager.initialize()

    # Test parameters
    audio_path = "audios/TIMIT_Ferrah_Normal_en-us_30s.wav"
    output_path = "result/alin_realtime_streaming_test.mp4"
    os.makedirs("result", exist_ok=True)
    
    # Get audio duration for FPS calculation
    audio_duration = get_audio_duration(audio_path)
    print(f"Audio duration: {audio_duration:.2f}s")
    expected_frames = int(audio_duration * 25)  # 25 FPS
    print(f"Expected frames at 25 FPS: {expected_frames}")
    
    # Initialize metrics
    metrics = StreamingMetrics(target_fps=60.0)
    
    # Setup video writer thread
    temp_path = output_path.replace('.mp4', '_temp.mp4')
    write_queue = queue.Queue(maxsize=100)  # Buffer up to 100 frames
    writer_stats = {}  # Shared statistics
    
    # Start writer thread
    writer_thread = threading.Thread(
        target=video_writer_thread,
        args=(write_queue, temp_path, 25, None, writer_stats, metrics),
        name="VideoWriter"
    )
    writer_thread.start()
    
    # Start performance monitor
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=performance_monitor_thread,
        args=(api, metrics, writer_stats, stop_monitor),
        name="PerformanceMonitor"
    )
    # monitor_thread.start()
    
    # Frame generation tracking
    frames_generated = 0
    first_frame_time = None
    start_time = time.time()
    max_queue_size = 0
    queue_full_events = 0
    
    try:
        print("\nStarting streaming test...")
        
        # audio_chunks = create_simple_audio_chunk_generator(audio_path, chunk_duration_ms=500)
        
        audio_chunks = create_audio_with_silences_chunk_generator(
            audio_path, 
            chunk_duration_ms=500,
            tts_duration_seconds=10.0,
            pause_duration_seconds=10.0
        )

        # Create audio chunk generator
        # audio_chunks = create_simple_audio_chunk_generator(
        #     audio_path, 
        #     chunk_duration_ms=500,
        #     tts_duration_seconds=10.0,
        #     pause_duration_seconds=5.0,
        #     simulate_tts=True,  # Enable TTS simulation
        #     simulate_abrupt_stops=True
        # )

        # audio_chunks = create_elevenlabs_tts_generator()

        # audio_chunks = create_choppy_audio_generator(
        #     audio_path,
        #     dropout_config=test_config_1,  # Example dropouts
        #     chunk_duration_ms=500
        # )

        # Process video frames
        with profiler.timer("Total Streaming"):
            for frame, img_idx, current, frame_audio in api.generate_video_stream(
                model_name="alin",
                audio_chunks=audio_chunks,
                is_silent=False,
                last_index=0,
                profiler=profiler,
                continue_silent=True,
                silent_duration_seconds=10,
                frame_generation_rate=25,
                buffer_duration_seconds=1.0,   # 2 second buffer for smoothness
            ):
                # Track frame submission (this is when we start processing)
                if current not in metrics.frame_metrics:
                    metrics.submit_frame(current)
                
                # Track frame reception
                metrics.receive_frame(current)
                
                if first_frame_time is None:
                    first_frame_time = time.time() - start_time
                    print(f"First frame latency: {first_frame_time:.2f}s")
                
                # Try to put frame in queue
                try:
                    write_queue.put((frame, current, frame_audio), block=True, timeout=0.1)
                    frames_generated += 1
                    
                    # Track max queue size
                    current_queue_size = write_queue.qsize()
                    max_queue_size = max(max_queue_size, current_queue_size)
                    
                except queue.Full:
                    # Queue is full - writer can't keep up
                    queue_full_events += 1
                    metrics.dropped_frames += 1
                    print(f"[WARNING] Write queue full! Dropping frame {current}. "
                          f"This is event #{queue_full_events}")
                    continue
                
                # Progress update
                if frames_generated % 25 == 0:
                    elapsed = time.time() - start_time
                    gen_fps = frames_generated / elapsed
                    queue_size = write_queue.qsize()
                    print(f"[Received Frames in Generator] Frame generated {frames_generated}: Gen FPS: {gen_fps:.1f}, "
                          f"Queue: {queue_size}, Time: {elapsed:.1f}s")
        
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Signal writer and monitor threads to stop
        print("\nSignaling threads to stop...")
        write_queue.put(None)
        stop_monitor.set()
        
        # Wait for threads to finish
        writer_thread.join(timeout=30.0)
        if writer_thread.is_alive():
            print("[WARNING] Writer thread did not finish in time!")
        
        # monitor_thread.join(timeout=2.0)
        
        # Get final timing stats
        total_time = time.time() - start_time
        final_stats = metrics.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"STREAMING RESULTS")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Frames generated: {frames_generated}")
        print(f"Frames written: {writer_stats.get('final_frames_written', 0)}")
        print(f"Max queue size: {max_queue_size}")
        print(f"Queue full events: {queue_full_events}")
        
        # Get frame statistics from writer thread
        frame_indices = writer_stats.get('frame_indices', [])
        frame_index_counts = Counter(writer_stats.get('frame_index_counts', {}))
        unique_frame_count = len(set(frame_indices))
        
        print(f"Unique frame indices: {unique_frame_count}")
        
        # Latency statistics
        if final_stats:
            print(f"\nLatency Statistics:")
            lat = final_stats['latency_ms']
            print(f"  Mean: {lat['mean']:.2f}ms")
            print(f"  P50 (median): {lat['p50']:.2f}ms")
            print(f"  P90: {lat['p90']:.2f}ms")
            print(f"  P95: {lat['p95']:.2f}ms")
            print(f"  P99: {lat['p99']:.2f}ms")
            print(f"  Max: {lat['max']:.2f}ms")
            print(f"  Min: {lat['min']:.2f}ms")
            
            print(f"\nFrame Timing:")
            timing = final_stats['frame_timing_ms']
            print(f"  Mean interval: {timing['mean']:.2f}ms ± {timing['std']:.2f}ms")
            print(f"  Max interval: {timing['max']:.2f}ms")
            print(f"  Effective FPS: {final_stats['effective_fps']:.1f}")
            print(f"  Late frames (>16.7ms): {final_stats['late_frame_percentage']:.1f}%")
        
        # FPS calculations
        if total_time > 0:
            # Generation FPS
            gen_fps = frames_generated / total_time
            print(f"\nGeneration FPS: {gen_fps:.1f}")
            
            # Writing FPS
            frames_written = writer_stats.get('final_frames_written', 0)
            if frames_written > 0:
                write_fps = frames_written / total_time
                print(f"Writing FPS: {write_fps:.1f}")
            
            # Content FPS (unique frames)
            content_fps = unique_frame_count / total_time
            print(f"Content FPS (unique frames): {content_fps:.1f}")
            
            # Performance metrics
            print(f"Target FPS: 25.0")
            print(f"Performance ratio: {content_fps/25.0:.2f}x real-time")
            
            if first_frame_time:
                print(f"First frame latency: {first_frame_time:.2f}s")
        
        # Frame completeness
        print(f"\nExpected frames for {audio_duration:.1f}s audio: {expected_frames}")
        print(f"Frame completeness: {unique_frame_count}/{expected_frames} "
              f"({100*unique_frame_count/expected_frames:.1f}%)")
        
        # Duplicate analysis
        duplicate_frames = [(idx, count) for idx, count in frame_index_counts.items() if count > 1]
        if duplicate_frames:
            print(f"\nWARNING: Found {len(duplicate_frames)} duplicate frame indices!")
            print("First 10 duplicates:")
            for idx, count in duplicate_frames[:10]:
                print(f"  Frame {idx}: appears {count} times")
        else:
            print("\n✓ No duplicate frames detected!")
        
        # Check for gaps in sequence
        if frame_indices:
            sorted_unique = sorted(set(frame_indices))
            gaps = []
            for i in range(1, len(sorted_unique)):
                if sorted_unique[i] != sorted_unique[i-1] + 1:
                    gap_start = sorted_unique[i-1] + 1
                    gap_end = sorted_unique[i] - 1
                    gaps.append((gap_start, gap_end, gap_end - gap_start + 1))
            
            if gaps:
                print(f"\nFound {len(gaps)} gaps in frame sequence!")
                print("First 10 gaps:")
                for start, end, size in gaps[:10]:
                    print(f"  Missing frames {start} to {end} ({size} frames)")
            else:
                print("✓ Frame sequence is continuous!")
        
        # Target achievement summary
        print(f"\n{'='*60}")
        print("TARGET ACHIEVEMENT SUMMARY")
        print(f"{'='*60}")
        if final_stats:
            achieved_60fps = final_stats['effective_fps'] >= 60
            low_latency = lat['p95'] <= 16.7  # One frame at 60 FPS
            
            print(f"60+ FPS Target: {'✓ ACHIEVED' if achieved_60fps else '✗ NOT MET'} "
                  f"({final_stats['effective_fps']:.1f} FPS)")
            print(f"Low Latency Target: {'✓ ACHIEVED' if low_latency else '✗ NOT MET'} "
                  f"(P95: {lat['p95']:.1f}ms, target: ≤16.7ms)")
            
            if achieved_60fps and low_latency:
                print("\n✓ SUCCESS: System meets real-time streaming requirements!")
            else:
                print("\n✗ System needs optimization to meet targets.")
        
        # Add audio to video
        if os.path.exists(temp_path) and writer_stats.get('final_frames_written', 0) > 0:
            print("\nAdding audio track...")
            if os.path.exists(temp_path) and writer_stats.get('final_frames_written', 0) > 0:
                reconstructed_audio_path = writer_stats.get('reconstructed_audio_path')
                
                if reconstructed_audio_path and os.path.exists(reconstructed_audio_path):
                    print("\nMerging video with reconstructed audio...")
                    
                    # First, let's compare the audio files
                    original_duration = get_audio_duration(audio_path)
                    reconstructed_duration = writer_stats.get('audio_duration', 0)
                    print(f"Audio comparison:")
                    print(f"  Original audio: {original_duration:.2f}s")
                    print(f"  Reconstructed audio: {reconstructed_duration:.2f}s")
                    print(f"  Difference: {abs(original_duration - reconstructed_duration):.3f}s")
                    
                    with profiler.timer("FFmpeg Audio Merge (Reconstructed)"):
                        ffmpeg_cmd = [
                            'ffmpeg', '-y',
                            '-i', temp_path,
                            '-i', reconstructed_audio_path,  # Use reconstructed audio
                            '-c:v', 'libx264',
                            '-c:a', 'aac',
                            '-strict', 'experimental',
                            '-shortest',  # Stop when shortest stream ends
                            output_path
                        ]
                        
                        result = subprocess.run(ffmpeg_cmd, capture_output=True)
                        if result.returncode != 0:
                            print(f"FFmpeg error: {result.stderr.decode()}")
                        else:
                            print(f"Output saved to: {output_path}")
                            
                            # Clean up temporary files
                            os.remove(temp_path)
                            os.remove(reconstructed_audio_path)
                            
                            # Also create a version with original audio for comparison
                            comparison_path = output_path.replace('.mp4', '_original_audio.mp4')
                            ffmpeg_cmd_original = [
                                'ffmpeg', '-y',
                                '-i', temp_path,
                                '-i', audio_path,  # Original audio
                                '-c:v', 'libx264',
                                '-c:a', 'aac',
                                '-strict', 'experimental',
                                '-shortest',
                                comparison_path
                            ]
                            subprocess.run(ffmpeg_cmd_original, capture_output=True, check=False)
                            print(f"Comparison video with original audio: {comparison_path}")
                else:
                    print("[WARNING] No reconstructed audio available, falling back to original audio")
                    with profiler.timer("FFmpeg Audio Merge"):
                        ffmpeg_cmd = [
                            'ffmpeg', '-y',
                            '-i', temp_path,
                            '-i', audio_path,
                            '-c:v', 'libx264',
                            '-c:a', 'aac',
                            '-strict', 'experimental',
                            '-shortest',  # Stop when shortest stream ends
                            output_path
                        ]
                        
                        result = subprocess.run(ffmpeg_cmd, capture_output=True)
                        if result.returncode != 0:
                            print(f"FFmpeg error: {result.stderr.decode()}")
                        else:
                            print(f"Output saved to: {output_path}")
                            os.remove(temp_path)
        
        # Save detailed metrics
        metrics_path = output_path.replace('.mp4', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'summary': {
                    'total_time': total_time,
                    'frames_generated': frames_generated,
                    'frames_written': writer_stats.get('final_frames_written', 0),
                    'unique_frames': unique_frame_count,
                    'expected_frames': expected_frames,
                    'first_frame_latency': first_frame_time
                },
                'performance': final_stats,
                'profiler_summary': profiler.get_summary()
            }, f, indent=2)
        print(f"Detailed metrics saved to: {metrics_path}")
        
        # Print profiler summary
        print("\n" + "="*60)
        print("PROFILER SUMMARY")
        print("="*60)
        profiler.print_summary(detailed=True)

if __name__ == "__main__":
    test_streaming_simple()