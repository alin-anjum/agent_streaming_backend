# inference_system/api.py
"""
Streaming Inference API for SyncTalk 2D

This module provides a high-performance streaming interface for generating lip-synced video frames
in real-time from audio input. It implements a sophisticated dual-thread architecture with dynamic
buffer management to achieve low latency while maintaining smooth playback.

Key Features:
- Real-time audio-to-video generation with lip sync
- Dynamic buffer management that adapts to audio availability
- Low-latency mode switching between active speech and silence
- Frame-accurate synchronization between audio and video
- Thread-safe processing with separate generation and yielding threads

Architecture Overview:
1. Audio Input Thread: Reads audio chunks from TTS/audio source
2. Processing Thread: Generates video frames based on audio input
3. Yielding Thread: Delivers frames at exact 25 FPS for smooth playback

The system uses two different buffer sizes:
- During active speech: Larger buffer (2.0s default) for resilience
- During silence: Smaller buffer (0.5s default) for low latency when speech resumes


DATA FLOW DIAGRAM:
==================

┌─────────────────┐
│   TTS/Audio     │
│    Generator    │  ← Produces audio chunks (bytes) or None (silence)
└────────┬────────┘
         │ audio_chunks
         ↓
┌─────────────────┐
│ Audio Iterator  │  ← Wrapped in iterator for processing
└────────┬────────┘
         │
         ↓
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          PROCESSING THREAD                                    ║
║                                                                               ║
║  ┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐      ║
║  │ _process_audio_ │     │ StreamingAud     │     │ Audio Feature      │      ║
║  │     input()     │ ──→ │   Dataset        │ ──→ │ Windows (16-frame) │      ║
║  └─────────────────┘     └──────────────────┘     └────────┬───────────┘      ║
║         ↓                                                  │                  ║
║  ┌─────────────────┐                                       │                  ║
║  │ Silence/Speech  │                                       │                  ║
║  │   Detection     │ ← Switches buffer modes               │                  ║
║  └─────────────────┘                                       │                  ║
║         ↓                                                  ↓                  ║
║  ┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐      ║
║  │ Buffer Manager  │     │ Frame & Landmark │     │   GPU Pipeline     │      ║
║  │ (Burst Refills) │ ──→ │    Retrieval     │ ──→ │  (Batch=4)         │      ║
║  └─────────────────┘     └──────────────────┘     └────────┬───────────┘      ║
║                                                            │                  ║
║                          ┌──────────────────┐              │                  ║
║                          │  Output Queue    │ ←────────────┘                  ║
║                          │  (Thread-safe)   │                                 ║
║                          └────────┬─────────┘                                 ║
╚══════════════════════════╪════════════════════════════════════════════════════╝
                           │ frames @ variable rate
                           ↓
╔══════════════════════════╪═══════════════════════════════════════════════════╗
║                          │        YIELDING THREAD                            ║
║                          ↓                                                   ║
║  ┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐     ║
║  │ Frame-accurate  │     │ Latency Tracking │     │   25 FPS Output    │     ║
║  │ Timing Control  │ ──→ │  & Measurement   │ ──→ │   Generator        │     ║
║  └─────────────────┘     └──────────────────┘     └────────┬───────────┘     ║
╚══════════════════════════════════════════════════════════════╪═══════════════╝
                                                               │
                                                               ↓
                                                    ┌─────────────────┐
                                                    │ Client/Consumer │
                                                    │ (video, audio)  │
                                                    └─────────────────┘

BUFFER MANAGEMENT STATES:
========================

1. BUILDING PHASE (Startup)
   ┌──────┐ Fill →  ┌──────────────────────┐
   │Empty │         │████████████░░░░░░░░░░│ Target: 2.0s
   └──────┘         └──────────────────────┘

2. ACTIVE SPEECH (Large Buffer - 2.0s)
   ┌──────────────────────┐     Refill at 50% →  ┌──────────────────────┐
   │████████░░░░░░░░░░░░░░│                      │████████████████░░░░░░│
   └──────────────────────┘                      └──────────────────────┘

3. TEMPORARY SILENCE (Small Buffer - 0.5s)
   ┌─────┐     Refill at 50% →  ┌─────┐
   │██░░░│                      │████░│
   └─────┘                      └─────┘

FRAME TIMING:
=============
- Input: Variable rate (depends on audio availability and GPU speed)
- Output: Constant 25 FPS (40ms per frame)
- Latency: ~0.5-0.7s during silence → speech transition   
"""

import os
import sys
import numpy as np
from typing import Dict, Generator, Optional, Tuple
import uuid
import time
from collections import deque
from typing import Dict, Generator, Optional, Tuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference_system.core.video_frame_manager import AllFramesMemory
from inference_system.core.landmark_manager import LandmarkManager
from inference_system.streaming.stream_wrapper import StreamingInferenceWrapper
from inference_system.streaming.per_stream_workers import PerStreamWorkersWrapper
from inference_system.streaming.silence_config import SilenceOptimizationConfig
from inference_system.inference_video import FrameData
from inference_system.utils.profiler import PerformanceProfiler
import queue
import threading

class InferenceAPI:
    """
    Main API class that manages multiple models and streaming sessions.
    
    This class provides:
    - Model preloading and warmup for optimal performance
    - Concurrent streaming sessions with different models
    - Resource management and cleanup
    """
    
    def __init__(self, dataset_base_path: str = "./dataset"):
        self.dataset_base_path = dataset_base_path
        self.models = {}  # model_name -> StreamingInferenceWrapper
        self.active_streams = {}  # stream_id -> stream_info
        self.silent_frame_cache = {} # Cache for silent frames (model_name, img_idx) -> (frame, audio)
        
    def preload_models(self, model_configs: Dict[str, Dict[str, str]]):
        """
        Preload models for inference.
        
        This method loads model checkpoints and initializes all necessary components
        for inference. Models are kept in memory for fast streaming startup.
        
        Args:
            model_configs: Dictionary mapping model names to their configurations
                          Each config can contain: checkpoint, mode, batch_size, 
                          crop_bbox, frame_range, video_path, resize_dims
        """
        for name, config in model_configs.items():
            dataset_dir = os.path.join(self.dataset_base_path, name)
            
            # Auto-detect latest checkpoint if not specified
            if 'checkpoint' not in config:
                checkpoint_path = os.path.join("./checkpoint", name)
                checkpoint_files = [f for f in os.listdir(checkpoint_path) 
                                   if f.endswith('.pth') and f.split('.')[0].isdigit()]
                checkpoint = os.path.join(checkpoint_path, 
                                        sorted(checkpoint_files, key=lambda x: int(x.split(".")[0]))[-1])
                print(f"Using checkpoint: {checkpoint}")
                config['checkpoint'] = checkpoint
            
            # Create base wrapper with specified configuration
            base_wrapper = StreamingInferenceWrapper(
                model_name=name,
                checkpoint_path=config['checkpoint'],
                dataset_dir=dataset_dir,
                mode=config.get('mode', 'ave'),
                batch_size=config.get('batch_size', 4),
                crop_bbox=config.get('crop_bbox', None),
                frame_range=config.get('frame_range', None),
                video_path=config.get('video_path', None),
                resize_dims=config.get('resize_dims', None),
                use_old_unet=config.get('use_old_unet', False),
                start_workers_on_init=False  # We will use PerStreamWorkersWrapper to manage workers
            )

            # Wrap with per-stream worker wrapper for true isolation
            wrapper = PerStreamWorkersWrapper(base_wrapper)

            self.models[name] = wrapper
            print(f"Loaded model: {name} (per-stream worker architecture)")
    
    def warmup_model(self, model_name: str, warmup_batches: int = 50):
        """
        Warmup a model by running dummy batches through the entire pipeline.
        
        GPU inference typically has high latency for the first few batches due to:
        - CUDA kernel compilation and caching
        - GPU memory allocation
        - Framework optimization
        
        This method runs dummy data through the model to eliminate these startup costs.
        
        Args:
            model_name: Name of the model to warmup
            warmup_batches: Number of batches to process (default: 50)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        wrapper = self.models[model_name]
        device = wrapper.device
        batch_size = wrapper.batch_size
        
        print(f"Warming up model '{model_name}' with {warmup_batches} batches...")
        start_time = time.time()
        
        # Clear any existing timing statistics
        if hasattr(StreamingInferenceWrapper, '_gpu_stats'):
            with StreamingInferenceWrapper._gpu_stats['lock']:
                StreamingInferenceWrapper._gpu_stats['inference_times'].clear()
        
        # Create realistic dummy data
        dummy_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        dummy_landmarks = np.random.rand(68, 2).astype(np.float32) * 200 + 400
        dummy_bbox = (400, 300, 700, 600)  # Typical face bounding box
        dummy_audio = np.random.randn(16, 512).astype(np.float32)
        
        # Submit warmup frames
        print("Submitting warmup frames...")
        # For per-stream, we need a dummy context to submit
        dummy_stream_id = "warmup_stream"
        context = wrapper.create_stream_context(dummy_stream_id)
        
        for i in range(warmup_batches * wrapper.batch_size):
            frame_data = FrameData(
                frame_idx=i,
                img_idx=i,
                img=dummy_img.copy(),
                landmarks=dummy_landmarks.copy(),
                bbox=dummy_bbox,
                audio_feat=dummy_audio.copy()
            )
            # Add stream_id for routing
            frame_data.stream_id = dummy_stream_id
            context.preprocess_queue.put(frame_data)
        
        # Wait for all warmup frames to complete
        print("Processing warmup frames...")
        warmup_completed = 0
        timeout = 60.0
        start_wait = time.time()
        
        # In per-stream, we check the context's output buffer
        output_buffer = context.output_buffer
        output_lock = context.output_lock
        next_output_idx = 0
        
        while warmup_completed < warmup_batches * wrapper.batch_size:
            # Collect completed frames from the context's buffer
            with output_lock:
                while next_output_idx in output_buffer:
                    output_buffer.pop(next_output_idx)
                    next_output_idx += 1
                    warmup_completed += 1
            
            if warmup_completed % 100 == 0 and warmup_completed > 0:
                print(f"  Processed {warmup_completed}/{warmup_batches * wrapper.batch_size} warmup frames")
            
            if time.time() - start_wait > timeout:
                print(f"Warning: Warmup timeout after {warmup_completed} frames")
                break
            
            time.sleep(0.001)

        # Clean up dummy context
        wrapper.remove_stream_context(dummy_stream_id)
        
        elapsed = time.time() - start_time
        print(f"Model '{model_name}' warmup complete in {elapsed:.2f}s")
        
        # Report final performance metrics
        if hasattr(StreamingInferenceWrapper, '_gpu_stats'):
            with StreamingInferenceWrapper._gpu_stats['lock']:
                times = StreamingInferenceWrapper._gpu_stats['inference_times']
                if times:
                    recent_times = times[-10:]
                    avg_time = np.mean(recent_times) * 1000
                    print(f"Post-warmup GPU performance: {avg_time:.1f}ms/batch")

    def stop_stream(self, model_name: str, stream_id: Optional[str] = None):
        """
        Stop streaming sessions for a model (circuit breaker).
        
        This method can be called to immediately halt streaming, useful for:
        - User-initiated stops
        - Error conditions
        - Resource cleanup
        
        Args:
            model_name: Name of the model
            stream_id: Optional specific stream ID to stop. If None, affects all streams for this model.
        """
        if model_name in self.models:
            wrapper = self.models[model_name]
            
            if stream_id:
                # Stop specific stream
                if hasattr(wrapper, 'remove_stream_context'):
                    try:
                        wrapper.remove_stream_context(stream_id)
                        print(f"Removed stream context for {stream_id} (model {model_name})")
                    except Exception as e:
                        print(f"Error removing stream context: {e}")

                # Also handle legacy streaming_dataset for circuit breaker
                if hasattr(wrapper, 'streaming_datasets') and stream_id in wrapper.streaming_datasets:
                    wrapper.streaming_datasets[stream_id].circuit_breaker = True
                    del wrapper.streaming_datasets[stream_id]
                    print(f"Circuit breaker activated for stream {stream_id}")
            else:
                # Stop all streams for this model
                if hasattr(wrapper, 'stream_workers'):
                    stream_ids = list(wrapper.stream_workers.keys())
                    for sid in stream_ids:
                        try:
                            wrapper.remove_stream_context(sid)
                        except Exception as e:
                            print(f"Error removing stream context {sid}: {e}")
                    print(f"Removed all stream contexts for model {model_name} ({len(stream_ids)} streams)")
                
                # Also handle legacy streaming_dataset for circuit breaker
                if hasattr(wrapper, 'streaming_datasets'):
                    for dataset in wrapper.streaming_datasets.values():
                        dataset.circuit_breaker = True
                    wrapper.streaming_datasets.clear()
                    print(f"Circuit breaker activated for all streams of model {model_name}")
                
                # Legacy single stream support
                if hasattr(wrapper, 'streaming_dataset') and wrapper.streaming_dataset:
                    wrapper.streaming_dataset.circuit_breaker = True
                    wrapper.streaming_dataset = None
                    
        else:
            print(f"Model {model_name} not found")
    


    def _initialize_stream_resources(self, model_name: str, video_path: Optional[str], 
                                frame_source: Optional[AllFramesMemory], 
                                landmark_source: Optional[LandmarkManager],
                                last_index: int, profiler: PerformanceProfiler, 
                                stream_id: str) -> tuple:
        """
        Initialize all resources needed for a streaming session.
        
        This method sets up:
        - Frame manager: Loads and manages video frames
        - Landmark manager: Provides facial landmarks for each frame
        - Streaming dataset: Processes audio and maintains sync
        
        The method supports multiple frame sources in priority order:
        1. Provided frame_source parameter
        2. Pre-loaded frames in the model wrapper
        3. Video file specified by video_path
        4. Image sequence in the dataset directory
        """
        wrapper = self.models[model_name]
        
        # Initialize frame manager with profiling
        with profiler.timer("Initialize Frame Manager"):
            dataset_dir = os.path.join(self.dataset_base_path, model_name)
            
            # Determine frame source
            if frame_source is not None:
                frame_manager = frame_source
                created_frame_manager = False
                print(f"Using provided frame manager")
            elif hasattr(wrapper, 'frame_manager') and wrapper.frame_manager is not None:
                frame_manager = wrapper.frame_manager
                created_frame_manager = False
                print(f"Using pre-loaded frame manager: {frame_manager.frames_loaded_count} frames available")
            elif video_path and os.path.exists(video_path):
                print(f"Loading frames from video: {video_path}")
                frame_manager = AllFramesMemory(
                    video_path, 
                    from_images=False,
                    crop_bbox=wrapper.crop_bbox,
                    frame_range=wrapper.frame_range,
                    resize_dims=wrapper.resize_dims
                )
                frame_manager.initialize()
                created_frame_manager = True
            else:
                # Fallback to image sequence
                img_dir = os.path.join(dataset_dir, "full_body_img/")
                if os.path.exists(img_dir):
                    print(f"Loading frames from images: {img_dir}")
                    frame_manager = AllFramesMemory(
                        img_dir, 
                        from_images=True,
                        crop_bbox=wrapper.crop_bbox,
                        frame_range=wrapper.frame_range,
                        resize_dims=wrapper.resize_dims
                    )
                    frame_manager.initialize()
                    created_frame_manager = True
                else:
                    raise ValueError(
                        f"No frames available for model {model_name}. "
                        f"Provide either:\n"
                        f"  1. 'video_path' in model config\n"
                        f"  2. frame_source parameter\n"
                        f"  3. video_path parameter\n"
                        f"  4. Image sequence in {img_dir}"
                    )
            
            # Handle landmark source
            if landmark_source is not None:
                landmark_manager = landmark_source
                print(f"Using provided landmark manager")
            else:
                landmark_manager = wrapper.landmark_manager
                print(f"Using preloaded landmarks: {len(landmark_manager.landmarks)} available")
        
        # Configure frame manager for streaming
        frame_manager.current_idx = last_index
        frame_manager.direction = 1  # Forward playback
        
        # Create streaming audio processor
        from inference_system.streaming.streaming_audio_dataset import StreamingAudDataset
        streaming_dataset = StreamingAudDataset(
            audio_encoder=wrapper.audio_encoder,
            profiler=profiler
        )
        
        print(f"\n[Stream {stream_id} - Inference Configuration]")
        print(f"  Model: {model_name}")
        print(f"  Frame shape: {frame_manager.video_shape}")
        print(f"  Total frames: {frame_manager.total_frames}")
        print(f"  Landmarks: {len(landmark_manager.landmarks)}")
        print(f"  Memory efficiency: {'✅ REUSING preloaded resources' if not created_frame_manager else '⚠️ Created new frame manager'}")
        
        # Create stream context for multi-stream support
        try:
            stream_context = wrapper.create_stream_context(stream_id, last_index)
            print(f"✅ Stream context created for {stream_id} at frame {last_index}")
        except Exception as e:
            print(f"❌ Error creating stream context: {e}")
            raise
        
        # Store reference for circuit breaker functionality
        # For concurrent streams, each stream manages its own dataset
        if not hasattr(wrapper, 'streaming_datasets'):
            wrapper.streaming_datasets = {}
        wrapper.streaming_datasets[stream_id] = streaming_dataset
        wrapper.streaming_dataset = streaming_dataset  # Keep for backwards compatibility
        
        return frame_manager, landmark_manager, streaming_dataset, created_frame_manager


    def _submit_frames_for_processing(self, frame_indices: list, stream_id: str,
                                     streaming_dataset, wrapper,
                                     submitted_frames: set, state: dict) -> int:
        """
        Submit a batch of frames to the GPU processing pipeline with stream awareness.
        
        This method:
        1. Uses the per-stream worker architecture for frame submission
        2. Maintains per-stream frame indexing for proper lip sync
        3. Associates audio features with correct stream frames
        4. Handles frame sequence isolation between streams
        
        The method handles errors gracefully and continues processing
        even if individual frames fail.
        
        Returns:
            Number of frames successfully submitted
        """
        if not frame_indices:
            return 0
        
        frames_to_process = sorted([f for f in frame_indices if f not in submitted_frames])
        
        if not frames_to_process:
            return 0
        
        context = wrapper.stream_workers.get(stream_id)
        if not context:
            print(f"❌ Stream {stream_id} context not found during frame submission.")
            return 0

        frames_submitted_from_cache = 0
        frames_submitted_to_gpu = 0
        
        non_cached_frames = []

        # First pass: handle cached frames
        for frame_idx in frames_to_process:
            if streaming_dataset.is_silent_frame(frame_idx):
                absolute_frame_idx = context.current_frame_idx + frame_idx
                frame_data_item = wrapper.frame_manager.get_frame_pingpong(absolute_frame_idx)
                if frame_data_item:
                    img_idx = frame_data_item.frame_idx
                    cache_key = (wrapper.model_name, img_idx)

                    if cache_key in self.silent_frame_cache:
                        cached_frame, cached_audio = self.silent_frame_cache[cache_key]
                        output_data = (cached_frame, img_idx, frame_idx, cached_audio)
                        with context.output_lock:
                            context.output_buffer[frame_idx] = output_data
                        
                        frames_submitted_from_cache += 1
                        submitted_frames.add(frame_idx)
                        continue
            non_cached_frames.append(frame_idx)
            
        # Second pass: submit non-cached frames to GPU
        for frame_idx in non_cached_frames:
            try:
                # Build audio features for this frame (16-frame window)
                audio_feat = streaming_dataset.build_audio_window(frame_idx)
                frame_audio_chunk = streaming_dataset.get_audio_for_frame(frame_idx)

                # Get frame and landmark data
                absolute_frame_idx = context.current_frame_idx + frame_idx
                frame_data_item = wrapper.frame_manager.get_frame_pingpong(absolute_frame_idx)
                if not frame_data_item:
                    print(f"❌ Stream {stream_id}: No frame data for index {absolute_frame_idx}")
                    continue
                
                landmark_data = wrapper.landmark_manager.get_landmark(frame_data_item.frame_idx)
                if not landmark_data:
                    print(f"❌ Stream {stream_id}: No landmark data for frame {frame_data_item.frame_idx}")
                    continue
                
                # Create FrameData with all necessary info
                frame_data = FrameData(
                    frame_idx=frame_idx, # Relative to stream
                    img_idx=frame_data_item.frame_idx, # Absolute in video
                    img=frame_data_item.frame,
                    landmarks=landmark_data.landmarks,
                    bbox=landmark_data.bbox,
                    audio_feat=audio_feat,
                    audio_chunk=frame_audio_chunk
                )
                frame_data.stream_id = stream_id
                
                # Put frame data into the stream's dedicated preprocess queue
                try:
                    context.preprocess_queue.put(frame_data, timeout=0.1)
                    frames_submitted_to_gpu += 1
                    submitted_frames.add(frame_idx)
                except queue.Full:
                    print(f"❌ Stream {stream_id}: Preprocess queue full at frame {frame_idx}")
                    break

            except Exception as e:
                print(f"❌ Stream {stream_id}: Error preparing frame {frame_idx} for submission: {e}")
                continue
        
        total_submitted = frames_submitted_from_cache + frames_submitted_to_gpu
        
        if frames_submitted_from_cache > 0:
            print(f"✅ Stream {stream_id}: Reused {frames_submitted_from_cache} silent frames from cache.")
        
        if frames_submitted_to_gpu > 0 and (frames_submitted_to_gpu <= 2 or frames_submitted_to_gpu % 50 == 0):
            print(f"[DEBUG] Stream {stream_id}: Submitted {frames_submitted_to_gpu} frames to preprocess queue")

        if total_submitted < len(frames_to_process):
             print(f"⚠️ Stream {stream_id}: Submitted {total_submitted}/{len(frames_to_process)} frames (queue might be full)")
        elif total_submitted > 0 and state['chunk_count'] % 5 == 0: # Log less frequently
            print(f"✅ Stream {stream_id}: Submitted batch of {total_submitted} frames")

        return total_submitted

    def _process_audio_input(self, state: dict) -> None:
        """
        Process incoming audio chunks and detect speech/silence transitions.
        
        This method implements the core audio processing logic:
        1. Reads audio chunks from the input iterator
        2. Detects transitions between speech and silence
        3. Manages audio buffering and synchronization
        4. Handles special markers (start_of_stream, end_of_stream)
        
        The method distinguishes between:
        - Real audio data: Added to the streaming dataset for processing
        - Silence (None): Triggers temporary silence mode for low latency
        - Markers: Queued for frame-accurate insertion in the output
        
        Temporary silence mode is crucial for low latency:
        - When silence is detected, we switch to a smaller buffer
        - This allows faster response when speech resumes
        - The generation clock is reset to prevent "catch-up" behavior
        """
        # Define control markers used by the streaming protocol
        START_MARKERS = {b'start_of_stream'}
        END_MARKERS = {b'end_of_stream'}
        ALL_MARKERS = START_MARKERS | END_MARKERS
        
        streaming_dataset = state['streaming_dataset']
        audio_iterator = state['audio_iterator']
        marker_queue = state['marker_queue']
        
        if not state['audio_exhausted']:
            try:
                audio_chunk = next(audio_iterator)
                
                # Handle pre-first-audio state
                # We don't start frame generation until we receive actual audio
                # This prevents generating frames during initial silence
                if not state.get('first_real_audio_received', False):
                    if audio_chunk is None:
                        # No audio yet, just wait
                        return
                    elif audio_chunk not in ALL_MARKERS and len(audio_chunk) > 0:
                        # First real audio received! Start frame generation
                        state['first_real_audio_received'] = True
                        print("[Audio] First real audio chunk received, starting normal processing")
                        
                        # Initialize generation timing
                        state['generation_start_time'] = time.time()
                        state['total_frames_generated'] = 0

                        # Enable buffer management
                        if 'buffer_state' in state:
                            state['buffer_state']['first_audio_received'] = True
                    # Markers are processed even before first audio
                
                # Process markers (can happen at any time)
                if audio_chunk in ALL_MARKERS:
                    if audio_chunk in START_MARKERS:
                        boundary_frame = state['next_frame_for_audio']
                    else:
                        boundary_frame = streaming_dataset.get_max_encodable_frame() + 1
                    
                    marker_queue.append((audio_chunk, boundary_frame))
                    print(f"Queued marker {audio_chunk.decode()} at frame boundary {boundary_frame}")
                    return
                
                # Handle silence detection (None from audio source)
                if audio_chunk is None:
                    if not state['in_temporary_silence']:
                        # Transition to temporary silence mode
                        state['in_temporary_silence'] = True
                        state['temporary_silence_start'] = time.time()
                        print("[Audio] Entering temporary silence (pause in TTS)")
                        
                        # Flush any pending audio to ensure clean transition
                        streaming_dataset.flush()
                        if hasattr(streaming_dataset, '_process_pending_audio'):
                            streaming_dataset._process_pending_audio()
                            
                elif len(audio_chunk) > 0:
                    state['chunk_count'] += 1
                    streaming_dataset.add_audio_chunk(audio_chunk)

                    # Real audio chunk received
                    if state['in_temporary_silence']:
                        # SMOOTH TRANSITION: Wait for a minimum audio buffer before switching
                        # This prevents FPS drops when audio resumes.
                        MIN_AUDIO_BUFFER_SECONDS = 0.25 # 250ms
                        
                        # Calculate how much processable audio we have
                        max_encodable_frame = streaming_dataset.get_max_encodable_frame()
                        last_submitted_frame = max(state['submitted_frames']) if state['submitted_frames'] else -1
                        
                        buffered_frames = max_encodable_frame - last_submitted_frame
                        buffered_seconds = buffered_frames / 25.0

                        if buffered_seconds >= MIN_AUDIO_BUFFER_SECONDS:
                            # Resuming from silence - this is a critical transition
                            silence_duration = time.time() - state['temporary_silence_start']
                            print(f"[Audio] Resuming from temporary silence after {silence_duration:.1f}s "
                                  f"(had {buffered_seconds:.2f}s of audio buffered)")
                            state['in_temporary_silence'] = False
                            state['temporary_silence_start'] = None
                            
                            # CRITICAL: Reset generation clock to prevent catch-up
                            # Without this, the system would try to generate all the frames
                            # it "missed" during silence, causing a burst of activity
                            current_time = time.time()
                            actual_frames = state['total_frames_generated']
                            
                            # Calculate what the start time should be to maintain current pace
                            state['generation_start_time'] = current_time - (actual_frames / state['frame_generation_rate'])
                            print(f"[Clock] Reset generation clock after {silence_duration:.1f}s silence. "
                                  f"Generated {actual_frames} frames so far")
                            
                            # Clear any pending silence
                            state['pending_silence_duration'] = 0.0
                            
                            # Start latency measurement for performance monitoring
                            state['latency_test'] = {
                                'audio_resumed_time': time.time(),
                                'target_frame': None,
                                'measured': False,
                                'buffer_size_at_resume': None
                            }
                            print(f"[LATENCY TEST] Started tracking audio resumption")
                    
                    # Track where new audio frames will appear
                    all_ready_frames = streaming_dataset.get_ready_video_frames()
                    if all_ready_frames:
                        state['next_frame_for_audio'] = max(all_ready_frames) + 1
                    
                    # Set target frame for latency measurement
                    if 'latency_test' in state and state['latency_test']['target_frame'] is None:
                        max_encodable = streaming_dataset.get_max_encodable_frame()
                        state['latency_test']['target_frame'] = max_encodable + 1
                        print(f"[LATENCY TEST] Audio will appear in frame {state['latency_test']['target_frame']}")
                        
            except StopIteration:
                # Audio source exhausted
                state['audio_exhausted'] = True
                print("Audio stream exhausted")
                streaming_dataset.flush()


    def _generate_frames_at_rate(self, state: dict, output_queue: queue.Queue) -> int:
        """
        Generate frames at a constant rate during buffer building phase.
        
        This method is used before buffer management is activated. It ensures
        frames are generated at exactly the target frame rate (25 FPS) by:
        1. Calculating how many frames should exist based on elapsed time
        2. Generating exactly that many frames
        3. Adding silence if necessary to maintain the rate
        
        This approach provides smooth, consistent frame generation and prevents
        the system from falling behind the target frame rate.
        
        Args:
            state: Current processing state
            output_queue: Queue for output frames (used for buffer size checks)
            
        Returns:
            Number of frames submitted for processing
        """
        current_time = time.time()
        streaming_dataset = state['streaming_dataset']
        submitted_frames = state['submitted_frames']
        frame_generation_rate = state['frame_generation_rate']
        
        # Calculate target frame count based on elapsed time
        elapsed_since_start = current_time - state['generation_start_time']
        target_frames_by_now = int(elapsed_since_start * frame_generation_rate)
        frames_to_generate = target_frames_by_now - state['total_frames_generated']
        
        if frames_to_generate <= 0:
            return 0  # Already ahead of schedule
        
        # Determine next frame index
        if submitted_frames:
            next_frame_idx = max(submitted_frames) + 1
        else:
            next_frame_idx = 0
        
        # Check available audio frames
        max_encodable = streaming_dataset.get_max_encodable_frame()
        frames_to_submit = []
        
        # Collect all available audio frames
        for i in range(frames_to_generate):
            frame_idx = next_frame_idx + i
            if frame_idx <= max_encodable and frame_idx not in submitted_frames:
                frames_to_submit.append(frame_idx)
            else:
                break  # No more audio frames available
        
        # Calculate frames still needed to maintain rate
        frames_still_needed = frames_to_generate - len(frames_to_submit)
        
        # Handle temporary silence mode
        if frames_still_needed > 0 and state['in_temporary_silence']:
            # Get buffer configuration for silence mode
            TARGET_BUFFER_SECONDS = state.get('silence_buffer_duration_seconds', 0.5)
            TARGET_BUFFER_FRAMES = int(TARGET_BUFFER_SECONDS * 25)
            
            # Critical threshold: 20% of target
            CRITICAL_PERCENT = 0.2
            CRITICAL_FRAMES = max(2, int(TARGET_BUFFER_FRAMES * CRITICAL_PERCENT))
            
            yield_buffer_size = output_queue.qsize()
            
            # Case 1: We have some audio frames but buffer is getting low
            if len(frames_to_submit) > 0 and yield_buffer_size < CRITICAL_FRAMES:
                # Submit available audio frames and top up with minimal silence
                print(f"[Hybrid] Have {len(frames_to_submit)} audio frames, "
                      f"but yield buffer critical ({yield_buffer_size} frames). "
                      f"Generating minimal silence to prevent underrun")
                
                # Generate just enough silence to stay above critical level
                silence_frames_needed = CRITICAL_FRAMES - yield_buffer_size + 5  # Small safety margin
                
                # For multiple streams, batch silence generation more efficiently
                active_stream_count = len(self.active_streams) if hasattr(self, 'active_streams') else 1
                if active_stream_count > 1:
                    # Generate slightly more silence to reduce frequency of generation
                    silence_frames_needed = max(silence_frames_needed, int(TARGET_BUFFER_FRAMES * 0.4))
                
                silence_duration = silence_frames_needed / 25.0
                streaming_dataset.add_silence(silence_duration)
                
                # Collect newly available silence frames
                new_max_encodable = streaming_dataset.get_max_encodable_frame()
                for i in range(len(frames_to_submit), frames_to_generate):
                    frame_idx = next_frame_idx + i
                    if frame_idx <= new_max_encodable and frame_idx not in submitted_frames:
                        frames_to_submit.append(frame_idx)
                    else:
                        break
            
            # Case 2: No audio frames and buffer is critical
            elif len(frames_to_submit) == 0 and yield_buffer_size < CRITICAL_FRAMES:
                # Calculate total frames in flight (buffer + GPU pipeline)
                pending_in_model = len(submitted_frames) - state['stream_info']['frames_processed']
                total_frames_available = yield_buffer_size + pending_in_model
                
                # Only generate if we need more frames
                if total_frames_available < TARGET_BUFFER_FRAMES:
                    # Generate exactly enough to reach target
                    frames_to_generate_silence = TARGET_BUFFER_FRAMES - total_frames_available
                    silence_duration = frames_to_generate_silence / 25.0
                    
                    print(f"[Silence] Yield buffer: {yield_buffer_size}/{TARGET_BUFFER_FRAMES} frames, "
                          f"generating {silence_duration:.2f}s to reach target")
                    
                    streaming_dataset.add_silence(silence_duration)
                    
                    # Collect newly available frames
                    new_max_encodable = streaming_dataset.get_max_encodable_frame()
                    for i in range(len(frames_to_submit), frames_to_generate):
                        frame_idx = next_frame_idx + i
                        if frame_idx <= new_max_encodable and frame_idx not in submitted_frames:
                            frames_to_submit.append(frame_idx)
                        else:
                            break
                else:
                    # Enough frames in pipeline, no action needed
                    print(f"[Silence] Yield buffer low ({yield_buffer_size} frames) but "
                          f"{pending_in_model} frames in GPU, total {total_frames_available} - no silence needed")
    
        # Handle final silence (after audio exhausted)
        elif frames_still_needed > 0 and state['audio_exhausted'] and state['continue_silent']:
            silence_duration = frames_still_needed / 25.0
            print(f"[FinalSilent] Adding {silence_duration:.2f}s of silence")
            streaming_dataset.add_silence(silence_duration)
            
            # Collect the generated silence frames
            new_max_encodable = streaming_dataset.get_max_encodable_frame()
            for i in range(len(frames_to_submit), frames_to_generate):
                frame_idx = next_frame_idx + i
                if frame_idx <= new_max_encodable and frame_idx not in submitted_frames:
                    frames_to_submit.append(frame_idx)
                else:
                    break
        
        # Submit collected frames for processing
        frames_submitted = 0
        if frames_to_submit:
            # Verify frame continuity to prevent gaps
            if submitted_frames:
                expected_start = max(submitted_frames) + 1
                actual_start = min(frames_to_submit)
                if actual_start != expected_start:
                    print(f"[ERROR] Frame continuity broken! Expected to submit {expected_start}, "
                        f"but submitting {actual_start}. Gap of {actual_start - expected_start} frames!")
                    return 0  # Don't submit - would create a gap
            
            # Submit frames to GPU pipeline
            frames_submitted = self._submit_frames_for_processing(
                frames_to_submit, state['stream_id'], streaming_dataset, 
                state['wrapper'], submitted_frames, state
            )
            state['frames_processed'] += frames_submitted
            state['total_frames_generated'] += frames_submitted
            
            # Track silent frames for duration limiting
            if state['audio_exhausted'] or state['in_temporary_silence']:
                state['silent_frames_generated'] += frames_submitted
            
            # Log progress periodically (less frequently during silence to reduce overhead)
            log_frequency = int(frame_generation_rate * 2) if state['in_temporary_silence'] else int(frame_generation_rate)
            if state['total_frames_generated'] % log_frequency == 0:
                elapsed = current_time - state['generation_start_time']
                actual_fps = state['total_frames_generated'] / elapsed if elapsed > 0 else 0
                
                # Calculate audio buffer status
                audio_frames_ready = max_encodable - next_frame_idx + 1
                audio_seconds_buffered = audio_frames_ready / 25.0 if audio_frames_ready > 0 else 0
                
                # Determine current mode for logging
                if state['in_temporary_silence']:
                    mode = "[TempSilent]"
                elif state['audio_exhausted']:
                    mode = "[FinalSilent]"
                else:
                    mode = "[Audio]"
                    
                # Only log if FPS is significantly different from target or in audio mode
                if state['stream_id'] in self.active_streams:
                    if not state['in_temporary_silence'] or abs(actual_fps - frame_generation_rate) > 5:
                        print(f"[Stream {state['stream_id']}] {mode} Generated {state['total_frames_generated']} frames in {elapsed:.1f}s "
                            f"(actual: {actual_fps:.1f} FPS, target: {frame_generation_rate} FPS, "
                            f"audio buffered: {audio_seconds_buffered:.1f}s)")
        
        return frames_submitted

    def _process_stream_iteration_threaded(self, state: dict, output_queue: queue.Queue, 
                                       shared_stats: dict, stats_lock: threading.Lock) -> tuple:
        """
        Process one iteration of the main streaming loop.
        
        This is the heart of the streaming system, called repeatedly by the processing thread.
        It implements sophisticated buffer management that adapts to the current state:
        
        1. Audio Processing: Reads and processes incoming audio chunks
        2. Buffer Management: Monitors output buffer and triggers refills
        3. Frame Generation: Generates frames in bursts for efficiency
        4. Output Collection: Collects processed frames from GPU
        5. Statistics: Tracks performance metrics
        
        The method uses different strategies based on state:
        - Building phase: Use rate-based generation to fill initial buffer
        - Active phase: Use burst refills when buffer drops below threshold
        - Silence phase: Use smaller buffer for lower latency
        
        Returns:
            Tuple of (continue_loop, frames_to_add)
        """
        frames_to_add = []
        continue_loop = True
        
        # Check circuit breaker early to stop processing immediately
        wrapper = state['wrapper']
        streaming_dataset = state['streaming_dataset']
        if streaming_dataset.circuit_breaker:
            print(f"[Stream {state['stream_id']}] Circuit breaker activated - stopping processing")
            return False, []
        
        # Check if stream is still active
        if state['stream_id'] not in self.active_streams:
            print(f"[Stream {state['stream_id']}] Stream no longer active - stopping processing")
            streaming_dataset.circuit_breaker = True
            return False, []
        
        submitted_frames = state['submitted_frames']
        
        # Dynamic buffer configuration based on current state
        # This is the key to low-latency silence-to-speech transitions
        # Each stream maintains its own buffer configuration
        active_stream_count = len(self.active_streams) if hasattr(self, 'active_streams') else 1
        
        if state['in_temporary_silence']:
            # Use optimized buffer sizes based on stream count
            TARGET_BUFFER_SECONDS = SilenceOptimizationConfig.get_buffer_size(
                active_stream_count, 
                base_size=state.get('silence_buffer_duration_seconds', 0.5)
            )
            
            if not state.get('multi_stream_buffer_logged', False):
                print(f"[Stream {state['stream_id']}] Multi-stream silence mode: {active_stream_count} streams, "
                      f"using {TARGET_BUFFER_SECONDS:.1f}s buffer (optimized)")
                state['multi_stream_buffer_logged'] = True
        else:
            # Use full buffer during active audio for resilience
            TARGET_BUFFER_SECONDS = state.get('buffer_duration_seconds', 2.0)
            state['multi_stream_buffer_logged'] = False  # Reset flag
        
        # Calculate all thresholds as percentages for consistency
        if state['in_temporary_silence']:
            # Use optimized thresholds for silence based on stream count
            REFILL_THRESHOLD_PERCENT = SilenceOptimizationConfig.get_refill_threshold(active_stream_count)
            REFILL_AMOUNT_PERCENT = 0.6     # Larger refills during silence
        else:
            REFILL_THRESHOLD_PERCENT = 0.5  # Standard refill at 50%
            REFILL_AMOUNT_PERCENT = 0.5     # Standard refill amount
        
        CRITICAL_BUFFER_PERCENT = 0.25  # Consider buffer critical at 25%
        MIN_REFILL_PERCENT = 0.2        # Minimum refill is 20% of buffer
    
        # Convert percentages to frame counts
        TARGET_BUFFER_FRAMES = int(TARGET_BUFFER_SECONDS * 25)
        REFILL_THRESHOLD = int(TARGET_BUFFER_FRAMES * REFILL_THRESHOLD_PERCENT)
        REFILL_AMOUNT = int(TARGET_BUFFER_FRAMES * REFILL_AMOUNT_PERCENT)
        CRITICAL_YIELD_BUFFER_FRAMES = int(TARGET_BUFFER_FRAMES * CRITICAL_BUFFER_PERCENT)
        MIN_SILENCE_FRAMES = int(TARGET_BUFFER_FRAMES * MIN_REFILL_PERCENT)
        
        # Ensure minimum values for very small buffers
        REFILL_AMOUNT = max(REFILL_AMOUNT, 5)
        CRITICAL_YIELD_BUFFER_FRAMES = max(CRITICAL_YIELD_BUFFER_FRAMES, 3)
        MIN_SILENCE_FRAMES = max(MIN_SILENCE_FRAMES, 5)

        # Initialize buffer state tracking
        if 'buffer_state' not in state:
            state['buffer_state'] = {
                'last_refill_time': time.time(),
                'total_refills': 0,
                'frames_in_current_burst': 0,
                'target_frames_for_burst': 0,
                'first_audio_received': False,
                'buffer_management_enabled': False,
                'peak_buffer_size': 0,
                'last_buffer_mode': 'audio',
                'silence_generated_for_yield_buffer': 0
            }
        
        # Log buffer mode transitions
        current_buffer_mode = 'silence' if state['in_temporary_silence'] else 'audio'
        if current_buffer_mode != state['buffer_state'].get('last_buffer_mode'):
            print(f"[Buffer] Mode changed to {current_buffer_mode}, target buffer: {TARGET_BUFFER_SECONDS}s ({TARGET_BUFFER_FRAMES} frames)")
            print(f"  - Refill threshold: {REFILL_THRESHOLD} frames ({REFILL_THRESHOLD_PERCENT*100:.0f}%)")
            print(f"  - Critical threshold: {CRITICAL_YIELD_BUFFER_FRAMES} frames ({CRITICAL_BUFFER_PERCENT*100:.0f}%)")
            state['buffer_state']['last_buffer_mode'] = current_buffer_mode
        
        # STEP 1: Process incoming audio
        self._process_audio_input(state)
        
        # Capture buffer size at audio resume for latency measurement
        if 'latency_test' in state and state['latency_test']['buffer_size_at_resume'] is None:
            state['latency_test']['buffer_size_at_resume'] = output_queue.qsize()
            print(f"[LATENCY TEST] Buffer size at audio resume: {state['latency_test']['buffer_size_at_resume']} frames")
            
            # Share latency info with yielding thread
            with stats_lock:
                shared_stats['latency_test'] = state['latency_test'].copy()
        
        # STEP 2: Monitor buffer status
        current_buffer_size = output_queue.qsize()
        buffer_state = state['buffer_state']

        # Track peak buffer size for statistics
        if current_buffer_size > buffer_state['peak_buffer_size']:
            buffer_state['peak_buffer_size'] = current_buffer_size

        # Enable buffer management once we reach target size
        if not buffer_state['buffer_management_enabled'] and current_buffer_size >= TARGET_BUFFER_FRAMES:
            buffer_state['buffer_management_enabled'] = True
            print(f"[Buffer] Reached target size ({TARGET_BUFFER_FRAMES} frames), enabling buffer management")

        # STEP 3: Generate frames using appropriate strategy
        if buffer_state['buffer_management_enabled']:
            # Use sophisticated buffer management with burst refills
            should_generate = False
            frames_to_submit = []
            
            if buffer_state['frames_in_current_burst'] > 0:
                # Continue current burst
                should_generate = True
            elif current_buffer_size <= REFILL_THRESHOLD:
                # Start new burst refill
                frames_needed = TARGET_BUFFER_FRAMES - current_buffer_size
                
                # During silence with multiple streams, generate larger bursts
                if state['in_temporary_silence']:
                    active_stream_count = len(self.active_streams) if hasattr(self, 'active_streams') else 1
                    if active_stream_count > 1:
                        # Generate larger bursts to reduce contention
                        burst_multiplier = min(1.5, 1 + (active_stream_count - 1) * 0.25)
                        buffer_state['target_frames_for_burst'] = min(frames_needed, 
                                                                     int(REFILL_AMOUNT * burst_multiplier))
                    else:
                        buffer_state['target_frames_for_burst'] = min(frames_needed, REFILL_AMOUNT)
                else:
                    buffer_state['target_frames_for_burst'] = min(frames_needed, REFILL_AMOUNT)
                
                buffer_state['frames_in_current_burst'] = 0
                buffer_state['last_refill_time'] = time.time()
                buffer_state['total_refills'] += 1
                should_generate = True
                
                mode_str = "[SilenceBuffer]" if state['in_temporary_silence'] else "[AudioBuffer]"
                if state['stream_id'] in self.active_streams:
                    print(f"[Stream {state['stream_id']}] {mode_str} Refill triggered: buffer={current_buffer_size} frames, "
                        f"targeting {buffer_state['target_frames_for_burst']} new frames "
                        f"(threshold: {REFILL_THRESHOLD})")
            
            if should_generate:
                # Execute burst generation
                remaining_in_burst = buffer_state['target_frames_for_burst'] - buffer_state['frames_in_current_burst']
                
                # Determine next frame to generate
                if submitted_frames:
                    next_frame_idx = max(submitted_frames) + 1
                else:
                    next_frame_idx = 0
                
                # First pass: collect available audio frames
                max_encodable = streaming_dataset.get_max_encodable_frame()
                
                for i in range(remaining_in_burst):
                    frame_idx = next_frame_idx + i
                    if frame_idx <= max_encodable and frame_idx not in submitted_frames:
                        frames_to_submit.append(frame_idx)
                
                # Second pass: generate silence if needed (only during silence mode)
                frames_still_needed = remaining_in_burst - len(frames_to_submit)
                
                if frames_still_needed > 0:
                    yield_buffer_size = output_queue.qsize()
                    
                    # Only generate silence if in silence mode AND buffer is critical
                    should_generate_silence = (
                        state['in_temporary_silence'] and
                        yield_buffer_size < CRITICAL_YIELD_BUFFER_FRAMES
                    )
                    
                    if should_generate_silence:
                        # Calculate total frames in flight
                        pending_in_model = len(submitted_frames) - state['stream_info']['frames_processed']
                        total_frames_in_flight = yield_buffer_size + pending_in_model
                        
                        # Only generate if below target
                        if total_frames_in_flight < TARGET_BUFFER_FRAMES:
                            # Use optimized batch size based on stream count
                            optimal_batch = SilenceOptimizationConfig.get_batch_size(active_stream_count)
                            frames_to_generate = min(optimal_batch, TARGET_BUFFER_FRAMES - total_frames_in_flight)
                            frames_to_generate = max(MIN_SILENCE_FRAMES, frames_to_generate)
                            silence_duration = frames_to_generate / 25.0
                            
                            print(f"[YieldBuffer] CRITICAL: Yield buffer at {yield_buffer_size} frames "
                                  f"({yield_buffer_size/TARGET_BUFFER_FRAMES*100:.0f}% of target), "
                                  f"generating {frames_to_generate} silent frames ({silence_duration:.2f}s)")
                            
                            streaming_dataset.add_silence(silence_duration)
                            buffer_state['silence_generated_for_yield_buffer'] += frames_to_generate
                            
                            # Collect newly available frames
                            new_max_encodable = streaming_dataset.get_max_encodable_frame()
                            
                            for i in range(len(frames_to_submit), len(frames_to_submit) + frames_to_generate):
                                frame_idx = next_frame_idx + i
                                if frame_idx <= new_max_encodable and frame_idx not in submitted_frames:
                                    frames_to_submit.append(frame_idx)
                                    if len(frames_to_submit) >= remaining_in_burst:
                                        break
                        else:
                            print(f"[YieldBuffer] Have {total_frames_in_flight} frames in flight "
                                  f"(yield: {yield_buffer_size}, GPU: {pending_in_model}), no silence needed")
                
                # Handle partial bursts during audio mode (expected behavior)
                if not state['in_temporary_silence'] and frames_still_needed > 0:
                    if frames_to_submit:
                        print(f"[Buffer] Only {len(frames_to_submit)} audio frames available "
                              f"(wanted {remaining_in_burst}), processing partial burst")
            
            # Submit collected frames
            if frames_to_submit:
                frames_submitted = self._submit_frames_for_processing(
                    frames_to_submit, state['stream_id'], streaming_dataset, 
                    wrapper, submitted_frames, state
                )
                
                state['frames_processed'] += frames_submitted
                state['total_frames_generated'] += frames_submitted
                buffer_state['frames_in_current_burst'] += frames_submitted
                
                # Track silent frames
                if state['audio_exhausted'] or state['in_temporary_silence']:
                    state['silent_frames_generated'] += frames_submitted
                
                # Check if burst is complete
                if (buffer_state['frames_in_current_burst'] >= buffer_state['target_frames_for_burst'] or
                    (not state['in_temporary_silence'] and not (state['audio_exhausted'] and state['continue_silent']))):
                    if state['stream_id'] in self.active_streams:
                        print(f"[Stream {state['stream_id']}] Burst complete: generated {buffer_state['frames_in_current_burst']} frames")
                    buffer_state['frames_in_current_burst'] = 0
                    buffer_state['target_frames_for_burst'] = 0
        else:
            # Buffer building phase - use rate-based generation
            frames_submitted = self._generate_frames_at_rate(state, output_queue)
        
        # STEP 4: Collect processed frames from GPU for this specific stream
        context = wrapper.stream_workers.get(state['stream_id'])
        if not context:
            print(f"❌ Stream {state['stream_id']} context not found. Stopping processing.")
            return False, []

        ready_frames = []
        with context.output_lock:
            # Debug: Check what's in the buffer
            if len(context.output_buffer) > 0 and state.get('debug_counter', 0) % 100 == 0:
                buffer_keys = sorted(list(context.output_buffer.keys()))[:5]
                print(f"[DEBUG] Stream {state['stream_id']}: Output buffer has {len(context.output_buffer)} frames, keys: {buffer_keys}, next_expected: {context.next_output_idx}")
            state['debug_counter'] = state.get('debug_counter', 0) + 1
            
            while context.next_output_idx in context.output_buffer:
                frame_data = context.output_buffer.pop(context.next_output_idx)
                ready_frames.append(frame_data)
                context.next_output_idx += 1

        for frame_data in ready_frames:
            if len(frame_data) == 4:
                final_frame, img_idx, frame_idx, frame_audio = frame_data
            else:
                final_frame, img_idx, frame_idx = frame_data
                frame_audio = streaming_dataset.get_audio_for_frame(frame_idx)
            
            # Cache silent frames. The frame_idx here is relative to the stream.
            if streaming_dataset.is_silent_frame(frame_idx):
                model_name = state['wrapper'].model_name
                cache_key = (model_name, img_idx)
                if cache_key not in self.silent_frame_cache:
                    self.silent_frame_cache[cache_key] = (final_frame, frame_audio)
                    if len(self.silent_frame_cache) % 25 == 0: # Log less often
                        print(f"[Cache] Stored silent frame for {model_name}, img_idx {img_idx}. Cache size: {len(self.silent_frame_cache)}")
            
            # Check for markers that should be inserted before this frame
            while state['marker_queue'] and state['marker_queue'][0][1] <= frame_idx:
                marker, boundary = state['marker_queue'].popleft()
                print(f"Inserting marker {marker.decode()} at frame {frame_idx}")
                frames_to_add.append((marker, None, None, marker))
            
            # Add the processed frame
            frames_to_add.append((final_frame, img_idx, frame_idx + state['last_index'], frame_audio))
            state['stream_info']['frames_processed'] += 1
            
            # Track first frame timing
            if not state['first_frame_ready']:
                state['first_frame_ready'] = True
                print(f"First frame ready: {time.time() - state['stream_start_time']:.3f}s")
        
        # STEP 5: Log statistics periodically
        current_time = time.time()
        if current_time - state['last_stats_time'] > 5.0:
            elapsed = current_time - state['stream_start_time']
            
            # Get stats from yielding thread
            with stats_lock:
                frames_yielded = shared_stats['frames_yielded']
                buffer_size = output_queue.qsize()
            
            # Calculate performance metrics
            generation_fps = state['stream_info']['frames_processed'] / elapsed if elapsed > 0 else 0
            yield_fps = frames_yielded / elapsed if elapsed > 0 else 0
            
            # Determine buffer status
            if buffer_state['buffer_management_enabled']:
                buffer_health = "Healthy" if buffer_size >= REFILL_THRESHOLD else "Low"
                buffer_status = f"Active [{buffer_health}]"
            else:
                buffer_status = f"Building (peak: {buffer_state['peak_buffer_size']})"
            
            # Only log if stream is still active
            if state['stream_id'] in self.active_streams:
                print(f"[Stream {state['stream_id']}] {elapsed:.1f}s: Ready={state['stream_info']['frames_processed']}, "
                    f"Yielded={frames_yielded}, "
                    f"Buffer: {buffer_size}/{TARGET_BUFFER_FRAMES} frames ({buffer_size/25.0:.1f}s) {buffer_status}, "
                    f"Refills: {buffer_state['total_refills']}, "
                    f"Gen FPS={generation_fps:.1f}, Yield FPS={yield_fps:.1f}")
            
            state['last_stats_time'] = current_time
        
        # STEP 6: Check exit conditions
        if state['audio_exhausted'] and not state['continue_silent']:
            # Process any remaining markers
            while state['marker_queue']:
                marker, _ = state['marker_queue'].popleft()
                frames_to_add.append((marker, None, None, marker))
            
            # Check if all frames are processed
            max_frame = streaming_dataset.get_max_encodable_frame()
            all_frames_submitted = all(f in submitted_frames for f in range(max_frame + 1))
            
            if all_frames_submitted and state['stream_info']['frames_processed'] >= state['frames_processed']:
                print(f"\nAll frames completed!")
                continue_loop = False
        
        # Check silent generation duration limit
        if state['audio_exhausted'] and state['continue_silent'] and state['silent_duration_seconds']:
            if state['silent_frames_generated'] >= state['target_silent_frames']:
                print(f"Silent generation complete: {state['silent_frames_generated']} frames")
                state['continue_silent'] = False
        
        # Check circuit breaker (redundant check for safety, should have been caught earlier)
        if streaming_dataset.circuit_breaker:
            print(f"[Stream {state['stream_id']}] Circuit breaker detected in main loop - stopping")
            continue_loop = False
        
        # Adaptive sleep based on system state with multi-stream optimization
        active_stream_count = len(self.active_streams) if hasattr(self, 'active_streams') else 1
        
        # Staggering should be applied if there's more than one active stream
        should_stagger = SilenceOptimizationConfig.should_stagger_streams(active_stream_count)
        
        if buffer_state['frames_in_current_burst'] > 0:
            # Actively generating frames (burst in progress)
            if state['in_temporary_silence']:
                # Use optimized sleep duration for silence generation
                sleep_time = SilenceOptimizationConfig.get_sleep_duration(active_stream_count, 'generating')
                if should_stagger:
                    sleep_time += SilenceOptimizationConfig.get_stream_offset(state['stream_id'], max_offset_ms=5)
                # time.sleep(sleep_time / 100)  # COMPLETELY REMOVED - Rapido handles timing
                pass  # NO SLEEP - RAPIDO HANDLES TIMING
            else:
                # ACTIVE AUDIO, GENERATING: REMOVE SLEEP FOR MAX SPEED
                # sleep_time = 0.001  # COMMENTED OUT
                # if should_stagger:
                #     sleep_time += SilenceOptimizationConfig.get_stream_offset(state['stream_id'], max_offset_ms=2)
                # time.sleep(sleep_time)  # COMMENTED OUT
                pass  # NO SLEEP DURING ACTIVE AUDIO GENERATION

        elif current_buffer_size > TARGET_BUFFER_FRAMES * 0.8:
            # Buffer is full, can sleep longer
            if state['in_temporary_silence']:
                sleep_time = SilenceOptimizationConfig.get_sleep_duration(active_stream_count, 'buffer_full')
                if should_stagger:
                    sleep_time += SilenceOptimizationConfig.get_stream_offset(state['stream_id'], max_offset_ms=10)
                # time.sleep(sleep_time / 50)  # COMPLETELY REMOVED - Rapido handles timing
                pass  # NO SLEEP - RAPIDO HANDLES TIMING
            else:
                # ACTIVE AUDIO, BUFFER FULL: REDUCE SLEEP FOR FASTER GENERATION
                # sleep_time = 0.02  # COMMENTED OUT
                # if should_stagger:
                #     sleep_time += SilenceOptimizationConfig.get_stream_offset(state['stream_id'], max_offset_ms=8)
                # time.sleep(sleep_time)  # COMMENTED OUT
                time.sleep(0.0005)  # MINIMAL SLEEP to prevent GPU queue overflow
        else:
            # Normal operation, buffer is not full but not actively generating a burst
            if state['in_temporary_silence']:
                sleep_time = SilenceOptimizationConfig.get_sleep_duration(active_stream_count, 'normal')
                if should_stagger:
                    sleep_time += SilenceOptimizationConfig.get_stream_offset(state['stream_id'], max_offset_ms=8)
                # time.sleep(sleep_time / 100)  # COMPLETELY REMOVED - Rapido handles timing
                pass  # NO SLEEP - RAPIDO HANDLES TIMING
            else:
                # ACTIVE AUDIO, NORMAL: MINIMAL SLEEP FOR MAX SPEED
                # sleep_time = 0.005  # COMMENTED OUT
                # if should_stagger:
                #     sleep_time += SilenceOptimizationConfig.get_stream_offset(state['stream_id'], max_offset_ms=4)
                # time.sleep(sleep_time)  # COMMENTED OUT
                time.sleep(0.001)  # MINIMAL SLEEP to prevent GPU queue overflow
        
        return continue_loop, frames_to_add

    def generate_video_stream(self, model_name: str, 
                    audio_chunks: Generator[bytes, None, None],
                    video_path: Optional[str] = None,
                    frame_source: Optional[AllFramesMemory] = None,
                    landmark_source: Optional[LandmarkManager] = None,
                    is_silent: bool = False,
                    last_index: int = 0,
                    profiler: Optional[PerformanceProfiler] = None,
                    continue_silent: bool = False,
                    silent_duration_seconds: Optional[float] = None,
                    frame_generation_rate: float = 25.0,
                    buffer_duration_seconds: float = 2.0,
                    silence_buffer_duration_seconds: float = 0.5,
                    stream_id: Optional[str] = None) -> Generator[Tuple, None, None]:
        """
        Generate a stream of lip-synced video frames from audio input.
        
        This is the main entry point for streaming inference. It sets up a sophisticated
        dual-thread architecture:
        
        1. Processing Thread: Reads audio, manages buffers, generates frames
        2. Yielding Thread: Delivers frames at exactly 25 FPS for smooth playback
        
        The method implements dynamic buffer management with two modes:
        - Active speech: Large buffer (2.0s) for resilience against processing variations
        - Silence: Small buffer (0.5s) for low latency when speech resumes
        
        This design achieves:
        - Smooth, stutter-free playback during active speech
        - Low latency (<0.7s) when transitioning from silence to speech
        - Efficient GPU utilization through burst processing
        - Accurate audio-video synchronization
        
        Args:
            model_name: Name of the pre-loaded model to use
            audio_chunks: Generator yielding audio chunks (bytes) or None for silence
            video_path: Optional path to video file (overrides model config)
            frame_source: Optional pre-loaded frame manager
            landmark_source: Optional pre-loaded landmark manager
            is_silent: If True, generate silent video (deprecated, use None audio chunks)
            last_index: Starting frame index for output
            profiler: Optional performance profiler instance
            continue_silent: If True, continue generating frames after audio ends
            silent_duration_seconds: Duration of silence to generate after audio
            frame_generation_rate: Output frame rate (must be 25.0 for now)
            buffer_duration_seconds: Buffer size during active speech (default: 2.0)
            silence_buffer_duration_seconds: Buffer size during silence (default: 0.5)
            
        Yields:
            Tuple of (frame, img_idx, frame_idx, audio_chunk) at 25 FPS
            
        Raises:
            ValueError: If model is not loaded
            Exception: Any errors during processing are re-raised after cleanup
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Create profiler if not provided
        if profiler is None:
            profiler = PerformanceProfiler(f"StreamingAPI_{model_name}")
        
        wrapper = self.models[model_name]
        
        # Use provided stream_id or generate a new one
        if stream_id is None:
            stream_id = str(uuid.uuid4())
        
        # CRITICAL FIX: Ensure clean state for new stream
        # With per-stream workers, we just need to create a new context.
        # The wrapper handles starting/stopping workers for that context.
        if hasattr(wrapper, 'stream_workers'):
            if stream_id in wrapper.stream_workers:
                print(f"🔄 Stream {stream_id} already exists. Removing old context before creating a new one.")
                wrapper.remove_stream_context(stream_id)
        
        # Use model's video path if not specified
        if video_path is None and hasattr(wrapper, 'video_path'):
            video_path = wrapper.video_path
        
        # Initialize all streaming resources
        frame_manager, landmark_manager, streaming_dataset, created_frame_manager = \
            self._initialize_stream_resources(model_name, video_path, frame_source, 
                                            landmark_source, last_index, profiler, stream_id)
        
        # Register this stream for management
        stream_info = {
            'model': model_name,
            'start_time': time.time(),
            'frames_processed': 0
        }
        self.active_streams[stream_id] = stream_info
        
        # Create thread-safe output queue with appropriate size
        # Size is 1.5x the maximum buffer to provide headroom
        max_queue_size = int(max(buffer_duration_seconds, silence_buffer_duration_seconds) * 25 * 1.5)
        output_queue = queue.Queue(maxsize=max_queue_size)
        processing_complete = threading.Event()
        processing_error = None
        
        # Shared statistics between threads
        stats_lock = threading.Lock()
        shared_stats = {
            'frames_yielded': 0,
            'first_frame_yielded': False,
            'buffer_high_water_mark': 0
        }
        
        def yielding_thread():
            """
            Thread that yields frames at exactly the specified frame rate.
            
            This thread is responsible for delivering frames at a constant rate
            regardless of processing speed. It implements frame-accurate timing
            to ensure smooth playback without stuttering or speed variations.
            
            The thread also performs latency measurements to track the time
            between audio resumption and when that audio appears in the output.
            """
            try:
                yield_interval = 1.0 / frame_generation_rate
                last_yield_time = time.time()
                local_frames_yielded = 0
                
                while True:
                    current_time = time.time()
                    
                    # Calculate how many frames should be yielded by now
                    elapsed = current_time - last_yield_time
                    frames_to_yield = int(elapsed / yield_interval)
                    
                    if frames_to_yield > 0:
                        for _ in range(frames_to_yield):
                            try:
                                # Try to get a frame with minimal timeout
                                frame_data = output_queue.get(timeout=0.001)
                                
                                # Check for stop signal
                                if frame_data is None:
                                    return
                                
                                # Perform latency measurement if needed
                                if len(frame_data) >= 3:
                                    _, _, frame_idx = frame_data[:3]
                                    
                                    with stats_lock:
                                        if ('latency_test' in shared_stats and
                                            shared_stats['latency_test'] and
                                            not shared_stats['latency_test'].get('measured', True) and
                                            shared_stats['latency_test'].get('target_frame') is not None):

                                            if frame_idx >= shared_stats['latency_test']['target_frame']:
                                                latency = current_time - shared_stats['latency_test']['audio_resumed_time']
                                                buffer_size = shared_stats['latency_test']['buffer_size_at_resume']
                                                
                                                print(f"\n[LATENCY TEST] ========== RESULTS ==========")
                                                print(f"  Silence-to-audio transition latency: {latency:.3f}s")
                                                print(f"  Buffer size at resume: {buffer_size} frames ({buffer_size/25.0:.2f}s)")
                                                print(f"  Target frame: {shared_stats['latency_test']['target_frame']}")
                                                print(f"  Actual frame yielded: {frame_idx}")
                                                print(f"=======================================\n")
                                                
                                                shared_stats['latency_test']['measured'] = True
                                
                                # Update statistics
                                with stats_lock:
                                    shared_stats['frames_yielded'] += 1
                                    if not shared_stats['first_frame_yielded']:
                                        shared_stats['first_frame_yielded'] = True
                                        print(f"First frame yielded: {time.time() - stream_info['start_time']:.3f}s "
                                            f"(yielding at {frame_generation_rate} FPS)")
                                
                                local_frames_yielded += 1
                                last_yield_time += yield_interval
                                
                                # Yield to the calling generator
                                yield frame_data
                                
                            except queue.Empty:
                                # No frame available, skip this yield slot
                                break
                    
                    # Check if we should exit
                    if processing_complete.is_set() and output_queue.empty():
                        return
                    
                    # COMPLETELY REMOVED SLEEP - Rapido handles all timing
                    # queue_size = output_queue.qsize()
                    # if queue_size < 5:
                    #     time.sleep(0.0001)  # REMOVED
                    # elif queue_size > 50:
                    #     time.sleep(0.001)   # REMOVED
                    # else:
                    #     time.sleep(0.0005)  # REMOVED
                    time.sleep(0.0001)  # ULTRA MINIMAL to prevent queue overflow
                        
            except Exception as e:
                print(f"Error in yielding thread: {e}")
                import traceback
                traceback.print_exc()
        
        def processing_thread():
            """
            Main processing thread that handles audio input and frame generation.
            
            This thread implements the core streaming logic:
            1. Reads audio chunks from the input generator
            2. Manages dynamic buffer sizing based on audio availability
            3. Generates frames in efficient bursts
            4. Handles transitions between speech and silence
            5. Monitors performance and adjusts behavior
            """
            nonlocal processing_error
            
            try:
                # Initialize processing state
                state = {
                    'wrapper': wrapper,
                    'streaming_dataset': streaming_dataset,
                    'frame_manager': frame_manager,
                    'landmark_manager': landmark_manager,
                    'audio_iterator': iter(audio_chunks),
                    'audio_exhausted': False,
                    'submitted_frames': set(),
                    'marker_queue': deque(),
                    'stream_info': stream_info,
                    'stream_id': stream_id,
                    'last_index': last_index,
                    'chunk_count': 0,
                    'frames_processed': 0,
                    'total_frames_generated': 0,
                    'silent_frames_generated': 0,
                    'generation_start_time': time.time(),
                    'frame_generation_rate': frame_generation_rate,
                    'target_silent_frames': int(silent_duration_seconds * 25) if silent_duration_seconds else None,
                    'in_temporary_silence': False,
                    'temporary_silence_start': None,
                    'continue_silent': continue_silent,
                    'silent_duration_seconds': silent_duration_seconds,
                    'stream_start_time': time.time(),
                    'last_stats_time': time.time(),
                    'first_frame_ready': False,
                    'next_frame_for_audio': 0,
                    'pending_silence_duration': 0.0,
                    'buffer_duration_seconds': buffer_duration_seconds,
                    'silence_buffer_duration_seconds': silence_buffer_duration_seconds,
                    'first_real_audio_received': False,
                    'in_temporary_silence': False,
                }
                
                print(f"Starting streaming with burst generation, yielding at {frame_generation_rate} FPS...")
                
                # Main processing loop
                while True:
                    # Process one iteration
                    continue_loop, frames_to_add = self._process_stream_iteration_threaded(
                        state, output_queue, shared_stats, stats_lock)
                    
                    # Add frames to output queue
                    for frame_data in frames_to_add:
                        try:
                            # Block for up to 5 seconds. If the queue is still full,
                            # it indicates a problem with the consumer.
                            output_queue.put(frame_data, block=True, timeout=5.0)
                            
                            # Track buffer high water mark
                            with stats_lock:
                                current_size = output_queue.qsize()
                                if current_size > shared_stats['buffer_high_water_mark']:
                                    shared_stats['buffer_high_water_mark'] = current_size
                                    
                        except queue.Full:
                            print(f"ERROR: Stream {stream_id} output buffer full for 5 seconds. "
                                  "Consumer seems stuck. Stopping stream.")
                            streaming_dataset.circuit_breaker = True
                            break # Exit the for loop
                    
                    if not continue_loop or streaming_dataset.circuit_breaker:
                        break
                
                # Signal model to stop processing
                # In per-stream worker model, we don't need to signal the model to stop,
                # as the stream's workers will be torn down.
                # The yielding thread is signaled as before.
                output_queue.put(None)
                
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback
                traceback.print_exc()
                processing_error = e
                output_queue.put(None)  # Ensure yielding thread exits
            
            finally:
                processing_complete.set()
        
        # Start processing thread
        proc_thread = threading.Thread(target=processing_thread, name="FrameProcessing")
        proc_thread.start()
        
        try:
            # Yield frames from the yielding thread
            for frame_data in yielding_thread():
                yield frame_data
                
        except Exception as e:
            print(f"Error during streaming: {e}")
            import traceback
            traceback.print_exc()
            streaming_dataset.circuit_breaker = True
            
        finally:
            # Ensure clean shutdown
            processing_complete.set()
            proc_thread.join(timeout=5.0)
            
            if proc_thread.is_alive():
                print("Warning: Processing thread did not terminate cleanly")
            
            # Print final statistics (only if stream was not forcefully stopped)
            if stream_id in self.active_streams:
                with stats_lock:
                    print(f"\n[Stream {stream_id}] Final Statistics:")
                    print(f"  Frames submitted: {'unknown' if processing_error else 'completed'}")
                    print(f"  Frames completed: {stream_info['frames_processed']}")
                    print(f"  Frames yielded: {shared_stats['frames_yielded']}")
                    print(f"  Max buffer size: {shared_stats['buffer_high_water_mark']} frames")
            
            # Clean up resources
            if created_frame_manager:
                frame_manager.cleanup()
            
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            
            # Clean up streaming dataset references for this stream
            if hasattr(wrapper, 'streaming_datasets') and stream_id in wrapper.streaming_datasets:
                del wrapper.streaming_datasets[stream_id]
                
            # Clean up stream context from PerStreamWorkersWrapper
            try:
                if hasattr(wrapper, 'remove_stream_context'):
                    wrapper.remove_stream_context(stream_id)
                    print(f"✅ Stream context and workers for {stream_id} cleaned up")
            except Exception as e:
                print(f"⚠️ Error cleaning up stream context {stream_id}: {e}")
                
            # Clean up legacy single reference if it was this stream's dataset
            if (hasattr(wrapper, 'streaming_dataset') and 
                hasattr(wrapper, 'streaming_datasets') and
                not wrapper.streaming_datasets):
                # No more concurrent streams, clear legacy reference
                wrapper.streaming_dataset = None
            
            # Re-raise any processing error
            if processing_error:
                raise processing_error