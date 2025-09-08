# inference_system/test/test_video_strategies.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import numpy as np
import threading
import queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
import argparse
import psutil
import torch
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from inference_system.core.video_frame_manager import (
    Strategy1_AllFramesInMemory,
    Strategy2_SmartCache,
    Strategy4_CompressedMemory
)
from inference_system.utils.profiler import PerformanceProfiler

@dataclass
class AudioChunk:
    """Simulated audio chunk"""
    duration_seconds: float
    chunk_id: int
    data: np.ndarray  # Dummy audio data

@dataclass
class ProcessingResult:
    frames_processed: int
    total_time: float
    time_to_first_frame: float
    average_latency: float
    memory_peak_mb: float
    fps_achieved: float

class VideoStrategyBenchmark:
    """Benchmark video loading strategies with real use cases"""
    
    def __init__(self, video_path: str, fps: int = 25):
        self.video_path = video_path
        self.fps = fps
        self.profiler = PerformanceProfiler("VideoStrategyBenchmark")
        
    def simulate_inference_time(self, batch_size: int = 1):
        """Simulate actual model inference time based on your profiling data"""
        # Based on your data: ~6.86ms per frame for inference
        inference_time = 0.00686 * batch_size
        time.sleep(inference_time)
        
    def simulate_preprocessing(self):
        """Simulate preprocessing time"""
        # Based on your data: ~0.85ms per frame
        time.sleep(0.00085)
        
    def simulate_postprocessing(self):
        """Simulate postprocessing time"""
        # Based on your data: ~4.28ms per frame
        time.sleep(0.00428)

    def get_frame_info(self, frame_data):
        """Get debug info about a frame"""
        if frame_data is None:
            return "None"
        elif hasattr(frame_data, 'frame'):
            # VideoFrameData object with .frame attribute
            if frame_data.frame is not None and hasattr(frame_data.frame, 'shape'):
                return f"VideoFrameData(idx={getattr(frame_data, 'frame_idx', '?')}, shape={frame_data.frame.shape})"
            else:
                return f"VideoFrameData(idx={getattr(frame_data, 'frame_idx', '?')}, frame=None)"
        elif hasattr(frame_data, 'shape'):
            # Raw numpy array
            return f"ndarray(shape={frame_data.shape})"
        else:
            # Unknown type
            return f"{type(frame_data).__name__}"

    def get_frame_with_timeout(self, strategy, frame_num: int, timeout: float = 5.0) -> Optional[object]:
        """Get a frame with timeout to prevent hanging"""
        frame_data = None
        exception = None
        
        def fetch_frame():
            nonlocal frame_data, exception
            try:
                frame_data = strategy.get_next_frame_pingpong()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=fetch_frame, name=f"FrameFetch-{frame_num}")
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            logger.error(f"Timeout getting frame {frame_num} after {timeout}s")
            # Try to diagnose the state
            if hasattr(strategy, 'frame_cache'):
                logger.debug(f"Cache size: {len(strategy.frame_cache)}")
            if hasattr(strategy, 'prefetch_queue'):
                logger.debug(f"Prefetch queue size: {strategy.prefetch_queue.qsize()}")
            if hasattr(strategy, 'prefetch_thread') and strategy.prefetch_thread:
                logger.debug(f"Prefetch thread alive: {strategy.prefetch_thread.is_alive()}")
            return None
        
        if exception:
            raise exception
        
        return frame_data

    def benchmark_batch_processing(self, strategy, audio_duration_seconds: float) -> ProcessingResult:
        """
        Simulate batch processing scenario:
        - Process entire video for given audio duration
        - Measure end-to-end time including all operations
        """
        logger.info(f"Starting batch processing benchmark for {audio_duration_seconds}s audio")
        print(f"\nBatch Processing Scenario ({audio_duration_seconds}s audio)")
        print("-" * 50)
        
        total_frames_needed = int(audio_duration_seconds * self.fps)
        frames_processed = 0
        start_time = time.time()
        time_to_first = None
        
        # Track memory
        process = psutil.Process()
        memory_start = process.memory_info().rss / 1024 / 1024
        memory_peak = memory_start
        
        logger.debug(f"Total frames needed: {total_frames_needed}")
        
        # For Smart Cache, give more specific info
        strategy_name = type(strategy).__name__
        if "SmartCache" in strategy_name or "Strategy2" in strategy_name:
            logger.info(f"Smart Cache info - cache_size: {getattr(strategy, 'cache_size', '?')}, "
                       f"prefetch_size: {getattr(strategy, 'prefetch_size', '?')}")
        
        with self.profiler.timer("batch_processing_total"):
            # Process all frames
            batch_size = 8  # Typical batch size
            
            while frames_processed < total_frames_needed:
                # Get frames
                logger.debug(f"Getting batch of frames, current progress: {frames_processed}/{total_frames_needed}")
                with self.profiler.timer("get_batch_frames"):
                    frames = []
                    for i in range(min(batch_size, total_frames_needed - frames_processed)):
                        frame_num = frames_processed + i
                        logger.debug(f"Getting frame {frame_num}")
                        
                        # Use timeout for Smart Cache
                        if "SmartCache" in strategy_name or "Strategy2" in strategy_name:
                            frame_data = self.get_frame_with_timeout(strategy, frame_num, timeout=5.0)
                            if frame_data is None:
                                logger.error(f"Failed to get frame {frame_num}, aborting test")
                                raise TimeoutError(f"Timeout getting frame {frame_num}")
                        else:
                            frame_data = strategy.get_next_frame_pingpong()
                        
                        logger.debug(f"Got frame {frame_num}: {self.get_frame_info(frame_data)}")
                        frames.append(frame_data)
                
                if time_to_first is None:
                    time_to_first = time.time() - start_time
                
                # Simulate processing
                with self.profiler.timer("processing_pipeline"):
                    self.simulate_preprocessing()
                    self.simulate_inference_time(len(frames))
                    self.simulate_postprocessing()
                
                frames_processed += len(frames)
                
                # Update memory peak
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_peak = max(memory_peak, current_memory)
                
                # Progress
                if frames_processed % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frames_processed / elapsed
                    print(f"  Processed {frames_processed}/{total_frames_needed} frames "
                          f"({fps:.1f} FPS)")
        
        total_time = time.time() - start_time
        
        result = ProcessingResult(
            frames_processed=frames_processed,
            total_time=total_time,
            time_to_first_frame=time_to_first,
            average_latency=total_time / frames_processed * 1000,  # ms
            memory_peak_mb=memory_peak - memory_start,
            fps_achieved=frames_processed / total_time
        )
        
        logger.info(f"Batch processing complete: {result.frames_processed} frames in {result.total_time:.2f}s")
        print(f"\nBatch Processing Results:")
        print(f"  Total frames: {result.frames_processed}")
        print(f"  Total time: {result.total_time:.2f}s")
        print(f"  Time to first frame: {result.time_to_first_frame*1000:.1f}ms")
        print(f"  Average FPS: {result.fps_achieved:.1f}")
        print(f"  Memory used: {result.memory_peak_mb:.1f}MB")
        
        return result

    def benchmark_streaming_scenario(self, strategy, 
                                   chunk_duration_seconds: float = 0.2,
                                   total_duration_seconds: float = 10.0) -> ProcessingResult:
        """
        Simulate streaming scenario:
        - Receive audio chunks periodically
        - Process corresponding frames with low latency
        - Measure latency for each chunk
        """
        logger.info(f"Starting streaming benchmark: {chunk_duration_seconds}s chunks, {total_duration_seconds}s total")
        print(f"\nStreaming Scenario (chunks: {chunk_duration_seconds}s)")
        print("-" * 50)
        
        frames_per_chunk = int(chunk_duration_seconds * self.fps)
        total_chunks = int(total_duration_seconds / chunk_duration_seconds)
        
        chunk_latencies = []
        time_to_first = None
        start_time = time.time()
        frames_processed = 0
        
        # Track memory
        process = psutil.Process()
        memory_start = process.memory_info().rss / 1024 / 1024
        memory_peak = memory_start
        
        strategy_name = type(strategy).__name__
        
        # Simulate streaming
        for chunk_id in range(total_chunks):
            logger.debug(f"Processing chunk {chunk_id}/{total_chunks}")
            # Simulate receiving audio chunk
            audio_chunk = AudioChunk(
                duration_seconds=chunk_duration_seconds,
                chunk_id=chunk_id,
                data=np.random.randn(int(16000 * chunk_duration_seconds))  # 16kHz audio
            )
            
            chunk_start = time.time()
            
            # Process frames for this chunk
            with self.profiler.timer(f"process_chunk_{chunk_id}"):
                chunk_frames = []
                
                # Get frames
                for frame_idx in range(frames_per_chunk):
                    logger.debug(f"Getting frame {frame_idx} for chunk {chunk_id}")
                    
                    if "SmartCache" in strategy_name or "Strategy2" in strategy_name:
                        frame_data = self.get_frame_with_timeout(
                            strategy, 
                            frames_processed + frame_idx, 
                            timeout=5.0
                        )
                        if frame_data is None:
                            logger.error(f"Failed to get frame {frame_idx} for chunk {chunk_id}")
                            raise TimeoutError(f"Timeout in chunk {chunk_id}")
                    else:
                        frame_data = strategy.get_next_frame_pingpong()
                    
                    logger.debug(f"Got frame {frame_idx} for chunk {chunk_id}: {self.get_frame_info(frame_data)}")
                    chunk_frames.append(frame_data)
                
                # Simulate processing
                self.simulate_preprocessing()
                self.simulate_inference_time(len(chunk_frames))
                self.simulate_postprocessing()
            
            chunk_time = time.time() - chunk_start
            chunk_latencies.append(chunk_time * 1000)  # ms
            
            if time_to_first is None:
                time_to_first = chunk_time
            
            frames_processed += len(chunk_frames)
            
            # Update memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_peak = max(memory_peak, current_memory)
            
            # Simulate real-time constraint
            sleep_time = max(0, chunk_duration_seconds - chunk_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(f"Chunk {chunk_id} took {chunk_time*1000:.1f}ms (>{chunk_duration_seconds*1000:.1f}ms)")
                print(f"  WARNING: Chunk {chunk_id} took {chunk_time*1000:.1f}ms "
                      f"(>{chunk_duration_seconds*1000:.1f}ms)")
        
        total_time = time.time() - start_time
        
        result = ProcessingResult(
            frames_processed=frames_processed,
            total_time=total_time,
            time_to_first_frame=time_to_first,
            average_latency=np.mean(chunk_latencies),
            memory_peak_mb=memory_peak - memory_start,
            fps_achieved=frames_processed / total_time
        )
        
        logger.info(f"Streaming complete: {result.frames_processed} frames in {result.total_time:.2f}s")
        print(f"\nStreaming Results:")
        print(f"  Chunks processed: {total_chunks}")
        print(f"  Total frames: {result.frames_processed}")
        print(f"  Time to first chunk: {result.time_to_first_frame*1000:.1f}ms")
        print(f"  Average chunk latency: {result.average_latency:.1f}ms")
        print(f"  P95 latency: {np.percentile(chunk_latencies, 95):.1f}ms")
        print(f"  P99 latency: {np.percentile(chunk_latencies, 99):.1f}ms")
        print(f"  Real-time factor: {total_duration_seconds/result.total_time:.2f}x")
        
        return result

def compare_strategies(video_path: str, test_duration: float = 30.0):
    """Compare all strategies with real scenarios"""
    
    benchmark = VideoStrategyBenchmark(video_path)
    
    strategies = [
        ("All Frames in Memory", Strategy1_AllFramesInMemory(video_path)),
        ("Compressed Memory (80%)", Strategy4_CompressedMemory(video_path, quality=80)),
        ("Compressed Memory (60%)", Strategy4_CompressedMemory(video_path, quality=60)),
    ]
    
    results = []
    
    for name, strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        logger.info(f"Starting test for strategy: {name}")
        
        # Initialize/preload
        logger.info(f"Initializing strategy: {name}")
        init_start = time.time()
        
        try:
            # Special handling for Smart Cache
            if "Smart Cache" in name:
                logger.debug(f"{name}: Initializing Smart Cache - starting async preload without waiting")
                if hasattr(strategy, 'preload_async'):
                    preload_thread = strategy.preload_async()
                    logger.debug(f"{name}: Preload thread started, not waiting for completion")
                    # Don't join the thread - let it run in background
                
                # Give it more time to load initial frames
                logger.debug(f"{name}: Waiting 2s for initial frames to load...")
                time.sleep(2.0)
                logger.debug(f"{name}: Proceeding with tests while preload continues in background")
                
            else:
                # Other strategies - wait for full initialization
                if hasattr(strategy, 'preload_async'):
                    logger.debug(f"{name}: Starting preload_async")
                    thread = strategy.preload_async()
                    if isinstance(thread, threading.Thread):
                        logger.debug(f"{name}: Waiting for preload thread to complete")
                        thread.join(timeout=30.0)  # Add timeout to prevent infinite wait
                        if thread.is_alive():
                            logger.error(f"{name}: Preload thread timed out after 30s")
                            raise TimeoutError("Preload thread timed out")
                        logger.debug(f"{name}: Preload thread completed")
                elif hasattr(strategy, 'load_all_frames'):
                    logger.debug(f"{name}: Loading all frames")
                    strategy.load_all_frames()
                    logger.debug(f"{name}: All frames loaded")
                elif hasattr(strategy, 'load_and_compress_frames'):
                    logger.debug(f"{name}: Loading and compressing frames")
                    strategy.load_and_compress_frames()
                    logger.debug(f"{name}: Frames loaded and compressed")
            
            init_time = time.time() - init_start
            logger.info(f"{name}: Initialization complete in {init_time:.2f}s")
            print(f"Initialization time: {init_time:.2f}s")
            
            # Start prefetch for smart cache
            if hasattr(strategy, 'start_prefetch'):
                logger.info(f"{name}: Starting prefetch thread")
                strategy.start_prefetch()
                logger.debug(f"{name}: Prefetch thread started")
                # Give prefetch thread a moment to start
                time.sleep(0.1)
            
            # Test batch processing
            logger.info(f"{name}: Starting batch processing test")
            batch_result = benchmark.benchmark_batch_processing(strategy, test_duration)
            
            # Reset for streaming test
            logger.debug(f"{name}: Resetting for streaming test")
            strategy.current_idx = 0
            strategy.direction = 1
            
            # Test streaming
            logger.info(f"{name}: Starting streaming test")
            stream_result = benchmark.benchmark_streaming_scenario(
                strategy, 
                chunk_duration_seconds=0.2,  # 200ms chunks
                total_duration_seconds=min(10.0, test_duration)
            )
            
            results.append({
                'name': name,
                'init_time': init_time,
                'batch': batch_result,
                'stream': stream_result
            })
            
            # Cleanup
            if hasattr(strategy, 'stop_prefetch'):
                logger.info(f"{name}: Stopping prefetch thread")
                strategy.stop_prefetch()
                logger.debug(f"{name}: Prefetch thread stopped")
                
        except Exception as e:
            logger.error(f"{name}: Test failed with error: {str(e)}", exc_info=True)
            print(f"ERROR: {name} test failed - {str(e)}")
            continue
    
    # Summary table
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Strategy':<25} {'Init(s)':<8} {'Batch FPS':<12} {'Stream Latency(ms)':<20} {'Memory(MB)':<12}")
    print(f"{'-'*25} {'-'*8} {'-'*12} {'-'*20} {'-'*12}")
    
    for r in results:
        print(f"{r['name']:<25} {r['init_time']:<8.1f} {r['batch'].fps_achieved:<12.1f} "
              f"{r['stream'].average_latency:<20.1f} {max(r['batch'].memory_peak_mb, r['stream'].memory_peak_mb):<12.1f}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if results:
        # Find best for batch
        best_batch = max(results, key=lambda x: x['batch'].fps_achieved)
        print(f"\nBest for batch processing: {best_batch['name']}")
        print(f"  - {best_batch['batch'].fps_achieved:.1f} FPS")
        print(f"  - {best_batch['batch'].time_to_first_frame*1000:.1f}ms to first frame")
        
        # Find best for streaming
        best_stream = min(results, key=lambda x: x['stream'].average_latency)
        print(f"\nBest for streaming: {best_stream['name']}")
        print(f"  - {best_stream['stream'].average_latency:.1f}ms average latency")
        print(f"  - {best_stream['stream'].time_to_first_frame*1000:.1f}ms to first frame")

def main():
    parser = argparse.ArgumentParser(description='Benchmark video loading strategies')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--duration', type=float, default=30.0, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    logger.info(f"Starting benchmarks with video: {args.video}, duration: {args.duration}s")
    compare_strategies(args.video, args.duration)

if __name__ == "__main__":
    main()