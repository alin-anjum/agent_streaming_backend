# inference_system/test/benchmark_frame_manager.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import psutil
import argparse
import numpy as np
from typing import Optional, Dict, List
from inference_system.core.video_frame_manager import AllFramesMemory

class FrameManagerBenchmark:
    """Benchmark different frame loading strategies"""
    
    def __init__(self, image_dir: str = None, video_path: str = None):
        self.image_dir = image_dir
        self.video_path = video_path
        self.process = psutil.Process()
    
    def get_memory_usage_mb(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def simulate_realistic_inference(self, manager, num_frames=500):
        """Simulate realistic inference with timing from actual code"""
        frame_times = []
        
        print(f"\n[Realistic Inference Simulation] Processing {num_frames} frames...")
        
        for i in range(num_frames):
            # Time frame access
            frame_start = time.time()
            frame_data = manager.get_next_frame_pingpong()
            frame_access_time = time.time() - frame_start
            
            if frame_data:
                frame_times.append(frame_access_time * 1000)  # ms
                
                # Simulate actual processing times from the inference code
                # Preprocessing: ~0.85ms
                time.sleep(0.00085)
                # Model inference: ~6.86ms
                time.sleep(0.00686)
                # Post-processing: ~4.28ms
                time.sleep(0.00428)
            
            # Progress update
            if i % 100 == 0 and i > 0:
                avg_access = np.mean(frame_times[-100:])
                print(f"  Processed {i} frames, avg access time: {avg_access:.3f}ms")
        
        return {
            'frame_times_ms': frame_times,
            'avg_access_ms': np.mean(frame_times),
            'p95_access_ms': np.percentile(frame_times, 95),
            'p99_access_ms': np.percentile(frame_times, 99)
        }
    
    def benchmark_baseline_images(self):
        """Benchmark the baseline approach - load from individual image files"""
        print("\n" + "="*80)
        print("BASELINE: Load from Individual Image Files")
        print("="*80)
        
        # Track memory before
        memory_start = self.get_memory_usage_mb()
        
        # Create manager loading from images
        manager = AllFramesMemory(
            self.image_dir,
            enable_async=False,  # Blocking load
            from_images=True
        )
        
        # Time initialization
        init_start = time.time()
        manager.initialize()
        init_time = time.time() - init_start
        
        # Get stats
        stats = manager.get_stats()
        
        # Run realistic inference simulation
        inference_results = self.simulate_realistic_inference(manager, num_frames=500)
        
        # Final memory usage
        memory_after = self.get_memory_usage_mb()
        memory_used = memory_after - memory_start
        
        print(f"\nResults:")
        print(f"  Source: {self.image_dir} (image files)")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Time to load all frames: {init_time:.3f}s")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Avg frame access time: {inference_results['avg_access_ms']:.3f}ms")
        print(f"  P95 frame access time: {inference_results['p95_access_ms']:.3f}ms")
        
        manager.cleanup()
        
        return {
            'mode': 'Baseline (Image Files)',
            'init_time': init_time,
            'time_to_first_frame': stats['time_to_first_frame'],
            'memory_mb': memory_used,
            'inference_results': inference_results
        }
    
    def benchmark_video_all_frames(self):
        """Benchmark loading all frames from video"""
        print("\n" + "="*80)
        print("OPTIMIZATION 1: Load All Frames from Video")
        print("="*80)
        
        # Track memory before
        memory_start = self.get_memory_usage_mb()
        
        # Create manager loading from video
        manager = AllFramesMemory(
            self.video_path,
            enable_async=True,  # Can start serving quickly
            from_images=False   # Load from video
        )
        
        # Time initialization
        init_start = time.time()
        manager.initialize()
        init_time = time.time() - init_start
        
        # Get stats
        stats = manager.get_stats()
        
        # Run realistic inference simulation
        inference_results = self.simulate_realistic_inference(manager, num_frames=500)
        
        # Wait for complete load if not done
        while not manager.load_complete:
            time.sleep(0.1)
        
        # Final memory usage
        memory_after = self.get_memory_usage_mb()
        memory_used = memory_after - memory_start
        
        print(f"\nResults:")
        print(f"  Source: {self.video_path} (video file)")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Time to first frame: {stats['time_to_first_frame']:.3f}s")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Avg frame access time: {inference_results['avg_access_ms']:.3f}ms")
        print(f"  P95 frame access time: {inference_results['p95_access_ms']:.3f}ms")
        
        manager.cleanup()
        
        return {
            'mode': 'Video (All Frames)',
            'init_time': init_time,
            'time_to_first_frame': stats['time_to_first_frame'],
            'memory_mb': memory_used,
            'inference_results': inference_results
        }
    
    def benchmark_video_memory_limited(self):
        """Benchmark loading frames from video with memory limit"""
        print("\n" + "="*80)
        print("OPTIMIZATION 2: Video with Memory Limit (750 frames)")
        print("="*80)
        
        # Track memory before
        memory_start = self.get_memory_usage_mb()
        
        # Create manager with memory limit
        manager = AllFramesMemory(
            self.video_path,
            enable_async=True,
            memory_limit_frames=750,
            load_batch_size=250,
            from_images=False
        )
        
        # Time initialization
        init_start = time.time()
        manager.initialize()
        init_time = time.time() - init_start
        
        # Get stats
        stats = manager.get_stats()
        
        # Run realistic inference simulation
        inference_results = self.simulate_realistic_inference(manager, num_frames=500)
        
        # Final memory usage
        memory_after = self.get_memory_usage_mb()
        memory_used = memory_after - memory_start
        
        final_stats = manager.get_stats()
        
        print(f"\nResults:")
        print(f"  Source: {self.video_path} (video file)")
        print(f"  Memory limit: 750 frames")
        print(f"  Frames in memory: {final_stats['frames_in_memory']}")
        print(f"  Time to first frame: {stats['time_to_first_frame']:.3f}s")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Avg frame access time: {inference_results['avg_access_ms']:.3f}ms")
        print(f"  P95 frame access time: {inference_results['p95_access_ms']:.3f}ms")
        
        manager.cleanup()
        
        return {
            'mode': 'Video (750 Frame Limit)',
            'init_time': init_time,
            'time_to_first_frame': stats['time_to_first_frame'],
            'memory_mb': memory_used,
            'inference_results': inference_results
        }
    
    def run_comparison(self):
        """Run full comparison between all approaches"""
        print("\n" + "#"*100)
        print("# FRAME LOADING STRATEGY COMPARISON")
        print("#"*100)
        print(f"\nImage Directory: {self.image_dir}")
        print(f"Video File: {self.video_path}")
        
        results = []
        
        # 1. Baseline - load from image files
        if self.image_dir and os.path.exists(self.image_dir):
            results.append(self.benchmark_baseline_images())
        
        # 2. Video - all frames
        if self.video_path and os.path.exists(self.video_path):
            results.append(self.benchmark_video_all_frames())
        
        # 3. Video - memory limited
        if self.video_path and os.path.exists(self.video_path):
            results.append(self.benchmark_video_memory_limited())
        
        # Summary comparison
        print("\n" + "="*100)
        print("COMPARISON SUMMARY")
        print("="*100)
        
        print(f"\n{'Mode':<30} {'Init Time (s)':<15} {'Memory (MB)':<15} {'Avg Access (ms)':<20} {'P95 Access (ms)':<20}")
        print("-"*100)
        
        for r in results:
            print(f"{r['mode']:<30} {r['init_time']:<15.3f} {r['memory_mb']:<15.1f} "
                  f"{r['inference_results']['avg_access_ms']:<20.3f} "
                  f"{r['inference_results']['p95_access_ms']:<20.3f}")
        
        # Calculate improvements vs baseline
        if len(results) >= 2:
            baseline = results[0]
            
            print("\n" + "="*100)
            print("IMPROVEMENTS vs BASELINE (Image Files)")
            print("="*100)
            
            for r in results[1:]:
                print(f"\n{r['mode']}:")
                
                # Disk space (rough estimate)
                image_size_mb = baseline['memory_mb']  # Rough estimate
                video_size_mb = image_size_mb * 0.1   # Video is ~10% of raw images
                print(f"  Disk space savings: ~{(1 - 0.1)*100:.0f}% (video vs images)")
                
                # Memory
                memory_ratio = r['memory_mb'] / baseline['memory_mb']
                print(f"  Memory usage: {memory_ratio:.1%} of baseline")
                
                # Speed
                init_improvement = baseline['time_to_first_frame'] / r['time_to_first_frame']
                print(f"  Time to first frame: {init_improvement:.1f}x faster")
                
                # Latency
                if r['inference_results']['avg_access_ms'] > baseline['inference_results']['avg_access_ms']:
                    access_penalty = r['inference_results']['avg_access_ms'] - baseline['inference_results']['avg_access_ms']
                    print(f"  Frame access penalty: +{access_penalty:.3f}ms")
                else:
                    access_improvement = baseline['inference_results']['avg_access_ms'] - r['inference_results']['avg_access_ms']
                    print(f"  Frame access improvement: -{access_improvement:.3f}ms")

def main():
    parser = argparse.ArgumentParser(description='Benchmark frame loading strategies')
    parser.add_argument('--images', type=str, help='Path to image directory (for baseline)')
    parser.add_argument('--video', type=str, help='Path to video file (for optimizations)')
    args = parser.parse_args()
    
    if not args.images and not args.video:
        print("Error: Provide at least --images or --video")
        sys.exit(1)
    
    benchmark = FrameManagerBenchmark(image_dir=args.images, video_path=args.video)
    benchmark.run_comparison()

if __name__ == "__main__":
    main()