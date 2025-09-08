# inference_system/test/test_frame_manager.py
import numpy as np
from inference_system.core.frame_manager import FrameManager
from inference_system.utils.profiler import Benchmark
import time

def test_frame_manager():
    """Test frame manager functionality and performance"""
    
    # Initialize frame manager
    print("Initializing Frame Manager...")
    fm = FrameManager(
        dataset_dir="./dataset",  # Relative to SyncTalk_2D
        name="alin",
        start_frame=0,
        use_parsing=False,
        preload=True,
        num_workers=4
    )
    
    # Print statistics
    fm.print_stats()
    
    # Test ping-pong functionality
    print("\n\nTesting ping-pong sequence:")
    positions = []
    for i in range(20):  # Get 20 frames to see the pattern
        frame = fm.get_next_frame()
        positions.append(frame.frame_index)
    
    print(f"Frame sequence: {positions}")
    
    # Benchmark frame access
    print("\n\nBenchmarking frame access speed:")
    
    def get_single_frame():
        return fm.get_next_frame()
    
    def get_batch_frames():
        return fm.get_frame_sequence(8)
    
    # Single frame access
    single_results = Benchmark.measure_throughput(
        get_single_frame, 
        iterations=1000, 
        warmup=100
    )
    
    # Batch frame access
    batch_results = Benchmark.measure_throughput(
        get_batch_frames,
        iterations=100,
        warmup=10
    )
    
    print(f"\nSingle frame access:")
    print(f"  Throughput: {single_results['throughput']:.0f} frames/sec")
    print(f"  Latency: {single_results['avg_time_ms']:.3f} ms/frame")
    
    print(f"\nBatch frame access (8 frames):")
    print(f"  Throughput: {batch_results['throughput'] * 8:.0f} frames/sec")
    print(f"  Latency: {batch_results['avg_time_ms'] / 8:.3f} ms/frame")
    
    # Test memory efficiency
    print("\n\nMemory efficiency test:")
    fm.profiler.print_summary()
    
    # Test crop region calculation
    print("\n\nTesting crop region calculation:")
    frame = fm.get_next_frame()
    xmin, ymin, xmax, ymax = fm.get_crop_region(frame.landmarks)
    print(f"  Crop region: ({xmin}, {ymin}) to ({xmax}, {ymax})")
    print(f"  Crop dimensions: {xmax-xmin} x {ymax-ymin}")

def test_frame_manager_patterns():
    """Test that ping-pong pattern works correctly"""
    # Create a mock frame manager with known frame count
    from inference_system.core.frame_manager import FrameData
    
    fm = FrameManager(
        dataset_dir="./dataset",
        name="alin",
        preload=False  # Don't load for this test
    )
    
    # Manually set frame count for testing
    fm.total_frames = 10
    
    # Create dummy frames
    for i in range(10):
        fm.frames[i] = FrameData(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            landmarks=np.zeros((68, 2), dtype=np.int32),
            frame_index=i,
            original_index=i
        )
    
    # Test pattern
    pattern = []
    for _ in range(30):  # Get enough to see full cycle
        frame = fm.get_next_frame()
        pattern.append(frame.frame_index)
    
    expected = [0,1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,0,0,1,2,3,4,5,6,7,8,9]
    
    print("\n\nPing-pong pattern test:")
    print(f"Pattern matches expected: {pattern == expected}")
    if pattern != expected:
        print(f"Expected: {expected}")
        print(f"Got: {pattern}")

if __name__ == "__main__":
    # Run tests
    test_frame_manager()
    test_frame_manager_patterns()