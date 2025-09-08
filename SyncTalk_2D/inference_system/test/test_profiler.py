# inference_system/test/test_profiler.py
import sys
sys.path.append('..')

import time
import torch
import numpy as np
from utils.profiler import PerformanceProfiler, Benchmark

def test_profiler():
    """Test the performance profiler"""
    profiler = PerformanceProfiler("test_run")
    
    # Test CPU operation
    with profiler.timer("cpu_operation"):
        # Simulate CPU work
        arr = np.random.randn(1000, 1000)
        result = np.matmul(arr, arr.T)
        time.sleep(0.1)
    
    # Test GPU operation if available
    if torch.cuda.is_available():
        with profiler.timer("gpu_operation"):
            # Simulate GPU work
            tensor = torch.randn(1000, 1000).cuda()
            result = torch.matmul(tensor, tensor.T)
            torch.cuda.synchronize()
    
    # Test nested operations
    with profiler.timer("parent_operation"):
        time.sleep(0.05)
        with profiler.timer("child_operation"):
            time.sleep(0.03)
    
    # Print results
    profiler.print_summary()
    
    # Save report
    profiler.save_report("test_profile_report.json")
    print("\nReport saved to test_profile_report.json")

def test_benchmark():
    """Test the benchmark utility"""
    def sample_function():
        # Simulate some work
        arr = np.random.randn(100, 100)
        return np.sum(arr)
    
    results = Benchmark.measure_throughput(sample_function, iterations=50, warmup=5)
    
    print("\nBenchmark Results:")
    print(f"Throughput: {results['throughput']:.2f} ops/sec")
    print(f"Average time: {results['avg_time_ms']:.2f} ms/op")

if __name__ == "__main__":
    print("Testing Performance Profiler...")
    test_profiler()
    
    print("\n" + "="*60)
    print("Testing Benchmark Utility...")
    test_benchmark()