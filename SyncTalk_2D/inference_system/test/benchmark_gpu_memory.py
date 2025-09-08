# inference_system/test/benchmark_gpu_memory.py
import os
import sys
import time
import torch
import numpy as np
import gc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from inference_system.core.gpu_memory_manager import GPUMemoryManager, BatchGPUMemoryManager

def benchmark_gpu_memory_transfer():
    """Benchmark GPU memory transfer optimizations"""
    
    print("="*60)
    print("GPU MEMORY TRANSFER BENCHMARK")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available, skipping GPU benchmarks")
        return
    
    # Test parameters
    num_frames = 500
    warmup_frames = 50
    
    # Create test data
    audio_feat_np = np.random.randn(768).astype(np.float32)
    img_real_np = np.random.randint(0, 255, (3, 320, 320)).astype(np.float32)
    img_masked_np = np.random.randint(0, 255, (3, 320, 320)).astype(np.float32)
    
    # Test 1: Baseline - Create new tensors every frame
    print("\n1. BASELINE: New tensor allocation per frame")
    print("-"*40)
    
    # Warmup
    for _ in range(warmup_frames):
        audio_tensor = torch.from_numpy(audio_feat_np.reshape(32, 16, 16)).unsqueeze(0).cuda()
        img_tensor = torch.cat([
            torch.from_numpy(img_real_np / 255.0),
            torch.from_numpy(img_masked_np / 255.0)
        ], dim=0).unsqueeze(0).cuda()
        del audio_tensor, img_tensor
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_frames):
        # Simulate current approach
        audio_tensor = torch.from_numpy(audio_feat_np.reshape(32, 16, 16)).unsqueeze(0).cuda()
        img_tensor = torch.cat([
            torch.from_numpy(img_real_np / 255.0),
            torch.from_numpy(img_masked_np / 255.0)
        ], dim=0).unsqueeze(0).cuda()
        
        # Simulate some computation
        _ = audio_tensor.sum()
        _ = img_tensor.sum()
        
        del audio_tensor, img_tensor
    
    torch.cuda.synchronize()
    baseline_time = time.time() - start
    baseline_per_frame = baseline_time / num_frames * 1000
    
    print(f"Total time: {baseline_time:.3f}s")
    print(f"Per frame: {baseline_per_frame:.3f}ms")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    # Test 2: Optimized - Pre-allocated GPU memory
    print("\n2. OPTIMIZED: Pre-allocated GPU memory")
    print("-"*40)
    
    manager = GPUMemoryManager(device, mode="ave")
    stats = manager.get_memory_stats()
    print(f"Pre-allocated memory: {stats['total']['mb']:.2f} MB")
    
    # Warmup
    for _ in range(warmup_frames):
        audio_tensor = manager.prepare_audio_features(audio_feat_np)
        img_tensor = manager.prepare_image_tensors(img_real_np, img_masked_np)
        _ = audio_tensor.sum()
        _ = img_tensor.sum()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_frames):
        audio_tensor = manager.prepare_audio_features(audio_feat_np)
        img_tensor = manager.prepare_image_tensors(img_real_np, img_masked_np)
        
        # Simulate computation
        _ = audio_tensor.sum()
        _ = img_tensor.sum()
    
    torch.cuda.synchronize()
    optimized_time = time.time() - start
    optimized_per_frame = optimized_time / num_frames * 1000
    
    print(f"Total time: {optimized_time:.3f}s")
    print(f"Per frame: {optimized_per_frame:.3f}ms")
    print(f"Speedup: {baseline_per_frame/optimized_per_frame:.2f}x")
    
    # Test 3: Batch processing
    print("\n3. BATCH PROCESSING: Process 4 frames at once")
    print("-"*40)
    
    batch_size = 4
    batch_manager = BatchGPUMemoryManager(device, mode="ave", batch_size=batch_size)
    
    # Create batch test data
    batch_audio = [audio_feat_np.copy() for _ in range(batch_size)]
    batch_img_real = [img_real_np.copy() for _ in range(batch_size)]
    batch_img_masked = [img_masked_np.copy() for _ in range(batch_size)]
    
    num_batches = num_frames // batch_size
    
    # Warmup
    for _ in range(warmup_frames // batch_size):
        for i in range(batch_size):
            batch_manager.add_to_batch(batch_audio[i], batch_img_real[i], batch_img_masked[i], i)
        audio_batch, img_batch, _ = batch_manager.prepare_batch()
        _ = audio_batch.sum()
        _ = img_batch.sum()
    
    torch.cuda.synchronize()
    start = time.time()
    
    for batch_idx in range(num_batches):
        # Add frames to batch
        for i in range(batch_size):
            batch_manager.add_to_batch(batch_audio[i], batch_img_real[i], batch_img_masked[i], i)
        
        # Process batch
        audio_batch, img_batch, indices = batch_manager.prepare_batch()
        
        # Simulate batch computation
        _ = audio_batch.sum()
        _ = img_batch.sum()
    
    torch.cuda.synchronize()
    batch_time = time.time() - start
    batch_per_frame = batch_time / (num_batches * batch_size) * 1000
    
    print(f"Total time: {batch_time:.3f}s")
    print(f"Per frame: {batch_per_frame:.3f}ms")
    print(f"Speedup vs baseline: {baseline_per_frame/batch_per_frame:.2f}x")
    print(f"Speedup vs optimized: {optimized_per_frame/batch_per_frame:.2f}x")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline (new tensors): {baseline_per_frame:.3f} ms/frame")
    print(f"Optimized (pre-allocated): {optimized_per_frame:.3f} ms/frame ({baseline_per_frame/optimized_per_frame:.1f}x)")
    print(f"Batch processing: {batch_per_frame:.3f} ms/frame ({baseline_per_frame/batch_per_frame:.1f}x)")
    print(f"\nEstimated inference speedup: {(baseline_per_frame-batch_per_frame)*num_frames/1000:.1f}s for {num_frames} frames")

if __name__ == "__main__":
    benchmark_gpu_memory_transfer()