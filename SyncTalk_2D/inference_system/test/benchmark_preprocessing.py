# inference_system/test/benchmark_preprocessing.py
import os
import sys
import time
import cv2
import torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from inference_system.core.preprocessing_gpu import GPUPreprocessor, HybridPreprocessor

def benchmark_preprocessing():
    """Benchmark preprocessing optimizations"""
    
    print("="*60)
    print("PREPROCESSING BENCHMARK")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available, skipping GPU benchmarks")
        return
    
    # Create test data
    test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    bbox = (600, 200, 900, 500)  # 300x300 face
    
    num_iterations = 500
    warmup = 50
    
    # Test 1: Original CPU preprocessing
    print("\n1. BASELINE: CPU preprocessing with OpenCV")
    print("-"*40)
    
    # Warmup
    for _ in range(warmup):
        xmin, ymin, xmax, ymax = bbox
        crop_img = test_img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        img_real_ex = crop_img[4:324, 4:324].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex_ori,(5,5,310,305),(0,0,0),-1)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
    start = time.time()
    for _ in range(num_iterations):
        xmin, ymin, xmax, ymax = bbox
        crop_img = test_img[ymin:ymax, xmin:xmax]
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        img_real_ex = crop_img[4:324, 4:324].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex_ori,(5,5,310,305),(0,0,0),-1)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
    cpu_time = time.time() - start
    cpu_per_frame = cpu_time / num_iterations * 1000
    
    print(f"Total time: {cpu_time:.3f}s")
    print(f"Per frame: {cpu_per_frame:.3f}ms")
    
    # Test 2: GPU preprocessing
    print("\n2. GPU PREPROCESSING: Fully GPU-based")
    print("-"*40)
    
    gpu_preprocessor = GPUPreprocessor(device)
    
    # Convert image to GPU tensor once
    img_tensor = torch.from_numpy(test_img).permute(2, 0, 1).float().to(device) / 255.0
    
    # Warmup
    for _ in range(warmup):
        real_img, masked_img, orig_size = gpu_preprocessor.preprocess_face_gpu(img_tensor, bbox)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        real_img, masked_img, orig_size = gpu_preprocessor.preprocess_face_gpu(img_tensor, bbox)
    
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    gpu_per_frame = gpu_time / num_iterations * 1000
    
    print(f"Total time: {gpu_time:.3f}s")
    print(f"Per frame: {gpu_per_frame:.3f}ms")
    print(f"Speedup: {cpu_per_frame/gpu_per_frame:.2f}x")
    
    # Test 3: Hybrid preprocessing
    print("\n3. HYBRID PREPROCESSING: CPU crop + GPU process")
    print("-"*40)
    
    hybrid_preprocessor = HybridPreprocessor(device)
    
    # Warmup
    for _ in range(warmup):
        real_img, masked_img, orig_size = hybrid_preprocessor.preprocess_frame(test_img, bbox)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        real_img, masked_img, orig_size = hybrid_preprocessor.preprocess_frame(test_img, bbox)
    
    torch.cuda.synchronize()
    hybrid_time = time.time() - start
    hybrid_per_frame = hybrid_time / num_iterations * 1000
    
    print(f"Total time: {hybrid_time:.3f}s")
    print(f"Per frame: {hybrid_per_frame:.3f}ms")
    print(f"Speedup vs CPU: {cpu_per_frame/hybrid_per_frame:.2f}x")
    
    # Test 4: Batch GPU preprocessing
    print("\n4. BATCH GPU PREPROCESSING: 4 frames at once")
    print("-"*40)
    
    batch_size = 4
    batch_imgs = img_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    batch_bboxes = [bbox] * batch_size
    
    # Warmup
    for _ in range(warmup // batch_size):
        real_imgs, masked_imgs, orig_sizes = gpu_preprocessor.preprocess_batch_gpu(batch_imgs, batch_bboxes)
    
    torch.cuda.synchronize()
    start = time.time()
    
    num_batches = num_iterations // batch_size
    for _ in range(num_batches):
        real_imgs, masked_imgs, orig_sizes = gpu_preprocessor.preprocess_batch_gpu(batch_imgs, batch_bboxes)
    
    torch.cuda.synchronize()
    batch_time = time.time() - start
    batch_per_frame = batch_time / (num_batches * batch_size) * 1000
    
    print(f"Total time: {batch_time:.3f}s")
    print(f"Per frame: {batch_per_frame:.3f}ms")
    print(f"Speedup vs CPU: {cpu_per_frame/batch_per_frame:.2f}x")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"CPU baseline: {cpu_per_frame:.3f} ms/frame")
    print(f"GPU processing: {gpu_per_frame:.3f} ms/frame ({cpu_per_frame/gpu_per_frame:.1f}x)")
    print(f"Hybrid (recommended): {hybrid_per_frame:.3f} ms/frame ({cpu_per_frame/hybrid_per_frame:.1f}x)")
    print(f"Batch GPU: {batch_per_frame:.3f} ms/frame ({cpu_per_frame/batch_per_frame:.1f}x)")

if __name__ == "__main__":
    benchmark_preprocessing()