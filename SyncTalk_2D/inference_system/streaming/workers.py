# inference_system/streaming/workers.py
"""
This module contains the shared worker thread functions for the streaming pipeline.
These workers are designed to be used by different wrapper architectures.
"""

import queue
import time
import torch
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import autocast

from ..utils.profiler import PerformanceProfiler

# --- Pre-processing ---

def _process_and_send_batch(batch_data, q_out, profiler):
    """Helper to process a pre-processing batch and send it to the next queue."""
    try:
        with profiler.timer(f"Streaming Preprocess ({len(batch_data)} frames)"):
            from ..inference_video import _process_batch_for_gpu
            b_real, b_masked, b_canvases, b_crops = _process_batch_for_gpu(batch_data)
        
        # Try to put with timeout to avoid infinite blocking
        try:
            q_out.put((batch_data, b_real, b_masked, b_canvases, b_crops), timeout=1.0)
            print(f"[PreProc] Sent batch of {len(batch_data)} frames to GPU queue")
        except queue.Full:
            print(f"[PreProc] ERROR: GPU queue is full! Cannot send batch of {len(batch_data)} frames")
            raise
    except Exception as e:
        print(f"[PreProc] ERROR in _process_and_send_batch: {e}")
        import traceback
        traceback.print_exc()
        raise

def streaming_cpu_pre_processing_worker(q_in: queue.Queue, q_out: queue.Queue, 
                                       batch_size: int, profiler: PerformanceProfiler):
    """
    CPU worker that performs pre-processing.
    It processes frames one-by-one and sends them to the shared GPU queue,
    letting the GPU worker handle the intelligent batching. This avoids
    double-batching issues and works with any batch_size.
    """
    worker_name = threading.current_thread().name
    print(f"[{worker_name}] Starting 1-to-1 pre-processing worker")
    processed_items = 0
    
    while True:
        try:
            item = q_in.get()  # Block until an item is available
            
            if item is None:
                print(f"[{worker_name}] Received stop signal")
                # Per-stream workers should not forward the stop signal to the shared queue
                if "PerStream" not in worker_name:
                    print(f"[{worker_name}] Forwarding stop signal to next queue")
                    q_out.put(None)
                else:
                    print(f"[{worker_name}] NOT forwarding stop signal (shared queue)")
                break
            
            # Process a single item and send it as a batch of 1
            try:
                # _process_and_send_batch expects a list
                _process_and_send_batch([item], q_out, profiler)
                processed_items += 1
                if processed_items <= 5 or processed_items % 100 == 0:
                    print(f"[{worker_name}] Processed {processed_items} items total")
            except Exception as e:
                print(f"[{worker_name}] Error processing item: {e}")
                import traceback
                traceback.print_exc()
                # Continue to the next item
                continue
                
        except Exception as e:
            print(f"[{worker_name}] Unexpected error in main loop: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[{worker_name}] Stopped after processing {processed_items} items")

# --- GPU Processing ---

def streaming_gpu_worker(q_in: queue.Queue, q_out: queue.Queue, device, net, mode, profiler, batch_size: int):
    """
    GPU worker that intelligently re-batches items from multiple streams for
    efficient inference. It waits to accumulate a full batch or for a short timeout.
    """
    try:
        print(f"[GPU Worker] Starting with batch_size={batch_size}, device={device}, mode={mode}")
        print(f"[GPU Worker] Network type: {type(net)}")
        print(f"[GPU Worker] Profiler type: {type(profiler)}")
        
        # Verify CUDA is available if using GPU
        if str(device) != 'cpu':
            if not torch.cuda.is_available():
                print(f"[GPU Worker] ERROR: CUDA not available but device is {device}")
                return
            print(f"[GPU Worker] CUDA is available: {torch.cuda.get_device_name()}")
        
        # Test the profiler to make sure it works
        try:
            with profiler.timer("GPU Worker Startup Test"):
                time.sleep(0.001)
            print(f"[GPU Worker] Profiler test successful")
        except Exception as e:
            print(f"[GPU Worker] WARNING: Profiler test failed: {e}")
        
        processed_batches = 0
        print(f"[GPU Worker] Entering main processing loop...")
        
        while True:
            batch_items = []
            
            # Wait for first item with timeout
            try:
                first_item = q_in.get(timeout=1.0)
                if first_item is None:
                    print("[GPU Worker] Received stop signal")
                    q_out.put(None)
                    break
                print(f"[GPU Worker] Received first item for batch (type: {type(first_item)})")
                batch_items.append(first_item)
            except queue.Empty:
                # No items available, continue waiting
                if processed_batches == 0:
                    print(f"[GPU Worker] Still waiting for first batch... (queue size: {q_in.qsize()})")
                continue
        
            # Try to accumulate more items for better GPU utilization
            # But don't wait too long - process what we have
            start_accumulate = time.time()
            
            # Adjust accumulation time based on queue depth (indicates load)
            queue_size = q_in.qsize() if hasattr(q_in, 'qsize') else 0
            if queue_size > 8:
                # High load - wait longer to batch more efficiently
                MAX_ACCUMULATE_TIME = 0.005  # 5ms for better batching
            elif queue_size > 4:
                # Medium load
                MAX_ACCUMULATE_TIME = 0.003  # 3ms
            else:
                # Low load - prioritize latency
                MAX_ACCUMULATE_TIME = 0.002  # 2ms default
            
            while len(batch_items) < batch_size and (time.time() - start_accumulate) < MAX_ACCUMULATE_TIME:
                try:
                    item = q_in.get_nowait()
                    if item is None:
                        # Put it back for proper shutdown handling later
                        q_in.put(item)
                        break
                    batch_items.append(item)
                except queue.Empty:
                    # No more items immediately available
                    if len(batch_items) >= 1:
                        # Process what we have rather than waiting
                        break
                    time.sleep(0.0001)  # Brief sleep before checking again
        
            # Combine all batch items
            combined_batch_data = []
            combined_real = []
            combined_masked = []
            combined_canvases = []
            combined_crops = []
            
            for i, item in enumerate(batch_items):
                try:
                    if not isinstance(item, tuple) or len(item) != 5:
                        print(f"[GPU Worker] ERROR: Invalid item format at index {i}: {type(item)}, len={len(item) if hasattr(item, '__len__') else 'N/A'}")
                        continue
                    
                    b_data, b_real, b_masked, b_canvases, b_crops = item
                    combined_batch_data.extend(b_data)
                    combined_real.extend(b_real)
                    combined_masked.extend(b_masked)
                    combined_canvases.extend(b_canvases)
                    combined_crops.extend(b_crops)
                except Exception as e:
                    print(f"[GPU Worker] ERROR unpacking item {i}: {e}")
                    continue
            
            batch_size_actual = len(combined_batch_data)
            if batch_size_actual == 0:
                continue
        
            # Process the batch on GPU
            try:
                with profiler.timer(f"Streaming GPU Batch ({batch_size_actual} frames)", gpu_sync=True):
                    # Convert to tensors
                    real_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in combined_real]
                    masked_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in combined_masked]
                    audio_tensors = [torch.from_numpy(d.audio_feat) for d in combined_batch_data]
                    
                    # Stack and move to GPU
                    real_batch = torch.stack(real_tensors).to(device, non_blocking=True).half() / 255.0
                    masked_batch = torch.stack(masked_tensors).to(device, non_blocking=True).half() / 255.0
                    audio_batch = torch.stack(audio_tensors).to(device, non_blocking=True).half()
                    
                    if mode == "ave":
                        audio_batch = audio_batch.view(batch_size_actual, 32, 16, 16)
                    
                    # Concatenate real and masked images
                    img_for_net = torch.cat([real_batch, masked_batch], dim=1)
                    
                    # Run inference
                    with autocast():
                        with torch.no_grad():
                            pred_batch = net(img_for_net, audio_batch)
                    
                    # Convert back to numpy
                    pred_batch_np = pred_batch.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                    
                    # Send to output queue
                    q_out.put((combined_batch_data, pred_batch_np, combined_canvases, combined_crops))
                    
                    processed_batches += 1
                    if processed_batches <= 5 or processed_batches % 100 == 0:
                        print(f"[GPU Worker] Processed batch {processed_batches} ({batch_size_actual} frames)")
                        
            except Exception as e:
                print(f"💥 GPU worker error processing batch: {e}")
                import traceback
                traceback.print_exc()
                # Don't crash the worker, just skip this batch
                continue
    
        print(f"[GPU Worker] Stopped after processing {processed_batches} batches")
        
    except Exception as e:
        print(f"[GPU Worker] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("[GPU Worker] Worker thread crashed!")

# --- Post-processing ---

def streaming_cpu_post_processing_worker(q_in: queue.Queue, output_buffer: dict, 
                                       output_lock: threading.Lock, profiler: PerformanceProfiler,
                                       release_callback=None):
    """CPU worker that performs post-processing on GPU output."""
    worker_name = threading.current_thread().name
    print(f"[{worker_name}] Starting post-processing worker")
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="StreamPostProc") as executor:
        while True:
            try:
                item = q_in.get(timeout=0.01)
                if item is None:
                    print(f"[{worker_name}] Received stop signal")
                    break
                
                p_batch_data, p_pred_batch_np, p_canvases, p_orig_crops = item
                batch_size = len(p_batch_data)
                
                with profiler.timer(f"Streaming Postprocess ({batch_size} frames)"):
                    from ..inference_video import _post_process_single_frame
                    futures = []
                    
                    for j in range(batch_size):
                        future = executor.submit(
                            _post_process_single_frame,
                            p_batch_data[j], p_pred_batch_np[j], 
                            p_canvases[j], p_orig_crops[j]
                        )
                        futures.append(future)
                    
                    for j, future in enumerate(futures):
                        try:
                            frame_idx, final_frame = future.result()
                            data = p_batch_data[j]
                            
                            with output_lock:
                                # Get audio chunk from frame data if available
                                audio_chunk = getattr(data, 'audio_chunk', None)
                                output_buffer[data.frame_idx] = (
                                    final_frame, data.img_idx, data.frame_idx, audio_chunk
                                )
                                processed_count += 1
                                if release_callback:
                                    release_callback(data.frame_idx)
                            
                            if processed_count <= 5 or processed_count % 50 == 0:
                                print(f"[{worker_name}] Processed {processed_count} frames total")
                        except Exception as e:
                            print(f"[{worker_name}] Error processing single frame: {e}")
                            import traceback
                            traceback.print_exc()
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[{worker_name}] Error in main loop: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"[{worker_name}] Stopped after processing {processed_count} frames")
