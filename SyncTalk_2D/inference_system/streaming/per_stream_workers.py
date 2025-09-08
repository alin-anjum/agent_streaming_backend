# inference_system/streaming/per_stream_workers.py
"""
Wrapper to provide isolated pre/post-processing workers for each stream
while sharing a single, expensive GPU worker.

Architecture:
- 1 Shared GPU Worker: Efficiently utilizes the GPU.
- 1 Shared Demultiplexer: Routes GPU output to the correct stream.
- N Per-Stream Workers:
  - 1 Pre-processing worker per stream.
  - 1 Post-processing worker per stream.
  - Isolated queues for each stream's workers.
"""

import threading
import queue
import time
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field

from .workers import (
    streaming_cpu_pre_processing_worker,
    streaming_gpu_worker,
    streaming_cpu_post_processing_worker
)

@dataclass
class StreamWorkerContext:
    """Manages all resources for a single stream's isolated workers."""
    stream_id: str
    
    # Per-stream frame tracking
    current_frame_idx: int = 0
    
    # Queues for this stream's pipeline (increased size to prevent blocking)
    preprocess_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=500))
    postprocess_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=500))
    output_buffer: Dict[int, Any] = field(default_factory=dict)
    output_lock: threading.Lock = field(default_factory=threading.Lock)
    next_output_idx: int = 0
    
    # Worker threads
    preprocess_worker: threading.Thread = None
    postprocess_worker: threading.Thread = None
    
    # State
    is_active: bool = True
    
    def start_workers(self, shared_gpu_queue: queue.Queue, base_wrapper: Any):
        """Starts the pre- and post-processing workers for this stream."""
        
        # Pre-processing worker (Stream -> Shared GPU)
        self.preprocess_worker = threading.Thread(
            target=streaming_cpu_pre_processing_worker,
            args=(self.preprocess_queue, shared_gpu_queue, base_wrapper.batch_size, base_wrapper.profiler),
            daemon=True,
            name=f"PerStream-PreProc-{self.stream_id[:8]}"
        )
        
        # Post-processing worker (Demultiplexer -> Stream Output)
        self.postprocess_worker = threading.Thread(
            target=streaming_cpu_post_processing_worker,
            args=(self.postprocess_queue, self.output_buffer, self.output_lock, base_wrapper.profiler),
            daemon=True,
            name=f"PerStream-PostProc-{self.stream_id[:8]}"
        )
        
        self.preprocess_worker.start()
        self.postprocess_worker.start()
        print(f"✅ [{self.stream_id}] Started isolated pre/post-processing workers.")

    def stop_workers(self):
        """Gracefully stops the workers for this stream."""
        if not self.is_active:
            return
        
        print(f"⏹️ [{self.stream_id}] Stopping isolated workers...")
        self.is_active = False
        
        # Send sentinel to stop workers
        self.preprocess_queue.put(None)
        self.postprocess_queue.put(None)
        
        # Wait for them to terminate
        if self.preprocess_worker and self.preprocess_worker.is_alive():
            self.preprocess_worker.join(timeout=2.0)
        if self.postprocess_worker and self.postprocess_worker.is_alive():
            self.postprocess_worker.join(timeout=2.0)
            
        print(f"✅ [{self.stream_id}] Isolated workers stopped.")

class PerStreamWorkersWrapper:
    def __init__(self, base_wrapper):
        print(f"[PerStreamWorkersWrapper] Initializing for model: {base_wrapper.model_name}")
        self.base_wrapper = base_wrapper
        self.model_name = base_wrapper.model_name
        self.profiler = base_wrapper.profiler

        # --- Shared Resources --- (increased sizes to prevent blocking)
        self.gpu_queue = queue.Queue(maxsize=500)         # INCREASED 5x
        self.demultiplexer_queue = queue.Queue(maxsize=500) # INCREASED 5x
        print(f"[{self.model_name}] Created GPU and demultiplexer queues")
        
        # --- Stream Management ---
        self.stream_workers: Dict[str, StreamWorkerContext] = {}
        self.contexts_lock = threading.Lock()
        
        # --- Start Shared Workers ---
        self.gpu_running = False
        self.demultiplexer_running = False
        self.shared_workers = []
        print(f"[{self.model_name}] About to start shared workers...")
        self.start_shared_workers()
        
        print(f"✅ [{self.model_name}] PerStreamWorkersWrapper initialized. GPU and Demux workers are running.")

    def start_shared_workers(self):
        """Starts the single GPU worker and the demultiplexer."""
        if self.gpu_running:
            print(f"[{self.model_name}] Shared workers already running, skipping start")
            return
            
        print(f"[{self.model_name}] Starting shared GPU and demultiplexer workers...")
        
        self.gpu_running = True
        gpu_worker = threading.Thread(
            target=streaming_gpu_worker,
            args=(self.gpu_queue, self.demultiplexer_queue, self.base_wrapper.device, self.base_wrapper.net, self.base_wrapper.mode, self.profiler, self.base_wrapper.batch_size),
            daemon=True,
            name=f"Shared-GPU-{self.model_name}"
        )
        
        self.demultiplexer_running = True
        demultiplexer_worker = threading.Thread(
            target=self._demultiplexer_thread,
            daemon=True,
            name=f"Shared-Demux-{self.model_name}"
        )

        try:
            gpu_worker.start()
            print(f"[{self.model_name}] GPU worker started: {gpu_worker.name}")
        except Exception as e:
            print(f"❌ [{self.model_name}] Failed to start GPU worker: {e}")
            import traceback
            traceback.print_exc()
            
        try:
            demultiplexer_worker.start()
            print(f"[{self.model_name}] Demultiplexer worker started: {demultiplexer_worker.name}")
        except Exception as e:
            print(f"❌ [{self.model_name}] Failed to start demultiplexer: {e}")
            import traceback
            traceback.print_exc()
        
        self.shared_workers = [gpu_worker, demultiplexer_worker]
        
        # Verify workers are alive with more detailed checking
        time.sleep(0.2)  # Give threads more time to start
        
        print(f"[{self.model_name}] Checking worker status...")
        print(f"  GPU worker alive: {gpu_worker.is_alive()}")
        print(f"  GPU queue size: {self.gpu_queue.qsize()}")
        print(f"  Demux worker alive: {demultiplexer_worker.is_alive()}")
        print(f"  Demux queue size: {self.demultiplexer_queue.qsize()}")
        
        if not gpu_worker.is_alive():
            print(f"❌ [{self.model_name}] CRITICAL: GPU worker failed to start or crashed immediately!")
            print(f"   This will cause the 'GPU queue full' error you're seeing.")
            
        if not demultiplexer_worker.is_alive():
            print(f"❌ [{self.model_name}] CRITICAL: Demultiplexer failed to start!")
        
    def _demultiplexer_thread(self):
        """Routes GPU output to the correct stream's post-processing queue."""
        print("🚦 Demultiplexer thread started.")
        while self.demultiplexer_running:
            try:
                item = self.demultiplexer_queue.get(timeout=1.0)
                if item is None:
                    continue # Wait for the stop signal from stop_workers

                batch_data, pred_batch_np, canvases, orig_crops = item
                if not batch_data:
                    continue

                # Efficiently group frames by stream_id
                grouped_by_stream: Dict[str, Dict[str, list]] = {}

                for i, frame_data in enumerate(batch_data):
                    stream_id = getattr(frame_data, 'stream_id', None)
                    if not stream_id:
                        print(f"⚠️ Demultiplexer found frame without stream_id. Discarding.")
                        continue

                    if stream_id not in grouped_by_stream:
                        grouped_by_stream[stream_id] = {
                            "batch_data": [], "pred_batch_np": [], "canvases": [], "orig_crops": []
                        }
                    
                    grouped_by_stream[stream_id]["batch_data"].append(frame_data)
                    grouped_by_stream[stream_id]["pred_batch_np"].append(pred_batch_np[i])
                    grouped_by_stream[stream_id]["canvases"].append(canvases[i])
                    grouped_by_stream[stream_id]["orig_crops"].append(orig_crops[i])
                
                # Re-batch and route to correct stream queues
                with self.contexts_lock:
                    for stream_id, data in grouped_by_stream.items():
                        if stream_id in self.stream_workers:
                            context = self.stream_workers[stream_id]
                            
                            # Construct the new, smaller batch for this stream
                            new_batch = (
                                data["batch_data"],
                                np.array(data["pred_batch_np"]),
                                np.array(data["canvases"]),
                                # Use dtype=object to handle potentially different crop shapes
                                np.array(data["orig_crops"], dtype=object)
                            )
                            context.postprocess_queue.put(new_batch)
                        else:
                            print(f"🚦 Demultiplexer: Stream {stream_id} is gone. Discarding {len(data['batch_data'])} frames.")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"💥 Demultiplexer thread error: {e}")
                import traceback
                traceback.print_exc()
        print("🚦 Demultiplexer thread stopped.")

    def create_stream_context(self, stream_id: str, last_index: int = 0):
        with self.contexts_lock:
            # First, check if GPU worker is alive before creating new stream
            if self.shared_workers and len(self.shared_workers) >= 2:
                gpu_worker = self.shared_workers[0]
                if not gpu_worker.is_alive():
                    print(f"❌ GPU worker is dead before creating stream {stream_id}!")
                    print(f"   This is why the second stream fails. Attempting restart...")
                    self._restart_gpu_worker()
            else:
                print(f"⚠️ No shared workers found when creating stream {stream_id}")
                print(f"   Starting shared workers...")
                self.start_shared_workers()
            
            if stream_id in self.stream_workers:
                print(f"⚠️ Stream context {stream_id} already exists. Recreating.")
                self.stream_workers[stream_id].stop_workers()

            context = StreamWorkerContext(stream_id=stream_id, current_frame_idx=last_index)
            context.next_output_idx = 0 # Ensure we start from the beginning
            context.start_workers(self.gpu_queue, self.base_wrapper)
            self.stream_workers[stream_id] = context
            print(f"✅ Created and started workers for stream {stream_id} (starts at frame {last_index})")
            return context

    def remove_stream_context(self, stream_id: str):
        with self.contexts_lock:
            if stream_id in self.stream_workers:
                print(f"🧹 Removing stream context and stopping workers for {stream_id}")
                context = self.stream_workers.pop(stream_id)
                context.stop_workers()
                
                # Check if shared workers are still alive
                print(f"[{self.model_name}] After removing stream {stream_id}:")
                print(f"  Remaining streams: {len(self.stream_workers)}")
                if self.shared_workers and len(self.shared_workers) >= 2:
                    gpu_worker = self.shared_workers[0]
                    demux_worker = self.shared_workers[1]
                    print(f"  GPU worker alive: {gpu_worker.is_alive()}")
                    print(f"  Demux worker alive: {demux_worker.is_alive()}")
                    
                    # If GPU worker died, this is critical
                    if not gpu_worker.is_alive():
                        print(f"❌ CRITICAL: GPU worker has died! This will break subsequent streams.")
                        print(f"   Attempting to restart GPU worker...")
                        self._restart_gpu_worker()
            else:
                print(f"⚠️ Attempted to remove non-existent stream context: {stream_id}")

    def _restart_gpu_worker(self):
        """Emergency restart of GPU worker if it has crashed"""
        print(f"[{self.model_name}] Emergency GPU worker restart initiated...")
        
        # Clear the GPU queue first
        while not self.gpu_queue.empty():
            try:
                self.gpu_queue.get_nowait()
            except queue.Empty:
                break
        print(f"[{self.model_name}] Cleared GPU queue")
        
        # Start a new GPU worker
        self.gpu_running = True
        gpu_worker = threading.Thread(
            target=streaming_gpu_worker,
            args=(self.gpu_queue, self.demultiplexer_queue, self.base_wrapper.device, 
                  self.base_wrapper.net, self.base_wrapper.mode, self.profiler, 
                  self.base_wrapper.batch_size),
            daemon=True,
            name=f"Shared-GPU-{self.model_name}-RESTARTED"
        )
        
        try:
            gpu_worker.start()
            print(f"[{self.model_name}] New GPU worker started: {gpu_worker.name}")
            
            # Replace in shared workers list
            if self.shared_workers and len(self.shared_workers) >= 1:
                self.shared_workers[0] = gpu_worker
            
            # Verify it's alive
            time.sleep(0.1)
            if gpu_worker.is_alive():
                print(f"✅ [{self.model_name}] GPU worker successfully restarted")
            else:
                print(f"❌ [{self.model_name}] GPU worker restart failed!")
        except Exception as e:
            print(f"❌ [{self.model_name}] Failed to restart GPU worker: {e}")
            import traceback
            traceback.print_exc()

    def stop_workers(self):
        """Stops all stream workers and shared workers."""
        print(f"[{self.model_name}] Stopping all per-stream and shared workers...")
        
        with self.contexts_lock:
            for context in self.stream_workers.values():
                context.stop_workers()
            self.stream_workers.clear()

        # Stop shared workers
        self.gpu_running = False
        self.demultiplexer_running = False
        self.gpu_queue.put(None)
        self.demultiplexer_queue.put(None) # To unblock the get()

        for worker in self.shared_workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
        
        print(f"✅ [{self.model_name}] All workers stopped.")

    def debug_worker_status(self):
        """Debug information about worker status"""
        with self.contexts_lock:
            stream_status = {}
            for stream_id, context in self.stream_workers.items():
                stream_status[stream_id] = {
                    'preprocess_worker_alive': context.preprocess_worker.is_alive() if context.preprocess_worker else False,
                    'postprocess_worker_alive': context.postprocess_worker.is_alive() if context.postprocess_worker else False,
                    'preprocess_queue_size': context.preprocess_queue.qsize() if hasattr(context.preprocess_queue, 'qsize') else 'unknown',
                    'postprocess_queue_size': context.postprocess_queue.qsize() if hasattr(context.postprocess_queue, 'qsize') else 'unknown',
                    'output_buffer_size': len(context.output_buffer),
                    'is_active': context.is_active
                }
        
        shared_gpu_alive = self.shared_workers[0].is_alive() if len(self.shared_workers) > 0 else False
        demux_alive = self.shared_workers[1].is_alive() if len(self.shared_workers) > 1 else False
        
        return {
            'per_stream_workers_wrapper': True,
            'gpu_worker_alive': shared_gpu_alive,
            'demultiplexer_alive': demux_alive,
            'gpu_queue_size': self.gpu_queue.qsize() if hasattr(self.gpu_queue, 'qsize') else 'unknown',
            'demux_queue_size': self.demultiplexer_queue.qsize() if hasattr(self.demultiplexer_queue, 'qsize') else 'unknown',
            'active_streams': len(self.stream_workers),
            'per_stream_status': stream_status
        }
    
    def __getattr__(self, name):
        """Delegate unknown attributes to base wrapper."""
        # This allows api.py to access things like frame_manager, landmark_manager etc.
        return getattr(self.base_wrapper, name)
