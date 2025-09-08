# inference_system/inference_video.py
# FINAL VERSION - 4-Stage Pipeline (Gather -> Pre-Process -> GPU -> Post-Process)

import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
# This sys.path.insert is important for the project structure.
# It assumes 'unet_328' and 'utils' are in the parent directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# These imports depend on the user's project structure.
# Make sure 'unet_328.py' and 'utils.py' are accessible.
from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features
from inference_system.core.video_frame_manager import AllFramesMemory
from inference_system.core.landmark_manager import LandmarkManager
from inference_system.utils.profiler import PerformanceProfiler
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
import queue

@dataclass
class FrameData:
    frame_idx: int
    img_idx: int
    img: np.ndarray
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    audio_feat: np.ndarray
    parsing: Optional[np.ndarray] = None
    audio_chunk: np.ndarray = field(default_factory=lambda: np.zeros(640, dtype=np.float32))


class OrderedFrameWriter:
    """
    A thread-safe, high-performance class that receives frames out of order
    and writes them to a video file in sequence. This version uses a dedicated
    write lock to prevent race conditions during file I/O.
    """
    def __init__(self, video_writer, total_frames, profiler):
        self.video_writer = video_writer
        self.total_frames = total_frames
        self.profiler = profiler
        self._buffer_lock = threading.Lock()  # Lock for the shared buffer
        self._write_lock = threading.Lock()   # NEW: Lock specifically for the write operation
        self._frame_buffer = {}
        self._next_frame_to_write = 0
        self._frames_written = 0
        self._done_event = threading.Event()

    def put(self, frame_idx, frame):
        """
        Receives a processed frame, adds it to a buffer, and writes any
        available sequential frames to disk in a thread-safe, ordered manner.
        """
        frames_to_write_now = []
        
        # --- Critical section for buffer manipulation (very fast) ---
        with self._buffer_lock:
            self._frame_buffer[frame_idx] = frame
            
            # Check for all consecutive frames that can now be written
            while self._next_frame_to_write in self._frame_buffer:
                frame_data = self._frame_buffer.pop(self._next_frame_to_write)
                frames_to_write_now.append(frame_data)
                self._next_frame_to_write += 1
        # --- End of buffer critical section ---

        # If we found frames to write, perform the slow I/O.
        # This part is now protected by its own lock to guarantee write order.
        if frames_to_write_now:
            with self._write_lock: # <-- THE CRITICAL FIX
                with self.profiler.timer("Frame Writing & Ordering"):
                    for frame_data in frames_to_write_now:
                        self.video_writer.write(frame_data)
                        # pass
            
            # Now, safely update the shared counter
            with self._buffer_lock:
                self._frames_written += len(frames_to_write_now)
                # If we've just written the last frame, signal completion
                if self._frames_written == self.total_frames:
                    self._done_event.set()

    def wait_for_completion(self, timeout=None):
        """Blocks the calling thread until all frames have been written."""
        # print("[INFO] Main thread is waiting for all frames to be written...")
        self._done_event.wait(timeout)

    def is_done(self):
        """Checks if all frames have been written."""
        return self._frames_written == self.total_frames

    def close(self):
        """Releases the video writer."""
        print("[INFO] Closing the ordered video writer.")
        self.video_writer.release()

# =================================================================================
#  STAGE 2: CPU PRE-PROCESSING WORKER
# =================================================================================
def cpu_pre_processing_worker(q_in: queue.Queue, q_out: queue.Queue, batch_size: int, 
                            profiler: PerformanceProfiler, total_frames: int = None):
    """
    Enhanced version that supports both batch and streaming modes.
    If total_frames is None, operates in streaming mode.
    """
    batch_data = []
    is_streaming = total_frames is None
    
    while True:
        with profiler.measure_block("PreProc_Wait_On_Get"):
            item = q_in.get()

        if item is None:
            if batch_data:
                # Handle final batch
                if not is_streaming and len(batch_data) < batch_size:
                    # Original padding logic for batch mode
                    padded_batch = list(batch_data)
                    num_to_pad = batch_size - len(padded_batch)
                    if num_to_pad > 0:
                        last_item = padded_batch[-1]
                        dummy_item = FrameData(
                            frame_idx=total_frames, 
                            img_idx=last_item.img_idx, 
                            img=last_item.img, 
                            landmarks=last_item.landmarks, 
                            bbox=last_item.bbox, 
                            audio_feat=last_item.audio_feat
                        )
                        padding = [dummy_item] * num_to_pad
                        padded_batch.extend(padding)
                    batch_data = padded_batch
                
                # Process final batch
                with profiler.timer("Batch CPU Pre-processing"):
                    b_real, b_masked, b_canvases, b_crops = _process_batch_for_gpu(batch_data)
                
                with profiler.measure_block("PreProc_Wait_On_Put"):
                    q_out.put((batch_data, b_real, b_masked, b_canvases, b_crops))
                    
            q_out.put(None)
            break
        
        batch_data.append(item)

        if len(batch_data) >= batch_size:
            with profiler.timer("Batch CPU Pre-processing"):
                b_real, b_masked, b_canvases, b_crops = _process_batch_for_gpu(batch_data)
            with profiler.measure_block("PreProc_Wait_On_Put"):
                q_out.put((batch_data, b_real, b_masked, b_canvases, b_crops))
            batch_data = []

def _process_batch_for_gpu(batch_data: List[FrameData]) -> Tuple:
    b_real, b_masked, b_canvases, b_crops = [], [], [], []
    for data in batch_data:
        xmin, ymin, xmax, ymax = data.bbox
        crop = data.img[ymin:ymax, xmin:xmax]
        b_crops.append(crop)
        resized_328 = cv2.resize(crop, (328, 328), interpolation=cv2.INTER_LINEAR)
        b_canvases.append(resized_328)
        img_real_ex = resized_328[4:324, 4:324]
        
        # Convert to pinned memory for faster GPU upload
        img_real_ex = np.ascontiguousarray(img_real_ex)
        masked = cv2.rectangle(img_real_ex.copy(), (5, 5, 310, 305), (0, 0, 0), -1)
        masked = np.ascontiguousarray(masked)
        
        b_real.append(img_real_ex)
        b_masked.append(masked)
    return b_real, b_masked, b_canvases, b_crops

# =================================================================================
#  STAGE 3: GPU WORKER THREAD FUNCTION
# =================================================================================
def gpu_worker(q_in: queue.Queue, q_out: queue.Queue, device, net, mode, profiler):
    """
    Takes pre-processed batches, handles all GPU work, and passes results
    to the post-processing queue.
    """
    while True:
        with profiler.measure_block("GPU_Wait_On_Get"):
            item = q_in.get()

        if item is None:
            q_out.put(None)
            break
        
        batch_data, batch_img_real_ex, batch_img_masked, batch_canvases, batch_original_crops = item

        with profiler.timer("Batch GPU Upload & Infer"):
            batch_size_actual = len(batch_data)
            real_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in batch_img_real_ex]
            masked_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in batch_img_masked]
            audio_tensors = [torch.from_numpy(d.audio_feat) for d in batch_data]

            real_batch = torch.stack(real_tensors).to(device, non_blocking=True).half() / 255.0
            masked_batch = torch.stack(masked_tensors).to(device, non_blocking=True).half() / 255.0
            audio_batch = torch.stack(audio_tensors).to(device, non_blocking=True).half()

            if mode == "ave": audio_batch = audio_batch.view(batch_size_actual, 32, 16, 16)

            img_for_net = torch.cat([real_batch, masked_batch], dim=1)

            with autocast():
                with torch.no_grad():
                    pred_batch = net(img_for_net, audio_batch)

            pred_batch_np = pred_batch.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        
        with profiler.measure_block("GPU_Wait_On_Put"):
            q_out.put((batch_data, pred_batch_np, batch_canvases, batch_original_crops))

def _post_process_single_frame(data, pred_np, canvas, original_crop):
    """Helper function to contain the post-processing logic for one frame."""
    full_frame = data.img
    xmin, ymin, xmax, ymax = data.bbox

    canvas[4:324, 4:324] = np.clip(pred_np, 0, 255).astype(np.uint8)
    h_crop, w_crop = original_crop.shape[:2]
    final_face = cv2.resize(canvas, (w_crop, h_crop), interpolation=cv2.INTER_LINEAR)
    
    final_frame = full_frame.copy()
    final_frame[ymin:ymax, xmin:xmax] = final_face
    return data.frame_idx, final_frame


def progress_reporter(progress_q: queue.Queue, pbar: tqdm):
    """A dedicated thread to safely update the tqdm progress bar."""
    while True:
        # Get the number of items that were just processed
        num_processed = progress_q.get()
        
        # Check for shutdown signal
        if num_processed is None:
            break
            
        # Update the progress bar, but don't let it go over the total
        # This handles the padding gracefully.
        current_val = pbar.n
        if current_val + num_processed > pbar.total:
            num_processed = pbar.total - current_val
        
        if num_processed > 0:
            pbar.update(num_processed)

# =================================================================================
#  STAGE 4: CPU POST-PROCESSING WORKER THREAD FUNCTION
# =================================================================================
def cpu_post_processing_worker(q_in: queue.Queue, ordered_writer: OrderedFrameWriter, progress_q: queue.Queue, profiler: PerformanceProfiler):
    """
    Takes results from the GPU, post-processes them, passes them to the
    writer, and reports progress to a dedicated queue.
    """
    with ThreadPoolExecutor(max_workers=os.cpu_count(), thread_name_prefix="PostProc_SubWorker") as executor:
        while True:
            with profiler.measure_block("PostProc_Wait_On_Get"):
                item = q_in.get()
            if item is None:
                break
            
            p_batch_data, p_pred_batch_np, p_canvases, p_orig_crops = item
            
            # --- We only want to report progress for REAL frames, not padded ones ---
            # The number of real frames is the original batch size before padding.
            num_real_frames = len([d for d in p_batch_data if d.frame_idx < ordered_writer.total_frames])
            
            with profiler.timer("Batch CPU Post-processing"):
                futures = [executor.submit(_post_process_single_frame, p_batch_data[j], p_pred_batch_np[j], p_canvases[j], p_orig_crops[j]) for j in range(len(p_batch_data))]

                for future in futures:
                    frame_idx, final_frame = future.result()
                    # Only put real frames into the writer. The writer's logic depends
                    # on getting exactly `total_frames`.
                    if frame_idx < ordered_writer.total_frames:
                        ordered_writer.put(frame_idx, final_frame)
            
            # Report the number of REAL frames processed to the progress reporter
            if num_real_frames > 0:
                progress_q.put(num_real_frames)


def find_optimal_workers():
    cpu_count = os.cpu_count()
    # Test with different worker counts
    # For CPU-bound post-processing, usually optimal is CPU_COUNT - 2
    return min(cpu_count - 2, 8)  # Cap at 8 to avoid thrashing

# =================================================================================
#  MAIN EXECUTION BLOCK (STAGE 1: Gatherer)
# =================================================================================
# =================================================================================
#  MAIN EXECUTION BLOCK (Final, Corrected Version with Progress Reporter)
# =================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipelined Batch Video Inference')
    parser.add_argument('--asr', type=str, default="ave")
    parser.add_argument('--name', type=str, default="May")
    parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--video_path', type=str, help='Path to video file')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of frames to process in parallel')
    args = parser.parse_args()

    profiler = PerformanceProfiler("Streaming-Pipelined-Inference")

    with profiler.timer("Setup and initialization"):
        checkpoint_path = os.path.join("./checkpoint", args.name)
        checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth') and f.split('.')[0].isdigit()]
        checkpoint = os.path.join(checkpoint_path, sorted(checkpoint_files, key=lambda x: int(x.split(".")[0]))[-1])
        print(f"Using checkpoint: {checkpoint}")
        save_path = os.path.join("./result", f"{args.name}_{os.path.basename(args.audio_path).split('.')[0]}_{checkpoint.split('/')[-1].split('.')[0]}.mp4")
        dataset_dir = os.path.join("./dataset", args.name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mode = args.asr
        batch_size = args.batch_size

    with profiler.timer("Model & Data Loading"):
        net = Model(6, mode).to(device).eval()
        net.load_state_dict(torch.load(checkpoint))
        net = net.half() #FP16

        audio_encoder = AudioEncoder().to(device).eval().half()
        ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth')
        audio_encoder.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})

        dataset = AudDataset(args.audio_path)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        print("Pre-computing all audio features...")
        outputs = []
        for mel in tqdm(data_loader, desc="Pre-computing audio features"):
            with torch.no_grad():
                out = audio_encoder(mel.to(device).half())
            outputs.append(out)
        audio_feats_tensor_gpu = torch.cat([outputs[0][:1], torch.cat(outputs, dim=0), outputs[-1][-1:]], dim=0)
        audio_feats_np = audio_feats_tensor_gpu.cpu().numpy()
        del outputs, audio_feats_tensor_gpu
        torch.cuda.empty_cache()
        print("Audio features ready.")

        num_frames_to_process = len(audio_feats_np)
        print(f"[INFO] Audio requires {num_frames_to_process} frames")

        if args.video_path and os.path.exists(args.video_path):
            frame_manager = AllFramesMemory(
                args.video_path, 
                from_images=False,
                max_frames_to_load=num_frames_to_process + args.start_frame  # Account for start offset
            )
        else:
            img_dir = os.path.join(dataset_dir, "full_body_img/")
            frame_manager = AllFramesMemory(
                img_dir, 
                from_images=True,
                max_frames_to_load=num_frames_to_process + args.start_frame  # Account for start offset
            )

        frame_manager.initialize()

        landmark_manager = LandmarkManager(os.path.join(dataset_dir, "landmarks/"))
        landmark_manager.initialize()

        exm_frame = frame_manager.get_frame(0).frame
        h, w = exm_frame.shape[:2]
        fps = 20 if mode == "wenet" else 25
        video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"),
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # --- Pipeline Setup ---
    q_size = args.batch_size * 20
    preprocess_queue = queue.Queue(maxsize=q_size)
    gpu_queue = queue.Queue(maxsize=q_size)
    postprocess_queue = queue.Queue(maxsize=q_size)
    progress_queue = queue.Queue() # New queue for dedicated progress reporting

    num_frames_to_process = len(audio_feats_np)
    ordered_writer = OrderedFrameWriter(video_writer, num_frames_to_process, profiler)

    # --- TQDM and Progress Reporter Thread ---
    pbar = tqdm(total=num_frames_to_process, desc="Streaming Pipelined Processing", file=sys.stderr)
    reporter_thread = threading.Thread(target=progress_reporter, args=(progress_queue, pbar), daemon=True, name="ProgressReporter")
    reporter_thread.start()

    # --- Start the Worker Threads ---
    num_post_proc_workers = find_optimal_workers()
    print(f"[INFO] Launching {num_post_proc_workers} parallel post-processing workers.")
    
    pre_proc_thread = threading.Thread(target=cpu_pre_processing_worker, args=(preprocess_queue, gpu_queue, args.batch_size, profiler, num_frames_to_process), daemon=True, name="Pre-Processor")
    gpu_thread = threading.Thread(target=gpu_worker, args=(gpu_queue, postprocess_queue, device, net, args.asr, profiler), daemon=True, name="GPU-Worker")
    
    post_proc_threads = []
    for i in range(num_post_proc_workers):
        thread = threading.Thread(target=cpu_post_processing_worker, args=(postprocess_queue, ordered_writer, progress_queue, profiler), daemon=True, name=f"Post-Processor-{i}")
        thread.start()
        post_proc_threads.append(thread)

    pre_proc_thread.start()
    gpu_thread.start()
    
    # --- Main Loop (Gatherer) ---
    print(f"Processing {num_frames_to_process} frames with streaming execution...")
    frame_manager.current_idx = args.start_frame
    frame_manager.direction = 1
    
    for i in range(num_frames_to_process):
        with profiler.timer("Frame Gathering"):
            # FIX: Use start_frame offset when getting frames
            actual_frame_idx = i + args.start_frame
            frame_data_item = frame_manager.get_frame_pingpong(actual_frame_idx)
            
            if not frame_data_item:
                print(f"[Warning] Could not get source frame for index {actual_frame_idx}. Skipping.")
                continue
            
            landmark_data = landmark_manager.get_landmark(frame_data_item.frame_idx)
            if not landmark_data: continue

            audio_feat = get_audio_features(audio_feats_np, i)  # Keep 'i' here for audio sync
            
            frame_info = FrameData(
                frame_idx=i,  # Keep this as 'i' for output ordering
                img_idx=frame_data_item.frame_idx,  # This is the actual source frame index
                img=frame_data_item.frame,
                landmarks=landmark_data.landmarks, 
                bbox=landmark_data.bbox,
                audio_feat=audio_feat
            )
        
        with profiler.measure_block("Gatherer_Wait_On_Put"):
            preprocess_queue.put(frame_info)

    # --- FINAL ROBUST Shutdown Sequence ---
    preprocess_queue.put(None)
    pre_proc_thread.join()
    gpu_thread.join()

    for _ in range(num_post_proc_workers):
        postprocess_queue.put(None)
    
    ordered_writer.wait_for_completion(timeout=60)

    for thread in post_proc_threads:
        thread.join()
        
    progress_queue.put(None)
    reporter_thread.join()

    pbar.close()

    # --- Final Steps ---
    with profiler.timer("Cleanup and Final Encoding"):
        ordered_writer.close()
        frame_manager.cleanup()
        landmark_manager.cleanup()
        print("Combining video and audio with ffmpeg...")
        os.system(f"ffmpeg -y -i {save_path.replace('.mp4', 'temp.mp4')} -i {args.audio_path} -c:v libx264 -c:a aac -crf 20 {save_path}")
        os.remove(save_path.replace('.mp4', 'temp.mp4'))

    # --- Final Analysis ---
    print(f"\n[INFO] ===== save video to {save_path} =====")
    profiler.print_summary()
    # profiler.print_timeline()
    
    timeline_path = save_path.replace(".mp4", "_timeline.csv")
    profiler.save_timeline_csv(timeline_path)

    print("\n" + "="*60)
    print("FINAL PERFORMANCE ANALYSIS")
    print("="*60)
    if hasattr(pbar, 'start_t') and pbar.start_t and hasattr(pbar, 'last_print_t') and pbar.last_print_t:
        total_frames_processed = ordered_writer._frames_written
        loop_duration = pbar.last_print_t - pbar.start_t
        if loop_duration > 0:
            print(f"  End-to-End FPS (Pipelined Loop): {total_frames_processed / loop_duration:.2f} FPS")

    num_batches = len(profiler.timings.get("Batch GPU Upload & Infer", []))
    if num_batches > 0:
        cpu_pre_time = sum(profiler.timings.get("Batch CPU Pre-processing", [0]))
        gpu_time = sum(profiler.timings.get("Batch GPU Upload & Infer", [0]))
        cpu_post_time = sum(profiler.timings.get("Batch CPU Post-processing", [0]))
        print("\nPer-batch timings (average):")
        print(f"  CPU Pre-processing:  {cpu_pre_time / num_batches * 1000:.2f} ms")
        print(f"  GPU Upload & Infer:  {gpu_time / num_batches * 1000:.2f} ms")
        print(f"  CPU Post-processing: {cpu_post_time / num_batches * 1000:.2f} ms")