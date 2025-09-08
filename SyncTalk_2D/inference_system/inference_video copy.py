# inference_system/inference_video.py
# FINAL VERSION - 3-Stage Pipeline (Producer -> GPU -> Consumer)

import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
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

# =================================================================================
#  STAGE 2: GPU WORKER THREAD FUNCTION
# =================================================================================
def gpu_worker(q_in: queue.Queue, q_out: queue.Queue, device, net, mode, profiler):
    """
    A dedicated thread that takes pre-processed data, handles GPU work,
    and passes results to the post-processing queue.
    """
    while True:
        item = q_in.get()
        if item is None:
            q_out.put(None) # Signal downstream worker to stop
            break

        batch_data, batch_img_real_ex, batch_img_masked, batch_canvases, batch_original_crops = item

        with profiler.timer("Batch GPU Upload & Infer"):
            batch_size_actual = len(batch_data)
            real_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in batch_img_real_ex]
            masked_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in batch_img_masked]
            audio_tensors = [torch.from_numpy(d.audio_feat) for d in batch_data]

            real_batch = torch.stack(real_tensors).to(device, non_blocking=True).float() / 255.0
            masked_batch = torch.stack(masked_tensors).to(device, non_blocking=True).float() / 255.0
            audio_batch = torch.stack(audio_tensors).to(device, non_blocking=True).float()

            if mode == "ave": audio_batch = audio_batch.view(batch_size_actual, 32, 16, 16)

            img_for_net = torch.cat([real_batch, masked_batch], dim=1)

            with autocast():
                with torch.no_grad():
                    pred_batch = net(img_for_net, audio_batch)

            # This .cpu() call is a synchronization point. Moving the tensor
            # to CPU memory is necessary before passing to the CPU-only post-processing thread.
            pred_batch_np = pred_batch.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        q_out.put((batch_data, pred_batch_np, batch_canvases, batch_original_crops))

# =================================================================================
#  STAGE 3: CPU POST-PROCESSING WORKER THREAD FUNCTION
# =================================================================================
def cpu_post_processing_worker(q_in: queue.Queue, frame_results: dict, pbar: tqdm, profiler: PerformanceProfiler):
    """
    A dedicated thread that takes results from the GPU worker, performs
    CPU-bound post-processing (resizing, pasting), and stores the final frame.
    """
    while True:
        item = q_in.get()
        if item is None: # Shutdown signal
            break

        p_batch_data, p_pred_batch_np, p_canvases, p_orig_crops = item

        with profiler.timer("Batch CPU Post-processing"):
            for j in range(len(p_batch_data)):
                pred_np = np.clip(p_pred_batch_np[j], 0, 255).astype(np.uint8)
                canvas, original_crop = p_canvases[j], p_orig_crops[j]
                data = p_batch_data[j]
                full_frame = data.img
                xmin, ymin, xmax, ymax = data.bbox

                canvas[4:324, 4:324] = pred_np
                h_crop, w_crop = original_crop.shape[:2]
                final_face = cv2.resize(canvas, (w_crop, h_crop), interpolation=cv2.INTER_CUBIC)

                # Use a copy to avoid potential race conditions if `data.img` is ever
                # referenced elsewhere. This is a safe practice in multithreading.
                final_frame = full_frame.copy()
                final_frame[ymin:ymax, xmin:xmax] = final_face

                # The frame_results dict is the shared state between this thread
                # and the final writing step. Access is safe because this is the only
                # writer thread, and the main thread only reads after this thread has joined.
                frame_results[data.frame_idx] = final_frame

        pbar.update(len(p_batch_data))


# =================================================================================
#  MAIN EXECUTION BLOCK (STAGE 1: Producer)
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

    profiler = PerformanceProfiler("PipelinedInference")

    with profiler.timer("Setup and initialization"):
        checkpoint_path = os.path.join("./checkpoint", args.name)
        checkpoint = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: int(x.split(".")[0]))[-1])
        print(f"Using checkpoint: {checkpoint}")
        save_path = os.path.join("./result", f"{args.name}_{os.path.basename(args.audio_path).split('.')[0]}_{checkpoint.split('/')[-1].split('.')[0]}_final.mp4")
        dataset_dir = os.path.join("./dataset", args.name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mode = args.asr
        batch_size = args.batch_size

    with profiler.timer("Model & Data Loading"):
        net = Model(6, mode).to(device).eval()
        net.load_state_dict(torch.load(checkpoint))

        audio_encoder = AudioEncoder().to(device).eval()
        ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth')
        audio_encoder.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})

        dataset = AudDataset(args.audio_path)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        print("Pre-computing all audio features...")
        outputs = []
        for mel in tqdm(data_loader, desc="Pre-computing audio features"):
            with torch.no_grad():
                out = audio_encoder(mel.to(device))
            outputs.append(out)
        audio_feats_tensor_gpu = torch.cat([outputs[0][:1], torch.cat(outputs, dim=0), outputs[-1][-1:]], dim=0)
        audio_feats_np = audio_feats_tensor_gpu.cpu().numpy()
        del outputs, audio_feats_tensor_gpu
        torch.cuda.empty_cache()
        print("Audio features ready.")

        if args.video_path and os.path.exists(args.video_path):
            frame_manager = AllFramesMemory(args.video_path, from_images=False)
        else:
            img_dir = os.path.join(dataset_dir, "full_body_img/")
            frame_manager = AllFramesMemory(img_dir, from_images=True)
        frame_manager.initialize()

        landmark_manager = LandmarkManager(os.path.join(dataset_dir, "landmarks/"))
        landmark_manager.initialize()

        exm_frame = frame_manager.get_frame(0).frame
        h, w = exm_frame.shape[:2]
        fps = 20 if mode == "wenet" else 25
        video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"),
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # --- Pipeline Setup ---
    # We now have two queues to connect the three stages.
    # A larger maxsize provides a bigger buffer, making the pipeline
    # more resilient to momentary slowdowns in any one stage.
    q_size = batch_size * 3
    gpu_queue = queue.Queue(maxsize=q_size) # Main thread -> GPU thread
    cpu_queue = queue.Queue(maxsize=q_size) # GPU thread -> Post-processing thread

    # This dictionary will store the final, processed frames. It's populated
    # by the consumer thread and read by the main thread after all threads finish.
    frame_results = {}

    num_frames_to_process = len(audio_feats_np)
    pbar = tqdm(total=num_frames_to_process, desc="Pipelined Processing")

    # --- Start the Worker Threads ---
    gpu_thread = threading.Thread(target=gpu_worker, args=(gpu_queue, cpu_queue, device, net, mode, profiler), daemon=True)
    cpu_post_thread = threading.Thread(target=cpu_post_processing_worker, args=(cpu_queue, frame_results, pbar, profiler), daemon=True)

    gpu_thread.start()
    cpu_post_thread.start()

    # --- Main Pipelined Loop (Producer only) ---
    print(f"Processing {num_frames_to_process} frames with 3-stage pipelined execution...")
    frame_manager.current_idx = args.start_frame
    frame_manager.direction = 1
    batch_data = []

    # The main thread's loop is now ONLY responsible for producing work for the GPU.
    # It gathers data, performs CPU pre-processing, and puts the batch onto the queue.
    for i in range(num_frames_to_process):
        with profiler.timer("Frame Gathering"):
            frame_data_item = frame_manager.get_next_frame_pingpong()
            if not frame_data_item: continue

            landmark_data = landmark_manager.get_landmark(frame_data_item.frame_idx)
            if not landmark_data: continue

            audio_feat = get_audio_features(audio_feats_np, i)

            frame_info = FrameData(
                frame_idx=i, img_idx=frame_data_item.frame_idx, img=frame_data_item.frame,
                landmarks=landmark_data.landmarks, bbox=landmark_data.bbox,
                audio_feat=audio_feat
            )
            batch_data.append(frame_info)

        # Submit a batch when it's full, or if it's the very last frame
        if len(batch_data) >= batch_size or (i == num_frames_to_process - 1 and batch_data):
            with profiler.timer("Batch CPU Pre-processing"):
                b_real, b_masked, b_canvases, b_crops = [], [], [], []
                for data in batch_data:
                    xmin, ymin, xmax, ymax = data.bbox
                    crop = data.img[ymin:ymax, xmin:xmax]
                    b_crops.append(crop)
                    resized_328 = cv2.resize(crop, (328, 328), interpolation=cv2.INTER_CUBIC)
                    b_canvases.append(resized_328)
                    img_real_ex = resized_328[4:324, 4:324]
                    b_real.append(img_real_ex)
                    b_masked.append(cv2.rectangle(img_real_ex.copy(), (5, 5, 310, 305), (0, 0, 0), -1))

            # The main thread's only job is to put work on the queue.
            # It will block here ONLY if the pipeline is completely full.
            gpu_queue.put((batch_data, b_real, b_masked, b_canvases, b_crops))
            batch_data = []

    # --- Shutdown Sequence ---
    # 1. Signal the first worker (GPU) that there's no more data.
    gpu_queue.put(None)

    # 2. Wait for the threads to finish their work in sequence.
    #    The GPU worker will process its remaining items, then put a `None`
    #    on the cpu_queue, signaling the post-processor to finish.
    gpu_thread.join()
    cpu_post_thread.join()
    pbar.close()

    # --- Final Steps (Writing, Cleanup, Reporting) ---
    with profiler.timer("Video writing"):
        # Sort the results by frame index to ensure correct video order
        sorted_keys = sorted(frame_results.keys())
        print(f"Writing {len(sorted_keys)} frames to video...")
        for i in sorted_keys:
            video_writer.write(frame_results[i])

    with profiler.timer("Cleanup and Final Encoding"):
        video_writer.release()
        frame_manager.cleanup()
        landmark_manager.cleanup()
        print("Combining video and audio with ffmpeg...")
        os.system(f"ffmpeg -i {save_path.replace('.mp4', 'temp.mp4')} -i {args.audio_path} -c:v libx264 -c:a aac -crf 20 -y {save_path}")
        os.remove(save_path.replace('.mp4', 'temp.mp4'))

    print(f"\n[INFO] ===== save video to {save_path} =====")
    profiler.print_summary()

    print("\n" + "="*60)
    print("FINAL PERFORMANCE ANALYSIS")
    print("="*60)
    total_frames_processed = len(frame_results)
    if hasattr(pbar, 'start_t') and pbar.start_t and hasattr(pbar, 'last_print_t') and pbar.last_print_t:
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