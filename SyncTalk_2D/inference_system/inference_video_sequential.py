# inference_system/inference_video_sequential.py
# BENCHMARK VERSION - Simple Sequential (Single-Threaded, No Pipeline)

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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# These imports depend on the user's project structure.
from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features
from inference_system.core.video_frame_manager import AllFramesMemory
from inference_system.core.landmark_manager import LandmarkManager
from inference_system.utils.profiler import PerformanceProfiler
from torch.cuda.amp import autocast

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

# NOTE: No threading or queue imports are needed for this version.

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
#  MAIN EXECUTION BLOCK
# =================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequential Batch Video Inference (Benchmark)')
    parser.add_argument('--asr', type=str, default="ave")
    parser.add_argument('--name', type=str, default="May")
    parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--video_path', type=str, help='Path to video file')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of frames to process in a batch')
    args = parser.parse_args()

    profiler = PerformanceProfiler("SequentialInference")

    # --- Setup and Initialization (Unchanged) ---
    with profiler.timer("Setup and initialization"):
        checkpoint_path = os.path.join("./checkpoint", args.name)
        checkpoint = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: int(x.split(".")[0]))[-1])
        print(f"Using checkpoint: {checkpoint}")
        save_path = os.path.join("./result", f"{args.name}_{os.path.basename(args.audio_path).split('.')[0]}_{checkpoint.split('/')[-1].split('.')[0]}_sequential_benchmark.mp4")
        dataset_dir = os.path.join("./dataset", args.name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mode = args.asr
        batch_size = args.batch_size

    # --- Model & Data Loading (Unchanged) ---
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

    # --- Main Sequential Loop ---
    num_frames_to_process = len(audio_feats_np)
    print(f"Processing {num_frames_to_process} frames with sequential execution...")
    frame_manager.current_idx = args.start_frame
    frame_manager.direction = 1
    batch_data = []
    frame_results = {}
    
    with tqdm(total=num_frames_to_process, desc="Sequential Processing") as pbar:
        for i in range(num_frames_to_process):
            # STAGE 1: Gather data for one frame
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

            # When the batch is full (or it's the last frame), process it completely.
            if len(batch_data) >= batch_size or (i == num_frames_to_process - 1 and batch_data):
                # STAGE 2: CPU Pre-processing for the entire batch
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

                # STAGE 3: GPU Inference for the entire batch
                with profiler.timer("Batch GPU Upload & Infer"):
                    batch_size_actual = len(batch_data)
                    real_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in b_real]
                    masked_tensors = [torch.from_numpy(img.transpose(2,0,1)) for img in b_masked]
                    audio_tensors = [torch.from_numpy(d.audio_feat) for d in batch_data]

                    real_batch = torch.stack(real_tensors).to(device).float() / 255.0
                    masked_batch = torch.stack(masked_tensors).to(device).float() / 255.0
                    audio_batch = torch.stack(audio_tensors).to(device).float()
                    
                    if mode == "ave": audio_batch = audio_batch.view(batch_size_actual, 32, 16, 16)
                    
                    img_for_net = torch.cat([real_batch, masked_batch], dim=1)

                    with autocast():
                        with torch.no_grad():
                            pred_batch = net(img_for_net, audio_batch)

                    pred_batch_np = pred_batch.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                # STAGE 4: CPU Post-processing for the entire batch
                with profiler.timer("Batch CPU Post-processing"):
                    for j in range(len(batch_data)):
                        pred_np = np.clip(pred_batch_np[j], 0, 255).astype(np.uint8)
                        canvas, original_crop = b_canvases[j], b_crops[j]
                        data = batch_data[j]
                        full_frame = data.img
                        xmin, ymin, xmax, ymax = data.bbox

                        canvas[4:324, 4:324] = pred_np
                        h_crop, w_crop = original_crop.shape[:2]
                        final_face = cv2.resize(canvas, (w_crop, h_crop), interpolation=cv2.INTER_CUBIC)
                        
                        final_frame = full_frame.copy()
                        final_frame[ymin:ymax, xmin:xmax] = final_face
                        frame_results[data.frame_idx] = final_frame

                pbar.update(len(batch_data))
                # Clear the batch to start fresh
                batch_data = []

    # --- Final Steps (Writing, Cleanup, Reporting) ---
    with profiler.timer("Video writing"):
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
    print("FINAL PERFORMANCE ANALYSIS (SEQUENTIAL)")
    print("="*60)
    total_frames_processed = len(frame_results)
    if hasattr(pbar, 'start_t') and pbar.start_t and hasattr(pbar, 'last_print_t') and pbar.last_print_t:
        loop_duration = pbar.last_print_t - pbar.start_t
        if loop_duration > 0:
            print(f"  End-to-End FPS (Sequential Loop): {total_frames_processed / loop_duration:.2f} FPS")

    num_batches = len(profiler.timings.get("Batch GPU Upload & Infer", []))
    if num_batches > 0:
        cpu_pre_time = sum(profiler.timings.get("Batch CPU Pre-processing", [0]))
        gpu_time = sum(profiler.timings.get("Batch GPU Upload & Infer", [0]))
        cpu_post_time = sum(profiler.timings.get("Batch CPU Post-processing", [0]))
        print("\nPer-batch timings (average):")
        print(f"  CPU Pre-processing:  {cpu_pre_time / num_batches * 1000:.2f} ms")
        print(f"  GPU Upload & Infer:  {gpu_time / num_batches * 1000:.2f} ms")
        print(f"  CPU Post-processing: {cpu_post_time / num_batches * 1000:.2f} ms")
        
        total_batch_time_ms = ((cpu_pre_time + gpu_time + cpu_post_time) / num_batches) * 1000
        print(f"  ------------------------------------------")
        print(f"  Total Sequential Time per Batch: {total_batch_time_ms:.2f} ms")