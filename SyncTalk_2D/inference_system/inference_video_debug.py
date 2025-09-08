# inference_system/inference_video_debug.py - Updated with batch processing
import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features
from inference_system.core.video_frame_manager import AllFramesMemory
from inference_system.core.landmark_manager import LandmarkManager
from inference_system.core.gpu_memory_manager import GPUMemoryManager, BatchGPUMemoryManager
from inference_system.utils.profiler import PerformanceProfiler

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class FrameData:
    """Container for all data needed to process a frame"""
    frame_idx: int
    img_idx: int
    img: np.ndarray
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    audio_feat: np.ndarray
    parsing: Optional[np.ndarray] = None
    crop_shape: Optional[Tuple[int, int]] = None
    crop_img_par: Optional[np.ndarray] = None
    crop_parsing_img: Optional[np.ndarray] = None

parser = argparse.ArgumentParser(description='Debug video inference',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="May")
parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--parsing', type=bool, default=False)
parser.add_argument('--video_path', type=str, help='Path to video file')

# Debug flags to disable optimizations
parser.add_argument('--disable_video_manager', action='store_true', help='Use original image loading')
parser.add_argument('--disable_landmark_manager', action='store_true', help='Use original landmark loading')
parser.add_argument('--disable_gpu_memory', action='store_true', help='Use original GPU memory allocation')
parser.add_argument('--disable_batching', action='store_true', help='Disable batch processing')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing')
parser.add_argument('--max_frames', type=int, default=100, help='Process only N frames for debugging')
parser.add_argument('--save_debug_frames', action='store_true', help='Save debug frames')

args = parser.parse_args()

# Initialize profiler
profiler = PerformanceProfiler("DebugInference")

print("="*60)
print("DEBUG MODE - Processing optimizations:")
print(f"  Video Manager: {'DISABLED' if args.disable_video_manager else 'ENABLED'}")
print(f"  Landmark Manager: {'DISABLED' if args.disable_landmark_manager else 'ENABLED'}")
print(f"  GPU Memory Manager: {'DISABLED' if args.disable_gpu_memory else 'ENABLED'}")
print(f"  Batch Processing: {'DISABLED' if args.disable_batching else f'ENABLED (batch_size={args.batch_size})'}")
print(f"  Max frames: {args.max_frames}")
print("="*60)

# Setup
checkpoint_path = os.path.join("./checkpoint", args.name)
checkpoint = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: int(x.split(".")[0]))[-1])
save_path = os.path.join("./result", args.name+"_debug.mp4")
dataset_dir = os.path.join("./dataset", args.name)
audio_path = args.audio_path
mode = args.asr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1 if args.disable_batching else args.batch_size

# Create debug output directory
if args.save_debug_frames:
    debug_dir = "./debug_frames"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Debug frames will be saved to: {debug_dir}")

# Initialize GPU memory manager (if enabled)
gpu_manager = None
if not args.disable_gpu_memory:
    if args.disable_batching:
        gpu_manager = GPUMemoryManager(device, mode=mode)
        print("GPU memory manager initialized (single frame)")
    else:
        gpu_manager = BatchGPUMemoryManager(device, mode=mode, batch_size=batch_size)
        print(f"Batch GPU memory manager initialized (batch_size={batch_size})")

# Audio feature extraction (always the same)
model = AudioEncoder().to(device).eval()
ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth')
model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})

dataset = AudDataset(audio_path)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

outputs = []
for mel in data_loader:
    mel = mel.to(device)
    with torch.no_grad():
        out = model(mel)
    outputs.append(out)
outputs = torch.cat(outputs, dim=0).cpu()
first_frame, last_frame = outputs[:1], outputs[-1:]
audio_feats = torch.cat([first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)],
                            dim=0).numpy()

# Frame loading
img_dir = os.path.join(dataset_dir, "full_body_img/")
if args.disable_video_manager:
    print("Using ORIGINAL image file loading")
    frame_manager = None
else:
    print("Using optimized video/frame manager")
    if args.video_path and os.path.exists(args.video_path):
        frame_manager = AllFramesMemory(args.video_path, enable_async=False, from_images=False)
    else:
        frame_manager = AllFramesMemory(img_dir, enable_async=False, from_images=True)
    frame_manager.initialize()

# Landmark loading
lms_dir = os.path.join(dataset_dir, "landmarks/")
if args.disable_landmark_manager:
    print("Using ORIGINAL landmark file loading")
    landmark_manager = None
else:
    print("Using optimized landmark manager")
    landmark_manager = LandmarkManager(lms_dir, enable_async=False)
    landmark_manager.initialize()

# Get example frame for video writer
if frame_manager:
    exm_frame_data = frame_manager.get_frame(0)
    h, w = exm_frame_data.frame.shape[:2]
else:
    exm_img = cv2.imread(os.path.join(img_dir, "0.jpg"))
    h, w = exm_img.shape[:2]

if args.parsing:
    parsing_dir = os.path.join(dataset_dir, "parsing/")

# Video writer
fps = 25 if mode in ["hubert", "ave"] else 20
video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"), 
                             cv2.VideoWriter_fourcc('M','J','P', 'G'), fps, (w, h))

# Load model
net = Model(6, mode).cuda()
net.load_state_dict(torch.load(checkpoint))
net.eval()

def process_single_frame(frame_data: FrameData, frame_count: int) -> np.ndarray:
    """Process a single frame (original method)"""
    img = frame_data.img
    img_idx = frame_data.img_idx
    xmin, ymin, xmax, ymax = frame_data.bbox
    
    # Preprocessing (ORIGINAL method)
    crop_img = img[ymin:ymax, xmin:xmax]  
    crop_img_par = crop_img.copy()  
    if args.parsing and frame_data.parsing is not None:
        crop_parsing_img = frame_data.parsing[ymin:ymax, xmin:xmax] 
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:324, 4:324].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex_ori,(5,5,310,305),(0,0,0),-1)
    
    # Save debug frames
    if args.save_debug_frames and frame_count < 10:
        cv2.imwrite(f"{debug_dir}/frame_{frame_count:04d}_real.jpg", img_real_ex)
        cv2.imwrite(f"{debug_dir}/frame_{frame_count:04d}_masked.jpg", img_masked)
    
    img_masked = img_masked.transpose(2,0,1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
    # Prepare tensors
    if gpu_manager and not args.disable_batching:
        # This will be handled in batch processing
        return None
    elif gpu_manager:
        # Single frame GPU memory manager
        audio_feat_gpu = gpu_manager.prepare_audio_features(frame_data.audio_feat)
        img_concat_gpu = gpu_manager.prepare_image_tensors(img_real_ex, img_masked)
    else:
        # Original tensor creation
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
        
        audio_feat = frame_data.audio_feat
        if isinstance(audio_feat, np.ndarray):
            if mode=="hubert":
                audio_feat = audio_feat.reshape(32,32,32)
            if mode=="wenet":
                audio_feat = audio_feat.reshape(256,16,32)
            if mode=="ave":
                audio_feat = audio_feat.reshape(32,16,16)
            audio_feat = audio_feat[None]
            audio_feat = torch.from_numpy(audio_feat)
        
        # Move to GPU (original way)
        audio_feat_gpu = audio_feat.cuda()
        img_concat_gpu = img_concat_T.cuda()
    
    # Model inference
    with torch.no_grad():
        pred = net(img_concat_gpu, audio_feat_gpu)[0]
    
    # Debug: Check prediction
    if args.save_debug_frames and frame_count < 10:
        pred_debug = pred.cpu().numpy().transpose(1,2,0)*255
        pred_debug = np.clip(pred_debug, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{debug_dir}/frame_{frame_count:04d}_pred.jpg", pred_debug)
        print(f"Frame {frame_count}: pred range [{pred.min():.3f}, {pred.max():.3f}]")
    
    # Post-processing (ORIGINAL method)
    pred = pred.cpu().numpy().transpose(1,2,0)*255
    pred = np.array(pred, dtype=np.uint8)
    crop_img_ori[4:324, 4:324] = pred
    
    if args.save_debug_frames and frame_count < 10:
        cv2.imwrite(f"{debug_dir}/frame_{frame_count:04d}_crop_after.jpg", crop_img_ori)
    
    crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
    if args.parsing and frame_data.parsing is not None:
        parsing_mask = (crop_parsing_img == [0, 0, 255]).all(axis=2) | (crop_parsing_img == [255, 255, 255]).all(axis=2)
        crop_img_ori[parsing_mask] = crop_img_par[parsing_mask]
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    
    if args.save_debug_frames and frame_count < 10:
        cv2.imwrite(f"{debug_dir}/frame_{frame_count:04d}_final.jpg", img)
    
    return img

def process_batch(batch_data: List[FrameData], start_frame_count: int) -> List[np.ndarray]:
    """Process a batch of frames together"""
    batch_size_actual = len(batch_data)
    
    # Prepare batch data
    for i, frame_data in enumerate(batch_data):
        # Crop and preprocess
        xmin, ymin, xmax, ymax = frame_data.bbox
        crop_img = frame_data.img[ymin:ymax, xmin:xmax]
        frame_data.crop_shape = crop_img.shape[:2]
        frame_data.crop_img_par = crop_img.copy()
        
        if args.parsing and frame_data.parsing is not None:
            frame_data.crop_parsing_img = frame_data.parsing[ymin:ymax, xmin:xmax]
        
        # Resize and prepare for model
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        frame_data.crop_img_ori = crop_img.copy()
        img_real_ex = crop_img[4:324, 4:324].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex_ori,(5,5,310,305),(0,0,0),-1)
        
        # Save debug frames
        if args.save_debug_frames and start_frame_count + i < 10:
            cv2.imwrite(f"{debug_dir}/frame_{start_frame_count+i:04d}_real.jpg", img_real_ex)
            cv2.imwrite(f"{debug_dir}/frame_{start_frame_count+i:04d}_masked.jpg", img_masked)
        
        # Convert to CHW format
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        # Add to batch
        ready = gpu_manager.add_to_batch(
            frame_data.audio_feat, 
            img_real_ex, 
            img_masked, 
            frame_data.frame_idx
        )
    
    # Process batch
    audio_batch, img_batch, indices = gpu_manager.prepare_batch()
    
    with torch.no_grad():
        # Process entire batch at once
        pred_batch = net(img_batch, audio_batch)
    
    # Post-process results
    results = []
    for i in range(batch_size_actual):
        # Get prediction for this frame
        pred = pred_batch[i].cpu().numpy().transpose(1,2,0)*255
        pred = np.array(pred, dtype=np.uint8)
        
        # Debug output
        if args.save_debug_frames and start_frame_count + i < 10:
            cv2.imwrite(f"{debug_dir}/frame_{start_frame_count+i:04d}_pred.jpg", pred)
            print(f"Batch frame {start_frame_count+i}: pred range [{pred_batch[i].min():.3f}, {pred_batch[i].max():.3f}]")
        
        # Apply to original frame
        frame_data = batch_data[i]
        frame_data.crop_img_ori[4:324, 4:324] = pred
        h, w = frame_data.crop_shape
        crop_img_ori = cv2.resize(frame_data.crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply parsing mask if needed
        if args.parsing and hasattr(frame_data, 'crop_parsing_img'):
            parsing_mask = (frame_data.crop_parsing_img == [0, 0, 255]).all(axis=2) | \
                          (frame_data.crop_parsing_img == [255, 255, 255]).all(axis=2)
            crop_img_ori[parsing_mask] = frame_data.crop_img_par[parsing_mask]
        
        # Update original image
        xmin, ymin, xmax, ymax = frame_data.bbox
        frame_data.img[ymin:ymax, xmin:xmax] = crop_img_ori
        
        if args.save_debug_frames and start_frame_count + i < 10:
            cv2.imwrite(f"{debug_dir}/frame_{start_frame_count+i:04d}_final.jpg", frame_data.img)
        
        results.append(frame_data.img)
    
    return results

# Initialize ping-pong
current_idx = args.start_frame
direction = 1

# Process frames
num_frames_to_process = min(audio_feats.shape[0], args.max_frames)
print(f"\nProcessing {num_frames_to_process} frames...")

if args.disable_batching:
    # Single frame processing
    for i in tqdm(range(num_frames_to_process)):
        # Get frame
        if frame_manager:
            frame_manager.current_idx = current_idx
            frame_manager.direction = direction
            frame_data = frame_manager.get_next_frame_pingpong()
            if frame_data is None:
                continue
            img = frame_data.frame.copy()
            img_idx = frame_data.frame_idx
        else:
            # Original image loading
            img_path = os.path.join(img_dir, f"{current_idx}.jpg")
            img = cv2.imread(img_path)
            img_idx = current_idx
        
        # Update ping-pong index
        current_idx += direction
        if current_idx >= 10479:
            current_idx = 10479
            direction = -1
        elif current_idx <= 0:
            current_idx = 0
            direction = 1
        
        # Load parsing if needed
        parsing = None
        if args.parsing:
            parsing_path = os.path.join(parsing_dir, f"{img_idx}.png")
            parsing = cv2.imread(parsing_path)
        
        # Get landmarks
        if landmark_manager:
            landmark_data = landmark_manager.get_landmark(img_idx)
            if landmark_data is None:
                continue
            lms = landmark_data.landmarks
            xmin, ymin, xmax, ymax = landmark_data.bbox
        else:
            # Original landmark loading
            lms_path = os.path.join(lms_dir, f"{img_idx}.lms")
            lms_list = []
            with open(lms_path, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    arr = line.split(" ")
                    arr = np.array(arr, dtype=np.float32)
                    lms_list.append(arr)
            lms = np.array(lms_list, dtype=np.int32)
            xmin = lms[1][0]
            ymin = lms[52][1]
            xmax = lms[31][0]
            width = xmax - xmin
            ymax = ymin + width
        
        # Get audio features
        audio_feat = get_audio_features(audio_feats, i)
        
        # Create frame data
        frame_info = FrameData(
            frame_idx=i,
            img_idx=img_idx,
            img=img,
            landmarks=lms,
            bbox=(xmin, ymin, xmax, ymax),
            audio_feat=audio_feat,
            parsing=parsing
        )
        
        # Process single frame
        result_img = process_single_frame(frame_info, i)
        if result_img is not None:
            video_writer.write(result_img)
else:
    # Batch processing
    batch_data = []
    frame_results = {}
    frames_processed = 0
    
    # Set up frame manager for batch mode
    if frame_manager:
        frame_manager.current_idx = current_idx
        frame_manager.direction = direction
    
    for i in tqdm(range(num_frames_to_process)):
        # Get frame
        if frame_manager:
            frame_data = frame_manager.get_next_frame_pingpong()
            if frame_data is None:
                continue
            img = frame_data.frame.copy()
            img_idx = frame_data.frame_idx
        else:
            # Original image loading
            img_path = os.path.join(img_dir, f"{current_idx}.jpg")
            img = cv2.imread(img_path)
            img_idx = current_idx
            
            # Update ping-pong index
            current_idx += direction
            if current_idx >= 10479:
                current_idx = 10479
                direction = -1
            elif current_idx <= 0:
                current_idx = 0
                direction = 1
        
        # Load parsing if needed
        parsing = None
        if args.parsing:
            parsing_path = os.path.join(parsing_dir, f"{img_idx}.png")
            parsing = cv2.imread(parsing_path)
        
        # Get landmarks
        if landmark_manager:
            landmark_data = landmark_manager.get_landmark(img_idx)
            if landmark_data is None:
                continue
            lms = landmark_data.landmarks
            xmin, ymin, xmax, ymax = landmark_data.bbox
        else:
            # Original landmark loading
            lms_path = os.path.join(lms_dir, f"{img_idx}.lms")
            lms_list = []
            with open(lms_path, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    arr = line.split(" ")
                    arr = np.array(arr, dtype=np.float32)
                    lms_list.append(arr)
            lms = np.array(lms_list, dtype=np.int32)
            xmin = lms[1][0]
            ymin = lms[52][1]
            xmax = lms[31][0]
            width = xmax - xmin
            ymax = ymin + width
        
        # Get audio features
        audio_feat = get_audio_features(audio_feats, i)
        
        # Create frame data
        frame_info = FrameData(
            frame_idx=i,
            img_idx=img_idx,
            img=img,
            landmarks=lms,
            bbox=(xmin, ymin, xmax, ymax),
            audio_feat=audio_feat,
            parsing=parsing
        )
        
        batch_data.append(frame_info)
        
        # Process batch when full
        if len(batch_data) >= batch_size:
            results = process_batch(batch_data, frames_processed)
            
            # Store results
            for j, result in enumerate(results):
                frame_results[batch_data[j].frame_idx] = result
            
            frames_processed += len(batch_data)
            batch_data.clear()
    
    # Process remaining frames
    if batch_data:
        results = process_batch(batch_data, frames_processed)
        for j, result in enumerate(results):
            frame_results[batch_data[j].frame_idx] = result
    
    # Write frames in order
    for i in range(num_frames_to_process):
        if i in frame_results:
            video_writer.write(frame_results[i])

# Cleanup
video_writer.release()
if frame_manager:
    frame_manager.cleanup()
if landmark_manager:
    landmark_manager.cleanup()

# Encode with audio
os.system(f"ffmpeg -i {save_path.replace('.mp4', 'temp.mp4')} -i {audio_path} -c:v libx264 -c:a aac -crf 20 {save_path} -y")
os.system(f"rm {save_path.replace('.mp4', 'temp.mp4')}")

print(f"\n[INFO] ===== Saved debug video to {save_path} =====")
if args.save_debug_frames:
    print(f"[INFO] Debug frames saved to {debug_dir}")