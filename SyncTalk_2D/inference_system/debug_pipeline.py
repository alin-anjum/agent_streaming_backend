# FILE: inference_system/debug_pipeline.py
# A focused script to numerically compare the CPU and GPU resizing operations.

import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Standard Imports ---
from unet_328 import Model
from utils import AudioEncoder, AudDataset, get_audio_features
from inference_system.core.video_frame_manager import AllFramesMemory
from inference_system.core.landmark_manager import LandmarkManager

# --- Debugging Configuration ---
os.makedirs("debug_dump", exist_ok=True)
DEBUG_FRAME_IDX = 100
DUMP_DIR = "debug_dump"

def get_data_for_debug(args, debug_audio_idx):
    """Loads all necessary data for a single frame, returning numpy arrays."""
    print("--- Loading data for debug ---")
    dataset_dir = os.path.join("./dataset", args.name)
    full_audio_feats = np.load(os.path.join(dataset_dir, "aud_ave.npy"))
    audio_feat_tensor = get_audio_features(full_audio_feats, debug_audio_idx)
    audio_feat_np = audio_feat_tensor.cpu().numpy() if isinstance(audio_feat_tensor, torch.Tensor) else audio_feat_tensor

    frame_manager = AllFramesMemory(os.path.join(dataset_dir, "full_body_img/"), from_images=True)
    frame_manager.initialize()
    landmark_manager = LandmarkManager(os.path.join(dataset_dir, "landmarks/"))
    landmark_manager.initialize()
    
    img_idx = debug_audio_idx
    frame_data = frame_manager.get_frame(img_idx)
    landmark_data = landmark_manager.get_landmark(img_idx)
    
    print(f"Loaded data for audio_idx={debug_audio_idx}, img_idx={img_idx}")
    return frame_data.frame, landmark_data.bbox

def generate_cpu_canvas(full_frame, bbox):
    """Generates and saves the 328x328 canvas using only CPU/cv2."""
    print("--- Generating CPU canvas ---")
    xmin, ymin, xmax, ymax = bbox
    crop_img = full_frame[ymin:ymax, xmin:xmax]
    resized_328 = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
    
    filename = os.path.join(DUMP_DIR, f"{DEBUG_FRAME_IDX}_1_Resized328_cpu.png")
    cv2.imwrite(filename, resized_328)
    print(f"  ... Saved {filename}")
    return resized_328

def generate_gpu_canvas(full_frame, bbox, device, align_corners_mode):
    """Generates and saves the 328x328 canvas using GPU/torch."""
    print(f"--- Generating GPU canvas (align_corners={align_corners_mode}) ---")
    full_frame_gpu = torch.from_numpy(full_frame.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    xmin, ymin, xmax, ymax = bbox
    crop_img_gpu = full_frame_gpu[:, :, ymin:ymax, xmin:xmax]
    
    # Use float32 for interpolation, as this is standard
    resized_328_gpu = F.interpolate(crop_img_gpu.float(), size=(328, 328), mode='bicubic', align_corners=align_corners_mode)
    
    # Convert back to uint8 for saving and comparison
    resized_328_np = torch.clamp(resized_328_gpu, 0, 255).byte().squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    filename = os.path.join(DUMP_DIR, f"{DEBUG_FRAME_IDX}_1_Resized328_gpu_ac{str(align_corners_mode)}.png")
    cv2.imwrite(filename, resized_328_np)
    print(f"  ... Saved {filename}")
    return resized_328_np

def compare_images(img1_path, img2_path):
    """Loads two images, calculates their difference, and prints stats."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print(f"Error: Could not load images for comparison. Check paths: {img1_path}, {img2_path}")
        return

    print(f"\n--- Comparing {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} ---")
    
    # Calculate absolute difference
    abs_diff = cv2.absdiff(img1, img2)
    
    # Enhance the difference to make it visible
    diff_enhanced = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    diff_filename = os.path.join(DUMP_DIR, f"DIFF_{os.path.basename(img1_path)}_vs_{os.path.basename(img2_path)}")
    cv2.imwrite(diff_filename, diff_enhanced)
    print(f"  ... Saved difference map to {diff_filename}")

    # Calculate statistics
    total_pixels = img1.size
    non_zero_pixels = np.count_nonzero(abs_diff)
    percentage_diff = (non_zero_pixels / total_pixels) * 100
    mean_abs_error = np.mean(abs_diff)
    max_abs_error = np.max(abs_diff)

    print(f"  - Pixels with any difference: {non_zero_pixels} / {total_pixels} ({percentage_diff:.2f}%)")
    print(f"  - Mean Absolute Error (per channel): {mean_abs_error:.4f}")
    print(f"  - Max Absolute Error (per channel): {max_abs_error}")
    
    if non_zero_pixels > 0:
        print("  - VERDICT: Images are DIFFERENT.")
    else:
        print("  - VERDICT: Images are IDENTICAL.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug pipeline resizing differences')
    parser.add_argument('--name', type=str, default="alin")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    full_frame, bbox = get_data_for_debug(args, DEBUG_FRAME_IDX)

    # 1. Generate CPU baseline
    cpu_canvas_path = os.path.join(DUMP_DIR, f"{DEBUG_FRAME_IDX}_1_Resized328_cpu.png")
    generate_cpu_canvas(full_frame.copy(), bbox)

    # 2. Generate GPU version with align_corners=False
    gpu_canvas_false_path = os.path.join(DUMP_DIR, f"{DEBUG_FRAME_IDX}_1_Resized328_gpu_acFalse.png")
    generate_gpu_canvas(full_frame.copy(), bbox, device, align_corners_mode=False)

    # 3. Generate GPU version with align_corners=True
    gpu_canvas_true_path = os.path.join(DUMP_DIR, f"{DEBUG_FRAME_IDX}_1_Resized328_gpu_acTrue.png")
    generate_gpu_canvas(full_frame.copy(), bbox, device, align_corners_mode=True)

    # 4. Compare the results numerically
    compare_images(cpu_canvas_path, gpu_canvas_false_path)
    compare_images(cpu_canvas_path, gpu_canvas_true_path)