import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet_328 import Model
from tqdm import tqdm
from utils import AudioEncoder, AudDataset, get_audio_features
# from unet2 import Model
# from unet_att import Model

import time
from collections import defaultdict
from contextlib import contextmanager

# Performance profiling utilities
class PerformanceProfiler:
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_timers = {}
    
    def start_timer(self, name):
        self.current_timers[name] = time.time()
    
    def end_timer(self, name):
        if name in self.current_timers:
            elapsed = time.time() - self.current_timers[name]
            self.timings[name].append(elapsed)
            del self.current_timers[name]
            return elapsed
        return 0
    
    @contextmanager
    def timer(self, name):
        self.start_timer(name)
        yield
        self.end_timer(name)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*60)
        
        total_time = sum(sum(times) for times in self.timings.values())
        
        for name, times in sorted(self.timings.items()):
            total = sum(times)
            avg = total / len(times) if times else 0
            percentage = (total / total_time * 100) if total_time > 0 else 0
            
            print(f"\n{name}:")
            print(f"  Total time: {total:.3f}s ({percentage:.1f}%)")
            print(f"  Average time: {avg:.3f}s")
            print(f"  Number of calls: {len(times)}")
            if len(times) > 1:
                print(f"  Min time: {min(times):.3f}s")
                print(f"  Max time: {max(times):.3f}s")
        
        print(f"\nTotal execution time: {total_time:.3f}s")
        print("="*60)

# Initialize profiler
profiler = PerformanceProfiler()

parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="May")
parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--parsing', type=bool, default=False)
args = parser.parse_args()

# Start overall timing
overall_start_time = time.time()

with profiler.timer("Setup and initialization"):
    checkpoint_path = os.path.join("./checkpoint", args.name)
    # 获取checkpoint_path目录下数字最大的.pth文件，按照int排序
    checkpoint = os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path), key=lambda x: int(x.split(".")[0]))[-1])
    print(checkpoint)
    save_path = os.path.join("./result", args.name+"_"+args.audio_path.split("/")[-1].split(".")[0]+"_"+checkpoint.split("/")[-1].split(".")[0]+".mp4")
    dataset_dir = os.path.join("./dataset", args.name)
    audio_path = args.audio_path
    mode = args.asr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Audio feature extraction
with profiler.timer("Audio encoder model loading"):
    model = AudioEncoder().to(device).eval()
    ckpt = torch.load('model/checkpoints/audio_visual_encoder.pth')
    model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})

with profiler.timer("Audio dataset preparation"):
    dataset = AudDataset(audio_path)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

with profiler.timer("Audio feature extraction"):
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

with profiler.timer("Directory setup and video writer initialization"):
    img_dir = os.path.join(dataset_dir, "full_body_img/")
    lms_dir = os.path.join(dataset_dir, "landmarks/")
    len_img = len(os.listdir(img_dir)) - 1
    exm_img = cv2.imread(img_dir+"0.jpg")
    h, w = exm_img.shape[:2]
    if args.parsing:
        parsing_dir = os.path.join(dataset_dir, "parsing/")

    if mode=="hubert" or mode=="ave":
        video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"), cv2.VideoWriter_fourcc('M','J','P', 'G'), 25, (w, h))
    if mode=="wenet":
        video_writer = cv2.VideoWriter(save_path.replace(".mp4", "temp.mp4"), cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (w, h))
    step_stride = 0
    img_idx = 0

with profiler.timer("Main model loading"):
    net = Model(6, mode).cuda()
    net.load_state_dict(torch.load(checkpoint))
    net.eval()

# Main inference loop
print(f"Processing {audio_feats.shape[0]} frames...")
for i in tqdm(range(audio_feats.shape[0])):
    if img_idx>len_img - 1:
        step_stride = -1
    if img_idx<1:
        step_stride = 1
    img_idx += step_stride
    
    with profiler.timer("Image loading"):
        img_path = img_dir + str(img_idx+args.start_frame)+'.jpg'
        img = cv2.imread(img_path)
        
        if args.parsing:
            parsing_path = parsing_dir + str(img_idx+args.start_frame)+'.png'
            parsing = cv2.imread(parsing_path)
    
    with profiler.timer("Landmark loading and processing"):
        lms_path = lms_dir + str(img_idx+args.start_frame)+'.lms'
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
    
    with profiler.timer("Image preprocessing"):
        crop_img = img[ymin:ymax, xmin:xmax]  
        crop_img_par = crop_img.copy()  
        if args.parsing:
            crop_parsing_img = parsing[ymin:ymax, xmin:xmax] 
        h, w = crop_img.shape[:2]
        crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
        crop_img_ori = crop_img.copy()
        img_real_ex = crop_img[4:324, 4:324].copy()
        img_real_ex_ori = img_real_ex.copy()
        img_masked = cv2.rectangle(img_real_ex_ori,(5,5,310,305),(0,0,0),-1)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
    
    with profiler.timer("Audio feature preparation"):
        audio_feat = get_audio_features(audio_feats, i)
        if mode=="hubert":
            audio_feat = audio_feat.reshape(32,32,32)
        if mode=="wenet":
            audio_feat = audio_feat.reshape(256,16,32)
        if mode=="ave":
            audio_feat = audio_feat.reshape(32,16,16)
        audio_feat = audio_feat[None]
        audio_feat = audio_feat.cuda()
        img_concat_T = img_concat_T.cuda()
    
    with profiler.timer("Model inference"):
        with torch.no_grad():
            pred = net(img_concat_T, audio_feat)[0]
    
    with profiler.timer("Post-processing"):
        pred = pred.cpu().numpy().transpose(1,2,0)*255
        pred = np.array(pred, dtype=np.uint8)
        crop_img_ori[4:324, 4:324] = pred
        crop_img_ori = cv2.resize(crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC)
        if args.parsing:
            parsing_mask = (crop_parsing_img == [0, 0, 255]).all(axis=2) | (crop_parsing_img == [255, 255, 255]).all(axis=2)
            crop_img_ori[parsing_mask] = crop_img_par[parsing_mask]
        img[ymin:ymax, xmin:xmax] = crop_img_ori
    
    with profiler.timer("Video writing"):
        video_writer.write(img)

with profiler.timer("Video writer release"):
    video_writer.release()

with profiler.timer("FFmpeg encoding"):
    os.system(f"ffmpeg -i {save_path.replace('.mp4', 'temp.mp4')} -i {audio_path} -c:v libx264 -c:a aac -crf 20 {save_path} -y")
    os.system(f"rm {save_path.replace('.mp4', 'temp.mp4')}")

print(f"[INFO] ===== save video to {save_path} =====")

# Print performance summary
overall_time = time.time() - overall_start_time
print(f"\nOverall execution time: {overall_time:.2f} seconds")
profiler.print_summary()

# Additional detailed analysis
print("\n" + "="*60)
print("PERFORMANCE ANALYSIS")
print("="*60)

# Calculate per-frame statistics
total_frames = audio_feats.shape[0]
inference_time = sum(profiler.timings["Model inference"])
preprocessing_time = sum(profiler.timings["Image preprocessing"])
postprocessing_time = sum(profiler.timings["Post-processing"])

print(f"\nPer-frame statistics:")
print(f"  Average inference time: {inference_time/total_frames*1000:.2f} ms/frame")
print(f"  Average preprocessing time: {preprocessing_time/total_frames*1000:.2f} ms/frame")
print(f"  Average postprocessing time: {postprocessing_time/total_frames*1000:.2f} ms/frame")
print(f"  Average FPS (inference only): {total_frames/inference_time:.2f}")
print(f"  Average FPS (total processing): {total_frames/(inference_time+preprocessing_time+postprocessing_time):.2f}")