import faulthandler
import time
faulthandler.enable()

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm


def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')
    
def extract_images(path):
    full_body_dir = path.replace(path.split("/")[-1], "full_body_img")
    if not os.path.exists(full_body_dir):
        os.mkdir(full_body_dir)
    
    counter = 0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps != 25:
        print(f"Converting video from {fps} fps to 25 fps...")
        # High quality conversion to 25fps using ffmpeg
        cmd = f'ffmpeg -i {path} -vf "fps=25" -c:v libx264 -c:a aac {path.replace(".mp4", "_25fps.mp4")}'
        os.system(cmd)
        path = path.replace(".mp4", "_25fps.mp4")
        cap.release()  # Release the old capture
        cap = cv2.VideoCapture(path)  # Open the new converted video
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        cap.release()
        raise ValueError("Your video fps should be 25!!!")
    
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    
    print(f"Video info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {video_duration:.2f} seconds")
    print(f"  - Output directory: {full_body_dir}")
    
    # High/highest quality JPEG parameters
    encode_params = [
        int(cv2.IMWRITE_JPEG_QUALITY),         95,   # 0–100
        int(cv2.IMWRITE_JPEG_LUMA_QUALITY),    100,  # Y  component
        int(cv2.IMWRITE_JPEG_CHROMA_QUALITY),  100,  # Cb/Cr components
        int(cv2.IMWRITE_JPEG_PROGRESSIVE),     1,
        int(cv2.IMWRITE_JPEG_OPTIMIZE),        1,
    ]
    
    print("\nExtracting frames...")
    start_time = time.time()
    
    # Create progress bar
    with tqdm(total=total_frames, 
              desc="Extracting frames", 
              unit="frame",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame with high quality
            frame_path = os.path.join(full_body_dir, f"{counter}.jpg")
            success = cv2.imwrite(frame_path, frame, encode_params)
            
            if not success:
                print(f"\nWarning: Failed to write frame {counter}")
            
            counter += 1
            
            # Update progress bar with additional info
            if counter % 100 == 0 or counter == total_frames:  # Update every 100 frames
                elapsed_time = time.time() - start_time
                fps_processing = counter / elapsed_time if elapsed_time > 0 else 0
                
                pbar.set_postfix({
                    'Saved': f"{counter}",
                    'Processing_FPS': f"{fps_processing:.1f}",
                    'Time_Elapsed': f"{elapsed_time:.1f}s"
                })
            
            pbar.update(1)
    
    cap.release()
    
    # Final summary
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = counter / total_time if total_time > 0 else 0
    
    print(f"\n{'='*60}")
    print("FRAME EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total frames extracted: {counter}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average processing speed: {avg_fps:.1f} frames/second")
    print(f"Output directory: {full_body_dir}")
    
    # Check if all frames were extracted
    if counter == total_frames:
        print("✓ All frames extracted successfully!")
    else:
        print(f"⚠ Warning: Expected {total_frames} frames, but extracted {counter}")
    
    print(f"{'='*60}")
    
    return full_body_dir, counter
        
def get_audio_feature(wav_path):
    
    print("Extracting audio features...")
    os.system("python ./data_utils/ave/test_w2l_audio.py --wav_path "+wav_path)
    
NUM_LANDMARK_POINTS = 68 * 2    # = 136 values written, change if your model differs

def get_landmark(video_path: str, landmarks_dir: str):
    """
    Detects landmarks for every frame image under full_body_img/, but
    skips frames whose .lms file already exists and looks complete.
    If the program crashes you can run it again and it will resume.
    """

    print('[INFO] detecting landmarks (resumable)…')

    # directory that already contains the extracted JPG frames
    full_img_dir = video_path.replace(video_path.split('/')[-1], 'full_body_img')

    # lazy import (keeps start-up fast if cuda init is heavy)
    from get_landmark import Landmark
    landmark_detector = Landmark()

    # work on a deterministic order so “resume” is obvious
    frame_list = sorted(f for f in os.listdir(full_img_dir) if f.lower().endswith('.jpg'))

    for img_name in tqdm(frame_list):
        img_path  = os.path.join(full_img_dir, img_name)
        lms_path  = os.path.join(landmarks_dir, img_name[:-4] + '.lms')
        tmp_path  = lms_path + '.tmp'

        # 1) Skip if a good file already exists
        if os.path.exists(lms_path):
            with open(lms_path, 'r') as f:
                if sum(1 for _ in f) >= NUM_LANDMARK_POINTS // 2:   # each line contains x y
                    continue            # looks complete → skip

        # 2) Clean up any half-written temp file from the previous crash
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # 3) Run detection
        try:
            pre_landmark, x1, y1 = landmark_detector.detect(img_path)
        except Exception as e:
            print(f'[WARN] landmark failed for {img_name}: {e}')
            continue                    # move on to next frame

        # 4) Write to temp, then rename → atomic
        try:
            with open(tmp_path, 'w') as f:
                for p in pre_landmark:
                    x, y = int(p[0] + x1), int(p[1] + y1)
                    f.write(f'{x} {y}\n')
            os.replace(tmp_path, lms_path)        # atomic on same filesystem
        except OSError as e:
            print(f'[ERROR] could not write {lms_path}: {e}')
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    extract_audio(opt.path, wav_path)
    extract_images(opt.path)
    get_landmark(opt.path, landmarks_dir)
    get_audio_feature(wav_path)
    
    