import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from torch import optim
import random
import argparse
import time
from tqdm import tqdm
from collections import deque


class Dataset(object):
    def __init__(self, dataset_dir, mode):
        
        self.img_path_list = []
        self.lms_path_list = []
        
        print(f"Loading dataset from {dataset_dir}...")
        for i in range(len(os.listdir(dataset_dir+"/full_body_img/"))):

            img_path = os.path.join(dataset_dir+"/full_body_img/", str(i)+".jpg")
            lms_path = os.path.join(dataset_dir+"/landmarks/", str(i)+".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)
                
        if mode=="wenet":
            audio_feats_path = dataset_dir+"/aud_wenet.npy"
        if mode=="hubert":
            audio_feats_path = dataset_dir+"/aud_hu.npy"
        if mode=="ave":
            audio_feats_path = dataset_dir+"/aud_ave.npy"
        self.mode = mode
        print(f"Loading audio features ({mode})...")
        self.audio_feats = np.load(audio_feats_path)
        self.audio_feats = self.audio_feats.astype(np.float32)
        print(f"Dataset loaded: {len(self.img_path_list)} samples")
        
    def __len__(self):
        return self.audio_feats.shape[0]-1

    def get_audio_features(self, features, index):
        
        left = index - 8
        right = index + 8
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = torch.from_numpy(features[left:right])
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds
    
    def process_img(self, img, lms_path, img_ex, lms_path_ex):

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
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (328, 328), cv2.INTER_AREA)
        img_real = crop_img[4:324, 4:324].copy()
        img_real_ori = img_real.copy()
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        
        return img_real_T

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        audio_feat = self.get_audio_features(self.audio_feats, idx)
        
        if self.mode=="ave":
            audio_feat = audio_feat.reshape(32,16,16)
        else:
            audio_feat = audio_feat.reshape(32,32,32)
        y = torch.ones(1).float()
        
        return img_real_T, audio_feat, y


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.residual = residual
    
    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=(7, 7), stride=1, padding=3),
            Conv2d(32, 32, kernel_size=5, stride=2, padding=1),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )
            
        p1 = 128
        p2 = (1, 2)
        if mode == "hubert":
            p1 = 32
            p2 = (2, 2)
        if mode == "ave":
            p1 = 32
            p2 = 1
        self.audio_encoder = nn.Sequential(
            Conv2d(p1, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=p2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, face_sequences, audio_sequences):
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        
        return audio_embedding, face_embedding


logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss


def format_time(seconds):
    """Convert seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"


def train(save_dir, dataset_dir, mode):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("="*80)
    print(f"SyncNet Training")
    print(f"Mode: {mode}")
    print(f"Save Directory: {save_dir}")
    print(f"Dataset: {dataset_dir}")
    print("="*80)
    
    # Dataset and DataLoader with optimizations
    train_dataset = Dataset(dataset_dir, mode=mode)
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True,
        num_workers=8,          # Reduced from 16 for better stability
        pin_memory=True,        # Speed up GPU transfer
        prefetch_factor=2,      # Prefetch batches
        persistent_workers=True # Keep workers alive between epochs
    )
    
    # Model setup
    print("\nInitializing model...")
    model = SyncNet_color(mode).cuda()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.001)
    
    # Training metrics
    best_loss = float('inf')
    loss_history = deque(maxlen=50)  # Keep last 50 losses for smoothing
    
    print(f"\nStarting training for 100 epochs...")
    print(f"Batches per epoch: {len(train_data_loader)}")
    print(f"Total iterations: {100 * len(train_data_loader)}")
    print("-"*80)
    
    total_start_time = time.time()
    
    for epoch in range(100):
        epoch_start_time = time.time()
        epoch_losses = []
        
        # Progress bar for each epoch
        progress_bar = tqdm(train_data_loader, 
                   desc=f"Epoch {epoch+1:3d}/100", 
                   leave=True,
                   unit="batch",
                   dynamic_ncols=True,  # Dynamically adjust to terminal width
                   ascii=False,  # Use Unicode characters for smoother bar
                   position=0)  
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Fast GPU transfer with non_blocking
            imgT, audioT, y = batch
            imgT = imgT.cuda(non_blocking=True)
            audioT = audioT.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            
            # Forward pass
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            
            # Backward pass (NO zero_grad - keeping original behavior!)
            loss.backward()
            optimizer.step()
            
            # Record loss
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            loss_history.append(loss_val)
            
            # Update progress bar
            smooth_loss = np.mean(list(loss_history))
            progress_bar.set_postfix({
                'loss': f'{loss_val:.4f}',
                'smooth': f'{smooth_loss:.4f}',
                'best': f'{best_loss:.4f}'
            })
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        epoch_loss = np.mean(epoch_losses)
        total_time = time.time() - total_start_time
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   • Average Loss: {epoch_loss:.6f}")
        print(f"   • Epoch Time: {format_time(epoch_time)}")
        print(f"   • Total Time: {format_time(total_time)}")
        print(f"   • Speed: {len(train_data_loader)/epoch_time:.1f} batches/sec")
        
        # Save best model
        if epoch_loss < best_loss:
            improvement = best_loss - epoch_loss if best_loss != float('inf') else 0
            best_loss = epoch_loss
            save_path = os.path.join(save_dir, f'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"   ✅ New best model saved! (improvement: {improvement:.8f})")
        
        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'best_loss': best_loss
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")
        
        # ETA calculation
        avg_epoch_time = total_time / (epoch + 1)
        eta_seconds = avg_epoch_time * (100 - epoch - 1)
        print(f"   ⏱️  ETA: {format_time(eta_seconds)}")
        print("-"*80)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print(f"   • Total Time: {format_time(time.time() - total_start_time)}")
    print(f"   • Best Loss: {best_loss:.6f}")
    print(f"   • Models saved in: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='test', type=str)
    parser.add_argument('--dataset_dir', default='./dataset/May', type=str)
    parser.add_argument('--asr', default='ave', type=str)
    opt = parser.parse_args()
    
    train(opt.save_dir, opt.dataset_dir, opt.asr)