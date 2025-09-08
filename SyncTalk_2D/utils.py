
from typing import List
import librosa
import librosa.filters
from scipy import signal
from os.path import basename
import numpy as np
import cv2
import os, math, warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


try:
    from skimage.metrics import structural_similarity as ssim_fn
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    warnings.warn("scikit-image not found → SSIM will not be computed")

try:
    from sewar.full_ref import vifp       # VIF-p
    _HAS_VIF = True
except ImportError:
    _HAS_VIF = False
    warnings.warn("sewar not found → VIF will not be computed")

try:
    import face_alignment
    _FA = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                       device='cpu', flip_input=False)
    _HAS_FA = True
except Exception:
    _HAS_FA = False
    warnings.warn("face_alignment not available → LMD will not be computed")

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out
    
def get_audio_features(features_np: np.ndarray, index: int):
    """
    Extracts a 16-frame window of audio features from a numpy array.
    This version assumes the input is always a numpy array.
    """
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0

    num_frames = features_np.shape[0]

    if left < 0:
        pad_left = -left
        left = 0
    
    if right > num_frames:
        pad_right = right - num_frames
        right = num_frames
    
    auds_slice = features_np[left:right]
    
    if pad_left > 0:
        padding = np.zeros((pad_left, auds_slice.shape[1]), dtype=auds_slice.dtype)
        auds = np.concatenate([padding, auds_slice], axis=0)
    else:
        auds = auds_slice

    if pad_right > 0:
        padding = np.zeros((pad_right, auds_slice.shape[1]), dtype=auds.dtype)
        auds = np.concatenate([auds, padding], axis=0)
    
    # Return a numpy array, the GPU worker will convert it to a tensor.
    return auds


def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]


def preemphasis(wav, k):
    return signal.lfilter([1, -k], [1], wav)


def melspectrogram(wav):
    D = _stft(preemphasis(wav, 0.97))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20

    return _normalize(S)


def _stft(y):
    return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)


def _linear_to_mel(spectogram):
    global _mel_basis
    _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    return librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)


def _amp_to_db(x):
    min_level = np.exp(-5 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    return np.clip((2 * 4.) * ((S - -100) / (--100)) - 4., -4., 4.)


class AudDataset(object):
    def __init__(self, wavpath):
        wav = load_wav(wavpath, 16000)

        self.orig_mel = melspectrogram(wav).T
        self.data_len = int((self.orig_mel.shape[0] - 16) / 80. * float(25)) + 2

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(25)))

        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            # print(end_idx, spec.shape[0])
            end_idx = spec.shape[0]
            start_idx = end_idx - 16

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        if (mel.shape[0] != 16):
            raise Exception('mel.shape[0] != 16')
        mel = torch.FloatTensor(mel.T).unsqueeze(0)

        return mel

def generate_sample_predictions(net, dataset, epoch, logger, num_samples=3, out_dir="./predictions"):
    """
    Generate visual comparison with multiple samples for current epoch
    
    Parameters
    ----------
    net : torch.nn.Module
        The trained model
    dataset : Dataset
        Dataset to sample from
    epoch : int
        Current epoch number
    logger : TrainingLogger
        Logger instance for logging messages
    num_samples : int
        Number of samples to generate (default: 3)
    out_dir : str
        Output directory for comparison images
    
    Returns
    -------
    path : str
        Path to saved comparison image
    metrics : dict
        Average metrics across all samples
    """
    from torch.cuda.amp import autocast
    import torch
    import random
    
    net.eval()
    
    # Get random samples
    pred_tensors = []
    real_tensors = []
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        img_concat, img_real, audio_feat = dataset[idx]
        
        img_concat_single = img_concat.unsqueeze(0).cuda()
        audio_feat_single = audio_feat.unsqueeze(0).cuda()

        with torch.no_grad(), autocast():
            pred = net(img_concat_single, audio_feat_single)[0]
        
        pred_tensors.append(pred.cpu())
        real_tensors.append(img_real)
    
    # Generate comparison image
    path, metrics = save_prediction(pred_tensors, real_tensors, epoch=epoch, out_dir=out_dir)
    
    # Log results
    logger.log_message(f"Saved comparison image: {path}")
    metrics_str = ", ".join([
        f"PSNR={metrics.get('PSNR', 0):.2f}",
        f"SSIM={metrics.get('SSIM', 0):.4f}",
        f"VIF={metrics.get('VIF', 0):.2f}"
    ])
    logger.log_message(f"Average metrics: {metrics_str}")
    
    return path, metrics

def save_prediction(pred_tensors: List[torch.Tensor],
                    real_tensors: List[torch.Tensor],
                    epoch: int,
                    out_dir: str = "./predictions",
                    left_label: str = "Prediction",
                    right_label: str = "Ground truth",
                    jpeg_quality: int = 95,
                    gutter: int = 8,
                    row_spacing: int = 4,
                    border_thickness: int = 4,
                    border_color: tuple = (255, 0, 0)):  # Red border (BGR format)
    """
    Build side-by-side canvas with 3 examples stacked vertically, compute per-frame quality metrics, save JPEG.

    Parameters
    ----------
    pred_tensors : List[torch.Tensor]  list of 3 prediction tensors
    real_tensors : List[torch.Tensor]  list of 3 ground truth tensors
    epoch : int
    out_dir : str
    left_label : str
    right_label : str
    jpeg_quality : int
    gutter : int                       horizontal spacing between pred/real
    row_spacing : int                  vertical spacing between rows
    border_thickness : int             thickness of border around images
    border_color : tuple               BGR color for border

    Returns
    -------
    path : str                absolute path of written file
    metrics : dict[str, float] averaged metrics across all 3 examples
    """
    
    assert len(pred_tensors) == 3 and len(real_tensors) == 3, "Need exactly 3 prediction and 3 real tensors"

    # ───────── tensor → uint8 HWC ────────
    def to_uint8(t: torch.Tensor) -> np.ndarray:
        if t.is_cuda:
            t = t.cpu()
        arr = t.numpy()
        if arr.ndim == 3:                  # CHW → HWC
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return arr.astype(np.uint8)

    # ───────── add border to image ────────
    def add_border(img: np.ndarray, thickness: int, color: tuple) -> np.ndarray:
        """Add a colored border around the image"""
        return cv2.copyMakeBorder(img, thickness, thickness, thickness, thickness,
                                  cv2.BORDER_CONSTANT, value=color)

    # Convert all tensors to uint8
    preds = [to_uint8(pred) for pred in pred_tensors]
    reals = [to_uint8(real) for real in real_tensors]

    # Get dimensions from first image (all should be same)
    img_h, img_w, _ = preds[0].shape
    
    # Add borders to all images
    preds_bordered = [add_border(pred, border_thickness, border_color) for pred in preds]
    reals_bordered = [add_border(real, border_thickness, border_color) for real in reals]
    
    # Dimensions with borders
    bordered_h = img_h + 2 * border_thickness
    bordered_w = img_w + 2 * border_thickness

    # ───────── calculate metrics (on original images without borders) ────────
    def psnr(a, b) -> float:
        mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
        return float("inf") if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))

    all_metrics = []
    for pred, real in zip(preds, reals):
        metrics = {"PSNR": psnr(pred, real)}
        
        if _HAS_SKIMAGE:
            metrics["SSIM"] = ssim_fn(pred, real, channel_axis=-1, data_range=255)
        
        if _HAS_VIF:
            metrics["VIF"] = vifp(pred, real)    # 0–1, higher better
        
        if _HAS_FA:
            def lip_lmd(a, b):
                # expects uint8 BGR → convert to RGB for face_alignment
                for img in (a, b):
                    if img.ndim != 3 or img.shape[2] != 3:
                        return None
                la = _FA.get_landmarks(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
                lb = _FA.get_landmarks(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
                if not la or not lb:
                    return None
                la, lb = la[0], lb[0]           # (68,2)
                mouth = slice(48, 68)
                eye_d = np.linalg.norm(la[36] - la[45])
                if eye_d < 1e-6:
                    return None
                return float(np.mean(np.linalg.norm(la[mouth] - lb[mouth], axis=1)) / eye_d)

            lmd_val = lip_lmd(pred, real)
            if lmd_val is not None:
                metrics["LMD"] = lmd_val
        
        all_metrics.append(metrics)

    # ───────── calculate average metrics ────────
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m and m[key] is not None]
        if values:
            avg_metrics[key] = sum(values) / len(values)

    # ───────── build canvas ────────────────
    font, fscale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    pad, label_h, foot_h = 4, 20 + 8, 24 + 8              # paddings / banners
    
    # Canvas dimensions
    single_row_h = bordered_h
    total_content_h = 3 * single_row_h + 2 * row_spacing  # 3 rows + 2 spacings
    canvas_h = label_h + total_content_h + foot_h
    canvas_w = 2 * bordered_w + gutter
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)

    # ───────── place images with borders ────────────────
    current_y = label_h
    for i, (pred, real) in enumerate(zip(preds_bordered, reals_bordered)):
        # Place prediction and real images
        canvas[current_y:current_y + single_row_h, :bordered_w] = pred
        canvas[current_y:current_y + single_row_h, bordered_w + gutter:] = real
        
        # Add small sample number on the left
        sample_text = f"{i+1}"
        cv2.putText(canvas, sample_text, (border_thickness + 2, current_y + border_thickness + 15), 
                   font, 0.4, (128, 128, 128), 1, cv2.LINE_AA)
        
        current_y += single_row_h + row_spacing

    # ───────── add labels ────────────────
    def put_center(text, y0, x0, width):
        (tw, th), _ = cv2.getTextSize(text, font, fscale, thick)
        x = int(x0 + (width - tw) / 2)
        y = int(y0 + pad + th)
        cv2.putText(canvas, text, (x, y), font, fscale, (0, 0, 0),
                    thick, cv2.LINE_AA)

    put_center(left_label, 0, 0, bordered_w)
    put_center(right_label, 0, bordered_w + gutter, bordered_w)

    # ───────── add footer with average metrics ────────
    footer_txt = "Avg: " + "  ".join(f"{k}: {v:.2f}" if k in ("PSNR", "VIF")
                                      else f"{k}: {v:.4f}"
                                      for k, v in avg_metrics.items())
    put_center(footer_txt, label_h + total_content_h, 0, canvas_w)

    # ───────── save ──────────────────────
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(out_dir, f"epoch_{epoch:04d}.jpg"))
    cv2.imwrite(path, canvas,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                 cv2.IMWRITE_JPEG_OPTIMIZE, 1])

    return path, avg_metrics
