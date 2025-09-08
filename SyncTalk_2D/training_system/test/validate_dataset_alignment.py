# validate_dataset_alignment.py

import numpy as np
import torch
import cv2
import os
from datasetsss_328 import MyDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def validate_dataset_alignment(dataset_dir, mode='hubert', output_dir='./alignment_validation'):
    """Comprehensive validation of frame-audio alignment"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Dataset Alignment Validation")
    print("="*80)
    
    # Test 1: Basic length validation
    print("\n1. Testing dataset length calculations...")
    dataset = MyDataset(dataset_dir, mode)
    
    # Get actual counts
    actual_images = len([f for f in os.listdir(os.path.join(dataset_dir, "full_body_img")) if f.endswith('.jpg')])
    actual_audio_shape = dataset.audio_feats.shape
    
    print(f"   Actual image files: {actual_images}")
    print(f"   Actual audio frames: {actual_audio_shape[0]}")
    print(f"   Dataset.__len__(): {len(dataset)}")
    print(f"   Safe length (with padding): {dataset.safe_length if hasattr(dataset, 'safe_length') else 'N/A'}")
    
    # Test 2: Boundary testing
    print("\n2. Testing boundary conditions...")
    test_indices = [
        0,  # First frame
        len(dataset) // 2,  # Middle frame
        len(dataset) - 1,  # Last valid frame
    ]
    
    for idx in test_indices:
        try:
            img_concat, img_real, audio_feat = dataset[idx]
            print(f"   ✓ Index {idx}: Successfully loaded (audio shape: {audio_feat.shape})")
        except Exception as e:
            print(f"   ✗ Index {idx}: Failed - {str(e)}")
    
    # Test 3: Audio window extraction validation
    print("\n3. Testing audio window extraction...")
    for idx in [0, 8, len(dataset)-1]:
        audio_window = dataset.get_audio_features(dataset.audio_feats, idx)
        expected_size = 16
        if audio_window.shape[0] != expected_size:
            print(f"   ✗ Index {idx}: Wrong window size {audio_window.shape[0]} (expected {expected_size})")
        else:
            # Check if padding was applied correctly
            is_padded_left = idx < 8
            is_padded_right = idx > dataset.audio_feats.shape[0] - 9
            print(f"   ✓ Index {idx}: Correct size, padded_left={is_padded_left}, padded_right={is_padded_right}")
    
    # Test 4: Frame correspondence test
    print("\n4. Testing frame-audio correspondence...")
    # This test assumes your preprocessing pipeline saves frames in order
    # We'll check if frame indices match what we expect
    
    sample_indices = np.linspace(0, len(dataset)-1, min(10, len(dataset)), dtype=int)
    misalignments = []
    
    for idx in sample_indices:
        # Load the image and check its filename
        img_path = dataset.img_path_list[idx]
        expected_frame_num = int(os.path.basename(img_path).split('.')[0])
        
        # Check if this matches our index
        if expected_frame_num != idx:
            misalignments.append((idx, expected_frame_num))
            print(f"   ✗ Misalignment at index {idx}: expected frame {idx}, got frame {expected_frame_num}")
    
    if not misalignments:
        print("   ✓ All sampled frames correctly aligned")
    else:
        print(f"   ✗ Found {len(misalignments)} misalignments")
    
    # Test 5: Audio continuity test
    print("\n5. Testing audio feature continuity...")
    # Check if consecutive audio frames are reasonably similar (no jumps)
    
    if len(dataset) >= 10:
        consecutive_diffs = []
        for i in range(1, min(100, dataset.audio_feats.shape[0])):
            diff = np.mean(np.abs(dataset.audio_feats[i] - dataset.audio_feats[i-1]))
            consecutive_diffs.append(diff)
        
        mean_diff = np.mean(consecutive_diffs)
        max_diff = np.max(consecutive_diffs)
        
        # Check for sudden jumps (might indicate frame dropping)
        jumps = [i for i, d in enumerate(consecutive_diffs) if d > mean_diff * 3]
        
        print(f"   Mean consecutive frame difference: {mean_diff:.4f}")
        print(f"   Max consecutive frame difference: {max_diff:.4f}")
        if jumps:
            print(f"   ⚠ Found {len(jumps)} potential discontinuities at frames: {jumps[:5]}...")
    
    # Test 6: Visual validation (create a PDF report)
    print("\n6. Creating visual validation report...")
    create_visual_validation_report(dataset, output_dir)
    
    # Test 7: Stress test with edge cases
    print("\n7. Stress testing edge cases...")
    edge_cases = [
        ("Very early frame", 0),
        ("Early frame needing left padding", 5),
        ("Late frame needing right padding", len(dataset) - 5 if len(dataset) > 10 else len(dataset) - 1),
        ("Last valid frame", len(dataset) - 1),
    ]
    
    for case_name, idx in edge_cases:
        try:
            img_concat, img_real, audio_feat = dataset[idx]
            audio_raw = dataset.get_audio_features(dataset.audio_feats, idx)
            print(f"   ✓ {case_name} (idx={idx}): Success")
        except Exception as e:
            print(f"   ✗ {case_name} (idx={idx}): Failed - {str(e)}")
    
    print("\n" + "="*80)
    print("Validation complete! Check the visual report in:", output_dir)
    print("="*80)

def create_visual_validation_report(dataset, output_dir):
    """Create a visual PDF report showing frame-audio alignment"""
    
    pdf_path = os.path.join(output_dir, 'alignment_report.pdf')
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Dataset overview
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Dataset Alignment Validation Report', fontsize=16)
        
        # Plot 1: Length comparison
        ax = axes[0, 0]
        lengths = [len(dataset.img_path_list), dataset.audio_feats.shape[0], len(dataset)]
        labels = ['Image Files', 'Audio Frames', 'Dataset Length']
        ax.bar(labels, lengths)
        ax.set_ylabel('Count')
        ax.set_title('Frame Count Comparison')
        for i, v in enumerate(lengths):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        # Plot 2: Audio feature statistics
        ax = axes[0, 1]
        audio_means = np.mean(dataset.audio_feats, axis=1)
        ax.plot(audio_means[:500])  # First 500 frames
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Mean Audio Feature')
        ax.set_title('Audio Feature Continuity (first 500 frames)')
        
        # Plot 3: Random frame samples
        ax = axes[1, 0]
        ax.axis('off')
        sample_indices = np.random.randint(0, len(dataset), min(6, len(dataset)))
        
        sample_text = "Sample Frame Checks:\n\n"
        for idx in sample_indices:
            try:
                _, img_real, audio_feat = dataset[idx]
                sample_text += f"Frame {idx}: ✓ Shape {tuple(img_real.shape)}\n"
            except Exception as e:
                sample_text += f"Frame {idx}: ✗ Error: {str(e)[:30]}...\n"
        
        ax.text(0.1, 0.9, sample_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Random Access Test')
        
        # Plot 4: Alignment summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
Alignment Summary:
─────────────────
Total Images:     {len(dataset.img_path_list)}
Total Audio:      {dataset.audio_feats.shape[0]}
Safe Length:      {getattr(dataset, 'safe_length', 'N/A')}
Dataset Length:   {len(dataset)}

Audio Shape:      {dataset.audio_feats.shape}
Audio dtype:      {dataset.audio_feats.dtype}

Status: {'✓ ALIGNED' if len(dataset.img_path_list) == dataset.audio_feats.shape[0] else '⚠ MISALIGNED'}
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Visual examples
        if len(dataset) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            fig.suptitle('Sample Frames with Audio Windows', fontsize=16)
            
            sample_indices = [0, len(dataset)//4, len(dataset)//2, 
                            3*len(dataset)//4, len(dataset)-1, len(dataset)-1]
            
            for i, (ax, idx) in enumerate(zip(axes.flat, sample_indices)):
                try:
                    _, img_real, audio_feat = dataset[idx]
                    # Convert tensor to numpy for display
                    img_np = img_real.permute(1, 2, 0).numpy()
                    ax.imshow(img_np)
                    ax.set_title(f'Frame {idx}\nAudio: {audio_feat.shape}')
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'Frame {idx}')
                    ax.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    print(f"   Visual report saved to: {pdf_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='hubert', choices=['hubert', 'wenet', 'ave'])
    parser.add_argument('--output_dir', type=str, default='./alignment_validation')
    
    args = parser.parse_args()
    
    validate_dataset_alignment(args.dataset_dir, args.mode, args.output_dir)