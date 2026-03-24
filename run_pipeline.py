import torch
import numpy as np
import os
from model import UNetWithOOD
from inference import predict_and_flag, calculate_aorta_to_artery_distance
from viz import show_cam_heatmap, show_uncertainty_overlay
from utils import load_checkpoint
from eval import generate_metrics  # New Import

# --- Configuration ---
CHECKPOINT_PATH = "my_checkpoint.pth.tar"
# For a single run, we use one image; for a "batch" run, we point to a directory
SAMPLE_IMAGE = "data/processed/images/test_slice_001.npy"
SAMPLE_MASK = "data/processed/masks/test_slice_001.npy" # Ground Truth
PIXEL_SPACING = 0.5  # mm per pixel

def run_full_analysis():
    device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model
    model = UNetWithOOD(in_channels=1, out_channels=3).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    load_checkpoint(checkpoint, model)

    # 2. Run Inference
    # seg_mask will have values 0, 1, 2
    seg_mask, ood_flag, entropy_map, ood_prob = predict_and_flag(
        model, SAMPLE_IMAGE, device, entropy_thresh=0.4
    )
    
    # 3. Load Ground Truth for Evaluation
    if os.path.exists(SAMPLE_MASK):
        ground_truth = np.load(SAMPLE_MASK).astype(np.uint8)
        # We wrap them in lists because generate_metrics expects a batch/list
        eval_results = generate_metrics([ground_truth], [seg_mask])
        
        print("\n" + "="*40)
        print("EVALUATION METRICS (Accuracy vs Ground Truth)")
        print(f"Artery Dice Score: {eval_results['dice_artery_mean']:.4f}  <-- KEY KPI")
        print(f"Aorta Dice Score:  {eval_results['dice_aorta_mean']:.4f}")
        print(f"Overall Dice:      {eval_results['dice_overall_mean']:.4f}")
        print("="*40)
    else:
        print("Warning: Ground truth mask not found. Skipping Dice calculation.")

    # 4. Clinical Metrics
    dist_px = calculate_aorta_to_artery_distance(seg_mask)
    
    print("\n" + "="*40)
    print(f"CLINICAL FINDINGS")
    print(f"OOD Flagged (Anomaly): {ood_flag}")
    print(f"Model Confidence Score: {1 - ood_prob:.4f}")
    if dist_px:
        print(f"Aorta-Artery Distance: {dist_px * PIXEL_SPACING:.2f} mm")
    print("="*40 + "\n")

    # 5. Visualizations
    raw_img = np.load(SAMPLE_IMAGE).astype(np.float32)
    show_uncertainty_overlay(raw_img, entropy_map)
    show_cam_heatmap(model, raw_img, model.final_conv, device)

if __name__ == "__main__":
    run_full_analysis()