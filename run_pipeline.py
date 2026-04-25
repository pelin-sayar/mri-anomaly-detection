import torch
import numpy as np
import os
from model import UNetWithOOD
from inference import predict_and_flag
from viz import show_cam_heatmap, show_uncertainty_overlay
from utils import load_checkpoint
from eval import generate_metrics

# --- Configuration ---
CHECKPOINT_PATH = "aoca_model_synced.pth.tar"
# For a single run, we use one image; for a "batch" run, we point to a directory
SAMPLE_IMAGE = "data/processed/val/images"
SAMPLE_MASK = "data/processed/val/masks"

def _resolve_sample(directory):
    if os.path.isdir(directory):
        candidates = sorted([f for f in os.listdir(directory) if f.endswith(".npy")])
        if not candidates:
            raise FileNotFoundError(f"No .npy files found in {directory}")
        return os.path.join(directory, candidates[0])
    return directory

def run_full_analysis():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 1. Load Model
    model = UNetWithOOD(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    load_checkpoint(checkpoint, model)

    image_path = _resolve_sample(SAMPLE_IMAGE)
    mask_path = _resolve_sample(SAMPLE_MASK)

    # 2. Run Inference
    result = predict_and_flag(
        model, image_path, device, entropy_thresh=0.4
    )
    seg_mask = result["seg_mask"]
    entropy_map = result["entropy_map"]
    
    # 3. Load Ground Truth for Evaluation
    if os.path.exists(mask_path):
        ground_truth = (np.load(mask_path) > 0).astype(np.uint8)
        eval_results = generate_metrics([ground_truth], [seg_mask])
        
        print("\n" + "="*40)
        print("EVALUATION METRICS (Accuracy vs Ground Truth)")
        print(f"Foreground Dice: {eval_results['dice_mean']:.4f}")
        print(f"Dice Std:        {eval_results['dice_std']:.4f}")
        print("="*40)
    else:
        print("Warning: Ground truth mask not found. Skipping Dice calculation.")
    
    print("\n" + "="*40)
    print(f"CLINICAL FINDINGS")
    print(f"High Uncertainty Flag: {result['uncertainty_flag']}")
    print(f"Mean Entropy: {result['mean_entropy']:.4f}")
    if result["aux_ood_prob"] is not None:
        print(f"Aux OOD Head Output: {result['aux_ood_prob']:.4f} (not supervised)")
    print(f"Predicted Foreground Pixels: {int(seg_mask.sum())}")
    print("="*40 + "\n")

    # 5. Visualizations
    raw_img = np.load(image_path).astype(np.float32)
    show_uncertainty_overlay(raw_img, entropy_map)
    show_cam_heatmap(model, raw_img, model.final_conv, device, class_index=0)

if __name__ == "__main__":
    run_full_analysis()
