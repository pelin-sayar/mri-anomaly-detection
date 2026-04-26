import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from scipy.ndimage import center_of_mass
from model import UNetWithOOD
from dataset import CoronaryArteryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "aoca_model_synced.pth.tar"
LOG_FILE = "alcapa_test_results.csv"
ALCAPA_DIR = "data/test_alcapa" 

def calculate_dice(preds, targets):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    intersection = np.sum(preds_flat[targets_flat > 0] > 0)
    return (2. * intersection) / (np.sum(preds_flat > 0) + np.sum(targets_flat > 0) + 1e-8)

def calculate_hybrid_label(preds_np, ent_score, ood_prob):
    """
    Detects anomalies using both anatomical distance and statistical uncertainty.
    """
    aorta_mask = (preds_np == 1)
    artery_mask = (preds_np == 2)
    
    # 1. Statistical Detection (The 'Smoke Detector')
    # If OOD is high or Entropy is high, it's anomalous even if segmentation fails.
    if ood_prob > 0.60 or ent_score > 0.08:
        if not np.any(artery_mask):
            return 0.0, "ANOMALOUS (High OOD/Statistical)"

    # 2. Anatomical Detection (The 'Ruler')
    if not np.any(aorta_mask) or not np.any(artery_mask):
        return 0.0, "UNKNOWN"

    ay, ax = center_of_mass(aorta_mask)
    artery_coords = np.argwhere(artery_mask)
    distances = np.sqrt(np.sum((artery_coords - [ay, ax])**2, axis=1))
    min_dist = np.min(distances)
    
    # Heuristic: > 55px away from Aorta is likely ALCAPA (Pulmonary origin)
    label = "ANOMALOUS (Distance)" if min_dist > 55 else "NORMAL"
    return min_dist, label

def get_medical_status(logits, threshold=0.08):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    uncertainty_score = torch.mean(entropy).item()
    status = "FLAG" if uncertainty_score > threshold else "CLEAR"
    return status, uncertainty_score, entropy.squeeze().cpu().numpy()

def log_to_csv(data):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def run_alcapa_test():
    model = UNetWithOOD(in_channels=1, out_channels=3).to(DEVICE)
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found.")
        return
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = CoronaryArteryDataset(base_dir=ALCAPA_DIR, split=None)
    
    if len(dataset) == 0:
        print(f"No data found in {ALCAPA_DIR}.")
        return

    print(f"Starting hybrid test on {len(dataset)} ALCAPA slices...")

    for idx in range(len(dataset)):
        image, mask = dataset[idx]
        image_tensor = image.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(image_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            ood_prob = torch.sigmoid(output[1]).item() if isinstance(output, tuple) else 0.0
            
            status, ent_score, uncertainty_map = get_medical_status(logits)
            
            # Use Soft Thresholding to find faint anomalous vessels
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            aorta_probs = probs[1]
            artery_probs = probs[2]
            
            preds = np.zeros_like(aorta_probs, dtype=np.uint8)
            preds[aorta_probs > 0.5] = 1
            preds[artery_probs > 0.12] = 2 # Lowered threshold specifically for ALCAPA
            
            dist, anatomy_label = calculate_hybrid_label(preds, ent_score, ood_prob)
            dice_val = calculate_dice(preds, mask.cpu().numpy())

        # Log to CSV
        log_to_csv({
            "Case_Index": idx,
            "Dice_Score": round(dice_val, 4),
            "Aorta_Distance_px": round(dist, 2),
            "Entropy_Score": round(ent_score, 4),
            "OOD_Prob": round(ood_prob, 4),
            "Anatomy_Result": anatomy_label
        })

        # Visualization for ALL flagged cases and every 10th slice for context
        if "🚨" in anatomy_label or idx % 10 == 0:
            plt.figure(figsize=(20, 7))
            plt.suptitle(f"Case {idx} | {anatomy_label}\nOOD: {ood_prob:.3f} | Max Art Prob: {np.max(artery_probs):.4f}", fontsize=12)
            
            plt.subplot(1, 4, 1); plt.imshow(image.squeeze().cpu().numpy(), cmap="gray"); plt.title("CT Scan"); plt.axis("off")
            plt.subplot(1, 4, 2); plt.imshow(mask.cpu().numpy(), cmap="jet"); plt.title("Ground Truth"); plt.axis("off")
            plt.subplot(1, 4, 3); plt.imshow(preds, cmap="jet"); plt.title("Pred (Soft Thresh)"); plt.axis("off")
            plt.subplot(1, 4, 4); plt.imshow(uncertainty_map, cmap="hot"); plt.title("Entropy Map"); plt.axis("off")
            
            os.makedirs("alcapa_results", exist_ok=True)
            plt.savefig(f"alcapa_results/case_{idx}_viz.png")
            plt.close()
            print(f"Slice {idx}: {anatomy_label}")

    print(f"Hybrid test complete. CSV: {LOG_FILE}")

if __name__ == "__main__":
    run_alcapa_test()