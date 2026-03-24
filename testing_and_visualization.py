import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import os
from scipy.ndimage import center_of_mass
from model import UNetWithOOD
from dataset import CoronaryArteryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "aoca_model_synced.pth.tar"
LOG_FILE = "midterm_results_log.csv"

def calculate_dice(preds, targets):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    intersection = np.sum(preds_flat[targets_flat > 0] > 0)
    return (2. * intersection) / (np.sum(preds_flat > 0) + np.sum(targets_flat > 0) + 1e-8)

def calculate_anatomy_stats(preds_np):
    aorta_mask = (preds_np == 1)
    artery_mask = (preds_np == 2)
    if not np.any(aorta_mask) or not np.any(artery_mask):
        return 0.0, "UNKNOWN"
    ay, ax = center_of_mass(aorta_mask)
    artery_coords = np.argwhere(artery_mask)
    distances = np.sqrt(np.sum((artery_coords - [ay, ax])**2, axis=1))
    min_dist = np.min(distances)
    label = "🚨 ANOMALOUS" if min_dist > 55 else "✅ NORMAL"
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

def visualize_prediction():
    model = UNetWithOOD(in_channels=1, out_channels=3).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = CoronaryArteryDataset(base_dir="data/processed", split="val")
    idx = random.randint(0, len(dataset) - 1)
    image, mask = dataset[idx]
    image_tensor = image.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        logits = output[0] if isinstance(output, tuple) else output
        ood_prob = torch.sigmoid(output[1]).item() if isinstance(output, tuple) else 0.0
        status, ent_score, uncertainty_map = get_medical_status(logits)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
        
        dist, anatomy_label = calculate_anatomy_stats(preds)
        dice_val = calculate_dice(preds, mask.cpu().numpy())

    # Log results for report
    log_to_csv({
        "Case_Index": idx,
        "Dice_Score": round(dice_val, 4),
        "Aorta_Distance_px": round(dist, 2),
        "Entropy_Score": round(ent_score, 4),
        "OOD_Prob": round(ood_prob, 4),
        "Anatomy_Result": anatomy_label
    })

    # Plotting (keeping your 4-panel view)
    plt.figure(figsize=(20, 7))
    plt.suptitle(f"Case {idx} | Dice: {dice_val:.3f} | {anatomy_label}\nOOD: {ood_prob:.3f} | Entropy: {ent_score:.4f}", fontsize=12)
    plt.subplot(1, 4, 1); plt.imshow(image.squeeze().cpu().numpy(), cmap="gray"); plt.title("CT"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(mask.cpu().numpy(), cmap="jet"); plt.title("Truth"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(preds, cmap="jet"); plt.title("Pred"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(uncertainty_map, cmap="hot"); plt.title("Uncertainty"); plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_prediction()