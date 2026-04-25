import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import os
from model import UNetWithOOD
from dataset import CoronaryArteryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "aoca_model_synced.pth.tar"
LOG_FILE = "midterm_results_log.csv"

def calculate_dice(preds, targets):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    intersection = np.sum((preds_flat > 0) & (targets_flat > 0))
    denom = np.sum(preds_flat > 0) + np.sum(targets_flat > 0)
    if denom == 0:
        return 1.0
    return (2. * intersection) / (denom + 1e-8)

def summarize_foreground(preds_np):
    foreground_pixels = int(np.sum(preds_np > 0))
    label = "PRESENT" if foreground_pixels > 0 else "ABSENT"
    return foreground_pixels, label

def get_medical_status(logits, threshold=0.08):
    probs = torch.sigmoid(logits)
    entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
    uncertainty_score = torch.mean(entropy).item()
    status = "HIGH_UNCERTAINTY" if uncertainty_score > threshold else "CLEAR"
    return status, uncertainty_score, entropy.squeeze(0).squeeze(0).cpu().numpy()

def log_to_csv(data):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def visualize_prediction():
    model = UNetWithOOD(in_channels=1, out_channels=1).to(DEVICE)
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
        ood_prob = output[1].item() if isinstance(output, tuple) else 0.0
        status, ent_score, uncertainty_map = get_medical_status(logits)
        preds = (torch.sigmoid(logits) > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        
        foreground_pixels, anatomy_label = summarize_foreground(preds)
        dice_val = calculate_dice(preds, mask.cpu().numpy())

    # Log results for report
    log_to_csv({
        "Case_Index": idx,
        "Dice_Score": round(dice_val, 4),
        "Aorta_Distance_px": foreground_pixels,
        "Entropy_Score": round(ent_score, 4),
        "OOD_Prob": round(ood_prob, 4),
        "Anatomy_Result": anatomy_label
    })

    # Plotting (keeping your 4-panel view)
    plt.figure(figsize=(20, 7))
    plt.suptitle(
        f"Case {idx} | Dice: {dice_val:.3f} | Foreground: {anatomy_label}\n"
        f"Status: {status} | Aux OOD head: {ood_prob:.3f} | Entropy: {ent_score:.4f}",
        fontsize=12,
    )
    plt.subplot(1, 4, 1); plt.imshow(image.squeeze().cpu().numpy(), cmap="gray"); plt.title("CT"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(mask.cpu().numpy(), cmap="jet"); plt.title("Truth"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(preds, cmap="jet"); plt.title("Pred"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(uncertainty_map, cmap="hot"); plt.title("Uncertainty"); plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_prediction()
