import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNetWithOOD
from dataset import CoronaryArteryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
EPOCHS = 30
NUM_WORKERS = 4
PIN_MEMORY = True
POS_WEIGHT = 20.0

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight=None, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.smooth = smooth

    def forward(self, logits, targets):
        targets = targets.unsqueeze(1)
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth))
        return bce + dice_loss.mean()

def dice_from_logits(logits, targets, threshold=0.5, eps=1e-8):
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = targets.unsqueeze(1)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denominator = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (denominator + eps)
    empty_masks = denominator == 0
    dice[empty_masks] = 1.0
    return dice.mean()

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    model.train()
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).float()

        autocast_context = torch.amp.autocast(device_type="cuda") if DEVICE == "cuda" else nullcontext()
        with autocast_context:
            outputs = model(data)
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    dice_score = 0.0
    loop = tqdm(loader, desc="Validating")
    
    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).float()
            
            outputs = model(x)
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            dice_score += dice_from_logits(predictions, y).item()

    mean_dice = dice_score / max(len(loader), 1)
    print(f"Val Dice Score: {mean_dice:.4f}")
    model.train()
    return mean_dice

def main():
    model = UNetWithOOD(in_channels=1, out_channels=1).to(DEVICE)
    pos_weight = torch.tensor([POS_WEIGHT], device=DEVICE)
    loss_fn = DiceBCELoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.4),
    ])
    val_transform = A.Compose([])

    train_ds = CoronaryArteryDataset(base_dir="data/processed", split="train", transform=train_transform)
    val_ds = CoronaryArteryDataset(base_dir="data/processed", split="val", transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    best_dice = -1.0
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_dice = check_accuracy(val_loader, model, device=DEVICE)

        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(checkpoint, "aoca_model_synced.pth.tar")
            print(f"Saved new best checkpoint with Dice {best_dice:.4f}")

if __name__ == "__main__":
    main()
