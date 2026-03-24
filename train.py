import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNetWithOOD
from dataset import CoronaryArteryDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 20
NUM_WORKERS = 4
PIN_MEMORY = True

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    model.train()
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE).long()

        # Fix for the UserWarning
        with torch.amp.autocast('cuda'):
            outputs = model(data)
            # Unpack the tuple if the model returns (seg, ood)
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        
        # Proper scaling and stepping
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def check_accuracy(loader, model, device="cuda"):
    model.eval()
    dice_score = 0
    loop = tqdm(loader, desc="Validating")
    
    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            y = y.to(device).long()
            
            # Unpack the tuple here
            outputs = model(x)
            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
            
            preds = torch.softmax(predictions, dim=1)
            preds = torch.argmax(preds, dim=1)
            
            # Simple Dice calculation for multi-class
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Val Dice Score: {dice_score/len(loader):.4f}")
    model.train()

def main():
    model = UNetWithOOD(in_channels=1, out_channels=3).to(DEVICE)
    weights = torch.tensor([0.1, 1.0, 5.0]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    train_ds = CoronaryArteryDataset(base_dir="data/processed", split="train")
    val_ds = CoronaryArteryDataset(base_dir="data/processed", split="val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(checkpoint, "aoca_model_synced.pth.tar")
        
        check_accuracy(val_loader, model, device=DEVICE)

if __name__ == "__main__":
    main()