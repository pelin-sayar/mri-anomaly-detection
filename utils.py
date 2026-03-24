import torch 
import torchvision
import dataset from ImageCAS_Dataset
import torch.utils.data from DataLoader
import os
import nibabel as nib
import numpy as np

def create_paired_slices(image_nifti, label_nifti, output_dir, case_id):
    # Create subfolders for organization
    img_out = os.path.join(output_dir, 'images')
    lbl_out = os.path.join(output_dir, 'labels')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # 1. Load both 3D volumes
    img_data = nib.load(image_nifti).get_fdata()
    lbl_data = nib.load(label_nifti).get_fdata()

    #basic Normalization for CT scan
    img_data = np.clip(img_data, -120, 800)
    img_data = (img_data + 120) / (800 + 120)

    #loop through the z-axis (the slices)
    num_slices = img_data.shape[2]
    for i in range(num_slices):
        #extract the same slice index for both
        img_slice = img_data[:, :, i]
        lbl_slice = lbl_data[:, :, i]

        #only save if there is actually an artery in the label
        #this saves space and prevents training on "empty" slices
        if np.any(lbl_slice > 0):
            slice_filename = f"case_{case_id}_slice_{i:03d}.npy"
            
            np.save(os.path.join(img_out, slice_filename), img_slice)
            np.save(os.path.join(lbl_out, slice_filename), lbl_slice)

    print(f"Case {case_id} processed: {num_slices} slices checked.")


def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers = 4, pin_memory = True):
    train_ds = ImageCAS_Dataset(
        image_dir = train_dir,
        mask_dir = train_maskdir,
        transform = train_transform
    )

    val_ds = ImageCAS_Dataset(
        image_dir = val_dir,
        mask_dir = val_maskdir,
        transform = val_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

