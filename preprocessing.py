import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import random

# --- Configuration ---
RAW_IMG_DIR = "data/raw/images/"   
RAW_MASK_DIR = "data/raw/masks/"   
BASE_PROCESSED_DIR = "data/processed/"

# Create the folder structure for training and validation
for split in ['train', 'val']:
    for folder in ['images', 'masks']:
        os.makedirs(os.path.join(BASE_PROCESSED_DIR, split, folder), exist_ok=True)

def preprocess_and_split(split_ratio=0.8):
    # Get all image files (e.g., 91.img.nii.gz)
    image_files = sorted([f for f in os.listdir(RAW_IMG_DIR) if f.endswith('.nii.gz')])
    
    # Shuffle for a fair distribution of patient cases
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    def process_list(file_list, split_name):
        for file_name in tqdm(file_list, desc=f"Processing {split_name}"):
            img_path = os.path.join(RAW_IMG_DIR, file_name)
            
            # MAPPING: '91.img.nii.gz' -> '91.label.nii.gz'
            mask_name = file_name.replace(".img.", ".label.")
            mask_path = os.path.join(RAW_MASK_DIR, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"\n[Warning] Mask not found for {file_name}, skipping.")
                continue
            
            # Load 3D Data
            img_nifty = nib.load(img_path).get_fdata()
            mask_nifty = nib.load(mask_path).get_fdata()

            # Normalization (Clip to Heart/Vessel HU range)
            img_nifty = np.clip(img_nifty, -200, 500)
            img_min, img_max = img_nifty.min(), img_nifty.max()
            img_nifty = (img_nifty - img_min) / (img_max - img_min + 1e-8)

            # Define num_slices from the 3rd axis (Depth)
            num_slices = img_nifty.shape[2]

            # Slicing along the axial plane
            for i in range(num_slices):
                img_slice = img_nifty[:, :, i]
                mask_slice = mask_nifty[:, :, i]

                # Only save slices that contain anatomy (Mask > 0)
                # This keeps the training efficient by ignoring empty space
                if np.any(mask_slice > 0):
                    slice_id = f"{file_name.replace('.nii.gz', '')}_s{i}.npy"
                    
                    save_img_path = os.path.join(BASE_PROCESSED_DIR, split_name, 'images', slice_id)
                    save_mask_path = os.path.join(BASE_PROCESSED_DIR, split_name, 'masks', slice_id)
                    
                    np.save(save_img_path, img_slice.astype(np.float32))
                    np.save(save_mask_path, mask_slice.astype(np.uint8))

    process_list(train_files, 'train')
    process_list(val_files, 'val') # Using train_files here for process_list logic

if __name__ == "__main__":
    preprocess_and_split()
    print("\nready to train on 2D slices.")