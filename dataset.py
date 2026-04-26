import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CoronaryArteryDataset(Dataset):
    def __init__(self, base_dir, split=None, transform=None):
        """
        Args:
            base_dir: Path to the data folder (e.g., 'data/test_alcapa')
            split: 'train', 'val', or None for direct folder access
            transform: Optional albumentations transform
        """
        self.transform = transform
        
        # 1. Path Logic: Handle split-based folders vs. flat folders
        if split and split != "None":
            self.img_dir = os.path.join(base_dir, split, "images")
            self.mask_dir = os.path.join(base_dir, split, "masks")
        else:
            # Check if 'images' subfolder exists, otherwise assume a flat directory
            potential_img_dir = os.path.join(base_dir, "images")
            if os.path.exists(potential_img_dir):
                self.img_dir = potential_img_dir
                self.mask_dir = os.path.join(base_dir, "masks")
            else:
                # Use base_dir directly if no 'images' subfolder is found
                self.img_dir = base_dir
                self.mask_dir = base_dir

        # 2. Safety Check: Verify the directory exists
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Directory not found: {self.img_dir}")

        # 3. File Listing: Filter for .npy files
        all_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.npy')])
        
        # 4. Filter out Masks from Image list if they are in the same folder
        if self.img_dir == self.mask_dir:
            # Assumes images don't have 'mask' in the name and masks do
            self.images = [f for f in all_files if "mask" not in f.lower()]
        else:
            self.images = all_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Logic to find the corresponding mask
        if self.img_dir == self.mask_dir:
            # If flat, mask is usually named 'filename_mask.npy' or similar
            # Adjust 'replace' string if your mask naming is different
            mask_name = img_name.replace(".npy", "_mask.npy")
            mask_path = os.path.join(self.mask_dir, mask_name)
        else:
            mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load arrays
        image = np.load(img_path).astype(np.float32)
        
        # Handle cases where a mask might be missing for ALCAPA
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.uint8)
        else:
            mask = np.zeros_like(image).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # PyTorch expects (Channels, H, W)
        image = np.expand_dims(image, axis=0)
        
        return torch.from_numpy(image), torch.from_numpy(mask)