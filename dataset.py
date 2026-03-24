import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CoronaryArteryDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None):
        """
        Args:
            base_dir: Path to 'data/processed'
            split: 'train' or 'val'
            transform: Optional albumentations transform
        """
        self.img_dir = os.path.join(base_dir, split, "images")
        self.mask_dir = os.path.join(base_dir, split, "masks")
        
        # Ensure we only list .npy files and they stay in the same order
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.npy')])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load numpy arrays
        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)
        
        # If using albumentations transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Add channel dimension to image: (H, W) -> (1, H, W)
        # PyTorch expects (Channels, Height, Width)
        image = np.expand_dims(image, axis=0)
        
        return torch.from_numpy(image), torch.from_numpy(mask)