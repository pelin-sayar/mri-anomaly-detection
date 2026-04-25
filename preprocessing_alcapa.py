import nibabel as nib
import numpy as np
import os
import cv2
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "./ImageALCAPA_raw/ImageALCAPA_BIBM_Publish" 
OUTPUT_DIR = "./data/test_alcapa"
TARGET_SIZE = (512, 512)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_intensity(slice_2d):
    """
    Standard Min-Max normalization. 
    Ensure this matches your ImageCAS preprocessing exactly!
    """
    m, M = slice_2d.min(), slice_2d.max()
    if M - m > 0:
        return (slice_2d - m) / (M - m)
    return slice_2d

def preprocess_alcapa():
    # filter out metadata files (._) and ground truth labels (_label)
    all_files = os.listdir(SOURCE_DIR)
    image_files = [f for f in all_files if f.endswith('.nii.gz') 
                   and not f.startswith('._') 
                   and 'label' not in f.lower()]
    
    print(f"Found {len(image_files)} valid 3D volumes. Skipping metadata and labels.")

    for filename in tqdm(image_files):
        file_path = os.path.join(SOURCE_DIR, filename)
        
        try:
            img_nifti = nib.load(file_path)
            data = img_nifti.get_fdata()
            
            # Slicing through the axial plane
            for z in range(data.shape[2]):
                axial_slice = data[:, :, z]
                
                # 1. Normalize
                normalized = normalize_intensity(axial_slice)
                
                # 2. Resize to 512x512
                resized = cv2.resize(normalized, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                
                # 3. Save as .npy
                slice_name = f"{filename.replace('.nii.gz', '')}_z{z:03d}.npy"
                np.save(os.path.join(OUTPUT_DIR, slice_name), resized)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    preprocess_alcapa()
    print(f"\nSuccess! Slices are ready in: {OUTPUT_DIR}")