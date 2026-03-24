import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_3d_mesh(volume_path, threshold=0.5, alpha=0.7, color='cyan'):
    """
    Marching Cubes to create a 3D view of the Aorta/Artery structure.
    """
    # Load 3D volume (expecting a .npy array of your stacked slices)
    volume = np.load(volume_path)
    
    # Generate mesh
    verts, faces, normals, values = measure.marching_cubes(volume, level=threshold)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the 3D polygons
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    
    plt.title('3D Anatomical Reconstruction')
    plt.show()


def show_cam_heatmap(model, image, target_layer, device, save_path=None):
    """
    Visualizes which pixels influenced the 'Artery' prediction the most.
    """
    model.eval()
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    
    # Prepare input
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    tensor.requires_grad = True

    activations = []
    gradients = []

    # Hooks to capture internal model data
    def forward_hook(module, input, output):
        activations.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Use 'full_backward_hook' for better compatibility with M4/Latest PyTorch
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    seg_out, _ = model(tensor)
    
    # FOCUS: We want the CAM for Class 2 (Coronary Artery)
    # seg_out shape: (Batch, Classes, H, W) -> (1, 3, H, W)
    class_score = seg_out[0, 2, :, :].mean() 
    
    model.zero_grad()
    class_score.backward()

    # Process CAM
    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]
    
    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
        
    cam = np.maximum(cam, 0) # ReLU on heatmap
    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image[0], cmap='gray')
    plt.title('Original CT')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image[0], cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM (Artery Focus)')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

    handle_fwd.remove()
    handle_bwd.remove()


def show_uncertainty_overlay(image, entropy_map, save_path=None):
    """
    Shows a heatmap of where the AI is uncertain about its prediction.
    Red areas = High Entropy (Model is 'confused').
    """
    if image.ndim == 3: # Handle (1, H, W)
        image = image[0]
        
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    # Overlay the entropy map (Red/Hot regions = High Uncertainty)
    plt.imshow(entropy_map, cmap='hot', alpha=0.4) 
    plt.colorbar(label='Pixel-wise Entropy')
    plt.title('Clinical Flagging: Uncertainty Map')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()