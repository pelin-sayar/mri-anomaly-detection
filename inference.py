import os
import numpy as np
import torch
import torch.nn.functional as F
from model import UNetWithOOD
from utils import load_checkpoint
from scipy.ndimage import distance_transform_edt

def softmax_entropy(logits):
	# logits: (B, C, H, W)
	probs = F.softmax(logits, dim=1)
	entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # (B, H, W)
	return entropy

@torch.no_grad()
def predict_and_flag(model, image_path, device, threshold=0.5, entropy_thresh=0.5):
	"""
	Predict segmentation and flag OOD using SoftMax entropy and OOD head.
	Args:
		model: Trained UNetWithOOD
		image_path: Path to .npy image slice
		device: torch device
		threshold: Segmentation threshold
		entropy_thresh: Entropy threshold for OOD flag
	Returns:
		seg_mask: np.ndarray, binary mask
		ood_flag: bool, True if anomaly detected
		entropy_map: np.ndarray
		ood_prob: float, OOD head output
	"""
	model.eval()
	image = np.load(image_path).astype(np.float32)
	if image.ndim == 2:
		image = np.expand_dims(image, axis=0)
	image = (image - image.min()) / (image.max() - image.min() + 1e-8)
	tensor = torch.from_numpy(image).unsqueeze(0).to(device)
	seg_logits, ood_prob = model(tensor)
	seg_mask = torch.argmax(seg_logits, dim=1).cpu().numpy().astype(np.uint8)[0]
	entropy_map = softmax_entropy(seg_logits).cpu().numpy()[0]
	ood_flag = (ood_prob.item() > 0.5) or (entropy_map.mean() > entropy_thresh)
	return seg_mask, ood_flag, entropy_map, ood_prob.item()

def calculate_aorta_to_artery_distance(seg_mask):
	"""
	Calculate minimum distance between aorta and coronary artery regions in the mask.
	Assumes mask labels: 1=aorta, 2=artery (example, adjust as needed).
	Returns: float (pixels)
	"""
	aorta = (seg_mask == 1)
	artery = (seg_mask == 2)
	if not np.any(aorta) or not np.any(artery):
		return None
	dist_map = distance_transform_edt(~artery)
	min_dist = dist_map[aorta].min()
	return min_dist

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="AOCA Inference and OOD Flagging")
	parser.add_argument('--model', type=str, default="my_checkpoint.pth.tar", help="Path to model checkpoint")
	parser.add_argument('--image', type=str, required=True, help="Path to .npy image slice")
	args = parser.parse_args()

	# Device selection
	if torch.backends.mps.is_available():
		device = torch.device("mps")
	elif torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	model = UNetWithOOD(in_channels=1, out_channels=3).to(device)
	load_checkpoint(torch.load(args.model, map_location=device), model)

	seg_mask, ood_flag, entropy_map, ood_prob = predict_and_flag(model, args.image, device)
	print(f"OOD Flag: {ood_flag} (OOD head prob: {ood_prob:.3f})")
	print(f"Segmentation mask shape: {seg_mask.shape}")
	print(f"Entropy mean: {entropy_map.mean():.3f}")
	dist = calculate_aorta_to_artery_distance(seg_mask)
	pixel_spacing = 0.5 # Example value from ImageCAS metadata
	if dist is not None:
		print(f"Aorta-to-artery min distance: {dist * pixel_spacing:.2f} mm")