import os
import numpy as np
import torch
from model import UNetWithOOD
from utils import load_checkpoint


def sigmoid_entropy(logits):
    # logits: (B, 1, H, W)
    probs = torch.sigmoid(logits)
    entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
    return entropy.squeeze(1)


@torch.no_grad()
def predict_and_flag(model, image_path, device, threshold=0.5, entropy_thresh=0.5):
    """
    Predict a binary segmentation mask and flag high-uncertainty slices.

    The model architecture includes an auxiliary OOD head, but it is not trained
    by the current training loop. This inference path therefore relies on entropy
    from the segmentation logits as the usable uncertainty signal.
    """
    model.eval()
    image = np.load(image_path).astype(np.float32)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    outputs = model(tensor)
    seg_logits = outputs[0] if isinstance(outputs, tuple) else outputs
    aux_ood_prob = outputs[1].item() if isinstance(outputs, tuple) else None
    seg_probs = torch.sigmoid(seg_logits)
    seg_mask = (seg_probs > threshold).cpu().numpy().astype(np.uint8)[0, 0]
    entropy_map = sigmoid_entropy(seg_logits).cpu().numpy()[0]
    mean_entropy = float(entropy_map.mean())
    uncertainty_flag = mean_entropy > entropy_thresh
    return {
        "seg_mask": seg_mask,
        "uncertainty_flag": uncertainty_flag,
        "entropy_map": entropy_map,
        "mean_entropy": mean_entropy,
        "aux_ood_prob": aux_ood_prob,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AOCA Inference and Uncertainty Flagging")
    parser.add_argument("--model", type=str, default="aoca_model_synced.pth.tar", help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to .npy image slice")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation probability threshold")
    parser.add_argument("--entropy-thresh", type=float, default=0.5, help="Mean entropy threshold for uncertainty flagging")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = UNetWithOOD(in_channels=1, out_channels=1).to(device)
    load_checkpoint(torch.load(args.model, map_location=device, weights_only=True), model)

    result = predict_and_flag(
        model,
        args.image,
        device,
        threshold=args.threshold,
        entropy_thresh=args.entropy_thresh,
    )
    print(f"Uncertainty Flag: {result['uncertainty_flag']}")
    print(f"Segmentation mask shape: {result['seg_mask'].shape}")
    print(f"Entropy mean: {result['mean_entropy']:.3f}")
    if result["aux_ood_prob"] is not None:
        print(f"Aux OOD head output (not supervised): {result['aux_ood_prob']:.3f}")
