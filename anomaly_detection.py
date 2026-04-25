import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from model import UNetWithOOD
from utils import load_checkpoint


EPS = 1e-8


def sigmoid_entropy(logits):
    probs = torch.sigmoid(logits)
    entropy = -(probs * torch.log(probs + EPS) + (1 - probs) * torch.log(1 - probs + EPS))
    return entropy.squeeze(1)


def load_normalized_image(image_path):
    image = np.load(image_path).astype(np.float32)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)
    image = (image - image.min()) / (image.max() - image.min() + EPS)
    return image


def compute_mask_morphology(mask):
    mask = mask.astype(np.uint8)
    height, width = mask.shape
    total_pixels = float(height * width)
    area = float(mask.sum())
    area_ratio = area / total_pixels

    if area == 0:
        return np.array(
            [
                0.0,  # area ratio
                0.0,  # component count
                0.0,  # largest component ratio
                0.0,  # bbox fill
                0.0,  # centroid y
                0.0,  # centroid x
                0.0,  # perimeter ratio
            ],
            dtype=np.float32,
        )

    num_components, labels = cv2.connectedComponents(mask)
    component_count = float(max(num_components - 1, 0))

    objects = ndimage.find_objects(labels)
    largest_component = 0.0
    for index, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        largest_component = max(largest_component, float((labels == index).sum()))
    largest_component_ratio = largest_component / total_pixels

    ys, xs = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_area = float((y_max - y_min + 1) * (x_max - x_min + 1))
    bbox_fill = area / max(bbox_area, 1.0)

    centroid_y = float(ys.mean() / max(height - 1, 1))
    centroid_x = float(xs.mean() / max(width - 1, 1))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(cnt, True) for cnt in contours))
    perimeter_ratio = perimeter / max(height + width, 1.0)

    return np.array(
        [
            area_ratio,
            component_count,
            largest_component_ratio,
            bbox_fill,
            centroid_y,
            centroid_x,
            perimeter_ratio,
        ],
        dtype=np.float32,
    )


@torch.no_grad()
def extract_slice_features(model, image_path, device, threshold=0.5):
    model.eval()
    image = load_normalized_image(image_path)
    tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    seg_logits, _, decoder_features = model(tensor, return_features=True)
    seg_probs = torch.sigmoid(seg_logits)
    seg_mask = (seg_probs > threshold).cpu().numpy().astype(np.uint8)[0, 0]

    entropy_map = sigmoid_entropy(seg_logits).cpu().numpy()[0]
    pooled_embedding = F.adaptive_avg_pool2d(decoder_features, 1).flatten(1).cpu().numpy()[0].astype(np.float32)

    morphology = compute_mask_morphology(seg_mask)
    uncertainty = np.array(
        [
            float(entropy_map.mean()),
            float(entropy_map.std()),
            float(seg_probs.mean().item()),
            float(seg_probs.max().item()),
        ],
        dtype=np.float32,
    )

    feature_vector = np.concatenate([pooled_embedding, morphology, uncertainty], axis=0)
    return {
        "image_path": image_path,
        "seg_mask": seg_mask,
        "entropy_map": entropy_map,
        "feature_vector": feature_vector,
        "morphology": morphology,
        "uncertainty": uncertainty,
    }


def fit_normal_reference(feature_matrix):
    mean = feature_matrix.mean(axis=0).astype(np.float32)
    std = feature_matrix.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    z_scores = np.abs((feature_matrix - mean) / std)
    train_scores = np.sqrt((z_scores ** 2).mean(axis=1)).astype(np.float32)

    return {
        "feature_mean": mean,
        "feature_std": std,
        "score_threshold": float(np.quantile(train_scores, 0.99)),
        "train_score_mean": float(train_scores.mean()),
        "train_score_std": float(train_scores.std()),
        "feature_dim": int(feature_matrix.shape[1]),
        "num_reference_slices": int(feature_matrix.shape[0]),
    }


def score_feature_vector(feature_vector, reference):
    z_scores = np.abs((feature_vector - reference["feature_mean"]) / reference["feature_std"])
    anomaly_score = float(np.sqrt((z_scores ** 2).mean()))
    is_anomalous = anomaly_score > reference["score_threshold"]
    return anomaly_score, is_anomalous, z_scores.astype(np.float32)


def fit_reference_from_directory(model, image_dir, device, threshold=0.5, limit=None):
    image_paths = sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.endswith(".npy")
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No .npy files found in {image_dir}")

    features = []
    for image_path in image_paths:
        result = extract_slice_features(model, image_path, device, threshold=threshold)
        features.append(result["feature_vector"])

    feature_matrix = np.stack(features)
    reference = fit_normal_reference(feature_matrix)
    reference["image_dir"] = image_dir
    reference["threshold"] = float(threshold)
    return reference


def save_reference(reference, output_path):
    np.savez_compressed(output_path, **reference)


def load_reference(reference_path):
    data = np.load(reference_path, allow_pickle=False)
    reference = {key: data[key] for key in data.files}
    reference["score_threshold"] = float(reference["score_threshold"])
    reference["train_score_mean"] = float(reference["train_score_mean"])
    reference["train_score_std"] = float(reference["train_score_std"])
    reference["feature_dim"] = int(reference["feature_dim"])
    reference["num_reference_slices"] = int(reference["num_reference_slices"])
    reference["threshold"] = float(reference["threshold"])
    if isinstance(reference.get("image_dir"), np.ndarray):
        reference["image_dir"] = str(reference["image_dir"].item())
    return reference


def load_model(checkpoint_path, device):
    model = UNetWithOOD(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    load_checkpoint(checkpoint, model)
    model.eval()
    return model


def command_fit(args):
    device = resolve_device(args.device)
    model = load_model(args.model, device)
    reference = fit_reference_from_directory(
        model,
        args.image_dir,
        device,
        threshold=args.threshold,
        limit=args.limit,
    )
    save_reference(reference, args.output)
    print(f"Saved normal reference to {args.output}")
    print(f"Reference slices: {reference['num_reference_slices']}")
    print(f"Feature dimension: {reference['feature_dim']}")
    print(f"Score threshold (99th percentile): {reference['score_threshold']:.4f}")


def command_score(args):
    device = resolve_device(args.device)
    model = load_model(args.model, device)
    reference = load_reference(args.reference)
    result = extract_slice_features(model, args.image, device, threshold=reference["threshold"])
    anomaly_score, is_anomalous, z_scores = score_feature_vector(result["feature_vector"], reference)

    print(f"Image: {args.image}")
    print(f"Anomaly score: {anomaly_score:.4f}")
    print(f"Anomalous: {is_anomalous}")
    print(f"Reference threshold: {reference['score_threshold']:.4f}")
    print(f"Mean entropy: {result['uncertainty'][0]:.4f}")
    print(f"Foreground area ratio: {result['morphology'][0]:.4f}")
    print(f"Top 5 feature z-scores: {np.sort(z_scores)[-5:][::-1]}")


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_parser():
    parser = argparse.ArgumentParser(description="Normal-only coronary anomaly scoring")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a normal reference model from normal slices")
    fit_parser.add_argument("--model", type=str, default="aoca_model_synced.pth.tar", help="Segmentation checkpoint path")
    fit_parser.add_argument("--image-dir", type=str, default="data/processed/train/images", help="Directory of normal .npy slices")
    fit_parser.add_argument("--output", type=str, default="normal_reference_stats.npz", help="Output .npz file")
    fit_parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    fit_parser.add_argument("--limit", type=int, default=None, help="Optional cap on reference slices")
    fit_parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda")
    fit_parser.set_defaults(func=command_fit)

    score_parser = subparsers.add_parser("score", help="Score a new slice against the normal reference")
    score_parser.add_argument("--model", type=str, default="aoca_model_synced.pth.tar", help="Segmentation checkpoint path")
    score_parser.add_argument("--reference", type=str, default="normal_reference_stats.npz", help="Reference .npz file")
    score_parser.add_argument("--image", type=str, required=True, help="Path to .npy slice")
    score_parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda")
    score_parser.set_defaults(func=command_score)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
