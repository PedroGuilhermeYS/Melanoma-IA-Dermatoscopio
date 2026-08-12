"""
predict.py — Standalone inference script for EVA-02 Melanoma Classifier
Model:      eva02_small_patch14_336.mim_in22k_ft_in1k
Checkpoint: model0001_epoch30.pt

Usage:
    # Single image
    python predict.py --checkpoint model0001_epoch30.pt --image lesion.jpg

    # Folder of images
    python predict.py --checkpoint model0001_epoch30.pt --folder ./images/

    # Custom threshold (default: 0.430 = 97% sensitivity operating point)
    python predict.py --checkpoint model0001_epoch30.pt --image lesion.jpg --threshold 0.640

    # Save results to CSV
    python predict.py --checkpoint model0001_epoch30.pt --folder ./images/ --output results.csv

    # Run on CPU explicitly
    python predict.py --checkpoint model0001_epoch30.pt --image lesion.jpg --device cpu

Thresholds (from test set evaluation):
    0.300  — 99% Sensitivity  (population screening, minimise missed cancers)
    0.430  — 97% Sensitivity  (default, recommended general screening)
    0.640  — 95% Sensitivity  (balanced screening)
    0.770  — Youden's J       (maximises sensitivity + specificity jointly)
    0.860  — Crossover        (sensitivity ≈ specificity ≈ 91.9%)

Requirements:
    pip install torch torchvision timm pillow
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    sys.exit("ERROR: timm not installed. Run: pip install timm")

# ── Supported image extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ── Model definition (mirrors training exactly) ───────────────────────────────
MODEL_NAME = "eva02_small_patch14_336.mim_in22k_ft_in1k"

class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, drop_path_rate=0.0, drop_rate=0.0):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=False,       # weights loaded from checkpoint
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)        # raw logits — sigmoid applied in predict()


# ── Checkpoint loader ─────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> ISICModel:
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"ERROR: Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Support both raw state_dict and wrapped checkpoint (training format)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", "unknown")
        auroc = ckpt.get("best_metric", None)
        print(f"  Epoch:          {epoch}")
        if auroc is not None:
            print(f"  Best val AUROC: {auroc:.4f}")
    else:
        state_dict = ckpt
        print("  (raw state dict — no epoch/metric info)")

    model = ISICModel(MODEL_NAME, num_classes=1, drop_path_rate=0.1, drop_rate=0.0)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  WARNING: missing keys: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    print(f"  Model ready on {device}\n")
    return model


# ── Transform (resolved from timm model config — guaranteed to match training) ─
def build_transform(model: ISICModel) -> object:
    data_cfg = resolve_data_config({}, model=model.model)
    transform = create_transform(**data_cfg, is_training=False)
    print(f"Input transform:  size={data_cfg.get('input_size')}, "
          f"mean={[round(v,3) for v in data_cfg.get('mean',(0,0,0))]}, "
          f"std={[round(v,3) for v in data_cfg.get('std',(1,1,1))]}\n")
    return transform


# ── Single image inference ────────────────────────────────────────────────────
def predict_image(image_path: str, model: ISICModel, transform, device: torch.device) -> float:
    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"  SKIP (unreadable): {image_path} — {e}")
        return None

    tensor = transform(img).unsqueeze(0).to(device)    # [1, C, H, W]

    with torch.no_grad():
        logit = model(tensor)                           # [1, 1]
        prob = torch.sigmoid(logit).item()

    return prob


# ── Batch inference over a folder ─────────────────────────────────────────────
def predict_folder(folder: str, model: ISICModel, transform, device: torch.device,
                   threshold: float) -> list[dict]:
    image_paths = sorted([
        p for p in Path(folder).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        sys.exit(f"ERROR: No images found in folder: {folder}")

    print(f"Found {len(image_paths)} images in {folder}\n")
    results = []

    for i, path in enumerate(image_paths, 1):
        prob = predict_image(str(path), model, transform, device)
        if prob is None:
            continue

        label = "MALIGNANT" if prob >= threshold else "benign"
        results.append({
            "image":       path.name,
            "path":        str(path),
            "probability": prob,
            "prediction":  label,
            "threshold":   threshold,
        })
        print(f"  [{i:>4}/{len(image_paths)}]  {path.name:<45}  "
              f"p={prob:.4f}  →  {label}")

    return results


# ── Output helpers ────────────────────────────────────────────────────────────
def print_summary(results: list[dict], threshold: float) -> None:
    if not results:
        return
    n_mal = sum(1 for r in results if r["prediction"] == "MALIGNANT")
    n_ben = len(results) - n_mal
    print(f"\n{'─'*60}")
    print(f"  Total images:  {len(results)}")
    print(f"  MALIGNANT:     {n_mal}  ({100*n_mal/len(results):.1f}%)")
    print(f"  benign:        {n_ben}  ({100*n_ben/len(results):.1f}%)")
    print(f"  Threshold:     {threshold}")
    print(f"{'─'*60}\n")


def save_csv(results: list[dict], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "path", "probability",
                                               "prediction", "threshold"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="EVA-02 Melanoma Classifier — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",   type=str, help="Path to a single image")
    src.add_argument("--folder",  type=str, help="Path to a folder of images")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--threshold",  type=float, default=0.430,
                        help="Classification threshold (default: 0.430 = 97%% sensitivity)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Optional: save results to this CSV file (folder mode)")
    parser.add_argument("--device",     type=str, default=None,
                        help="Device: 'cuda', 'cpu', or 'cuda:0' (auto-detected if omitted)")

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Build transform (derived from model config — no hardcoding)
    transform = build_transform(model)

    # Inference
    if args.image:
        prob = predict_image(args.image, model, transform, device)
        if prob is not None:
            label = "MALIGNANT" if prob >= args.threshold else "benign"
            print(f"Image:       {args.image}")
            print(f"Probability: {prob:.4f}")
            print(f"Threshold:   {args.threshold}")
            print(f"Prediction:  {label}")

    elif args.folder:
        results = predict_folder(args.folder, model, transform, device, args.threshold)
        print_summary(results, args.threshold)
        if args.output:
            save_csv(results, args.output)


if __name__ == "__main__":
    main()
