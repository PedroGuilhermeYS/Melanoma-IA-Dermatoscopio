"""
Single-image CLI for the melanoma classifier.

Usage:
    python -m melanoma_ia.cli.predict --image lesao.jpg
    python -m melanoma_ia.cli.predict --image lesao.jpg --checkpoint /path/to/local_model.pt
    python -m melanoma_ia.cli.predict --image lesao.jpg --json

"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from PIL import Image

from ..config import ModelSettings
from ..inference import InferenceEngine
from ..model import load_model, load_model_from_local_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify a single dermoscopy image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to the lesion image (jpg/png).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a local checkpoint (.pt) to use instead of downloading "
        "the published Hub release. Useful for evaluating a specific training epoch.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the decision threshold (default: from MELANOMA_THRESHOLD / 0.430).",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        print(
            "Tip: run python scripts/download_sample_image.py to fetch a "
            "public sample image for a quick smoke test.",
            file=sys.stderr,
        )
        return 1

    settings = ModelSettings.from_env()
    if args.threshold is not None:
        settings = dataclasses.replace(settings, threshold=args.threshold)

    if args.checkpoint:
        load_result = load_model_from_local_checkpoint(settings.model_name, args.checkpoint)
    else:
        load_result = load_model(settings)

    engine = InferenceEngine(load_result, settings)
    if engine.load_warning:
        print(f"Warning: {engine.load_warning}", file=sys.stderr)

    image = Image.open(args.image)
    result = engine.predict(image)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Probabilidade: {result.probability:.4f}")
        print(f"Classificação: {result.label} (threshold={result.threshold})")
        print(f"Modelo: {result.model_name} (revision={result.model_revision})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
