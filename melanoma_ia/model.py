"""
Single source of truth for the model architecture and checkpoint loading.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .config import ModelSettings

logger = logging.getLogger(__name__)


class ISICModel(nn.Module):
    """
    EVA-02 backbone with a single-logit binary head (malignant vs. benign).
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, drop_path_rate=0.1)
        self.model.head = nn.Linear(self.model.head.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class LoadResult:
    model: ISICModel
    device: torch.device
    checkpoint_path: Path
    warning: Optional[str] = None


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _remap_state_dict_prefix(model: nn.Module, state_dict: dict) -> dict:
    """
    Handle the classic model. prefix mismatch between a checkpoint saved with a wrapper class and one saved from the bare timm model.
    """
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    ckpt_has_prefix = bool(checkpoint_keys) and all(k.startswith("model.") for k in checkpoint_keys)
    model_has_prefix = any(k.startswith("model.") for k in model_keys)

    if ckpt_has_prefix and not model_has_prefix:
        return {k[len("model."):] if k.startswith("model.") else k: v for k, v in state_dict.items()}
    if not ckpt_has_prefix and model_has_prefix:
        return {f"model.{k}": v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model: ISICModel, checkpoint_path: Path, device: torch.device) -> Optional[str]:
    """
    Load weights into model in place. Returns a human-readable warning string if the load had to fall back to strict=False, or None if it loaded cleanly.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = _remap_state_dict_prefix(model, state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        return None
    except RuntimeError as exc:
        if "Missing key(s)" not in str(exc) and "Unexpected key(s)" not in str(exc):
            raise
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        warning = (
            f"Checkpoint loaded with mismatches -- missing={len(missing)}, "
            f"unexpected={len(unexpected)}. Model may not predict correctly. "
            f"First missing: {missing[:5]}. First unexpected: {unexpected[:5]}."
        )
        logger.warning(warning)
        return warning


def load_model(settings: ModelSettings) -> LoadResult:
    """
    Download (if needed) and load the fine-tuned checkpoint from the Hugging Face Hub, returning a ready-to-use, eval() -mode model.
    """
    device = resolve_device(settings.device)

    logger.info(
        "Downloading checkpoint %s@%s from %s...",
        settings.hf_filename,
        settings.hf_revision or "(default branch -- not pinned)",
        settings.hf_repo_id,
    )
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=settings.hf_filename,
            revision=settings.hf_revision or None,
        )
    )

    model = ISICModel(settings.model_name)
    warning = load_checkpoint(model, checkpoint_path, device)
    model.to(device).eval()

    return LoadResult(model=model, device=device, checkpoint_path=checkpoint_path, warning=warning)


def load_model_from_local_checkpoint(model_name: str, checkpoint_path: Path, device: Optional[torch.device] = None) -> LoadResult:
    device = device or resolve_device("auto")
    model = ISICModel(model_name)
    warning = load_checkpoint(model, checkpoint_path, device)
    model.to(device).eval()
    return LoadResult(model=model, device=device, checkpoint_path=Path(checkpoint_path), warning=warning)
