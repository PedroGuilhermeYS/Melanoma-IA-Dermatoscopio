"""
Shared pytest fixtures.

None of these tests download the real EVA-02 checkpoint from the Hugging Face Hub, that would require network access and a multi-hundred-MB download on every CI run. Instead, they build a tiny, randomly-initialized ViT through the exact same ISICModel / load_checkpoint code path, save it to a temp file, and reload it. Which still exercises the real architecture wrapper, the real checkpoint (de)serialization and the real prefix-remapping logic, just with a small, fast, offline model.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from melanoma_ia.config import ModelSettings
from melanoma_ia.inference import InferenceEngine
from melanoma_ia.model import ISICModel, load_model_from_local_checkpoint

TINY_MODEL_NAME = "vit_tiny_patch16_224"


@pytest.fixture(scope="session")
def tiny_checkpoint_path(tmp_path_factory) -> Path:
    """
    A real ISICModel(TINY_MODEL_NAME), saved in the same {model_state_dict: ...} format the real checkpoints use.
    """
    model = ISICModel(TINY_MODEL_NAME)
    path = tmp_path_factory.mktemp("checkpoints") / "tiny_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


@pytest.fixture
def model_settings(tiny_checkpoint_path) -> ModelSettings:
    return ModelSettings(model_name=TINY_MODEL_NAME, threshold=0.5, device="cpu")


@pytest.fixture
def engine(model_settings, tiny_checkpoint_path) -> InferenceEngine:
    load_result = load_model_from_local_checkpoint(
        model_settings.model_name, tiny_checkpoint_path, device=torch.device("cpu")
    )
    return InferenceEngine(load_result, model_settings)


@pytest.fixture
def sample_image() -> Image.Image:
    import numpy as np

    array = (np.random.rand(96, 96, 3) * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")
