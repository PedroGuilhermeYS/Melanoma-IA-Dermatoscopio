"""
Inference engine: owns a loaded model + its matching preprocessing transform, and exposes a single predict(image) call.
"""
from __future__ import annotations

import logging

import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from .config import ModelSettings
from .model import LoadResult, load_model
from .schemas import PredictionResult

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Wraps a loaded model + its preprocessing transform for reuse across the CLI, the HTTP API and the Grad-CAM GUI.
    """

    def __init__(self, load_result: LoadResult, settings: ModelSettings):
        self._model = load_result.model
        self._device = load_result.device
        self._settings = settings
        self._checkpoint_path = load_result.checkpoint_path
        self._load_warning = load_result.warning

        self._transform = create_transform(
            **resolve_data_config({}, model=self._model.model), is_training=False
        )

        if self._load_warning:
            logger.warning("Model loaded with warnings: %s", self._load_warning)

    @classmethod
    def from_settings(cls, settings: ModelSettings) -> "InferenceEngine":
        return cls(load_model(settings), settings)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def transform(self):
        return self._transform

    @property
    def load_warning(self) -> str | None:
        return self._load_warning

    @property
    def settings(self) -> ModelSettings:
        return self._settings

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> PredictionResult:
        tensor = self.preprocess(image)
        probability = torch.sigmoid(self._model(tensor)).item()
        label = "malignant" if probability >= self._settings.threshold else "benign"
        return PredictionResult(
            probability=probability,
            label=label,
            threshold=self._settings.threshold,
            model_name=self._settings.model_name,
            model_revision=self._settings.hf_revision or "unpinned",
        )
