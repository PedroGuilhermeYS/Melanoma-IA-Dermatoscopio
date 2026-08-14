from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from melanoma_ia.gradcam import GradCAMHooks, compute_gradcam, find_target_layer


class _TinyCNN(nn.Module):
    """
    Minimal CNN classifier exposing .features, used only to exercise the 4D-activation branch of compute_gradcam, EVA-02 itself is a ViT, but the function is written to support both.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_find_target_layer_vit(engine):
    layer = find_target_layer(engine.model)
    assert layer is engine.model.model.blocks[-1]


def test_find_target_layer_cnn():
    model = _TinyCNN()
    layer = find_target_layer(model)
    assert layer is model.features[-1]


def test_find_target_layer_raises_for_unsupported_model():
    class Empty(nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(ValueError):
        find_target_layer(Empty())


def test_compute_gradcam_vit_branch(engine, sample_image):
    input_tensor = engine.preprocess(sample_image)
    target_layer = find_target_layer(engine.model)
    output_size = input_tensor.shape[-1]

    with GradCAMHooks(target_layer) as hooks:
        result = compute_gradcam(engine.model, input_tensor, hooks, output_size=output_size)

    assert result.heatmap.shape == (output_size, output_size)
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0 + 1e-6
    assert 0.0 <= result.probability <= 1.0


def test_compute_gradcam_cnn_branch():
    model = _TinyCNN()
    model.eval()
    input_tensor = torch.randn(1, 3, 32, 32)
    target_layer = find_target_layer(model)

    with GradCAMHooks(target_layer) as hooks:
        result = compute_gradcam(model, input_tensor, hooks, output_size=32)

    assert result.heatmap.shape == (32, 32)
    assert isinstance(result.heatmap, np.ndarray)


def test_gradcam_hooks_remove_stops_capturing():
    model = _TinyCNN()
    target_layer = find_target_layer(model)
    hooks = GradCAMHooks(target_layer)
    hooks.remove()

    hooks.activations = None
    with torch.no_grad():
        model(torch.randn(1, 3, 16, 16))
    assert hooks.activations is None
