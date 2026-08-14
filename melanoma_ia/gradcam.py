"""
Grad-CAM computation, extracted from what used to be private logic inside the PyQt6 GUI's GradCAMWorker / ModelLoadWorker classes. Framework agnostic so it's reusable by the GUI, the API server, and any future consumer, all from a single implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_target_layer(model: nn.Module) -> nn.Module:
    """
    Locate a reasonable last-conv/last-block layer to hook for Grad-CAM, trying the common attribute layouts used by timm ViT and CNN models.
    """
    if hasattr(model, "model") and hasattr(model.model, "blocks"):
        return model.model.blocks[-1]
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError(
        "Could not find a suitable target layer for Grad-CAM on this model "
        "(expected .model.blocks[-1], .blocks[-1] or .features[-1])."
    )


class GradCAMHooks:
    """
    Registers forward/backward hooks on a target layer and captures the activations/gradients needed for Grad-CAM.
    """

    def __init__(self, target_layer: nn.Module):
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.clone().detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GradCAMHooks":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()


@dataclass
class GradCAMOutput:
    heatmap: np.ndarray
    probability: float


def compute_gradcam(model: nn.Module, input_tensor: torch.Tensor, hooks: GradCAMHooks,  output_size: int,) -> GradCAMOutput:
    """
    Run a forward+backward pass and turn the captured activations / gradients into a normalized Grad-CAM heatmap resized to output_size x output_size.
    """
    with torch.enable_grad():
        model.zero_grad()
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
        logits.backward()

    if hooks.activations is None or hooks.gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    activations = hooks.activations.clone()
    gradients = hooks.gradients.clone()

    if activations.dim() == 3:
        # Vision Transformer: [B, seq_len(+CLS), C] -> drop CLS, reshape to grid.
        batch_size, seq_len, channels = activations.shape
        activations = activations[:, 1:, :]
        gradients = gradients[:, 1:, :]

        seq_len = activations.shape[1]
        spatial_dim = int(seq_len**0.5)
        if spatial_dim * spatial_dim != seq_len:
            raise ValueError(f"Cannot reshape seq_len {seq_len} into a square grid")

        activations = activations.transpose(1, 2).reshape(batch_size, channels, spatial_dim, spatial_dim)
        gradients = gradients.transpose(1, 2).reshape(batch_size, channels, spatial_dim, spatial_dim)

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(channels):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()

    elif activations.dim() == 4:
        # CNN: [B, C, H, W]
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()

    else:
        raise ValueError(f"Unexpected activation shape: {tuple(activations.shape)}")

    heatmap_np = F.relu(heatmap).cpu().numpy()
    if heatmap_np.max() > 0:
        heatmap_np = heatmap_np / heatmap_np.max()
    heatmap_np = cv2.resize(heatmap_np, (output_size, output_size))

    return GradCAMOutput(heatmap=heatmap_np, probability=probability)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET, opacity: float = 0.45, activation_threshold: float = 0.0) -> np.ndarray:
    mask = heatmap >= activation_threshold
    thresholded = heatmap * mask
    colored = cv2.applyColorMap(np.uint8(255 * thresholded), colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - opacity, colored_rgb, opacity, 0)
