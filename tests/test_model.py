from __future__ import annotations

import torch

from melanoma_ia.model import (
    ISICModel,
    load_checkpoint,
    load_model_from_local_checkpoint,
    resolve_device,
)
from tests.conftest import TINY_MODEL_NAME


def test_isic_model_forward_shape():
    model = ISICModel(TINY_MODEL_NAME)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_resolve_device_explicit():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_is_cpu_or_cuda():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda"}


def test_load_checkpoint_round_trip_no_warning(tiny_checkpoint_path):
    model = ISICModel(TINY_MODEL_NAME)
    warning = load_checkpoint(model, tiny_checkpoint_path, torch.device("cpu"))
    assert warning is None 


def test_load_checkpoint_handles_model_prefix_mismatch(tmp_path):
    """
    Simulates the bug this loader exists to catch: a checkpoint saved from the bare timm backbone loaded into an ISICModel wrapper, whose own state_dict keys are prefixed with model. Without remapping, this would raise a Missing/Unexpected key(s) RuntimeError.
    """
    source = ISICModel(TINY_MODEL_NAME)
    unprefixed_state_dict = source.model.state_dict()
    ckpt_path = tmp_path / "unprefixed.pt"
    torch.save({"model_state_dict": unprefixed_state_dict}, ckpt_path)

    target = ISICModel(TINY_MODEL_NAME)
    warning = load_checkpoint(target, ckpt_path, torch.device("cpu"))

    assert warning is None

    x = torch.randn(1, 3, 224, 224)
    source.eval()
    target.eval()
    with torch.no_grad():
        assert torch.allclose(source(x), target(x), atol=1e-5)


def test_load_model_from_local_checkpoint(tiny_checkpoint_path):
    result = load_model_from_local_checkpoint(TINY_MODEL_NAME, tiny_checkpoint_path, device=torch.device("cpu"))
    assert result.warning is None
    assert not result.model.training
