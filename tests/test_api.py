from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from melanoma_ia.api.main import create_app
from melanoma_ia.config import APISettings, Settings


def _image_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def api_client(engine):
    app = create_app(settings=Settings(), engine=engine)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_client_with_auth(engine):
    settings = Settings(api=APISettings(api_token="secret-token"))
    app = create_app(settings=settings, engine=engine)
    with TestClient(app) as client:
        yield client


def test_health_reports_model_loaded(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_prediction(api_client, sample_image):
    files = {"file": ("lesion.png", _image_bytes(sample_image), "image/png")}
    resp = api_client.post("/predict", files=files)

    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in {"malignant", "benign"}
    assert "threshold" in body
    assert "model_name" in body


def test_predict_gradcam_returns_heatmap(api_client, sample_image):
    files = {"file": ("lesion.png", _image_bytes(sample_image), "image/png")}
    resp = api_client.post("/predict/gradcam", files=files)

    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "heatmap_png_base64" in body
    assert len(body["heatmap_png_base64"]) > 0


def test_predict_rejects_invalid_image(api_client):
    files = {"file": ("not_an_image.txt", b"hello world", "text/plain")}
    resp = api_client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_without_token_is_rejected_when_auth_enabled(api_client_with_auth, sample_image):
    files = {"file": ("lesion.png", _image_bytes(sample_image), "image/png")}
    resp = api_client_with_auth.post("/predict", files=files)
    assert resp.status_code == 401


def test_predict_with_valid_token_succeeds(api_client_with_auth, sample_image):
    files = {"file": ("lesion.png", _image_bytes(sample_image), "image/png")}
    resp = api_client_with_auth.post(
        "/predict", files=files, headers={"Authorization": "Bearer secret-token"}
    )
    assert resp.status_code == 200


def test_health_does_not_require_auth(api_client_with_auth):
    resp = api_client_with_auth.get("/health")
    assert resp.status_code == 200
