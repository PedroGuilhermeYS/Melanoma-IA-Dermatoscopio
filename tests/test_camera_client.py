from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from melanoma_ia.clients.camera_client import CameraClient


def _fake_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


@patch("melanoma_ia.clients.camera_client.requests.get")
def test_get_snapshot_parses_image_and_timestamp(mock_get):
    mock_response = MagicMock()
    mock_response.content = _fake_jpeg_bytes()
    mock_response.headers = {"X-Capture-Timestamp": "1699999999.123"}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    client = CameraClient("http://camera.local:5000")
    frame = client.get_snapshot()

    assert frame.image.size == (8, 8)
    assert frame.capture_timestamp == 1699999999.123
    mock_get.assert_called_once()
    called_url = mock_get.call_args[0][0]
    assert called_url == "http://camera.local:5000/snapshot"


@patch("melanoma_ia.clients.camera_client.requests.get")
def test_get_snapshot_sends_bearer_token_when_configured(mock_get):
    mock_response = MagicMock()
    mock_response.content = _fake_jpeg_bytes()
    mock_response.headers = {}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    client = CameraClient("http://camera.local:5000", token="my-token")
    client.get_snapshot()

    _, kwargs = mock_get.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer my-token"}


@patch("melanoma_ia.clients.camera_client.requests.get")
def test_get_snapshot_omits_auth_header_when_no_token(mock_get):
    mock_response = MagicMock()
    mock_response.content = _fake_jpeg_bytes()
    mock_response.headers = {}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    client = CameraClient("http://camera.local:5000")
    client.get_snapshot()

    _, kwargs = mock_get.call_args
    assert kwargs["headers"] == {}


@patch("melanoma_ia.clients.camera_client.requests.get")
def test_health_returns_json(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ok", "camera_ready": True}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    client = CameraClient("http://camera.local:5000")
    result = client.health()

    assert result == {"status": "ok", "camera_ready": True}
