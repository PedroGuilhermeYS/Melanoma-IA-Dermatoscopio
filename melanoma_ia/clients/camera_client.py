"""
Thin HTTP client for the camera service's /snapshot endpoint.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import requests
from PIL import Image


@dataclass
class CameraFrame:
    image: Image.Image
    capture_timestamp: Optional[float]


class CameraClient:
    def __init__(self, base_url: str, token: str = "", timeout: float = 5.0):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def get_snapshot(self) -> CameraFrame:
        response = requests.get(
            f"{self._base_url}/snapshot", headers=self._headers(), timeout=self._timeout
        )
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        ts_header = response.headers.get("X-Capture-Timestamp")
        return CameraFrame(image=image, capture_timestamp=float(ts_header) if ts_header else None)

    def health(self) -> dict:
        response = requests.get(f"{self._base_url}/health", timeout=self._timeout)
        response.raise_for_status()
        return response.json()
