"""
HTTP surface of the inference server.

    POST /predict: multipart image upload -> JSON PredictionResult
    POST /predict/gradcam : multipart image upload -> JSON {prediction, heatmap_png_base64}
    GET /health : model/device status, used by the orchestrator and Docker/compose healthchecks
"""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image

from ..config import Settings
from ..gradcam import GradCAMHooks, compute_gradcam, find_target_layer, overlay_heatmap
from ..schemas import HealthStatus, PredictionResult

_bearer = HTTPBearer(auto_error=False)


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    def _require_auth(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
        if not settings.api.api_token:
            return
        if credentials is None or credentials.credentials != settings.api.api_token:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def _get_engine(request: Request):
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            detail = getattr(request.app.state, "model_error", "model not loaded")
            raise HTTPException(status_code=503, detail=f"model unavailable: {detail}")
        return engine

    async def _read_image(file: UploadFile) -> Image.Image:
        try:
            contents = await file.read()
            return Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    @router.get("/health", response_model=HealthStatus)
    def health(request: Request) -> HealthStatus:
        engine = getattr(request.app.state, "engine", None)
        error = getattr(request.app.state, "model_error", None)
        if engine is not None:
            return HealthStatus(status="ok", model_loaded=True, device=str(engine.device))
        return HealthStatus(status="error", model_loaded=False, detail=error or "loading")

    @router.post("/predict", response_model=PredictionResult, dependencies=[Depends(_require_auth)])

    async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResult:
        engine = _get_engine(request)
        image = await _read_image(file)
        return engine.predict(image)

    @router.post("/predict/gradcam", dependencies=[Depends(_require_auth)])
    async def predict_gradcam(request: Request, file: UploadFile = File(...)) -> dict:
        engine = _get_engine(request)
        image = await _read_image(file)

        input_tensor = engine.preprocess(image)
        target_layer = find_target_layer(engine.model)

        with GradCAMHooks(target_layer) as hooks:
            output_size = input_tensor.shape[-1]
            result = compute_gradcam(engine.model, input_tensor, hooks, output_size=output_size)

        resized_image = np.array(image.resize((output_size, output_size)))
        overlay = overlay_heatmap(resized_image, result.heatmap)

        success, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            raise HTTPException(status_code=500, detail="failed to encode heatmap overlay")

        threshold = engine.settings.threshold
        prediction = PredictionResult(
            probability=result.probability,
            label="malignant" if result.probability >= threshold else "benign",
            threshold=threshold,
            model_name=engine.settings.model_name,
            model_revision=engine.settings.hf_revision or "unpinned",
        )

        return {
            "prediction": prediction.to_dict(),
            "heatmap_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    return router
