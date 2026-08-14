"""
Shared data types, used by the inference engine, the HTTP API and the CLI/orchestrator. Keeping this in one place is what lets the FastAPI response_model and the JSONL log written by the orchestrator stay in sync automatically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class PredictionResult:
    probability: float
    label: str 
    threshold: float
    model_name: str
    model_revision: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GradCAMResult:
    prediction: PredictionResult
    heatmap_png_base64: str


@dataclass
class HealthStatus:
    status: str  # ok / loading / error
    model_loaded: bool
    device: Optional[str] = None
    detail: Optional[str] = None
