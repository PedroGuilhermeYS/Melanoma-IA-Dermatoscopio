"""
Centralized, environment-driven configuration for the AI package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

def _float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


@dataclass(frozen=True)
class ModelSettings:
    model_name: str = "eva02_small_patch14_336.mim_in22k_ft_in1k"

    # Hugging Face Hub source for the fine-tuned checkpoint.
    hf_repo_id: str = "fawo/eva02-small-melanoma-classifier"
    hf_filename: str = "model_0001.pt"

    hf_revision: str = ""

    # Decision threshold. Default matches the ~97% sensitivity operating point documented in results/threshold_key_indicators.csv
    threshold: float = 0.430

    device: str = "auto"  # auto | cpu | cuda

    @classmethod
    def from_env(cls) -> "ModelSettings":
        return cls(
            model_name=os.getenv("MELANOMA_MODEL_NAME", cls.model_name),
            hf_repo_id=os.getenv("MELANOMA_HF_REPO_ID", cls.hf_repo_id),
            hf_filename=os.getenv("MELANOMA_HF_FILENAME", cls.hf_filename),
            hf_revision=os.getenv("MELANOMA_HF_REVISION", cls.hf_revision),
            threshold=_float_env("MELANOMA_THRESHOLD", cls.threshold),
            device=os.getenv("MELANOMA_DEVICE", cls.device),
        )

@dataclass(frozen=True)
class APISettings:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    # Shared secret required as Authorization: Bearer <token>. Empty = auth disabled.
    api_token: str = ""

    @classmethod
    def from_env(cls) -> "APISettings":
        return cls(
            host=os.getenv("MELANOMA_API_HOST", cls.host),
            port=_int_env("MELANOMA_API_PORT", cls.port),
            log_level=os.getenv("MELANOMA_API_LOG_LEVEL", cls.log_level),
            api_token=os.getenv("MELANOMA_API_TOKEN", cls.api_token),
        )


@dataclass(frozen=True)
class OrchestratorSettings:
    """
    Config for the camera-polling -> AI-prediction pipeline.
    """

    camera_url: str = "http://127.0.0.1:5000"
    camera_token: str = ""

    api_url: str = "http://127.0.0.1:8000"

    poll_interval_seconds: float = 2.0
    batch_size: int = 10

    results_path: str = "results/orchestrator_log.jsonl"

    @classmethod
    def from_env(cls) -> "OrchestratorSettings":
        return cls(
            camera_url=os.getenv("MELANOMA_CAMERA_URL", cls.camera_url),
            camera_token=os.getenv("MELANOMA_CAMERA_TOKEN", cls.camera_token),
            api_url=os.getenv("MELANOMA_API_URL", cls.api_url),
            poll_interval_seconds=_float_env("MELANOMA_POLL_INTERVAL_SECONDS", cls.poll_interval_seconds),
            batch_size=_int_env("MELANOMA_BATCH_SIZE", cls.batch_size),
            results_path=os.getenv("MELANOMA_RESULTS_PATH", cls.results_path),
        )


@dataclass(frozen=True)
class Settings:
    model: ModelSettings = field(default_factory=ModelSettings)
    api: APISettings = field(default_factory=APISettings)
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=ModelSettings.from_env(),
            api=APISettings.from_env(),
            orchestrator=OrchestratorSettings.from_env(),
        )
