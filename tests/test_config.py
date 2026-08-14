from __future__ import annotations

from melanoma_ia.config import ModelSettings, OrchestratorSettings, Settings


def test_model_settings_defaults():
    settings = ModelSettings()
    assert settings.threshold == 0.430
    assert settings.hf_revision == "" 


def test_model_settings_from_env(monkeypatch):
    monkeypatch.setenv("MELANOMA_THRESHOLD", "0.75")
    monkeypatch.setenv("MELANOMA_HF_REVISION", "abc123")
    monkeypatch.setenv("MELANOMA_DEVICE", "cpu")

    settings = ModelSettings.from_env()

    assert settings.threshold == 0.75
    assert settings.hf_revision == "abc123"
    assert settings.device == "cpu"


def test_orchestrator_settings_from_env(monkeypatch):
    monkeypatch.setenv("MELANOMA_CAMERA_URL", "http://camera.local:5000")
    monkeypatch.setenv("MELANOMA_BATCH_SIZE", "5")

    settings = OrchestratorSettings.from_env()

    assert settings.camera_url == "http://camera.local:5000"
    assert settings.batch_size == 5


def test_settings_from_env_combines_all(monkeypatch):
    monkeypatch.setenv("MELANOMA_API_PORT", "9001")
    settings = Settings.from_env()
    assert settings.api.port == 9001
    assert isinstance(settings.model, ModelSettings)
