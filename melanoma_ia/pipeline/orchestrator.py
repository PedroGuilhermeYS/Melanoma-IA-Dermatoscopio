"""
Orchestrator: polls the camera service for frames and classifies each one.

Usage:
    python -m melanoma_ia.pipeline.orchestrator
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from ..clients.camera_client import CameraClient
from ..config import OrchestratorSettings, Settings
from ..inference import InferenceEngine

logger = logging.getLogger(__name__)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    orch: OrchestratorSettings = settings.orchestrator
    results_path = Path(orch.results_path)

    logger.info("Loading model...")
    engine = InferenceEngine.from_settings(settings.model)
    if engine.load_warning:
        logger.warning("Model loaded with warnings: %s", engine.load_warning)

    camera = CameraClient(orch.camera_url, token=orch.camera_token)

    logger.info(
        "Orchestrator started: camera=%s poll_interval=%ss batch_size=%s results=%s",
        orch.camera_url,
        orch.poll_interval_seconds,
        orch.batch_size,
        results_path,
    )

    frame_count = 0
    batch_count = 0
    try:
        while True:
            loop_start = time.time()
            try:
                frame = camera.get_snapshot()
                prediction = engine.predict(frame.image)
                frame_count += 1

                record = {
                    "frame_index": frame_count,
                    "processed_at": time.time(),
                    "capture_timestamp": frame.capture_timestamp,
                    **asdict(prediction),
                }
                _append_jsonl(results_path, record)

                logger.info(
                    "[%d] %s (p=%.4f, threshold=%.3f)",
                    frame_count,
                    prediction.label,
                    prediction.probability,
                    prediction.threshold,
                )

                if frame_count % orch.batch_size == 0:
                    batch_count += 1
                    logger.info("Completed batch %d (%d frames)", batch_count, orch.batch_size)

            except Exception:
                logger.exception("Error processing a frame; retrying after the poll interval")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, orch.poll_interval_seconds - elapsed))

    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
