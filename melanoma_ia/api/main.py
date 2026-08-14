"""
FastAPI application factory.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from ..config import Settings
from ..inference import InferenceEngine
from .routes import build_router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, engine: Optional[InferenceEngine] = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
            app.state.model_error = None
            yield
            return

        logger.info("Loading model at startup...")
        try:
            app.state.engine = InferenceEngine.from_settings(settings.model)
            app.state.model_error = None
            logger.info("Model loaded on device=%s", app.state.engine.device)
        except Exception as exc:
            logger.exception("Model failed to load at startup")
            app.state.engine = None
            app.state.model_error = str(exc)
        yield

    app = FastAPI(
        title="Melanoma IA Dermatoscopio API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(build_router(settings))
    return app
