"""FastAPI application entry-point.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from app import __version__
from app.api.routes import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)

app = FastAPI(
    title="AI-Driven Carbon Efficiency Benchmarking",
    description=(
        "Transforms passive network telemetry into actionable environmental "
        "and financial ROI by clustering sites via K-Means and flagging "
        "anomalous power consumption for the NOC."
    ),
    version=__version__,
)

app.include_router(api_router)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}
