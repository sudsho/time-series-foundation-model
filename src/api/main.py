"""
FastAPI service for /forecast.

loads a finetuned forecaster checkpoint at startup and exposes:
  GET  /health     -> liveness
  POST /forecast   -> body: {"series": [...], "context_length": 256}
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..predict import forecast, load_forecaster

CKPT_PATH = os.environ.get("MODEL_CKPT", "artifacts/finetune/best.pt")


class ForecastRequest(BaseModel):
    series: List[float] = Field(..., description="1D history of the series")
    context_length: Optional[int] = Field(
        default=None, description="how much of the tail to use as context"
    )


class ForecastResponse(BaseModel):
    horizon: int
    forecast: List[float]


app = FastAPI(title="ts-foundation-model", version="0.1.0")

state = {"model": None, "meta": None, "loaded": False}


@app.on_event("startup")
def _load_model():
    if not os.path.exists(CKPT_PATH):
        # don't crash the container if no weights yet, /forecast will 503
        print(f"[startup] no checkpoint found at {CKPT_PATH}, /forecast disabled")
        return
    model, meta = load_forecaster(CKPT_PATH)
    state["model"] = model
    state["meta"] = meta
    state["loaded"] = True
    print(f"[startup] loaded checkpoint from {CKPT_PATH}, meta={meta}")


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": state["loaded"]}


@app.post("/forecast", response_model=ForecastResponse)
def post_forecast(req: ForecastRequest):
    if not state["loaded"]:
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(req.series) < 8:
        raise HTTPException(status_code=400, detail="series too short")

    series = np.asarray(req.series, dtype=np.float32)
    meta = state["meta"]
    L = req.context_length or (meta["n_patches"] * meta["patch_len"])
    if series.shape[0] < L:
        raise HTTPException(
            status_code=400,
            detail=f"need at least {L} points; got {series.shape[0]}",
        )
    out = forecast(state["model"], series, L, meta["patch_len"])
    return ForecastResponse(horizon=int(meta["horizon"]), forecast=out.tolist())
