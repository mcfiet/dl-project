from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.points_scored_model import ModelBundle, prepare_features

MODEL_PATH = Path("models/tabpfn_points_scored.pkl")

app = FastAPI(title="Points Scored Inference")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PointsScoredRequest(BaseModel):
    driver_id: str
    constructor_id: str
    circuit_id: str
    year: int
    grid_position: int
    quali_delta: float
    quali_tm_delta: float
    season_pts_driver: float
    season_pts_team: float
    last_3_avg: float
    is_street_circuit: int
    is_wet: int


class PointsScoredResponse(BaseModel):
    prediction: int
    probability: float


def load_bundle() -> ModelBundle:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


bundle: ModelBundle | None = None


@app.on_event("startup")
async def startup() -> None:
    global bundle
    bundle = load_bundle()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PointsScoredResponse)
async def predict(payload: PointsScoredRequest) -> PointsScoredResponse:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = pd.DataFrame([payload.dict()])
    try:
        X = prepare_features(data, bundle)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    proba = bundle.model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)

    return PointsScoredResponse(prediction=pred, probability=float(proba))
