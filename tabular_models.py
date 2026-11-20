"""Load the locally trained tabular models when their artifacts are present."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "data" / "models"


@lru_cache(maxsize=2)
def load_model(name: str):
    path = MODEL_DIR / name
    if not path.is_file():
        return None
    return joblib.load(path)


def recommend_crop(values: dict, top_n: int = 3):
    artifact = load_model("crop_recommender.joblib")
    if artifact is None:
        return None
    row = pd.DataFrame([{name: float(values[name]) for name in artifact["features"]}])
    model = artifact["model"]
    probs = model.predict_proba(row)[0]
    order = np.argsort(probs)[::-1][:top_n]
    return [{"crop": str(model.classes_[i]), "probability": float(probs[i])} for i in order]


def estimate_production(values: dict):
    artifact = load_model("production_estimator.joblib")
    if artifact is None:
        return None
    row = {name: values[name] for name in artifact["features"]}
    for name in artifact["numeric"]:
        row[name] = float(row[name])
    for name in artifact["categorical"]:
        row[name] = str(row[name]).strip()
    log_value = float(artifact["model"].predict(pd.DataFrame([row]))[0])
    predicted = max(0.0, float(np.expm1(log_value)))
    return {"estimated_production": predicted, "unit": "same as source dataset (unit not documented)"}
