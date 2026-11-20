"""Train AgriVision crop recommender and production estimator from scratch.

No pretrained weights or hosted AI APIs are used. Inputs and labels come from
the two user-supplied CSV files. Run from the project root:
  python training/train_tabular_models.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             median_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training"
OUT = ROOT / "data" / "models"
OUT.mkdir(parents=True, exist_ok=True)


def train_recommender():
    path = DATA / "crop_recommendation.csv"
    df = pd.read_csv(path)
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    target = "label"
    missing = set(features + [target]) - set(df.columns)
    if missing:
        raise ValueError(f"Crop recommendation CSV is missing columns: {sorted(missing)}")
    df = df.dropna(subset=features + [target]).copy()
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features)
    df[target] = df[target].astype(str).str.strip().str.lower()
    X, y = df[features], df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=400, max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    class_list = model.named_steps["forest"].classes_
    top3 = np.mean([y_test.iloc[i] in class_list[top3_idx[i]] for i in range(len(y_test))])
    metrics = {
        "rows_used": int(len(df)), "train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
        "classes": int(y.nunique()), "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "top3_accuracy": float(top3),
        "split": "random stratified 80/20, seed 42",
        "input_columns": features,
        "units_note": "The supplied CSV does not document measurement units; enter values using the same conventions as its source.",
    }
    joblib.dump({"model": model, "features": features, "classes": list(class_list), "metrics": metrics},
                OUT / "crop_recommender.joblib", compress=3)
    return metrics


def train_production_estimator():
    path = DATA / "crop_production.csv"
    df = pd.read_csv(path)
    numeric = ["Crop_Year", "Temperature", "Humidity", "Soil_Moisture", "Area"]
    categorical = ["State_Name", "District_Name", "Season", "Crop"]
    target = "Production"
    required = numeric + categorical + [target]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Crop production CSV is missing columns: {sorted(missing)}")
    df = df[required].copy()
    for col in numeric + [target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical:
        df[col] = df[col].astype("string").str.strip()
    df = df.dropna(subset=required)
    df = df[(df[target] >= 0) & (df["Area"] > 0)].copy()
    # Preserve 2014 as a time-based holdout; do not train on the test year.
    test = df[df["Crop_Year"] == 2014].copy()
    train = df[df["Crop_Year"] < 2014].copy()
    if train.empty or test.empty:
        raise ValueError("Expected historical rows before 2014 and a 2014 holdout in this dataset.")
    features = numeric + categorical
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical),
    ])
    model = Pipeline([
        ("prepare", pre),
        ("forest", RandomForestRegressor(
            n_estimators=180, max_depth=24, min_samples_leaf=2,
            max_features=0.9, random_state=42, n_jobs=-1,
        )),
    ])
    y_train = np.log1p(train[target].to_numpy(dtype=float))
    y_test = test[target].to_numpy(dtype=float)
    model.fit(train[features], y_train)
    pred_log = model.predict(test[features])
    pred = np.maximum(0, np.expm1(pred_log))
    metrics = {
        "rows_used": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_years": [int(train["Crop_Year"].min()), int(train["Crop_Year"].max())],
        "test_year": 2014,
        "mae_dataset_units": float(mean_absolute_error(y_test, pred)),
        "median_absolute_error_dataset_units": float(median_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "rmsle": float(np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(pred)))),
        "input_columns": features,
        "target_note": "Production and area units are not documented in the supplied CSV; output is an estimate in the CSV's production units.",
        "missing_production_rows_dropped": int(pd.to_numeric(pd.read_csv(path)[target], errors="coerce").isna().sum()),
        "split": "train on years before 2014; hold out 2014",
    }
    joblib.dump({"model": model, "features": features, "numeric": numeric,
                 "categorical": categorical, "metrics": metrics,
                 "target_transform": "log1p/expm1"},
                OUT / "production_estimator.joblib", compress=3)
    return metrics


def main():
    report = {
        "created_from_random_initialization": True,
        "source_files": ["data/training/crop_recommendation.csv", "data/training/crop_production.csv"],
        "crop_recommender": train_recommender(),
        "production_estimator": train_production_estimator(),
        "important_limit": "These models recommend crop labels or estimate historical production only. They do not diagnose diseases, inspect leaf images, or provide pesticide advice.",
    }
    (OUT / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
