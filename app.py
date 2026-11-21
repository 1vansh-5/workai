"""Local AgriVision prototype: knowledge retrieval and reviewed user contributions.

No hosted AI is used and no generative model is claimed to be trained.
"""
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tabular_models import estimate_production, load_model, recommend_crop
from vision_model import load_vision_model, predict_plant_image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
KB_FILE = DATA / "knowledge.json"
PENDING_FILE = DATA / "pending_contributions.jsonl"
app = FastAPI(title="AgriVision Local", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=1000)
    crop: str | None = Field(default=None, max_length=80)
    language: str = "en"


class Contribution(BaseModel):
    crop: str = Field(min_length=2, max_length=80)
    problem: str = Field(min_length=2, max_length=120)
    observation: str = Field(min_length=5, max_length=1500)
    suggested_advice: str = Field(min_length=5, max_length=2000)
    source: str = Field(min_length=2, max_length=300)


class CropRecommendationInput(BaseModel):
    N: float = Field(ge=0)
    P: float = Field(ge=0)
    K: float = Field(ge=0)
    temperature: float
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0)


class ProductionInput(BaseModel):
    State_Name: str = Field(min_length=2, max_length=100)
    District_Name: str = Field(min_length=2, max_length=100)
    Crop_Year: int = Field(ge=1997, le=2100)
    Season: str = Field(min_length=2, max_length=40)
    Crop: str = Field(min_length=2, max_length=100)
    Temperature: float
    Humidity: float = Field(ge=0, le=100)
    Soil_Moisture: float = Field(ge=0, le=100)
    Area: float = Field(gt=0)


def load_kb():
    try:
        return json.loads(KB_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []


def words(text):
    return {w for w in re.findall(r"[a-zA-Z][a-zA-Z-]{1,}|[\u0900-\u097f]+", text.lower()) if len(w) > 2}


def retrieve(message, crop=None, limit=4):
    query = words(f"{message} {crop or ''}")
    ranked = []
    for entry in load_kb():
        aliases = [str(x).lower() for x in entry.get("c", [])]
        keys = [str(x).lower() for x in entry.get("k", [])]
        name = str(entry.get("n", ""))
        score = 2 * len(query & words(" ".join([name, *keys, *aliases])))
        if crop and any(crop.lower() == a or crop.lower() in a or a in crop.lower() for a in aliases):
            score += 5
        score += sum(4 for k in keys if len(k) > 2 and k in message.lower())
        if score:
            ranked.append((score, entry))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in ranked[:limit]]


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "running", "knowledge_entries": len(load_kb()),
            "generative_model": "not trained / not loaded",
            "scan_model": "ready" if load_vision_model() else "not trained / not loaded",
            "crop_recommender": "ready" if load_model("crop_recommender.joblib") else "not trained",
            "production_estimator": "ready" if load_model("production_estimator.joblib") else "not trained"}


@app.post("/api/recommend-crop")
def recommend_crop_endpoint(req: CropRecommendationInput):
    result = recommend_crop(req.model_dump(), top_n=3)
    if result is None:
        raise HTTPException(status_code=503, detail="Crop recommender is not trained. Run training/train_tabular_models.py first.")
    return {"recommendations": result,
            "note": "Ranking from the supplied crop recommendation dataset. Scores are model scores, not guaranteed probabilities of success. The CSV does not document measurement units; enter matching values."}


@app.post("/api/predict-production")
def predict_production_endpoint(req: ProductionInput):
    result = estimate_production(req.model_dump())
    if result is None:
        raise HTTPException(status_code=503, detail="Production estimator is not trained. Run training/train_tabular_models.py first.")
    return {**result,
            "note": "Estimate based on historical records from 1997–2014 in 7 states. The CSV does not document production or area units. Treat this as an experimental estimate, not a yield guarantee."}


@app.post("/api/chat")
def chat(req: ChatRequest):
    matches = retrieve(req.message, req.crop)
    if not matches:
        return {"mode": "knowledge_retrieval", "matches": [],
                "message": "No sufficiently relevant note was found. Add the crop and visible symptoms, or contact your local KVK. No diagnosis was generated."}
    return {"mode": "knowledge_retrieval", "matches": [
        {"title": e.get("n", "Knowledge note"), "treatment": e.get("cure", []),
         "prevention": e.get("prev", []), "quality": e.get("q", []),
         "source": "AgriVision crop knowledge base"} for e in matches],
        "message": "These are the closest notes retrieved from the current knowledge base, not a model-generated diagnosis. Confirm treatment with your local KVK."}


@app.post("/api/contributions", status_code=202)
def contribute(req: Contribution):
    DATA.mkdir(exist_ok=True)
    record = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
              "status": "pending_review", **req.model_dump()}
    with PENDING_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"id": record["id"], "status": "pending_review",
            "message": "Saved for expert review. This will not become trusted advice automatically."}


@app.post("/api/scan")
def scan(file: UploadFile = File(...), crop: str | None = Form(default=None)):
    max_bytes = 10 * 1024 * 1024
    image_bytes = file.file.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum upload size is 10 MB.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Choose an image before scanning.")
    try:
        predictions = predict_plant_image(image_bytes, crop=crop)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if predictions is None:
        raise HTTPException(status_code=503, detail="The trained photo checkpoint is missing. Place best_model.pt at data/models/plant_disease_cnn.pt and redeploy.")
    return {"predictions": predictions,
            "note": "Experimental classifier trained on a small image set. Scores are not calibrated probabilities. Confirm any suspected disease with a local KVK or plant pathologist; do not use this result alone to choose treatment."}


@app.get("/api/admin/pending")
def pending():
    # Local-only review helper. Do not expose this service publicly without authentication.
    if __import__("os").getenv("AGRIVISION_LOCAL_ADMIN") != "I_UNDERSTAND_LOCAL_ONLY":
        raise HTTPException(status_code=403, detail="Enable the explicit local-only admin flag to inspect submissions.")
    if not PENDING_FILE.exists():
        return {"pending": []}
    records = [json.loads(x) for x in PENDING_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {"pending": [x for x in records if x.get("status") == "pending_review"]}
