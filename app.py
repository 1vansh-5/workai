"""Local AgriVision prototype: knowledge retrieval and reviewed user contributions.

No hosted AI is used and no generative model is claimed to be trained.
"""
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tabular_models import estimate_production, load_model, recommend_crop

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
KB_FILE = DATA / "knowledge.json"
PENDING_FILE = DATA / "pending_contributions.jsonl"
KCC_RESOURCE_ID = os.getenv(
    "DATA_GOV_IN_RESOURCE_ID",
    "cef25fe2-9231-4128-8aec-2c948fedd43f",
)
KCC_API_URL = f"https://api.data.gov.in/resource/{KCC_RESOURCE_ID}"
KCC_CACHE_SECONDS = 60 * 60
KCC_CACHE = {}
KCC_STATUS = {"status": "not_configured", "records": 0}
LOGGER = logging.getLogger("uvicorn.error")

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
    return {
        word
        for word in re.findall(
            r"[a-zA-Z][a-zA-Z-]{1,}|[\u0900-\u097f]+",
            text.lower(),
        )
        if len(word) > 2
    }


GENERIC_MATCH_TERMS = {
    "plant", "crop", "leaf", "leaves", "spot", "spots",
    "blight", "disease", "symptom", "symptoms", "affected",
}


def retrieve(message, crop=None, limit=4):
    """Find symptom-specific notes and avoid matching on generic crop/leaf words."""
    query = words(message) - words(crop or "") - GENERIC_MATCH_TERMS
    ranked = []

    for entry in load_kb():
        aliases = [str(value).lower() for value in entry.get("c", [])]

        if crop and not any(
            crop.lower() == alias
            or crop.lower() in alias
            or alias in crop.lower()
            for alias in aliases
        ):
            continue

        keys = [str(value).lower() for value in entry.get("k", [])]
        name = str(entry.get("n", ""))
        note_terms = words(" ".join([name, *keys])) - GENERIC_MATCH_TERMS
        score = len(query & note_terms)

        if score > 0:
            ranked.append((score, entry))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in ranked[:limit]]

def _field(record, *names):
    """Get a KCC API field without depending on schema capitalization."""
    lowered = {str(key).casefold(): value for key, value in record.items()}

    for name in names:
        value = lowered.get(name.casefold())
        if value not in (None, ""):
            return str(value).strip()

    return ""


def fetch_kcc_records(crop):
    """Fetch and cache a crop-specific KCC API sample."""
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    crop_value = (crop or "").strip().title()

    if not api_key:
        KCC_STATUS.update(status="not_configured", records=0)
        return []

    if crop_value.casefold() not in {"rice", "maize"}:
        return []

    cache_key = crop_value.casefold()
    cached = KCC_CACHE.get(cache_key)
    now = time.time()

    # Reuse a successful response for one hour.
    if cached and now - cached[0] < KCC_CACHE_SECONDS:
        return cached[1]

    params = {
        "api-key": api_key,
        "format": "json",
        "limit": "1000",
        "offset": "0",
        "filters[Crop]": crop_value,
    }

    request = Request(
        KCC_API_URL + "?" + urlencode(params),
        headers={
            "Accept": "application/json",
            "User-Agent": "AgriVision/1.0",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        KCC_STATUS.update(status=f"http_error:{exc.code}", records=0)
        LOGGER.warning("KCC API returned HTTP %s", exc.code)
        return cached[1] if cached else []

    except URLError as exc:
        # Do not log the request URL because it contains the API key.
        reason = getattr(exc, "reason", exc)
        KCC_STATUS.update(
            status=f"connection_error:{type(reason).__name__}",
            records=0,
        )
        LOGGER.warning("KCC API connection failed: %r", reason)
        return cached[1] if cached else []

    except (
        TimeoutError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        KCC_STATUS.update(
            status=f"request_failed:{type(exc).__name__}",
            records=0,
        )
        LOGGER.warning("KCC API request failed: %r", exc)
        return cached[1] if cached else []

    records = payload.get("records", []) if isinstance(payload, dict) else []

    if not isinstance(records, list):
        records = []

    records = [row for row in records if isinstance(row, dict)]
    KCC_CACHE[cache_key] = (now, records)

    KCC_STATUS.update(
        status="connected" if records else "connected_no_records",
        records=len(records),
    )
    LOGGER.info(
        "KCC API returned %s records for %s",
        len(records),
        crop_value,
    )

    return records


def retrieve_kcc(message, crop=None, limit=3):
    """Rank KCC answers and return source metadata with each match."""
    if not crop:
        return []

    query_words = words(message)
    ranked = []

    for record in fetch_kcc_records(crop):
        question = _field(record, "QueryText", "Query", "FarmerQuery")
        answer = _field(record, "KccAns", "KCCAns", "Answer", "Response")
        record_crop = _field(record, "Crop")

        if not question or not answer:
            continue

        if record_crop and crop.casefold() not in record_crop.casefold():
            continue

        overlap = len(query_words & words(question))
        if overlap < 1:
            continue

        state = _field(record, "StateName", "State")
        district = _field(record, "DistrictName", "District")
        created = _field(record, "CreatedOn", "Date", "Created")

        ranked.append(
            (
                overlap,
                {
                    "title": question,
                    "answer": answer,
                    "source": "Kisan Call Centre (KCC), data.gov.in",
                    "updated": created or None,
                    "region": ", ".join(
                        value for value in (district, state) if value
                    ) or None,
                },
            )
        )

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in ranked[:limit]]


def crop_from_message(message):
    match = re.search(r"\b(rice|maize)\b", message, re.IGNORECASE)
    return match.group(1).title() if match else None


def retrieve_for_prediction(crop, condition, limit=2):
    """Match an image label to notes for the same crop and condition."""
    def clean(value):
        return " ".join(
            re.findall(r"[a-z0-9]+", str(value).casefold())
        )

    crop_key = clean(crop)
    condition_key = clean(condition)
    condition_tokens = set(condition_key.split())
    ranked = []

    for entry in load_kb():
        aliases = [clean(value) for value in entry.get("c", [])]

        if crop_key not in aliases and "any" not in aliases:
            continue

        title = clean(entry.get("n", ""))
        keywords = [clean(value) for value in entry.get("k", [])]
        phrases = [title, *keywords]

        exact = any(
            phrase
            and (
                phrase == condition_key
                or phrase in condition_key
                or condition_key in phrase
            )
            for phrase in phrases
        )

        overlap = max(
            (
                len(condition_tokens & set(phrase.split()))
                for phrase in phrases
            ),
            default=0,
        )

        # Avoid matching on a vague shared word such as "blight".
        score = 100 if exact else overlap

        if score >= 2:
            ranked.append((score, entry))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "title": entry.get("n", "Knowledge note"),
            "cause": entry.get("cause"),
            "treatment": entry.get("cure", []),
            "prevention": entry.get("prev", []),
            "quality": entry.get("q", []),
            "source": entry.get("source", "Source not recorded in this note"),
            "region": entry.get("region"),
            "updated": entry.get("updated"),
        }
        for _, entry in ranked[:limit]
    ]


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "running",
        "knowledge_entries": len(load_kb()),
        "kcc_api": {
            **KCC_STATUS,
            "configured": bool(os.getenv("DATA_GOV_IN_API_KEY")),
        },
        "generative_model": "not trained / not loaded",
        "scan_model": "browser ONNX model (static/plant_disease.onnx)",
        "crop_recommender": (
            "ready"
            if load_model("crop_recommender.joblib")
            else "not trained"
        ),
        "production_estimator": (
            "ready"
            if load_model("production_estimator.joblib")
            else "not trained"
        ),
    }


@app.post("/api/recommend-crop")
def recommend_crop_endpoint(req: CropRecommendationInput):
    result = recommend_crop(req.model_dump(), top_n=3)

    if result is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Crop recommender is not trained. "
                "Run training/train_tabular_models.py first."
            ),
        )

    return {
        "recommendations": result,
        "note": (
            "Ranking from the supplied crop recommendation dataset. "
            "Scores are model scores, not guaranteed probabilities of success. "
            "The CSV does not document measurement units; enter matching values."
        ),
    }


@app.post("/api/predict-production")
def predict_production_endpoint(req: ProductionInput):
    result = estimate_production(req.model_dump())

    if result is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Production estimator is not trained. "
                "Run training/train_tabular_models.py first."
            ),
        )

    return {
        **result,
        "note": (
            "Estimate based on historical records from 1997–2014 in 7 states. "
            "The CSV does not document production or area units. "
            "Treat this as an experimental estimate, not a yield guarantee."
        ),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    crop = req.crop or crop_from_message(req.message)
    local_matches = retrieve(req.message, crop)
    kcc_matches = retrieve_kcc(req.message, crop)

    matches = [
        {
            "title": entry.get("n", "Knowledge note"),
            "treatment": entry.get("cure", []),
            "prevention": entry.get("prev", []),
            "quality": entry.get("q", []),
            "cause": entry.get("cause"),
            "source": entry.get("source", "Source not recorded in this note"),
            "region": entry.get("region"),
            "updated": entry.get("updated"),
        }
        for entry in local_matches
    ]

    matches.extend(kcc_matches)
    matches = matches[:6]

    if not matches:
        return {
            "mode": "knowledge_retrieval",
            "matches": [],
            "message": (
                "No sufficiently relevant note or KCC answer was found. "
                "Choose Rice or Maize and describe the symptoms, "
                "or contact your local KVK. No diagnosis was generated."
            ),
        }

    if kcc_matches:
        response_message = (
            "These include retrieved KCC answers and local notes. "
            "They are references, not a confirmed diagnosis. "
            "Check each source and date, and confirm treatment with your local KVK."
        )
    else:
        response_message = (
            "These are matching local crop notes, not a confirmed diagnosis. "
            "Confirm treatment with your local KVK."
        )

    return {
        "mode": "knowledge_retrieval",
        "matches": matches,
        "message": response_message,
    }


@app.post("/api/contributions", status_code=202)
def contribute(req: Contribution):
    DATA.mkdir(exist_ok=True)

    record = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending_review",
        **req.model_dump(),
    }

    with PENDING_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "id": record["id"],
        "status": "pending_review",
        "message": (
            "Saved for expert review. "
            "This will not become trusted advice automatically."
        ),
    }


@app.get("/api/admin/pending")
def pending():
    # Local-only review helper. Do not expose publicly without authentication.
    if os.getenv("AGRIVISION_LOCAL_ADMIN") != "I_UNDERSTAND_LOCAL_ONLY":
        raise HTTPException(
            status_code=403,
            detail=(
                "Enable the explicit local-only admin flag "
                "to inspect submissions."
            ),
        )

    if not PENDING_FILE.exists():
        return {"pending": []}

    records = [
        json.loads(line)
        for line in PENDING_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return {
        "pending": [
            record
            for record in records
            if record.get("status") == "pending_review"
        ]
    }
