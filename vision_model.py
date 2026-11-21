"""Load the trained AgriVision image classifier and classify uploaded photos."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from training.model import AgriVisionCNN

MODEL_PATH = Path(__file__).resolve().parent / "data" / "models" / "plant_disease_cnn.pt"


@lru_cache(maxsize=1)
def load_vision_model():
    if not MODEL_PATH.is_file():
        return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    model = AgriVisionCNN(len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels, int(checkpoint.get("image_size", 224))


def predict_plant_image(image_bytes: bytes, crop: str | None = None, top_n: int = 3):
    loaded = load_vision_model()
    if loaded is None:
        return None
    model, labels, image_size = loaded
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a readable image. Please use JPG or PNG.") from exc
    image = ImageOps.fit(
        image, (image_size, image_size), method=Image.Resampling.BILINEAR
    )
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor)[0], dim=0).cpu().numpy()

    candidates = list(range(len(labels)))
    if crop:
        crop_key = crop.strip().casefold()
        candidates = [i for i, label in enumerate(labels)
                      if label.split("::", 1)[0].strip().casefold() == crop_key]
        if not candidates:
            supported = sorted({label.split("::", 1)[0] for label in labels})
            raise ValueError("This photo model supports only: " + ", ".join(supported))
    order = sorted(candidates, key=lambda i: float(probabilities[i]), reverse=True)[:top_n]
    results = []
    for i in order:
        crop_name, _, condition = labels[i].partition("::")
        results.append({"crop": crop_name, "condition": condition or labels[i],
                        "score": float(probabilities[i])})
    return results
