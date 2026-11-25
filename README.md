# AgriVision

AgriVision is an experimental Rice and Maize image-classification demo with crop-note search, optional Kisan Call Centre (KCC) retrieval, crop recommendations, and historical production estimates.

The photo classifier runs in the browser using ONNX Runtime Web. Chat searches existing notes and, when available, KCC records. There is no trained generative language model and the app does not generate diagnoses.

## Current image model

- 852 image records in the manifest across 17 crop-specific labels.
- 596 training images, 119 validation images, and 137 held-out test images.
- The CNN was initialized from random weights and trained from scratch.
- The highest recorded validation accuracy was 50.4% on 119 images. The selected best-validation-loss checkpoint recorded 47.9%.
- The training script does not load or evaluate the test split. Do not describe validation results as field or test accuracy.
- A temporary 35% model-score cutoff suppresses guidance for low-scoring scans. This is a display safeguard, not a calibrated probability threshold.

The training photos are not stored in this repository. The manifest records metadata and image paths; the model files contain learned weights. Keep image files in your local dataset folder and confirm their license and consent before use.

## Run locally

Use a supported Python installation, then run these commands from the repository folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The browser loads `static/plant_disease.onnx` and `static/labels.json`. The selected photo is processed in the browser; the image file is not sent to the Python API for classification.

## Train the image model

Keep the extracted Rice and Maize image folders on your computer. The CSV manifest's `relative_path` values must point to files under `--image-root`.

Example in Windows PowerShell:

```powershell
python training\train_vision.py --manifest training\image_cases.csv --image-root "C:\path\to\aidata" --output runs\my-run --epochs 25 --batch-size 16
```

The script trains from random initialization. It uses only rows marked `train` for gradient updates and `val` for checkpoint selection. It explicitly skips rows marked `test`. Keep the test split untouched until final evaluation.

To use a newly trained checkpoint on the website, export it to ONNX and update the matching labels file in `static/`. Check the export script's arguments before running it.

## Existing tabular models

- **Crop recommender:** random forest classifier trained on 2,200 rows and 22 crop labels. The recorded held-out random-split accuracy is 99.3%; top-three accuracy is 100%.
- **Historical production estimator:** random forest trained on 49,784 usable rows. It trains on years before 2014 and holds out 2014. The recorded R² is 0.865; mean absolute error is about 251,238 dataset units and median absolute error is about 193 dataset units.

The source CSVs do not document measurement units. These are dataset results, not guarantees for a specific farm. See `data/models/training_report.json` for the recorded metrics and split details.

## API routes

- `GET /api/health` - reports service, model, knowledge-base, and KCC status.
- `POST /api/chat` - searches the local knowledge notes and, if the service responds, matching KCC records.
- `POST /api/recommend-crop` - returns ranked crop matches for the supplied soil and weather values.
- `POST /api/predict-production` - estimates historical production from the supplied location, year, crop, weather, soil, and area values.
- `POST /api/contributions` - accepts a suggestion for review. Suggestions are not automatically trusted or used to train a model.

The scan classifier is browser-side; there is no `/api/scan` route.

## Knowledge notes and image matching

The notes are stored in `data/knowledge.json`. Each note may include crop aliases (`c`), search terms (`k`), condition title (`n`), treatment notes (`cure`), prevention (`prev`), quality notes (`q`), and optional cause/source/date fields.

The search filters notes by crop and symptom-specific terms. Generic words such as “leaf” and “blight” are ignored when matching so they do not, by themselves, attach advice for another condition. If a useful matching note is missing, add one only after checking the diagnosis and source. The image classifier currently has classes for Rice and Maize; the knowledge file does not yet contain a reviewed note for every image class.

## KCC configuration

To enable the optional data.gov.in lookup, set these server-side environment variables in your deployment:

- `DATA_GOV_IN_API_KEY` - your private data.gov.in API key.
- `DATA_GOV_IN_RESOURCE_ID` - optional; the app has a default resource ID.

The app caches successful Rice and Maize responses in memory for one hour. If the key is missing or the external API request fails, local note search remains available. Check `/api/health` and the deployment's runtime logs for KCC status. Never place the API key in browser code, a committed file, or a public URL. KCC records are historical answers; verify dates, region, and current local guidance.

## Improve the model safely

1. Evaluate the 137 held-out test images with a separate evaluator. Report overall and per-class results plus a confusion matrix.
2. Check labels and group related photos from the same plant, plot, or visit into one data split.
3. Collect more varied, permitted field photos for classes the evaluation shows are confused.
4. Change one training setting at a time and compare validation results. Do not tune using the test set.
5. Calibrate the rejection cutoff on validation data; a model score is not automatically a true probability.
6. Keep user submissions pending until a knowledgeable reviewer verifies the label and source.

For agronomic treatment decisions, confirm the diagnosis and advice with a qualified local expert or KVK. This project is a research and learning demo, not a field-validated diagnostic service.

## Publishing and data handling

Check source licenses, attribution, and consent before publishing datasets or model artifacts. Do not commit private farmer photos or personal details. The contribution queue currently writes to a local JSONL file; serverless deployments may not provide durable storage. Use an external database or storage service before relying on submissions in a public deployment.
