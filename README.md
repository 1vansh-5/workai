# AgriVision local project

This folder contains the designed local web app, crop knowledge search, reviewed user-submission queue, two trained tabular models, and the user's from-scratch rice/maize photo classifier checkpoint.

## New trained models

- **Crop recommender:** a random forest classifier trained from random initialization on `data/training/crop_recommendation.csv` (2,200 rows, 22 crop labels). It returns the three highest-scoring crop matches from soil/weather measurements.
- **Historical production estimator:** a random forest regressor trained from random initialization on `data/training/crop_production.csv` (49,784 usable rows). It estimates historical production from location, year, season, crop, weather, soil moisture and area.
- Training results and the exact data split are recorded in `data/models/training_report.json`.

The crop recommender's held-out random 20% accuracy was 99.3% and top-three accuracy was 100%. The compact production model (60 trees, depth limited to 18) was trained on 1997–2013 and evaluated on 2014: R² was 0.865, median absolute error was about 193 dataset units, and mean absolute error was about 251,238 dataset units. This large difference reflects a highly skewed production target and outliers; do not interpret the R² as a guarantee for a particular farm. The source CSVs do not document measurement or production units. Keep the same conventions as the source data.

The photo classifier was trained from random initialization using 852 images in 17 Rice/Maize labels. Its best-validation-loss checkpoint reached 47.9% accuracy on the 119-image validation split (peak shown during training was 50.4%); the held-out test split has not been evaluated. Treat photo predictions as an early demonstration, not as reliable diagnosis. It only supports Rice and Maize labels. Image scanning now retrieves same-crop, condition-matched notes and displays available causes, management, prevention, and source metadata. It will say when no matching note or cause is recorded instead of inventing one. The existing notes do not yet cover every image class or include all causes/source citations. These models are not a general language model, and chat searches the included knowledge notes rather than generating free-form diagnoses. User submissions are saved for human review; they do not automatically retrain the models or become advice.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The tabular models and `data/models/plant_disease_cnn.pt` are loaded locally; photo uploads are sent to `/api/scan`. The photo model runs on CPU by default. This is a local demonstration setup, not a hardened public production service.

## Retrain the tabular models

Keep the CSVs at the paths under `data/training/`, install `requirements.txt`, then run:

```bash
python training/train_tabular_models.py
```

The script overwrites the two model artifacts and training report. Crop recommendation uses a stratified 80/20 split. Production estimation keeps 2014 out of training and uses it as a time-based holdout.

## APIs

- `POST /api/recommend-crop` accepts `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`.
- `POST /api/predict-production` accepts state, district, year, season, crop, temperature, humidity, soil moisture and area.
- `POST /api/chat` retrieves matching entries from the local knowledge file and, when configured, relevant rice/maize KCC Q&A.
- `POST /api/contributions` saves a proposed note for review.
- `POST /api/scan` returns an explicit unavailable response until a separate expert-labeled image dataset is collected and a vision model is trained.

## Other project functions

- Search the included crop notes and display matching treatment, prevention and quality notes.
- Submit farmer observations into a pending queue.
- Review suggestions with `python review_contributions.py`; only approved entries are added to `data/knowledge.json`.
- Upload Rice or Maize photos to get the classifier's top matches and any matching knowledge-base notes or KCC Q&A.

## Connect Kisan Call Centre (KCC) answers

The backend can optionally retrieve a bounded, crop-filtered sample of KCC Q&A
records from data.gov.in for chat and image scans. It uses resource ID
`cef25fe2-9231-4128-8aec-2c948fedd43f`, filters to Rice or Maize, and keeps a
one-hour in-memory cache per crop. If the key is absent or the API is
unavailable, local knowledge-base search continues to work.

In Render, add the environment variable `DATA_GOV_IN_API_KEY` with your private
data.gov.in key. Optionally set `DATA_GOV_IN_RESOURCE_ID` if the official API
details page provides a different resource ID. Never place the key in
`static/index.html`, a committed file, or a public URL. The API page checks the
key server-side in `app.py`; the browser only calls this app's `/api/chat` and
`/api/scan` endpoints.

KCC results are historical answers. The page shows their source, date and
available region; review treatment advice against current local guidance before
acting. This integration does not train the image model or a language model.

## Extend the crop knowledge notes

Each object in `data/knowledge.json` can include the existing fields `c` (crop aliases), `k` (search terms), `n` (condition title), `cure`, `prev`, and `q`, plus `cause`, `source`, `region`, `updated`, and `reviewed_by`. Add concise, crop-specific, expert-reviewed notes and include the original source and review date. The image scanner only attaches a note when both crop and condition match; it does not turn a model label into a cause by itself.

## Publishing

The two CSVs were supplied for this project. Check their source, license and attribution requirements before publishing the CSVs or trained artifacts in a public repository. Do not commit private farmer photos or contact details. The starter app also needs authentication, HTTPS, rate limits and data-retention controls before public deployment.

The knowledge file came from the linked project: https://github.com/1vansh-5/Agrivision-Ai-major-pro. Have a qualified local agronomist or KVK review advice before field use.
