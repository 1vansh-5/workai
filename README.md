# AgriVision local project

This folder contains the designed local web app, its crop knowledge search, reviewed user-submission queue, two trained tabular machine-learning models, and the datasets used to train them.

## New trained models

- **Crop recommender:** a random forest classifier trained from random initialization on `data/training/crop_recommendation.csv` (2,200 rows, 22 crop labels). It returns the three highest-scoring crop matches from soil/weather measurements.
- **Historical production estimator:** a random forest regressor trained from random initialization on `data/training/crop_production.csv` (49,784 usable rows). It estimates historical production from location, year, season, crop, weather, soil moisture and area.
- Training results and the exact data split are recorded in `data/models/training_report.json`.

The crop recommender's held-out random 20% accuracy was 99.3% and top-three accuracy was 100%. The compact production model (60 trees, depth limited to 18) was trained on 1997–2013 and evaluated on 2014: R² was 0.865, median absolute error was about 193 dataset units, and mean absolute error was about 251,238 dataset units. This large difference reflects a highly skewed production target and outliers; do not interpret the R² as a guarantee for a particular farm. The source CSVs do not document measurement or production units. Keep the same conventions as the source data.

These are supervised tabular models, not a general language model. The crop files contain no leaf images or disease labels. Therefore, the photo scanner remains unavailable, and chat searches the included knowledge notes rather than generating free-form diagnoses. User submissions are saved for human review; they do not automatically retrain the models or become advice.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Both trained model files are under `data/models/`; the app loads them locally and does not call a hosted AI service. This is the local demonstration setup, not a hardened public production service.

## Retrain the tabular models

Keep the CSVs at the paths under `data/training/`, install `requirements.txt`, then run:

```bash
python training/train_tabular_models.py
```

The script overwrites the two model artifacts and training report. Crop recommendation uses a stratified 80/20 split. Production estimation keeps 2014 out of training and uses it as a time-based holdout.

## APIs

- `POST /api/recommend-crop` accepts `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`.
- `POST /api/predict-production` accepts state, district, year, season, crop, temperature, humidity, soil moisture and area.
- `POST /api/chat` retrieves matching entries from the local knowledge file.
- `POST /api/contributions` saves a proposed note for review.
- `POST /api/scan` returns an explicit unavailable response until a separate expert-labeled image dataset is collected and a vision model is trained.

## Other project functions

- Search the included crop notes and display matching treatment, prevention and quality notes.
- Submit farmer observations into a pending queue.
- Review suggestions with `python review_contributions.py`; only approved entries are added to `data/knowledge.json`.
- Use the photo picker interface, which clearly says when no trained image model is loaded.

## Publishing

The two CSVs were supplied for this project. Check their source, license and attribution requirements before publishing the CSVs or trained artifacts in a public repository. Do not commit private farmer photos or contact details. The starter app also needs authentication, HTTPS, rate limits and data-retention controls before public deployment.

The knowledge file came from the linked project: https://github.com/1vansh-5/Agrivision-Ai-major-pro. Have a qualified local agronomist or KVK review advice before field use.
