# AgriVision local starter

This is a complete, local-first project folder with a designed web page, a FastAPI server, the linked project's crop notes, a reviewed-contribution queue, and the random-initialization photo-model training baseline.

## What works now

- Search the current crop notes and show the matched treatment, prevention and quality notes.
- Submit farmer observations into a pending queue.
- Review suggested entries interactively with `review_contributions.py`; only approved, rewritten notes enter `data/knowledge.json`.
- Use the designed photo-picker interface. The scan endpoint deliberately reports that no trained image model is loaded instead of guessing.
- Validate a photo manifest and train a small image classifier from random weights after expert-labeled data is collected.

## What is not included

There is no trained model in this package. The chat is knowledge retrieval, not natural language generation. The training code covers a starter image classifier only; it does not train a language model. A credible from-scratch chat model requires a large, licensed text corpus, expert-reviewed instruction data and GPU compute. Do not present this prototype as a diagnosis system.

## Run locally

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. The app stores farmer submissions in `data/pending_contributions.jsonl`. Review them on the same machine with:

```bash
python review_contributions.py
```

The review utility updates the local JSON knowledge file; keep a backup and have a qualified agronomy reviewer approve any treatment advice. This starter is intended for local development. Add authentication, HTTPS, rate limits, data retention and privacy protections before hosting it for public use.

## Train the photo model when data is ready

Put images under a data folder and complete `training/image_cases_template.csv`. Only expert-approved rows are used. The `split` column must be grouped by farm/plant/outbreak to prevent leakage.

```bash
python training/validate_manifest.py --manifest training/image_cases.csv --image-root data
python training/train_vision.py --manifest training/image_cases.csv --image-root data --output runs/vision-v0 --epochs 25 --batch-size 16
```

The training script selects by validation loss and does not inspect the test split. Its checkpoint is not safe for treatment decisions until a locked, farm-separated evaluation is completed and reviewed.

## Put it in your GitHub repository

Copy the contents of this folder into the root of your repository. Keep your existing project in a backup or separate branch first. Then review the changes, commit, and push from your own GitHub setup. Do not commit private farmer photos or contact details.

## Data and advice provenance

`data/knowledge.json` was copied from the linked AgriVision repository. Have a local agronomist/KVK review it before field use. Public repository: https://github.com/1vansh-5/Agrivision-Ai-major-pro
