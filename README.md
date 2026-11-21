# AgriVision training starter (from random initialization)

This folder contains a first, deliberately small image-classification training pipeline. The CNN weights start randomly; no pretrained model or external AI service is used.

It is **not a trained model** and is not suitable for farmer diagnosis yet. It only becomes useful after you collect and expert-review enough field images. It does not train the language/chat model; that needs a separate, licensed text corpus and reviewed Q&A set.

## Setup

Use Python 3.10 or newer, then install:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build suited to your computer from the official PyTorch install selector if the basic install does not detect your GPU. The pipeline also runs on CPU for small experiments, but training may be slow.

## Data layout

Use the `image_cases_template.csv` from the data collection pack as the manifest. Keep image files in a data directory and set `relative_path` to a path under it, for example:

```text
data/
  images/
    case001_leaf.jpg
  image_cases.csv
```

In the CSV, `relative_path` would be `images/case001_leaf.jpg`. Set `review_status` to `approved` only after expert review. Set `split` to `train`, `val`, or `test` after grouping every photo from the same plant/farm visit/outbreak together. Keep a locked test set from unseen farms; do not tune on it.

Use specific labels, including `healthy`, `unknown`, and `not_enough_evidence`. Do not relabel uncertainty as a disease. The classifier treats `crop_common` plus `condition_label` as one class, e.g. `rice::blast`.

## Check the manifest

For the ICAR folder layout (`Rice`/`Maize`, then `Healthy`, `Disease`, or
`Insect-pests`), generate a starting manifest from the dataset folders:

```powershell
python build_folder_manifest.py --image-root "C:\Users\tripa\OneDrive\Desktop\aidata" --output "C:\Users\tripa\OneDrive\Desktop\aidata\image_cases.csv"
```

The generator keeps disease and pest subfolders as separate labels and uses a
seeded per-label split (70/15/15 when there are enough examples). It defaults
all records to `pending`, so the training script will intentionally skip them.
Review the folder-derived labels and ICAR data-use terms, then change
`review_status` to `approved` only for labels you have verified. If you know
that photos show the same plant or field visit, give them a shared `group_id`
and keep that group in one split; the folder layout does not provide that
identity, so the generated per-image IDs cannot detect duplicate or related
photos across splits.

```bash
python validate_manifest.py --manifest data/image_cases.csv --image-root data
```

Fix all errors before training. Warnings about small classes are expected early on. Group leakage is an error: the same `group_id` cannot occur in multiple splits.

## Train the vision baseline

```bash
python train_vision.py --manifest data/image_cases.csv --image-root data --output runs/vision-v0 --epochs 25 --batch-size 16 --image-size 224
```

The script trains only rows marked `approved` and uses their predefined `train` and `val` splits. It does not use the test split to choose a model. It saves the best validation checkpoint and a JSON history file. Evaluate the locked test set only after selecting the model and freezing the code; add a dedicated evaluation script before reporting results to farmers.

## Important limits

- This is a learning baseline, not a diagnosis product. A small dataset may simply be memorized.
- Image labels should be confirmed by qualified agronomy reviewers. Keep disagreements and ambiguous cases explicit.
- A classifier only predicts among labels it was trained on; it cannot safely handle arbitrary unseen problems. Do not use its output for pesticide decisions.
- A photo classifier does not answer general farming questions. The chat agent needs a separately trained language model and should retrieve the current vetted knowledge base for every answer.
- Never report training accuracy as real-world accuracy. Use a locked, farm-separated field test and publish per-class results and abstention behavior.
