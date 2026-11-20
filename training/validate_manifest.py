"""Validate the AgriVision image manifest and check split leakage."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED = ["image_id", "case_id", "group_id", "relative_path", "crop_common",
            "condition_label", "review_status", "split"]
SPLITS = {"train", "val", "test"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--image-root", required=True, type=Path)
    args = p.parse_args()

    errors, warnings = [], []
    counts = Counter()
    group_splits = defaultdict(set)
    seen_ids = set()
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit("Missing columns: " + ", ".join(missing))
        rows = list(reader)

    if not rows:
        raise SystemExit("Manifest has no data rows.")

    for line, row in enumerate(rows, start=2):
        prefix = f"row {line}"
        vals = {k: (row.get(k) or "").strip() for k in REQUIRED}
        for field in ("image_id", "case_id", "group_id", "relative_path", "crop_common", "condition_label"):
            if not vals[field]:
                errors.append(f"{prefix}: {field} is empty")
        if vals["image_id"] in seen_ids and vals["image_id"]:
            errors.append(f"{prefix}: duplicate image_id {vals['image_id']!r}")
        seen_ids.add(vals["image_id"])
        if vals["split"] not in SPLITS:
            errors.append(f"{prefix}: split must be one of {sorted(SPLITS)}")
        if vals["review_status"].lower() != "approved":
            warnings.append(f"{prefix}: row is not approved and will be skipped by training")
        rel = Path(vals["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"{prefix}: relative_path must stay under image root")
        elif not (args.image_root / rel).is_file():
            errors.append(f"{prefix}: image not found: {rel}")
        if vals["group_id"] and vals["split"] in SPLITS:
            group_splits[vals["group_id"]].add(vals["split"])
        counts[(vals["split"], f"{vals['crop_common']}::{vals['condition_label']}")] += 1

    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"group_id {group!r} leaks across splits: {sorted(splits)}")

    approved = [r for r in rows if (r.get("review_status") or "").strip().lower() == "approved"]
    for split in sorted(SPLITS):
        labels = {f"{r['crop_common'].strip()}::{r['condition_label'].strip()}"
                  for r in approved if (r.get("split") or "").strip() == split}
        if not labels:
            warnings.append(f"no approved examples in {split} split")
    label_counts = Counter(f"{r['crop_common'].strip()}::{r['condition_label'].strip()}"
                           for r in approved if (r.get("split") or "").strip() == "train")
    for label, n in sorted(label_counts.items()):
        if n < 20:
            warnings.append(f"training label {label!r} has only {n} approved examples")

    print(f"Rows: {len(rows)} | approved: {len(approved)}")
    print("Counts by split and label:")
    for (split, label), n in sorted(counts.items()):
        print(f"  {split:5} {label}: {n}")
    for msg in warnings:
        print("WARNING:", msg)
    for msg in errors:
        print("ERROR:", msg)
    if errors:
        raise SystemExit(f"Manifest has {len(errors)} error(s).")
    print("Manifest checks passed. This does not validate the correctness of expert labels.")


if __name__ == "__main__":
    main()
