"""Create an AgriVision CSV manifest from the ICAR Rice/Maize folder layout.

The generated rows are pending review by default, so the training script will
not use them until the labels are checked and approved in the CSV.
"""
import argparse
import csv
import hashlib
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")
SPLIT_WEIGHTS = (0.70, 0.15, 0.15)


def label_for(image: Path, root: Path):
    parts = image.relative_to(root).parts
    if len(parts) < 3:
        return None
    crop = parts[0].strip().lower()
    folder = parts[1].strip().lower()
    if crop not in {"rice", "maize"}:
        return None
    if folder == "healthy":
        condition = "healthy"
    elif folder in {"disease", "insect-pests", "insect_pests", "insect pests"}:
        if len(parts) < 4:
            return None
        condition = parts[2]
    else:
        return None
    condition = condition.lower()
    while condition and condition[0].isdigit():
        condition = condition[1:]
    condition = condition.strip(" _-").replace("-", "_").replace(" ", "_")
    return crop, condition


def split_sizes(count):
    if count == 1:
        return (1, 0, 0)
    if count == 2:
        return (1, 0, 1)
    train = max(1, round(count * SPLIT_WEIGHTS[0]))
    val = max(1, round(count * SPLIT_WEIGHTS[1]))
    # Reserve at least one image for test and keep all allocations in bounds.
    train = min(train, count - 2)
    val = min(val, count - train - 1)
    return (train, val, count - train - val)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path,
                        help="Dataset directory containing Rice and Maize folders")
    parser.add_argument("--output", type=Path, default=Path("image_cases.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--review-status", choices=("pending", "approved"), default="pending",
                        help="Use approved only after confirming the dataset labels and permission")
    args = parser.parse_args()
    root = args.image_root.resolve()
    if not root.is_dir():
        parser.error(f"image root is not a folder: {root}")

    grouped = {}
    skipped = []
    for image in sorted(root.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_for(image, root)
        if label is None:
            skipped.append(image)
            continue
        grouped.setdefault(label, []).append(image)

    if not grouped:
        parser.error("No images found in the expected Rice/Maize folder structure.")

    rng = random.Random(args.seed)
    rows = []
    for (crop, condition), images in sorted(grouped.items()):
        rng.shuffle(images)
        train_count, val_count, _ = split_sizes(len(images))
        for index, image in enumerate(images):
            rel = image.relative_to(root).as_posix()
            image_id = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
            rows.append({
                "image_id": image_id,
                "case_id": image_id,
                "group_id": image_id,
                "relative_path": rel,
                "crop_common": crop,
                "crop_scientific": "Oryza sativa" if crop == "rice" else "Zea mays",
                "condition_label": condition,
                "condition_type": "healthy" if condition == "healthy" else "disease_or_pest",
                "healthy_or_unknown": "healthy" if condition == "healthy" else "",
                "review_status": args.review_status,
                "split": ("train" if index < train_count else
                          "val" if index < train_count + val_count else "test"),
                "license_or_source": "ICAR dataset; verify applicable data-use terms",
                "notes": "Folder-derived label; verify before training. Group identity is per image because plant/visit IDs were not provided.",
            })

    columns = [
        "image_id", "case_id", "group_id", "relative_path", "crop_common",
        "crop_scientific", "condition_label", "condition_type", "healthy_or_unknown",
        "review_status", "split", "license_or_source", "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} image rows across {len(grouped)} labels to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} images outside the expected class folders.")
    print("Check labels and license_or_source. Rows marked pending are intentionally excluded from training.")
    print("If multiple photos show the same plant or visit, edit group_id so they share one value and keep them in one split.")


if __name__ == "__main__":
    main()
