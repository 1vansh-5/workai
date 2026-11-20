"""Train a small AgriVision image classifier from random initialization."""
import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import AgriVisionCNN


class CaseDataset(Dataset):
    def __init__(self, rows, image_root, labels, image_size, augment=False):
        self.rows = rows
        self.image_root = Path(image_root)
        self.labels = labels
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.image_root / row["relative_path"]
        image = Image.open(path).convert("RGB")
        image = ImageOps.fit(image, (self.image_size, self.image_size), method=Image.Resampling.BILINEAR)
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.25:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        # RGB to float CHW, scaled to [0, 1], then normalized to [-1, 1].
        pixels = torch.tensor(list(image.getdata()), dtype=torch.float32)
        pixels = pixels.view(self.image_size, self.image_size, 3).permute(2, 0, 1) / 255.0
        pixels = (pixels - 0.5) / 0.5
        label = f"{row['crop_common'].strip()}::{row['condition_label'].strip()}"
        return pixels, self.labels[label]


def read_rows(manifest):
    with Path(manifest).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def run_epoch(model, loader, loss_fn, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    if total == 0:
        raise ValueError("A split has no examples.")
    return {"loss": total_loss / total, "accuracy": total_correct / total, "examples": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--image-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64:
        p.error("epochs and batch-size must be positive; image-size must be at least 64")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_rows(args.manifest)
    approved = [r for r in rows if (r.get("review_status") or "").strip().lower() == "approved"]
    labels = sorted({f"{r['crop_common'].strip()}::{r['condition_label'].strip()}" for r in approved})
    if len(labels) < 2:
        raise SystemExit("Need at least two expert-approved labels to train a classifier.")
    label_to_id = {name: i for i, name in enumerate(labels)}
    train_rows = [r for r in approved if (r.get("split") or "").strip() == "train"]
    val_rows = [r for r in approved if (r.get("split") or "").strip() == "val"]
    if not train_rows or not val_rows:
        raise SystemExit("Need approved train and val examples. Keep test examples out of model selection.")
    train_classes = {f"{r['crop_common'].strip()}::{r['condition_label'].strip()}" for r in train_rows}
    absent = set(labels) - train_classes
    if absent:
        raise SystemExit("Labels absent from train split: " + ", ".join(sorted(absent)))

    args.output.mkdir(parents=True, exist_ok=True)
    train_ds = CaseDataset(train_rows, args.image_root, label_to_id, args.image_size, augment=True)
    val_ds = CaseDataset(val_rows, args.image_root, label_to_id, args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AgriVisionCNN(len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_val = float("inf")
    history = []
    print(f"Device: {device}; classes: {len(labels)}; train: {len(train_rows)}; val: {len(val_rows)}")
    print("Initialization: random; no pretrained weights loaded.")

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = run_epoch(model, val_loader, loss_fn, None, device)
        item = {"epoch": epoch, "train": train_metrics, "val": val_metrics,
                "seconds": round(time.time() - started, 1)}
        history.append(item)
        print(f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.3f} val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.3f}")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "labels": labels,
                "image_size": args.image_size,
                "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                "initialization": "random",
                "warning": "Research baseline only; not validated for diagnosis or treatment.",
            }, args.output / "best_model.pt")

    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "labels.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")
    print(f"Saved best validation checkpoint to {args.output / 'best_model.pt'}")
    print("Test split was not loaded. Do not report validation accuracy as field performance.")


if __name__ == "__main__":
    main()
