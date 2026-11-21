"""Export AgriVision's trained CNN for browser inference (run locally only)."""
import argparse
import json
from pathlib import Path

import torch

from model import AgriVisionCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to runs/icar-v2/best_model.pt")
    parser.add_argument("--output", type=Path, default=Path("static/plant_disease.onnx"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install ONNX locally first: python -m pip install onnx") from exc

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = checkpoint["labels"]
    size = int(checkpoint.get("image_size", 224))
    model = AgriVisionCNN(len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(model, dummy, args.output, input_names=["image"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    (args.output.parent / "labels.json").write_text(
        json.dumps({"labels": labels, "image_size": size}, indent=2), encoding="utf-8")
    print(f"Created {args.output} and {args.output.parent / 'labels.json'}")


if __name__ == "__main__":
    main()
