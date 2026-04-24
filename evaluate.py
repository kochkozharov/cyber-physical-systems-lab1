"""Evaluate trained detectors on the test split and print a metrics table.

Usage:
    python evaluate.py                                  # all available models
    python evaluate.py yolo11n_baseline rtdetr_l_baseline
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from ultralytics import RTDETR, YOLO

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_YAML = PROJECT_ROOT / "data" / "dataset.yaml"
EVAL_RUNS = PROJECT_ROOT / "runs" / "eval"


def pick_device() -> str:
    """Return the best PyTorch device available (MPS preferred on macOS)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()

MODELS = {
    "yolo11n_baseline": (YOLO, PROJECT_ROOT / "runs" / "baseline" / "yolo11n" / "weights" / "best.pt"),
    "rtdetr_l_baseline": (RTDETR, PROJECT_ROOT / "runs" / "baseline" / "rtdetr_l" / "weights" / "best.pt"),
    "yolo11n_improved": (YOLO, PROJECT_ROOT / "runs" / "improved" / "yolo11n" / "weights" / "best.pt"),
    "rtdetr_l_improved": (RTDETR, PROJECT_ROOT / "runs" / "improved" / "rtdetr_l" / "weights" / "best.pt"),
}


def evaluate_model(model_cls, weights_path: Path, name: str) -> dict:
    """Run ``.val(split='test')`` on ``weights_path`` and return key metrics."""
    model = model_cls(str(weights_path))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        device=DEVICE,
        project=str(EVAL_RUNS),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    return {
        "P": float(metrics.box.mp),
        "R": float(metrics.box.mr),
        "mAP@0.5": float(metrics.box.map50),
        "mAP@0.5:0.95": float(metrics.box.map),
    }


def main() -> int:
    """Evaluate every requested model key and print a summary markdown table."""
    keys = sys.argv[1:] or list(MODELS.keys())
    results: list[tuple[str, dict]] = []
    for key in keys:
        if key not in MODELS:
            print(f"Unknown model key: {key}", file=sys.stderr)
            return 1
        model_cls, path = MODELS[key]
        if not path.exists():
            print(f"Weights missing for {key}: {path}")
            continue
        print(f"\n=== Evaluating {key} ===")
        m = evaluate_model(model_cls, path, key)
        results.append((key, m))

    if not results:
        print("No models were evaluated — did training finish?", file=sys.stderr)
        return 1

    print("\n| Model | P | R | mAP@0.5 | mAP@0.5:0.95 |")
    print("|---|---|---|---|---|")
    for key, m in results:
        print(
            f"| {key} | {m['P']:.3f} | {m['R']:.3f} | "
            f"{m['mAP@0.5']:.3f} | {m['mAP@0.5:0.95']:.3f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
