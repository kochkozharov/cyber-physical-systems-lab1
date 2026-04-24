"""Retrain both detectors with a stronger augmentation and schedule recipe.

Hypotheses being tested (reported in the README):
 * Stronger augmentation (mosaic, mixup, HSV, flip, small rotation) improves
   generalisation on a modest-size dataset (~1500 images).
 * A cosine LR schedule with a short warmup stabilises training of the
   transformer detector.
 * Weight decay plus label smoothing regularise the CNN against overfitting.

Training runs for 15 epochs so the cosine schedule and augmentation have room
to actually contribute — the 5-epoch budget used for the baseline would make
warmup eat most of training.
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
RUNS_DIR = PROJECT_ROOT / "runs" / "improved"

EPOCHS = 15
IMGSZ = 640
# batch=4 avoids a torch 2.11 + MPS indexing bug in Ultralytics' TAL assigner;
# see ``train_baseline.py`` for details.
BATCH = 4
SEED = 42


def pick_device() -> str:
    """Return the best PyTorch device available (MPS preferred on macOS)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()

IMPROVED = dict(
    mosaic=1.0,
    mixup=0.15,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    fliplr=0.5,
    cos_lr=True,
    warmup_epochs=2,
    weight_decay=5e-4,
    label_smoothing=0.05,
)


def train_yolo11n_improved() -> None:
    """Fit YOLOv11n with the improved hyperparameter recipe."""
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        device=DEVICE,
        project=str(RUNS_DIR),
        name="yolo11n",
        exist_ok=True,
        verbose=True,
        **IMPROVED,
    )


def train_rtdetr_l_improved() -> None:
    """Fit RT-DETR-l with the improved hyperparameter recipe.

    RT-DETR ignores ``label_smoothing`` in its loss, so we drop it before
    passing kwargs to ``train`` to avoid an Ultralytics "unused argument"
    warning in the logs.
    """
    model = RTDETR("rtdetr-l.pt")
    params = {k: v for k, v in IMPROVED.items() if k != "label_smoothing"}
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        device=DEVICE,
        project=str(RUNS_DIR),
        name="rtdetr_l",
        exist_ok=True,
        verbose=True,
        **params,
    )


def main() -> int:
    """Entry point: ensure data is ready, then train both improved detectors."""
    if not DATA_YAML.exists():
        print(f"Missing {DATA_YAML}. Run prepare_dataset.py first.", file=sys.stderr)
        return 1
    train_yolo11n_improved()
    train_rtdetr_l_improved()
    return 0


if __name__ == "__main__":
    sys.exit(main())
