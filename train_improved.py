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
    model = RTDETR("rtdetr-l.pt")
    # RT-DETR ignores ``label_smoothing`` — drop it to avoid an Ultralytics
    # "unused argument" warning in the logs.
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
    if not DATA_YAML.exists():
        print(f"Missing {DATA_YAML}. Run prepare_dataset.py first.", file=sys.stderr)
        return 1
    train_yolo11n_improved()
    train_rtdetr_l_improved()
    return 0


if __name__ == "__main__":
    sys.exit(main())
