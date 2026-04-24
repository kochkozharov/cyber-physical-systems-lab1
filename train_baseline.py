from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from ultralytics import RTDETR, YOLO

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_YAML = PROJECT_ROOT / "data" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "baseline"

EPOCHS = 5
IMGSZ = 640
# batch=4 avoids a torch 2.11 + MPS indexing bug in Ultralytics' TAL assigner
# (``torch.AcceleratorError: index N out of bounds`` when mask_gt has
# more than ~100 True entries). Using 4 keeps training reliable on M-series.
BATCH = 4
SEED = 42


def pick_device() -> str:
    # Ultralytics 8.4.x doesn't auto-select MPS on macOS — without an explicit
    # ``device`` it silently falls back to CPU (~10× slower on M-series).
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


def train_yolo11n() -> None:
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
    )


def train_rtdetr_l() -> None:
    model = RTDETR("rtdetr-l.pt")
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
    )


def main() -> int:
    if not DATA_YAML.exists():
        print(f"Missing {DATA_YAML}. Run prepare_dataset.py first.", file=sys.stderr)
        return 1
    train_yolo11n()
    train_rtdetr_l()
    return 0


if __name__ == "__main__":
    sys.exit(main())
