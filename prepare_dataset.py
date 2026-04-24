from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import requests
import yaml
from tqdm import tqdm


DATASET_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
)
PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_DIR = DATASET_ROOT / "african-wildlife"
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = DATA_DIR / "dataset.yaml"
CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]
SPLITS = ("train", "val", "test")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1 << 15):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))


def _splits_ready() -> bool:
    for split in SPLITS:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        if not img_dir.exists() or not any(img_dir.iterdir()):
            return False
        if not lbl_dir.exists() or not any(lbl_dir.iterdir()):
            return False
    return True


def ensure_dataset() -> None:
    if _splits_ready():
        print(f"Dataset already present at {DATASET_DIR}")
        return

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_ROOT / "african-wildlife.zip"
    if not zip_path.exists():
        print(f"Downloading dataset from {DATASET_URL}")
        download_file(DATASET_URL, zip_path)

    print(f"Extracting into {DATASET_ROOT}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATASET_ROOT)

    # The archive extracts as ``images/`` and ``labels/`` directly next to the
    # zip file — consolidate those into ``dataset/african-wildlife/``.
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("images", "labels"):
        src = DATASET_ROOT / subdir
        dst = DATASET_DIR / subdir
        if src.exists() and src.resolve() != dst.resolve():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))

    # Remove byproducts left at the extraction root.
    for leftover in ("african-wildlife.yaml", "LICENSE.txt"):
        p = DATASET_ROOT / leftover
        if p.exists():
            p.unlink()
    zip_path.unlink(missing_ok=True)

    if not _splits_ready():
        raise RuntimeError(
            f"Dataset extraction finished but expected splits are missing under {DATASET_DIR}"
        )
    print(f"Dataset prepared at {DATASET_DIR}")


def write_config() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote config to {CONFIG_PATH}")


def count_split(split: str) -> int:
    img_dir = DATASET_DIR / "images" / split
    if not img_dir.exists():
        return 0
    return sum(1 for p in img_dir.iterdir() if p.is_file())


def main() -> int:
    ensure_dataset()
    write_config()

    train = count_split("train")
    val = count_split("val")
    test = count_split("test")
    print(f"Dataset splits: {train} train / {val} val / {test} test")
    print(f"Config: {CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
