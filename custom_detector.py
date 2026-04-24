"""A four-class single-prediction detector implemented from scratch in PyTorch.

Architecture:
  * Backbone: MobileNet-style stem + six depthwise-separable blocks,
    progressively downsampling to a 512-dim global feature vector.
  * Two heads, each an MLP: one regresses the box ``(cx, cy, w, h)`` in
    normalised image coordinates (sigmoid output), the other classifies the
    object among the four wildlife classes.
  * Loss: SmoothL1 on the box + CrossEntropy on the class.

Simplification: the head outputs exactly one prediction per image. During
training each image's label file is collapsed to the single largest-area box;
at evaluation we match the one prediction against *any* GT box with IoU >= 0.5
and the same class. This is obviously not competitive with anchor-based heads
on crowded scenes, but it's a tractable educational baseline that exercises
the whole detection pipeline end-to-end.

Two variants are trained for comparison: one without augmentation, one with
(colour jitter + random horizontal flip).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "dataset" / "african-wildlife"
RUNS_DIR = PROJECT_ROOT / "runs" / "custom"
NUM_CLASSES = 4
IMG_SIZE = 224
SEED = 42


def get_device() -> torch.device:
    """Pick the best available PyTorch device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class WildlifeDataset(Dataset):
    """YOLO-format dataset collapsed to one (largest) box per image.

    Every image that has at least one ground-truth box contributes one
    training sample. The target is a 5-vector ``(cx, cy, w, h, cls)``.
    """

    def __init__(self, split: str, augment: bool = False):
        self.img_dir = DATA_DIR / "images" / split
        self.label_dir = DATA_DIR / "labels" / split
        self.samples = self._collect_samples()
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

    def _collect_samples(self) -> list[tuple[Path, int, float, float, float, float]]:
        """Walk the split folder and keep images whose label file has ≥1 box."""
        samples: list[tuple[Path, int, float, float, float, float]] = []
        if not self.img_dir.exists():
            return samples
        for img_path in sorted(self.img_dir.iterdir()):
            if not img_path.is_file():
                continue
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            boxes: list[tuple[int, float, float, float, float]] = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w, h = (float(x) for x in parts)
                    boxes.append((int(cls), cx, cy, w, h))
            if boxes:
                cls, cx, cy, w, h = max(boxes, key=lambda b: b[3] * b[4])
                samples.append((img_path, cls, cx, cy, w, h))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls, cx, cy, w, h = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        if self.augment:
            img = self.color_jitter(img)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                cx = 1.0 - cx
        img = self.normalize(self.to_tensor(img))
        target = torch.tensor([cx, cy, w, h, float(cls)], dtype=torch.float32)
        return img, target


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 with BN and ReLU between."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return x


class CustomDetector(nn.Module):
    """Backbone → GAP → box-regression head + classification head."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(32, 64),
            DepthwiseSeparableBlock(64, 128, stride=2),
            DepthwiseSeparableBlock(128, 128),
            DepthwiseSeparableBlock(128, 256, stride=2),
            DepthwiseSeparableBlock(256, 256),
            DepthwiseSeparableBlock(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.box_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.box_head(x), self.cls_head(x)


def compute_iou(a, b) -> float:
    """IoU between two boxes in (cx, cy, w, h) normalised coordinates."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def read_all_boxes(split: str, stem: str):
    """Return every ``(cls, cx, cy, w, h)`` tuple in the label file for ``stem``."""
    path = DATA_DIR / "labels" / split / f"{stem}.txt"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = (float(x) for x in parts)
                out.append((int(cls), cx, cy, w, h))
    return out


def evaluate(model: nn.Module, split: str, device: torch.device):
    """Return (precision, recall, map50_proxy) on ``split``.

    Each image contributes at most one matched prediction. The "mAP@0.5 proxy"
    reported is the fraction of predictions that matched any GT with IoU≥0.5
    and the correct class — this matches the reporting in the example README
    for a single-prediction head.
    """
    ds = WildlifeDataset(split)
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for img_path, *_ in ds.samples:
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            x = ds.normalize(ds.to_tensor(img)).unsqueeze(0).to(device)
            box_pred, cls_logits = model(x)
            box_pred = box_pred[0].cpu().tolist()
            cls_pred_idx = int(cls_logits[0].argmax().item())
            gts = read_all_boxes(split, img_path.stem)
            matched = False
            for gcls, gcx, gcy, gw, gh in gts:
                if gcls == cls_pred_idx and compute_iou(
                    box_pred, (gcx, gcy, gw, gh)
                ) >= 0.5:
                    matched = True
                    break
            if matched:
                tp += 1
                fn += max(0, len(gts) - 1)
            else:
                fp += 1
                fn += len(gts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    map50_proxy = precision
    return precision, recall, map50_proxy


def train_variant(
    variant_name: str,
    epochs: int,
    lr: float = 1e-3,
    augment: bool = False,
) -> tuple[float, float, float]:
    """Fit one variant and persist its weights under ``runs/custom/<name>``."""
    device = get_device()
    train_ds = WildlifeDataset("train", augment=augment)
    if len(train_ds) == 0:
        raise RuntimeError(
            "Training split is empty — did prepare_dataset.py run successfully?"
        )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    model = CustomDetector().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            # ``.contiguous()`` on the slices avoids an MPS error in
            # ``smooth_l1_loss`` ("view size is not compatible with input
            # tensor's size and stride") under torch 2.11.
            box_gt = targets[:, :4].contiguous()
            cls_gt = targets[:, 4].long().contiguous()
            box_pred, cls_logits = model(imgs)
            box_loss = F.smooth_l1_loss(box_pred, box_gt)
            cls_loss = F.cross_entropy(cls_logits, cls_gt)
            loss = box_loss + cls_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * imgs.size(0)
        avg = total / len(train_ds)
        print(f"[{variant_name}] epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    out_dir = RUNS_DIR / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best.pt")

    p, r, m = evaluate(model, "test", device)
    print(f"[{variant_name}] test  P={p:.3f}  R={r:.3f}  mAP@0.5={m:.3f}")
    return p, r, m


def main() -> int:
    """Entry point: train both variants and print a summary table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=== Custom detector: no augmentation ===")
    p0, r0, m0 = train_variant("no_aug", epochs=args.epochs, augment=False)
    random.seed(SEED)
    torch.manual_seed(SEED)
    print("\n=== Custom detector: with augmentation ===")
    p1, r1, m1 = train_variant("with_aug", epochs=args.epochs, augment=True)

    print("\n| Variant | P | R | mAP@0.5 |")
    print("|---|---|---|---|")
    print(f"| custom (no aug) | {p0:.3f} | {r0:.3f} | {m0:.3f} |")
    print(f"| custom (aug) | {p1:.3f} | {r1:.3f} | {m1:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
