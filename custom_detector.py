"""A four-class grid-based detector implemented from scratch in PyTorch.

Architecture (YOLOv1-style, anchor-free, single-scale):
  * Backbone: MobileNet-style stem + seven depthwise-separable blocks,
    progressively downsampling 224x224 → 7x7 with 512 channels.
  * Grid head: a 1x1 conv producing ``(5 + num_classes)`` channels per cell.
    For each of the 49 cells we predict:
      - ``obj`` (sigmoid): probability that a box centre lies in this cell
      - ``(dx, dy)`` (sigmoid): centre offset within the cell, in [0, 1]
      - ``(w, h)`` (sigmoid): box width/height as a fraction of the image
      - class logits (softmax over the four wildlife classes)
  * Loss: BCE(obj) over all cells + SmoothL1(box) + CE(class), the latter two
    only on the cell that contains each GT centre. Object-loss is weighted
    down vs the others to handle the 5:44 positive/negative imbalance.
  * Inference: per-class NMS (IoU ≥ 0.5) over cell predictions above a
    confidence threshold.

Metrics: proper mAP@0.5 computed by sorting all predictions by confidence and
walking the per-class precision-recall curve (COCO-style 101-point
interpolation).
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
GRID_S = 7
CELL = 1.0 / GRID_S
MAX_BOXES = 8  # per image; excess ground-truth boxes are dropped
SEED = 42


def get_device() -> torch.device:
    """Pick the best available PyTorch device (CUDA → MPS → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class WildlifeDataset(Dataset):
    """YOLO-format dataset returning image + padded box tensor.

    Each sample yields ``(image, boxes)`` where ``boxes`` is a
    ``(MAX_BOXES, 5)`` tensor with rows ``(cls, cx, cy, w, h)``; unused rows
    are filled with ``cls=-1`` sentinel so losses can mask them out. Images
    with zero GT boxes are dropped at load time.
    """

    def __init__(self, split: str, augment: bool = False):
        self.img_dir = DATA_DIR / "images" / split
        self.label_dir = DATA_DIR / "labels" / split
        self.augment = augment
        self.samples = self._collect_samples()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)

    def _collect_samples(self) -> list[tuple[Path, list[tuple[int, float, float, float, float]]]]:
        """Walk the split and keep ``(image_path, [box_tuples])`` pairs."""
        samples = []
        if not self.img_dir.exists():
            return samples
        for img_path in sorted(self.img_dir.iterdir()):
            if not img_path.is_file():
                continue
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            boxes = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w, h = (float(x) for x in parts)
                    boxes.append((int(cls), cx, cy, w, h))
            if boxes:
                samples.append((img_path, boxes))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        if self.augment:
            img = self.color_jitter(img)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = [(c, 1.0 - cx, cy, w, h) for (c, cx, cy, w, h) in boxes]
        img = self.normalize(self.to_tensor(img))

        padded = torch.full((MAX_BOXES, 5), -1.0, dtype=torch.float32)
        for i, (c, cx, cy, w, h) in enumerate(boxes[:MAX_BOXES]):
            padded[i] = torch.tensor([c, cx, cy, w, h], dtype=torch.float32)
        return img, padded


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 with BN and ReLU6 between."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu6(self.bn1(self.dw(x)), inplace=True)
        x = F.relu6(self.bn2(self.pw(x)), inplace=True)
        return x


class GridDetector(nn.Module):
    """Backbone → 1x1 grid head, outputs ``(B, 5+C, S, S)``."""

    def __init__(self, num_classes: int = NUM_CLASSES, grid_s: int = GRID_S):
        super().__init__()
        self.num_classes = num_classes
        self.grid_s = grid_s
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # 112
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableBlock(32, 64),                      # 112
            DepthwiseSeparableBlock(64, 128, stride=2),           # 56
            DepthwiseSeparableBlock(128, 128),                    # 56
            DepthwiseSeparableBlock(128, 256, stride=2),          # 28
            DepthwiseSeparableBlock(256, 256),                    # 28
            DepthwiseSeparableBlock(256, 512, stride=2),          # 14
            DepthwiseSeparableBlock(512, 512, stride=2),          # 7
        )
        self.head = nn.Conv2d(512, 5 + num_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        raw = self.head(x)  # (B, 5+C, S, S)
        return raw


def build_targets(boxes: torch.Tensor, grid_s: int = GRID_S):
    """Convert padded GT boxes into dense grid targets.

    ``boxes`` is ``(B, MAX_BOXES, 5)`` with ``-1`` padding in class column.
    Returns:
      * ``obj_t``: ``(B, S, S)`` float, 1.0 where a GT centre falls
      * ``box_t``: ``(B, S, S, 4)`` float, ``(dx, dy, w, h)`` in cell-local
        coords (offset) + image-fraction (size). Only meaningful where
        ``obj_t == 1``; zeros elsewhere.
      * ``cls_t``: ``(B, S, S)`` long, class index at each positive cell
        (``-1`` elsewhere — to mask cross-entropy).
    """
    device = boxes.device
    B = boxes.shape[0]
    obj_t = torch.zeros(B, grid_s, grid_s, device=device)
    box_t = torch.zeros(B, grid_s, grid_s, 4, device=device)
    cls_t = torch.full((B, grid_s, grid_s), -1, dtype=torch.long, device=device)

    # Iterate over each ground-truth box; cheap because MAX_BOXES is small.
    for b in range(B):
        for k in range(boxes.shape[1]):
            cls = int(boxes[b, k, 0].item())
            if cls < 0:
                continue
            cx, cy, w, h = boxes[b, k, 1:].tolist()
            gi = min(int(cx * grid_s), grid_s - 1)
            gj = min(int(cy * grid_s), grid_s - 1)
            # If two GT centres collide in the same cell, last one wins —
            # acceptable given MAX_BOXES ≤ 8 and the cell grid is 7×7=49.
            obj_t[b, gj, gi] = 1.0
            dx = cx * grid_s - gi
            dy = cy * grid_s - gj
            box_t[b, gj, gi, 0] = dx
            box_t[b, gj, gi, 1] = dy
            box_t[b, gj, gi, 2] = w
            box_t[b, gj, gi, 3] = h
            cls_t[b, gj, gi] = cls
    return obj_t, box_t, cls_t


def split_predictions(raw: torch.Tensor):
    """Split raw ``(B, 5+C, S, S)`` head output into activated components."""
    obj_logits = raw[:, 0]                         # (B, S, S)
    box_raw = raw[:, 1:5]                          # (B, 4, S, S)
    cls_logits = raw[:, 5:]                        # (B, C, S, S)
    box_act = torch.sigmoid(box_raw.permute(0, 2, 3, 1))  # (B, S, S, 4)
    return obj_logits, box_act, cls_logits


def grid_loss(raw: torch.Tensor, obj_t, box_t, cls_t, lambda_obj: float = 1.0,
              lambda_noobj: float = 0.5, lambda_box: float = 5.0,
              lambda_cls: float = 1.0):
    """Combined objectness + box + class loss.

    Box and class losses are masked to positive cells. Objectness uses
    separate weights for positive (with-object) and negative (empty) cells to
    compensate for the 5:44 imbalance — the same trick YOLOv1 used.
    """
    obj_logits, box_act, cls_logits = split_predictions(raw)
    pos = obj_t > 0.5
    neg = ~pos

    obj_target_float = obj_t
    # Per-cell BCE(with logits) over objectness
    obj_bce = F.binary_cross_entropy_with_logits(
        obj_logits, obj_target_float, reduction="none"
    )
    obj_loss = lambda_obj * obj_bce[pos].sum() + lambda_noobj * obj_bce[neg].sum()
    obj_loss = obj_loss / max(pos.sum().item() + 1, 1)

    if pos.any():
        # SmoothL1 on box coords, only on positive cells
        box_pos = box_act[pos]          # (N, 4)
        box_gt = box_t[pos]             # (N, 4)
        box_loss = lambda_box * F.smooth_l1_loss(box_pos, box_gt)

        # Cross-entropy on class, only on positive cells. cls_logits has shape
        # (B, C, S, S); flatten to (B*S*S, C) then index with the pos mask.
        cls_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, cls_logits.shape[1])
        pos_flat = pos.reshape(-1)
        cls_targets = cls_t.reshape(-1)[pos_flat].contiguous()
        cls_loss = lambda_cls * F.cross_entropy(cls_flat[pos_flat], cls_targets)
    else:
        box_loss = raw.sum() * 0.0
        cls_loss = raw.sum() * 0.0

    return obj_loss + box_loss + cls_loss, {
        "obj": float(obj_loss.item()),
        "box": float(box_loss.item()) if torch.is_tensor(box_loss) else 0.0,
        "cls": float(cls_loss.item()) if torch.is_tensor(cls_loss) else 0.0,
    }


def decode_predictions(raw: torch.Tensor, conf_thresh: float = 0.2):
    """Convert raw head output into a list of per-image predictions.

    Returns a list of length ``B``; each entry is a tensor ``(N_i, 6)`` with
    columns ``(cls, cx, cy, w, h, conf)``, unfiltered by NMS.
    """
    B, _, S, _ = raw.shape
    obj_logits, box_act, cls_logits = split_predictions(raw)
    obj_prob = torch.sigmoid(obj_logits)                                  # (B,S,S)
    cls_prob = F.softmax(cls_logits, dim=1)                               # (B,C,S,S)
    cls_scores, cls_ids = cls_prob.max(dim=1)                             # (B,S,S)

    preds = []
    device = raw.device
    cell_idx = torch.arange(S, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(cell_idx, cell_idx, indexing="ij")            # (S,S)

    for b in range(B):
        mask = obj_prob[b] > conf_thresh
        if mask.sum() == 0:
            preds.append(torch.zeros((0, 6), device=device))
            continue
        box_b = box_act[b][mask]                # (N, 4): dx, dy, w, h
        dx, dy, w, h = box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3]
        gx_m = gx[mask]
        gy_m = gy[mask]
        cx = (gx_m + dx) / S
        cy = (gy_m + dy) / S
        conf = obj_prob[b][mask] * cls_scores[b][mask]
        cls = cls_ids[b][mask].float()
        pred = torch.stack([cls, cx, cy, w, h, conf], dim=1)
        preds.append(pred)
    return preds


def iou_xywh(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for two sets of (cx, cy, w, h) boxes, returns (Na, Nb)."""
    a_x1 = boxes_a[:, 0] - boxes_a[:, 2] / 2
    a_y1 = boxes_a[:, 1] - boxes_a[:, 3] / 2
    a_x2 = boxes_a[:, 0] + boxes_a[:, 2] / 2
    a_y2 = boxes_a[:, 1] + boxes_a[:, 3] / 2
    b_x1 = boxes_b[:, 0] - boxes_b[:, 2] / 2
    b_y1 = boxes_b[:, 1] - boxes_b[:, 3] / 2
    b_x2 = boxes_b[:, 0] + boxes_b[:, 2] / 2
    b_y2 = boxes_b[:, 1] + boxes_b[:, 3] / 2

    x1 = torch.max(a_x1[:, None], b_x1[None, :])
    y1 = torch.max(a_y1[:, None], b_y1[None, :])
    x2 = torch.min(a_x2[:, None], b_x2[None, :])
    y2 = torch.min(a_y2[:, None], b_y2[None, :])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def nms_per_class(preds: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    """Greedy NMS applied independently per class on ``(N, 6)`` predictions.

    Columns: ``(cls, cx, cy, w, h, conf)``. Returns the kept rows.
    """
    if preds.numel() == 0:
        return preds
    keep = []
    for c in preds[:, 0].unique():
        idx = (preds[:, 0] == c).nonzero(as_tuple=False).flatten()
        p = preds[idx]
        order = torch.argsort(p[:, 5], descending=True)
        p = p[order]
        selected = []
        while p.shape[0]:
            top = p[:1]
            selected.append(top)
            if p.shape[0] == 1:
                break
            ious = iou_xywh(top[:, 1:5], p[1:, 1:5]).squeeze(0)
            p = p[1:][ious < iou_thresh]
        keep.append(torch.cat(selected, dim=0))
    return torch.cat(keep, dim=0)


def compute_map50(pred_lists: list[torch.Tensor], gt_lists: list[list[tuple]],
                  num_classes: int = NUM_CLASSES, iou_thresh: float = 0.5):
    """Proper per-class AP@0.5 averaged to mAP@0.5.

    ``pred_lists[i]`` is a ``(N, 6)`` tensor of predictions for image ``i`` —
    ``(cls, cx, cy, w, h, conf)``. ``gt_lists[i]`` is the list of GT tuples
    ``(cls, cx, cy, w, h)`` for image ``i``. Returns
    ``(mean_precision, mean_recall, map50, per_class_ap)``.
    """
    aps = []
    per_class_stats = []
    for c in range(num_classes):
        preds = []  # (img_idx, conf, cx, cy, w, h)
        for i, p in enumerate(pred_lists):
            if p.numel() == 0:
                continue
            mask = p[:, 0] == c
            if mask.sum() == 0:
                continue
            for row in p[mask]:
                preds.append((i, float(row[5]), row[1:5].tolist()))

        gt_per_img: dict[int, list[tuple]] = {}
        total_gt = 0
        for i, gts in enumerate(gt_lists):
            items = [(cx, cy, w, h) for (cc, cx, cy, w, h) in gts if cc == c]
            if items:
                gt_per_img[i] = items
                total_gt += len(items)

        if total_gt == 0 and not preds:
            continue
        if total_gt == 0:
            aps.append(0.0)
            per_class_stats.append((0.0, 0.0))
            continue
        if not preds:
            aps.append(0.0)
            per_class_stats.append((0.0, 0.0))
            continue

        preds.sort(key=lambda x: -x[1])
        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))
        matched = {i: [False] * len(gt_per_img.get(i, [])) for i in gt_per_img}

        for k, (img_i, _conf, box) in enumerate(preds):
            gts = gt_per_img.get(img_i, [])
            if not gts:
                fp[k] = 1
                continue
            bt = torch.tensor(box).unsqueeze(0)
            gt_t = torch.tensor(gts)
            ious = iou_xywh(bt, gt_t).squeeze(0)
            best = int(ious.argmax())
            if float(ious[best]) >= iou_thresh and not matched[img_i][best]:
                tp[k] = 1
                matched[img_i][best] = True
            else:
                fp[k] = 1

        cum_tp = torch.cumsum(tp, 0)
        cum_fp = torch.cumsum(fp, 0)
        recall = cum_tp / total_gt
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)

        # COCO 101-point interpolation of PR curve
        ap = 0.0
        for t in torch.linspace(0, 1, 101):
            mask = recall >= t
            p_at_t = float(precision[mask].max()) if mask.any() else 0.0
            ap += p_at_t / 101
        aps.append(ap)
        per_class_stats.append((float(precision[-1]), float(recall[-1])))

    map50 = sum(aps) / len(aps) if aps else 0.0
    mean_p = sum(p for p, _ in per_class_stats) / len(per_class_stats) if per_class_stats else 0.0
    mean_r = sum(r for _, r in per_class_stats) / len(per_class_stats) if per_class_stats else 0.0
    return mean_p, mean_r, map50


def evaluate(model: nn.Module, split: str, device: torch.device,
             conf_thresh: float = 0.05):
    """Evaluate a trained grid detector on a split and return (P, R, mAP@0.5)."""
    ds = WildlifeDataset(split)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model.eval()
    pred_lists: list[torch.Tensor] = []
    gt_lists: list[list[tuple]] = []
    with torch.no_grad():
        for imgs, boxes in loader:
            imgs = imgs.to(device)
            raw = model(imgs)
            preds = decode_predictions(raw, conf_thresh=conf_thresh)
            for i in range(imgs.shape[0]):
                p = preds[i]
                p = nms_per_class(p) if p.numel() else p
                pred_lists.append(p.cpu())
                gts = []
                for row in boxes[i]:
                    c = int(row[0].item())
                    if c < 0:
                        continue
                    gts.append((c, float(row[1]), float(row[2]),
                                float(row[3]), float(row[4])))
                gt_lists.append(gts)
    return compute_map50(pred_lists, gt_lists)


def train_variant(variant_name: str, epochs: int, lr: float = 1e-3,
                  augment: bool = False):
    """Fit one variant end-to-end and persist weights + return test metrics."""
    device = get_device()
    train_ds = WildlifeDataset("train", augment=augment)
    if len(train_ds) == 0:
        raise RuntimeError(
            "Training split is empty — did prepare_dataset.py run successfully?"
        )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    model = GridDetector().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        total_loss_components = {"obj": 0.0, "box": 0.0, "cls": 0.0}
        for imgs, boxes in train_loader:
            imgs = imgs.to(device)
            boxes = boxes.to(device)
            obj_t, box_t, cls_t = build_targets(boxes)

            raw = model(imgs)
            loss, parts = grid_loss(raw, obj_t, box_t, cls_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * imgs.size(0)
            for k, v in parts.items():
                total_loss_components[k] += v * imgs.size(0)
        sched.step()
        avg = total / len(train_ds)
        comps = {k: v / len(train_ds) for k, v in total_loss_components.items()}
        print(f"[{variant_name}] epoch {epoch + 1}/{epochs}  "
              f"loss={avg:.3f}  obj={comps['obj']:.3f}  "
              f"box={comps['box']:.3f}  cls={comps['cls']:.3f}")

    out_dir = RUNS_DIR / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best.pt")

    p, r, m = evaluate(model, "test", device)
    print(f"[{variant_name}] TEST  P={p:.3f}  R={r:.3f}  mAP@0.5={m:.3f}")
    return p, r, m


def main() -> int:
    """Train both variants (no-aug vs aug) and print a summary table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=== Custom grid detector: no augmentation ===")
    p0, r0, m0 = train_variant("no_aug", epochs=args.epochs, augment=False)
    random.seed(SEED)
    torch.manual_seed(SEED)
    print("\n=== Custom grid detector: with augmentation ===")
    p1, r1, m1 = train_variant("with_aug", epochs=args.epochs, augment=True)

    print("\n| Variant | P | R | mAP@0.5 |")
    print("|---|---|---|---|")
    print(f"| custom (no aug) | {p0:.3f} | {r0:.3f} | {m0:.3f} |")
    print(f"| custom (aug) | {p1:.3f} | {r1:.3f} | {m1:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
