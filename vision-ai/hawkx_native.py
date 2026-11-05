"""HAWK-X native grid detector: an educational CNN trained from random initialization.

No pretrained detection model is loaded. This is an experimental offline baseline,
NOT a validated human detector or a flight-control component.
"""
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

IMAGE_SIZE = 128
GRID_SIZE = 8


class HAWKXDetector(nn.Module):
    """Single-class, one-box-per-cell detector (8 x 8 prediction grid)."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 16, 32, 64, 96]
        for i in range(4):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ])
        self.network = nn.Sequential(
            *layers,
            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 5, 1),
        )

    def forward(self, images):
        # [batch, 5, 8, 8] = objectness + cx_offset + cy_offset + w + h
        return self.network(images)


def image_tensor(image):
    import numpy as np
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(resized, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def targets_from_boxes(boxes):
    target = torch.zeros(5, GRID_SIZE, GRID_SIZE, dtype=torch.float32)
    for box in boxes:
        if len(box) != 4:
            raise ValueError("Each box must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = [float(v) for v in box]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Box coordinates must be normalized to [0, 1]")
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cell_x = min(int(cx * GRID_SIZE), GRID_SIZE - 1)
        cell_y = min(int(cy * GRID_SIZE), GRID_SIZE - 1)
        # If two people occupy the same grid cell, only one can be represented.
        if target[0, cell_y, cell_x] == 1:
            continue
        target[0, cell_y, cell_x] = 1.0
        target[1, cell_y, cell_x] = cx * GRID_SIZE - cell_x
        target[2, cell_y, cell_x] = cy * GRID_SIZE - cell_y
        target[3, cell_y, cell_x] = x2 - x1
        target[4, cell_y, cell_x] = y2 - y1
    return target


class PersonBoxDataset(Dataset):
    """JSON: {"images":[{"file":"images/a.jpg","boxes":[[x1,y1,x2,y2]]}]}."""

    def __init__(self, annotation_file):
        path = Path(annotation_file)
        document = json.loads(path.read_text(encoding="utf-8"))
        self.root = path.parent
        self.items = document["images"]
        if not self.items:
            raise ValueError(f"No images in {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(self.root / item["file"]) as image:
            pixels = image_tensor(image)
        return pixels, targets_from_boxes(item.get("boxes", []))


def detection_loss(logits, targets):
    objectness = nn.functional.binary_cross_entropy_with_logits(
        logits[:, :1], targets[:, :1],
        pos_weight=logits.new_tensor([8.0]),
    )
    positive = targets[:, :1].bool().expand_as(logits[:, 1:])
    if positive.any():
        box_loss = nn.functional.smooth_l1_loss(
            logits[:, 1:].sigmoid()[positive], targets[:, 1:][positive]
        )
    else:
        box_loss = logits.sum() * 0.0
    return objectness + 5.0 * box_loss


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


@torch.no_grad()
def decode_detections(logits, image_width, image_height, threshold=0.5, iou_threshold=0.4):
    """Convert a single image's [5, 8, 8] raw logits to pixel-space boxes."""
    probabilities = logits.sigmoid().cpu()
    candidates = []
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            confidence = float(probabilities[0, gy, gx])
            if confidence < threshold:
                continue
            cx = (gx + float(probabilities[1, gy, gx])) / GRID_SIZE
            cy = (gy + float(probabilities[2, gy, gx])) / GRID_SIZE
            w = float(probabilities[3, gy, gx])
            h = float(probabilities[4, gy, gx])
            bounds = [
                max(0.0, (cx - w / 2) * image_width),
                max(0.0, (cy - h / 2) * image_height),
                min(float(image_width), (cx + w / 2) * image_width),
                min(float(image_height), (cy + h / 2) * image_height),
            ]
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            candidates.append({"confidence": round(confidence, 4), "bbox_xyxy": [
                round(v, 1) for v in bounds
            ]})
    candidates.sort(key=lambda entry: entry["confidence"], reverse=True)
    accepted = []
    for candidate in candidates:
        if all(_iou(candidate["bbox_xyxy"], old["bbox_xyxy"]) < iou_threshold for old in accepted):
            accepted.append(candidate)
    return accepted
