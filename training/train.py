"""Fine-tune pretrained YOLO11n for HAWK-X on a labeled local dataset.

This program uses the third-party Ultralytics library and pretrained weights.
It does not claim to train a detector from scratch. Run from the HAWK-X root:
    py training/train.py --dataset datasets/hawkx-person --epochs 30
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_dataset(dataset, splits=("train", "val")):
    """Validate a YOLO person-only dataset. No external dependencies."""
    dataset = Path(dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset folder not found: {dataset}")

    counts = {}
    for split in splits:
        image_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split
        if not image_dir.is_dir():
            raise ValueError(f"Missing image folder: {image_dir}")
        images = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No images found in: {image_dir}")

        persons = 0
        for image in images:
            relative = image.relative_to(image_dir)
            annotation = label_dir / relative.with_suffix(".txt")
            if not annotation.exists():
                continue  # In YOLO format, absence of a label file means empty image.
            for line_no, line in enumerate(annotation.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError(f"{annotation}:{line_no}: expected '0 x_center y_center width height'")
                try:
                    x, y, w, h = map(float, fields[1:])
                except ValueError as exc:
                    raise ValueError(f"{annotation}:{line_no}: invalid coordinates") from exc
                if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError(f"{annotation}:{line_no}: coordinates must be normalized")
                if not (0 <= x-w/2 <= x+w/2 <= 1 and 0 <= y-h/2 <= y+h/2 <= 1):
                    raise ValueError(f"{annotation}:{line_no}: bounding box outside image")
                persons += 1
        counts[split] = (len(images), persons)

    if counts["train"][1] == 0 or counts["val"][1] == 0:
        raise ValueError("Both train and val need some labeled people. Empty images are also useful.")
    return dataset, counts


def write_config(dataset):
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / "hawkx_person_data.yaml"
    # Absolute dataset root avoids ambiguity in Ultralytics datasets_dir settings.
    path.write_text(
        "path: " + json.dumps(dataset.as_posix()) + "\n"
        + "train: images/train\n"
        + "val: images/val\n"
        + "test: images/test\n"
        + "names:\n  0: person\n",
        encoding="utf-8",
    )
    return path


def main():
    parser = argparse.ArgumentParser(description="HAWK-X: fine-tune pretrained YOLO11n for person detection")
    parser.add_argument("--dataset", default="datasets/hawkx-person", help="YOLO-format dataset root")
    parser.add_argument("--model", default="yolo11n.pt", help="Upstream pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="cpu, 0 (first CUDA GPU), etc.")
    parser.add_argument("--name", default="hawkx-person")
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) < 1:
        parser.error("--epochs, --imgsz and --batch must be positive")

    try:
        dataset, counts = check_dataset(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))

    print("HAWK-X fine-tuning — Ultralytics YOLO11n pretrained baseline")
    for split, (images, persons) in counts.items():
        print(f"  {split}: {images} images, {persons} labeled person boxes")
    config = write_config(dataset)

    from ultralytics import YOLO  # Requires installation: py -m pip install -e ./vision-ai

    options = {
        "data": str(config),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": 0,  # Windows-friendly
        "project": str(RUNS),
        "name": args.name,
        "seed": 42,
        "patience": 10,
    }
    if args.device:
        options["device"] = args.device

    model = YOLO(args.model)
    model.train(**options)
    print("\nTraining finished. Find best.pt under training/runs/<run-name>/weights/.")
    print("Evaluate on a separately held-out test set before claiming accuracy.")


if __name__ == "__main__":
    main()
