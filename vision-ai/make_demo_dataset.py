"""Create synthetic shapes for a pipeline smoke test, NOT person-detection training."""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="vision-ai/demo-data")
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("samples must be >= 2")
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(23)
    items = []
    for i in range(args.samples):
        background = tuple(rng.randrange(50, 130) for _ in range(3))
        image = Image.new("RGB", (128, 128), background)
        draw = ImageDraw.Draw(image)
        boxes = []
        if i % 4 != 0:  # Some images have no foreground object.
            x, y = rng.randint(15, 72), rng.randint(10, 65)
            width, height = rng.randint(18, 35), rng.randint(28, 45)
            draw.rectangle((x, y, x + width, y + height),
                           fill=(225, 220, 200))
            boxes = [[x / 128, y / 128, (x + width) / 128, (y + height) / 128]]
        name = f"images/sample_{i:03}.png"
        image.save(out / name)
        items.append({"file": name, "boxes": boxes})
    (out / "annotations.json").write_text(
        json.dumps({"images": items}, indent=2), encoding="utf-8")
    print(f"Created {len(items)} synthetic shape examples in {out}")


if __name__ == "__main__":
    main()
