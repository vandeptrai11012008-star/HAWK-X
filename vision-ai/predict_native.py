"""Run offline HAWK-X inference on an image and export a human-review report."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from hawkx_native import HAWKXDetector, decode_detections, image_tensor


def main():
    parser = argparse.ArgumentParser(description="HAWK-X native offline image inference")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="vision-ai/checkpoints/hawkx_native.pt")
    parser.add_argument("--output", default="vision-ai/output")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("threshold must be between 0 and 1")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "hawkx-grid-v0.1":
        raise ValueError("This checkpoint is not compatible with HAWK-X grid v0.1")
    model = HAWKXDetector()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with Image.open(args.image) as loaded:
        image = loaded.convert("RGB")
    with torch.no_grad():
        logits = model(image_tensor(image).unsqueeze(0))[0]
    detections = decode_detections(logits, image.width, image.height,
                                   threshold=args.threshold)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "0.1",
        "source": "hawkx-native-experimental",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_image": str(Path(args.image)),
        "review_status": "unreviewed",
        "location": {"valid": False, "frame_id": None, "xyz_m": None},
        "detections": [
            {"class": "potential_person", "confidence": d["confidence"],
             "bbox_xyxy": d["bbox_xyxy"], "review_status": "unreviewed"}
            for d in detections
        ],
        "note": "Experimental output; not verified for real-world rescue decisions",
    }
    (out / "detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    annotated = image.copy()
    drawer = ImageDraw.Draw(annotated)
    for detection in detections:
        box = detection["bbox_xyxy"]
        drawer.rectangle(box, outline="red", width=2)
        drawer.text((box[0], max(0, box[1] - 12)),
                    f"candidate {detection['confidence']:.2f}", fill="red")
    annotated.save(out / "preview.jpg")
    print(f"{len(detections)} candidates; files saved in {out}")


if __name__ == "__main__":
    main()
