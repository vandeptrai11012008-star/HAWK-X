"""HAWK-X pretrained-person-detection adapter for local image/video analysis.

This script uses the third-party Ultralytics YOLO11n pretrained model.
The model and upstream library are NOT original HAWK-X code.
See vision-ai/README.md for the source and licensing information.

Usage:
    py vision-ai/pretrained_detect.py --source path/to/photo.jpg --save --json
    py vision-ai/pretrained_detect.py --source 0 --show
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="HAWK-X: pretrained person-detection baseline")
    parser.add_argument("--source", default="0", help="Image/video path or camera index (0 = webcam)")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model file/name")
    parser.add_argument("--conf", type=float, default=0.4, help="Minimum model confidence (0 to 1)")
    parser.add_argument("--show", action="store_true", help="Display live/image preview")
    parser.add_argument("--save", action="store_true", help="Save annotated frames for file inputs")
    parser.add_argument("--json", action="store_true", help="Save candidate detections for file inputs")
    parser.add_argument("--output", default="vision-ai/output/pretrained", help="Local output directory")
    args = parser.parse_args()
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be in [0, 1]")
    return args


def main():
    args = parse_args()
    camera = args.source.isdigit()
    source = int(args.source) if camera else args.source

    if camera and (args.save or args.json):
        raise SystemExit(
            "Use --show for webcam. --save/--json are restricted to files to avoid "
            "accumulating unbounded recordings."
        )
    if not camera and not Path(str(source)).exists():
        raise SystemExit(f"Input file not found: {source}")

    output_dir = Path(args.output)
    if args.save or args.json:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Trained upstream weights are retrieved automatically by Ultralytics
    # when the named model is not already present in the local cache.
    model = YOLO(args.model)
    results = model.predict(
        source=source, classes=[0], conf=args.conf,
        show=args.show, save=False, stream=True, verbose=False
    )

    for index, result in enumerate(results):
        candidates = []
        if result.boxes is not None:
            for box in result.boxes:
                candidates.append({
                    "class": "potential_person",
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                    "review_status": "unreviewed",
                })
        print(f"Frame {index}: {len(candidates)} candidate(s)")
        if camera:
            continue

        # Keep original imagery on the user's machine only.
        stem = f"{Path(str(result.path)).stem}_{index:05d}"
        if args.save:
            result.save(filename=str(output_dir / f"{stem}_annotated.jpg"))
        if args.json:
            report = {
                "schema_version": "0.1",
                "source": "third-party-pretrained-yolo11n",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "source_file": str(result.path),
                "review_status": "unreviewed",
                "location": {"valid": False, "frame_id": None, "xyz_m": None},
                "detections": candidates,
                "note": "Computer-vision candidates; not confirmed rescue events",
            }
            (output_dir / f"{stem}.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )


if __name__ == "__main__":
    main()
