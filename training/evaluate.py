"""Evaluate a fine-tuned HAWK-X detector on separately held-out images."""
import argparse

from train import ROOT, RUNS, check_dataset, write_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate HAWK-X checkpoint on held-out test images")
    parser.add_argument("--dataset", default="datasets/hawkx-person")
    parser.add_argument("--model", required=True, help="Path to trained best.pt")
    parser.add_argument("--device", default=None, help="cpu or 0, optional")
    args = parser.parse_args()
    try:
        dataset, counts = check_dataset(args.dataset, splits=("train", "val", "test"))
    except ValueError as exc:
        parser.error(str(exc))

    from ultralytics import YOLO

    options = {
        "data": str(write_config(dataset)),
        "split": "test",
        "project": str(RUNS),
        "name": "heldout-evaluation",
        "plots": True,
    }
    if args.device:
        options["device"] = args.device
    result = YOLO(args.model).val(**options)
    print("Held-out test evaluation:")
    print(f"mAP@0.50 = {result.box.map50:.4f}")
    print(f"mAP@0.50:0.95 = {result.box.map:.4f}")
    print(f"Precision = {result.box.mp:.4f}; Recall = {result.box.mr:.4f}")
    print("Compare these metrics with the original pretrained model on the same test set.")


if __name__ == "__main__":
    main()
