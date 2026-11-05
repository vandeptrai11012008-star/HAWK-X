# HAWK-X Native Vision (experimental)

This module contains an **independently implemented educational CNN** for offline, single-class object-detection experiments. Its network starts with random weights and uses PyTorch. It does not import third-party detector frameworks or use pretrained detector weights.

**Not a production model:** the 128-pixel input and 8x8 grid are deliberately small. Real-person detection accuracy, safety, latency and rescue suitability have **not** been measured. Multiple objects in one cell cannot be represented. Predictions are *candidates* requiring human review; there is no flight or pursuit integration.

## Components

- `hawkx_native.py`: small CNN, dataset parser, grid-box encoding, loss function, decoding and duplicate filtering.
- `train_native.py`: local image/annotation training; initializes network from scratch.
- `predict_native.py`: offline prediction to image preview + JSON event for review.
- `make_demo_dataset.py`: creates **synthetic rectangles only** to test that code executes. They are NOT people.
- `test_native.py`: unit tests for output dimensions, backpropagation and box handling.
- `detect.py`: legacy detector script on main; this branch replaces it with a native offline entry point.

## Quick start (from HAWK-X repository root)

Use a Python version for which the required libraries are available. Install dependencies:

```powershell
py -m pip install -r vision-ai\requirements-native.txt
```

Create demo shapes and train a smoke-test checkpoint:

```powershell
py vision-ai\make_demo_dataset.py
py vision-ai\train_native.py --data vision-ai\demo-data\annotations.json --epochs 5
py vision-ai\predict_native.py --image vision-ai\demo-data\images\sample_001.png
py -m unittest discover -s vision-ai -p "test_native.py"
```

Output files are `vision-ai/output/detections.json` and `vision-ai/output/preview.jpg`. The smoke-test checkpoint is **not** a person detector.

## Training on real images

Use properly licensed or consented training images. Create an annotations file such as:

```json
{
  "images": [
    {
      "file": "images/example.jpg",
      "boxes": [[0.15, 0.25, 0.55, 0.90]]
    }
  ]
}
```

Each box is normalized `[x1, y1, x2, y2]` in [0,1] coordinates. Empty `boxes` means no person. Put the annotation JSON beside the `images/` folder and use `--data path/to/annotations.json`. Split training and validation by video/scene rather than randomly splitting near-identical frames; report false positives and recall against held-out labels before claiming useful detection.

## Roadmap

1. Add real annotation validation and held-out precision/recall metrics.
2. Improve small-object handling, multi-scale detection, dataset augmentation, and device compatibility.
3. Add human-review event ingestion into the Rescue HQ demo.
4. Validate reproducibility and false-alert rates on authorized test footage.

## Authorship and provenance

HAWK-X Native Vision code in this folder is a new experimental implementation developed for HAWK-X. General detection concepts and PyTorch APIs are used; no external detection source code or pretrained weights are bundled. Dependencies and any future imported datasets or components remain subject to their individual licenses. Earlier Git history may contain a separate legacy experiment; this native branch does not rely on it.
