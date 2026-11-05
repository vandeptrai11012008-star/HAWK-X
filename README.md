# HAWK-X — AI Search-and-Rescue Research Prototype

HAWK-X is an early-stage project exploring onboard perception, 3D mapping, and an operator-reviewed Rescue HQ dashboard. Its current repository is a research/demo prototype, **not** a validated rescue system or autonomous flight implementation.

## Architecture

```text
RGB camera                 LiDAR / IMU
    |                          |
vision-ai/                  mapping-3d/
(upstream source)                |
    |                           |
    +------ future integration -+
                 |
              core-hq/
       Human-operated Rescue HQ
```

## Vision AI — original upstream source

The entire `vision-ai/` directory is a **Git submodule** pointing to the official [Ultralytics source repository](https://github.com/ultralytics/ultralytics). No HAWK-X detection algorithm is represented as original work here. The repository is pinned to a specific upstream commit for reproducibility and can be updated deliberately later.

**Ultralytics code and YOLO pretrained models were created and trained by the upstream authors, not by HAWK-X.** They are subject to the upstream [AGPL-3.0 license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and any applicable model/commercial license terms. Keep upstream notices and honor applicable terms. Downloaded checkpoint files are not included in this repository.

## Download the full source

```powershell
git clone --recurse-submodules https://github.com/vandeptrai11012008-star/HAWK-X.git
cd HAWK-X
```

Already cloned without the submodule? From the repo root:

```powershell
git pull origin main
git submodule update --init --recursive
```

## Run pretrained YOLO11n on Windows

With Python and a supported PyTorch environment available, from the HAWK-X repo root:

```powershell
py -m pip install -e .\vision-ai
yolo predict model=yolo11n.pt source=0 classes=0 show=True
```

Alternatively use an image file: `yolo predict model=yolo11n.pt source="C:\path\to\photo.jpg" classes=0`.

The `yolo11n.pt` checkpoint is pretrained and is downloaded or loaded through Ultralytics. Running prediction does **not** mean HAWK-X trained the model. Do not use unverified detections as a basis for emergency decisions.

## Fine-tune HAWK-X on labeled person imagery

Training code for fine-tuning the pretrained model is available in [training/](training/README.md).
A properly licensed/consented, YOLO-labeled dataset is required; none is distributed in this repository.

```powershell
py training\train.py --dataset datasets\hawkx-person --epochs 30 --batch 8
```

The checkpoint produced by fine-tuning is derived from the upstream pretrained model; it is not an independently trained HAWK-X model. Measure performance on held-out data before drawing conclusions.

## Other project folders

- `mapping-3d/`: planned local point-cloud representations and example data.
- `core-hq/`: offline Rescue HQ browser demo, with synthetic events.
- `shared/`: example event payloads.
- `docs/`: architecture notes and validation plan.

To open the offline HQ demo, run `py -m http.server 8000` in the repository root and navigate to `http://localhost:8000/core-hq/`.

## Status and safety

The HQ demo and example detections remain synthetic. No live LiDAR integration, validated rescue detection, autonomous pursuit, or flight-control code is represented as complete. Future hardware work requires proper supervision, independent safety checks, consented or licensed test footage, reliable evaluation and operator review.

HAWK-X is a Code Metal FORGE Hackathon project concept. The submodule is third-party software, not original HAWK-X code.
