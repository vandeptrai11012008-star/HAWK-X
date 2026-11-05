# HAWK-X Vision AI

This directory contains an early-stage **person-detection baseline** for evaluating test photos, video files and local webcam input. It does not implement validated rescue detection, autonomous navigation or flight control.

## Pretrained baseline — runs immediately

- `detect.py`: original simple webcam demo.
- `pretrained_detect.py`: enhanced CLI for image, video or webcam, with optional locally saved annotated images and JSON candidate reports.
- `requirements-pretrained.txt`: Python dependency for this baseline.

From the HAWK-X repository root in Windows PowerShell:

```powershell
py -m pip install -r vision-ai\requirements-pretrained.txt
py vision-ai\pretrained_detect.py --source 0 --show
```

To test an existing **local** image and save a preview plus a machine-readable report:

```powershell
py vision-ai\pretrained_detect.py --source C:\path\to\your\photo.jpg --save --json
```

For a test video, provide its file path with `--source`. Press `q` in the display window to stop preview if supported by your environment. Predictions are candidate detections needing human verification. The code neither uploads images to a server nor asserts geographic location.

When you use a file input, output is written to `vision-ai/output/pretrained/`, which is excluded from version control. For webcam input, use `--show`, without `--save` or `--json`.

## Independent HAWK-X research

A separate experimental model, independently implemented in PyTorch and initialized with random weights, is under development in the [native vision branch](https://github.com/vandeptrai11012008-star/HAWK-X/tree/feature/native-vision-pytorch/vision-ai). It is not pretrained or production-ready. Do not assume it can identify people until trained and independently evaluated.

## Source, authorship and licensing

**HAWK-X's integration code and HQ project are a separate project; the pretrained detector itself was not invented or trained by the HAWK-X developers.**

The pretrained baseline imports the [Ultralytics](https://github.com/ultralytics/ultralytics) package and loads `yolo11n.pt`, a YOLO11n pretrained checkpoint from that project. Upstream code is licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); models and commercial deployments also carry licensing requirements described by the provider. Keep applicable notices and comply with the relevant terms when using, distributing or modifying the software or weights. No upstream library source files or pretrained checkpoint are committed into this repository.

For future evaluations, use appropriately licensed or consented image data, document dataset provenance, report precision/recall and false alarms, and keep any actual flight stack independently safety-reviewed.
