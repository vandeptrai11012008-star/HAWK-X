# HAWK-X Person-Detection Fine-Tuning

**Status:** training code is available; **no HAWK-X-specific model has been trained or evaluated** in this repository because labeled footage has not been provided.

This is fine-tuning of **upstream Ultralytics YOLO11n pretrained weights**, not a new neural architecture or independently pretrained model. Observe upstream licensing requirements, including for derivative weights and redistributed software. The official upstream source is linked as the `vision-ai/` git submodule.

## 1. Install

From the HAWK-X root in VS Code PowerShell:

```powershell
git pull origin main
git submodule update --init --recursive
py -m pip install -e .\vision-ai
```

Use a Python version compatible with your installed PyTorch and Ultralytics. GPU training needs a supported GPU/PyTorch setup; CPU training works but is generally much slower.

## 2. Prepare a person-detection dataset

Use images/video frames you have permission to train on. For HAWK-X, prioritize daytime forest/tree cover scenes and include negative examples with **no person**. Use **different scenes or recording sessions** in train/val/test to avoid misleading scores from near-duplicate frames.

```text
HAWK-X/
  datasets/
    hawkx-person/
      images/
        train/     # images used to update weights
        val/       # images used to select hyperparameters/checkpoints
        test/      # held-out images, evaluated only at the end
      labels/
        train/     # one .txt per image with visible people
        val/
        test/
  training/
    train.py
    evaluate.py
```

A label file for a person image (example `labels/train/photo01.txt`) contains one row per visible person:

```text
0 0.500 0.500 0.250 0.600
```

The five fields mean `class_id x_center y_center width height`. Coordinates are **normalized between 0 and 1**, with `class_id=0` for `person`. Use an annotation tool that exports YOLO detection format. Images without people may have an empty or absent label file. The checker verifies that each training and validation split contains at least one person label; a realistic dataset needs considerably more examples. Avoid uploading private/identifiable training photos to a public GitHub repository.

## 3. Fine-tune from pretrained YOLO11n

```powershell
py training\train.py --dataset datasets\hawkx-person --epochs 30 --batch 8
```

Specify `--device cpu` to force CPU or `--device 0` when a compatible CUDA GPU is present. If training runs out of memory, reduce `--batch`, e.g. `--batch 4`. Training cannot start until the dataset directories and labels exist.

The training script checks the labels, creates a local absolute-path YOLO dataset configuration, loads `yolo11n.pt`, and fine-tunes it. A training run saves checkpoints under:

```text
training/runs/hawkx-person/weights/best.pt
```

Repeated runs may get an incremented name (e.g., `hawkx-person2`). Find the actual `best.pt` in the run's `weights/` folder.

## 4. Evaluate on held-out data

```powershell
py training\evaluate.py --dataset datasets\hawkx-person --model training\runs\hawkx-person\weights\best.pt
```

The script reports person-detection precision, recall and mAP. For a fair comparison, also evaluate the original `yolo11n.pt` on **the same held-out test set**. Tune on validation, not on test; retain failure examples and false positives for human review.

## 5. Predict on your own test photo

```powershell
yolo predict model="training/runs/hawkx-person/weights/best.pt" source="C:/path/to/approved/test.jpg" classes=0 save=True
```

These predictions are not independently validated for actual rescue decisions. This folder contains no drone operation or automatic tracking functionality.

## Notes

- `training/runs/`, `datasets/`, and `*.pt` are intentionally excluded from GitHub commits because they may be large and include private images or licensed weights.
- The `vision-ai/` submodule stays unchanged; new HAWK-X training scripts live outside it.
- Upstream YOLO11n's 80-class baseline was pretrained previously; domain fine-tuning on a one-class HAWK-X dataset adapts that model. It does not erase the model's provenance.
