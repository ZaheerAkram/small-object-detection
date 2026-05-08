# Small Object Detection on xView (YOLOv11)

**Author:** Zaheer Ud Din Akram &nbsp;·&nbsp; **Last updated:** May-2026

This repository replicates the **YOLOv11** experiments from:

> Yuan, X., Chakravarty, A., Lichtenberg, E. M., Gu, L., Wei, Z., & Chen, T.
> *An empirical analysis of deep learning methods for small object detection from satellite imagery.*
> **Expert Systems With Applications, 307 (2026) 131061.**

The paper compares six detectors (YOLOv11, Faster R-CNN, SSD, Cascade R-CNN, Deformable DETR, RT-DETR). YOLOv11 is the most balanced for localization and adaptability, so this repo focuses on reproducing its xView results.

The whole pipeline is one notebook: [`small_object_detection_v1.ipynb`](small_object_detection_v1.ipynb).

---

## What the notebook does

1. Auto-downloads the xView dataset from Kaggle into `./dataset/xview/` (idempotent — cached on re-runs).
2. Crops the labelled training images into 512×512 tiles with overlap; caches them under `outputs/tiles/`.
3. Filters annotations to four small-vehicle classes (paper Table 3): Small Car, Passenger Vehicle, Pickup Truck, Utility Truck. Object area is bounded to 10–1000 px².
4. Runs **6-fold cross-validation**, splitting at the source-image level so tiles never leak between train and val.
5. Trains YOLOv11 per fold, saves best/last checkpoints, and aggregates AP / AR / F1 across folds.
6. Writes plots, per-fold metrics, and a final summary into `outputs/`.

The notebook is **resume-safe**: kernel crashes, reboots, and re-runs continue from the last saved epoch.

---

## Paper hyperparameters (Table 5, YOLOv11 row)

| Parameter | Value |
| --- | --- |
| Optimizer | SGD |
| Learning rate | 1.01 × 10⁻⁴ |
| Batch size | 8 |
| Weight decay | 5 × 10⁻⁴ |
| Momentum | 0.89 |
| Iterations | 94,000 |
| Tile / image size | 512 × 512 |
| Augmentation | Horizontal flip only (p = 0.5) |
| Cross-validation | 6-fold |

> **Note:** the notebook defaults to `BATCH_SIZE = 4` and uses an epoch budget with early stopping (patience = 15) instead of a fixed iteration count. This is to fit an 8 GB GPU + 15 GB RAM machine. To match the paper exactly, set `BATCH_SIZE = 8` in cell 1.1 and run on a larger GPU.

---

## Repository layout

```text
.
├── small_object_detection_v1.ipynb   # the entire pipeline (config -> train -> eval -> plots)
├── base_version.ipynb                # earlier reference version, kept for diffing
├── requirements.txt                  # Python dependencies
├── yolo11m.pt / yolo26n.pt           # local YOLO base weights
├── kaggle.json                       # your Kaggle API token (gitignored)
├── dataset/xview/                    # auto-populated from Kaggle (gitignored)
└── outputs/                          # all run artefacts (gitignored)
    ├── checkpoints/yolov11_fold{N}/  # ultralytics run dir per fold
    ├── weights/fold{N}_best.pt       # tidy copy of each fold's best
    ├── tiles/                        # 512x512 tiles + YOLO labels
    ├── yolo_dataset/fold_{N}/        # per-fold dataset.yaml + train.txt + val.txt
    ├── metrics/                      # per-fold CSV + aggregated JSON
    ├── plots/                        # figures + training curves
    └── logs/
```

---

## Requirements

- Python 3.10+
- An **NVIDIA GPU** (the notebook hard-aborts on CPU). Defaults are tuned for 8 GB VRAM.
- A Kaggle account + API token

---

## Install

```bash
git clone <this-repo>
cd small-object-detection
pip install -r requirements.txt
```

PyTorch with the right CUDA build is best installed manually. There's a commented `pip install` line at the top of the notebook for `cu124` if you need a reference.

---

## Kaggle credentials (one-time)

Generate `kaggle.json` at <https://www.kaggle.com/settings> → **Create New API Token**, then place it in **either**:

- `./kaggle.json` (project root — the notebook copies it into `~/.kaggle/` for you), or
- `~/.kaggle/kaggle.json`

The notebook downloads `hassanmojab/xview-dataset` on first run and skips the download on subsequent runs.

---

## Run

1. Open `small_object_detection_v1.ipynb`.
2. In the config cell (1.1), choose:
   - `DEBUG_MODE = True` — single-epoch smoke test on fold 0 (a few minutes)
   - `DEBUG_MODE = False` — full 6-fold run
3. **Run all cells.** The first run downloads the dataset, builds the tile cache, and starts training. Subsequent runs resume from the last saved checkpoint per fold.

### Resume behaviour

| Scenario | What happens on re-run |
| --- | --- |
| Crashed mid-epoch / hit Ctrl-C | Loads `last.pt` (with optimizer state), continues at the next epoch |
| You deleted only `last.pt` | Warm-starts from `best.pt` |
| You deleted the whole fold dir | Trains fresh from the local YOLO weights |
| Run already early-stopped | Skips re-training, runs eval only |
| You deleted `dataset/xview/` | Re-downloads from Kaggle on next run |

---

## Results

After each fold finishes, `outputs/metrics/fold{N}_results.csv` is written. The aggregation cell loads all available per-fold CSVs and prints **mean ± std** for AP@50, AP@50–95, Precision, Recall, and F1, plus saves `outputs/metrics/aggregated_results.json`. Partial runs are reportable — you don't have to wait for all 6 folds.

To compare against published numbers, see the YOLOv11 rows in the paper's tables.

---

## Citation

```bibtex
@article{yuan2026empirical,
  author  = {Yuan, Xiaohui and Chakravarty, Aniv and Lichtenberg, Elinor M.
             and Gu, Lichuan and Wei, Zhenchun and Chen, Tian},
  title   = {An empirical analysis of deep learning methods for small object
             detection from satellite imagery},
  journal = {Expert Systems With Applications},
  volume  = {307},
  pages   = {131061},
  year    = {2026}
}
```
