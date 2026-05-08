# YOLOv11 Small Object Detection on xView

Replication of the **YOLOv11 portion** of:

> Yuan, X., Chakravarty, A., Lichtenberg, E. M., Gu, L., Wei, Z., & Chen, T.
> *An empirical analysis of deep learning methods for small object detection from satellite imagery.*

That paper compares six deep-learning detectors — **YOLOv11**, Faster R-CNN, SSD, Cascade R-CNN, Deformable DETR, and RT-DETR — on three high-resolution satellite datasets, and concludes that **YOLOv11 achieves the most balanced performance for localization and adaptability**. Anchor-based methods (SSD / Faster R-CNN / Cascade R-CNN) are sensitive to anchor box size; Deformable DETR struggles on dense small-object clusters; RT-DETR is competitive but training-intensive. Because YOLOv11 is the strongest method overall, this repository reproduces just the YOLOv11 experiments end-to-end on the **xView** dataset.

The whole pipeline lives in a single notebook — [`small_object_detection_v1.ipynb`](small_object_detection_v1.ipynb) — and runs on a local NVIDIA GPU:

- 512×512 image tiling of the xView training set
- 6-fold cross-validation, split at the **source-image level** (no leakage between train and val tiles)
- Training hyperparameters: SGD, lr = 1.01e-4, momentum = 0.89, weight-decay = 5e-4, horizontal-flip-only augmentation
- Four target vehicle classes: Small Car, Passenger Vehicle, Pickup Truck, Utility Truck
- Small-object filter: 10 px² ≤ object area ≤ 1000 px²

The notebook is **interruption-safe** by design: kernel restarts, machine reboots, and accidental cell re-runs all resume cleanly from the last saved epoch.

---

## Repository layout

```
.
├── small_object_detection_v1.ipynb   # the entire pipeline (config → train → eval → plots)
├── base_version.ipynb                # earlier reference version, kept for diffing
├── requirements.txt                  # Python dependencies
├── yolo11m.pt / yolo26n.pt           # local YOLO base weights (no internet needed at train time)
├── kaggle.json                       # your Kaggle API token (gitignored)
├── dataset/xview/                    # auto-populated from Kaggle on first run (gitignored)
└── outputs/                          # all run artefacts (gitignored)
    ├── checkpoints/yolov11_fold{N}/  # ultralytics run dir per fold (best.pt, last.pt, results.csv)
    ├── weights/fold{N}_best.pt       # tidy copy of each fold's best
    ├── tiles/                        # 512×512 PNG tiles + YOLO labels
    ├── yolo_dataset/fold_{N}/        # per-fold dataset.yaml + train.txt + val.txt
    ├── metrics/                      # per-fold CSV + aggregated JSON
    ├── plots/                        # paper-style figures + training curves
    └── logs/
```

---

## Requirements

- Python 3.10+
- An NVIDIA GPU (the notebook hard-aborts on CPU). The defaults are tuned for **8 GB GPU + 15 GB RAM**: `batch=4`, `workers=2`, AMP enabled.
- A Kaggle account with an API token

Install Python deps:

```bash
pip install -r requirements.txt
```

PyTorch with the right CUDA build is best installed manually — see the commented `pip install` line at the top of the notebook for an example (`cu124`).

---

## Dataset (auto-downloaded from Kaggle)

The notebook pulls the xView dataset from
[`hassanmojab/xview-dataset`](https://www.kaggle.com/datasets/hassanmojab/xview-dataset)
on first run and caches it under `./dataset/xview/`.
Subsequent runs detect the cached copy and skip the download.

To enable the download, place your Kaggle API token in **either** location:

- `./kaggle.json` (project root — the notebook copies it into `~/.kaggle/` for you), **or**
- `~/.kaggle/kaggle.json` (Kaggle's standard location)

Generate the token at <https://www.kaggle.com/settings> → **Create New API Token**.

Only the labelled training split is used. The public xView `val_images/` set has no annotations and so cannot be used for training or evaluation; instead the notebook builds its own train/val splits via 6-fold CV at the source-image level.

---

## Running

1. Drop `kaggle.json` into the project root (one-time).
2. Open `small_object_detection_v1.ipynb` and decide between:
   - `DEBUG_MODE = True` — single-epoch smoke test on fold 0, end-to-end in a few minutes
   - `DEBUG_MODE = False` — full 6-fold run with the training hyperparameters above
3. **Run all cells.** On first run the dataset downloads, gets tiled, and training starts. Re-running the notebook resumes from the last saved checkpoint of every fold.

### Re-run / resume cheat sheet

| Scenario | What happens on re-run |
|---|---|
| Kernel crashed mid-epoch | `last.pt` exists with optimizer state → continues at next epoch |
| You hit Ctrl-C | Same as above |
| You deleted `last.pt` only | Falls back to `best.pt` as a warm-start (epoch counter resets) |
| You deleted the whole fold directory | Trains fresh from the local YOLO weights |
| Run already converged (early stop) | Skips re-training, runs eval only |
| You deleted `dataset/xview/` | Re-downloads from Kaggle on the next run |

---

## Results

The notebook's aggregation cell loads every per-fold CSV in `outputs/metrics/` and prints the **mean ± std** for AP@50, AP@50–95, Precision, Recall, and F1 as folds finish, so partial 6-fold runs are reportable too. Aggregated metrics are also written to `outputs/metrics/aggregated_results.json`.

To compare against the paper's published numbers, see Yuan et al.'s tables for YOLOv11 on xView in the source paper.

---

## Citation

If you build on this replication, please cite the original paper:

```
Yuan, X., Chakravarty, A., Lichtenberg, E. M., Gu, L., Wei, Z., & Chen, T.
An empirical analysis of deep learning methods for small object detection
  from satellite imagery.
```
