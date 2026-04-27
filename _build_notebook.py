"""Generate small_object_detection_v1.ipynb for local GPU training.

Run with: python _build_notebook.py
This produces a clean, local-only YOLOv11 training notebook with
robust resume, incremental saves, and 8GB-GPU-friendly defaults.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "small_object_detection_v1.ipynb"

cells = []


def md(src: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    })


def code(src: str):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    })


# ──────────────────────────────────────────────────────────────────────
# 0. Header
# ──────────────────────────────────────────────────────────────────────
md(r"""# YOLOv11 Small Object Detection — xView (Paper Replication)
### Yuan et al. (2026), ESWA 307 · 6-Fold Cross-Validation
### Local GPU build — interruption-safe, RAM/GPU friendly

---

This notebook replicates the paper using a local NVIDIA GPU. It is built to
survive kernel restarts, machine reboots, and accidental cell re-runs:

| Capability | How it works |
|---|---|
| Resume from interruption | Loads `last.pt` (with optimizer state) and continues at the next epoch |
| Warm-start fallback | If `last.pt` is corrupt, falls back to `best.pt` as initial weights |
| Incremental checkpoints | `save_period=1` writes a checkpoint every epoch |
| Iterative metric saves | Per-fold results written as soon as a fold finishes |
| Tiling cache | Tile images and a manifest are flushed every 5 source images |
| Memory ceiling | 8 GB GPU + 15 GB RAM friendly defaults (`batch=4`, `workers=2`, AMP on) |

### Paper hyperparameters (Table 5)
| Param | Value |
|---|---|
| Optimizer | SGD |
| Learning rate | 1.01e-4 |
| Momentum | 0.89 |
| Weight decay | 5e-4 |
| Image size / tile | 512 × 512 |
| Augmentation | Horizontal flip only (p=0.5) |

### Paper results (Table 13 — xView, 6-fold CV, mean ± std)
| Metric | YOLOv11 |
|---|---|
| AP50 | 69.47 (±3.07) |
| AP50:95 | 29.58 (±1.40) |
| F1 | 73.97 (±1.81) |
""")

# ──────────────────────────────────────────────────────────────────────
# 1. Config section
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 1. Configuration

### Quick start
1. Place the xView dataset under `./dataset/xview/` (or set `LOCAL_DATASET_DIR`).
   The folder must contain a GeoJSON file and a `train_images/` directory.
2. Set `DEBUG_MODE = True` for a 1-epoch smoke test, or `False` for the full run.
3. Run all cells. Training auto-resumes from the last saved epoch on every re-run.
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  1.1  EDIT THESE VARIABLES
# ══════════════════════════════════════════════════════════════════

# ---- Mode ------------------------------------------------------------
DEBUG_MODE = False           # True = 1 epoch smoke test, False = full training

# ---- Local dataset ---------------------------------------------------
# Folder must contain the xView train images + the GeoJSON annotations.
# Auto-detection scans this directory (and all subdirectories) for a
# *.geojson file and an images folder (train_images/, images/, ...).
LOCAL_DATASET_DIR = "./dataset/xview"

# ---- Cross-validation (Paper: 6-fold) --------------------------------
NUM_FOLDS    = 6
CURRENT_FOLD = 0             # 0..5, or "all" to loop through every fold

# ---- Model (Paper Table 14: YOLOv11 ≈ 25.37M params) -----------------
YOLO_VARIANT = "yolo11m.pt"  # Closest public variant to paper scale (~20M).

# ---- Paper hyperparameters (Table 5) ---------------------------------
TILE_SIZE      = 512
TRAIN_RATIO    = 0.70
BATCH_SIZE     = 4           # Tuned for 8 GB GPU running yolo11m @ 512
LEARNING_RATE  = 1.01e-4
MOMENTUM       = 0.89
WEIGHT_DECAY   = 5e-4
IMG_SIZE       = 512
FLIP_PROB      = 0.5
CONF_THRESHOLD = 0.25
SEED           = 42

# ---- Small-object definition (Paper Section 3.1) ---------------------
MAX_OBJ_PX = 1000
MIN_OBJ_PX = 10

# ---- Behaviour -------------------------------------------------------
RESUME_TRAINING  = True      # Auto-resume from last.pt / best.pt if present
SKIP_TRAINING    = False     # True = only run the eval pass on existing weights
FORCE_RETILE     = False     # True = wipe the tile cache and rebuild

# ---- Training budget -------------------------------------------------
# 1000 epochs is an upper bound — early stopping (PATIENCE=15) will halt
# training as soon as the validation metric stops improving.
EPOCHS    = 1000
PATIENCE  = 15

# ---- Hardware / memory knobs (15 GB RAM, 8 GB GPU) -------------------
NUM_WORKERS   = 2            # Keep low; the OS already uses ~7-8 GB RAM
AMP           = True         # Mixed precision => smaller GPU footprint
CACHE_IMAGES  = False        # Never cache the dataset in RAM
SAVE_PERIOD   = 1            # Save a checkpoint every epoch (resume-friendly)

# CUDA allocator tweaks reduce fragmentation on small GPUs and let
# PyTorch grow allocations without re-allocating big slabs.
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                       "expandable_segments:True,max_split_size_mb:128")
_os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

# ══════════════════════════════════════════════════════════════════
#  DEBUG overrides (cheap smoke test for the whole pipeline)
# ══════════════════════════════════════════════════════════════════
if DEBUG_MODE:
    EPOCHS       = 1
    BATCH_SIZE   = 2
    YOLO_VARIANT = "yolo11n.pt"
    if isinstance(CURRENT_FOLD, str):
        CURRENT_FOLD = 0
    print("=" * 60)
    print("  DEBUG MODE: 1 epoch, yolo11n, fold 0 only")
    print("=" * 60)
else:
    print("=" * 60)
    print(f"  LOCAL TRAINING  Model: {YOLO_VARIANT}  Folds: {NUM_FOLDS}")
    print(f"  Fold(s) to run: {CURRENT_FOLD}")
    print(f"  Epochs (max)  : {EPOCHS}    Patience: {PATIENCE}")
    print("=" * 60)

print(f"  Model    : {YOLO_VARIANT}")
print(f"  Batch    : {BATCH_SIZE}  Workers: {NUM_WORKERS}  AMP: {AMP}")
print(f"  Resume   : {RESUME_TRAINING}  Save every: {SAVE_PERIOD} epoch(s)")
''')

# ──────────────────────────────────────────────────────────────────────
# 2. Environment setup
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 2. Environment Setup

Installs missing packages, configures torch / OpenCV thread counts so the
DataLoader doesn't oversubscribe CPUs, and probes the GPU.
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  2.1  Install + import + GPU probe + output directories
# ══════════════════════════════════════════════════════════════════
import os, sys, platform, subprocess, shutil, zipfile, time, gc, json, random, warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

print("[2.1] Setting up environment ...")

# Lightweight pip install only when a package is missing — keeps re-runs fast.
def _pip(import_name, package_name=None):
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package_name or import_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

for imp, pkg in [
    ("numpy", None), ("pandas", None), ("matplotlib", None), ("seaborn", None),
    ("cv2", "opencv-python-headless"), ("tqdm", None), ("sklearn", "scikit-learn"),
    ("yaml", "pyyaml"), ("torch", None), ("torchvision", None),
    ("ultralytics", None),
]:
    _pip(imp, pkg)

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")           # safe inside notebooks without a display
import matplotlib.pyplot as plt, seaborn as sns, cv2, yaml
import torch, torchvision
from tqdm.auto import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.figsize": (12, 6), "figure.dpi": 100, "font.size": 11})
sns.set_style("whitegrid")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Cap thread counts so the DataLoader workers don't fight the main thread
# for CPU on a memory-pressured machine.
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
except Exception:
    pass
torch.backends.cudnn.benchmark = True   # faster on fixed-size inputs (512x512)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ---- GPU probe -------------------------------------------------------
HW = {"device": "cpu", "name": "CPU", "count": 0, "mem_gb": 0}
if torch.cuda.is_available():
    HW.update(
        device="cuda",
        count=torch.cuda.device_count(),
        name=torch.cuda.get_device_name(0),
        mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    )
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    HW.update(device="mps", name="Apple MPS", count=1)

print(f"  Device   : {HW['device']}")
print(f"  GPU      : {HW['name']}  ({HW['mem_gb']} GB)")
print(f"  Torch    : {torch.__version__}  CUDA: {torch.version.cuda}")
if HW["device"] == "cpu":
    sys.stderr.write("\nERROR: No CUDA-capable GPU detected. This notebook requires a GPU to run.\n")
    sys.exit(1)

# Free any leftover GPU memory from prior cell runs.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ---- Output layout ---------------------------------------------------
# Everything lives under ./outputs/ so a single rmtree clears the run.
BASE_DIR = Path(".").resolve()
DATA_DIR = Path(LOCAL_DATASET_DIR).expanduser().resolve()
OUT_DIR  = BASE_DIR / "outputs"

P = {
    "DATA":    DATA_DIR,
    "OUTPUT":  OUT_DIR,
    "CKPT":    OUT_DIR / "checkpoints",   # YOLO writes per-fold runs here
    "WEIGHTS": OUT_DIR / "weights",       # final/best.pt copies per fold
    "PLOTS":   OUT_DIR / "plots",
    "METRICS": OUT_DIR / "metrics",
    "TILES":   OUT_DIR / "tiles",         # 512x512 PNG tiles + YOLO labels
    "YOLO_DS": OUT_DIR / "yolo_dataset",  # per-fold YOLO directory layout
    "LOGS":    OUT_DIR / "logs",
}
for k, d in P.items():
    if k != "DATA":
        d.mkdir(parents=True, exist_ok=True)

print(f"  Dataset  : {P['DATA']}")
print(f"  Output   : {P['OUTPUT']}")
print("[2.1] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 3. Dataset discovery
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 3. Dataset Loading (local)

The Kaggle xView dataset contains:
- `train_images/` + GeoJSON annotations → **labelled, we use this**
- `val_images/` → **NO public annotations** → cannot use for training/evaluation

We use **only the labelled training data**, tile it, then create our own
train/val splits via 6-fold CV at the source-image level (no data leakage).

### Channel handling
The raw xView imagery is 8-band WorldView-3, but the standard repackaging
stores 3-channel (RGB) TIFs that OpenCV reads directly — exactly what YOLO
expects. 8-band processing would require a custom backbone.
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  3.1  Locate the local xView dataset
# ══════════════════════════════════════════════════════════════════
print("[3.1] Locating local dataset ...")

def safe_list_dirs(root: Path):
    try:
        return [p for p in root.rglob("*") if p.is_dir()]
    except Exception:
        return []

def find_xview(search_root: Path):
    """Recursively look for an images folder + a GeoJSON / labels folder."""
    empty = {"valid": False, "geojson": None, "images_dir": None, "labels_dir": None}
    if not search_root.exists():
        return empty

    all_dirs = [search_root] + safe_list_dirs(search_root)

    # ---- GeoJSON: prefer files that look like xView train annotations ----
    geo = None
    for pattern in ["*train*.geojson", "*xview*.geojson", "*xView*.geojson",
                    "*.geojson", "*.json"]:
        for p in search_root.rglob(pattern):
            if p.is_file():
                geo = p
                break
        if geo is not None:
            break

    # ---- Image folder ----------------------------------------------------
    img = None
    preferred_img = {"train_images", "images", "train", "imgs", "image", "jpegimages"}
    for d in all_dirs:
        try:
            if d.name.lower() in preferred_img:
                if any(d.glob("*.tif")) or any(d.glob("*.png")) \
                   or any(d.glob("*.jpg")) or any(d.glob("*.jpeg")):
                    img = d
                    break
        except Exception:
            pass
    if img is None:                      # any folder containing image files
        for d in all_dirs:
            try:
                if any(d.glob("*.tif")) or any(d.glob("*.png")) \
                   or any(d.glob("*.jpg")) or any(d.glob("*.jpeg")):
                    img = d
                    break
            except Exception:
                pass

    # ---- Labels folder (txt / geojson) -----------------------------------
    lbl = None
    preferred_lbl = {"train_labels", "labels", "label", "annotations", "annots", "anno"}
    for d in all_dirs:
        try:
            if d.name.lower() in preferred_lbl:
                if any(d.glob("*.txt")) or any(d.glob("*.geojson")) or any(d.glob("*.json")):
                    lbl = d
                    break
        except Exception:
            pass
    if lbl is None:
        for d in all_dirs:
            try:
                if any(d.glob("*.txt")) or any(d.glob("*.geojson")) or any(d.glob("*.json")):
                    lbl = d
                    break
            except Exception:
                pass
    if geo is not None and lbl is None:
        lbl = geo.parent

    valid = (img is not None) and ((geo is not None) or (lbl is not None))
    return {"valid": valid, "geojson": geo, "images_dir": img, "labels_dir": lbl}


DS_INFO = find_xview(P["DATA"])
if not DS_INFO["valid"]:
    raise FileNotFoundError(
        f"\n  xView dataset not found inside: {P['DATA']}\n"
        f"  Place the dataset there (must include a GeoJSON file and an\n"
        f"  images folder such as 'train_images/'), or set LOCAL_DATASET_DIR\n"
        f"  in cell 1.1 to the correct path."
    )

IMAGES_DIR   = DS_INFO["images_dir"]
GEOJSON_PATH = DS_INFO["geojson"]
LABELS_DIR   = DS_INFO["labels_dir"]

# Decide which label format we have
if GEOJSON_PATH is not None:
    ANNO_TYPE = "geojson"
elif LABELS_DIR is not None and list(LABELS_DIR.glob("*.geojson")):
    GEOJSON_PATH = list(LABELS_DIR.glob("*.geojson"))[0]
    ANNO_TYPE = "geojson"
else:
    ANNO_TYPE = "txt"

print(f"  DATA_DIR  : {P['DATA']}")
print(f"  Images    : {IMAGES_DIR}")
print(f"  Labels    : {LABELS_DIR}")
print(f"  Anno type : {ANNO_TYPE}")
print(f"  GeoJSON   : {GEOJSON_PATH}")
print("[3.1] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 4. Annotation loading + class selection
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 4. Annotation Loading & Class Selection
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  4.1  Load annotations
# ══════════════════════════════════════════════════════════════════
print("[4.1] Loading annotations ...")
t0 = time.time()

if ANNO_TYPE == "geojson":
    with open(GEOJSON_PATH) as f:
        raw = json.load(f)
    print(f"  GeoJSON features: {len(raw['features']):,}")
    records = []
    for feat in tqdm(raw["features"], desc="  Parsing"):
        props = feat.get("properties", {})
        coords = props.get("bounds_imcoords", "")
        if not coords:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in coords.split(","))
        except Exception:
            continue
        w, h = abs(x2 - x1), abs(y2 - y1)
        records.append({
            "image_id": props.get("image_id", ""),
            "class_id": int(props.get("type_id", -1)),
            "px_x1": min(x1, x2), "px_y1": min(y1, y2),
            "px_x2": max(x1, x2), "px_y2": max(y1, y2),
            "px_w": w, "px_h": h, "px_area": w * h,
        })
    del raw                                   # release the parsed JSON
    bbox_df = pd.DataFrame(records)
else:
    raise NotImplementedError(f"TXT label loading not implemented (ANNO_TYPE={ANNO_TYPE})")

print(f"  Boxes: {len(bbox_df):,}  Images: {bbox_df['image_id'].nunique()}")
print(f"  Time : {time.time() - t0:.1f}s")
print("[4.1] Done.")
''')

code(r'''# ══════════════════════════════════════════════════════════════════
#  4.2  Target class selection (Paper Table 3)
# ══════════════════════════════════════════════════════════════════
print("[4.2] Selecting target classes ...")

TARGET_CLS = {18: "Small Car", 17: "Passenger Vehicle",
              20: "Pickup Truck", 21: "Utility Truck"}
CLS_TO_IDX = {18: 0, 17: 1, 20: 2, 21: 3}

bbox_df["is_target"] = bbox_df["class_id"].isin(TARGET_CLS)
small_df = bbox_df[
    bbox_df["is_target"]
    & (bbox_df["px_area"] >= MIN_OBJ_PX)
    & (bbox_df["px_area"] <= MAX_OBJ_PX)
].copy()

for cid, cn in TARGET_CLS.items():
    s = small_df[small_df["class_id"] == cid]
    if len(s):
        print(f"  {cn:22s} ID={cid:2d}  {len(s):>8,} objs  "
              f"{s['image_id'].nunique():>4} imgs  "
              f"area {s['px_area'].min():.0f}-{s['px_area'].max():.0f}")
print(f"  Total: {len(small_df):,} objects across {small_df['image_id'].nunique()} images")
print("[4.2] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 5. Figure 3
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 5. Size & Aspect-Ratio Analysis (Paper Figure 3)
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  5.1  Figure 3 + Table 3
# ══════════════════════════════════════════════════════════════════
print("[5.1] Plotting distributions ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
bins_s = np.logspace(0, np.log10(100_000), 20)
axes[0].hist(bbox_df["px_area"].clip(upper=100_000), bins=bins_s,
             alpha=.7, color="steelblue", edgecolor="w", label="All")
axes[0].hist(small_df["px_area"].clip(upper=100_000), bins=bins_s,
             alpha=.7, color="indianred", edgecolor="w", label="Target")
axes[0].axvline(1000, color="red", ls="--", alpha=.6)
axes[0].set_xscale("log")
axes[0].set_xlabel("Area (px)")
axes[0].set_title("(a) Size Distribution")
axes[0].legend()

short = np.minimum(small_df["px_w"], small_df["px_h"])
long_ = np.maximum(small_df["px_w"], small_df["px_h"])
ar = (short / long_.replace(0, np.nan)).dropna()
axes[1].hist(ar, bins=np.arange(0, 1.05, .1), color="indianred", edgecolor="w")
axes[1].set_xlabel("Aspect Ratio")
axes[1].set_title("(b) Aspect Ratio")

plt.tight_layout()
plt.savefig(P["PLOTS"] / "fig3.png", dpi=150, bbox_inches="tight")
plt.show()

# Table 3
rows = []
for cid, cn in TARGET_CLS.items():
    s = small_df[small_df["class_id"] == cid]
    if len(s):
        rows.append({
            "Class": cn,
            "Images": s["image_id"].nunique(),
            "Objects": len(s),
            "Min": int(s["px_area"].min()),
            "Median": int(s["px_area"].median()),
            "Max": int(s["px_area"].max()),
        })
t3 = pd.DataFrame(rows)
print(t3.to_string(index=False))
t3.to_csv(P["METRICS"] / "table3.csv", index=False)
print("[5.1] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 6. Tiling
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 6. Image Tiling (Paper Section 4.1: 512×512)

Tiles are written to disk and a CSV manifest is flushed every 5 source
images, so an interrupted tiling run resumes seamlessly. Set
`FORCE_RETILE = True` in cell 1.1 to wipe the cache and rebuild.
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  6.1  Tiling pipeline (RAM-friendly, iterative save)
# ══════════════════════════════════════════════════════════════════
print("[6.1] Tiling ...")
t0 = time.time()

def tile_pos(h, w, ts=512):
    """Return (y, x) origins covering an image with ~uniform overlap."""
    nr, nc = int(np.ceil(h / ts)), int(np.ceil(w / ts))
    out = []
    for r in range(nr):
        for c in range(nc):
            y = r * (h - ts) // max(nr - 1, 1) if nr > 1 else 0
            x = c * (w - ts) // max(nc - 1, 1) if nc > 1 else 0
            out.append((min(y, max(0, h - ts)), min(x, max(0, w - ts))))
    return out

def clip_box(bx, ty, tx, ts, mf=0.2):
    """Clip a box to the tile, dropping it if <20% of its area survives."""
    x1 = max(bx["px_x1"] - tx, 0)
    y1 = max(bx["px_y1"] - ty, 0)
    x2 = min(bx["px_x2"] - tx, ts)
    y2 = min(bx["px_y2"] - ty, ts)
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        return None
    if bx["px_area"] > 0 and cw * ch / bx["px_area"] < mf:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": cw, "h": ch, "cid": bx["class_id"]}


img_out = P["TILES"] / "images"
lbl_out = P["TILES"] / "labels"
img_out.mkdir(parents=True, exist_ok=True)
lbl_out.mkdir(parents=True, exist_ok=True)
mf_csv = P["TILES"] / "manifest.csv"
done_src_txt = P["TILES"] / "_done_src.txt"

# Optional: nuke the cache and rebuild from scratch.
if FORCE_RETILE:
    print("  FORCE_RETILE=True — clearing tile cache ...")
    shutil.rmtree(P["TILES"], ignore_errors=True)
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

# Resume-friendly: load the previous manifest (if any) so we skip
# already-tiled source images.
existing_records = []
done_srcs = set()
if mf_csv.exists():
    try:
        prev = pd.read_csv(mf_csv)
        existing_records = prev.to_dict("records")
        done_srcs = set(prev["src"].unique().tolist())
        print(f"  Cached manifest: {len(prev)} tiles from {len(done_srcs)} source images")
    except Exception:
        existing_records, done_srcs = [], set()
if done_src_txt.exists():
    try:
        for line in done_src_txt.read_text().splitlines():
            line = line.strip()
            if line:
                done_srcs.add(line)
    except Exception:
        pass

target_imgs = list(small_df["image_id"].unique())
if DEBUG_MODE:
    target_imgs = target_imgs[:30]
todo_imgs = [i for i in target_imgs if i not in done_srcs]

if not todo_imgs:
    print(f"  All {len(target_imgs)} source images already tiled — using cache")
else:
    print(f"  Tiling {len(todo_imgs)} / {len(target_imgs)} images (skipping {len(done_srcs)} cached) ...")
    # Group boxes by image once — avoids re-filtering the DataFrame per tile.
    boxes_by_img = {iid: g for iid, g in small_df.groupby("image_id")}
    new_records = []
    FLUSH_EVERY = 5             # persist manifest + done list every 5 source images
    processed = 0

    for img_id in tqdm(todo_imgs, desc="  Tiling"):
        ip = IMAGES_DIR / str(img_id)
        if not ip.exists():                  # try common extensions
            for ext in [".tif", ".tiff", ".png"]:
                c = IMAGES_DIR / (str(img_id).split(".")[0] + ext)
                if c.exists():
                    ip = c
                    break
        if not ip.exists():
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        iboxes = boxes_by_img.get(img_id)
        if iboxes is None or len(iboxes) == 0:
            del img
            continue

        for idx, (ty, tx) in enumerate(tile_pos(h, w, TILE_SIZE)):
            tile = img[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
            if tile.shape[0] < TILE_SIZE or tile.shape[1] < TILE_SIZE:
                pad = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                pad[:tile.shape[0], :tile.shape[1]] = tile
                tile = pad
            tboxes = []
            for _, bx in iboxes.iterrows():
                cb = clip_box(bx, ty, tx, TILE_SIZE)
                if cb:
                    tboxes.append(cb)
            if not tboxes:
                continue
            tn = f"{Path(img_id).stem}_t{idx:04d}"
            cv2.imwrite(str(img_out / f"{tn}.png"), tile)
            lines = []
            for b in tboxes:
                cx = (b["x1"] + b["x2"]) / 2 / TILE_SIZE
                cy = (b["y1"] + b["y2"]) / 2 / TILE_SIZE
                bw = b["w"] / TILE_SIZE
                bh = b["h"] / TILE_SIZE
                lines.append(f"{CLS_TO_IDX.get(b['cid'], 0)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_out / f"{tn}.txt").write_text("\n".join(lines))
            new_records.append({"tile": tn, "src": img_id, "n_obj": len(tboxes)})

        del img                              # free image memory before next iter
        done_srcs.add(img_id)
        processed += 1

        # Iterative save — survives kernel interruption mid-tiling.
        if processed % FLUSH_EVERY == 0:
            combined = pd.DataFrame(existing_records + new_records)
            combined.drop_duplicates(subset=["tile"], keep="last", inplace=True)
            combined.to_csv(mf_csv, index=False)
            with open(done_src_txt, "w") as f:
                f.write("\n".join(sorted(done_srcs)))
            gc.collect()

    # Final flush.
    combined = pd.DataFrame(existing_records + new_records)
    combined.drop_duplicates(subset=["tile"], keep="last", inplace=True)
    combined.to_csv(mf_csv, index=False)
    with open(done_src_txt, "w") as f:
        f.write("\n".join(sorted(done_srcs)))
    del boxes_by_img
    gc.collect()

manifest = pd.read_csv(mf_csv)
print(f"  Tiles: {len(manifest):,}  Objects: {manifest['n_obj'].sum():,}")
print(f"  Time : {time.time() - t0:.1f}s")
print("[6.1] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 7. Cross validation split
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 7. 6-Fold Cross-Validation Split

### Paper Section 4.1
> "We conducted 6-fold cross-validation and report the average performance."

The split is done at the **source-image level** — every tile from one raw
image goes into the same fold, which prevents data leakage between train
and val.
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  7.1  Generate 6-fold split (deterministic; persisted to disk)
# ══════════════════════════════════════════════════════════════════
print("[7.1] Generating 6-fold CV splits ...")
from sklearn.model_selection import KFold

src_images = manifest["src"].unique()
print(f"  Source images: {len(src_images)}")
print(f"  Total tiles  : {len(manifest)}")

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

fold_splits = {}
for fold_idx, (train_src_idx, val_src_idx) in enumerate(kf.split(src_images)):
    train_srcs = set(src_images[train_src_idx])
    val_srcs   = set(src_images[val_src_idx])

    train_tiles = manifest[manifest["src"].isin(train_srcs)]["tile"].tolist()
    val_tiles   = manifest[manifest["src"].isin(val_srcs)]["tile"].tolist()
    train_objs  = manifest[manifest["src"].isin(train_srcs)]["n_obj"].sum()
    val_objs    = manifest[manifest["src"].isin(val_srcs)]["n_obj"].sum()

    fold_splits[fold_idx] = {"train": train_tiles, "val": val_tiles}

    print(f"  Fold {fold_idx}: train={len(train_tiles):>5} tiles ({train_objs:>7,} objs)"
          f"  val={len(val_tiles):>5} tiles ({val_objs:>6,} objs)"
          f"  [src: {len(train_srcs)} / {len(val_srcs)}]")

# Save fold metadata so a later run can sanity-check the split is identical.
fold_meta = {f"fold_{i}": {"train": len(s["train"]), "val": len(s["val"])}
             for i, s in fold_splits.items()}
fold_meta["num_folds"] = NUM_FOLDS
fold_meta["seed"] = SEED
with open(P["METRICS"] / "fold_splits.json", "w") as f:
    json.dump(fold_meta, f, indent=2)

print(f"\n  Fold metadata saved to {P['METRICS'] / 'fold_splits.json'}")
print("[7.1] Done.")
''')

code(r'''# ══════════════════════════════════════════════════════════════════
#  7.2  Training budget summary
# ══════════════════════════════════════════════════════════════════
print("[7.2] Training budget ...")

n_train_tiles   = len(fold_splits[0]["train"])
iters_per_epoch = int(np.ceil(n_train_tiles / BATCH_SIZE))

print(f"  Max epochs      : {EPOCHS}    (early stopping kicks in much earlier)")
print(f"  Patience        : {PATIENCE}")
print(f"  Train tiles     : {n_train_tiles:,}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  Iters / epoch   : {iters_per_epoch:,}")
print("[7.2] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 8. Training (the big one)
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 8. YOLO Dataset Setup & Training

For each fold:
1. Build a YOLO-style directory layout (writes `train.txt` / `val.txt` lists
   pointing at the cached tiles — no copying, saves disk space).
2. Write `dataset.yaml`.
3. Train YOLOv11 with the paper hyperparameters and `save_period=1` so a
   checkpoint exists at every epoch.
4. **Resume logic**: if a previous run was interrupted, the training loop
   loads `last.pt` (with optimizer state) and picks up at the next epoch.
   If `last.pt` is corrupt, it warm-starts from `best.pt`. Only when both
   are missing does it train from the pretrained YOLO weights.
5. Evaluate the best checkpoint and write the per-fold metrics CSV.
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  8.1  Setup + Train + Evaluate per fold
#
#  Resume strategy:
#    - last.pt with optimizer state -> exact resume (continues at next epoch)
#    - best.pt only / last.pt corrupt -> warm-start from those weights
#    - neither present                -> fresh training from YOLO_VARIANT
#
#  Every checkpoint, every per-fold metric, and every YOLO log is written to
#  ./outputs/, so an interrupted run loses at most the in-flight epoch.
# ══════════════════════════════════════════════════════════════════
print("[8.1] Starting fold training loop ...")


def setup_fold_dataset(fold_idx, fold_data, paths):
    """Create the YOLO directory layout for one fold using path lists.

    Instead of copying tiles into per-fold folders (slow on Windows, no
    symlink permission needed), we write `train.txt` / `val.txt` files
    that contain absolute paths to the cached tile images. Ultralytics
    reads these natively. The label files sit next to each image's
    location string by replacing /images/ with /labels/ — but YOLO also
    accepts a parallel labels directory, so we keep the standard layout
    and point the YAML at it.
    """
    fold_dir = paths["YOLO_DS"] / f"fold_{fold_idx}"
    yaml_p   = fold_dir / "dataset.yaml"
    fold_dir.mkdir(parents=True, exist_ok=True)

    si = paths["TILES"] / "images"
    sl = paths["TILES"] / "labels"

    # Images directory: a single shared folder for all folds, since
    # tile names are unique. We just write per-fold path lists pointing
    # into it. This avoids duplicating ~tens of thousands of PNGs per fold.
    train_list = fold_dir / "train.txt"
    val_list   = fold_dir / "val.txt"

    def write_list(path: Path, tile_names):
        with open(path, "w") as f:
            for tn in tile_names:
                ip = si / f"{tn}.png"
                if ip.exists():
                    f.write(str(ip.resolve()) + "\n")

    write_list(train_list, fold_data["train"])
    write_list(val_list,   fold_data["val"])

    # The labels directory is shared too; YOLO finds <name>.txt next to
    # each <name>.png by replacing /images/ with /labels/, so this works
    # as long as both `si` and `sl` are siblings under TILES/.
    yd = {
        "path":  str(paths["TILES"].resolve()),
        "train": str(train_list.resolve()),
        "val":   str(val_list.resolve()),
        "nc":    4,
        "names": {0: "Small_Car", 1: "Passenger_Vehicle",
                  2: "Pickup_Truck", 3: "Utility_Truck"},
    }
    with open(yaml_p, "w") as f:
        yaml.dump(yd, f, default_flow_style=False)

    n_tr = sum(1 for _ in open(train_list))
    n_vl = sum(1 for _ in open(val_list))
    print(f"    Fold {fold_idx}: {n_tr} train, {n_vl} val images (path-list mode)")
    return yaml_p


def _inspect_ckpt(pt_path: Path):
    """Return (epoch, has_optimizer) for a YOLO checkpoint, or (None, False).

    `epoch` is the epoch index of the LAST COMPLETED epoch. Ultralytics
    sets it to -1 once the run finishes (so the next run can't resume past
    completion).
    """
    try:
        ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        ep = ck.get("epoch", None)
        has_opt = ck.get("optimizer", None) is not None
        del ck
        gc.collect()
        return ep, has_opt
    except Exception as e:
        print(f"    Could not inspect {pt_path.name}: {e}")
        return None, False


def find_resume_source(exp_weights: Path):
    """Decide how to resume training for a fold.

    Returns (path, mode) where mode is one of:
      "resume" : exact resume with optimizer state + epoch counter
      "warm"   : load weights only and start training from epoch 0
      None     : nothing to resume from — train fresh from YOLO_VARIANT
    """
    last_pt = exp_weights / "last.pt"
    best_pt = exp_weights / "best.pt"

    # 1) Exact resume needs last.pt with optimizer state and a sane epoch.
    if last_pt.exists() and last_pt.stat().st_size > 0:
        ep, has_opt = _inspect_ckpt(last_pt)
        if has_opt and isinstance(ep, int) and ep >= 0:
            return last_pt, "resume"
        print(f"    last.pt present but not resumable (epoch={ep}, has_opt={has_opt})")

    # 2) Fall back to best.pt as warm-start weights.
    if best_pt.exists() and best_pt.stat().st_size > 0:
        return best_pt, "warm"

    # 3) Last resort: last.pt without optimizer (e.g. stripped after run finished).
    if last_pt.exists() and last_pt.stat().st_size > 0:
        return last_pt, "warm"

    return None, None


def train_fold(fold_idx, yaml_path, paths):
    """Train YOLOv11 for one fold with resume + memory-aware settings."""
    exp_name    = f"yolov11_fold{fold_idx}"
    exp_dir     = paths["CKPT"] / exp_name
    exp_weights = exp_dir / "weights"

    resume_src, mode = (None, None)
    if RESUME_TRAINING:
        resume_src, mode = find_resume_source(exp_weights)

    # All training arguments in one place. `resume=True` ignores most of
    # these and re-reads the saved args.yaml — which is what we want for
    # an exact resume.
    args = dict(
        data=str(yaml_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        lr0=LEARNING_RATE, lrf=0.01,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
        # Paper uses horizontal flip only.
        flipud=0., fliplr=FLIP_PROB,
        mosaic=0., mixup=0., scale=0.,
        hsv_h=0., hsv_s=0., hsv_v=0.,
        optimizer="SGD", seed=SEED + fold_idx, deterministic=True,
        workers=NUM_WORKERS,
        patience=PATIENCE,
        project=str(paths["CKPT"]), name=exp_name, exist_ok=True,
        save=True, save_period=SAVE_PERIOD,
        plots=True, verbose=True,
        cache=CACHE_IMAGES,
        amp=AMP,
        rect=False,
        close_mosaic=0,
    )

    # Sanity: make sure we haven't accidentally aimed at a default COCO yaml.
    assert "coco" not in args["data"].lower(), f"BUG: data={args['data']}"

    # Free GPU memory left over from earlier cells.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if mode == "resume":
        print(f"    RESUMING fold {fold_idx} from {resume_src.name}")
        ep, _ = _inspect_ckpt(resume_src)
        print(f"    Last completed epoch: {ep}  -> continuing at epoch {ep + 1}")
        model = YOLO(str(resume_src))
        # `resume=True` makes ultralytics restore optimizer + epoch counter.
        model.train(resume=True)
    elif mode == "warm":
        print(f"    WARM-START fold {fold_idx} from {resume_src.name} (no optimizer state)")
        model = YOLO(str(resume_src))
        model.train(**args)
    else:
        print(f"    FRESH training fold {fold_idx} from {YOLO_VARIANT}")
        model = YOLO(YOLO_VARIANT)
        model.train(**args)

    # Verify training actually used our dataset (guards against a stale
    # args.yaml that pointed at COCO somehow).
    if (exp_dir / "args.yaml").exists():
        with open(exp_dir / "args.yaml") as f:
            used = yaml.safe_load(f)
        if "coco" in str(used.get("data", "")).lower():
            print(f"    ERROR: Fold {fold_idx} trained on COCO, not xView!")
            return None

    # Copy the final best.pt into a tidy output dir (independent of the
    # ultralytics run folder).
    best_src = exp_weights / "best.pt"
    if best_src.exists():
        shutil.copy2(best_src, paths["WEIGHTS"] / f"fold{fold_idx}_best.pt")

    # Free GPU between folds.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def evaluate_fold(fold_idx, yaml_path, paths):
    """Evaluate one fold and return a metrics dict."""
    best_pt = paths["CKPT"] / f"yolov11_fold{fold_idx}" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"    No best.pt for fold {fold_idx}")
        return None

    model = YOLO(str(best_pt))
    r = model.val(
        data=str(yaml_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
        split="val", plots=True, verbose=False,
        batch=BATCH_SIZE, workers=NUM_WORKERS, half=AMP,
    )

    met = {
        "fold":      fold_idx,
        "AP50":      round(float(r.box.map50) * 100, 2),
        "AP50_95":   round(float(r.box.map)   * 100, 2),
        "Precision": round(float(r.box.mp)    * 100, 2),
        "Recall":    round(float(r.box.mr)    * 100, 2),
    }
    p, rc = met["Precision"] / 100, met["Recall"] / 100
    met["F1"] = round(2 * p * rc / (p + rc) * 100, 2) if (p + rc) > 0 else 0.0

    print(f"    Fold {fold_idx}: AP50={met['AP50']:.2f}  AP50:95={met['AP50_95']:.2f}  "
          f"F1={met['F1']:.2f}")

    del model, r
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return met


# ══════════════════════════════════════════════════════════════════
#  Determine which folds to run, then loop
# ══════════════════════════════════════════════════════════════════
if isinstance(CURRENT_FOLD, int):
    folds_to_run = [CURRENT_FOLD]
elif CURRENT_FOLD == "all":
    folds_to_run = list(range(NUM_FOLDS))
else:
    folds_to_run = [0]

print(f"  Folds to run : {folds_to_run}")
print(f"  Max epochs   : {EPOCHS}    Patience: {PATIENCE}")
print(f"  Batch / LR   : {BATCH_SIZE} / {LEARNING_RATE}")
print(f"  Model        : {YOLO_VARIANT}")
print(f"  Resume       : {RESUME_TRAINING}    Save every: {SAVE_PERIOD} epoch(s)")

all_metrics = []

for fi in folds_to_run:
    print(f"\n{'=' * 60}")
    print(f"  FOLD {fi}/{NUM_FOLDS - 1}")
    print(f"{'=' * 60}")

    # Setup
    yaml_p = setup_fold_dataset(fi, fold_splits[fi], P)

    # Train
    if not SKIP_TRAINING:
        t0 = time.time()
        try:
            train_fold(fi, yaml_p, P)
        except KeyboardInterrupt:
            # User hit stop — checkpoints are already on disk thanks to
            # save_period=1, so the next run will pick up automatically.
            print(f"\n    Training interrupted by user. last.pt kept on disk for resume.")
            raise
        print(f"    Training time: {(time.time() - t0) / 60:.1f} min")

    # Evaluate (always runs, even if SKIP_TRAINING=True, as long as best.pt exists)
    met = evaluate_fold(fi, yaml_p, P)
    if met:
        all_metrics.append(met)
        # Iterative save — one CSV per fold, written immediately.
        pd.DataFrame([met]).to_csv(
            P["METRICS"] / f"fold{fi}_results.csv", index=False)

print(f"\n  Completed {len(all_metrics)} fold(s)")
print("[8.1] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 9. Aggregate
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 9. Results Aggregation (mean ± std)

Loads every per-fold CSV from `outputs/metrics/` so partial runs are
reportable too — no need to wait for all 6 folds to finish.
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  9.1  Aggregate fold results
# ══════════════════════════════════════════════════════════════════
print("[9.1] Aggregating results ...")

# Pick up every per-fold CSV that has been written so far.
saved_results = []
for fi in range(NUM_FOLDS):
    fp = P["METRICS"] / f"fold{fi}_results.csv"
    if fp.exists():
        saved_results.append(pd.read_csv(fp).iloc[0].to_dict())

if saved_results:
    results_df = pd.DataFrame(saved_results)
    n_folds_done = len(results_df)
    print(f"  Results from {n_folds_done}/{NUM_FOLDS} folds:\n")
    print(results_df.to_string(index=False))

    metrics = ["AP50", "AP50_95", "Precision", "Recall", "F1"]
    print(f"\n  {'Metric':12s}  {'Mean':>8s}  {'Std':>7s}  Paper (mean±std)")
    print("  " + "-" * 55)
    paper = {"AP50": (69.47, 3.07), "AP50_95": (29.58, 1.40), "F1": (73.97, 1.81)}
    for m in metrics:
        if m in results_df.columns:
            mean_v = results_df[m].mean()
            std_v  = results_df[m].std() if n_folds_done > 1 else 0
            p_str = ""
            if m in paper:
                pm, ps = paper[m]
                p_str = f"{pm:.2f} ± {ps:.2f}"
            print(f"  {m:12s}  {mean_v:8.2f}  {std_v:7.2f}  {p_str}")

    agg = {m: {"mean": float(results_df[m].mean()),
               "std":  float(results_df[m].std()) if n_folds_done > 1 else 0.0}
           for m in metrics if m in results_df.columns}
    agg["n_folds"] = n_folds_done
    with open(P["METRICS"] / "aggregated_results.json", "w") as f:
        json.dump(agg, f, indent=2)
    results_df.to_csv(P["METRICS"] / "all_folds_results.csv", index=False)
    print(f"\n  Saved: aggregated_results.json, all_folds_results.csv")

    if n_folds_done < NUM_FOLDS:
        print(f"\n  NOTE: {n_folds_done}/{NUM_FOLDS} folds done.")
        print(f"  Set CURRENT_FOLD = {n_folds_done} (or 'all') in cell 1.1 and re-run.")

    if DEBUG_MODE:
        print("\n  DEBUG results — not paper-comparable.")
else:
    print("  No fold results found yet. Train at least one fold first.")

print("[9.1] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 10. Visualization
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 10. Visualization
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  10.1  Prediction overlays from the most recently run fold
# ══════════════════════════════════════════════════════════════════
print("[10.1] Prediction overlays ...")

last_fold = folds_to_run[-1] if folds_to_run else 0
best_pt   = P["CKPT"] / f"yolov11_fold{last_fold}" / "weights" / "best.pt"
fold_ds   = P["YOLO_DS"] / f"fold_{last_fold}"

if not best_pt.exists():
    print("  No model — skipping.")
else:
    vm = YOLO(str(best_pt))
    # We use path lists, so read val.txt to enumerate val images.
    val_list = fold_ds / "val.txt"
    val_imgs = [Path(p.strip()) for p in val_list.read_text().splitlines() if p.strip()] \
               if val_list.exists() else []

    if not val_imgs:
        print("  No val images.")
    else:
        sel = random.sample(val_imgs, min(6, len(val_imgs)))
        nr  = (len(sel) + 2) // 3
        fig, axes = plt.subplots(nr, 3, figsize=(20, 7 * nr))
        axes = np.array(axes).flatten()
        for i, ip in enumerate(sel):
            ax = axes[i]
            img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
            # Label file lives next to the image under labels/ instead of images/.
            lp = P["TILES"] / "labels" / (ip.stem + ".txt")
            ng = 0
            if lp.exists():
                for ln in lp.read_text().strip().split("\n"):
                    p = ln.split()
                    if len(p) == 5:
                        _, cx, cy, w, h = map(float, p)
                        cv2.rectangle(
                            img,
                            (int((cx - w / 2) * 512), int((cy - h / 2) * 512)),
                            (int((cx + w / 2) * 512), int((cy + h / 2) * 512)),
                            (0, 255, 0), 2,
                        )
                        ng += 1
            r = vm.predict(str(ip), conf=CONF_THRESHOLD, verbose=False)
            n_pred = 0
            if r and len(r[0].boxes):
                for b in r[0].boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    n_pred += 1
            ax.imshow(img)
            ax.set_title(f"GT:{ng} Pred:{n_pred}")
            ax.axis("off")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"Fold {last_fold}: Green=GT, Red=Pred", fontsize=14)
        plt.tight_layout()
        plt.savefig(P["PLOTS"] / "predictions.png", dpi=150, bbox_inches="tight")
        plt.show()
print("[10.1] Done.")
''')

code(r'''# ══════════════════════════════════════════════════════════════════
#  10.2  Training curves (loss / mAP / P&R / LR)
# ══════════════════════════════════════════════════════════════════
print("[10.2] Training curves ...")
csv_p = P["CKPT"] / f"yolov11_fold{last_fold}" / "results.csv"
if not csv_p.exists():
    print("  No results.csv")
else:
    df = pd.read_csv(csv_p)
    df.columns = [c.strip() for c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (t, v, ttl) in enumerate([
        ("train/box_loss", "val/box_loss", "Box"),
        ("train/cls_loss", "val/cls_loss", "Cls"),
        ("train/dfl_loss", "val/dfl_loss", "DFL"),
    ]):
        ax = axes[0, i]
        if t in df: ax.plot(df["epoch"], df[t], label="Train", lw=2)
        if v in df: ax.plot(df["epoch"], df[v], label="Val",   lw=2)
        ax.set_title(ttl); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1, 0]
    for c in [x for x in df.columns if "map" in x.lower()]:
        ax.plot(df["epoch"], df[c], label=c, lw=2)
    ax.set_title("mAP"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    if "metrics/precision(B)" in df:
        ax.plot(df["epoch"], df["metrics/precision(B)"], label="P")
        ax.plot(df["epoch"], df["metrics/recall(B)"],    label="R")
    ax.set_title("P & R"); ax.legend()

    ax = axes[1, 2]
    for c in [x for x in df.columns if "lr" in x.lower()]:
        ax.plot(df["epoch"], df[c], label=c)
    ax.set_title("LR"); ax.legend(fontsize=8)

    for a in axes.flatten():
        a.set_xlabel("Epoch"); a.grid(alpha=.3)
    plt.suptitle(f"Fold {last_fold} Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(P["PLOTS"] / "curves.png", dpi=150, bbox_inches="tight")
    plt.show()
print("[10.2] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 11. Error analysis
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 11. Error Analysis (size & density breakdown)
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  11.1  Per-size, per-density recall on a random val sample
# ══════════════════════════════════════════════════════════════════
print("[11.1] Error analysis ...")

if not best_pt.exists():
    print("  No model.")
else:
    em = YOLO(str(best_pt))

    val_list = fold_ds / "val.txt"
    vi = [Path(p.strip()) for p in val_list.read_text().splitlines() if p.strip()] \
         if val_list.exists() else []
    vl = P["TILES"] / "labels"

    ns = min(50 if DEBUG_MODE else 200, len(vi))
    samp = random.sample(vi, ns) if ns else []

    S = {
        "tp": 0, "fp": 0, "fn": 0, "gt": 0,
        "sz": defaultdict(lambda: {"gt": 0, "tp": 0}),
        "dn": defaultdict(lambda: {"gt": 0, "tp": 0}),
    }

    for ip in tqdm(samp, desc="  Analysis"):
        lp = vl / (ip.stem + ".txt")
        gt = []
        if lp.exists():
            for ln in lp.read_text().strip().split("\n"):
                p = ln.split()
                if len(p) == 5:
                    _, cx, cy, w, h = map(float, p)
                    gt.append({"cx": cx, "cy": cy, "w": w, "h": h,
                               "a": w * h * 512 * 512})
        r = em.predict(str(ip), conf=CONF_THRESHOLD, verbose=False)
        pds = []
        if r and len(r[0].boxes):
            for b in r[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy() / 512
                pds.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        S["gt"] += len(gt)
        mg, mp = set(), set()
        for pi, p in enumerate(pds):
            bi, bv = -1, 0
            for gi, g in enumerate(gt):
                if gi in mg:
                    continue
                gx1, gy1 = g["cx"] - g["w"] / 2, g["cy"] - g["h"] / 2
                gx2, gy2 = g["cx"] + g["w"] / 2, g["cy"] + g["h"] / 2
                ix = max(0, min(p["x2"], gx2) - max(p["x1"], gx1))
                iy = max(0, min(p["y2"], gy2) - max(p["y1"], gy1))
                inter = ix * iy
                un = (p["x2"] - p["x1"]) * (p["y2"] - p["y1"]) + g["w"] * g["h"] - inter
                iou = inter / max(un, 1e-9)
                if iou > bv:
                    bv = iou; bi = gi
            if bv >= .5:
                mg.add(bi); mp.add(pi); S["tp"] += 1
        S["fp"] += len(pds) - len(mp)
        S["fn"] += len(gt)  - len(mg)
        for gi, g in enumerate(gt):
            a = g["a"]
            bn = "<100" if a < 100 else "100-300" if a < 300 \
                 else "300-600" if a < 600 else "600+"
            S["sz"][bn]["gt"] += 1
            if gi in mg:
                S["sz"][bn]["tp"] += 1
        d = "sparse" if len(gt) <= 5 else "moderate" if len(gt) <= 20 else "dense"
        S["dn"][d]["gt"] += len(gt)
        S["dn"][d]["tp"] += len(mg)

    pr = S["tp"] / max(S["tp"] + S["fp"], 1)
    rc = S["tp"] / max(S["tp"] + S["fn"], 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    print(f"  Precision={pr:.3f}  Recall={rc:.3f}  F1={f1:.3f}")
    print(f"  By size:")
    for b in ["<100", "100-300", "300-600", "600+"]:
        s = S["sz"][b]
        if s["gt"]:
            print(f"    {b:10s} {s['gt']:5d} GT  recall={s['tp'] / s['gt']:.3f}")
    print(f"  By density:")
    for d in ["sparse", "moderate", "dense"]:
        s = S["dn"][d]
        if s["gt"]:
            print(f"    {d:10s} {s['gt']:5d} GT  recall={s['tp'] / s['gt']:.3f}")
    with open(P["METRICS"] / "error_analysis.json", "w") as f:
        json.dump({"precision": pr, "recall": rc, "f1": f1}, f, indent=2)

print("[11.1] Done.")
''')

# ──────────────────────────────────────────────────────────────────────
# 12. Summary
# ──────────────────────────────────────────────────────────────────────
md(r"""---
## 12. Summary

### Output layout
```
outputs/
├── checkpoints/yolov11_fold{N}/      # ultralytics run dir per fold
│   ├── weights/last.pt               # rolling checkpoint (resume source)
│   ├── weights/best.pt               # best val mAP so far
│   ├── results.csv                   # per-epoch metrics + losses
│   └── args.yaml                     # frozen training args
├── weights/fold{N}_best.pt           # tidy copy of each fold's best
├── metrics/                          # per-fold + aggregated CSV/JSON
├── plots/                            # fig3, predictions, curves
├── tiles/                            # 512x512 PNG tiles + YOLO labels
├── yolo_dataset/fold_{N}/            # dataset.yaml + train.txt + val.txt
└── logs/
```

### Resume cheat sheet
| Scenario | What happens on re-run |
|---|---|
| Kernel crashed mid-epoch | `last.pt` exists with optimizer → continues at next epoch |
| You hit Ctrl-C | Same as above |
| You deleted `last.pt` only | Falls back to `best.pt` as warm-start (epoch counter resets) |
| You deleted the whole fold dir | Trains fresh from `YOLO_VARIANT` |
| Run already converged (early stop) | Skips re-training, runs eval only |
""")

code(r'''# ══════════════════════════════════════════════════════════════════
#  12.1  Save run summary
# ══════════════════════════════════════════════════════════════════
print("[12.1] Saving summary ...")

summary = {
    "paper":           "Yuan et al. 2026, ESWA 307",
    "model":           YOLO_VARIANT,
    "debug":           DEBUG_MODE,
    "folds":           NUM_FOLDS,
    "folds_completed": len(all_metrics) if "all_metrics" in dir() else 0,
    "epochs_max":      EPOCHS,
    "patience":        PATIENCE,
    "batch":           BATCH_SIZE,
    "lr":              LEARNING_RATE,
    "momentum":        MOMENTUM,
    "weight_decay":    WEIGHT_DECAY,
    "optimizer":       "SGD",
    "tile_size":       TILE_SIZE,
    "img_size":        IMG_SIZE,
    "amp":             AMP,
    "workers":         NUM_WORKERS,
    "save_period":     SAVE_PERIOD,
    "gpu":             HW["name"],
    "gpu_mem_gb":      HW["mem_gb"],
    "timestamp":       datetime.now().isoformat(),
}
with open(P["METRICS"] / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Saved: {P['METRICS'] / 'summary.json'}")
print(f"\n  {'=' * 50}")
print(f"  COMPLETE")
print(f"  {'=' * 50}")
print(f"  Mode  : {'DEBUG' if DEBUG_MODE else 'PAPER'}")
print(f"  Model : {YOLO_VARIANT}")
print(f"  Folds : {len(all_metrics) if 'all_metrics' in dir() else '?'}/{NUM_FOLDS}")

if DEBUG_MODE:
    print(f"\n  Set DEBUG_MODE=False for paper results.")
    print(f"  Set CURRENT_FOLD='all' (or 0..{NUM_FOLDS - 1}) to train more folds.")

print("[12.1] Done.")
''')


# ──────────────────────────────────────────────────────────────────────
# Assemble notebook
# ──────────────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB_PATH} with {len(cells)} cells")
