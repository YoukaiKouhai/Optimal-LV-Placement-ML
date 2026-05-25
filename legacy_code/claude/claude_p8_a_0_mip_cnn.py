"""
CRT Lead Detection — Step 8: MIP Projection CNN (PyTorch + CUDA)
================================================================
IDEA (from your friend + your 1.1 suggestion)
----------------------------------------------
Finding a 3D point (x,y,z) directly is hard.
Split it into three easier 2D problems:

  Projection 1 — collapse along Z (axial MIP):
    Image shape: (Y, X)  →  model predicts (x, y) of each electrode

  Projection 2 — collapse along Y (coronal MIP):
    Image shape: (Z, X)  →  model predicts (x, z) of each electrode

  Projection 3 — collapse along X (sagittal MIP):
    Image shape: (Z, Y)  →  model predicts (y, z) of each electrode

Then triangulate 3D:
    x = mean(x from proj1, x from proj2)
    y = mean(y from proj1, y from proj3)
    z = mean(z from proj2, z from proj3)

PER-ELECTRODE MODELS (idea 1.1)
---------------------------------
Train one separate CNN per electrode per projection direction.
Total: 6 electrodes × 3 projections = 18 small models.

With only 59 GT patients, a single model predicting 6 electrodes at
once is dividing its attention 6 ways. One model per electrode keeps
the full dataset focused on one target.

WHY MIP WORKS FOR LEADS
------------------------
Pacing leads are metal (>2000 HU) — extremely bright in CT.
Maximum Intensity Projection preserves the brightest voxel along each
ray, so lead electrodes appear as clear bright spots in 2D even if
they are only 3×3×3 voxels in 3D.

ARCHITECTURE — Small CNN (avoids overfitting on n=59)
------------------------------------------------------
Input:  (1, IMG_SIZE, IMG_SIZE)  — single-channel MIP image
        Conv2d → BN → ReLU → MaxPool  ×3
        AdaptiveAvgPool → Flatten
        FC(128→64) → Dropout → FC(64→2)
Output: (x_norm, y_norm) in [0,1] image coordinates

TRAINING STRATEGY
-----------------
  • 5-fold cross-validation (LOO with 59 patients × 18 models × 50
    epochs would take hours; 5-fold is the standard compromise)
  • Data augmentation: horizontal/vertical flip, small rotation, brightness jitter
  • GPU (CUDA) if available — should train in minutes total
  • Early stopping per fold

Requirements:
    pip install torch torchvision nibabel scikit-learn matplotlib scipy

Usage:
    python claude_p8_mip_cnn.py
"""

import json
import math
import warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib  import Path
from copy     import deepcopy
from scipy    import ndimage as ndi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data    import Dataset, DataLoader
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")
INVENTORY_JSON = BASE_DIR / "data_inventory.json"
CENTROIDS_JSON = BASE_DIR / "centroids_results.json"   # GT world coords

OUTPUT_DIR     = BASE_DIR / "mip_models"               # saved model weights
OUTPUT_RESULTS = BASE_DIR / "mip_results.json"
OUTPUT_REPORT  = BASE_DIR / "mip_report.txt"
OUTPUT_PLOT    = BASE_DIR / "mip_results_plot.png"

OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE    = 128      # resize all MIPs to this (pixels). 128 is fast; try 256 later
N_FOLDS     = 5        # k-fold CV (use 3 if you want more data per fold)
EPOCHS      = 80       # per fold
BATCH_SIZE  = 8
LR          = 1e-3
WEIGHT_DECAY= 1e-4
EARLY_STOP  = 15       # stop if val loss stagnant for N epochs

# Metal HU threshold for MIP clipping — electrodes are >2000 HU
# We clip the MIP at this value before normalising to [0,1]
# so the electrode spots are preserved at maximum brightness
MIP_CLIP_HU = 3000

# Electrodes to train models for
ELECTRODES = ["LL1", "LL2", "LL3", "LL4", "RL1", "RL2"]

# Seg label → electrode name (from Step 2)
SEG_TO_NAME = {4004:"LL1", 4005:"LL2", 4006:"LL3", 4007:"LL4",
               4008:"RL1", 4009:"RL2"}

# Projection axes and what 2D coords they predict
# Each entry: (axis_to_collapse, coord_axes_predicted)
# coord_axes_predicted refers to indices in the LPS world vector [X, Y, Z]
PROJECTIONS = {
    "axial":    {"collapse": 0, "pred_world": [0, 1]},  # collapse dim0 → predicts X,Y
    "coronal":  {"collapse": 1, "pred_world": [0, 2]},  # collapse dim1 → predicts X,Z
    "sagittal": {"collapse": 2, "pred_world": [1, 2]},  # collapse dim2 → predicts Y,Z
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  MIP GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_mip(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Maximum Intensity Projection along *axis*.
    Returns a 2D float32 array, values clipped to [HU_min, MIP_CLIP_HU]
    and normalised to [0, 1].
    """
    mip = data.max(axis=axis)               # take max along the chosen axis
    mip = np.clip(mip, -1000, MIP_CLIP_HU)
    mip = (mip + 1000) / (MIP_CLIP_HU + 1000)   # normalise to [0,1]
    return mip.astype(np.float32)


def resize_mip(mip: np.ndarray, size: int) -> np.ndarray:
    """Resize a 2D MIP to (size, size) using bilinear interpolation."""
    # use scipy zoom — no torchvision dependency at this point
    factors = (size / mip.shape[0], size / mip.shape[1])
    return ndi.zoom(mip, factors, order=1).astype(np.float32)


def world_lps_to_pixel(world_xyz: list,
                        affine: np.ndarray,
                        data_shape: tuple,
                        collapse_axis: int,
                        img_size: int) -> tuple[float, float]:
    """
    Convert a world LPS coordinate to (row_norm, col_norm) in the MIP image.

    Steps:
      1. LPS → RAS (negate X, Y)
      2. RAS → voxel (inv affine)
      3. Drop the collapsed axis → 2D pixel (row, col)
      4. Scale to the resized image → normalise to [0,1]

    Returns (row_norm, col_norm), both in [0,1].
    Row = first remaining axis after dropping collapse_axis.
    Col = second remaining axis.
    """
    xyz_lps = np.array(world_xyz, dtype=float)
    xyz_ras = np.array([-xyz_lps[0], -xyz_lps[1], xyz_lps[2], 1.0])
    vox     = np.linalg.inv(affine) @ xyz_ras    # (i, j, k, 1)
    ijk     = vox[:3]

    # the two axes that remain after collapsing
    remaining = [a for a in range(3) if a != collapse_axis]
    row_idx = ijk[remaining[0]]   # first remaining axis = rows
    col_idx = ijk[remaining[1]]   # second remaining axis = cols

    # normalise by original volume dimension then scale to img_size
    row_norm = row_idx / data_shape[remaining[0]]
    col_norm = col_idx / data_shape[remaining[1]]

    # clamp to [0,1] — small overruns can happen near borders
    row_norm = float(np.clip(row_norm, 0.0, 1.0))
    col_norm = float(np.clip(col_norm, 0.0, 1.0))

    return row_norm, col_norm


def pixel_to_world_lps(row_norm: float, col_norm: float,
                        affine: np.ndarray,
                        data_shape: tuple,
                        collapse_axis: int,
                        collapse_mid: float) -> np.ndarray:
    """
    Convert normalised 2D pixel → world LPS.

    For the collapsed dimension we use *collapse_mid* (the midpoint of
    the volume along that axis) as a placeholder — it gets averaged away
    during triangulation.
    """
    remaining = [a for a in range(3) if a != collapse_axis]

    ijk    = np.zeros(3)
    ijk[remaining[0]] = row_norm * data_shape[remaining[0]]
    ijk[remaining[1]] = col_norm * data_shape[remaining[1]]
    ijk[collapse_axis] = collapse_mid

    ras = affine @ np.array([ijk[0], ijk[1], ijk[2], 1.0])
    return np.array([-ras[0], -ras[1], ras[2]])   # LPS


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MIPDataset(Dataset):
    """
    Each item:
        image  : (1, IMG_SIZE, IMG_SIZE)  float32  — normalised MIP
        target : (2,)                     float32  — (row_norm, col_norm)
        pid    : str
    """
    def __init__(self, items: list, augment: bool = False):
        self.items   = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img  = item["mip"].copy()       # (H, W) float32
        tgt  = item["target"].copy()    # (2,)   float32

        if self.augment:
            # horizontal flip (along cols)
            if np.random.rand() < 0.5:
                img  = img[:, ::-1].copy()
                tgt  = np.array([tgt[0], 1.0 - tgt[1]])

            # vertical flip (along rows)
            if np.random.rand() < 0.5:
                img  = img[::-1, :].copy()
                tgt  = np.array([1.0 - tgt[0], tgt[1]])

            # brightness jitter — shift then rescale
            shift  = np.random.uniform(-0.05, 0.05)
            img    = np.clip(img + shift, 0.0, 1.0)

        img_t = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)
        tgt_t = torch.from_numpy(tgt.astype(np.float32))
        return img_t, tgt_t, item["pid"]


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL — Small CNN
# ─────────────────────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    """
    Tiny convolutional regressor.

    3 conv blocks: Conv → BN → ReLU → MaxPool(2)
    Then global average pool → two FC layers → 2 outputs (row, col)

    Why small? Only 59 patients. A ResNet would memorize every image.
    This architecture has ~180K parameters — right-sized for our data.
    """
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 128→64
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 64→32
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 32→16
        )
        self.pool   = nn.AdaptiveAvgPool2d(1)                   # (128,1,1)
        self.drop   = nn.Dropout(0.4)
        self.fc1    = nn.Linear(128, 64)
        self.fc2    = nn.Linear(64, 2)                          # row, col

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)   # (B, 128)
        x = F.relu(self.fc1(self.drop(x)))
        return torch.sigmoid(self.fc2(x))   # sigmoid → [0,1]


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(train_items, val_items, epochs, lr, weight_decay,
                   early_stop_patience) -> tuple[nn.Module, list]:
    """
    Train one SmallCNN for one (electrode, projection, fold) combination.
    Returns (best_model, val_loss_history).
    """
    train_set = MIPDataset(train_items, augment=True)
    val_set   = MIPDataset(val_items,  augment=False)
    train_dl  = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=len(train_items) > BATCH_SIZE)
    val_dl    = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    model     = SmallCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=lr/20)
    criterion = nn.MSELoss()

    best_val   = float("inf")
    best_state = deepcopy(model.state_dict())
    patience   = 0
    history    = []

    for epoch in range(epochs):
        # ── train ──
        model.train()
        for imgs, tgts, _ in train_dl:
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, tgts)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ── validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, tgts, _ in val_dl:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                pred = model(imgs)
                val_losses.append(criterion(pred, tgts).item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        history.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = deepcopy(model.state_dict())
            patience   = 0
        else:
            patience  += 1
            if patience >= early_stop_patience:
                break

    model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
#  PIXEL-SPACE ERROR → WORLD-SPACE ERROR
# ─────────────────────────────────────────────────────────────────────────────

def pixel_error_to_mm(row_err: float, col_err: float,
                       data_shape: tuple, collapse_axis: int,
                       affine: np.ndarray) -> float:
    """
    Convert normalised pixel errors → approximate mm error.

    We estimate mm/pixel by examining the affine's voxel spacing
    and the volume dimension along each remaining axis.
    """
    remaining = [a for a in range(3) if a != collapse_axis]
    # voxel spacing along each axis (mm per voxel)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))  # (3,) mm/voxel

    row_mm = row_err * data_shape[remaining[0]] * spacing[remaining[0]]
    col_mm = col_err * data_shape[remaining[1]] * spacing[remaining[1]]
    return float(math.sqrt(row_mm**2 + col_mm**2))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print(f"  CRT Lead — Step 8: MIP Projection CNN")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # ── load data ─────────────────────────────────────────────────────────
    with open(INVENTORY_JSON,  encoding="utf-8") as f: inventory  = json.load(f)
    with open(CENTROIDS_JSON,  encoding="utf-8") as f: centroids  = json.load(f)

    # ── build patient list: need both raw CT AND ground truth centroids ───
    print("\n  Finding GT patients with raw CT images...")
    gt_patients = []
    for ds_key in ("dataset_1", "dataset_2"):
        for pid, rec in inventory.get(ds_key, {}).items():
            if rec.get("img_nii") and pid in centroids:
                if not any(p["pid"] == pid for p in gt_patients):
                    gt_patients.append({
                        "pid":      pid,
                        "img_path": rec["img_nii"],
                    })

    print(f"  Found {len(gt_patients)} GT patients with raw CT\n")

    if len(gt_patients) < N_FOLDS:
        print(f"⚠  Need at least {N_FOLDS} patients. Exiting.")
        return

    # ─────────────────────────────────────────────────────────────────────
    #  PRE-COMPUTE ALL MIPs + TARGETS
    # ─────────────────────────────────────────────────────────────────────
    print("  Pre-computing MIPs for all patients (this takes a few minutes)...")
    print(f"  {'PID':<8}  {'Shape':>16}  {'MIPs generated'}")
    print("  " + "─" * 42)

    # patient_data[pid] = {
    #   "axial":    { "mip": (H,W) np array,  "shape": ..., "affine": ... },
    #   "coronal":  { ... },
    #   "sagittal": { ... },
    #   "targets":  { "LL1": {"axial": (r,c), "coronal": (r,c), ...}, ... }
    # }
    patient_data = {}

    for pt in gt_patients:
        pid      = pt["pid"]
        img_path = pt["img_path"]

        try:
            nii    = nib.load(img_path)
            data   = nii.get_fdata(dtype=np.float32)
            affine = nii.affine
        except Exception as e:
            print(f"  {pid:<8}  ERROR: {e}")
            continue

        shape = data.shape
        mips  = {}
        for proj_name, cfg in PROJECTIONS.items():
            ax  = cfg["collapse"]
            mip = make_mip(data, ax)
            mips[proj_name] = {
                "mip":    resize_mip(mip, IMG_SIZE),
                "shape":  shape,
                "affine": affine,
                "collapse_mid": shape[ax] / 2.0,
            }

        # ── compute 2D pixel targets from GT world coords ─────────────────
        gt_cent = centroids[pid].get("centroids", {})
        targets = {}
        for lbl_str, entry in gt_cent.items():
            lbl_int = int(lbl_str)
            name    = SEG_TO_NAME.get(lbl_int)
            if name is None:
                continue
            world_xyz = entry["world_xyz"]
            targets[name] = {}
            for proj_name, cfg in PROJECTIONS.items():
                ax = cfg["collapse"]
                r, c = world_lps_to_pixel(
                    world_xyz, affine, shape, ax, IMG_SIZE
                )
                targets[name][proj_name] = (r, c)

        patient_data[pid] = {
            "mips":    mips,
            "targets": targets,
            "shape":   shape,
            "affine":  affine,
        }
        print(f"  {pid:<8}  {str(shape):>16}  "
              f"{list(mips.keys())}")

    valid_pids = list(patient_data.keys())
    print(f"\n  {len(valid_pids)} patients ready for training\n")

    # ─────────────────────────────────────────────────────────────────────
    #  5-FOLD CV — one model per electrode per projection
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  Training — 1 CNN per electrode per projection")
    print(f"  {len(ELECTRODES)} electrodes × 3 projections × {N_FOLDS} folds"
          f" = {len(ELECTRODES)*3*N_FOLDS} total training runs")
    print("=" * 65)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    pid_array = np.array(valid_pids)

    # Store per-patient predictions for final 3D triangulation
    # pred_store[pid][electrode][proj_name] = (row_pred, col_pred)
    pred_store = {pid: {elec: {} for elec in ELECTRODES}
                  for pid in valid_pids}

    # Results: results[electrode][proj_name] = list of pixel errors (normalised)
    results = {e: {p: [] for p in PROJECTIONS} for e in ELECTRODES}

    for electrode in ELECTRODES:
        for proj_name, proj_cfg in PROJECTIONS.items():
            collapse_ax = proj_cfg["collapse"]

            print(f"\n  [{electrode}] [{proj_name}] "
                  f"— training {N_FOLDS} folds...", flush=True)

            fold_errors = []

            for fold_idx, (train_idx, val_idx) in enumerate(
                    kf.split(pid_array)):

                train_pids = pid_array[train_idx].tolist()
                val_pids   = pid_array[val_idx].tolist()

                # ── filter to patients that have this electrode ────────────
                def make_items(pids):
                    items = []
                    for pid in pids:
                        if pid not in patient_data:
                            continue
                        tgt_dict = patient_data[pid]["targets"].get(electrode)
                        if tgt_dict is None:
                            continue
                        rc = tgt_dict.get(proj_name)
                        if rc is None:
                            continue
                        items.append({
                            "pid":    pid,
                            "mip":    patient_data[pid]["mips"][proj_name]["mip"],
                            "target": np.array(rc, dtype=np.float32),
                        })
                    return items

                train_items = make_items(train_pids)
                val_items   = make_items(val_pids)

                if len(train_items) < 3 or len(val_items) == 0:
                    continue

                # ── train ────────────────────────────────────────────────
                model, _ = train_one_fold(
                    train_items, val_items,
                    EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP
                )

                # ── evaluate on val set ───────────────────────────────────
                model.eval()
                fold_val_errs = []
                with torch.no_grad():
                    for item in val_items:
                        img_t = torch.from_numpy(
                            item["mip"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        pred  = model(img_t).cpu().numpy()[0]   # (2,)
                        true  = item["target"]                  # (2,)

                        # pixel error (normalised, [0,1] scale)
                        pix_err = float(np.linalg.norm(pred - true))
                        fold_val_errs.append(pix_err)
                        results[electrode][proj_name].append(pix_err)

                        # store prediction for triangulation
                        pid = item["pid"]
                        pred_store[pid][electrode][proj_name] = (
                            float(pred[0]), float(pred[1])
                        )

                mean_fold = np.mean(fold_val_errs) if fold_val_errs else float("nan")
                fold_errors.append(mean_fold)
                print(f"    fold {fold_idx+1}/{N_FOLDS}  "
                      f"n_train={len(train_items)}  "
                      f"n_val={len(val_items)}  "
                      f"mean_pix_err={mean_fold:.4f}", flush=True)

            # save the final model trained on all data
            if fold_errors:
                all_items = make_items(valid_pids)
                if len(all_items) >= 3:
                    final_model, _ = train_one_fold(
                        all_items, all_items[:3],
                        EPOCHS, LR, WEIGHT_DECAY, EARLY_STOP
                    )
                    save_path = (OUTPUT_DIR /
                                 f"model_{electrode}_{proj_name}.pt")
                    torch.save(final_model.state_dict(), save_path)

    # ─────────────────────────────────────────────────────────────────────
    #  TRIANGULATE 3D POSITIONS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Triangulating 3D positions from 2D predictions")
    print("=" * 65)

    triangulated = {}   # pid → electrode → {"pred_xyz", "gt_xyz", "err_mm"}

    for pid in valid_pids:
        pt_data     = patient_data[pid]
        affine      = pt_data["affine"]
        shape       = pt_data["shape"]
        gt_targets  = pt_data["targets"]
        triangulated[pid] = {}

        for electrode in ELECTRODES:
            proj_preds = pred_store[pid][electrode]
            if len(proj_preds) < 2:
                continue   # need at least 2 projections to triangulate

            # collect world coordinate estimates from each projection
            xyz_estimates = []
            for proj_name, (r_pred, c_pred) in proj_preds.items():
                collapse_ax  = PROJECTIONS[proj_name]["collapse"]
                collapse_mid = pt_data["mips"][proj_name]["collapse_mid"]
                world_est = pixel_to_world_lps(
                    r_pred, c_pred, affine, shape,
                    collapse_ax, collapse_mid
                )
                xyz_estimates.append((proj_name, world_est))

            if not xyz_estimates:
                continue

            # triangulate: for each world axis, average the two projections
            # that provide it (ignore the collapsed one from each)
            # X: from axial (proj0=X,Y) and coronal (proj0=X,Z)
            # Y: from axial (proj1=Y) and sagittal (proj1=Y)
            # Z: from coronal (proj1=Z) and sagittal (proj0=Z)
            pred_by_proj = {name: xyz for name, xyz in xyz_estimates}

            x_ests, y_ests, z_ests = [], [], []
            if "axial"    in pred_by_proj:
                x_ests.append(pred_by_proj["axial"][0])
                y_ests.append(pred_by_proj["axial"][1])
            if "coronal"  in pred_by_proj:
                x_ests.append(pred_by_proj["coronal"][0])
                z_ests.append(pred_by_proj["coronal"][2])
            if "sagittal" in pred_by_proj:
                y_ests.append(pred_by_proj["sagittal"][1])
                z_ests.append(pred_by_proj["sagittal"][2])

            pred_xyz = np.array([
                float(np.mean(x_ests)) if x_ests else 0.0,
                float(np.mean(y_ests)) if y_ests else 0.0,
                float(np.mean(z_ests)) if z_ests else 0.0,
            ])

            # ground truth
            gt_entry = centroids[pid].get("centroids", {})
            gt_xyz   = None
            for lbl_str, entry in gt_entry.items():
                if SEG_TO_NAME.get(int(lbl_str)) == electrode:
                    gt_xyz = np.array(entry["world_xyz"])
                    break

            err_mm = (float(np.linalg.norm(pred_xyz - gt_xyz))
                      if gt_xyz is not None else None)

            triangulated[pid][electrode] = {
                "pred_xyz": pred_xyz.tolist(),
                "gt_xyz":   gt_xyz.tolist() if gt_xyz is not None else None,
                "err_mm":   round(err_mm, 2) if err_mm is not None else None,
            }

    # ─────────────────────────────────────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    report_lines = ["Step 8 — MIP CNN Results", "=" * 65]

    all_errors = []
    per_electrode_errors = {e: [] for e in ELECTRODES}

    for pid, elec_dict in triangulated.items():
        for electrode, rec in elec_dict.items():
            if rec["err_mm"] is not None:
                all_errors.append(rec["err_mm"])
                per_electrode_errors[electrode].append(rec["err_mm"])

    print(f"\n  {'Electrode':<8}  {'N':>4}  {'Mean err':>10}  "
          f"{'Median':>8}  {'<5mm':>8}  {'<10mm':>8}")
    print("  " + "─" * 52)
    report_lines.append(
        f"  {'Electrode':<8}  {'N':>4}  {'Mean err':>10}  "
        f"{'Median':>8}  {'<5mm':>8}  {'<10mm':>8}"
    )

    for electrode in ELECTRODES:
        errs = per_electrode_errors[electrode]
        if not errs:
            continue
        mean_e  = np.mean(errs)
        med_e   = np.median(errs)
        pct5    = 100 * sum(e < 5  for e in errs) / len(errs)
        pct10   = 100 * sum(e < 10 for e in errs) / len(errs)
        row     = (f"  {electrode:<8}  {len(errs):>4}  {mean_e:>8.2f} mm"
                   f"  {med_e:>6.2f} mm  {pct5:>6.1f}%  {pct10:>6.1f}%")
        print(row)
        report_lines.append(row)

    if all_errors:
        print("  " + "─" * 52)
        overall = (f"  {'OVERALL':<8}  {len(all_errors):>4}  "
                   f"{np.mean(all_errors):>8.2f} mm  "
                   f"{np.median(all_errors):>6.2f} mm  "
                   f"{100*sum(e<5 for e in all_errors)/len(all_errors):>6.1f}%  "
                   f"{100*sum(e<10 for e in all_errors)/len(all_errors):>6.1f}%")
        print(overall)
        report_lines.append(overall)

    comparison = [
        "",
        "COMPARISON ACROSS ALL METHODS:",
        f"  Step 2  Seg centroid  (GT seg needed)  : ~0.1 mm  [gold standard]",
        f"  Step 5b Classical CV  (no seg needed)  : ~1.4 mm  when detected",
        f"  Step 6  PointNet blob (no seg needed)  : ~1.2 mm  when detected",
        f"  Step 8  MIP CNN       (no seg needed)  : "
        f"{np.mean(all_errors):.1f} mm  ← this step" if all_errors else "",
        "",
        "INTERPRETATION OF MIP CNN:",
        "  Error < 5mm  → clinically acceptable for lead localization",
        "  Error < 10mm → useful for approximate normalization",
        "  Error > 10mm → model needs more data or larger image size",
        "",
        "TIPS TO IMPROVE:",
        f"  • Increase IMG_SIZE from {IMG_SIZE} to 256 (sharper electrode spots)",
        "  • Add more augmentation (small rotations ±10°)",
        "  • Use focal loss weighted to electrode pixel locations",
        "  • Train on pseudo-labeled raw patients from Step 6c",
        "  • Try heatmap regression instead of direct coordinate output",
        "=" * 65,
    ]
    for line in comparison:
        print(line)
    report_lines.extend(comparison)

    # ── plot ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Step 8 — MIP CNN: 3D Triangulation Results\n"
                 "Per-electrode position error after 2D→3D triangulation",
                 fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"LL1":"#4e79a7","LL2":"#59a14f","LL3":"#f28e2b",
              "LL4":"#e15759","RL1":"#b07aa1","RL2":"#76b7b2"}

    # per-electrode error bars
    ax0 = fig.add_subplot(gs[0, :2])
    elec_means = [np.mean(per_electrode_errors[e]) if per_electrode_errors[e]
                  else 0 for e in ELECTRODES]
    elec_stds  = [np.std(per_electrode_errors[e])  if per_electrode_errors[e]
                  else 0 for e in ELECTRODES]
    bars = ax0.bar(ELECTRODES, elec_means, yerr=elec_stds,
                   color=[colors[e] for e in ELECTRODES],
                   alpha=0.85, capsize=5, edgecolor="k", lw=0.6)
    ax0.axhline(5,  color="green",  linestyle="--", lw=1.2, label="5mm target")
    ax0.axhline(10, color="orange", linestyle="--", lw=1.2, label="10mm threshold")
    ax0.set_ylabel("Mean 3D position error (mm)")
    ax0.set_title("Per-Electrode Triangulation Error\n(5-fold CV, mean ± std)")
    ax0.legend(fontsize=9)
    ax0.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, elec_means):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{m:.1f}", ha="center", fontsize=8)

    # error histogram
    ax1 = fig.add_subplot(gs[0, 2])
    if all_errors:
        ax1.hist(all_errors, bins=20, color="#4e79a7", alpha=0.8,
                 edgecolor="white")
        ax1.axvline(5,  color="green",  lw=1.5, linestyle="--", label="5mm")
        ax1.axvline(10, color="orange", lw=1.5, linestyle="--", label="10mm")
        ax1.set_xlabel("3D Error (mm)")
        ax1.set_ylabel("Count")
        ax1.set_title("Error Distribution\n(all electrodes combined)")
        ax1.legend(fontsize=8)

    # per-projection pixel errors
    for p_idx, (proj_name, _) in enumerate(PROJECTIONS.items()):
        ax = fig.add_subplot(gs[1, p_idx])
        means = [np.mean(results[e][proj_name]) * IMG_SIZE
                 if results[e][proj_name] else 0
                 for e in ELECTRODES]
        ax.bar(ELECTRODES, means,
               color=[colors[e] for e in ELECTRODES], alpha=0.85,
               edgecolor="k", lw=0.5)
        ax.set_ylabel("Mean pixel error (px)")
        ax.set_title(f"{proj_name.capitalize()} projection\n2D pixel error")
        ax.grid(axis="y", alpha=0.3)

    plt.savefig(OUTPUT_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n✅  Plot saved → {OUTPUT_PLOT}")

    # ── save JSON + report ────────────────────────────────────────────────
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"IMG_SIZE": IMG_SIZE, "N_FOLDS": N_FOLDS,
                       "EPOCHS": EPOCHS, "DEVICE": str(DEVICE)},
            "triangulated": triangulated,
            "per_electrode_mean_err": {
                e: round(float(np.mean(v)), 3) if v else None
                for e, v in per_electrode_errors.items()
            },
            "overall_mean_err": round(float(np.mean(all_errors)), 3)
                                 if all_errors else None,
        }, f, indent=2)
    print(f"✅  Results saved → {OUTPUT_RESULTS}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved  → {OUTPUT_REPORT}")


if __name__ == "__main__":
    missing = []
    try:
        import torch
        print(f"  PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing.append("torch")
    try:    import nibabel
    except ImportError: missing.append("nibabel")
    try:    import skimage
    except ImportError: missing.append("scikit-image")
    try:    import matplotlib
    except ImportError: missing.append("matplotlib")

    if missing:
        print(f"\n⚠  Missing: pip install {' '.join(missing)}")
        if "torch" in missing:
            print("   PyTorch install: https://pytorch.org/get-started/locally/")
    else:
        run()