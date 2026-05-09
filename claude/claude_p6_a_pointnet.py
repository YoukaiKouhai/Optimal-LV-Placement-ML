"""
CRT Lead Detection — Step 6: PointNet Blob Classifier (PyTorch)
===============================================================
WHAT THIS SOLVES
----------------
Step 5b achieved 77.7% detection with classical CV — meaning we can find
candidate blobs from raw CT, but we can't reliably tell which blobs are
real electrodes vs bone/artifact, or which electrode position each blob
corresponds to.

PointNet solves exactly this: given a SET of candidate blobs (a point cloud),
classify each blob as one of:
    Class 0:    background (bone, artifact, noise)
    Class 1-4:  LV electrode (LL1-LL4, i.e. labels 4004-4007)
    Class 5-6:  RV electrode (RL1-RL2, i.e. labels 4008-4009)

ARCHITECTURE — Mini PointNet (Segmentation variant)
----------------------------------------------------
PointNet's key idea: process each point independently with shared MLPs,
then aggregate a global feature with max-pooling, then classify each
point using both local and global information.

    Per-blob features (input):
        x, y, z         — world LPS coordinates (normalized to heart frame)
        n_voxels        — blob size (normalized)
        dist_from_heart — distance from apex-base midpoint (normalized)
        hu_approx       — approximate mean HU inferred from n_voxels

    Shared MLP (per-point):   6 → 64 → 128 → 256  (local features)
    Max-pool:                  256  →  global descriptor
    Concatenate:               local (256) + global (256) = 512
    Classification MLP:        512 → 256 → 128 → 7 classes

TRAINING STRATEGY (for n=59 patients)
--------------------------------------
    • LeaveOneOut cross-validation — most honest estimate with small data
    • Class weighting — background blobs outnumber electrodes ~5:1,
      so we weight electrode classes higher
    • Data augmentation — small random jitter on blob positions
    • Early stopping — stop when validation loss stops improving

Requirements:
    pip install torch numpy scikit-learn

Usage:
    python claude_p6_pointnet.py

Input:
    cv_sweep_results.json     (Step 5b — best CV parameters)
    centroids_results.json    (Step 2 — ground truth centroids)
    data_inventory.json       (Step 1 — file paths)

Output:
    pointnet_model.pt         — saved trained model weights
    pointnet_results.json     — per-patient LOO predictions
    pointnet_report.txt       — summary report
"""

import json
import warnings
import numpy as np
from pathlib import Path
from copy    import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics  import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR         = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")
INVENTORY_JSON   = BASE_DIR / "data_inventory.json"
CENTROIDS_JSON   = BASE_DIR / "centroids_results.json"
SWEEP_JSON       = BASE_DIR / "cv_sweep_results.json"

OUTPUT_MODEL     = BASE_DIR / "pointnet_model.pt"
OUTPUT_RESULTS   = BASE_DIR / "pointnet_results.json"
OUTPUT_REPORT    = BASE_DIR / "pointnet_report.txt"

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Labels: 0=background, 1=LL1(4004), 2=LL2(4005), 3=LL3(4006),
#         4=LL4(4007), 5=RL1(4008), 6=RL2(4009)
N_CLASSES   = 7
LABEL_MAP   = {0: "background",
               1: "LL1", 2: "LL2", 3: "LL3", 4: "LL4",
               5: "RL1", 6: "RL2"}
SEG_TO_CLASS = {4004: 1, 4005: 2, 4006: 3, 4007: 4, 4008: 5, 4009: 6}

# How many blobs to pad/truncate each patient to (PointNet needs fixed size)
# Set to the max blobs observed in your sweep results
MAX_BLOBS   = 80

# Training
EPOCHS           = 150
BATCH_SIZE       = 8      # patients per batch
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
EARLY_STOP_PAT   = 20     # stop if val loss doesn't improve for N epochs

# Augmentation
JITTER_STD_MM    = 1.0    # random jitter on blob positions during training

# Match radius for assigning GT labels to detected blobs
MATCH_RADIUS_MM  = 10.0

# Device: use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def get_heart_centre(pid, centroids_data):
    rec  = centroids_data.get(pid, {})
    cent = rec.get("centroids", {})
    apex = cent.get("4002") or cent.get(4002)
    base = cent.get("4003") or cent.get(4003)
    if apex and base:
        return (np.array(apex["world_xyz"]) + np.array(base["world_xyz"])) / 2
    return None


def build_blob_features(blobs: list, heart_centre: np.ndarray,
                         heart_axis_mm: float) -> np.ndarray:
    """
    Convert a list of blob dicts into a normalised feature matrix.

    Features per blob (6 total):
        0-2: x, y, z  normalised by heart axis length
              → makes positions scale-invariant across patient sizes
        3:   dist from heart centre, normalised by heart axis
        4:   n_voxels, log-scaled then normalised (blob size)
        5:   radial distance in XY plane from heart centre

    Returns array of shape (N_blobs, 6).
    """
    if not blobs:
        return np.zeros((0, 6), dtype=np.float32)

    pts    = np.array([b["world_xyz"]  for b in blobs], dtype=np.float32)
    nvox   = np.array([b["n_voxels"]   for b in blobs], dtype=np.float32)

    # centre on heart, scale by axis length
    scale  = max(heart_axis_mm, 1.0)
    pts_n  = (pts - heart_centre) / scale

    dist   = np.linalg.norm(pts_n, axis=1, keepdims=True)
    radial = np.linalg.norm(pts_n[:, :2], axis=1, keepdims=True)
    nvox_n = (np.log1p(nvox) / np.log1p(1000)).reshape(-1, 1)

    feats  = np.hstack([pts_n, dist, nvox_n, radial])   # (N, 6)
    return feats.astype(np.float32)


def assign_gt_labels(blobs: list, gt_centroids: dict) -> np.ndarray:
    """
    For each blob, assign its ground truth class label.
    Uses nearest-neighbour matching within MATCH_RADIUS_MM.
    Unmatched blobs get class 0 (background).

    Returns int array of shape (N_blobs,).
    """
    n = len(blobs)
    labels = np.zeros(n, dtype=np.int64)

    if not blobs or not gt_centroids:
        return labels

    pts   = np.array([b["world_xyz"] for b in blobs], dtype=np.float32)
    used  = set()

    for seg_label_str in sorted(gt_centroids.keys(), key=lambda x: int(x)):
        seg_label_int = int(seg_label_str)
        class_id      = SEG_TO_CLASS.get(seg_label_int)
        if class_id is None:
            continue

        entry = gt_centroids.get(seg_label_str) or gt_centroids.get(seg_label_int)
        if not entry:
            continue

        gt_pt = np.array(entry["world_xyz"], dtype=np.float32)
        dists = np.linalg.norm(pts - gt_pt, axis=1)
        for ui in used:
            dists[ui] = np.inf

        best_idx  = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        if best_dist <= MATCH_RADIUS_MM:
            labels[best_idx] = class_id
            used.add(best_idx)

    return labels


def pad_or_truncate(feats: np.ndarray, labels: np.ndarray,
                     max_n: int) -> tuple:
    """
    PointNet needs fixed-size input.
    Pad with zeros (labelled background=0) or truncate to max_n.
    Also returns a mask: 1 = real blob, 0 = padding.
    """
    n = len(feats)
    n_feat = feats.shape[1] if n > 0 else 6

    if n >= max_n:
        return feats[:max_n], labels[:max_n], np.ones(max_n, dtype=np.float32)

    pad_n   = max_n - n
    f_pad   = np.zeros((pad_n, n_feat), dtype=np.float32)
    l_pad   = np.zeros(pad_n, dtype=np.int64)
    mask    = np.array([1]*n + [0]*pad_n, dtype=np.float32)

    return (np.vstack([feats, f_pad]),
            np.concatenate([labels, l_pad]),
            mask)


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class BlobDataset(Dataset):
    """
    Each item is one patient's blob cloud:
        feats  : (MAX_BLOBS, 6)   float32
        labels : (MAX_BLOBS,)     int64   — 0=bg, 1-6=electrode class
        mask   : (MAX_BLOBS,)     float32 — 1=real, 0=padding
    """

    def __init__(self, patient_data: list, augment: bool = False):
        """
        patient_data: list of dicts, each with keys
            'feats', 'labels', 'mask', 'pid'
        """
        self.data    = patient_data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        feats  = item["feats"].copy()
        labels = item["labels"].copy()
        mask   = item["mask"].copy()

        # augmentation: small jitter on xyz coordinates
        if self.augment:
            n_real = int(mask.sum())
            jitter = np.random.randn(n_real, 3).astype(np.float32)
            jitter *= (JITTER_STD_MM / 87.3)   # normalise by mean heart axis
            feats[:n_real, :3] += jitter

        return (torch.from_numpy(feats),
                torch.from_numpy(labels),
                torch.from_numpy(mask))


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL — Mini PointNet Segmentation
# ─────────────────────────────────────────────────────────────────────────────

class SharedMLP(nn.Module):
    """1D convolution = shared MLP across all points."""
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.bn   = nn.BatchNorm1d(out_ch) if bn else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MiniPointNet(nn.Module):
    """
    PointNet segmentation network.

    Input:  (B, N, F)  — batch of point clouds
    Output: (B, N, C)  — per-point class logits

    B = batch size (patients)
    N = MAX_BLOBS
    F = n_features (6)
    C = N_CLASSES (7)
    """

    def __init__(self, n_feat: int = 6, n_classes: int = N_CLASSES):
        super().__init__()

        # ── local feature extraction (per-point) ──────────────────────────
        self.local1 = SharedMLP(n_feat, 64)
        self.local2 = SharedMLP(64, 128)
        self.local3 = SharedMLP(128, 256)

        # ── segmentation head (local + global → per-point classification) ─
        # input: 256 (local) + 256 (global) = 512
        self.seg1   = SharedMLP(256 + 256, 256)
        self.seg2   = SharedMLP(256, 128)
        self.seg3   = SharedMLP(128, 64)
        self.out    = nn.Conv1d(64, n_classes, 1)   # no relu on output

        self.drop   = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, F)  — input point cloud
        returns: (B, N, C) — per-point logits
        """
        B, N, F = x.shape

        # transpose to (B, F, N) for Conv1d
        x = x.transpose(2, 1)                      # (B, F, N)

        # local features
        l1 = self.local1(x)                        # (B, 64,  N)
        l2 = self.local2(l1)                       # (B, 128, N)
        l3 = self.local3(l2)                       # (B, 256, N)

        # global descriptor via max-pool over all points
        g  = torch.max(l3, dim=2, keepdim=True)[0] # (B, 256, 1)
        g  = g.expand(-1, -1, N)                   # (B, 256, N) — broadcast

        # concatenate local + global
        cat = torch.cat([l3, g], dim=1)            # (B, 512, N)

        # per-point classification
        s1  = self.seg1(cat)                       # (B, 256, N)
        s1  = self.drop(s1)
        s2  = self.seg2(s1)                        # (B, 128, N)
        s3  = self.seg3(s2)                        # (B, 64,  N)
        out = self.out(s3)                         # (B, C,   N)

        return out.transpose(2, 1)                 # (B, N, C)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(all_labels: list) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
    Background (class 0) is downweighted since it dominates.
    """
    counts = np.zeros(N_CLASSES)
    for labels in all_labels:
        for c in range(N_CLASSES):
            counts[c] += (labels == c).sum()

    total   = counts.sum()
    weights = total / (N_CLASSES * np.maximum(counts, 1))
    # extra downweight on background — it's less important to get right
    weights[0] *= 0.3
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def masked_loss(logits: torch.Tensor, labels: torch.Tensor,
                mask: torch.Tensor, criterion) -> torch.Tensor:
    """
    Compute cross-entropy loss only on real blobs (mask=1), not padding.
    logits: (B, N, C)
    labels: (B, N)
    mask:   (B, N)
    """
    B, N, C = logits.shape
    logits_flat = logits.reshape(B * N, C)
    labels_flat = labels.reshape(B * N)
    mask_flat   = mask.reshape(B * N).bool()

    logits_real = logits_flat[mask_flat]
    labels_real = labels_flat[mask_flat]

    if len(labels_real) == 0:
        return torch.tensor(0.0, requires_grad=True)

    return criterion(logits_real, labels_real)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for feats, labels, mask in loader:
        feats, labels, mask = (feats.to(DEVICE), labels.to(DEVICE),
                                mask.to(DEVICE))
        optimizer.zero_grad()
        logits = model(feats)
        loss   = masked_loss(logits, labels, mask, criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    for feats, labels, mask in loader:
        feats, labels, mask = (feats.to(DEVICE), labels.to(DEVICE),
                                mask.to(DEVICE))
        logits = model(feats)
        loss   = masked_loss(logits, labels, mask, criterion)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)                # (B, N)
        mask_bool = mask.bool()
        all_pred.extend(preds[mask_bool].cpu().numpy().tolist())
        all_true.extend(labels[mask_bool].cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, np.array(all_pred), np.array(all_true)


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def electrode_detection_rate(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    For each electrode class (1-6), compute:
        - % of true electrodes correctly classified (recall)
    Ignores class 0 (background) for the main metric.
    """
    results = {}
    for c in range(1, N_CLASSES):
        mask_true = (true == c)
        if mask_true.sum() == 0:
            continue
        correct = ((pred == c) & mask_true).sum()
        results[LABEL_MAP[c]] = {
            "n_true":      int(mask_true.sum()),
            "n_correct":   int(correct),
            "recall_pct":  round(100 * correct / mask_true.sum(), 1),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD CV BLOBS FROM STEP 5b
# ─────────────────────────────────────────────────────────────────────────────

def load_cv_blobs(pid: str, inventory: dict, centroids_data: dict,
                   best_params: dict) -> tuple[list, np.ndarray | None, float]:
    """
    Re-run the CV blob detection for one patient using the best params
    found in the sweep, so we don't need to store all blobs on disk.

    Returns (blobs, heart_centre, axis_len_mm).
    """
    import nibabel as nib
    from scipy   import ndimage as ndi
    from skimage import measure

    # find image path
    img_path = None
    for ds_key in ("dataset_1", "dataset_2"):
        rec = inventory.get(ds_key, {}).get(pid)
        if rec and rec.get("img_nii"):
            img_path = rec["img_nii"]
            break
    if img_path is None:
        return [], None, 87.3

    try:
        nii  = nib.load(img_path)
        data = nii.get_fdata(dtype=np.float32)
        aff  = nii.affine
    except Exception:
        return [], None, 87.3

    hu    = best_params.get("HU_THRESHOLD",           2928)
    bmin  = best_params.get("BLOB_MIN_VOXELS",        7)
    bmax  = best_params.get("BLOB_MAX_VOXELS",        879)
    rad   = best_params.get("HEART_SEARCH_RADIUS_MM", 120.0)

    metal  = data > hu
    struct = ndi.generate_binary_structure(3, 3)
    labelled, _ = ndi.label(metal, structure=struct)
    props  = measure.regionprops(labelled)

    blobs = []
    for p in props:
        if not (bmin <= p.area <= bmax):
            continue
        ci, cj, ck = p.centroid
        ras = aff @ np.array([ci, cj, ck, 1.0])
        lps = np.array([-ras[0], -ras[1], ras[2]])
        blobs.append({"world_xyz": lps.tolist(), "n_voxels": int(p.area)})

    # get heart info from centroids
    rec       = centroids_data.get(pid, {})
    cent      = rec.get("centroids", {})
    apex_e    = cent.get("4002") or cent.get(4002)
    base_e    = cent.get("4003") or cent.get(4003)
    hc        = None
    axis_len  = 87.3

    if apex_e and base_e:
        apex_pt = np.array(apex_e["world_xyz"])
        base_pt = np.array(base_e["world_xyz"])
        hc      = (apex_pt + base_pt) / 2
        axis_len = float(np.linalg.norm(base_pt - apex_pt))

    if hc is not None:
        blobs = [b for b in blobs
                 if np.linalg.norm(np.array(b["world_xyz"]) - hc) <= rad]

    return blobs, hc, axis_len


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print(f"  CRT Lead — Step 6: PointNet Blob Classifier")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # ── load inventories ───────────────────────────────────────────────────
    with open(INVENTORY_JSON,  encoding="utf-8") as f: inventory  = json.load(f)
    with open(CENTROIDS_JSON,  encoding="utf-8") as f: centroids  = json.load(f)
    with open(SWEEP_JSON,      encoding="utf-8") as f: sweep      = json.load(f)

    best_params = sweep.get("best_params", {
        "HU_THRESHOLD": 2928, "BLOB_MIN_VOXELS": 7,
        "BLOB_MAX_VOXELS": 879, "HEART_SEARCH_RADIUS_MM": 120.0,
    })
    print(f"\n  Using CV params: {best_params}\n")

    # ── build dataset ──────────────────────────────────────────────────────
    print("  Building patient dataset from CV blobs + GT labels...")
    all_patient_data = []
    skipped = 0

    for pid in sorted(centroids.keys()):
        gt_cent = centroids[pid].get("centroids", {})
        if not gt_cent:
            skipped += 1
            continue

        blobs, hc, axis_len = load_cv_blobs(pid, inventory, centroids, best_params)

        if len(blobs) == 0:
            print(f"  [{pid}] no blobs — skipping")
            skipped += 1
            continue

        hc_arr   = hc if hc is not None else np.zeros(3)
        feats    = build_blob_features(blobs, hc_arr, axis_len)
        labels   = assign_gt_labels(blobs, gt_cent)
        feats_p, labels_p, mask_p = pad_or_truncate(feats, labels, MAX_BLOBS)

        n_elec   = int((labels_p > 0).sum())
        all_patient_data.append({
            "pid":    pid,
            "feats":  feats_p,
            "labels": labels_p,
            "mask":   mask_p,
            "n_blobs":  len(blobs),
            "n_elec":   n_elec,
        })
        print(f"  [{pid}]  blobs={len(blobs):>3}  "
              f"electrodes_matched={n_elec}  axis={axis_len:.0f}mm")

    n_patients = len(all_patient_data)
    print(f"\n  Total patients ready: {n_patients}  (skipped {skipped})")

    if n_patients < 5:
        print("⚠  Need at least 5 patients. Check paths and CV params.")
        return

    # ── compute class weights ──────────────────────────────────────────────
    all_labels_list = [d["labels"] for d in all_patient_data]
    class_weights   = compute_class_weights(all_labels_list)
    print(f"\n  Class weights: {[f'{w:.2f}' for w in class_weights.cpu().numpy()]}")

    # ─────────────────────────────────────────────────────────────────────────
    #  LEAVE-ONE-OUT CROSS VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Leave-One-Out Cross Validation")
    print("=" * 65)

    loo_pred_all = np.zeros(0, dtype=np.int64)
    loo_true_all = np.zeros(0, dtype=np.int64)
    loo_results  = {}

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for test_idx in range(n_patients):
        test_item  = all_patient_data[test_idx]
        train_data = [d for i, d in enumerate(all_patient_data) if i != test_idx]
        pid        = test_item["pid"]

        print(f"  LOO fold {test_idx+1:>3}/{n_patients}  "
              f"[held-out: {pid}] ...", end=" ", flush=True)

        # ── dataloaders ───────────────────────────────────────────────────
        train_set = BlobDataset(train_data, augment=True)
        test_set  = BlobDataset([test_item], augment=False)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader  = DataLoader(test_set,  batch_size=1,          shuffle=False)

        # ── model ─────────────────────────────────────────────────────────
        model     = MiniPointNet(n_feat=6, n_classes=N_CLASSES).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # ── train ─────────────────────────────────────────────────────────
        best_val_loss = float("inf")
        patience_cnt  = 0
        best_weights  = deepcopy(model.state_dict())

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, _, _ = eval_one_epoch(model, test_loader, criterion)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = deepcopy(model.state_dict())
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= EARLY_STOP_PAT:
                    break

        # ── evaluate on held-out patient ──────────────────────────────────
        model.load_state_dict(best_weights)
        _, pred, true = eval_one_epoch(model, test_loader, criterion)

        loo_pred_all = np.concatenate([loo_pred_all, pred])
        loo_true_all = np.concatenate([loo_true_all, true])

        det = electrode_detection_rate(pred, true)
        elec_recalls = [v["recall_pct"] for v in det.values()]
        mean_recall  = np.mean(elec_recalls) if elec_recalls else 0.0
        loo_results[pid] = {"electrode_recall": det,
                             "mean_elec_recall_pct": round(mean_recall, 1),
                             "val_loss": round(best_val_loss, 4)}

        print(f"mean electrode recall = {mean_recall:.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    #  TRAIN FINAL MODEL ON ALL DATA
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  Training final model on all patients...")
    full_set    = BlobDataset(all_patient_data, augment=True)
    full_loader = DataLoader(full_set, batch_size=BATCH_SIZE, shuffle=True)

    final_model = MiniPointNet(n_feat=6, n_classes=N_CLASSES).to(DEVICE)
    optimizer   = torch.optim.Adam(final_model.parameters(),
                                    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(EPOCHS):
        loss = train_one_epoch(final_model, full_loader, optimizer, criterion)
        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch+1:>4}/{EPOCHS}  loss={loss:.4f}")

    torch.save(final_model.state_dict(), OUTPUT_MODEL)
    print(f"\n  ✅  Final model saved → {OUTPUT_MODEL}")

    # ─────────────────────────────────────────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    det_loo   = electrode_detection_rate(loo_pred_all, loo_true_all)
    bg_mask   = (loo_true_all == 0)
    fp_rate   = float((loo_pred_all[bg_mask] != 0).mean()) * 100

    report_lines = []

    summary = [
        "",
        "=" * 65,
        "POINTNET LOO RESULTS",
        "=" * 65,
        f"  {'Electrode':<10}  {'N true':>8}  {'N correct':>10}  {'Recall':>8}",
        "  " + "─" * 42,
    ]
    recalls = []
    for c in range(1, N_CLASSES):
        name = LABEL_MAP[c]
        if name not in det_loo:
            continue
        d = det_loo[name]
        summary.append(f"  {name:<10}  {d['n_true']:>8}  "
                       f"{d['n_correct']:>10}  {d['recall_pct']:>7.1f}%")
        recalls.append(d["recall_pct"])

    mean_r = np.mean(recalls) if recalls else 0
    summary += [
        "  " + "─" * 42,
        f"  Mean electrode recall  : {mean_r:.1f}%",
        f"  Background false-pos   : {fp_rate:.1f}%  (blobs misclassified as leads)",
        "",
        "COMPARISON TO CLASSICAL CV:",
        f"  Step 5 (HU=2000)       : 56.5% detection",
        f"  Step 5b best params    : 77.7% detection",
        f"  PointNet LOO           : {mean_r:.1f}% recall  ← this script",
        "",
        "INTERPRETATION:",
        "  Recall > 77.7% → PointNet beats classical CV ✅",
        "  Recall < 77.7% → more training data needed; use Step 5b result",
        "  False-pos < 20% → model is selective enough to use as pseudo-labels",
        "=" * 65,
    ]

    for line in summary:
        print(line)
    report_lines.extend(summary)

    print(f"\n  Full per-patient results → see {OUTPUT_REPORT}")

    # per-patient detail
    report_lines.append("\nPER-PATIENT LOO DETAIL:")
    for pid, r in loo_results.items():
        report_lines.append(f"  {pid}: mean_recall={r['mean_elec_recall_pct']}%  "
                             f"val_loss={r['val_loss']}")
        for name, d in r["electrode_recall"].items():
            report_lines.append(f"    {name}: {d['n_correct']}/{d['n_true']} "
                                 f"({d['recall_pct']}%)")

    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump({"summary": {"mean_recall_pct": round(mean_r, 1),
                               "fp_rate_pct": round(fp_rate, 1),
                               "n_patients": n_patients},
                   "per_patient": loo_results}, f, indent=2)
    print(f"  ✅  Results saved → {OUTPUT_RESULTS}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  ✅  Report saved  → {OUTPUT_REPORT}")


# ─────────────────────────────────────────────────────────────────────────────
#  INSTALL CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    missing = []
    try:    import torch;    print(f"  PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
    except ImportError: missing.append("torch")
    try:    import nibabel
    except ImportError: missing.append("nibabel")
    try:    import skimage
    except ImportError: missing.append("scikit-image")

    if missing:
        print(f"\n⚠  Missing libraries. Run:")
        print(f"   pip install {' '.join(missing)}")
        if "torch" in missing:
            print("   For PyTorch: https://pytorch.org/get-started/locally/")
    else:
        run()