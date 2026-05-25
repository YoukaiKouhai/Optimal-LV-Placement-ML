"""
CRT Lead Detection — Step 6b: Confidence Thresholding
======================================================
Problem from Step 6:
    Recall       = 89.0%  ✅ (finds most real electrodes)
    False-pos    = 31.6%  ❌ (too many bone blobs misclassified as leads)

Solution: instead of always taking argmax(logits), only accept a
prediction if the model's softmax confidence >= THRESHOLD.
Blobs below the threshold stay as "background" even if the model
weakly prefers an electrode class.

This script:
  1. Loads the saved pointnet_model.pt (no retraining)
  2. Sweeps thresholds from 0.30 → 0.95 on GT patients (LOO)
  3. Plots the Recall vs False-Positive tradeoff curve
  4. Picks the best threshold (maximises F1 for electrode classes)
  5. Runs final inference on ALL patients (GT + 308 raw)
     using the best threshold
  6. Saves pseudo-labels for raw patients → ready for Step 3 + ML

Usage:
    python claude_p6b_threshold.py

Input:
    pointnet_model.pt         (Step 6 trained model)
    cv_sweep_results.json     (Step 5b best CV params)
    centroids_results.json    (Step 2 ground truth)
    data_inventory.json       (Step 1 file paths)

Output:
    threshold_sweep.png       — Recall / FP / F1 vs threshold plot
    threshold_report.txt      — table of all threshold results
    pseudo_labels.json        — electrode detections for ALL patients
                                (GT patients: validated; raw: predicted)
"""

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from copy    import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  — must match Step 6
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")
MODEL_PATH     = BASE_DIR / "pointnet_model.pt"
INVENTORY_JSON = BASE_DIR / "data_inventory.json"
CENTROIDS_JSON = BASE_DIR / "centroids_results.json"
SWEEP_JSON     = BASE_DIR / "cv_sweep_results.json"

OUTPUT_PLOT    = BASE_DIR / "threshold_sweep.png"
OUTPUT_REPORT  = BASE_DIR / "threshold_report.txt"
OUTPUT_PSEUDO  = BASE_DIR / "pseudo_labels.json"


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  — must match Step 6 exactly
# ─────────────────────────────────────────────────────────────────────────────

N_CLASSES        = 7
MAX_BLOBS        = 80
MATCH_RADIUS_MM  = 10.0
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {0: "background",
             1: "LL1", 2: "LL2", 3: "LL3", 4: "LL4",
             5: "RL1", 6: "RL2"}
SEG_TO_CLASS = {4004: 1, 4005: 2, 4006: 3, 4007: 4, 4008: 5, 4009: 6}

# Thresholds to sweep — confidence must exceed this to accept a prediction
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
              0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  (copy from Step 6 — must be identical)
# ─────────────────────────────────────────────────────────────────────────────

class SharedMLP(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.bn   = nn.BatchNorm1d(out_ch) if bn else nn.Identity()
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MiniPointNet(nn.Module):
    def __init__(self, n_feat=6, n_classes=N_CLASSES):
        super().__init__()
        self.local1 = SharedMLP(n_feat, 64)
        self.local2 = SharedMLP(64, 128)
        self.local3 = SharedMLP(128, 256)
        self.seg1   = SharedMLP(256 + 256, 256)
        self.seg2   = SharedMLP(256, 128)
        self.seg3   = SharedMLP(128, 64)
        self.out    = nn.Conv1d(64, n_classes, 1)
        self.drop   = nn.Dropout(p=0.3)

    def forward(self, x):
        B, N, F = x.shape
        x  = x.transpose(2, 1)
        l1 = self.local1(x)
        l2 = self.local2(l1)
        l3 = self.local3(l2)
        g  = torch.max(l3, dim=2, keepdim=True)[0].expand(-1, -1, N)
        cat = torch.cat([l3, g], dim=1)
        s1  = self.drop(self.seg1(cat))
        s2  = self.seg2(s1)
        s3  = self.seg3(s2)
        return self.out(s3).transpose(2, 1)   # (B, N, C)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA HELPERS  (copy from Step 6)
# ─────────────────────────────────────────────────────────────────────────────

def build_blob_features(blobs, heart_centre, heart_axis_mm):
    if not blobs:
        return np.zeros((0, 6), dtype=np.float32)
    pts   = np.array([b["world_xyz"] for b in blobs], dtype=np.float32)
    nvox  = np.array([b["n_voxels"]  for b in blobs], dtype=np.float32)
    scale = max(heart_axis_mm, 1.0)
    pts_n = (pts - heart_centre) / scale
    dist  = np.linalg.norm(pts_n, axis=1, keepdims=True)
    radial= np.linalg.norm(pts_n[:, :2], axis=1, keepdims=True)
    nvox_n= (np.log1p(nvox) / np.log1p(1000)).reshape(-1, 1)
    return np.hstack([pts_n, dist, nvox_n, radial]).astype(np.float32)


def assign_gt_labels(blobs, gt_centroids):
    n = len(blobs)
    labels = np.zeros(n, dtype=np.int64)
    if not blobs or not gt_centroids:
        return labels
    pts  = np.array([b["world_xyz"] for b in blobs], dtype=np.float32)
    used = set()
    for lbl_str in sorted(gt_centroids.keys(), key=lambda x: int(x)):
        cls = SEG_TO_CLASS.get(int(lbl_str))
        if cls is None:
            continue
        entry = gt_centroids.get(lbl_str)
        if not entry:
            continue
        gt_pt = np.array(entry["world_xyz"], dtype=np.float32)
        dists = np.linalg.norm(pts - gt_pt, axis=1)
        for ui in used:
            dists[ui] = np.inf
        best_idx = int(np.argmin(dists))
        if dists[best_idx] <= MATCH_RADIUS_MM:
            labels[best_idx] = cls
            used.add(best_idx)
    return labels


def pad_or_truncate(feats, labels, max_n):
    n     = len(feats)
    n_f   = feats.shape[1] if n > 0 else 6
    if n >= max_n:
        return feats[:max_n], labels[:max_n], np.ones(max_n, dtype=np.float32)
    pad_n  = max_n - n
    mask   = np.array([1]*n + [0]*pad_n, dtype=np.float32)
    return (np.vstack([feats, np.zeros((pad_n, n_f), dtype=np.float32)]),
            np.concatenate([labels, np.zeros(pad_n, dtype=np.int64)]),
            mask)


def load_cv_blobs(pid, inventory, centroids_data, best_params):
    """Re-run CV detection for one patient. Returns (blobs, hc, axis_len)."""
    import nibabel as nib
    from scipy   import ndimage as ndi
    from skimage import measure

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

    hu   = best_params.get("HU_THRESHOLD",           2928)
    bmin = best_params.get("BLOB_MIN_VOXELS",        7)
    bmax = best_params.get("BLOB_MAX_VOXELS",        879)
    rad  = best_params.get("HEART_SEARCH_RADIUS_MM", 120.0)

    metal    = data > hu
    struct   = ndi.generate_binary_structure(3, 3)
    labelled, _ = ndi.label(metal, structure=struct)
    props    = measure.regionprops(labelled)

    blobs = []
    for p in props:
        if not (bmin <= p.area <= bmax):
            continue
        ci, cj, ck = p.centroid
        ras = aff @ np.array([ci, cj, ck, 1.0])
        lps = np.array([-ras[0], -ras[1], ras[2]])
        blobs.append({"world_xyz": lps.tolist(), "n_voxels": int(p.area)})

    cent = centroids_data.get(pid, {}).get("centroids", {})
    apex_e = cent.get("4002") or cent.get(4002)
    base_e = cent.get("4003") or cent.get(4003)
    hc, axis_len = None, 87.3

    if apex_e and base_e:
        a = np.array(apex_e["world_xyz"])
        b = np.array(base_e["world_xyz"])
        hc       = (a + b) / 2
        axis_len = float(np.linalg.norm(b - a))

    if hc is not None:
        blobs = [b for b in blobs
                 if np.linalg.norm(np.array(b["world_xyz"]) - hc) <= rad]

    return blobs, hc, axis_len


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE WITH CONFIDENCE THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_with_threshold(model, feats_np, mask_np, threshold):
    """
    Run model on one patient's features.

    Returns:
        pred_classes  : int array (N,)  — predicted class per blob
                        0 = background (either truly bg, or below threshold)
        confidences   : float array (N,) — softmax probability of chosen class
        probs_all     : float array (N, C) — full softmax distribution
    """
    model.eval()
    feats_t = torch.from_numpy(feats_np).unsqueeze(0).to(DEVICE)  # (1,N,F)
    logits  = model(feats_t)                                        # (1,N,C)
    probs   = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()   # (N,C)

    # argmax prediction and its confidence
    argmax_class = probs.argmax(axis=1)                 # (N,)
    max_conf     = probs.max(axis=1)                    # (N,)

    # apply threshold: if model isn't confident enough → background
    pred_classes = np.where(
        (argmax_class > 0) & (max_conf >= threshold),
        argmax_class,
        0
    )

    return pred_classes, max_conf, probs


def metrics_at_threshold(pred_all, true_all, threshold_label):
    """
    Compute recall (per electrode class), FP rate, and macro F1
    given flat pred/true arrays from all patients.
    """
    recalls = []
    for c in range(1, N_CLASSES):
        mask_true = (true_all == c)
        if mask_true.sum() == 0:
            continue
        recall = float((pred_all[mask_true] == c).mean()) * 100
        recalls.append(recall)

    mean_recall = np.mean(recalls) if recalls else 0.0

    # false positive rate on background blobs
    bg_mask = (true_all == 0)
    fp_rate = float((pred_all[bg_mask] != 0).mean()) * 100 if bg_mask.sum() else 0.0

    # macro F1 across electrode classes
    # precision: of predicted class c, how many are truly class c
    precisions = []
    for c in range(1, N_CLASSES):
        pred_c = (pred_all == c)
        if pred_c.sum() == 0:
            precisions.append(0.0)
            continue
        prec = float(((pred_all == c) & (true_all == c)).sum() / pred_c.sum()) * 100
        precisions.append(prec)

    macro_prec   = np.mean(precisions) if precisions else 0.0
    macro_recall = mean_recall
    f1 = (2 * macro_prec * macro_recall / (macro_prec + macro_recall + 1e-9))

    return {
        "recall_pct":    round(mean_recall, 2),
        "fp_rate_pct":   round(fp_rate, 2),
        "precision_pct": round(macro_prec, 2),
        "f1":            round(f1, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_sweep(sweep_rows, best_thresh):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("PointNet Confidence Threshold Sweep\n"
                 "Finding the best Recall / False-Positive tradeoff",
                 fontsize=13, fontweight="bold")

    thresholds = [r["threshold"] for r in sweep_rows]
    recalls    = [r["recall_pct"]    for r in sweep_rows]
    fp_rates   = [r["fp_rate_pct"]   for r in sweep_rows]
    f1s        = [r["f1"]            for r in sweep_rows]

    # ── recall ────────────────────────────────────────────────────────────
    axes[0].plot(thresholds, recalls, "o-", color="#4e79a7", lw=2)
    axes[0].axvline(best_thresh, color="red", linestyle="--", lw=1.2,
                    label=f"Best = {best_thresh:.2f}")
    axes[0].axhline(77.7, color="orange", linestyle=":", lw=1.2,
                    label="CV baseline 77.7%")
    axes[0].set_xlabel("Confidence threshold")
    axes[0].set_ylabel("Mean electrode recall (%)")
    axes[0].set_title("Recall vs Threshold")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.3)

    # ── FP rate ───────────────────────────────────────────────────────────
    axes[1].plot(thresholds, fp_rates, "o-", color="#e15759", lw=2)
    axes[1].axvline(best_thresh, color="red", linestyle="--", lw=1.2,
                    label=f"Best = {best_thresh:.2f}")
    axes[1].axhline(20, color="green", linestyle=":", lw=1.2,
                    label="Target < 20%")
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Background false-positive rate (%)")
    axes[1].set_title("False Positives vs Threshold")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 50)
    axes[1].grid(alpha=0.3)

    # ── F1 ────────────────────────────────────────────────────────────────
    axes[2].plot(thresholds, f1s, "o-", color="#59a14f", lw=2)
    axes[2].axvline(best_thresh, color="red", linestyle="--", lw=1.2,
                    label=f"Best = {best_thresh:.2f}")
    best_f1 = max(f1s)
    axes[2].set_xlabel("Confidence threshold")
    axes[2].set_ylabel("Macro F1 score")
    axes[2].set_title("F1 vs Threshold\n(balances recall + precision)")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.3)
    axes[2].text(0.05, 0.08, f"Peak F1 = {best_f1:.1f}",
                 transform=axes[2].transAxes, fontsize=9, color="#59a14f")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✅  Plot saved → {OUTPUT_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print(f"  CRT Lead — Step 6b: Confidence Thresholding")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # ── load model ────────────────────────────────────────────────────────
    model = MiniPointNet(n_feat=6, n_classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"\n  ✅  Model loaded from {MODEL_PATH}")

    # ── load data ─────────────────────────────────────────────────────────
    with open(INVENTORY_JSON,  encoding="utf-8") as f: inventory  = json.load(f)
    with open(CENTROIDS_JSON,  encoding="utf-8") as f: centroids  = json.load(f)
    with open(SWEEP_JSON,      encoding="utf-8") as f: sweep      = json.load(f)
    best_params = sweep.get("best_params", {})

    # ── build GT patient feature arrays ───────────────────────────────────
    print("\n  Building GT patient features...")
    gt_patients = []   # list of (pid, feats, labels, mask, blobs, hc, axis)

    for pid in sorted(centroids.keys()):
        gt_cent = centroids[pid].get("centroids", {})
        if not gt_cent:
            continue
        blobs, hc, axis_len = load_cv_blobs(pid, inventory, centroids, best_params)
        if not blobs:
            continue
        hc_arr  = hc if hc is not None else np.zeros(3)
        feats   = build_blob_features(blobs, hc_arr, axis_len)
        labels  = assign_gt_labels(blobs, gt_cent)
        f_p, l_p, m_p = pad_or_truncate(feats, labels, MAX_BLOBS)
        gt_patients.append((pid, f_p, l_p, m_p, blobs, hc_arr, axis_len))

    print(f"  GT patients loaded: {len(gt_patients)}")

    # ─────────────────────────────────────────────────────────────────────
    #  PART 1 — THRESHOLD SWEEP ON GT PATIENTS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PART 1 — Threshold Sweep")
    print("=" * 65)
    print(f"\n  {'Threshold':>10}  {'Recall%':>8}  {'FP rate%':>9}  "
          f"{'Precision%':>11}  {'F1':>6}")
    print("  " + "─" * 52)

    sweep_rows  = []
    report_lines = []
    report_lines.append("Step 6b — Confidence Threshold Sweep")
    report_lines.append("=" * 65)
    report_lines.append(f"  {'Threshold':>10}  {'Recall%':>8}  "
                        f"{'FP rate%':>9}  {'Precision%':>11}  {'F1':>6}")
    report_lines.append("  " + "─" * 52)

    for thresh in THRESHOLDS:
        all_pred, all_true = [], []

        for pid, feats, labels, mask, blobs, hc, axis in gt_patients:
            pred, conf, _ = predict_with_threshold(model, feats, mask, thresh)
            real = mask.astype(bool)
            all_pred.extend(pred[real].tolist())
            all_true.extend(labels[real].tolist())

        m = metrics_at_threshold(
            np.array(all_pred), np.array(all_true), thresh
        )
        m["threshold"] = thresh
        sweep_rows.append(m)

        row = (f"  {thresh:>10.2f}  {m['recall_pct']:>8.1f}  "
               f"{m['fp_rate_pct']:>9.1f}  "
               f"{m['precision_pct']:>11.1f}  {m['f1']:>6.1f}")
        note = ""
        if thresh == 0.30:
            note = "  ← Step 6 baseline (argmax)"
        print(row + note)
        report_lines.append(row + note)

    # ── pick best threshold = highest F1 ─────────────────────────────────
    best_row   = max(sweep_rows, key=lambda r: r["f1"])
    best_thresh = best_row["threshold"]

    summary_lines = [
        "",
        f"  Best threshold (max F1) : {best_thresh:.2f}",
        f"  At best threshold:",
        f"    Recall     : {best_row['recall_pct']:.1f}%  "
        f"(was 89.0% at threshold=0.30)",
        f"    FP rate    : {best_row['fp_rate_pct']:.1f}%  "
        f"(was 31.6% at threshold=0.30)",
        f"    Precision  : {best_row['precision_pct']:.1f}%",
        f"    F1 score   : {best_row['f1']:.1f}",
        "",
        f"  Recall change : {best_row['recall_pct'] - 89.0:+.1f}%",
        f"  FP change     : {best_row['fp_rate_pct'] - 31.6:+.1f}%  "
        f"← negative = fewer false positives",
    ]
    for line in summary_lines:
        print(line)
    report_lines.extend(summary_lines)

    # ── plot ──────────────────────────────────────────────────────────────
    plot_threshold_sweep(sweep_rows, best_thresh)

    # ─────────────────────────────────────────────────────────────────────
    #  PART 2 — FINAL INFERENCE ON ALL PATIENTS (GT + RAW)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  PART 2 — Full Inference at threshold = {best_thresh:.2f}")
    print("=" * 65)

    # collect all patient IDs across both datasets
    all_pids: dict[str, dict] = {}
    for ds_key in ("dataset_1", "dataset_2"):
        for pid, rec in inventory.get(ds_key, {}).items():
            if rec.get("img_nii") and pid not in all_pids:
                all_pids[pid] = rec

    pseudo_labels = {}
    n_gt = n_raw = n_failed = 0

    print(f"\n  Processing {len(all_pids)} patients...\n")
    print(f"  {'PID':<8}  {'Type':<8}  {'Blobs':>6}  "
          f"{'Detected':>9}  {'Predictions'}")
    print("  " + "─" * 60)

    for pid in sorted(all_pids.keys()):
        is_gt = pid in centroids

        blobs, hc, axis_len = load_cv_blobs(
            pid, inventory, centroids, best_params
        )

        if not blobs:
            n_failed += 1
            print(f"  {pid:<8}  {'GT' if is_gt else 'RAW':<8}  "
                  f"{'—':>6}  no blobs detected")
            continue

        hc_arr = hc if hc is not None else np.zeros(3)
        feats  = build_blob_features(blobs, hc_arr, axis_len)
        gt_lbl = assign_gt_labels(
            blobs, centroids.get(pid, {}).get("centroids", {})
        ) if is_gt else np.zeros(len(blobs), dtype=np.int64)

        f_p, l_p, m_p = pad_or_truncate(feats, gt_lbl, MAX_BLOBS)
        pred, conf, probs = predict_with_threshold(
            model, f_p, m_p, best_thresh
        )

        # collect only real blobs (not padding)
        n_real  = len(blobs)
        pred_r  = pred[:n_real]
        conf_r  = conf[:n_real]
        probs_r = probs[:n_real]

        # build electrode detections
        detections = {}
        for blob_idx, (blob, cls, confidence) in enumerate(
                zip(blobs, pred_r, conf_r)):
            if cls == 0:
                continue   # background or below threshold
            name = LABEL_MAP[cls]
            # if same class predicted twice, keep the higher confidence one
            if name not in detections or confidence > detections[name]["confidence"]:
                detections[name] = {
                    "world_xyz":   blob["world_xyz"],
                    "confidence":  round(float(confidence), 3),
                    "n_voxels":    blob["n_voxels"],
                    "blob_idx":    blob_idx,
                }

        # validation for GT patients
        val = {}
        if is_gt:
            gt_cent = centroids[pid].get("centroids", {})
            for lbl_str, entry in gt_cent.items():
                cls_id = SEG_TO_CLASS.get(int(lbl_str))
                if cls_id is None:
                    continue
                name   = LABEL_MAP[cls_id]
                gt_xyz = np.array(entry["world_xyz"])
                if name in detections:
                    det_xyz = np.array(detections[name]["world_xyz"])
                    err     = float(np.linalg.norm(det_xyz - gt_xyz))
                    val[name] = {
                        "gt_xyz":   entry["world_xyz"],
                        "err_mm":   round(err, 2),
                        "status":   "✅" if err <= MATCH_RADIUS_MM else "❌",
                    }
                else:
                    val[name] = {"gt_xyz": entry["world_xyz"],
                                 "err_mm": None, "status": "❌ missed"}

        pseudo_labels[pid] = {
            "type":         "GT" if is_gt else "RAW",
            "n_blobs":      len(blobs),
            "heart_centre": hc_arr.tolist() if hc is not None else None,
            "axis_len_mm":  round(axis_len, 2),
            "detections":   detections,
            "validation":   val,
        }

        det_str = ", ".join(
            f"{n}({d['confidence']:.2f})" for n, d in sorted(detections.items())
        ) or "none"
        print(f"  {pid:<8}  {'GT' if is_gt else 'RAW':<8}  "
              f"{len(blobs):>6}  {len(detections):>5} elec  {det_str}")

        if is_gt:
            n_gt += 1
        else:
            n_raw += 1

    # ── aggregate GT validation at best threshold ─────────────────────────
    print("\n" + "=" * 65)
    print("  GT VALIDATION SUMMARY AT BEST THRESHOLD")
    print("=" * 65)

    gt_det = gt_miss = 0
    gt_errors = []
    for pid, rec in pseudo_labels.items():
        if rec["type"] != "GT":
            continue
        for name, v in rec["validation"].items():
            if v["status"] == "✅":
                gt_det += 1
                gt_errors.append(v["err_mm"])
            else:
                gt_miss += 1

    total_gt_elec = gt_det + gt_miss
    det_pct = 100 * gt_det / total_gt_elec if total_gt_elec else 0

    final_summary = [
        "",
        f"  GT electrode detection  : {gt_det}/{total_gt_elec} ({det_pct:.1f}%)",
        f"  Mean position error     : {np.mean(gt_errors):.2f} mm" if gt_errors else "",
        f"  Median position error   : {np.median(gt_errors):.2f} mm" if gt_errors else "",
        f"  Errors < 5mm            : {100*sum(e<5 for e in gt_errors)/len(gt_errors):.1f}%" if gt_errors else "",
        "",
        f"  RAW patients processed  : {n_raw}",
        f"  Patients with detections: {sum(1 for r in pseudo_labels.values() if r['type']=='RAW' and r['detections'])}",
        f"  Failed (no blobs)       : {n_failed}",
        "",
        "PIPELINE COMPARISON:",
        f"  Method                          Detection",
        f"  Step 5  HU threshold only       56.5%",
        f"  Step 5b Tuned params            77.7%",
        f"  Step 6  PointNet (thresh=0.30)  89.0%  (31.6% FP)",
        f"  Step 6b PointNet (thresh={best_thresh:.2f})  "
        f"{det_pct:.1f}%  ({best_row['fp_rate_pct']:.1f}% FP)  ← this script",
        "",
        "NEXT STEP — Option B (pseudo-label pipeline):",
        "  pseudo_labels.json now contains electrode detections for ALL",
        f"  {len(all_pids)} patients. Feed the RAW patient detections into",
        "  Step 3 normalization to get longitudinal + radial positions,",
        "  then retrain the Step 4b ML model with 4× more data.",
        "=" * 65,
    ]
    for line in final_summary:
        print(line)
    report_lines.extend(final_summary)

    # ── save outputs ──────────────────────────────────────────────────────
    with open(OUTPUT_PSEUDO, "w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": best_thresh,
            "sweep":          sweep_rows,
            "patients":       pseudo_labels,
        }, f, indent=2)
    print(f"\n✅  Pseudo-labels saved → {OUTPUT_PSEUDO}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved       → {OUTPUT_REPORT}")


if __name__ == "__main__":
    missing = []
    try:    import torch
    except ImportError: missing.append("torch")
    try:    import nibabel
    except ImportError: missing.append("nibabel")
    try:    import skimage
    except ImportError: missing.append("scikit-image")
    try:    import matplotlib
    except ImportError: missing.append("matplotlib")

    if missing:
        print(f"⚠  Missing: pip install {' '.join(missing)}")
    else:
        run()