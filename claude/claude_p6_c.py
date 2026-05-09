"""
CRT Lead Detection — Step 6c: Heart Centre Estimation + Option B
================================================================
PROBLEM DIAGNOSED FROM STEP 6b:
    Only 33/222 raw patients got any electrode detection.
    Root cause: raw patients have no segmentation, so heart_centre
    defaulted to np.zeros(3) — completely wrong world coordinates.
    This scrambled all normalized features before they reached PointNet.

THIS SCRIPT FIXES THAT IN TWO STEPS:

  STEP A — Heart Centre Estimation (image-based, no segmentation needed)
    Algorithm:
      1. Load raw CT
      2. Threshold at a soft-tissue level (~-100 to +200 HU) to find
         the body region
      3. Find the bounding box of the thorax (exclude table/air)
      4. Within the thorax, find the centroid of cardiac-density tissue
         (~20–80 HU — blood pool is ~40–60 HU)
      5. Refine using the metal blobs: the heart centre should be near
         the centre of mass of the detected lead candidates
      Fallback: use the scan midpoint if tissue estimation fails.

  STEP B — Re-run full inference pipeline on ALL patients
    Now that raw patients have a valid heart centre estimate,
    re-run blob detection → feature engineering → PointNet
    and report how many raw patients now get detections.

  OUTPUT:
    heart_centres.json      — estimated heart centres for all patients
    pseudo_labels_v2.json   — corrected pseudo-labels for all patients
    option_b_report.txt     — summary ready for Step 3 normalization

Usage:
    python claude_p6c_option_b.py
"""

import json
import warnings
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from scipy   import ndimage as ndi
from skimage import measure

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")
MODEL_PATH     = BASE_DIR / "pointnet_model.pt"
INVENTORY_JSON = BASE_DIR / "data_inventory.json"
CENTROIDS_JSON = BASE_DIR / "centroids_results.json"
SWEEP_JSON     = BASE_DIR / "cv_sweep_results.json"

OUTPUT_CENTRES = BASE_DIR / "heart_centres.json"
OUTPUT_PSEUDO  = BASE_DIR / "pseudo_labels_v2.json"
OUTPUT_REPORT  = BASE_DIR / "option_b_report.txt"
OUTPUT_PLOT    = BASE_DIR / "option_b_detection_summary.png"


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG  (must match Steps 5b / 6 / 6b)
# ─────────────────────────────────────────────────────────────────────────────

N_CLASSES       = 7
MAX_BLOBS       = 80
MATCH_RADIUS_MM = 10.0
BEST_THRESHOLD  = 0.60   # from Step 6b output — update if yours differed
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP    = {0: "background",
                1: "LL1", 2: "LL2", 3: "LL3", 4: "LL4",
                5: "RL1", 6: "RL2"}
SEG_TO_CLASS = {4004: 1, 4005: 2, 4006: 3, 4007: 4, 4008: 5, 4009: 6}

# HU windows for heart centre estimation
HU_BODY_MIN    = -200    # above air (-1000) and below bone (+200)
HU_BODY_MAX    =  200
HU_CARDIAC_MIN =   20    # blood pool / myocardium
HU_CARDIAC_MAX =  100

# Fraction of image (axial) to search for heart
# Heart sits in the middle third of the thorax height
THORAX_FRAC_LO = 0.25   # skip top 25% (above heart)
THORAX_FRAC_HI = 0.75   # skip bottom 25% (below heart)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  (identical to Steps 6 / 6b)
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
        self.seg1   = SharedMLP(512, 256)
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
        s1 = self.drop(self.seg1(torch.cat([l3, g], dim=1)))
        s2 = self.seg2(s1)
        s3 = self.seg3(s2)
        return self.out(s3).transpose(2, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  STEP A — HEART CENTRE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def voxel_to_lps(affine, i, j, k):
    ras = affine @ np.array([i, j, k, 1.0])
    return np.array([-ras[0], -ras[1], ras[2]])


def estimate_heart_centre_from_image(data: np.ndarray,
                                      affine: np.ndarray) -> np.ndarray:
    """
    Estimate heart centre in LPS mm from a raw CT volume.

    Strategy (3 methods, fallback chain):
      1. Cardiac HU window (20–100 HU) centre of mass in middle thorax slices
      2. Body tissue COM in middle slices
      3. Geometric centre of the scan volume

    Returns LPS mm coordinates of the estimated heart centre.
    """
    n_slices = data.shape[0]   # first dim is slice index in nibabel

    # restrict to middle third of axial range (where heart lives)
    lo_slice = int(n_slices * THORAX_FRAC_LO)
    hi_slice = int(n_slices * THORAX_FRAC_HI)
    roi      = data[lo_slice:hi_slice, :, :]

    # ── Method 1: cardiac-density tissue ────────────────────────────────────
    cardiac_mask = (roi >= HU_CARDIAC_MIN) & (roi <= HU_CARDIAC_MAX)
    if cardiac_mask.sum() > 1000:   # enough tissue found
        com = ndi.center_of_mass(cardiac_mask.astype(np.float32))
        # com is (dim0, dim1, dim2) within roi → add slice offset
        ci = com[0] + lo_slice
        cj, ck = com[1], com[2]
        return voxel_to_lps(affine, ci, cj, ck)

    # ── Method 2: broad body tissue COM ─────────────────────────────────────
    body_mask = (roi >= HU_BODY_MIN) & (roi <= HU_BODY_MAX)
    if body_mask.sum() > 5000:
        com = ndi.center_of_mass(body_mask.astype(np.float32))
        ci  = com[0] + lo_slice
        cj, ck = com[1], com[2]
        return voxel_to_lps(affine, ci, cj, ck)

    # ── Method 3: geometric centre of scan ──────────────────────────────────
    ci = n_slices / 2
    cj = data.shape[1] / 2
    ck = data.shape[2] / 2
    return voxel_to_lps(affine, ci, cj, ck)


def refine_with_blobs(estimate: np.ndarray,
                       blobs: list,
                       max_refine_dist_mm: float = 80.0) -> np.ndarray:
    """
    Refine the heart centre estimate using detected metal blobs.

    Electrodes (true leads) cluster near the heart. Take the centroid
    of all blobs within max_refine_dist_mm of the initial estimate.
    This pulls the centre toward where the leads actually are.
    """
    if not blobs:
        return estimate

    pts   = np.array([b["world_xyz"] for b in blobs])
    dists = np.linalg.norm(pts - estimate, axis=1)
    near  = pts[dists < max_refine_dist_mm]

    if len(near) < 2:
        return estimate

    return near.mean(axis=0)


def get_heart_centre_for_patient(pid: str,
                                  centroids_data: dict,
                                  data: np.ndarray,
                                  affine: np.ndarray,
                                  blobs: list) -> tuple[np.ndarray, str]:
    """
    Get the best available heart centre for a patient.

    Priority:
      1. GT centroid (apex+base midpoint) — most accurate
      2. Image-based estimation + blob refinement — for raw patients
    Returns (centre_lps, method_used).
    """
    # ── GT: use apex/base midpoint ───────────────────────────────────────────
    cent   = centroids_data.get(pid, {}).get("centroids", {})
    apex_e = cent.get("4002") or cent.get(4002)
    base_e = cent.get("4003") or cent.get(4003)

    if apex_e and base_e:
        apex = np.array(apex_e["world_xyz"])
        base = np.array(base_e["world_xyz"])
        return (apex + base) / 2.0, "gt_centroid"

    # ── RAW: estimate from image ─────────────────────────────────────────────
    estimate = estimate_heart_centre_from_image(data, affine)
    refined  = refine_with_blobs(estimate, blobs)
    return refined, "image_estimated"


def get_axis_len(pid: str, centroids_data: dict) -> float:
    """Heart axis length from GT if available, else population mean."""
    cent   = centroids_data.get(pid, {}).get("centroids", {})
    apex_e = cent.get("4002") or cent.get(4002)
    base_e = cent.get("4003") or cent.get(4003)
    if apex_e and base_e:
        a = np.array(apex_e["world_xyz"])
        b = np.array(base_e["world_xyz"])
        return float(np.linalg.norm(b - a))
    return 87.3   # population mean from Step 3


# ─────────────────────────────────────────────────────────────────────────────
#  BLOB DETECTION  (same as Steps 5b / 6 / 6b)
# ─────────────────────────────────────────────────────────────────────────────

def detect_blobs_raw(data, affine, hu, bmin, bmax):
    metal    = data > hu
    struct   = ndi.generate_binary_structure(3, 3)
    labelled, _ = ndi.label(metal, structure=struct)
    props    = measure.regionprops(labelled)
    blobs    = []
    for p in props:
        if not (bmin <= p.area <= bmax):
            continue
        ci, cj, ck = p.centroid
        ras = affine @ np.array([ci, cj, ck, 1.0])
        lps = np.array([-ras[0], -ras[1], ras[2]])
        blobs.append({"world_xyz": lps.tolist(), "n_voxels": int(p.area)})
    return blobs


def spatial_filter(blobs, hc, radius_mm):
    if hc is None:
        return blobs
    return [b for b in blobs
            if np.linalg.norm(np.array(b["world_xyz"]) - hc) <= radius_mm]


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING  (same as Steps 6 / 6b)
# ─────────────────────────────────────────────────────────────────────────────

def build_blob_features(blobs, hc, axis_len):
    if not blobs:
        return np.zeros((0, 6), dtype=np.float32)
    pts    = np.array([b["world_xyz"] for b in blobs], dtype=np.float32)
    nvox   = np.array([b["n_voxels"]  for b in blobs], dtype=np.float32)
    scale  = max(axis_len, 1.0)
    pts_n  = (pts - hc) / scale
    dist   = np.linalg.norm(pts_n, axis=1, keepdims=True)
    radial = np.linalg.norm(pts_n[:, :2], axis=1, keepdims=True)
    nvox_n = (np.log1p(nvox) / np.log1p(1000)).reshape(-1, 1)
    return np.hstack([pts_n, dist, nvox_n, radial]).astype(np.float32)


def pad_or_truncate(feats, max_n):
    n   = len(feats)
    n_f = feats.shape[1] if n > 0 else 6
    if n >= max_n:
        return feats[:max_n], np.ones(max_n, dtype=np.float32)
    pad_n = max_n - n
    mask  = np.array([1]*n + [0]*pad_n, dtype=np.float32)
    return (np.vstack([feats, np.zeros((pad_n, n_f), dtype=np.float32)]),
            mask)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, feats_np, threshold):
    model.eval()
    t      = torch.from_numpy(feats_np).unsqueeze(0).to(DEVICE)
    logits = model(t)
    probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    argmax = probs.argmax(axis=1)
    maxconf= probs.max(axis=1)
    pred   = np.where((argmax > 0) & (maxconf >= threshold), argmax, 0)
    return pred, maxconf, probs


def collect_detections(blobs, pred, conf):
    """
    Build electrode detection dict from blob predictions.
    If the same electrode class is predicted for multiple blobs,
    keep the one with the highest confidence.
    """
    detections = {}
    for blob, cls, confidence in zip(blobs, pred, conf):
        if cls == 0:
            continue
        name = LABEL_MAP[cls]
        if name not in detections or confidence > detections[name]["confidence"]:
            detections[name] = {
                "world_xyz":  blob["world_xyz"],
                "confidence": round(float(confidence), 3),
                "n_voxels":   blob["n_voxels"],
            }
    return detections


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_plot(results: dict):
    """Bar chart comparing GT vs RAW detection rates per electrode."""
    gt_recs  = [r for r in results.values() if r["type"] == "GT"]
    raw_recs = [r for r in results.values() if r["type"] == "RAW"]

    electrode_names = ["LL1", "LL2", "LL3", "LL4", "RL1", "RL2"]

    def detection_rate(recs, name):
        found = sum(1 for r in recs if name in r["detections"])
        return 100 * found / len(recs) if recs else 0

    gt_rates  = [detection_rate(gt_recs,  n) for n in electrode_names]
    raw_rates = [detection_rate(raw_recs, n) for n in electrode_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Step 6c — Option B: Pseudo-Label Detection Summary\n"
                 "Fixed heart centre estimation for raw patients",
                 fontsize=13, fontweight="bold")

    x   = np.arange(len(electrode_names))
    w   = 0.35
    ax  = axes[0]
    ax.bar(x - w/2, gt_rates,  w, label="GT patients",  color="#4e79a7", alpha=0.9)
    ax.bar(x + w/2, raw_rates, w, label="RAW patients", color="#f28e2b", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(electrode_names)
    ax.set_ylabel("% patients with detection")
    ax.set_title("Detection Rate per Electrode Class")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.axhline(77.7, color="red", linestyle="--", lw=1,
               label="CV baseline 77.7%")
    ax.grid(axis="y", alpha=0.3)

    # confidence distribution for raw patients
    ax2 = axes[1]
    all_confs = []
    for r in raw_recs:
        for name, d in r["detections"].items():
            all_confs.append(d["confidence"])

    if all_confs:
        ax2.hist(all_confs, bins=20, color="#59a14f", alpha=0.85,
                 edgecolor="white")
        ax2.axvline(BEST_THRESHOLD, color="red", linestyle="--", lw=1.5,
                    label=f"Threshold = {BEST_THRESHOLD}")
        ax2.set_xlabel("Prediction confidence")
        ax2.set_ylabel("Count")
        ax2.set_title("Confidence Distribution\n(RAW patient detections)")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No RAW detections", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title("Confidence Distribution (RAW)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✅  Plot saved → {OUTPUT_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print(f"  CRT Lead — Step 6c: Heart Centre Fix + Option B")
    print(f"  Device: {DEVICE}  |  Threshold: {BEST_THRESHOLD}")
    print("=" * 65)

    # ── load ──────────────────────────────────────────────────────────────
    model = MiniPointNet(n_feat=6, n_classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"\n  ✅  Model loaded")

    with open(INVENTORY_JSON,  encoding="utf-8") as f: inventory  = json.load(f)
    with open(CENTROIDS_JSON,  encoding="utf-8") as f: centroids  = json.load(f)
    with open(SWEEP_JSON,      encoding="utf-8") as f: sweep_data = json.load(f)

    best_params = sweep_data.get("best_params", {
        "HU_THRESHOLD": 2928, "BLOB_MIN_VOXELS": 7,
        "BLOB_MAX_VOXELS": 879, "HEART_SEARCH_RADIUS_MM": 120.0,
    })

    # collect all patients
    all_patients: dict[str, str] = {}   # pid → img_path
    for ds_key in ("dataset_1", "dataset_2"):
        for pid, rec in inventory.get(ds_key, {}).items():
            if rec.get("img_nii") and pid not in all_patients:
                all_patients[pid] = rec["img_nii"]

    print(f"\n  Total patients: {len(all_patients)}")
    print(f"  GT (have seg) : {sum(1 for p in all_patients if p in centroids)}")
    print(f"  RAW (no seg)  : {sum(1 for p in all_patients if p not in centroids)}")

    # ─────────────────────────────────────────────────────────────────────
    #  PROCESS ALL PATIENTS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Processing all patients with fixed heart centre estimation...")
    print("=" * 65)

    results       = {}
    heart_centres = {}
    report_lines  = []

    n_gt_det = n_gt_miss = 0
    n_raw_with_det = n_raw_no_det = 0
    gt_errors = []

    print(f"\n  {'PID':<8}  {'Type':<6}  {'HC method':<16}  "
          f"{'Blobs':>6}  {'Detected':>9}  {'Electrodes'}")
    print("  " + "─" * 68)

    for pid in sorted(all_patients.keys()):
        img_path = all_patients[pid]
        is_gt    = pid in centroids

        # ── load CT ───────────────────────────────────────────────────────
        try:
            nii  = nib.load(img_path)
            data = nii.get_fdata(dtype=np.float32)
            aff  = nii.affine
        except Exception as e:
            print(f"  {pid:<8}  ERROR loading: {e}")
            continue

        # ── blob detection ─────────────────────────────────────────────────
        blobs_all = detect_blobs_raw(
            data, aff,
            best_params.get("HU_THRESHOLD",    2928),
            best_params.get("BLOB_MIN_VOXELS", 7),
            best_params.get("BLOB_MAX_VOXELS", 879),
        )

        # ── heart centre (THE FIX) ─────────────────────────────────────────
        hc, hc_method = get_heart_centre_for_patient(
            pid, centroids, data, aff, blobs_all
        )

        heart_centres[pid] = {
            "method":     hc_method,
            "centre_lps": hc.tolist(),
        }

        # ── spatial filter with correct heart centre ───────────────────────
        blobs = spatial_filter(
            blobs_all, hc,
            best_params.get("HEART_SEARCH_RADIUS_MM", 120.0)
        )

        axis_len = get_axis_len(pid, centroids)

        # ── features → PointNet ───────────────────────────────────────────
        feats    = build_blob_features(blobs, hc, axis_len)
        if len(feats) == 0:
            results[pid] = {"type": "GT" if is_gt else "RAW",
                            "detections": {}, "validation": {},
                            "hc_method": hc_method}
            print(f"  {pid:<8}  {'GT' if is_gt else 'RAW':<6}  "
                  f"{hc_method:<16}  {'0':>6}  no blobs")
            continue

        f_p, m_p = pad_or_truncate(feats, MAX_BLOBS)
        pred, conf, _ = predict(model, f_p, BEST_THRESHOLD)

        n_real     = len(blobs)
        detections = collect_detections(blobs, pred[:n_real], conf[:n_real])

        # ── GT validation ─────────────────────────────────────────────────
        validation = {}
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
                    status  = "✅" if err <= MATCH_RADIUS_MM else "❌"
                    validation[name] = {"gt_xyz": entry["world_xyz"],
                                        "err_mm": round(err, 2),
                                        "status": status}
                    if status == "✅":
                        n_gt_det  += 1
                        gt_errors.append(err)
                    else:
                        n_gt_miss += 1
                else:
                    validation[name] = {"gt_xyz": entry["world_xyz"],
                                        "err_mm": None,
                                        "status": "❌ missed"}
                    n_gt_miss += 1

        results[pid] = {
            "type":       "GT" if is_gt else "RAW",
            "n_blobs":    len(blobs),
            "hc_method":  hc_method,
            "detections": detections,
            "validation": validation,
            "axis_len_mm": round(axis_len, 2),
            "heart_centre_lps": hc.tolist(),
        }

        if not is_gt:
            if detections:
                n_raw_with_det += 1
            else:
                n_raw_no_det += 1

        elec_str = ", ".join(sorted(detections.keys())) or "—"
        print(f"  {pid:<8}  {'GT' if is_gt else 'RAW':<6}  "
              f"{hc_method:<16}  {len(blobs):>6}  "
              f"{len(detections):>5} elec  {elec_str}")

    # ─────────────────────────────────────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    n_gt_total = n_gt_det + n_gt_miss
    gt_det_pct = 100 * n_gt_det / n_gt_total if n_gt_total else 0

    n_raw_total = n_raw_with_det + n_raw_no_det
    raw_det_pct = 100 * n_raw_with_det / n_raw_total if n_raw_total else 0

    summary = [
        "",
        "=" * 65,
        "OPTION B SUMMARY — Fixed Heart Centre Estimation",
        "=" * 65,
        "",
        "GT PATIENTS (validation):",
        f"  Electrode detections : {n_gt_det}/{n_gt_total} ({gt_det_pct:.1f}%)",
        f"  Mean position error  : {np.mean(gt_errors):.2f} mm" if gt_errors else "",
        f"  Median error         : {np.median(gt_errors):.2f} mm" if gt_errors else "",
        f"  Errors < 5mm         : {100*sum(e<5 for e in gt_errors)/len(gt_errors):.1f}%" if gt_errors else "",
        "",
        "RAW PATIENTS (pseudo-labels for Option B):",
        f"  Total raw patients   : {n_raw_total}",
        f"  With ≥1 detection    : {n_raw_with_det} ({raw_det_pct:.1f}%)",
        f"  No detections        : {n_raw_no_det}",
        "",
        "PIPELINE PROGRESSION:",
        f"  Step 5  HU only          : 56.5% GT detection",
        f"  Step 5b Tuned params     : 77.7% GT detection",
        f"  Step 6  PointNet thresh  : 89.0% GT recall (31.6% FP)",
        f"  Step 6b Conf threshold   : 74.9% GT detect  (8.6% FP) [broken HC]",
        f"  Step 6c Fixed HC         : {gt_det_pct:.1f}% GT detect  "
        f"← this script (RAW: {raw_det_pct:.1f}%)",
        "",
        "NEXT STEPS FOR OPTION B:",
        f"  1. pseudo_labels_v2.json has world_xyz for ~{n_raw_with_det} raw patients",
        "  2. Feed those coords into Step 3 normalize_coords.py",
        "     (you need apex/base — these come from the image-estimated HC,",
        "      so Step 3 normalization will be approximate for raw patients)",
        "  3. Add the normalized [t, angle] features to your Step 4b dataset",
        "  4. Retrain — more data usually helps even if labels are noisy",
        "",
        "NOTE ON APPROXIMATE NORMALIZATION:",
        "  Raw patients have estimated HC, not true apex/base from a human.",
        "  Their [t, angle] values will have ~5-10mm uncertainty.",
        "  Label this data as 'pseudo' in your ML training so you can",
        "  ablate its effect (train with vs without pseudo data).",
        "=" * 65,
    ]

    for line in summary:
        print(line)
    report_lines.extend(summary)

    # ── save ──────────────────────────────────────────────────────────────
    with open(OUTPUT_CENTRES, "w", encoding="utf-8") as f:
        json.dump(heart_centres, f, indent=2)
    print(f"\n✅  Heart centres saved  → {OUTPUT_CENTRES}")

    with open(OUTPUT_PSEUDO, "w", encoding="utf-8") as f:
        json.dump({"threshold": BEST_THRESHOLD,
                   "patients":  results}, f, indent=2)
    print(f"✅  Pseudo-labels saved  → {OUTPUT_PSEUDO}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved         → {OUTPUT_REPORT}")

    make_summary_plot(results)


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
        print(f"⚠  pip install {' '.join(missing)}")
    else:
        run()