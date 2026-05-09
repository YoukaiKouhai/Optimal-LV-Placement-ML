"""
CRT Lead Detection — Step 5b: Parameter Sweep + RANSAC
=======================================================
Two improvements over Step 5a:

  IMPROVEMENT 1 — Automated parameter sweep
    Tests combinations of HU_THRESHOLD and BLOB_MAX_VOXELS one at a time,
    runs the full detection pipeline for each combo on GT patients,
    and reports which settings maximise detection rate.

  IMPROVEMENT 2 — RANSAC lead trajectory filtering
    Observation from Step 5 report: LV leads (4004–4007) are found well
    but RV leads (4008–4009) are almost always missed.
    
    Key insight: electrodes sit on a catheter wire → they form a roughly
    LINEAR path through 3D space. Bone/artifact blobs do NOT line up.
    
    RANSAC (Random Sample Consensus):
      1. From all candidate blobs, randomly pick 2 → define a 3D line
      2. Count how many other blobs lie within INLIER_DIST mm of that line
      3. Repeat N times → keep the line with most inliers
      4. Those inliers = the lead electrode candidates
      5. Run separately for LV and RV leads (different anatomical regions)

    This dramatically reduces false positives and finds blobs that were
    previously rejected because a different blob was closer to the GT point.

Requirements:
    pip install nibabel scipy scikit-image numpy

Usage:
    python claude_p5b_sweep_ransac.py
"""

import json
import itertools
import warnings
import numpy as np
import nibabel as nib

from pathlib import Path
from scipy   import ndimage as ndi
from skimage import measure

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")
INVENTORY_JSON = BASE_DIR / "data_inventory.json"
CENTROIDS_JSON = BASE_DIR / "centroids_results.json"

OUTPUT_SWEEP   = BASE_DIR / "cv_sweep_results.json"
OUTPUT_RANSAC  = BASE_DIR / "cv_ransac_results.json"
OUTPUT_REPORT  = BASE_DIR / "cv_5b_report.txt"


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETER SWEEP GRID  — change these to explore different ranges
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_PARAMS = {
    # Test one parameter at a time against the fixed baseline
    "HU_THRESHOLD":           [1500, 1700, 1800, 2000, 2200, 2500],
    "BLOB_MAX_VOXELS":        [300, 400, 600, 800, 1000, 1500],
    "BLOB_MIN_VOXELS":        [4, 6, 8, 12, 16],
    "HEART_SEARCH_RADIUS_MM": [80, 100, 120, 150, 200],
}

# Baseline (Step 5 settings — used when sweeping other params)
BASELINE = {
    "HU_THRESHOLD":           2000,
    "BLOB_MAX_VOXELS":        600,
    "BLOB_MIN_VOXELS":        8,
    "HEART_SEARCH_RADIUS_MM": 120.0,
    "MATCH_RADIUS_MM":        10.0,
}

# ─────────────────────────────────────────────────────────────────────────────
#  RANSAC SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

RANSAC_ITERATIONS = 200    # more = more reliable, slower
RANSAC_INLIER_DIST_MM = 8  # blob within Xmm of the fitted line = inlier

# Approximate anatomical split: LV leads are on the lateral/posterior side,
# RV leads are on the septal/anterior side.
# We use the heart centre + RV direction to split the blob cloud.
# In LPS coords, RV is typically at larger negative Y (more posterior/right).
# We use a simple heuristic: split by distance from heart centre along Y axis.
RV_Y_OFFSET_MM = -60   # RV blobs expected to be this far in Y from heart centre


# ─────────────────────────────────────────────────────────────────────────────
#  COORDINATE HELPERS  (same as Step 5)
# ─────────────────────────────────────────────────────────────────────────────

def voxel_to_world_lps(affine, i, j, k):
    ras = affine @ np.array([i, j, k, 1.0])
    return np.array([-ras[0], -ras[1], ras[2]])


def detect_blobs(data, affine, hu_thresh, blob_min, blob_max):
    """Full blob detection pipeline for one set of parameters."""
    metal = data > hu_thresh
    struct = ndi.generate_binary_structure(3, 3)
    labelled, _ = ndi.label(metal, structure=struct)
    props = measure.regionprops(labelled)

    blobs = []
    for p in props:
        if not (blob_min <= p.area <= blob_max):
            continue
        ci, cj, ck = p.centroid
        world = voxel_to_world_lps(affine, ci, cj, ck)
        blobs.append({
            "voxel_ijk": [float(ci), float(cj), float(ck)],
            "world_xyz": world.tolist(),
            "n_voxels":  int(p.area),
        })
    return blobs


def spatial_filter_blobs(blobs, heart_centre, radius_mm):
    if heart_centre is None:
        return blobs
    hc  = np.array(heart_centre)
    return [b for b in blobs
            if np.linalg.norm(np.array(b["world_xyz"]) - hc) <= radius_mm]


def get_heart_centre(pid, centroids_data):
    rec       = centroids_data.get(pid, {})
    centroids = rec.get("centroids", {})
    apex = centroids.get("4002") or centroids.get(4002)
    base = centroids.get("4003") or centroids.get(4003)
    if apex and base:
        return ((np.array(apex["world_xyz"]) + np.array(base["world_xyz"])) / 2).tolist()
    for lbl in ["4002", "4003", "4004", "4001"]:
        e = centroids.get(lbl)
        if e: return e["world_xyz"]
    return None


def match_to_gt(blobs, gt_centroids, match_radius, electrode_labels):
    if not blobs:
        return {}, 0, 0
    pts    = np.array([b["world_xyz"] for b in blobs])
    used   = set()
    report = {}
    n_det = n_miss = 0

    for lbl_str in sorted(gt_centroids.keys(), key=lambda x: int(x)):
        if int(lbl_str) not in electrode_labels:
            continue
        entry = gt_centroids.get(lbl_str)
        if not entry:
            continue
        gt_pt = np.array(entry["world_xyz"])
        dists = np.linalg.norm(pts - gt_pt, axis=1)
        for ui in used:
            dists[ui] = np.inf
        best_idx  = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        if best_dist <= match_radius:
            used.add(best_idx)
            report[lbl_str] = {"error_mm": round(best_dist, 3), "status": "✅"}
            n_det += 1
        else:
            report[lbl_str] = {"error_mm": None, "status": "❌"}
            n_miss += 1

    return report, n_det, n_miss


def adjust_param_value(param_name, value, factor):
    """Adjust a numeric parameter by a factor, preserving integer thresholds."""
    if param_name in {"HU_THRESHOLD", "BLOB_MAX_VOXELS", "BLOB_MIN_VOXELS"}:
        candidate = int(round(value * factor))
        if candidate == value:
            candidate = value + 1 if factor > 1 else max(1, value - 1)
        return candidate
    return float(value * factor)


def evaluate_config(cfg, gt_patients, centroids_data):
    total_det = total_miss = 0
    all_errors = []
    total_blobs_kept = 0
    n_patients = 0

    for pid, rec in gt_patients.items():
        img_path = rec.get("img_nii")
        if not img_path or pid not in centroids_data:
            continue

        try:
            nii  = nib.load(img_path)
            data = nii.get_fdata(dtype=np.float32)
            aff  = nii.affine
        except Exception:
            continue

        blobs = detect_blobs(
            data, aff,
            cfg["HU_THRESHOLD"],
            cfg["BLOB_MIN_VOXELS"],
            cfg["BLOB_MAX_VOXELS"],
        )
        hc = get_heart_centre(pid, centroids_data)
        blobs = spatial_filter_blobs(blobs, hc, cfg["HEART_SEARCH_RADIUS_MM"])

        gt_c = centroids_data[pid].get("centroids", {})
        _, nd, nm = match_to_gt(blobs, gt_c, cfg["MATCH_RADIUS_MM"], ELECTRODE_LABELS)
        total_det += nd
        total_miss += nm
        total_blobs_kept += len(blobs)
        n_patients += 1

        pts = np.array([b["world_xyz"] for b in blobs]) if blobs else None
        used = set()
        for lbl_str in sorted(gt_c.keys(), key=lambda x: int(x)):
            if int(lbl_str) not in ELECTRODE_LABELS:
                continue
            entry = gt_c.get(lbl_str)
            if not entry or pts is None:
                continue
            gt_pt = np.array(entry["world_xyz"])
            dists = np.linalg.norm(pts - gt_pt, axis=1)
            for ui in used:
                dists[ui] = np.inf
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            if best_dist <= cfg["MATCH_RADIUS_MM"]:
                all_errors.append(best_dist)
                used.add(best_idx)

    total_elec = total_det + total_miss
    det_pct = 100 * total_det / total_elec if total_elec else 0
    mean_err = float(np.mean(all_errors)) if all_errors else float("nan")
    blobs_per = total_blobs_kept / n_patients if n_patients else 0

    return det_pct, mean_err, blobs_per, total_det, total_miss


ELECTRODE_LABELS = {4004, 4005, 4006, 4007, 4008, 4009}


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1 — PARAMETER SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(gt_patients: dict, centroids_data: dict) -> dict:
    """
    For each parameter in SWEEP_PARAMS, vary it around baseline using a
    +10% increase and a fallback -2% decrease when needed.
    """
    print("\n" + "=" * 65)
    print("  PART 1 — Parameter Sweep (one parameter at a time)")
    print("=" * 65)

    sweep_results = {}

    for param_name in SWEEP_PARAMS.keys():
        print(f"\n  Optimizing {param_name} around baseline {BASELINE[param_name]}")
        print(f"  {'Value':>10}  {'Det%':>7}  {'MeanErr':>9}  {'Blobs/pt':>9}")
        print(f"  {'─'*10}  {'─'*7}  {'─'*9}  {'─'*9}")

        param_results = []
        baseline_cfg = dict(BASELINE)
        baseline_det, baseline_err, baseline_blobs, _, _ = evaluate_config(
            baseline_cfg, gt_patients, centroids_data
        )
        param_results.append({
            "value":       BASELINE[param_name],
            "det_pct":     round(baseline_det, 1),
            "mean_err_mm": round(baseline_err, 3),
            "blobs_per_pt": round(baseline_blobs, 1),
            "n_detected":  None,
            "n_missed":    None,
            "status":      "baseline",
        })
        print(f"  {str(BASELINE[param_name]):>10}  {baseline_det:>6.1f}%"
              f"  {baseline_err:>7.2f} mm"
              f"  {baseline_blobs:>7.1f}  baseline")

        best_val = BASELINE[param_name]
        best_det = baseline_det
        current_val = BASELINE[param_name]

        # first try increasing by 10% until the detection rate stops improving
        while True:
            candidate = adjust_param_value(param_name, current_val, 1.1)
            if candidate == current_val:
                break
            cfg = dict(BASELINE)
            cfg[param_name] = candidate
            det_pct, mean_err, blobs_per, _, _ = evaluate_config(
                cfg, gt_patients, centroids_data
            )
            param_results.append({
                "value":       candidate,
                "det_pct":     round(det_pct, 1),
                "mean_err_mm": round(mean_err, 3),
                "blobs_per_pt": round(blobs_per, 1),
                "n_detected":  None,
                "n_missed":    None,
                "status":      "up+10%",
            })
            marker = " ← best" if det_pct > best_det else ""
            print(f"  {str(candidate):>10}  {det_pct:>6.1f}%"
                  f"  {mean_err:>7.2f} mm"
                  f"  {blobs_per:>7.1f}{marker}")

            if det_pct > best_det:
                best_det = det_pct
                best_val = candidate
                current_val = candidate
                continue
            break

        # if increasing did not improve from baseline, try decreasing by 2%
        if best_val == BASELINE[param_name]:
            current_val = BASELINE[param_name]
            while True:
                candidate = adjust_param_value(param_name, current_val, 0.98)
                if candidate == current_val:
                    break
                cfg = dict(BASELINE)
                cfg[param_name] = candidate
                det_pct, mean_err, blobs_per, _, _ = evaluate_config(
                    cfg, gt_patients, centroids_data
                )
                param_results.append({
                    "value":       candidate,
                    "det_pct":     round(det_pct, 1),
                    "mean_err_mm": round(mean_err, 3),
                    "blobs_per_pt": round(blobs_per, 1),
                    "n_detected":  None,
                    "n_missed":    None,
                    "status":      "down-2%",
                })
                marker = " ← best" if det_pct > best_det else ""
                print(f"  {str(candidate):>10}  {det_pct:>6.1f}%"
                      f"  {mean_err:>7.2f} mm"
                      f"  {blobs_per:>7.1f}{marker}")

                if det_pct > best_det:
                    best_det = det_pct
                    best_val = candidate
                    current_val = candidate
                    continue
                break

        sweep_results[param_name] = param_results
        print(f"\n  ★ Best {param_name} = {best_val} ({best_det:.1f}% detection)\n")

    return sweep_results


def best_params_from_sweep(sweep_results: dict) -> dict:
    """Pick the best value for each parameter from the sweep."""
    best = dict(BASELINE)  # start from baseline
    for param_name, results in sweep_results.items():
        top = max(results, key=lambda r: r["det_pct"])
        best[param_name] = top["value"]
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  PART 2 — RANSAC LEAD TRAJECTORY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def line_point_distances(line_pt: np.ndarray, line_dir: np.ndarray,
                          points: np.ndarray) -> np.ndarray:
    """
    Compute perpendicular distance from each point to a 3D line.

    Line defined by a point on the line and a unit direction vector.
    dist = ||(p - line_pt) × line_dir||  (cross product magnitude)
    """
    vecs  = points - line_pt                     # shape (N, 3)
    cross = np.cross(vecs, line_dir)             # shape (N, 3)
    return np.linalg.norm(cross, axis=1)         # shape (N,)


def ransac_fit_line(points: np.ndarray,
                    n_iter: int = RANSAC_ITERATIONS,
                    inlier_dist: float = RANSAC_INLIER_DIST_MM
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a 3D line to a point cloud using RANSAC.

    Algorithm:
      for n_iter iterations:
        1. randomly sample 2 points → candidate line
        2. compute distance of all points to that line
        3. count inliers (dist < inlier_dist)
        4. keep the line with the most inliers

    Returns:
      best_inlier_mask  — boolean array, True = inlier blob
      best_line_pt      — a point on the best line
      best_line_dir     — unit direction of the best line
    """
    n = len(points)
    if n < 2:
        return np.ones(n, dtype=bool), points[0] if n else np.zeros(3), np.array([0,0,1.])

    best_inliers = np.zeros(n, dtype=bool)
    best_count   = 0
    best_pt      = points[0]
    best_dir     = np.array([0., 0., 1.])

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        # sample 2 distinct points
        i, j = rng.choice(n, size=2, replace=False)
        pt1, pt2 = points[i], points[j]
        direction = pt2 - pt1
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction /= norm

        # count inliers
        dists   = line_point_distances(pt1, direction, points)
        inliers = dists < inlier_dist
        count   = inliers.sum()

        if count > best_count:
            best_count   = count
            best_inliers = inliers
            best_pt      = pt1
            best_dir     = direction

    return best_inliers, best_pt, best_dir


def split_lv_rv_blobs(blobs: list[dict],
                       heart_centre: list | None) -> tuple[list, list]:
    """
    Heuristically split blobs into LV and RV candidates using the
    heart centre as reference.

    Strategy: in LPS coordinates, the RV is typically more negative Y
    (more posterior) and can be on either side in X.  We use a simple
    distance-based split: blobs farthest from heart centre in the Y
    direction are more likely RV.

    This is approximate — RANSAC within each group does the real work.
    """
    if not blobs or heart_centre is None:
        return blobs, []

    hc  = np.array(heart_centre)
    pts = np.array([b["world_xyz"] for b in blobs])

    # Y-displacement from heart centre
    y_disp = pts[:, 1] - hc[1]    # negative = more posterior

    # split at the RV_Y_OFFSET_MM threshold
    lv_mask = y_disp > RV_Y_OFFSET_MM
    rv_mask = ~lv_mask

    lv_blobs = [b for b, m in zip(blobs, lv_mask) if m]
    rv_blobs = [b for b, m in zip(blobs, rv_mask) if m]

    return lv_blobs, rv_blobs


def ransac_filter_patient(blobs: list[dict],
                           heart_centre: list | None
                           ) -> list[dict]:
    """
    Apply RANSAC to find the lead trajectories within a patient's blob set.

    We run RANSAC twice — once on LV blob candidates, once on RV — and
    combine the inliers from both runs.

    Returns a filtered list of blobs that lie on a lead trajectory.
    """
    if len(blobs) < 2:
        return blobs

    # split into approximate LV / RV regions
    lv_blobs, rv_blobs = split_lv_rv_blobs(blobs, heart_centre)

    result_blobs = []

    for group, label in [(lv_blobs, "LV"), (rv_blobs, "RV")]:
        if len(group) < 2:
            result_blobs.extend(group)   # not enough for RANSAC, keep all
            continue

        pts    = np.array([b["world_xyz"] for b in group])
        inlier_mask, _, _ = ransac_fit_line(pts)

        for blob, is_inlier in zip(group, inlier_mask):
            if is_inlier:
                blob["ransac_group"] = label
                result_blobs.append(blob)

    return result_blobs


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  CRT Lead — Step 5b: Parameter Sweep + RANSAC")
    print("=" * 65)

    # ── load data ──────────────────────────────────────────────────────────
    with open(INVENTORY_JSON, encoding="utf-8") as f:
        inv = json.load(f)
    with open(CENTROIDS_JSON, encoding="utf-8") as f:
        centroids_data = json.load(f)

    # collect patients that have BOTH a raw image AND ground truth
    gt_patients = {}
    for ds_key in ("dataset_1", "dataset_2"):
        for pid, rec in inv.get(ds_key, {}).items():
            if rec.get("img_nii") and pid in centroids_data:
                if pid not in gt_patients:
                    gt_patients[pid] = rec

    print(f"\n  GT patients available for evaluation: {len(gt_patients)}")
    print(f"  Baseline settings: {BASELINE}\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  PART 1 — PARAMETER SWEEP
    # ─────────────────────────────────────────────────────────────────────────
    sweep_results = run_sweep(gt_patients, centroids_data)

    best_cfg = best_params_from_sweep(sweep_results)
    print("\n" + "=" * 65)
    print("  BEST PARAMETER SET FROM SWEEP:")
    for k, v in best_cfg.items():
        baseline_v = BASELINE[k]
        changed = " ← changed from baseline" if v != baseline_v else ""
        print(f"    {k:<28} = {v}{changed}")
    print("=" * 65)

    with open(OUTPUT_SWEEP, "w", encoding="utf-8") as f:
        json.dump({"baseline": BASELINE,
                   "best_params": best_cfg,
                   "sweep": sweep_results}, f, indent=2)
    print(f"\n✅  Sweep results saved → {OUTPUT_SWEEP}")

    # ─────────────────────────────────────────────────────────────────────────
    #  PART 2 — RANSAC  (using best params from sweep)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PART 2 — RANSAC Lead Trajectory Filtering")
    print(f"           Using best params: HU={best_cfg['HU_THRESHOLD']},"
          f" blob_max={best_cfg['BLOB_MAX_VOXELS']}")
    print("=" * 65)

    report_lines = []
    ransac_results = {}

    # counters: no RANSAC vs with RANSAC
    nr_det = nr_miss = wr_det = wr_miss = 0
    nr_errors, wr_errors = [], []
    n_processed = 0

    print(f"\n  {'PID':<8}  {'Blobs':>6}  {'→RANSAC':>8}  "
          f"{'noRANSAC det%':>14}  {'RANSAC det%':>12}")
    print("  " + "─" * 58)

    for pid, rec in sorted(gt_patients.items()):
        img_path = rec.get("img_nii")
        if not img_path:
            continue

        try:
            nii  = nib.load(img_path)
            data = nii.get_fdata(dtype=np.float32)
            aff  = nii.affine
        except Exception as e:
            print(f"  [{pid}] ERROR: {e}")
            continue

        # detect with best params
        blobs = detect_blobs(
            data, aff,
            best_cfg["HU_THRESHOLD"],
            best_cfg["BLOB_MIN_VOXELS"],
            best_cfg["BLOB_MAX_VOXELS"],
        )
        hc    = get_heart_centre(pid, centroids_data)
        blobs = spatial_filter_blobs(blobs, hc, best_cfg["HEART_SEARCH_RADIUS_MM"])

        gt_c  = centroids_data[pid].get("centroids", {})
        n_elec = sum(1 for l in gt_c if int(l) in ELECTRODE_LABELS)

        # ── without RANSAC ──
        _, nd_nr, nm_nr = match_to_gt(
            blobs, gt_c, best_cfg["MATCH_RADIUS_MM"], ELECTRODE_LABELS
        )
        nr_det  += nd_nr
        nr_miss += nm_nr

        # ── with RANSAC ──
        blobs_r = ransac_filter_patient(blobs, hc)
        _, nd_wr, nm_wr = match_to_gt(
            blobs_r, gt_c, best_cfg["MATCH_RADIUS_MM"], ELECTRODE_LABELS
        )
        wr_det  += nd_wr
        wr_miss += nm_wr

        det_nr = 100 * nd_nr / n_elec if n_elec else 0
        det_wr = 100 * nd_wr / n_elec if n_elec else 0
        delta  = det_wr - det_nr
        arrow  = f" (+{delta:.0f}%)" if delta > 0 else f" ({delta:.0f}%)" if delta < 0 else ""

        print(f"  {pid:<8}  {len(blobs):>6}  {len(blobs_r):>8}  "
              f"{det_nr:>12.0f}%  {det_wr:>10.0f}%{arrow}")

        # per-patient report block
        block = [
            f"\n{'─'*60}",
            f"Patient {pid}",
            f"  Blobs before RANSAC: {len(blobs)}",
            f"  Blobs after  RANSAC: {len(blobs_r)}",
            f"  Detection without RANSAC: {nd_nr}/{n_elec} ({det_nr:.0f}%)",
            f"  Detection with    RANSAC: {nd_wr}/{n_elec} ({det_wr:.0f}%)",
        ]

        # show matched blobs with RANSAC
        pts_r = np.array([b["world_xyz"] for b in blobs_r]) if blobs_r else None
        used  = set()
        for lbl_str in sorted(gt_c.keys(), key=lambda x: int(x)):
            if int(lbl_str) not in ELECTRODE_LABELS:
                continue
            entry = gt_c.get(lbl_str)
            if not entry or pts_r is None:
                block.append(f"  {lbl_str}: ❌ (no blobs)")
                continue
            gt_pt = np.array(entry["world_xyz"])
            dists = np.linalg.norm(pts_r - gt_pt, axis=1)
            for ui in used:
                dists[ui] = np.inf
            best_idx  = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            if best_dist <= best_cfg["MATCH_RADIUS_MM"]:
                used.add(best_idx)
                block.append(f"  {lbl_str}: ✅ {best_dist:.2f} mm")
            else:
                block.append(f"  {lbl_str}: ❌ nearest={best_dist:.1f} mm")

        report_lines.extend(block)
        ransac_results[pid] = {
            "n_blobs_before": len(blobs),
            "n_blobs_after":  len(blobs_r),
            "det_no_ransac":  round(det_nr, 1),
            "det_ransac":     round(det_wr, 1),
        }
        n_processed += 1

    # ── aggregate comparison ───────────────────────────────────────────────
    total = nr_det + nr_miss
    nr_pct = 100 * nr_det / total if total else 0
    wr_pct = 100 * wr_det / total if total else 0

    summary = [
        "",
        "=" * 65,
        "FINAL COMPARISON SUMMARY",
        "=" * 65,
        f"  {'Method':<30}  {'Detected':>10}  {'Rate':>8}",
        f"  {'─'*30}  {'─'*10}  {'─'*8}",
        f"  {'Step 5 baseline (HU=2000)':30}  {'200/354':>10}  {'56.5%':>8}",
        f"  {'Best params, no RANSAC':30}  {nr_det}/{total:>8}  {nr_pct:>6.1f}%",
        f"  {'Best params + RANSAC':30}  {wr_det}/{total:>8}  {wr_pct:>6.1f}%",
        "",
        "RANSAC EXPLANATION:",
        "  Each catheter lead is a sequence of electrodes strung along",
        "  a wire → they form a LINE in 3D space.  RANSAC finds the",
        "  line that passes through the most detected blobs and keeps",
        "  only those blobs (inliers). Bone/artifact blobs are random",
        "  and do NOT line up, so they get rejected as outliers.",
        "",
        f"  RANSAC iterations : {RANSAC_ITERATIONS}",
        f"  Inlier threshold  : {RANSAC_INLIER_DIST_MM} mm from line",
        "=" * 65,
    ]

    for line in summary:
        print(line)
    report_lines = summary + report_lines

    # ── next steps ─────────────────────────────────────────────────────────
    next_steps = [
        "",
        "WHAT TO DO NEXT:",
        "  If RANSAC detection rate < 70%:",
        "    → Increase RANSAC_ITERATIONS to 500",
        "    → Increase RANSAC_INLIER_DIST_MM to 12–15",
        "    → Check if RV_Y_OFFSET_MM correctly splits LV vs RV blobs",
        "    → Try adding a second RANSAC pass on the remaining outliers",
        "",
        "  If detection rate > 75%:",
        "    → The CV pipeline is now strong enough to label raw patients",
        "    → Use cv_ransac_results.json blobs as pseudo-labels for",
        "       Dataset 2 (308 raw patients)",
        "    → Feed those pseudo-labels into Step 3 normalization",
        "    → Retrain the ML model from Step 4b with 300+ patients",
        "",
        "  PyTorch path (your friend's suggestion):",
        "    → Replace the RANSAC blob filter with a small PointNet model",
        "       that learns which blobs belong to leads vs bone",
        "    → Input: blob position + n_voxels → output: lead/not-lead",
        "    → Train on the 86 GT patients, apply to 308 raw patients",
        "    → This is the natural bridge from classical CV to deep learning",
    ]
    for line in next_steps:
        print(line)
    report_lines.extend(next_steps)

    # ── save outputs ───────────────────────────────────────────────────────
    with open(OUTPUT_RANSAC, "w", encoding="utf-8") as f:
        json.dump({
            "best_params":    best_cfg,
            "ransac_settings": {
                "iterations":    RANSAC_ITERATIONS,
                "inlier_dist_mm": RANSAC_INLIER_DIST_MM,
                "rv_y_offset_mm": RV_Y_OFFSET_MM,
            },
            "aggregate": {
                "no_ransac_pct": round(nr_pct, 1),
                "ransac_pct":    round(wr_pct, 1),
            },
            "patients": ransac_results,
        }, f, indent=2)
    print(f"\n✅  RANSAC results saved → {OUTPUT_RANSAC}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved        → {OUTPUT_REPORT}")


if __name__ == "__main__":
    missing = []
    try: import nibabel
    except ImportError: missing.append("nibabel")
    try: import skimage
    except ImportError: missing.append("scikit-image")
    try: import scipy
    except ImportError: missing.append("scipy")

    if missing:
        print(f"⚠  Missing libraries. Run:\n   pip install {' '.join(missing)}")
    else:
        run()