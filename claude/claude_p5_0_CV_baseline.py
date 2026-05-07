"""
CRT Lead Detection Project — Step 5: Classical CV Baseline
===========================================================
Detects pacing lead electrodes in RAW CT scans (no segmentation needed)
using classical computer vision:

    1. Load raw CT (.nii.gz)  →  Hounsfield Unit volume
    2. HU threshold  (>2000 HU)  →  binary metal mask
    3. 3D connected components   →  individual blobs
    4. Blob filtering by volume  →  electrode candidates
       (electrodes ≈ 3x3x3 vox;  ribs/sternum are much larger)
    5. Spatial filtering         →  keep only blobs near the heart
       (uses apex/base from centroids_results.json when available,
        otherwise uses a coarse anatomical heuristic)
    6. Affine transform          →  world LPS mm coordinates
    7. Hungarian matching        →  pair candidates to known labels
       (for ground-truth patients only — to measure detection accuracy)
    8. Save results              →  cv_results.json + cv_report.txt

Why this matters
----------------
• 86 patients have segmentations (Steps 2-4 covered these).
• 308+ patients have only raw images — this script runs on ALL of them.
• The detection output becomes a non-ML scientific baseline AND
  provides candidate coordinates that could seed a future labelling pass.

Requirements
------------
    pip install nibabel scipy numpy pandas scikit-image

Usage
-----
    python claude_p5_classical_cv.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import nibabel as nib

from pathlib  import Path
from scipy    import ndimage as ndi
from skimage  import measure          # regionprops — blob statistics

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  — edit these to match your machine
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR         = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")
INVENTORY_JSON   = BASE_DIR / "data_inventory.json"
CENTROIDS_JSON   = BASE_DIR / "centroids_results.json"   # Step 2 output

OUTPUT_JSON      = BASE_DIR / "cv_results.json"
OUTPUT_REPORT    = BASE_DIR / "cv_report.txt"


# ─────────────────────────────────────────────────────────────────────────────
#  TUNABLE PARAMETERS  — adjust if detection is too noisy or misses leads
# ─────────────────────────────────────────────────────────────────────────────

# Metal threshold: pacing leads typically appear > 2000 HU.
# Ribs/sternum are cortical bone (~400–1800 HU).  Start at 2000 — lower
# to 1800 if leads are being missed; raise to 2500 if too many bone hits.
HU_THRESHOLD = 2000

# Electrode blob size limits (in voxels).
# The professor's ground truth uses 3×3×3 boxes = 27 voxels.
# In practice, partial-volume effects make real electrodes 10–80 voxels.
# Ribs / sternum fragments are usually > 500 voxels after thresholding.
BLOB_MIN_VOXELS =  8     # smaller = noise
BLOB_MAX_VOXELS = 600    # larger  = bone / metal artefact

# Spatial search radius around the known heart region (mm).
# Blobs further than this from the heart centre are discarded.
# Set to None to disable spatial filtering (slower but catches edge cases).
HEART_SEARCH_RADIUS_MM = 120.0

# For patients WITH ground truth: maximum distance (mm) to call a
# detected blob a "match" to a known electrode position.
MATCH_RADIUS_MM = 10.0

# Labels that correspond to lead electrodes (not anatomical markers)
ELECTRODE_LABELS = {4004, 4005, 4006, 4007, 4008, 4009}
ANCHOR_LABELS    = {4001, 4002, 4003}   # ANT, APEX, BASE


# ─────────────────────────────────────────────────────────────────────────────
#  COORDINATE HELPER  (same convention as Step 2)
# ─────────────────────────────────────────────────────────────────────────────

def voxel_to_world_lps(affine: np.ndarray, i: float, j: float, k: float
                        ) -> np.ndarray:
    """
    Voxel indices (i,j,k) → world LPS mm coordinates.
    nibabel affine gives RAS; negate X and Y to get Horos-compatible LPS.
    """
    ras   = affine @ np.array([i, j, k, 1.0])
    return np.array([-ras[0], -ras[1], ras[2]])


def world_to_voxel(affine: np.ndarray, lps_xyz: np.ndarray) -> np.ndarray:
    """LPS mm → voxel indices (i,j,k). Inverse of above."""
    ras_xyz = np.array([-lps_xyz[0], -lps_xyz[1], lps_xyz[2], 1.0])
    return (np.linalg.inv(affine) @ ras_xyz)[:3]


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — HU THRESHOLDING
# ─────────────────────────────────────────────────────────────────────────────

def threshold_metal(data: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of voxels that are likely metal (leads/electrodes).

    Note: CT data may be stored as float after get_fdata().  HU values for
    metal implants are typically clipped at the scanner maximum (~3071 HU)
    but the physical signal is far higher — they appear as saturated bright
    spots, often surrounded by beam-hardening streak artefacts.
    """
    return data > HU_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — 3D CONNECTED COMPONENTS + BLOB FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def find_blobs(metal_mask: np.ndarray, affine: np.ndarray) -> list[dict]:
    """
    Label connected components in the metal mask, filter by size,
    and return a list of blob dicts with centroid in both voxel and world space.

    Uses 26-connectivity (full 3D neighbourhood) so diagonal voxels
    within the same electrode cluster as one blob.
    """
    struct = ndi.generate_binary_structure(3, 3)   # 26-connectivity
    labelled, n_blobs = ndi.label(metal_mask, structure=struct)

    if n_blobs == 0:
        return []

    props = measure.regionprops(labelled)

    blobs = []
    for p in props:
        n_vox = p.area          # number of voxels in this blob

        if n_vox < BLOB_MIN_VOXELS or n_vox > BLOB_MAX_VOXELS:
            continue            # too small (noise) or too large (bone)

        # centroid in voxel space (i,j,k)
        ci, cj, ck = p.centroid
        world_lps  = voxel_to_world_lps(affine, ci, cj, ck)

        blobs.append({
            "voxel_ijk":  [float(ci), float(cj), float(ck)],
            "world_xyz":  world_lps.tolist(),
            "n_voxels":   int(n_vox),
            "bbox":       list(p.bbox),             # (min_i,min_j,min_k,max_i,...)
        })

    return blobs


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — SPATIAL FILTERING  (keep blobs near the heart)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_heart_centre(pid: str, centroids_data: dict) -> np.ndarray | None:
    """
    Return an approximate heart centre in LPS mm from the Step 2 centroids,
    or None if not available (raw-only patient).

    We use the midpoint of APEX (4002) and BASE (4003) as the heart centre.
    """
    rec = centroids_data.get(pid, {})
    centroids = rec.get("centroids", {})

    apex = (centroids.get("4002") or centroids.get(4002))
    base = (centroids.get("4003") or centroids.get(4003))

    if apex and base:
        a = np.array(apex["world_xyz"])
        b = np.array(base["world_xyz"])
        return (a + b) / 2.0

    # fallback: use any available centroid as rough centre
    for lbl in ["4002", "4003", "4004", "4001"]:
        entry = centroids.get(lbl)
        if entry:
            return np.array(entry["world_xyz"])

    return None   # raw-only patient — no anchor available


def spatial_filter(blobs: list[dict], heart_centre: np.ndarray | None,
                   radius_mm: float) -> list[dict]:
    """
    Remove blobs that are more than *radius_mm* away from *heart_centre*.
    If heart_centre is None (no ground truth anchor), return all blobs
    but flag them as unfiltered.
    """
    if heart_centre is None or radius_mm is None:
        for b in blobs:
            b["spatially_filtered"] = False
        return blobs

    kept = []
    for b in blobs:
        pt   = np.array(b["world_xyz"])
        dist = float(np.linalg.norm(pt - heart_centre))
        b["dist_from_heart_mm"] = round(dist, 2)
        if dist <= radius_mm:
            b["spatially_filtered"] = True
            kept.append(b)

    return kept


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — HUNGARIAN MATCHING  (ground-truth patients only)
# ─────────────────────────────────────────────────────────────────────────────

def match_blobs_to_ground_truth(blobs: list[dict],
                                 gt_centroids: dict,
                                 match_radius: float) -> dict:
    """
    For each known electrode label, find the closest detected blob.
    Returns a match report dict.

    Uses a greedy nearest-neighbour match (good enough for ≤9 electrodes).
    """
    if not blobs:
        return {}

    blob_pts   = np.array([b["world_xyz"] for b in blobs])
    matched    = set()    # blob indices already claimed
    report     = {}

    # sort electrodes so distal (LL1=4004) is matched first
    for lbl_str in sorted(gt_centroids.keys(), key=lambda x: int(x)):
        lbl_int = int(lbl_str)
        if lbl_int not in ELECTRODE_LABELS:
            continue

        entry = gt_centroids[lbl_str] or gt_centroids.get(lbl_int)
        if entry is None:
            continue

        gt_pt  = np.array(entry["world_xyz"])
        dists  = np.linalg.norm(blob_pts - gt_pt, axis=1)

        # mask out already-matched blobs
        for used_idx in matched:
            dists[used_idx] = np.inf

        best_idx  = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        if best_dist <= match_radius:
            matched.add(best_idx)
            report[lbl_str] = {
                "gt_xyz":     gt_pt.tolist(),
                "blob_xyz":   blobs[best_idx]["world_xyz"],
                "error_mm":   round(best_dist, 3),
                "n_voxels":   blobs[best_idx]["n_voxels"],
                "status":     "✅ DETECTED",
            }
        else:
            report[lbl_str] = {
                "gt_xyz":     gt_pt.tolist(),
                "blob_xyz":   None,
                "error_mm":   round(best_dist, 3) if best_dist < np.inf else None,
                "n_voxels":   None,
                "status":     "❌ MISSED",
            }

    return report


# ─────────────────────────────────────────────────────────────────────────────
#  PER-PATIENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_patient(pid: str, img_path: str,
                    centroids_data: dict) -> dict:
    """
    Run the full classical CV pipeline for one patient.
    Returns a result dict.
    """
    # ── load raw CT ────────────────────────────────────────────────────────
    nii    = nib.load(img_path)
    data   = nii.get_fdata(dtype=np.float32)
    affine = nii.affine

    # ── 1: threshold ────────────────────────────────────────────────────────
    metal_mask      = threshold_metal(data)
    n_metal_voxels  = int(metal_mask.sum())

    # ── 2: connected components + size filter ───────────────────────────────
    all_blobs = find_blobs(metal_mask, affine)

    # ── 3: spatial filter ───────────────────────────────────────────────────
    heart_centre = estimate_heart_centre(pid, centroids_data)
    blobs        = spatial_filter(all_blobs, heart_centre, HEART_SEARCH_RADIUS_MM)

    # ── 4: match to ground truth (if seg data available) ────────────────────
    gt_rec      = centroids_data.get(pid, {})
    gt_centroid = gt_rec.get("centroids", {})
    has_gt      = bool(gt_centroid)

    match_report = {}
    if has_gt:
        match_report = match_blobs_to_ground_truth(
            blobs, gt_centroid, MATCH_RADIUS_MM
        )

    return {
        "img_path":       img_path,
        "has_gt":         has_gt,
        "volume_shape":   list(data.shape),
        "n_metal_voxels": n_metal_voxels,
        "n_blobs_raw":    len(all_blobs),
        "n_blobs_kept":   len(blobs),
        "heart_centre":   heart_centre.tolist() if heart_centre is not None else None,
        "blobs":          blobs,
        "gt_match":       match_report,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  REPORTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def detection_summary(match_report: dict) -> tuple[int, int, list[float]]:
    """Returns (n_detected, n_missed, list_of_errors_mm)."""
    n_det, n_miss, errors = 0, 0, []
    for v in match_report.values():
        if v["status"] == "✅ DETECTED":
            n_det  += 1
            errors.append(v["error_mm"])
        else:
            n_miss += 1
    return n_det, n_miss, errors


def format_match_table(match_report: dict) -> list[str]:
    lines = []
    header = (f"  {'Label':<6}  {'GT XYZ':>26}  "
              f"{'Blob XYZ':>26}  {'Error':>8}  Status")
    lines.append(header)
    lines.append("  " + "─" * 80)
    for lbl, v in sorted(match_report.items(), key=lambda x: int(x[0])):
        gt   = f"({v['gt_xyz'][0]:6.1f},{v['gt_xyz'][1]:6.1f},{v['gt_xyz'][2]:6.1f})"
        blob = (f"({v['blob_xyz'][0]:6.1f},{v['blob_xyz'][1]:6.1f},{v['blob_xyz'][2]:6.1f})"
                if v["blob_xyz"] else f"{'— not found —':>26}")
        err  = f"{v['error_mm']:6.2f} mm" if v["error_mm"] is not None else "     N/A"
        lines.append(f"  {lbl:<6}  {gt}  {blob}  {err}  {v['status']}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  CRT Lead — Step 5: Classical CV Baseline")
    print("=" * 65)

    # ── load inventories ───────────────────────────────────────────────────
    with open(INVENTORY_JSON, encoding="utf-8") as f:
        inv = json.load(f)
    with open(CENTROIDS_JSON, encoding="utf-8") as f:
        centroids_data = json.load(f)

    # ── collect ALL patients with a raw image ──────────────────────────────
    # merge dataset_1 and dataset_2; skip duplicates
    all_patients: dict[str, dict] = {}
    for ds_key in ("dataset_1", "dataset_2"):
        for pid, rec in inv.get(ds_key, {}).items():
            if rec.get("img_nii") and pid not in all_patients:
                all_patients[pid] = rec

    gt_pids   = set(centroids_data.keys())
    raw_pids  = set(all_patients.keys()) - gt_pids

    print(f"\n  Total patients with raw CT : {len(all_patients)}")
    print(f"  With ground truth (Step 2) : {len(gt_pids & set(all_patients))}")
    print(f"  Raw-only (no seg)          : {len(raw_pids)}")
    print(f"\n  HU threshold   : {HU_THRESHOLD}")
    print(f"  Blob size range: {BLOB_MIN_VOXELS} – {BLOB_MAX_VOXELS} voxels")
    print(f"  Search radius  : {HEART_SEARCH_RADIUS_MM} mm around heart centre")
    print(f"  Match radius   : {MATCH_RADIUS_MM} mm (GT validation)\n")

    all_results  = {}
    report_lines = []

    # counters for aggregate stats
    agg_det = agg_miss = agg_gt_patients = 0
    all_errors: list[float] = []

    for idx, (pid, rec) in enumerate(sorted(all_patients.items()), 1):
        img_path = rec["img_nii"]
        print(f"[{idx:>3}/{len(all_patients)}]  {pid}  ", end="", flush=True)

        try:
            result = process_patient(pid, img_path, centroids_data)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[pid] = {"error": str(e)}
            continue

        n_kept = result["n_blobs_kept"]
        print(f"metal={result['n_metal_voxels']:>7} vox  "
              f"blobs_raw={result['n_blobs_raw']:>4}  "
              f"blobs_kept={n_kept:>3}", end="")

        # ── per-patient report ─────────────────────────────────────────────
        block = [
            f"\n{'─'*65}",
            f"Patient {pid}   "
            f"({'has GT' if result['has_gt'] else 'RAW ONLY'})",
            f"  Image : {img_path}",
            f"  Shape : {result['volume_shape']}",
            f"  Metal voxels (>{HU_THRESHOLD} HU) : {result['n_metal_voxels']}",
            f"  Blobs after size filter   : {result['n_blobs_raw']}",
            f"  Blobs after spatial filter: {result['n_blobs_kept']}",
        ]

        if result["has_gt"] and result["gt_match"]:
            n_det, n_miss, errors = detection_summary(result["gt_match"])
            agg_det  += n_det
            agg_miss += n_miss
            agg_gt_patients += 1
            all_errors.extend(errors)
            mean_err = np.mean(errors) if errors else float("nan")
            dr_str   = f"{n_det}/{n_det+n_miss}"
            print(f"  GT: {dr_str} detected  mean_err={mean_err:.2f} mm")
            block += [
                f"  Detection: {dr_str} electrodes found",
                f"  Mean error: {mean_err:.2f} mm",
            ]
            block += format_match_table(result["gt_match"])
        else:
            print(f"  (no GT — {n_kept} candidates stored)")
            block.append(f"  Candidate blobs (for future labelling):")
            for i, b in enumerate(result["blobs"][:10], 1):   # show first 10
                xyz = b["world_xyz"]
                block.append(
                    f"    [{i:>2}]  ({xyz[0]:7.1f}, {xyz[1]:7.1f}, {xyz[2]:7.1f}) mm"
                    f"  n_vox={b['n_voxels']}"
                )
            if len(result["blobs"]) > 10:
                block.append(f"    ... and {len(result['blobs'])-10} more")

        report_lines.extend(block)
        all_results[pid] = result

    # ── aggregate summary ──────────────────────────────────────────────────
    total_electrodes = agg_det + agg_miss
    det_rate = 100 * agg_det / total_electrodes if total_electrodes else 0
    mean_err_all = float(np.mean(all_errors)) if all_errors else float("nan")
    med_err_all  = float(np.median(all_errors)) if all_errors else float("nan")
    pct_under5   = (100 * sum(e < 5  for e in all_errors) / len(all_errors)
                    if all_errors else 0)
    pct_under10  = (100 * sum(e < 10 for e in all_errors) / len(all_errors)
                    if all_errors else 0)

    summary = [
        "",
        "=" * 65,
        "CLASSICAL CV DETECTION SUMMARY",
        "=" * 65,
        f"  GT patients evaluated  : {agg_gt_patients}",
        f"  Electrodes searched    : {total_electrodes}",
        f"  Detected (≤{MATCH_RADIUS_MM} mm)    : {agg_det}  ({det_rate:.1f}%)",
        f"  Missed               : {agg_miss}",
        f"  Mean position error  : {mean_err_all:.2f} mm",
        f"  Median error         : {med_err_all:.2f} mm",
        f"  % within  5 mm       : {pct_under5:.1f}%",
        f"  % within 10 mm       : {pct_under10:.1f}%",
        "",
        "HOW TO READ THIS:",
        "  Detection rate > 80% → threshold is well-tuned for your scanners",
        "  Mean error    < 5 mm → classical CV is a viable non-ML baseline",
        "  Many 'MISSED' with large errors → lower HU_THRESHOLD or increase",
        "                                    BLOB_MAX_VOXELS in the config",
        "  Blobs_kept >> 6 per patient     → lower HU_THRESHOLD or decrease",
        "                                    HEART_SEARCH_RADIUS_MM",
        "",
        "COMPARISON TO STEP 2 (segmentation centroid):",
        "  Step 2 accuracy : 100.0%  (uses expert seg — gold standard)",
        f"  Step 5 accuracy :  {det_rate:.1f}%  (no seg needed — CV baseline)",
        "  Gap = cost of removing the human segmentation step.",
        "=" * 65,
    ]

    for line in summary:
        print(line)
    report_lines = summary + report_lines

    # ── parameter tuning guide ─────────────────────────────────────────────
    tuning = [
        "",
        "PARAMETER TUNING GUIDE",
        "─" * 40,
        "If detection rate is LOW (< 70%):",
        "  → Lower HU_THRESHOLD to 1800 or 1500",
        "  → Increase BLOB_MAX_VOXELS to 1000",
        "  → Increase MATCH_RADIUS_MM to 15",
        "",
        "If too many false blobs per patient:",
        "  → Raise HU_THRESHOLD to 2500",
        "  → Lower BLOB_MAX_VOXELS to 300",
        "  → Decrease HEART_SEARCH_RADIUS_MM to 80",
        "",
        "Current settings:",
        f"  HU_THRESHOLD           = {HU_THRESHOLD}",
        f"  BLOB_MIN_VOXELS        = {BLOB_MIN_VOXELS}",
        f"  BLOB_MAX_VOXELS        = {BLOB_MAX_VOXELS}",
        f"  HEART_SEARCH_RADIUS_MM = {HEART_SEARCH_RADIUS_MM}",
        f"  MATCH_RADIUS_MM        = {MATCH_RADIUS_MM}",
    ]
    for line in tuning:
        print(line)
    report_lines.extend(tuning)

    # ── save ───────────────────────────────────────────────────────────────
    # blobs list contains numpy types — convert to plain Python for JSON
    def jsonify(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):return float(obj)
        return obj

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=jsonify)
    print(f"\n✅  Results saved → {OUTPUT_JSON}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved  → {OUTPUT_REPORT}")


# ─────────────────────────────────────────────────────────────────────────────
#  INSTALL CHECK
# ─────────────────────────────────────────────────────────────────────────────

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