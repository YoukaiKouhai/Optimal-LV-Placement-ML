"""
CRT Lead Detection Project — Step 2: Centroid Extractor
========================================================
For every patient with a segmentation file, this script:
  1. Loads the seg .nii.gz
  2. Extracts the centroid (center of mass) for each label 4001–4008
  3. Applies the affine matrix → world coords in mm
  4. Loads the matching _leads.csv (if present) for validation
  5. Computes Euclidean distance between auto vs manual coords
  6. Saves all results to centroids_results.json + centroids_report.txt

Requirements:
    pip install nibabel scipy numpy pandas

Usage:
    python claude_p2_extract_centroids.py
"""

import json
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.ndimage import center_of_mass


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INVENTORY_JSON  = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\data_inventory.json")
OUTPUT_JSON     = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\centroids_results.json")
OUTPUT_REPORT   = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\centroids_report.txt")

# Label definitions
# NOTE: verified by comparing auto centroids to Horos CSV clicks.
# The actual seg file label assignments are shifted from the original description:
#   4001 = ANT  (anterior wall marker, not a lead)
#   4002 = APEX
#   4003 = BASE
#   4004 = LL1  (LV distal)
#   ...
LABEL_MAP = {
    4001: "ANT",           # Anterior wall reference marker (not a lead)
    4002: "Apex",
    4003: "Base",
    4004: "LV_distal",     # LL1
    4005: "LV_2",          # LL2
    4006: "LV_3",          # LL3
    4007: "LV_proximal",   # LL4
    4008: "RV_distal",     # RL1
    4009: "RV_proximal",   # RL2
}

# How CSV landmark names map to seg labels (validated from data)
CSV_TO_LABEL = {
    "ANT":  4001,
    "APEX": 4002,
    "BASE": 4003,
    "LL1":  4004,
    "LL2":  4005,
    "LL3":  4006,
    "LL4":  4007,
    "RL1":  4008,
    "RL2":  4009,
}

# Validation threshold: < 2mm = success
SUCCESS_THRESHOLD_MM = 2.0


# ─────────────────────────────────────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def voxel_to_world(affine: np.ndarray, com_i: float, com_j: float, com_k: float) -> np.ndarray:
    """
    Convert center-of-mass voxel indices to world mm coords in LPS space
    (the same system Horos uses when saving click-point CSVs).

    COORDINATE SYSTEM:
    - nibabel affine → RAS (Right, Anterior, Superior) — NIfTI standard.
    - Horos CSV      → LPS (Left, Posterior, Superior)  — DICOM standard.
    - Conversion: negate X and Y (Z/Superior is identical in both).
    """
    voxel_homog  = np.array([com_i, com_j, com_k, 1.0])
    world_ras    = affine @ voxel_homog        # shape (4,), RAS + homog
    world_lps    = world_ras[:3].copy()
    world_lps[0] = -world_ras[0]              # Right  → Left
    world_lps[1] = -world_ras[1]              # Anterior → Posterior
    return world_lps


def extract_centroids(seg_path: str) -> dict:
    """
    Load a segmentation NIfTI, find all labels 4001–4008,
    compute their world-space centroids.

    Returns a dict:  { label_int: {"name": str, "world_xyz": [X, Y, Z]} }
    Labels not found in the volume are skipped.
    """
    nii   = nib.load(seg_path)
    data  = nii.get_fdata(dtype=np.float32)   # shape: (Z, Y, X) or (X, Y, Z)?
    aff   = nii.affine                         # 4x4 float64

    # Quick sanity: print shape once for debugging
    # print(f"  Volume shape: {data.shape}, affine:\n{aff}")

    results = {}
    for label, name in LABEL_MAP.items():
        mask = (data == label)
        voxel_count = int(mask.sum())

        if voxel_count == 0:
            # Label not present in this scan — skip silently
            continue

        # scipy returns (dim0, dim1, dim2) — pass directly, no reordering
        com_i, com_j, com_k = center_of_mass(mask)
        world_xyz = voxel_to_world(aff, com_i, com_j, com_k)

        results[label] = {
            "name":        name,
            "voxel_count": voxel_count,
            "voxel_ijk":   [float(com_i), float(com_j), float(com_k)],
            "world_xyz":   [float(world_xyz[0]),
                            float(world_xyz[1]),
                            float(world_xyz[2])],
        }

    return results


def load_csv_landmarks(csv_path: str) -> dict:
    """
    Load a headerless _leads.csv and return a dict:
       { "LL1": [X, Y, Z], "APEX": [X, Y, Z], ... }
    """
    df = pd.read_csv(csv_path, header=None, names=["name", "X", "Y", "Z"])
    landmarks = {}
    for _, row in df.iterrows():
        key = str(row["name"]).strip().upper()
        landmarks[key] = [float(row["X"]), float(row["Y"]), float(row["Z"])]
    return landmarks


def euclidean_distance(p1: list, p2: list) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def validate_against_csv(centroids: dict, csv_landmarks: dict) -> dict:
    """
    Compare auto-detected centroids to manual CSV clicks.
    Returns per-label validation results.
    """
    validation = {}

    for csv_name, label_id in CSV_TO_LABEL.items():
        csv_upper = csv_name.upper()
        if csv_upper not in csv_landmarks:
            continue                        # this landmark wasn't in the CSV
        if label_id not in centroids:
            validation[csv_name] = {
                "csv_xyz":    csv_landmarks[csv_upper],
                "auto_xyz":   None,
                "error_mm":   None,
                "status":     "MISSING — label not found in seg",
            }
            continue

        auto_xyz = centroids[label_id]["world_xyz"]
        csv_xyz  = csv_landmarks[csv_upper]
        err      = euclidean_distance(auto_xyz, csv_xyz)

        validation[csv_name] = {
            "csv_xyz":  csv_xyz,
            "auto_xyz": auto_xyz,
            "error_mm": round(err, 3),
            "status":   "✅ PASS" if err < SUCCESS_THRESHOLD_MM else "❌ FAIL",
        }

    return validation


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  CRT Lead — Step 2: Centroid Extractor")
    print("=" * 60)

    # Load the inventory built in Step 1
    with open(INVENTORY_JSON, encoding="utf-8") as f:
        inv = json.load(f)

    all_results   = {}
    report_lines  = []
    n_pass = n_fail = n_missing = 0

    # ── iterate over ALL patients that have a seg file ───────────────────────
    # Pull from both datasets; prefer records with a seg_nii
    all_patients = {}
    for ds_key in ("dataset_1", "dataset_2"):
        for pid, rec in inv.get(ds_key, {}).items():
            if rec.get("seg_nii"):
                # If patient appears in both datasets, dataset_1 (smaller, curated) wins
                if pid not in all_patients:
                    all_patients[pid] = rec

    print(f"\nProcessing {len(all_patients)} patients with segmentation files...\n")

    for pid in sorted(all_patients.keys()):
        rec      = all_patients[pid]
        seg_path = rec["seg_nii"]
        csv_path = rec.get("leads_csv")

        print(f"[{pid}] seg: {Path(seg_path).name}", end="")

        try:
            centroids = extract_centroids(seg_path)
        except Exception as e:
            print(f"  ⚠  ERROR loading seg: {e}")
            all_results[pid] = {"error": str(e)}
            continue

        labels_found = list(centroids.keys())
        print(f"  → {len(labels_found)} labels found: {labels_found}")

        patient_result = {
            "seg_path":    seg_path,
            "csv_path":    csv_path,
            "centroids":   centroids,
            "validation":  None,
        }

        # ── validation (only if CSV exists) ──────────────────────────────────
        if csv_path:
            try:
                csv_landmarks = load_csv_landmarks(csv_path)
                validation    = validate_against_csv(centroids, csv_landmarks)
                patient_result["validation"] = validation
                patient_result["csv_landmarks"] = csv_landmarks

                # report per patient
                report_lines.append(f"\n{'─'*55}")
                report_lines.append(f"Patient {pid}")
                report_lines.append(f"{'─'*55}")
                for lm_name, v in validation.items():
                    err_str = f"{v['error_mm']:.2f} mm" if v["error_mm"] is not None else "N/A"
                    report_lines.append(
                        f"  {lm_name:<6}  auto={_fmt(v['auto_xyz'])}  "
                        f"csv={_fmt(v['csv_xyz'])}  "
                        f"err={err_str}  {v['status']}"
                    )
                    if v["error_mm"] is not None:
                        if v["error_mm"] < SUCCESS_THRESHOLD_MM:
                            n_pass += 1
                        else:
                            n_fail += 1
                    else:
                        n_missing += 1

            except Exception as e:
                print(f"    ⚠  CSV validation error: {e}")
        else:
            report_lines.append(f"\nPatient {pid}  (no CSV — skipping validation)")
            for label, info in centroids.items():
                report_lines.append(
                    f"  {info['name']:<14} world=({info['world_xyz'][0]:7.2f}, "
                    f"{info['world_xyz'][1]:7.2f}, {info['world_xyz'][2]:7.2f}) mm"
                )

        all_results[pid] = patient_result

    # ── summary ──────────────────────────────────────────────────────────────
    total_validated = n_pass + n_fail + n_missing
    summary = [
        "",
        "=" * 55,
        "VALIDATION SUMMARY",
        "=" * 55,
        f"  Landmarks validated : {total_validated}",
        f"  ✅ PASS (< {SUCCESS_THRESHOLD_MM} mm)  : {n_pass}",
        f"  ❌ FAIL (≥ {SUCCESS_THRESHOLD_MM} mm)  : {n_fail}",
        f"  ⚠  Missing in seg  : {n_missing}",
    ]
    if total_validated > 0:
        pct = 100 * n_pass / (n_pass + n_fail) if (n_pass + n_fail) > 0 else 0
        summary.append(f"  Accuracy           : {pct:.1f}%  (pass / detected)")
    summary.append("=" * 55)

    for line in summary:
        print(line)
    report_lines = summary + report_lines

    # ── save JSON ─────────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅  Results saved  → {OUTPUT_JSON}")

    # ── save report ───────────────────────────────────────────────────────────
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved   → {OUTPUT_REPORT}")


def _fmt(xyz):
    """Pretty-print an [X,Y,Z] list, or 'None'."""
    if xyz is None:
        return "      None      "
    return f"({xyz[0]:7.2f},{xyz[1]:7.2f},{xyz[2]:7.2f})"


# ─────────────────────────────────────────────────────────────────────────────
#  INSTALL CHECK — friendly message if libraries are missing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    missing = []
    try:
        import nibabel
    except ImportError:
        missing.append("nibabel")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    if missing:
        print("⚠  Missing libraries. Run:")
        print(f"   pip install {' '.join(missing)}")
    else:
        run()