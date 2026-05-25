"""
CRT Lead Detection — Step 7: Pseudo-Label Normalization + Retrain
=================================================================
This script completes Option B by:

  PART A — Normalize raw patient pseudo-labels
    Raw patients have no APEX/BASE/ANT from segmentation.
    We estimate them from:
      • heart_centre_lps   (image-estimated in Step 6c)
      • axis_len_mm        (87.3mm population mean if unknown)
      • population-mean axis direction + ANT direction
        (learned from the 59 GT patients in normalized_results.json)
    Then apply the exact same Step 3 math (build_heart_frame, 
    longitudinal_t, radial_angle_deg) to get [t, angle] per electrode.

  PART B — Combine GT + pseudo datasets
    Creates three dataset splits for a fair comparison:
      gt_only   — 59 patients, expert labels       (Step 3 output)
      pseudo    — 221 patients, PointNet labels     (Step 6c output)
      combined  — gt_only + pseudo (280 patients)

  PART C — Retrain Step 4b model on each split + compare LOO
    Uses the same LinearRidge model (best generalizer from Step 4b).
    Compares LOO performance across splits to answer:
    "Did pseudo-labeling the raw patients actually help?"

Usage:
    python claude_p7_retrain.py

Inputs:
    normalized_results.json     Step 3  — GT normalized coords
    pseudo_labels_v2.json       Step 6c — raw patient detections
    centroids_results.json      Step 2  — GT centroids (for axis stats)

Outputs:
    pseudo_normalized.json      — normalized coords for raw patients
    combined_dataset.json       — GT + pseudo merged, ready for ML
    retrain_results.json        — LOO comparison across splits
    retrain_report.txt          — human-readable summary
    retrain_plot.png            — comparison bar chart
"""

import json
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib  import Path
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import Ridge
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.base            import clone
from sklearn.metrics         import mean_absolute_error

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR          = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude")
GT_NORMALIZED     = BASE_DIR / "normalized_results.json"
PSEUDO_LABELS     = BASE_DIR / "pseudo_labels_v2.json"
CENTROIDS_JSON    = BASE_DIR / "centroids_results.json"

OUTPUT_PSEUDO_NORM = BASE_DIR / "pseudo_normalized.json"
OUTPUT_COMBINED    = BASE_DIR / "combined_dataset.json"
OUTPUT_RESULTS     = BASE_DIR / "retrain_results.json"
OUTPUT_REPORT      = BASE_DIR / "retrain_report.txt"
OUTPUT_PLOT        = BASE_DIR / "retrain_plot.png"


# ─────────────────────────────────────────────────────────────────────────────
#  ELECTRODE + FEATURE DEFINITIONS  (same as Step 4b)
# ─────────────────────────────────────────────────────────────────────────────

ELECTRODE_NAMES = ["LL1", "LL2", "LL3", "LL4", "RL1", "RL2"]

FEATURE_NAMES = [
    "RL1_t", "RL1_ang", "RL2_t", "RL2_ang",
    "rv_spread_t", "rv_spread_ang", "rv_mean_t", "rv_mean_ang",
    "axis_len_mm",
]
TARGET_NAMES = ["LL1_t", "LL1_ang"]   # distal LV lead — most important


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 MATH  (copied verbatim from claude_p3_normalize_coords.py)
# ─────────────────────────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError(f"Cannot normalise near-zero vector: {v}")
    return v / n


def build_heart_frame(apex_xyz, base_xyz, ant_xyz):
    apex = np.array(apex_xyz, dtype=float)
    base = np.array(base_xyz, dtype=float)
    ant  = np.array(ant_xyz,  dtype=float)
    axis_vec  = base - apex
    axis_len  = float(np.linalg.norm(axis_vec))
    axis_unit = unit(axis_vec)
    ant_vec   = ant - apex
    t_ant     = np.dot(ant_vec, axis_unit)
    ant_perp  = ant_vec - t_ant * axis_unit
    ant_unit  = unit(ant_perp)
    lat_unit  = np.cross(axis_unit, ant_unit)
    return axis_unit, ant_unit, lat_unit, axis_len


def longitudinal_t(point, apex, axis_unit):
    return float(np.dot(np.array(point) - np.array(apex), axis_unit))


def radial_angle_deg(point, apex, axis_unit, ant_unit, lat_unit):
    vec_from_apex = np.array(point) - np.array(apex)
    t             = np.dot(vec_from_apex, axis_unit)
    radial_vec    = vec_from_apex - t * axis_unit
    comp_ant      = np.dot(radial_vec, ant_unit)
    comp_lat      = np.dot(radial_vec, lat_unit)
    angle_rad     = math.atan2(comp_lat, comp_ant)
    return math.degrees(angle_rad) % 360


def angular_diff(a1: float, a2: float) -> float:
    diff = abs(a1 - a2) % 360
    return diff if diff <= 180 else 360 - diff


def circular_mean_deg(a1: float, a2: float) -> float:
    a1r, a2r  = math.radians(a1), math.radians(a2)
    mean_sin  = (math.sin(a1r) + math.sin(a2r)) / 2
    mean_cos  = (math.cos(a1r) + math.cos(a2r)) / 2
    return math.degrees(math.atan2(mean_sin, mean_cos)) % 360


# ─────────────────────────────────────────────────────────────────────────────
#  PART A — POPULATION FRAME ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_population_frame(gt_normalized: dict) -> dict:
    """
    Learn the mean heart axis direction and ANT direction from GT patients.
    These serve as priors for raw patients that have no APEX/BASE/ANT.

    Returns a dict with:
      mean_axis_dir  — mean unit vector Apex→Base in LPS space
      mean_ant_dir   — mean unit vector toward ANT wall in LPS space
      mean_axis_len  — mean axis length in mm
    """
    axis_dirs = []
    ant_dirs  = []
    axis_lens = []

    for pid, rec in gt_normalized.items():
        apex = np.array(rec["apex_xyz"], dtype=float)
        base = np.array(rec["base_xyz"], dtype=float)
        ant  = np.array(rec["ant_xyz"],  dtype=float)

        axis_vec = base - apex
        axis_len = float(np.linalg.norm(axis_vec))
        if axis_len < 1:
            continue

        axis_u = axis_vec / axis_len
        axis_dirs.append(axis_u)
        axis_lens.append(axis_len)

        # ANT direction in perpendicular plane
        ant_vec  = ant - apex
        t_ant    = np.dot(ant_vec, axis_u)
        ant_perp = ant_vec - t_ant * axis_u
        ant_len  = np.linalg.norm(ant_perp)
        if ant_len > 1:
            ant_dirs.append(ant_perp / ant_len)

    mean_axis_dir = np.mean(axis_dirs, axis=0)
    mean_axis_dir = mean_axis_dir / np.linalg.norm(mean_axis_dir)

    mean_ant_dir  = np.mean(ant_dirs, axis=0)
    mean_ant_dir  = mean_ant_dir / np.linalg.norm(mean_ant_dir)

    mean_axis_len = float(np.mean(axis_lens))

    print(f"  Population frame from {len(axis_dirs)} GT patients:")
    print(f"    Mean axis direction : {mean_axis_dir.round(3)}")
    print(f"    Mean ANT direction  : {mean_ant_dir.round(3)}")
    print(f"    Mean axis length    : {mean_axis_len:.1f} mm")

    return {
        "mean_axis_dir": mean_axis_dir.tolist(),
        "mean_ant_dir":  mean_ant_dir.tolist(),
        "mean_axis_len": mean_axis_len,
        "n_gt_patients": len(axis_dirs),
    }


def estimate_anchors(heart_centre: list, axis_len_mm: float,
                      pop_frame: dict) -> tuple:
    """
    Estimate APEX, BASE, ANT for a raw patient given:
      • heart_centre   — image-estimated midpoint (Step 6c)
      • axis_len_mm    — estimated axis length
      • pop_frame      — population-mean directions

    Strategy:
      APEX = heart_centre - (axis_len/2) * mean_axis_dir
      BASE = heart_centre + (axis_len/2) * mean_axis_dir
      ANT  = APEX + mean_ant_dir * (axis_len * 0.3)
             (ANT is typically ~30% of axis length away from apex,
              perpendicular to the axis)

    These are approximate (±5-15mm uncertainty) but consistent.
    """
    hc       = np.array(heart_centre, dtype=float)
    axis_dir = np.array(pop_frame["mean_axis_dir"], dtype=float)
    ant_dir  = np.array(pop_frame["mean_ant_dir"],  dtype=float)
    half     = axis_len_mm / 2.0

    apex_est = hc - half * axis_dir
    base_est = hc + half * axis_dir
    ant_est  = apex_est + ant_dir * (axis_len_mm * 0.3)

    return apex_est.tolist(), base_est.tolist(), ant_est.tolist()


# ─────────────────────────────────────────────────────────────────────────────
#  PART A — NORMALIZE RAW PATIENTS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_raw_patient(pid: str, pseudo_rec: dict,
                            pop_frame: dict) -> dict | None:
    """
    Convert raw patient pseudo-labels into [t, angle] normalized coords.

    Returns a record in the same format as normalized_results.json,
    with an extra 'is_pseudo': True flag so downstream code can
    distinguish GT from pseudo data.
    """
    detections   = pseudo_rec.get("detections", {})
    hc           = pseudo_rec.get("heart_centre_lps")
    axis_len_mm  = pseudo_rec.get("axis_len_mm", pop_frame["mean_axis_len"])

    if not detections or hc is None:
        return None

    # check we have at least one RV electrode (needed as ML feature)
    has_rv = any(k in detections for k in ("RL1", "RL2"))
    if not has_rv:
        return None

    # estimate anchors from population prior
    apex_xyz, base_xyz, ant_xyz = estimate_anchors(hc, axis_len_mm, pop_frame)

    # build heart frame (same function as Step 3)
    try:
        axis_unit, ant_unit, lat_unit, axis_len_actual = build_heart_frame(
            apex_xyz, base_xyz, ant_xyz
        )
    except ValueError:
        return None

    # normalize each detected electrode
    electrodes = {}
    for name in ELECTRODE_NAMES:
        if name not in detections:
            continue
        det  = detections[name]
        pt   = np.array(det["world_xyz"], dtype=float)
        t    = longitudinal_t(pt, apex_xyz, axis_unit)
        t_n  = t / axis_len_actual
        ang  = radial_angle_deg(pt, apex_xyz, axis_unit, ant_unit, lat_unit)

        electrodes[name] = {
            "world_xyz":         det["world_xyz"],
            "longitudinal_t":    round(t_n, 4),
            "radial_angle_deg":  round(ang, 2),
            "confidence":        det.get("confidence", None),
            "dist_from_apex_mm": round(t, 2),
        }

    if not electrodes:
        return None

    return {
        "apex_xyz":    apex_xyz,
        "base_xyz":    base_xyz,
        "ant_xyz":     ant_xyz,
        "axis_len_mm": round(axis_len_mm, 2),
        "electrodes":  electrodes,
        "is_pseudo":   True,          # flag: labels come from PointNet, not expert
        "hc_method":   pseudo_rec.get("hc_method", "unknown"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PART B — BUILD ML FEATURE ROWS
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_row(rec: dict) -> dict | None:
    """
    Convert a normalized patient record into an ML feature + target row.
    Returns None if any required electrode is missing.
    Same feature engineering as Step 4b.
    """
    elec = rec.get("electrodes", {})

    rl1 = elec.get("RL1")
    rl2 = elec.get("RL2")
    ll1 = elec.get("LL1")
    if rl1 is None or rl2 is None or ll1 is None:
        return None

    rl1_t   = rl1["longitudinal_t"]
    rl1_ang = rl1["radial_angle_deg"]
    rl2_t   = rl2["longitudinal_t"]
    rl2_ang = rl2["radial_angle_deg"]

    features = {
        "RL1_t":        rl1_t,
        "RL1_ang":      rl1_ang,
        "RL2_t":        rl2_t,
        "RL2_ang":      rl2_ang,
        "rv_spread_t":  rl2_t - rl1_t,
        "rv_spread_ang": angular_diff(rl1_ang, rl2_ang),
        "rv_mean_t":    (rl1_t + rl2_t) / 2,
        "rv_mean_ang":  circular_mean_deg(rl1_ang, rl2_ang),
        "axis_len_mm":  rec.get("axis_len_mm", 87.3),
    }

    targets = {
        "LL1_t":   ll1["longitudinal_t"],
        "LL1_ang": ll1["radial_angle_deg"],
    }

    return {
        "features": features,
        "targets":  targets,
        "is_pseudo": rec.get("is_pseudo", False),
    }


def build_numpy_arrays(rows: list) -> tuple:
    """Convert list of row dicts to numpy X, y arrays."""
    X = np.array([[r["features"][f] for f in FEATURE_NAMES]
                   for r in rows], dtype=np.float32)
    y = np.array([[r["targets"][t]  for t in TARGET_NAMES]
                   for r in rows], dtype=np.float32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  PART C — LOO EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def angular_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.abs(y_true - y_pred) % 360
    diff = np.where(diff > 180, 360 - diff, diff)
    return float(diff.mean())


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE for each target. Angle uses circular MAE."""
    return {
        "LL1_t_mae":   round(float(mean_absolute_error(
                           y_true[:, 0], y_pred[:, 0])), 4),
        "LL1_ang_mae": round(angular_mae(y_true[:, 1], y_pred[:, 1]), 2),
    }


def run_loo(X: np.ndarray, y: np.ndarray) -> dict:
    """Leave-One-Out CV with LinearRidge. Returns score dict."""
    pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=0.1))])
    loo  = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        m = clone(pipe)
        m.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = m.predict(X[test_idx])

    return score(y, y_pred)


def mean_baseline(y: np.ndarray) -> dict:
    """Score of always predicting the training mean."""
    loo  = LeaveOneOut()
    y_pred = np.zeros_like(y)
    for train_idx, test_idx in loo.split(y):
        y_pred[test_idx] = y[train_idx].mean(axis=0)
    return score(y, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(split_results: dict, mean_axis_mm: float):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Step 7 — Does Pseudo-Labeling Help?\n"
        "LOO MAE comparison: GT-only vs GT+Pseudo vs Pseudo-only",
        fontsize=13, fontweight="bold"
    )

    splits  = list(split_results.keys())
    colors  = {"gt_only": "#4e79a7",
               "pseudo_only": "#f28e2b",
               "combined": "#59a14f",
               "MeanBaseline": "#bbbbbb"}

    t_vals   = [split_results[s]["LL1_t_mae"]   for s in splits]
    ang_vals = [split_results[s]["LL1_ang_mae"]  for s in splits]
    n_vals   = [split_results[s]["n_patients"]   for s in splits]
    labels   = [f"{s}\n(n={n})" for s, n in zip(splits, n_vals)]
    bar_cols = [colors.get(s, "#aaaaaa") for s in splits]

    # ── longitudinal t MAE ────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(labels, t_vals, color=bar_cols, alpha=0.88, edgecolor="k",
                  linewidth=0.6)
    baseline_t = split_results.get("MeanBaseline", {}).get("LL1_t_mae", None)
    if baseline_t:
        ax.axhline(baseline_t, color="red", linestyle="--", lw=1.3,
                   label=f"Mean baseline ({baseline_t:.3f})")
    ax.set_ylabel("LOO MAE  (t-units)")
    ax.set_title(f"Longitudinal Position Error — LL1_t\n"
                 f"(× {mean_axis_mm:.0f} mm/unit)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, t_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f"{v:.3f}\n≈{v*mean_axis_mm:.1f}mm",
                ha="center", va="bottom", fontsize=8)

    # ── angle MAE ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    bars2 = ax2.bar(labels, ang_vals, color=bar_cols, alpha=0.88,
                    edgecolor="k", linewidth=0.6)
    baseline_ang = split_results.get("MeanBaseline", {}).get("LL1_ang_mae", None)
    if baseline_ang:
        ax2.axhline(baseline_ang, color="red", linestyle="--", lw=1.3,
                    label=f"Mean baseline ({baseline_ang:.1f}°)")
    ax2.set_ylabel("LOO MAE  (degrees)")
    ax2.set_title("Radial Angle Error — LL1_ang\n(0°=Anterior, CW viewed from Base)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars2, ang_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{v:.1f}°", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✅  Plot saved → {OUTPUT_PLOT}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  CRT Lead — Step 7: Pseudo-Label Normalization + Retrain")
    print("=" * 65)

    # ── load data ─────────────────────────────────────────────────────────
    with open(GT_NORMALIZED,  encoding="utf-8") as f: gt_norm    = json.load(f)
    with open(PSEUDO_LABELS,  encoding="utf-8") as f: pseudo_raw = json.load(f)
    with open(CENTROIDS_JSON, encoding="utf-8") as f: centroids  = json.load(f)

    patients_raw = pseudo_raw.get("patients", pseudo_raw)

    print(f"\n  GT normalized patients  : {len(gt_norm)}")
    print(f"  Raw pseudo-label records: {len(patients_raw)}")

    # ─────────────────────────────────────────────────────────────────────
    #  PART A — NORMALIZE RAW PATIENTS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  PART A — Computing population frame + normalizing raw patients")
    print("─" * 65)

    pop_frame = compute_population_frame(gt_norm)

    pseudo_normalized = {}
    n_ok = n_skip = 0

    for pid, rec in sorted(patients_raw.items()):
        # skip GT patients — they already have expert normalization
        if rec.get("type") == "GT":
            continue
        if not rec.get("detections"):
            n_skip += 1
            continue

        result = normalize_raw_patient(pid, rec, pop_frame)
        if result is None:
            n_skip += 1
            continue

        pseudo_normalized[pid] = result
        n_ok += 1

    print(f"\n  Raw patients normalized  : {n_ok}")
    print(f"  Raw patients skipped     : {n_skip}  "
          f"(no detections or missing RV/LV leads)")

    with open(OUTPUT_PSEUDO_NORM, "w", encoding="utf-8") as f:
        json.dump(pseudo_normalized, f, indent=2)
    print(f"✅  Pseudo-normalized saved → {OUTPUT_PSEUDO_NORM}")

    # ─────────────────────────────────────────────────────────────────────
    #  PART B — BUILD THREE DATASET SPLITS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  PART B — Building dataset splits")
    print("─" * 65)

    # ── split 1: GT only ─────────────────────────────────────────────────
    gt_rows = []
    for pid, rec in gt_norm.items():
        row = build_feature_row(rec)
        if row:
            row["pid"] = pid
            gt_rows.append(row)
    print(f"\n  GT-only rows    : {len(gt_rows)}")

    # ── split 2: pseudo only ─────────────────────────────────────────────
    pseudo_rows = []
    for pid, rec in pseudo_normalized.items():
        row = build_feature_row(rec)
        if row:
            row["pid"] = pid
            pseudo_rows.append(row)
    print(f"  Pseudo-only rows: {len(pseudo_rows)}")

    # ── split 3: combined ────────────────────────────────────────────────
    combined_rows = gt_rows + pseudo_rows
    print(f"  Combined rows   : {len(combined_rows)}")

    # save combined dataset
    combined_payload = {
        "gt_patients":     [r["pid"] for r in gt_rows],
        "pseudo_patients": [r["pid"] for r in pseudo_rows],
        "all_rows": [
            {"pid": r["pid"], "features": r["features"],
             "targets": r["targets"], "is_pseudo": r["is_pseudo"]}
            for r in combined_rows
        ],
    }
    with open(OUTPUT_COMBINED, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2)
    print(f"✅  Combined dataset saved  → {OUTPUT_COMBINED}")

    # ─────────────────────────────────────────────────────────────────────
    #  PART C — LOO COMPARISON
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  PART C — LOO Cross-Validation Comparison")
    print("─" * 65)

    mean_axis_mm = pop_frame["mean_axis_len"]

    splits = {
        "MeanBaseline": None,   # always predict mean
        "gt_only":      gt_rows,
        "pseudo_only":  pseudo_rows,
        "combined":     combined_rows,
    }

    all_split_results = {}
    report_lines      = []

    header = (f"\n  {'Split':<16}  {'N':>5}  {'LL1_t MAE':>12}  "
              f"{'≈mm':>7}  {'LL1_ang MAE':>12}  {'vs baseline':>12}")
    print(header)
    print("  " + "─" * 70)
    report_lines += [header, "  " + "─"*70]

    baseline_t = baseline_ang = None

    for split_name, rows in splits.items():
        if rows is None or len(rows) < 3:
            if split_name == "MeanBaseline":
                # compute on GT data
                X_gt, y_gt = build_numpy_arrays(gt_rows)
                bl = mean_baseline(y_gt)
                baseline_t   = bl["LL1_t_mae"]
                baseline_ang = bl["LL1_ang_mae"]
                all_split_results[split_name] = {
                    **bl, "n_patients": len(gt_rows)
                }
                row_str = (f"  {'MeanBaseline':<16}  {len(gt_rows):>5}  "
                           f"{baseline_t:>10.4f}t  "
                           f"{baseline_t*mean_axis_mm:>5.1f}  "
                           f"{baseline_ang:>10.1f}°  (reference)")
                print(row_str)
                report_lines.append(row_str)
            else:
                print(f"  {split_name:<16}  too few rows — skipped")
            continue

        X, y = build_numpy_arrays(rows)
        print(f"  Running LOO on {split_name} (n={len(rows)})...",
              end=" ", flush=True)
        sc = run_loo(X, y)
        all_split_results[split_name] = {**sc, "n_patients": len(rows)}

        delta_t   = sc["LL1_t_mae"]   - baseline_t   if baseline_t   else 0
        delta_ang = sc["LL1_ang_mae"] - baseline_ang if baseline_ang else 0
        sign_t    = "+" if delta_t   >= 0 else ""
        sign_a    = "+" if delta_ang >= 0 else ""

        row_str = (f"  {split_name:<16}  {len(rows):>5}  "
                   f"{sc['LL1_t_mae']:>10.4f}t  "
                   f"{sc['LL1_t_mae']*mean_axis_mm:>5.1f}  "
                   f"{sc['LL1_ang_mae']:>10.1f}°  "
                   f"({sign_t}{delta_t:.4f} / {sign_a}{delta_ang:.1f}°)")
        print("done.")
        print(row_str)
        report_lines.append(row_str)

    # ── interpretation ────────────────────────────────────────────────────
    gt_sc = all_split_results.get("gt_only", {})
    cb_sc = all_split_results.get("combined", {})

    t_improved   = cb_sc.get("LL1_t_mae",   99) < gt_sc.get("LL1_t_mae",   0)
    ang_improved = cb_sc.get("LL1_ang_mae", 99) < gt_sc.get("LL1_ang_mae", 0)

    interpretation = [
        "",
        "=" * 65,
        "INTERPRETATION",
        "=" * 65,
        "",
        f"  GT-only   LL1_t  MAE : {gt_sc.get('LL1_t_mae',   'N/A'):.4f} t  "
        f"≈ {gt_sc.get('LL1_t_mae', 0)*mean_axis_mm:.1f} mm",
        f"  Combined  LL1_t  MAE : {cb_sc.get('LL1_t_mae',   'N/A'):.4f} t  "
        f"≈ {cb_sc.get('LL1_t_mae', 0)*mean_axis_mm:.1f} mm  "
        f"{'✅ improved' if t_improved else '❌ no improvement'}",
        "",
        f"  GT-only   LL1_ang MAE: {gt_sc.get('LL1_ang_mae', 'N/A'):.1f}°",
        f"  Combined  LL1_ang MAE: {cb_sc.get('LL1_ang_mae', 'N/A'):.1f}°  "
        f"{'✅ improved' if ang_improved else '❌ no improvement'}",
        "",
        "WHAT THE RESULTS MEAN:",
        "",
        "  SCENARIO 1 — Combined beats GT-only on BOTH targets:",
        "    The pseudo-labels from PointNet are noisy but carry real",
        "    signal. More data helped even with label noise.",
        "    → Next step: filter pseudo data by confidence score",
        "      (keep only detections with confidence > 0.75)",
        "",
        "  SCENARIO 2 — Combined is WORSE than GT-only:",
        "    The approximate normalization (population-mean axis/ANT)",
        "    introduced too much noise. The pseudo-labels hurt more than",
        "    they helped.",
        "    → This is also a valid research finding:",
        "      'Pseudo-labels with ~10mm anchor uncertainty do not",
        "       improve prediction beyond 59 expert-labeled patients.'",
        "    → Consider: a semi-supervised approach where pseudo-labels",
        "      are used only for pre-training, then fine-tuned on GT.",
        "",
        "  SCENARIO 3 — Pseudo-only is close to GT-only:",
        "    PointNet detections are accurate enough to replace manual",
        "    labeling for the normalization step. This is the strongest",
        "    possible result for this project.",
        "",
        "FULL PIPELINE SUMMARY (all steps):",
        "  Step 1  Data inventory          → 86 GT + 308 raw patients found",
        "  Step 2  Centroid extraction     → 100% accuracy vs manual clicks",
        "  Step 3  Normalization           → heart-relative [t, angle] coords",
        "  Step 4  Overfit ML experiment   → signal exists but weak (RV→LV)",
        "  Step 4b Improved ML             → LOO LL1_t ≈14mm, ang ≈16°",
        "  Step 5  Classical CV baseline   → 56.5% electrode detection",
        "  Step 5b Param sweep + RANSAC    → 77.7% (RANSAC hurt, removed)",
        "  Step 6  PointNet classifier     → 89.0% recall, 31.6% FP",
        "  Step 6b Conf threshold          → 74.9% recall, 8.6% FP",
        "  Step 6c Fixed heart centre      → 74.9% GT / 99.5% RAW covered",
        f"  Step 7  Pseudo-label retrain   → see results above",
        "=" * 65,
    ]

    for line in interpretation:
        print(line)
    report_lines.extend(interpretation)

    # ── save ──────────────────────────────────────────────────────────────
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump({
            "pop_frame":      pop_frame,
            "split_results":  all_split_results,
            "mean_axis_mm":   mean_axis_mm,
        }, f, indent=2)
    print(f"\n✅  Results saved  → {OUTPUT_RESULTS}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved   → {OUTPUT_REPORT}")

    make_plot(all_split_results, mean_axis_mm)


if __name__ == "__main__":
    missing = []
    try:    import sklearn
    except ImportError: missing.append("scikit-learn")
    try:    import matplotlib
    except ImportError: missing.append("matplotlib")

    if missing:
        print(f"⚠  pip install {' '.join(missing)}")
    else:
        run()