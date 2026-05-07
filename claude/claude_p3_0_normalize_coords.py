"""
CRT Lead Detection Project — Step 3: Coordinate Normalization
=============================================================
Takes the world-space (LPS mm) centroids from Step 2 and converts
each electrode position into a heart-relative coordinate system:

  Longitudinal (t):  0.0 = at the Apex, 1.0 = at the Base
  Radial angle (°):  0° = Anterior wall (ANT), clockwise when
                     viewed from the Base looking toward the Apex.
                     Typical anatomy: ~90° = Lateral wall,
                     ~180° = Posterior, ~270° = Septal

These two numbers are the same regardless of patient size or scanner
position — making them suitable as ML features / targets.

Usage:
    python claude_p3_normalize_coords.py

Input:
    centroids_results.json   (output of Step 2)

Output:
    normalized_results.json  — per-patient normalized coords + raw
    normalized_report.txt    — human-readable summary table
"""

import json
import math
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURE YOUR PATHS HERE
# ─────────────────────────────────────────────────────────────────────────────

CENTROIDS_JSON   = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\centroids_results.json")
OUTPUT_JSON      = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\normalized_results.json")
OUTPUT_REPORT    = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\normalized_report.txt")

# Labels we want to normalize (everything except the anchors themselves)
ELECTRODE_CSV_NAMES = ["LL1", "LL2", "LL3", "LL4", "RL1", "RL2"]
ANCHOR_CSV_NAMES    = ["ANT", "APEX", "BASE"]

# Seg label numbers for the three anchor points
LABEL_ANT  = 4001
LABEL_APEX = 4002
LABEL_BASE = 4003


# ─────────────────────────────────────────────────────────────────────────────
#  MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    """Return the unit (normalised) vector of v."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError(f"Cannot normalise near-zero vector: {v}")
    return v / n


def longitudinal_t(point: np.ndarray,
                   apex:  np.ndarray,
                   axis_unit: np.ndarray) -> float:
    """
    Project *point* onto the apex→base axis.
    Returns t where t=0 is exactly at the Apex, t=1 is exactly at the Base.
    Values outside [0,1] are fine — they just mean the point is beyond
    the apex or base, which can happen for proximal RV leads.
    """
    return float(np.dot(point - apex, axis_unit))


def radial_angle_deg(point:     np.ndarray,
                     apex:      np.ndarray,
                     axis_unit: np.ndarray,
                     ant_unit:  np.ndarray,
                     lat_unit:  np.ndarray) -> float:
    """
    Compute the clock-face angle of *point* in the plane perpendicular
    to the heart axis, using the Anterior wall as 0° / 12 o'clock.

    Coordinate frame in the perpendicular plane:
        ant_unit  →  0°  (12 o'clock, Anterior)
        lat_unit  →  90° (3 o'clock, Lateral — left side of the heart
                          when viewed from Base toward Apex)

    Returns angle in [0, 360) degrees, clockwise viewed from Base.

    Steps:
      1. Remove the longitudinal component of (point - apex) to get the
         purely radial vector.
      2. Decompose that radial vector into (ant, lat) components.
      3. atan2 → angle, clockwise convention.
    """
    vec_from_apex = point - apex
    # subtract the along-axis part → purely radial
    t = np.dot(vec_from_apex, axis_unit)
    radial_vec = vec_from_apex - t * axis_unit

    # decompose onto the two perpendicular basis vectors
    comp_ant = np.dot(radial_vec, ant_unit)
    comp_lat = np.dot(radial_vec, lat_unit)

    # atan2(lat, ant): 0° = anterior, 90° = lateral
    angle_rad = math.atan2(comp_lat, comp_ant)
    angle_deg = math.degrees(angle_rad) % 360   # force into [0, 360)
    return angle_deg


def build_heart_frame(apex_xyz: list, base_xyz: list, ant_xyz: list):
    """
    Build a heart-relative coordinate frame from the three anchor points.

    Returns:
        axis_unit  — unit vector from Apex → Base (longitudinal axis)
        ant_unit   — unit vector toward Anterior wall (⊥ to axis)
        lat_unit   — unit vector toward Lateral wall (⊥ to both above)
        axis_len   — distance Apex→Base in mm (useful for sanity checks)

    The three returned vectors form a right-handed orthonormal basis.
    lat_unit = cross(axis_unit, ant_unit), which points to the lateral
    wall (left side of the heart) when viewed from the Base.
    """
    apex = np.array(apex_xyz, dtype=float)
    base = np.array(base_xyz, dtype=float)
    ant  = np.array(ant_xyz,  dtype=float)

    axis_vec  = base - apex
    axis_len  = float(np.linalg.norm(axis_vec))
    axis_unit = unit(axis_vec)

    # Project ANT onto the perpendicular plane
    ant_vec   = ant - apex
    t_ant     = np.dot(ant_vec, axis_unit)
    ant_perp  = ant_vec - t_ant * axis_unit   # radial component of ANT

    ant_unit  = unit(ant_perp)                # 0° / 12 o'clock direction
    lat_unit  = np.cross(axis_unit, ant_unit) # 90° / 3 o'clock direction
    # Note: cross product of two unit vectors is already unit length if ⊥

    return axis_unit, ant_unit, lat_unit, axis_len


# ─────────────────────────────────────────────────────────────────────────────
#  CLOCK LABEL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def angle_to_clock(angle_deg: float) -> str:
    """
    Convert an angle (0° = Anterior / 12 o'clock, clockwise) to a
    human-readable anatomical wall label.

         0°  ±22.5°  → Anterior
        45°  ±22.5°  → Anterolateral
        90°  ±22.5°  → Lateral
       135°  ±22.5°  → Posterolateral
       180°  ±22.5°  → Posterior
       225°  ±22.5°  → Posteromedial / Inferoseptal
       270°  ±22.5°  → Septal / Inferior
       315°  ±22.5°  → Anteroseptal
    """
    a = angle_deg % 360
    boundaries = [
        (22.5,  "Anterior"),
        (67.5,  "Anterolateral"),
        (112.5, "Lateral"),
        (157.5, "Posterolateral"),
        (202.5, "Posterior"),
        (247.5, "Posteromedial"),
        (292.5, "Septal/Inferior"),
        (337.5, "Anteroseptal"),
        (360.0, "Anterior"),
    ]
    for threshold, label in boundaries:
        if a < threshold:
            return label
    return "Anterior"


# ─────────────────────────────────────────────────────────────────────────────
#  PER-PATIENT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalise_patient(pid: str, patient_data: dict) -> dict | None:
    """
    Normalise one patient's centroids.
    Returns a result dict, or None if anchors are missing.
    """
    centroids  = patient_data.get("centroids", {})
    validation = patient_data.get("validation", {})

    # ── pull anchor coords from centroids (auto-extracted) ───────────────────
    def get_centroid(label_id: int):
        entry = centroids.get(str(label_id)) or centroids.get(label_id)
        return entry["world_xyz"] if entry else None

    apex_xyz = get_centroid(LABEL_APEX)
    base_xyz = get_centroid(LABEL_BASE)
    ant_xyz  = get_centroid(LABEL_ANT)

    if apex_xyz is None or base_xyz is None or ant_xyz is None:
        print(f"  [{pid}] ⚠  Skipped — missing APEX/BASE/ANT centroid.")
        return None

    # ── build the heart coordinate frame ─────────────────────────────────────
    try:
        axis_unit, ant_unit, lat_unit, axis_len_mm = build_heart_frame(
            apex_xyz, base_xyz, ant_xyz
        )
    except ValueError as e:
        print(f"  [{pid}] ⚠  Frame error: {e}")
        return None

    # ── collect electrode coords (prefer auto centroids, fall back to CSV) ───
    # We have centroids for label 4004–4009 (LL1–LL4, RL1, RL2).
    # Map CSV names → label IDs for easy lookup.
    label_for_csv = {
        "LL1": 4004, "LL2": 4005, "LL3": 4006, "LL4": 4007,
        "RL1": 4008, "RL2": 4009,
    }

    electrodes = {}
    for csv_name in ELECTRODE_CSV_NAMES:
        label_id = label_for_csv[csv_name]
        entry    = centroids.get(str(label_id)) or centroids.get(label_id)
        if entry is None:
            continue
        pt = np.array(entry["world_xyz"], dtype=float)

        # longitudinal position
        t = longitudinal_t(pt, np.array(apex_xyz), axis_unit)
        # normalise by axis length so 0=apex, 1=base
        t_norm = t / axis_len_mm

        # radial angle
        angle = radial_angle_deg(pt, np.array(apex_xyz),
                                 axis_unit, ant_unit, lat_unit)

        electrodes[csv_name] = {
            "world_xyz":         entry["world_xyz"],
            "longitudinal_t":    round(t_norm, 4),   # 0=apex, 1=base
            "radial_angle_deg":  round(angle, 2),    # 0=ANT, CW viewed from base
            "wall_label":        angle_to_clock(angle),
            "dist_from_apex_mm": round(t, 2),
        }

    return {
        "apex_xyz":    apex_xyz,
        "base_xyz":    base_xyz,
        "ant_xyz":     ant_xyz,
        "axis_len_mm": round(axis_len_mm, 2),
        "electrodes":  electrodes,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  CRT Lead — Step 3: Coordinate Normalization")
    print("=" * 60)

    with open(CENTROIDS_JSON, encoding="utf-8") as f:
        all_centroids = json.load(f)

    all_normalised = {}
    report_lines   = []
    n_ok = n_skip  = 0

    for pid in sorted(all_centroids.keys()):
        patient_data = all_centroids[pid]
        if "error" in patient_data:
            continue

        result = normalise_patient(pid, patient_data)
        if result is None:
            n_skip += 1
            continue

        all_normalised[pid] = result
        n_ok += 1

        # ── per-patient report block ──────────────────────────────────────────
        block = [
            f"\n{'─'*58}",
            f"Patient {pid}   (heart axis = {result['axis_len_mm']:.1f} mm)",
            f"{'─'*58}",
            f"  {'Lead':<5}  {'Long t':>7}  {'Angle°':>7}  {'Wall':>16}  {'mm from Apex':>13}",
            f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*16}  {'─'*13}",
        ]
        for name, e in result["electrodes"].items():
            block.append(
                f"  {name:<5}  {e['longitudinal_t']:>7.3f}  "
                f"{e['radial_angle_deg']:>7.1f}  "
                f"{e['wall_label']:>16}  "
                f"{e['dist_from_apex_mm']:>11.1f} mm"
            )
        report_lines.extend(block)
        # also print to terminal
        for line in block:
            print(line)

    # ── summary ──────────────────────────────────────────────────────────────
    summary = [
        "",
        "=" * 58,
        "NORMALIZATION SUMMARY",
        "=" * 58,
        f"  Patients normalised  : {n_ok}",
        f"  Patients skipped     : {n_skip}  (missing APEX/BASE/ANT)",
        "=" * 58,
    ]
    for line in summary:
        print(line)
    report_lines = summary + report_lines

    # ── save outputs ──────────────────────────────────────────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_normalised, f, indent=2)
    print(f"\n✅  Normalized results → {OUTPUT_JSON}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved       → {OUTPUT_REPORT}")

    # ── print a quick feature matrix preview ─────────────────────────────────
    print("\n── Feature matrix preview (what ML will train on) ────────────")
    print(f"  {'PID':<7}", end="")
    cols = []
    for name in ELECTRODE_CSV_NAMES:
        cols += [f"{name}_t", f"{name}_ang"]
    for c in cols:
        print(f"  {c:>9}", end="")
    print()

    for pid, r in list(all_normalised.items())[:5]:   # first 5 patients
        print(f"  {pid:<7}", end="")
        for name in ELECTRODE_CSV_NAMES:
            e = r["electrodes"].get(name)
            if e:
                print(f"  {e['longitudinal_t']:>9.3f}", end="")
                print(f"  {e['radial_angle_deg']:>9.1f}", end="")
            else:
                print(f"  {'N/A':>9}", end="")
                print(f"  {'N/A':>9}", end="")
        print()

    if n_ok > 5:
        print(f"  ... ({n_ok - 5} more patients)")


if __name__ == "__main__":
    run()