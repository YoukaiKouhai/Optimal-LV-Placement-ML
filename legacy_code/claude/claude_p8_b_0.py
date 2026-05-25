"""
CRT Lead Detection — Step 8b: MIP CNN Fixed (Voxel-Space Triangulation)
========================================================================
FIX FROM STEP 8 (55mm errors)
-------------------------------
Root cause:  CT affine matrices are NOT diagonal — scanner axes don't
             align perfectly with world axes.  The old code set the
             collapsed voxel axis to volume_midpoint, ran the full affine,
             and the off-diagonal terms contaminated X, Y, Z outputs even
             for the axes that were supposedly predicted correctly.

Fix:         Work entirely in NORMALISED VOXEL SPACE for targets and
             triangulation.  Convert to world LPS exactly ONCE at the end.

  OLD (broken):
    predict row_norm, col_norm
    → pixel_to_world_lps with fake collapsed axis
    → average broken world coords from each projection

  NEW (correct):
    Target = (dim_a_norm, dim_b_norm) in voxel space
    Predict = (dim_a_pred, dim_b_pred) in voxel space
    Triangulate: average overlapping voxel dims across projections
    Once: voxel_norm_to_world_lps(i_avg, j_avg, k_avg)

PROJECTION → VOXEL DIMENSION MAPPING
--------------------------------------
  axial    (collapse nibabel axis 0): predicts dim1_norm, dim2_norm → (j, k)
  coronal  (collapse nibabel axis 1): predicts dim0_norm, dim2_norm → (i, k)
  sagittal (collapse nibabel axis 2): predicts dim0_norm, dim1_norm → (i, j)

TRIANGULATION
-------------
  i_norm = mean( coronal_row,  sagittal_row )
  j_norm = mean( axial_row,    sagittal_col )
  k_norm = mean( axial_col,    coronal_col  )

  world_lps = voxel_norm_to_world_lps(i_norm, j_norm, k_norm)

Architecture: heatmap + soft-argmax head (replaces global-pool MLP — required
for localising a small bright electrode in a full-field MIP).  Optional
``DualMIPLocalizer`` fuses those logits with a lightweight residual U-Net on the
MIP.  Training blends Gaussian heatmap matching with SmoothL1 on coordinates.
Inference can average predictions over flips (TTA).  Optional 2D ROI: default **polar** mode — `ROI_POLY_N` rays from an
intensity-weighted seed, outward to the farthest in-mask pixel per direction;
the axis-aligned bbox of that star polygon (optionally **unioned** with a
centered square) is cropped and targets remapped.  Legacy **square** mode:
`--roi-square` or `roi_mode="square"`.
``--auto-tune`` probes VRAM then grid-searches IMG_SIZE / σ / loss weights on a
fast surrogate task.  ANT / APEX / BASE are trained like leads for bullseye JSON.

Requirements:
    pip install torch torchvision nibabel scikit-learn matplotlib scipy

Usage:
    python claude_p8_b_0.py
    python claude_p8_b_0.py --auto-tune              # VRAM probe + grid → mip_tune_best.json
    python claude_p8_b_0.py --auto-tune-apply        # tune then full run with best config
    python claude_p8_b_0.py --config mip_tune_best.json
    python claude_p8_b_0.py --no-dual --no-tta --no-roi
"""

import argparse
import json
import sys
import math
import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, List, Optional, Tuple
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
CENTROIDS_JSON = BASE_DIR / "centroids_results.json"

OUTPUT_DIR     = BASE_DIR / "mip_models_v2"
OUTPUT_RESULTS = BASE_DIR / "mip_results_v2.json"
OUTPUT_REPORT  = BASE_DIR / "mip_report_v2.txt"
OUTPUT_PLOT    = BASE_DIR / "mip_results_v2_plot.png"

OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE    = 160        # 128→160: sharper MIPs without 256× memory cost
N_FOLDS     = 5
EPOCHS      = 100
BATCH_SIZE  = 8
LR          = 1e-3
WEIGHT_DECAY= 1e-4
EARLY_STOP  = 20

# Sharper softmax peaks → tighter soft-argmax (lower = peakier; too low = unstable)
SOFTMAX_TEMP = 1.25

# Gaussian heatmap σ as fraction of [0,1] plane (wider = easier; narrower = sharper peaks)
HM_SIGMA_NORM = 0.018
# Weight for heatmap matching vs coordinate regression (sum should stabilize training)
LOSS_HM_WEIGHT = 0.65
LOSS_COORD_WEIGHT = 0.35

MIP_CLIP_HU = 3000       # electrodes > 2000 HU survive at full brightness

# 2D MIP ROI (no GT bbox leak — inference-time geometry only)
ROI_MIP_ENABLE   = True
ROI_MODE         = "polar"   # "polar" = 360-ray star hull bbox; "square" = legacy CoM window
ROI_POLY_N       = 360       # rays around weighted center (polygon → tight axis-aligned crop)
ROI_MARGIN_FRAC  = 0.10      # pad bbox as fraction of max(w,h) of the hull
ROI_MIN_CROP_FRAC = 0.40     # never crop tighter than this × min(full image sides) on disk
ROI_MASK_DILATE  = 1         # binary-dilate bright mask (bridges tiny gaps in MIP)
ROI_POLAR_UNION_SQUARE = True  # union polar bbox with legacy centered square (disjoint leads)
ROI_WINDOW_FRAC  = 0.58      # min(side) × frac — square mode & union window
ROI_BRIGHT_Q     = 0.88      # slightly lower quantile → larger bright mask, fewer missed leads

# Auto-tune quick pass (full run unchanged unless --auto-tune)
TUNE_SUBSET_PATIENTS = 12
TUNE_EPOCHS          = 16
TUNE_EARLY_STOP      = 7
TUNE_N_FOLDS         = 3
TUNE_STRUCT          = "LL1"
TUNE_PROJ            = "axial"
TUNE_BEST_JSON       = BASE_DIR / "mip_tune_best.json"

# All structures we train / triangulate (anatomy first — used for bullseye normalization)
STRUCTURES   = ["ANT", "APEX", "BASE", "LL1", "LL2", "LL3", "LL4", "RL1", "RL2"]
ELECTRODES   = ["LL1", "LL2", "LL3", "LL4", "RL1", "RL2"]  # LV/RV leads only

# Segmentation label → canonical name (matches centroids_results.json naming)
LABEL_TO_CANON = {
    4001: "ANT", 4002: "APEX", 4003: "BASE",
    4004: "LL1", 4005: "LL2", 4006: "LL3", 4007: "LL4",
    4008: "RL1", 4009: "RL2",
}
# Alternate names stored in some JSON rows (defensive)
CENTROID_NAME_TO_CANON = {
    "ant": "ANT", "apex": "APEX", "base": "BASE",
    "lv_distal": "LL1", "lv_2": "LL2", "lv_3": "LL3", "lv_proximal": "LL4",
    "rv_distal": "RL1", "rv_proximal": "RL2",
}


def canon_name_for_centroid_entry(lbl_int: int, entry: dict) -> Optional[str]:
    """Resolve centroid dict entry to a STRUCTURES key."""
    if lbl_int in LABEL_TO_CANON:
        return LABEL_TO_CANON[lbl_int]
    raw = (entry or {}).get("name") or ""
    key = raw.strip().lower().replace(" ", "_")
    return CENTROID_NAME_TO_CANON.get(key)

# Projection definitions: which nibabel axis to collapse,
# and which two nibabel dims the CNN predicts (row=first, col=second)
PROJECTIONS = {
    "axial":    {"collapse": 0, "pred_dims": (1, 2)},  # predicts j_norm, k_norm
    "coronal":  {"collapse": 1, "pred_dims": (0, 2)},  # predicts i_norm, k_norm
    "sagittal": {"collapse": 2, "pred_dims": (0, 1)},  # predicts i_norm, j_norm
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    """Hyperparameters + feature flags for one training / eval run."""
    img_size: int = field(default_factory=lambda: IMG_SIZE)
    hm_sigma_norm: float = field(default_factory=lambda: HM_SIGMA_NORM)
    loss_hm_weight: float = field(default_factory=lambda: LOSS_HM_WEIGHT)
    loss_coord_weight: float = field(default_factory=lambda: LOSS_COORD_WEIGHT)
    softmax_temp: float = field(default_factory=lambda: SOFTMAX_TEMP)
    n_folds: int = field(default_factory=lambda: N_FOLDS)
    epochs: int = field(default_factory=lambda: EPOCHS)
    batch_size: int = field(default_factory=lambda: BATCH_SIZE)
    early_stop: int = field(default_factory=lambda: EARLY_STOP)
    lr: float = field(default_factory=lambda: LR)
    weight_decay: float = field(default_factory=lambda: WEIGHT_DECAY)
    use_roi_mip: bool = field(default_factory=lambda: ROI_MIP_ENABLE)
    roi_mode: str = field(default_factory=lambda: ROI_MODE)
    roi_poly_n: int = field(default_factory=lambda: ROI_POLY_N)
    roi_margin_frac: float = field(default_factory=lambda: ROI_MARGIN_FRAC)
    roi_min_crop_frac: float = field(default_factory=lambda: ROI_MIN_CROP_FRAC)
    roi_mask_dilate: int = field(default_factory=lambda: ROI_MASK_DILATE)
    roi_polar_union_square: bool = field(default_factory=lambda: ROI_POLAR_UNION_SQUARE)
    roi_window_frac: float = field(default_factory=lambda: ROI_WINDOW_FRAC)
    roi_bright_q: float = field(default_factory=lambda: ROI_BRIGHT_Q)
    use_dual_head: bool = True
    use_tta_inference: bool = True

    @classmethod
    def from_globals(cls) -> "RunConfig":
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_path(cls, path: Path) -> "RunConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d.get("config"), dict):
            d = d["config"]
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ─────────────────────────────────────────────────────────────────────────────
#  COORDINATE HELPERS  (the fixed versions)
# ─────────────────────────────────────────────────────────────────────────────

def world_lps_to_voxel_norm(world_xyz: list,
                              affine:    np.ndarray,
                              shape:     tuple) -> np.ndarray:
    """
    World LPS mm  →  normalised voxel (i_norm, j_norm, k_norm) in [0,1].

    Steps:
      1. LPS → RAS (negate X and Y)
      2. RAS → voxel (apply inv affine)
      3. Divide by volume shape to get [0,1]

    This is the ONLY target representation we use — everything stays in
    voxel space until the very last step.
    """
    xyz_lps = np.array(world_xyz, dtype=float)
    xyz_ras = np.array([-xyz_lps[0], -xyz_lps[1], xyz_lps[2], 1.0])
    ijk     = (np.linalg.inv(affine) @ xyz_ras)[:3]
    return np.clip(ijk / np.array(shape, dtype=float), 0.0, 1.0)


def voxel_norm_to_world_lps(ijk_norm:  np.ndarray,
                              affine:    np.ndarray,
                              shape:     tuple) -> np.ndarray:
    """
    Normalised voxel (i_norm, j_norm, k_norm)  →  world LPS mm.
    The SINGLE affine conversion after triangulation.
    """
    ijk = ijk_norm * np.array(shape, dtype=float)
    ras = affine @ np.array([ijk[0], ijk[1], ijk[2], 1.0])
    return np.array([-ras[0], -ras[1], ras[2]])


def longitudinal_height(p_lps: np.ndarray,
                        apex_lps: np.ndarray,
                        base_lps: np.ndarray) -> float:
    """
    Normalised height along Apex→Base: 0 at apex, 1 at base (clamped).
    Values outside the segment still clamp to [0,1] for stability.
    """
    p = np.asarray(p_lps, dtype=float).reshape(3)
    a = np.asarray(apex_lps, dtype=float).reshape(3)
    b = np.asarray(base_lps, dtype=float).reshape(3)
    u = b - a
    L = float(np.linalg.norm(u))
    if L < 1e-6:
        return float("nan")
    u = u / L
    t = float(np.dot(p - a, u) / L)
    return float(np.clip(t, 0.0, 1.0))


def clock_degrees_short_axis(p_lps: np.ndarray,
                             apex_lps: np.ndarray,
                             base_lps: np.ndarray,
                             ant_lps: np.ndarray) -> float:
    """
    Angle (degrees, [0,360)) in the short-axis plane at the electrode:
    0° aligns with the projection of (ANT − Apex) onto that plane
    (anterior reference); increasing angle follows right-hand rule about
    Apex→Base axis.
    """
    p = np.asarray(p_lps, dtype=float).reshape(3)
    a = np.asarray(apex_lps, dtype=float).reshape(3)
    b = np.asarray(base_lps, dtype=float).reshape(3)
    ant = np.asarray(ant_lps, dtype=float).reshape(3)
    u = b - a
    Lu = float(np.linalg.norm(u))
    if Lu < 1e-6:
        return float("nan")
    u = u / Lu
    ref = ant - a
    ref_p = ref - np.dot(ref, u) * u
    v = p - a
    v_p = v - np.dot(v, u) * u
    nr = float(np.linalg.norm(ref_p))
    nv = float(np.linalg.norm(v_p))
    if nr < 1e-3 or nv < 1e-3:
        return float("nan")
    e1 = ref_p / nr
    e2 = np.cross(u, e1)
    ne2 = float(np.linalg.norm(e2))
    if ne2 < 1e-6:
        return float("nan")
    e2 = e2 / ne2
    ang = math.degrees(math.atan2(float(np.dot(v_p, e2)), float(np.dot(v_p, e1))))
    return (ang % 360.0 + 360.0) % 360.0


def sanity_check_roundtrip(world_xyz, affine, shape, tol_mm=0.5):
    """
    Verify world → voxel_norm → world round-trips cleanly.
    Prints a warning if the round-trip error exceeds tol_mm.
    """
    norm = world_lps_to_voxel_norm(world_xyz, affine, shape)
    recovered = voxel_norm_to_world_lps(norm, affine, shape)
    err = float(np.linalg.norm(np.array(world_xyz) - recovered))
    if err > tol_mm:
        print(f"  ⚠  Round-trip error {err:.2f} mm > {tol_mm} mm — "
              f"check affine!")
    return err


# ─────────────────────────────────────────────────────────────────────────────
#  MIP GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_mip(data: np.ndarray, collapse_axis: int) -> np.ndarray:
    """Maximum Intensity Projection, clipped and normalised to [0,1]."""
    mip = data.max(axis=collapse_axis)
    mip = np.clip(mip, -1000, MIP_CLIP_HU)
    mip = (mip + 1000) / (MIP_CLIP_HU + 1000)
    return mip.astype(np.float32)


def resize_2d(img: np.ndarray, size: int) -> np.ndarray:
    factors = (size / img.shape[0], size / img.shape[1])
    return ndi.zoom(img, factors, order=1).astype(np.float32)


MipRecord = Dict[str, object]  # {"img": ndarray, "roi": None | (r0,r1,c0,c1)}


def mip_plane_from_record(rec: MipRecord) -> np.ndarray:
    return np.asarray(rec["img"], dtype=np.float32)


def brightness_roi_crop(
    mip_full: np.ndarray,
    out_size: int,
    window_frac: float,
    bright_q: float,
    enabled: bool,
) -> Tuple[np.ndarray, Optional[Tuple[float, float, float, float]]]:
    """
    Crop a square window around the intensity-weighted CoM of bright voxels,
    then resize to out_size.  Returns (patch, roi) where roi is
    (r0, r1, c0, c1) on the *normalised* full MIP grid [0,1]×[0,1].
    """
    h, w = mip_full.shape
    if not enabled or h < 8 or w < 8:
        return mip_full.astype(np.float32), None

    thr = float(np.quantile(mip_full, bright_q))
    mask = mip_full >= thr
    if not np.any(mask):
        return mip_full.astype(np.float32), None

    yy, xx = np.nonzero(mask)
    cy = float(yy.mean())
    cx = float(xx.mean())
    side = int(max(16, min(h, w) * window_frac))
    y0 = int(np.clip(cy - side / 2, 0, max(0, h - side)))
    x0 = int(np.clip(cx - side / 2, 0, max(0, w - side)))
    y1 = int(min(h, y0 + side))
    x1 = int(min(w, x0 + side))
    if y1 <= y0 or x1 <= x0:
        return mip_full.astype(np.float32), None

    patch = mip_full[y0:y1, x0:x1]
    patch = resize_2d(patch, out_size)
    r0, r1 = y0 / h, y1 / h
    c0, c1 = x0 / w, x1 / w
    return patch.astype(np.float32), (r0, r1, c0, c1)


def _weighted_centroid_on_mask(
    mip: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """Intensity²-weighted centroid on bright pixels (emphasises metal over bone rim)."""
    yy, xx = np.nonzero(mask)
    if len(yy) == 0:
        return mip.shape[0] / 2.0, mip.shape[1] / 2.0
    vals = mip[yy, xx].astype(np.float64)
    wts = np.maximum(vals - float(np.quantile(mip, 0.5)), 0.0) ** 2 + 1e-8
    sw = float(wts.sum())
    cy = float((yy.astype(np.float64) * wts).sum() / sw)
    cx = float((xx.astype(np.float64) * wts).sum() / sw)
    return cy, cx


def _snap_center_into_mask(
    cy: float,
    cx: float,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """If centroid falls in a gap, snap to nearest masked pixel."""
    h, w = mask.shape
    ri, ci = int(np.clip(round(cy), 0, h - 1)), int(np.clip(round(cx), 0, w - 1))
    if mask[ri, ci]:
        return cy, cx
    dist = ndi.distance_transform_edt(~mask)
    jj, ii = np.unravel_index(np.argmin(dist), dist.shape)
    return float(jj), float(ii)


def polar_polygon_roi_crop(
    mip_full: np.ndarray,
    out_size: int,
    bright_q: float,
    n_rays: int,
    margin_frac: float,
    min_crop_frac: float,
    mask_dilate: int,
    union_square: bool,
    window_frac: float,
    enabled: bool,
) -> Tuple[np.ndarray, Optional[Tuple[float, float, float, float]]]:
    """
    “360-gon” polar ROI: from an intensity-weighted seed inside the bright mask,
    march along each ray *outward* and record the farthest masked pixel per angle.
    The axis-aligned bbox of that star-shaped polygon is padded and cropped — this
    tends to surround elongated / multi-blob metal (leads + staples) better than
    a fixed square on the global CoM.

    Falls back to full field if the hull is degenerate.
    """
    h, w = mip_full.shape
    if not enabled or h < 8 or w < 8:
        return mip_full.astype(np.float32), None

    thr = float(np.quantile(mip_full, bright_q))
    mask = mip_full >= thr
    if not np.any(mask):
        return mip_full.astype(np.float32), None

    if mask_dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=int(mask_dilate))

    cy, cx = _weighted_centroid_on_mask(mip_full, mask)
    cy, cx = _snap_center_into_mask(cy, cx, mask)

    r_max_len = float(math.hypot(h, w)) + 2.0
    dt = 0.75
    verts_r: List[float] = []
    verts_c: List[float] = []
    n_rays = max(36, min(n_rays, 720))

    for k in range(n_rays):
        ang = 2.0 * math.pi * k / n_rays
        sin_a, cos_a = math.sin(ang), math.cos(ang)
        r_best = 0.0
        t = 0.0
        while t <= r_max_len:
            ri = int(round(cy + t * sin_a))
            ci = int(round(cx + t * cos_a))
            if ri < 0 or ri >= h or ci < 0 or ci >= w:
                break
            if mask[ri, ci]:
                r_best = t
            t += dt
        # hull vertex at farthest in-mask distance along ray (from center)
        verts_r.append(cy + r_best * sin_a)
        verts_c.append(cx + r_best * cos_a)

    # Include center so degenerate rays still yield a minimal box
    verts_r.append(cy)
    verts_c.append(cx)

    yr = np.clip(np.array(verts_r, dtype=np.float64), 0, h - 1)
    xc = np.clip(np.array(verts_c, dtype=np.float64), 0, w - 1)
    y0f, y1f = float(yr.min()), float(yr.max())
    x0f, x1f = float(xc.min()), float(xc.max())

    mh = max(y1f - y0f, 1.0)
    mw = max(x1f - x0f, 1.0)
    pad = margin_frac * max(mh, mw, float(min(h, w)))
    y0 = int(math.floor(y0f - pad))
    y1 = int(math.ceil(y1f + pad))
    x0 = int(math.floor(x0f - pad))
    x1 = int(math.ceil(x1f + pad))

    if union_square:
        side = int(max(16, min(h, w) * window_frac))
        sq_y0 = int(np.clip(round(cy - side / 2), 0, max(0, h - side)))
        sq_x0 = int(np.clip(round(cx - side / 2), 0, max(0, w - side)))
        sq_y1 = int(min(h, sq_y0 + side))
        sq_x1 = int(min(w, sq_x0 + side))
        y0 = min(y0, sq_y0)
        y1 = max(y1, sq_y1)
        x0 = min(x0, sq_x0)
        x1 = max(x1, sq_x1)

    # Enforce minimum crop footprint on full-res MIP (avoid tiny crops that clip leads)
    min_side = min_crop_frac * float(min(h, w))
    ch, cw = y1 - y0, x1 - x0
    if ch < min_side:
        d = (min_side - ch) / 2.0
        y0 = int(math.floor(y0 - d))
        y1 = int(math.ceil(y1 + d))
    if cw < min_side:
        d = (min_side - cw) / 2.0
        x0 = int(math.floor(x0 - d))
        x1 = int(math.ceil(x1 + d))

    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, y0 + 1, h))
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, x0 + 1, w))

    patch = mip_full[y0:y1, x0:x1]
    patch = resize_2d(patch, out_size)
    r0, r1 = y0 / h, y1 / h
    c0, c1 = x0 / w, x1 / w
    return patch.astype(np.float32), (r0, r1, c0, c1)


def roi_crop_mip(
    mip_full: np.ndarray,
    out_size: int,
    cfg: RunConfig,
) -> Tuple[np.ndarray, Optional[Tuple[float, float, float, float]]]:
    """Dispatch ROI: polar star-hull vs legacy square."""
    if not cfg.use_roi_mip:
        return mip_full.astype(np.float32), None
    mode = (cfg.roi_mode or "polar").lower().strip()
    if mode == "square":
        return brightness_roi_crop(
            mip_full, out_size, cfg.roi_window_frac, cfg.roi_bright_q, True)
    return polar_polygon_roi_crop(
        mip_full,
        out_size,
        cfg.roi_bright_q,
        cfg.roi_poly_n,
        cfg.roi_margin_frac,
        cfg.roi_min_crop_frac,
        cfg.roi_mask_dilate,
        cfg.roi_polar_union_square,
        cfg.roi_window_frac,
        True,
    )


def remap_target_to_roi(
    row_norm: float,
    col_norm: float,
    roi: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Map full-plane normalised coords into ROI-normalised [0,1]."""
    r0, r1, c0, c1 = roi
    dr = max(r1 - r0, 1e-6)
    dc = max(c1 - c0, 1e-6)
    tr = (row_norm - r0) / dr
    tc = (col_norm - c0) / dc
    return (
        float(np.clip(tr, 0.0, 1.0)),
        float(np.clip(tc, 0.0, 1.0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MIPDataset(Dataset):
    """
    Each item: one (patient, projection, electrode) combination.
        image  : (1, IMG_SIZE, IMG_SIZE)  — normalised MIP
        target : (2,)  — (row_vox_norm, col_vox_norm)  ← VOXEL SPACE
        pid    : str
    """
    def __init__(self, items, augment=False):
        self.items   = items
        self.augment = augment

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img  = item["mip"].copy()
        tgt  = item["target"].copy()

        if self.augment:
            # horizontal flip
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                tgt = np.array([tgt[0], 1.0 - tgt[1]])
            # vertical flip
            if np.random.rand() < 0.5:
                img = img[::-1, :].copy()
                tgt = np.array([1.0 - tgt[0], tgt[1]])
            # brightness jitter
            img = np.clip(img + np.random.uniform(-0.04, 0.04), 0.0, 1.0)

        return (torch.from_numpy(img).unsqueeze(0),
                torch.from_numpy(tgt.astype(np.float32)),
                item["pid"])


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL  — heatmap + soft-argmax (global average pool cannot localize)
# ─────────────────────────────────────────────────────────────────────────────

def spatial_softmax_coords(
    logits: torch.Tensor,
    temp: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits: (B, H, W)  →  coord (B, 2) in [0,1], p same shape (normalised mass).
    """
    flat = logits.flatten(1) / temp
    p = torch.softmax(flat, dim=1).view_as(logits)
    h, w = p.shape[1], p.shape[2]
    device, dtype = p.device, p.dtype
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w)
    row = (p * ys).sum(dim=(1, 2))
    col = (p * xs).sum(dim=(1, 2))
    coord = torch.stack([row, col], dim=1)
    return coord, p


class MIPCoordNet(nn.Module):
    """
    Predicts (row_norm, col_norm) via a 1-channel heatmap upsampled to IMG_SIZE
    and a differentiable soft-argmax.  This keeps full spatial resolution
    instead of collapsing the feature map to a single vector (Step 8's main
    weakness for a tiny bright spot in a large MIP).
    """
    def __init__(self, img_size: int = IMG_SIZE, temp: float = SOFTMAX_TEMP):
        super().__init__()
        self.img_size = img_size
        self.temp     = temp
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2))
        self.hm = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x, return_logits: bool = False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        hm = self.hm(x)
        hm = F.interpolate(hm, size=(self.img_size, self.img_size),
                           mode="bilinear", align_corners=False)
        logits = hm.squeeze(1)
        coord, _p = spatial_softmax_coords(logits, self.temp)
        if return_logits:
            return coord, logits
        return coord


class ResidualConv2d(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.c2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.bn1(self.c1(x)), inplace=True)
        z = self.bn2(self.c2(z))
        return F.relu(z + self.skip(x), inplace=True)


class LightResidualUNet(nn.Module):
    """
    Lightweight residual U-Net on the MIP — second path for fused logits.
    """
    def __init__(self, img_size: int):
        super().__init__()
        self.img_size = img_size
        self.in_stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), ResidualConv2d(32, 64))
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), ResidualConv2d(64, 96))
        self.mid = ResidualConv2d(96, 96)
        self.dec2 = ResidualConv2d(96 + 64, 64)
        self.dec1 = ResidualConv2d(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        xm = self.mid(x2)
        u2 = F.interpolate(xm, size=x1.shape[2:], mode="bilinear", align_corners=False)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.dec2(u2)
        u1 = F.interpolate(u2, size=x0.shape[2:], mode="bilinear", align_corners=False)
        u1 = torch.cat([u1, x0], dim=1)
        u1 = self.dec1(u1)
        lo = self.out(u1)
        if lo.shape[2] != self.img_size or lo.shape[3] != self.img_size:
            lo = F.interpolate(lo, size=(self.img_size, self.img_size),
                               mode="bilinear", align_corners=False)
        return lo.squeeze(1)


class DualMIPLocalizer(nn.Module):
    """Fuses CNN heatmap logits + U-Net logits before spatial softmax."""
    def __init__(self, img_size: int, temp: float = SOFTMAX_TEMP):
        super().__init__()
        self.img_size = img_size
        self.temp = float(temp)
        self.cnn = MIPCoordNet(img_size=img_size, temp=temp)
        self.unet = LightResidualUNet(img_size=img_size)
        self.unet_gain = nn.Parameter(torch.tensor(0.85))

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        # CNN logits (strip final softmax by re-running backbone)
        t = self.cnn.temp
        z = self.cnn.block1(x)
        z = self.cnn.block2(z)
        z = self.cnn.block3(z)
        hm = self.cnn.hm(z)
        hm = F.interpolate(hm, size=(self.img_size, self.img_size),
                           mode="bilinear", align_corners=False)
        logits_cnn = hm.squeeze(1)
        logits_u = self.unet(x)
        logits = logits_cnn + self.unet_gain * logits_u
        coord, _p = spatial_softmax_coords(logits, t)
        if return_logits:
            return coord, logits
        return coord


def build_localizer(cfg: RunConfig) -> nn.Module:
    if cfg.use_dual_head:
        return DualMIPLocalizer(cfg.img_size, cfg.softmax_temp).to(DEVICE)
    return MIPCoordNet(cfg.img_size, cfg.softmax_temp).to(DEVICE)


@torch.no_grad()
def tta_predict_coords(model: nn.Module, mip: np.ndarray, use_tta: bool) -> np.ndarray:
    """Average (row,col) over identity + flips (inference only)."""
    x = torch.from_numpy(mip).view(1, 1, mip.shape[0], mip.shape[1]).to(DEVICE)
    if not use_tta:
        return model(x).cpu().numpy()[0]

    acc = torch.zeros(1, 2, device=DEVICE, dtype=x.dtype)
    n = 0
    for hf in (False, True):
        for vf in (False, True):
            xt = x
            if hf:
                xt = torch.flip(xt, dims=(3,))
            if vf:
                xt = torch.flip(xt, dims=(2,))
            pred = model(xt)
            pr = pred.clone()
            if hf:
                pr[:, 1] = 1.0 - pr[:, 1]
            if vf:
                pr[:, 0] = 1.0 - pr[:, 0]
            acc += pr
            n += 1
    return (acc / n).cpu().numpy()[0]


def probe_max_img_size(
    dual: bool,
    batch_size: int,
    device: torch.device,
) -> int:
    """Largest spatial size that survives a forward+backward OOM probe."""
    if device.type != "cuda":
        return min(192, IMG_SIZE)

    sizes = [256, 224, 192, 176, 160, 144, 128]
    best = 128
    for sz in sizes:
        torch.cuda.empty_cache()
        try:
            cfg_try = RunConfig(
                img_size=sz, batch_size=batch_size,
                use_dual_head=dual, use_roi_mip=False,
                use_tta_inference=False)
            m = build_localizer(cfg_try)
            x = torch.zeros(batch_size, 1, sz, sz, device=device)
            y = m(x).sum()
            y.backward()
            del m, x, y
            torch.cuda.empty_cache()
            best = sz
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise
    return best


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_maps_2d(
    row_norm: torch.Tensor,
    col_norm: torch.Tensor,
    h: int,
    w: int,
    sigma: float,
) -> torch.Tensor:
    """
    Normalised 2D Gaussians (each sums to 1) at (row_norm, col_norm) in [0,1].
    row_norm → first image axis (y), col_norm → second (x), matching soft-argmax.
    """
    device, dtype = row_norm.device, row_norm.dtype
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype).view(1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype).view(1, 1, w)
    br = row_norm.view(-1, 1, 1)
    bc = col_norm.view(-1, 1, 1)
    g = torch.exp(-((ys - br) ** 2 + (xs - bc) ** 2) / (2.0 * sigma ** 2)) + 1e-12
    return g / g.sum(dim=(1, 2), keepdim=True)


def train_one_fold(train_items, val_items, cfg: RunConfig):
    bs = cfg.batch_size
    train_dl = DataLoader(MIPDataset(train_items, augment=True),
                          batch_size=bs, shuffle=True,
                          drop_last=len(train_items) > bs)
    val_dl   = DataLoader(MIPDataset(val_items, augment=False),
                          batch_size=bs, shuffle=False)

    model     = build_localizer(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.epochs, eta_min=cfg.lr / 20)
    coord_loss = nn.SmoothL1Loss()

    best_val, best_state, patience = float("inf"), deepcopy(model.state_dict()), 0

    def batch_loss(imgs: torch.Tensor, tgts: torch.Tensor) -> torch.Tensor:
        pred_coord, logits = model(imgs, return_logits=True)
        flat = logits.flatten(1) / float(model.temp)
        p = torch.softmax(flat, dim=1).view_as(logits)
        g = gaussian_maps_2d(tgts[:, 0], tgts[:, 1], logits.shape[1], logits.shape[2],
                             cfg.hm_sigma_norm)
        hm_term = F.mse_loss(p, g)
        c_term = coord_loss(pred_coord, tgts)
        return (cfg.loss_hm_weight * hm_term
                + cfg.loss_coord_weight * c_term)

    for epoch in range(cfg.epochs):
        model.train()
        for imgs, tgts, _ in train_dl:
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            optimizer.zero_grad()
            batch_loss(imgs, tgts).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = []
        with torch.no_grad():
            for imgs, tgts, _ in val_dl:
                imgs_d, tgts_d = imgs.to(DEVICE), tgts.to(DEVICE)
                val_loss.append(batch_loss(imgs_d, tgts_d).item())
        vl = float(np.mean(val_loss)) if val_loss else float("inf")

        if vl < best_val:
            best_val, best_state, patience = vl, deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= cfg.early_stop:
                break

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  DATA BUILD + AUTO-TUNE
# ─────────────────────────────────────────────────────────────────────────────

def compute_patient_data(
    gt_patients: List[dict],
    centroids: dict,
    cfg: RunConfig,
) -> dict:
    """
    Load CT, build per-projection MIPs (optional ROI), voxel-space targets.
    Returns patient_data dict keyed by pid.
    """
    patient_data = {}
    print("\n  Pre-computing MIPs and voxel-space targets...")
    print(f"  img_size={cfg.img_size}  ROI={cfg.use_roi_mip} "
          f"mode={getattr(cfg, 'roi_mode', 'polar')}  dual={cfg.use_dual_head}")
    print(f"  {'PID':<8}  {'Shape':>16}  Round-trip sanity")
    print("  " + "─" * 46)

    for pt in gt_patients:
        pid, img_path = pt["pid"], pt["img_path"]
        try:
            nii    = nib.as_closest_canonical(nib.load(img_path))
            data   = nii.get_fdata(dtype=np.float32)
            affine = nii.affine
        except Exception as e:
            print(f"  {pid:<8}  ERROR: {e}")
            continue

        shape = data.shape
        gt_cent = centroids[pid].get("centroids", {})
        ijk_by_name: Dict[str, np.ndarray] = {}
        rt_errs = []

        for lbl_str, entry in gt_cent.items():
            name = canon_name_for_centroid_entry(int(lbl_str), entry)
            if name is None or name not in STRUCTURES:
                continue
            rt_err = sanity_check_roundtrip(entry["world_xyz"], affine, shape)
            rt_errs.append(rt_err)
            ijk_by_name[name] = world_lps_to_voxel_norm(
                entry["world_xyz"], affine, shape)

        mips: Dict[str, MipRecord] = {}
        targets: Dict[str, Dict[str, Tuple[float, float]]] = {}

        for proj_name, pcfg in PROJECTIONS.items():
            raw_mip = make_mip(data, pcfg["collapse"])
            mip_full = resize_2d(raw_mip, cfg.img_size)
            mip_img, roi = roi_crop_mip(mip_full, cfg.img_size, cfg)
            mips[proj_name] = {"img": mip_img, "roi": roi}

            d0, d1 = pcfg["pred_dims"]
            for name, ijk in ijk_by_name.items():
                row_norm = float(ijk[d0])
                col_norm = float(ijk[d1])
                if roi is not None:
                    row_norm, col_norm = remap_target_to_roi(row_norm, col_norm, roi)
                if name not in targets:
                    targets[name] = {}
                targets[name][proj_name] = (row_norm, col_norm)

        patient_data[pid] = {
            "mips": mips,
            "targets": targets,
            "shape": shape,
            "affine": affine,
        }
        mean_rt = float(np.mean(rt_errs)) if rt_errs else 0.0
        print(f"  {pid:<8}  {str(shape):>16}  "
              f"round-trip={mean_rt:.3f} mm {'✅' if mean_rt < 0.5 else '⚠'}")

    return patient_data


def run_auto_tune(
    gt_patients: List[dict],
    centroids: dict,
    base_cfg: RunConfig,
) -> RunConfig:
    """
    VRAM-safe IMG_SIZE probe + coarse grid on HM σ and loss weights.
    Uses a tiny surrogate task (one structure / one projection / subset).
    """
    print("\n" + "=" * 65)
    print("  AUTO-TUNE: VRAM probe + quick surrogate grid")
    print("=" * 65)

    max_img = probe_max_img_size(
        dual=base_cfg.use_dual_head, batch_size=base_cfg.batch_size,
        device=DEVICE)
    print(f"  Probe: max img_size ≈ {max_img} (batch={base_cfg.batch_size}, "
          f"dual={base_cfg.use_dual_head})")

    img_grid = [s for s in (256, 224, 192, 176, 160, 144, 128) if s <= max_img]
    if not img_grid:
        img_grid = [128]
    # keep largest few to cap wall time
    img_grid = img_grid[: min(4, len(img_grid))]

    sigmas = (0.014, 0.018, 0.024)
    loss_grid = ((0.7, 0.3), (0.65, 0.35), (0.55, 0.45))

    sub_pids = [p["pid"] for p in gt_patients[:TUNE_SUBSET_PATIENTS]]
    sub_pts = [p for p in gt_patients if p["pid"] in set(sub_pids)]

    best = None  # (score, cfg_dict)
    for img_sz in img_grid:
        cfg_sz = RunConfig(**{**base_cfg.to_dict(), "img_size": img_sz})
        pdata = compute_patient_data(sub_pts, centroids, cfg_sz)
        if len(pdata) < 4:
            continue

        pid_arr = np.array(list(pdata.keys()))
        kf = KFold(n_splits=min(TUNE_N_FOLDS, len(pid_arr)), shuffle=True,
                   random_state=42)

        for sig in sigmas:
            for hm_w, c_w in loss_grid:
                cfg_try = RunConfig(**{
                    **cfg_sz.to_dict(),
                    "hm_sigma_norm": sig,
                    "loss_hm_weight": hm_w,
                    "loss_coord_weight": c_w,
                    "epochs": TUNE_EPOCHS,
                    "early_stop": TUNE_EARLY_STOP,
                    "n_folds": min(TUNE_N_FOLDS, len(pid_arr)),
                })
                errs = []
                for tr_idx, va_idx in kf.split(pid_arr):
                    def _items(pids):
                        items = []
                        for pid in pids:
                            if pid not in pdata:
                                continue
                            td = pdata[pid]["targets"].get(TUNE_STRUCT)
                            if not td:
                                continue
                            rc = td.get(TUNE_PROJ)
                            if rc is None:
                                continue
                            items.append({
                                "pid": pid,
                                "mip": mip_plane_from_record(
                                    pdata[pid]["mips"][TUNE_PROJ]),
                                "target": np.array(rc, dtype=np.float32),
                            })
                        return items

                    tr_it = _items(pid_arr[tr_idx].tolist())
                    va_it = _items(pid_arr[va_idx].tolist())
                    if len(tr_it) < 3 or not va_it:
                        continue
                    model = train_one_fold(tr_it, va_it, cfg_try)
                    model.eval()
                    fold_e = []
                    with torch.no_grad():
                        for it in va_it:
                            pr = tta_predict_coords(
                                model, it["mip"], cfg_try.use_tta_inference)
                            fold_e.append(float(np.linalg.norm(pr - it["target"])))
                    errs.append(float(np.mean(fold_e)) if fold_e else 99.0)

                score = float(np.mean(errs)) if errs else 99.0
                print(f"    img={img_sz} σ={sig:.3f} hm={hm_w:.2f}/c={c_w:.2f} "
                      f"→ surrogate MAE={score:.4f}")
                cand = RunConfig(**{
                    **base_cfg.to_dict(),
                    "img_size": img_sz,
                    "hm_sigma_norm": sig,
                    "loss_hm_weight": hm_w,
                    "loss_coord_weight": c_w,
                })
                if best is None or score < best[0]:
                    best = (score, cand)

    if best is None:
        print("  ⚠  Auto-tune found no valid combo; using base config.")
        out = RunConfig(**base_cfg.to_dict())
    else:
        out = RunConfig(**best[1].to_dict())
        print(f"\n  ✅ Best surrogate MAE={best[0]:.4f}  img_size={out.img_size}  "
              f"σ={out.hm_sigma_norm}  hm_w={out.loss_hm_weight}")

    # clamp img to probed max for the final full run
    out.img_size = min(out.img_size, max_img)
    with open(TUNE_BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(out.to_dict(), f, indent=2)
    print(f"  Saved → {TUNE_BEST_JSON}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_patients():
    with open(INVENTORY_JSON, encoding="utf-8") as f:
        inventory = json.load(f)
    with open(CENTROIDS_JSON, encoding="utf-8") as f:
        centroids = json.load(f)
    gt_patients = []
    for ds_key in ("dataset_1", "dataset_2"):
        for pid, rec in inventory.get(ds_key, {}).items():
            if rec.get("img_nii") and pid in centroids:
                if not any(p["pid"] == pid for p in gt_patients):
                    gt_patients.append({"pid": pid, "img_path": rec["img_nii"]})
    return inventory, centroids, gt_patients


def run(cfg: Optional[RunConfig] = None):
    cfg = cfg or RunConfig.from_globals()
    print("=" * 65)
    print(f"  CRT Lead — Step 8b: MIP CNN + dual-head / TTA / ROI")
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  img_size={cfg.img_size}  dual={cfg.use_dual_head}  "
          f"TTA={cfg.use_tta_inference}  ROI={cfg.use_roi_mip}  "
          f"roi_mode={getattr(cfg, 'roi_mode', 'polar')}")
    print("=" * 65)

    _inventory, centroids, gt_patients = load_gt_patients()
    print(f"\n  GT patients with raw CT: {len(gt_patients)}")

    patient_data = compute_patient_data(gt_patients, centroids, cfg)

    valid_pids = list(patient_data.keys())
    print(f"\n  {len(valid_pids)} patients ready\n")

    # ─────────────────────────────────────────────────────────────────────
    #  K-fold CV — one localizer per structure per projection
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  Training: {len(STRUCTURES)} structures × 3 projections × "
          f"{cfg.n_folds} folds = {len(STRUCTURES)*3*cfg.n_folds} runs")
    print("=" * 65)

    kf         = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=42)
    pid_arr    = np.array(valid_pids)

    # pred_store[pid][structure][proj] = (row_pred_vox_norm, col_pred_vox_norm)
    pred_store = {pid: {s: {} for s in STRUCTURES} for pid in valid_pids}

    # 2D pixel errors (in voxel-norm units) for reporting
    pix_errors = {s: {p: [] for p in PROJECTIONS} for s in STRUCTURES}

    def make_items(pids, structure, proj_name):
        items = []
        for pid in pids:
            if pid not in patient_data:
                continue
            tgt_dict = patient_data[pid]["targets"].get(structure)
            if tgt_dict is None:
                continue
            rc = tgt_dict.get(proj_name)
            if rc is None:
                continue
            items.append({
                "pid":    pid,
                "mip":    mip_plane_from_record(
                    patient_data[pid]["mips"][proj_name]),
                "target": np.array(rc, dtype=np.float32),
            })
        return items

    for structure in STRUCTURES:
        for proj_name in PROJECTIONS:
            print(f"\n  [{structure}] [{proj_name}] training {cfg.n_folds} folds...",
                  flush=True)
            fold_maes = []

            for fold_i, (tr_idx, va_idx) in enumerate(kf.split(pid_arr)):
                train_items = make_items(pid_arr[tr_idx].tolist(),
                                         structure, proj_name)
                val_items   = make_items(pid_arr[va_idx].tolist(),
                                         structure, proj_name)
                if len(train_items) < 3 or not val_items:
                    continue

                model = train_one_fold(train_items, val_items, cfg)
                model.eval()

                fold_errs = []
                with torch.no_grad():
                    for item in val_items:
                        pred = tta_predict_coords(
                            model, item["mip"], cfg.use_tta_inference)
                        true  = item["target"]
                        err   = float(np.linalg.norm(pred - true))
                        fold_errs.append(err)
                        pix_errors[structure][proj_name].append(err)

                        # store voxel-norm prediction
                        pred_store[item["pid"]][structure][proj_name] = (
                            float(pred[0]), float(pred[1])
                        )

                mae = np.mean(fold_errs) if fold_errs else float("nan")
                fold_maes.append(mae)
                print(f"    fold {fold_i+1}  n_val={len(val_items):<3}  "
                      f"vox_norm_MAE={mae:.4f}", flush=True)

            # save final model (trained on all data)
            all_items = make_items(valid_pids, structure, proj_name)
            if len(all_items) >= 3:
                final_m = train_one_fold(all_items, all_items[:3], cfg)
                torch.save(final_m.state_dict(),
                           OUTPUT_DIR / f"model_{structure}_{proj_name}.pt")

    # ─────────────────────────────────────────────────────────────────────
    #  TRIANGULATE IN VOXEL SPACE → CONVERT ONCE TO WORLD
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Triangulating: voxel-space median fusion → one affine conversion")
    print("=" * 65)

    triangulated = {}
    all_errors   = []
    per_struct_err = {s: [] for s in STRUCTURES}

    for pid in valid_pids:
        pt          = patient_data[pid]
        affine      = pt["affine"]
        shape       = pt["shape"]
        gt_cent     = centroids[pid].get("centroids", {})
        triangulated[pid] = {}

        for structure in STRUCTURES:
            proj_preds = pred_store[pid][structure]
            if not proj_preds:
                continue

            # Accumulate voxel-norm estimates per dimension
            # dim 0 (i): predicted by coronal_row and sagittal_row
            # dim 1 (j): predicted by axial_row   and sagittal_col
            # dim 2 (k): predicted by axial_col   and coronal_col
            i_ests, j_ests, k_ests = [], [], []

            if "axial" in proj_preds:
                row, col = proj_preds["axial"]
                j_ests.append(row)   # axial pred_dims = (1,2) → row=j, col=k
                k_ests.append(col)

            if "coronal" in proj_preds:
                row, col = proj_preds["coronal"]
                i_ests.append(row)   # coronal pred_dims = (0,2) → row=i, col=k
                k_ests.append(col)

            if "sagittal" in proj_preds:
                row, col = proj_preds["sagittal"]
                i_ests.append(row)   # sagittal pred_dims = (0,1) → row=i, col=j
                j_ests.append(col)

            # median in voxel space (more robust than mean when one MIP is off)
            ijk_norm = np.array([
                float(np.median(i_ests)) if i_ests else 0.5,
                float(np.median(j_ests)) if j_ests else 0.5,
                float(np.median(k_ests)) if k_ests else 0.5,
            ])

            # ONE affine conversion
            pred_xyz = voxel_norm_to_world_lps(ijk_norm, affine, shape)

            # ground truth
            gt_xyz = None
            for lbl_str, entry in gt_cent.items():
                if canon_name_for_centroid_entry(int(lbl_str), entry) == structure:
                    gt_xyz = np.array(entry["world_xyz"])
                    break

            err_mm = (float(np.linalg.norm(pred_xyz - gt_xyz))
                      if gt_xyz is not None else None)

            triangulated[pid][structure] = {
                "pred_xyz":  pred_xyz.tolist(),
                "gt_xyz":    gt_xyz.tolist() if gt_xyz is not None else None,
                "ijk_norm":  ijk_norm.tolist(),
                "err_mm":    round(err_mm, 2) if err_mm is not None else None,
                "n_projections_used": len(proj_preds),
            }

            if err_mm is not None:
                all_errors.append(err_mm)
                per_struct_err[structure].append(err_mm)

    # ── Apex–Base–ANT normalised metrics for LV leads (project goal) ─────
    bullseye_metrics = {}

    def _gt_world(pid: str, name: str) -> Optional[np.ndarray]:
        gc = centroids.get(pid, {}).get("centroids", {})
        for lbl_str, entry in gc.items():
            if canon_name_for_centroid_entry(int(lbl_str), entry) == name:
                return np.array(entry["world_xyz"], dtype=float)
        return None

    def _pred_world(pid: str, name: str) -> Optional[np.ndarray]:
        rec = triangulated.get(pid, {}).get(name)
        if rec and rec.get("pred_xyz") is not None:
            return np.array(rec["pred_xyz"], dtype=float)
        return None

    for pid in valid_pids:
        bullseye_metrics[pid] = {}
        pa, ba, an = (_pred_world(pid, "APEX"), _pred_world(pid, "BASE"),
                      _pred_world(pid, "ANT"))
        apex_b = _gt_world(pid, "APEX")
        base_b = _gt_world(pid, "BASE")
        ant_b = _gt_world(pid, "ANT")
        apex_u = pa if pa is not None else apex_b
        base_u = ba if ba is not None else base_b
        ant_u = an if an is not None else ant_b
        if pa is not None and ba is not None and an is not None:
            lm_src = "predicted_landmarks"
        elif apex_b is not None and base_b is not None and ant_b is not None:
            lm_src = "gt_landmarks_fallback"
        else:
            lm_src = "insufficient_landmarks"

        for lv in ("LL1", "LL2", "LL3", "LL4"):
            lv_rec = triangulated.get(pid, {}).get(lv)
            if not lv_rec or lv_rec.get("pred_xyz") is None:
                continue
            p = np.array(lv_rec["pred_xyz"], dtype=float)
            if lm_src == "insufficient_landmarks" or apex_u is None \
                    or base_u is None or ant_u is None:
                bullseye_metrics[pid][lv] = {
                    "height_norm_along_apex_base": None,
                    "clock_deg_from_ANT": None,
                    "landmark_source": lm_src,
                }
                continue
            bullseye_metrics[pid][lv] = {
                "height_norm_along_apex_base": longitudinal_height(
                    p, apex_u, base_u),
                "clock_deg_from_ANT": clock_degrees_short_axis(
                    p, apex_u, base_u, ant_u),
                "landmark_source": lm_src,
            }

    n_lv_bull = sum(len(bullseye_metrics[pid]) for pid in bullseye_metrics)
    print(f"\n  Bullseye LV metrics (height + clock): {n_lv_bull} rows "
          f"→ field `bullseye_lv_from_predictions` in {OUTPUT_RESULTS.name}")

    # ─────────────────────────────────────────────────────────────────────
    #  RESULTS
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    header = (f"\n  {'Structure':<10}  {'N':>4}  {'Mean err':>10}  "
              f"{'Median':>8}  {'<5mm':>8}  {'<10mm':>8}")
    divider = "  " + "─" * 54
    print(header)
    print(divider)

    report_lines = [
        "Step 8b — MIP CNN Fixed (Voxel-Space Triangulation)",
        "=" * 65,
        header.strip(),
        divider.strip(),
    ]

    for structure in STRUCTURES:
        errs = per_struct_err[structure]
        if not errs:
            continue
        mean_e = np.mean(errs)
        med_e  = np.median(errs)
        pct5   = 100 * sum(e < 5  for e in errs) / len(errs)
        pct10  = 100 * sum(e < 10 for e in errs) / len(errs)
        row = (f"  {structure:<10}  {len(errs):>4}  {mean_e:>8.2f} mm"
               f"  {med_e:>6.2f} mm  {pct5:>6.1f}%  {pct10:>6.1f}%")
        print(row)
        report_lines.append(row)

    if all_errors:
        print(divider)
        ov = (f"  {'OVERALL':<8}  {len(all_errors):>4}  "
              f"{np.mean(all_errors):>8.2f} mm  "
              f"{np.median(all_errors):>6.2f} mm  "
              f"{100*sum(e<5  for e in all_errors)/len(all_errors):>6.1f}%  "
              f"{100*sum(e<10 for e in all_errors)/len(all_errors):>6.1f}%")
        print(ov)
        report_lines += [divider.strip(), ov]

    # per-structure 2D pixel error (sanity check on CNN learning)
    print("\n  2D Pixel Accuracy (voxel-norm MAE — per projection):")
    print(f"  {'Structure':<10}  {'Axial':>12}  {'Coronal':>12}  {'Sagittal':>12}")
    pix_hdr = (f"  {'Structure':<10}  {'Axial':>12}  {'Coronal':>12}  "
               f"{'Sagittal':>12}")
    report_lines += ["", "2D Pixel Accuracy:", pix_hdr]
    for structure in STRUCTURES:
        vals = [pix_errors[structure].get(p, []) for p in PROJECTIONS]
        means = [f"{np.mean(v):.4f}" if v else "N/A" for v in vals]
        row = (f"  {structure:<10}  {means[0]:>12}  {means[1]:>12}  "
               f"{means[2]:>12}")
        print(row)
        report_lines.append(row)

    comparison = [
        "",
        "COMPARISON:",
        f"  Step 2  Seg centroid (GT seg needed)  : ~0.1 mm",
        f"  Step 5b Classical CV (no seg)         :  1.4 mm (when detected)",
        f"  Step 6  PointNet     (no seg)         :  1.2 mm (when detected)",
        f"  Step 8  MIP CNN      (broken)         : 55.6 mm (coord bug)",
        f"  Step 8b MIP CNN      (fixed)          : "
        f"{np.mean(all_errors):.1f} mm  ← this script" if all_errors else "",
        "",
        "INTERPRETING RESULTS:",
        "  Error < 5mm   → MIP CNN is competitive with classical CV",
        "  Error 5-15mm  → useful seed for PointNet; increase IMG_SIZE",
        "  Error > 15mm  → try these improvements:",
        f"    • IMG_SIZE: {cfg.img_size} → 256 (sharper electrode spots)",
        "    • Tune HM_SIGMA_NORM / LOSS_HM_WEIGHT (Gaussian heatmap head)",
        "    • Pre-crop to heart ROI before MIP (reduces background noise)",
        "    • Add pseudo-labeled raw patients from Step 6c",
        "=" * 65,
    ]
    for line in comparison:
        print(line)
    report_lines.extend(comparison)

    # ─────────────────────────────────────────────────────────────────────
    #  PLOT
    # ─────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "Step 8b — MIP CNN: Voxel triangulation + Gaussian heatmap loss\n"
        "Per-structure 3D position error  (5-fold CV)",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    palette = {
        "ANT": "#8c564b", "APEX": "#17becf", "BASE": "#bcbd22",
        "LL1": "#4e79a7", "LL2": "#59a14f", "LL3": "#f28e2b",
        "LL4": "#e15759", "RL1": "#b07aa1", "RL2": "#76b7b2",
    }

    # per-structure error bar
    ax0 = fig.add_subplot(gs[0, :2])
    means = [np.mean(per_struct_err[s]) if per_struct_err[s] else 0
             for s in STRUCTURES]
    stds  = [np.std(per_struct_err[s])  if per_struct_err[s] else 0
             for s in STRUCTURES]
    xpos = np.arange(len(STRUCTURES))
    bars  = ax0.bar(xpos, means, yerr=stds,
                    color=[palette[s] for s in STRUCTURES],
                    alpha=0.85, capsize=4, edgecolor="k", lw=0.6)
    ax0.set_xticks(xpos)
    ax0.set_xticklabels(STRUCTURES, rotation=35, ha="right", fontsize=9)
    ax0.axhline(5,  color="green",  linestyle="--", lw=1.2, label="5mm target")
    ax0.axhline(10, color="orange", linestyle="--", lw=1.2, label="10mm threshold")
    ax0.set_ylabel("3D Error (mm)")
    ax0.set_title("Triangulation error: anatomy + leads (mean ± std)")
    ax0.legend(fontsize=9); ax0.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        if m <= 0:
            continue
        ax0.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3, f"{m:.1f}", ha="center", fontsize=7)

    # histogram
    ax1 = fig.add_subplot(gs[0, 2])
    if all_errors:
        ax1.hist(all_errors, bins=25, color="#4e79a7", alpha=0.8,
                 edgecolor="white")
        ax1.axvline(5,  color="green",  lw=1.5, linestyle="--", label="5mm")
        ax1.axvline(10, color="orange", lw=1.5, linestyle="--", label="10mm")
        ax1.set_xlabel("3D Error (mm)"); ax1.set_ylabel("Count")
        ax1.set_title("Error Distribution"); ax1.legend(fontsize=8)

    # per-projection 2D errors
    for p_idx, proj_name in enumerate(PROJECTIONS):
        ax = fig.add_subplot(gs[1, p_idx])
        p_means = [np.mean(pix_errors[s][proj_name]) if pix_errors[s][proj_name]
                   else 0 for s in STRUCTURES]
        ax.bar(xpos, p_means,
               color=[palette[s] for s in STRUCTURES],
               alpha=0.85, edgecolor="k", lw=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(STRUCTURES, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("Voxel-norm MAE")
        ax.set_title(f"{proj_name.capitalize()} — 2D pixel error")
        ax.grid(axis="y", alpha=0.3)

    plt.savefig(OUTPUT_PLOT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n✅  Plot saved    → {OUTPUT_PLOT}")

    # ── save ──────────────────────────────────────────────────────────────
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                **cfg.to_dict(),
                "DEVICE": str(DEVICE),
                "fix": "voxel_space_triangulation_dual_tta_roi",
            },
            "per_structure_mean_err": {
                s: round(float(np.mean(v)), 2) if v else None
                for s, v in per_struct_err.items()},
            "per_electrode_mean_err": {
                e: round(float(np.mean(per_struct_err[e])), 2)
                if per_struct_err.get(e) else None
                for e in ELECTRODES},
            "overall_mean_err": round(float(np.mean(all_errors)), 2)
                                 if all_errors else None,
            "bullseye_lv_from_predictions": bullseye_metrics,
            "triangulated": triangulated,
        }, f, indent=2)
    print(f"✅  Results saved → {OUTPUT_RESULTS}")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅  Report saved  → {OUTPUT_REPORT}")


if __name__ == "__main__":
    missing = []
    try:
        import torch
        print(f"  PyTorch {torch.__version__}  |  "
              f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        missing.append("torch")
    for lib in [("nibabel","nibabel"),("skimage","scikit-image"),
                ("matplotlib","matplotlib"),("sklearn","scikit-learn")]:
        try: __import__(lib[0])
        except ImportError: missing.append(lib[1])

    if missing:
        print(f"\n⚠  Missing: pip install {' '.join(missing)}")
        sys.exit(1)

    ap = argparse.ArgumentParser(
        description="CRT MIP localiser — dual U-Net head, TTA, ROI MIPs, auto-tune")
    ap.add_argument("--config", type=str, default=None,
                    help="JSON from mip_tune_best.json or results 'config' block")
    ap.add_argument("--auto-tune", action="store_true",
                    help="VRAM probe + quick grid; writes mip_tune_best.json")
    ap.add_argument("--auto-tune-apply", action="store_true",
                    help="Run --auto-tune then full training with best config")
    ap.add_argument("--no-dual", action="store_true", help="Disable U-Net fusion head")
    ap.add_argument("--no-tta", action="store_true", help="Disable test-time augmentation")
    ap.add_argument("--no-roi", action="store_true", help="Disable 2D MIP ROI crop")
    ap.add_argument("--roi-square", action="store_true",
                    help="Use legacy square ROI instead of polar 360-ray hull")
    args = ap.parse_args()

    cfg = RunConfig.from_globals()
    if args.config:
        cfg = RunConfig.from_json_path(Path(args.config))
    if args.no_dual:
        cfg.use_dual_head = False
    if args.no_tta:
        cfg.use_tta_inference = False
    if args.no_roi:
        cfg.use_roi_mip = False
    if args.roi_square:
        cfg.roi_mode = "square"

    if args.auto_tune_apply:
        _inv, centroids, gt_patients = load_gt_patients()
        cfg = run_auto_tune(gt_patients, centroids, cfg)
        run(cfg)
    elif args.auto_tune:
        _inv, centroids, gt_patients = load_gt_patients()
        run_auto_tune(gt_patients, centroids, cfg)
    else:
        run(cfg)