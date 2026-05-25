# ==============================
# STEP 1: DATA LOADING + PREPROCESSING
# ==============================
from __future__ import annotations

import gzip
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import HausdorffDistanceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Rand3DElasticd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.utils import set_determinism


# ------------------------------
# User-validated raw label map
# ------------------------------
RAW_LABEL_MAP: Dict[int, str] = {
    4001: "ANT",           # Anterior wall reference marker
    4002: "Apex",
    4003: "Base",
    4004: "LV_distal",     # LL1
    4005: "LV_2",          # LL2
    4006: "LV_3",          # LL3
    4007: "LV_proximal",   # LL4
    4008: "RV_distal",     # RL1
    4009: "RV_proximal",   # RL2
}

CSV_TO_LABEL: Dict[str, int] = {
    "ANT": 4001,
    "APEX": 4002,
    "BASE": 4003,
    "LL1": 4004,
    "LL2": 4005,
    "LL3": 4006,
    "LL4": 4007,
    "RL1": 4008,
    "RL2": 4009,
}

# ------------------------------
# Contiguous training-space class order
# 0 = background
# 1..6 = electrodes
# 7..9 = anatomy
# ------------------------------
CLASS_NAMES: List[str] = [
    "Background",
    "LL1",
    "LL2",
    "LL3",
    "LL4",
    "RL1",
    "RL2",
    "ANT",
    "Apex",
    "Base",
]

RAW_TO_CONTIG: Dict[int, int] = {
    0: 0,
    4004: 1,  # LL1
    4005: 2,  # LL2
    4006: 3,  # LL3
    4007: 4,  # LL4
    4008: 5,  # RL1
    4009: 6,  # RL2
    4001: 7,  # ANT
    4002: 8,  # Apex
    4003: 9,  # Base
}
CONTIG_TO_RAW = {v: k for k, v in RAW_TO_CONTIG.items() if k != 0}

ELECTRODE_CLASSES = [1, 2, 3, 4, 5, 6]
ANATOMY_CLASSES = [7, 8, 9]

PATIENT_ID_REGEX = re.compile(r"(\d{5})")

DEFAULT_DATASET_ROOTS: Tuple[str, ...] = (
    r"C:\Users\ayw005\Desktop\BENG 280C Project\BENG280C_pacing_lead_data_1st20",
    r"C:\Users\ayw005\Desktop\BENG 280C Project\HCT2_lead_segmentation_training",
)

IMAGE_SUBDIR = "HCT2_img_nii"
LABEL_SUBDIR = "HCT2_leads_seg_nii"


@dataclass
class Config:
    # Raw data layout
    raw_dataset_roots: Tuple[str, ...] = DEFAULT_DATASET_ROOTS
    raw_labeled_images_dir: Optional[str] = None
    raw_labeled_labels_dir: Optional[str] = None
    raw_unlabeled_images_dir: Optional[str] = None

    # Working directories
    work_dir: str = "runs/cardiac_leads_ssl_landmark_debug"

    # Reproducibility
    seed: int = 42

    # Data
    num_classes: int = 10
    labels_already_contiguous: bool = False
    label_dilation_radius_voxels: int = 6
    target_spacing_dhw: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    hu_clip_min: float = -1000.0
    hu_clip_max: float = 3000.0

    # Training shapes
    spatial_dims: int = 3  # set to 2 for slice-wise experiments
    patch_size_3d: Tuple[int, int, int] = (160, 160, 160)
    train_batch_size: int = 1  # recommended = 1 with patch-sampling + AMP
    val_batch_size: int = 1
    samples_per_volume: int = 2
    num_workers: int = 4
    pin_memory: bool = True

    # Class-balanced patch sampling ratios for RandCropByLabelClassesd
    # [bg, LL1, LL2, LL3, LL4, RL1, RL2, ANT, Apex, Base]
    class_sampling_ratios: Tuple[float, ...] = (1, 4, 4, 4, 4, 4, 4, 2, 2, 2)

    # Model
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: Tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    dropout: float = 0.0

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    supervised_epochs: int = 50
    finetune_epochs: int = 0
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4
    amp: bool = True

    # Inference
    infer_overlap: float = 0.5
    sw_batch_size: int = 2

    # Pseudo-labeling
    pseudo_min_prob_electrode: float = 0.985
    pseudo_min_prob_anatomy: float = 0.95
    pseudo_max_entropy: float = 0.20   # normalized entropy threshold
    pseudo_min_foreground_voxels: int = 20
    pseudo_min_case_mean_conf: float = 0.98
    enable_pseudo_labeling: bool = False
    min_supervised_dice_for_pseudo: float = 0.10
    pseudo_weight: float = 0.5

    # Loss
    lambda_dice: float = 1.0
    lambda_ce: float = 1.0

    # Metrics
    hausdorff_percentile: Optional[float] = 95.0  # set None for full HD

    # Visualization
    max_overlay_cases: int = 8

    @property
    def cache_dir(self) -> Path:
        return Path(self.work_dir) / "cache"

    @property
    def labeled_cache_dir(self) -> Path:
        return self.cache_dir / "labeled"

    @property
    def unlabeled_cache_dir(self) -> Path:
        return self.cache_dir / "unlabeled"

    @property
    def pseudo_cache_dir(self) -> Path:
        return self.cache_dir / "pseudo"

    @property
    def weights_dir(self) -> Path:
        return Path(self.work_dir) / "weights"

    @property
    def metrics_dir(self) -> Path:
        return Path(self.work_dir) / "metrics"

    @property
    def plots_dir(self) -> Path:
        return Path(self.work_dir) / "plots"

    @property
    def overlays_dir(self) -> Path:
        return Path(self.work_dir) / "overlays"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(cfg: Config) -> None:
    for d in [
        cfg.cache_dir,
        cfg.labeled_cache_dir,
        cfg.unlabeled_cache_dir,
        cfg.pseudo_cache_dir,
        cfg.weights_dir,
        cfg.metrics_dir,
        cfg.plots_dir,
        cfg.overlays_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def robust_stem(path: Path) -> str:
    name = path.name
    for suffix in [".nii.gz", ".nii", ".npz", ".npy.gz", ".npy"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def supported_files(directory: str) -> List[Path]:
    directory = Path(directory)
    paths = []
    for ext in ["*.nii.gz", "*.nii", "*.npz", "*.npy.gz", "*.npy"]:
        paths.extend(sorted(directory.glob(ext)))
    return sorted(set(paths))


def extract_patient_id(path: Path) -> Optional[str]:
    match = PATIENT_ID_REGEX.search(path.name)
    return match.group(1) if match else None


def _paths_by_patient_id(paths: Sequence[Path]) -> Dict[str, Path]:
    indexed: Dict[str, Path] = {}
    for path in sorted(paths):
        patient_id = extract_patient_id(path)
        if patient_id is not None:
            indexed[patient_id] = path
    return indexed


def pair_labeled_cases(images_dir: str, labels_dir: str) -> List[Tuple[Path, Path]]:
    image_paths = {robust_stem(p): p for p in supported_files(images_dir)}
    label_paths = {robust_stem(p): p for p in supported_files(labels_dir)}
    common = sorted(set(image_paths).intersection(label_paths))
    if not common:
        image_paths = _paths_by_patient_id(supported_files(images_dir))
        label_paths = _paths_by_patient_id(supported_files(labels_dir))
        common = sorted(set(image_paths).intersection(label_paths))
    if not common:
        raise RuntimeError("No paired labeled image/label files were found.")
    return [(image_paths[k], label_paths[k]) for k in common]


def list_unlabeled_cases(images_dir: str) -> List[Path]:
    paths = supported_files(images_dir)
    if not paths:
        raise RuntimeError("No unlabeled image files were found.")
    return paths


def discover_cases_from_dataset_roots(dataset_roots: Sequence[str]) -> Dict[str, List]:
    labeled_by_patient: Dict[str, Tuple[Path, Path]] = {}
    unlabeled_by_patient: Dict[str, Path] = {}

    for root_str in dataset_roots:
        root = Path(root_str)
        images_dir = root / IMAGE_SUBDIR
        labels_dir = root / LABEL_SUBDIR

        if not images_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {labels_dir}")

        image_by_patient = _paths_by_patient_id(supported_files(images_dir))
        label_by_patient = _paths_by_patient_id(supported_files(labels_dir))

        labeled_ids = sorted(set(image_by_patient).intersection(label_by_patient))
        unlabeled_ids = sorted(set(image_by_patient) - set(label_by_patient))

        for patient_id in labeled_ids:
            labeled_by_patient[patient_id] = (image_by_patient[patient_id], label_by_patient[patient_id])
            unlabeled_by_patient.pop(patient_id, None)

        for patient_id in unlabeled_ids:
            if patient_id not in labeled_by_patient:
                unlabeled_by_patient[patient_id] = image_by_patient[patient_id]

    labeled_pairs = [labeled_by_patient[patient_id] for patient_id in sorted(labeled_by_patient)]
    unlabeled_images = [unlabeled_by_patient[patient_id] for patient_id in sorted(unlabeled_by_patient)]

    if not labeled_pairs:
        raise RuntimeError("No paired labeled image/label files were found in raw_dataset_roots.")
    if not unlabeled_images:
        raise RuntimeError("No unlabeled image files were found in raw_dataset_roots.")

    return {
        "labeled_pairs": sorted(labeled_pairs),
        "unlabeled_images": sorted(unlabeled_images),
    }


def _load_np_numpy_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        arr = np.load(f, allow_pickle=False)
    return np.asarray(arr)


def _load_npz(path: Path, key: Optional[str] = None) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key is not None and key in data.files:
            return np.asarray(data[key])
        for candidate in ["image", "label", "mask", "arr_0"]:
            if candidate in data.files:
                return np.asarray(data[candidate])
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]])
    raise KeyError(f"Could not infer array key from {path}.")


def load_volume_any(path: Path, array_key: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
    """
    Returns:
        arr_dhw: np.ndarray with shape [D, H, W]
        meta: dict including spacing_dhw and affine when available
    """
    path = Path(path)
    suffixes = "".join(path.suffixes)

    if suffixes.endswith(".nii.gz") or path.suffix == ".nii":
        nii = nib.as_closest_canonical(nib.load(str(path)))
        arr_xyz = np.asarray(nii.get_fdata(dtype=np.float32))
        if arr_xyz.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI, got shape {arr_xyz.shape} from {path}.")
        # nibabel returns [X, Y, Z]; DL pipeline uses [D, H, W] == [Z, Y, X]
        arr_dhw = np.transpose(arr_xyz, (2, 1, 0))
        zooms_xyz = np.asarray(nii.header.get_zooms()[:3], dtype=np.float32)
        spacing_dhw = zooms_xyz[[2, 1, 0]]
        meta = {
            "spacing_dhw": spacing_dhw,
            "affine": nii.affine.astype(np.float32),
            "source_path": str(path),
        }
        return arr_dhw, meta

    if path.suffix == ".npz":
        arr = _load_npz(path, key=array_key).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array in {path}, got shape {arr.shape}.")
        meta = {
            "spacing_dhw": np.ones(3, dtype=np.float32),
            "affine": np.eye(4, dtype=np.float32),
            "source_path": str(path),
        }
        return arr.astype(np.float32), meta

    if suffixes.endswith(".npy.gz"):
        arr = _load_np_numpy_gz(path).astype(np.float32)
    elif path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        raise ValueError(f"Unsupported input format: {path}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in {path}, got shape {arr.shape}.")
    meta = {
        "spacing_dhw": np.ones(3, dtype=np.float32),
        "affine": np.eye(4, dtype=np.float32),
        "source_path": str(path),
    }
    return arr, meta


def remap_labels_to_contiguous(label_dhw: np.ndarray, cfg: Config) -> np.ndarray:
    label_dhw = np.asarray(label_dhw)

    if cfg.labels_already_contiguous:
        valid = set(range(cfg.num_classes))
        uniques = set(np.unique(label_dhw).tolist())
        unknown = sorted(uniques - valid)
        if unknown:
            raise ValueError(f"Found unexpected contiguous labels: {unknown}")
        return label_dhw.astype(np.uint8)

    uniques = set(np.unique(label_dhw).tolist())
    unknown = sorted(uniques - set(RAW_TO_CONTIG.keys()))
    if unknown:
        raise ValueError(f"Found raw labels not in RAW_TO_CONTIG: {unknown}")

    out = np.zeros_like(label_dhw, dtype=np.uint8)
    for raw_id, class_id in RAW_TO_CONTIG.items():
        if raw_id == 0:
            continue
        out[label_dhw == raw_id] = class_id
    return out


def dilate_sparse_labels(label_dhw: np.ndarray, radius_voxels: int) -> np.ndarray:
    """
    Converts single-voxel landmark labels into small cubic training masks.

    The source labels are sparse lead/anatomy points, not organ segmentations.
    A pure voxel-wise segmentation loss receives too little foreground signal
    unless each point is expanded into a small target region.
    """
    if radius_voxels <= 0:
        return label_dhw.astype(np.uint8)

    label_dhw = label_dhw.astype(np.uint8)
    expanded = np.zeros_like(label_dhw, dtype=np.uint8)
    depth, height, width = label_dhw.shape

    for class_id in range(1, int(label_dhw.max()) + 1):
        coords = np.argwhere(label_dhw == class_id)
        for z, y, x in coords:
            z0, z1 = max(0, z - radius_voxels), min(depth, z + radius_voxels + 1)
            y0, y1 = max(0, y - radius_voxels), min(height, y + radius_voxels + 1)
            x0, x1 = max(0, x - radius_voxels), min(width, x + radius_voxels + 1)
            expanded[z0:z1, y0:y1, x0:x1] = class_id

    return expanded


def normalize_ct(image_dhw: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    image = np.clip(image_dhw.astype(np.float32), hu_min, hu_max)
    image = (image - hu_min) / max(hu_max - hu_min, 1e-8)
    return image.astype(np.float32)


def resample_volume_torch(
    volume_dhw: np.ndarray,
    in_spacing_dhw: Sequence[float],
    out_spacing_dhw: Sequence[float],
    is_label: bool = False,
) -> np.ndarray:
    """
    Resamples a [D, H, W] volume using torch interpolation.
    """
    in_spacing = np.asarray(in_spacing_dhw, dtype=np.float32)
    out_spacing = np.asarray(out_spacing_dhw, dtype=np.float32)

    if np.allclose(in_spacing, out_spacing, atol=1e-5):
        return volume_dhw.astype(np.float32 if not is_label else np.uint8)

    scale = in_spacing / out_spacing
    new_shape = np.maximum(1, np.round(np.asarray(volume_dhw.shape) * scale).astype(int))

    tensor = torch.from_numpy(volume_dhw.astype(np.float32))[None, None, ...]  # [1,1,D,H,W]
    mode = "nearest" if is_label else "trilinear"

    with torch.no_grad():
        resized = F.interpolate(
            tensor,
            size=tuple(int(x) for x in new_shape),
            mode=mode,
            align_corners=False if mode == "trilinear" else None,
        )

    out = resized[0, 0].cpu().numpy()
    if is_label:
        out = np.rint(out).astype(np.uint8)
    else:
        out = out.astype(np.float32)
    return out


def preprocess_case_to_npz(
    image_path: Path,
    output_npz: Path,
    cfg: Config,
    label_path: Optional[Path] = None,
) -> None:
    image_dhw, image_meta = load_volume_any(image_path, array_key="image")
    image_rs = resample_volume_torch(
        image_dhw,
        in_spacing_dhw=image_meta["spacing_dhw"],
        out_spacing_dhw=cfg.target_spacing_dhw,
        is_label=False,
    )
    image_rs = normalize_ct(image_rs, cfg.hu_clip_min, cfg.hu_clip_max)

    payload = {
        "image": image_rs.astype(np.float32),
        "case_id": robust_stem(image_path),
        "spacing_dhw": np.asarray(cfg.target_spacing_dhw, dtype=np.float32),
        "source_image": str(image_path),
    }

    if label_path is not None:
        label_dhw, label_meta = load_volume_any(label_path, array_key="label")
        label_dhw = remap_labels_to_contiguous(label_dhw, cfg)
        label_rs = resample_volume_torch(
            label_dhw,
            in_spacing_dhw=label_meta["spacing_dhw"],
            out_spacing_dhw=cfg.target_spacing_dhw,
            is_label=True,
        )
        label_rs = dilate_sparse_labels(label_rs, cfg.label_dilation_radius_voxels)
        payload["label"] = label_rs.astype(np.uint8)
        payload["source_label"] = str(label_path)
        payload["label_dilation_radius_voxels"] = np.int16(cfg.label_dilation_radius_voxels)

    np.savez_compressed(output_npz, **payload)


def build_preprocessed_cache(cfg: Config) -> Dict[str, List[Path]]:
    ensure_dirs(cfg)
    discovered = None

    if cfg.raw_labeled_images_dir and cfg.raw_labeled_labels_dir:
        labeled_pairs = pair_labeled_cases(cfg.raw_labeled_images_dir, cfg.raw_labeled_labels_dir)
    else:
        discovered = discover_cases_from_dataset_roots(cfg.raw_dataset_roots)
        labeled_pairs = discovered["labeled_pairs"]

    if not cfg.enable_pseudo_labeling:
        unlabeled_images = []
    elif cfg.raw_unlabeled_images_dir:
        unlabeled_images = list_unlabeled_cases(cfg.raw_unlabeled_images_dir)
    else:
        if discovered is None:
            discovered = discover_cases_from_dataset_roots(cfg.raw_dataset_roots)
        unlabeled_images = discovered["unlabeled_images"]

    labeled_npz_paths: List[Path] = []
    for image_path, label_path in tqdm(labeled_pairs, desc="Preprocessing labeled"):
        out_path = cfg.labeled_cache_dir / f"{robust_stem(image_path)}.npz"
        if not out_path.exists():
            preprocess_case_to_npz(image_path, out_path, cfg, label_path=label_path)
        labeled_npz_paths.append(out_path)

    unlabeled_npz_paths: List[Path] = []
    for image_path in tqdm(unlabeled_images, desc="Preprocessing unlabeled"):
        out_path = cfg.unlabeled_cache_dir / f"{robust_stem(image_path)}.npz"
        if not out_path.exists():
            preprocess_case_to_npz(image_path, out_path, cfg, label_path=None)
        unlabeled_npz_paths.append(out_path)

    meta_path = Path(cfg.work_dir) / "config.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    return {
        "labeled": sorted(labeled_npz_paths),
        "unlabeled": sorted(unlabeled_npz_paths),
    }
