# ==============================
# STEP 14: TRAINING + DATA PIPELINE DIAGNOSTICS
# ==============================
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from S1_DataLoading_Preprocessing import (
    CLASS_NAMES,
    Config,
    IMAGE_SUBDIR,
    LABEL_SUBDIR,
    build_preprocessed_cache,
    ensure_dirs,
    extract_patient_id,
    load_volume_any,
    seed_everything,
    supported_files,
)
from S2_DatasetPreparation_Augmentation import (
    NPZSegmentationDataset,
    build_train_transform,
    build_val_transform,
    create_train_loader,
    create_train_val_split,
    create_val_loader,
)
from S3_ModelDefintion import build_model


def _load_training_module():
    """
    Description
    -----------
    Load data, configuration, weights, or metadata from disk. This function implements the load training module step.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    Any
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    module_path = Path(__file__).with_name("S4_S5_Training_Semi-Supervised_Pseudo-Lableing.py")
    spec = importlib.util.spec_from_file_location("S4_S5_Training_Semi_Supervised_Pseudo_Lableing", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_training_module = _load_training_module()
build_loss_fn = _training_module.build_loss_fn
compute_class_weights = _training_module.compute_class_weights
fit_model = _training_module.fit_model
labels_to_multichannel_targets = _training_module.labels_to_multichannel_targets


def _repo_root() -> Path:
    """
    Description
    -----------
    Implement the repo root helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    Path
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str | Path | None, base: Path) -> Path | None:
    """
    Description
    -----------
    Resolve paths or configuration references into concrete runtime values. This function implements the resolve path step.
    
    Parameters
    ----------
    path_value : str | Path | None (input)
        Filesystem location used for reading inputs or writing outputs.
    base : Path (input)
        The base value supplied to this function.
    
    Returns
    -------
    Path | None
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def _case_id_from_npz(path: Path) -> str:
    """
    Description
    -----------
    Derive case id from npz for downstream CRT lead localization steps.
    
    Parameters
    ----------
    path : Path (input)
        Filesystem path used by this step.
    
    Returns
    -------
    str
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    with np.load(path, allow_pickle=False) as data:
        if "case_id" in data.files:
            return str(data["case_id"])
    return path.stem


def _patient_id_from_case_id(case_id: str) -> str:
    """
    Description
    -----------
    Derive patient id from case id for downstream CRT lead localization steps.
    
    Parameters
    ----------
    case_id : str (input)
        Internal dataset identifier for a patient or case.
    
    Returns
    -------
    str
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    match = extract_patient_id(Path(case_id))
    return match or case_id


def _counts_for_labels(label: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Description
    -----------
    Implement the counts for labels helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    label : np.ndarray (input)
        Ground-truth label map or target tensor.
    num_classes : int (input)
        Class identifier, class name, or number of modeled classes. Units: count.
    
    Returns
    -------
    np.ndarray
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    unique, freq = np.unique(label.astype(np.int64), return_counts=True)
    valid = (unique >= 0) & (unique < num_classes)
    counts[unique[valid]] = freq[valid]
    return counts


def _summarize_npz(path: Path, split: str, cfg: Config) -> Dict[str, object]:
    """
    Description
    -----------
    Summarize run outputs or records into compact metrics. This function implements the summarize npz step.
    
    Parameters
    ----------
    path : Path (input)
        Filesystem path used by this step.
    split : str (input)
        The split value supplied to this function.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    Dict[str, object]
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    with np.load(path, allow_pickle=False) as data:
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.int64)
        spacing = data["spacing_dhw"].astype(np.float32) if "spacing_dhw" in data.files else np.ones(3, dtype=np.float32)
        source_image = str(data["source_image"]) if "source_image" in data.files else ""
        source_label = str(data["source_label"]) if "source_label" in data.files else ""
        case_id = str(data["case_id"]) if "case_id" in data.files else path.stem

    counts = _counts_for_labels(label, cfg.num_classes)
    row: Dict[str, object] = {
        "split": split,
        "case_id": case_id,
        "patient_id": _patient_id_from_case_id(case_id),
        "npz_path": str(path),
        "source_image": source_image,
        "source_label": source_label,
        "image_shape": "x".join(str(v) for v in image.shape),
        "label_shape": "x".join(str(v) for v in label.shape),
        "shape_match": bool(image.shape == label.shape),
        "spacing_dhw": ",".join(f"{float(v):.4g}" for v in spacing),
        "image_min": float(np.nanmin(image)),
        "image_max": float(np.nanmax(image)),
        "image_mean": float(np.nanmean(image)),
        "image_std": float(np.nanstd(image)),
        "foreground_voxels": int(counts[1:].sum()),
        "background_fraction": float(counts[0] / max(counts.sum(), 1)),
        "unique_labels": ",".join(str(int(v)) for v in np.unique(label)),
    }
    for class_id, class_name in enumerate(CLASS_NAMES):
        row[f"voxels_{class_name}"] = int(counts[class_id])
    for class_id, class_name in enumerate(CLASS_NAMES[1:], start=1):
        row[f"missing_{class_name}"] = bool(counts[class_id] == 0)
    return row


def _inspect_raw_pairing(cfg: Config, out_dir: Path) -> pd.DataFrame:
    """
    Description
    -----------
    Inspect data or metadata for diagnostics. This function implements the inspect raw pairing step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    out_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    
    Returns
    -------
    pd.DataFrame
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    rows: List[Dict[str, object]] = []
    for root_str in cfg.raw_dataset_roots:
        root = Path(root_str)
        images = supported_files(root / IMAGE_SUBDIR)
        labels = supported_files(root / LABEL_SUBDIR)
        image_ids = {extract_patient_id(path): path for path in images if extract_patient_id(path)}
        label_ids = {extract_patient_id(path): path for path in labels if extract_patient_id(path)}
        all_ids = sorted(set(image_ids) | set(label_ids))
        for patient_id in all_ids:
            rows.append(
                {
                    "dataset_root": str(root),
                    "patient_id": patient_id,
                    "image_exists": patient_id in image_ids,
                    "label_exists": patient_id in label_ids,
                    "image_path": str(image_ids.get(patient_id, "")),
                    "label_path": str(label_ids.get(patient_id, "")),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw_image_label_pairing.csv", index=False)
    return df


def _write_split_audit(train_files: Sequence[Path], val_files: Sequence[Path], out_dir: Path) -> pd.DataFrame:
    """
    Description
    -----------
    Write a diagnostic or summary artifact to disk. This function implements the write split audit step.
    
    Parameters
    ----------
    train_files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    val_files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    out_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    
    Returns
    -------
    pd.DataFrame
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    rows = []
    for split, files in [("train", train_files), ("val", val_files)]:
        for path in files:
            case_id = _case_id_from_npz(path)
            rows.append(
                {
                    "split": split,
                    "case_id": case_id,
                    "patient_id": _patient_id_from_case_id(case_id),
                    "npz_path": str(path),
                }
            )
    df = pd.DataFrame(rows)
    train_ids = set(df.loc[df["split"] == "train", "patient_id"])
    val_ids = set(df.loc[df["split"] == "val", "patient_id"])
    overlap = sorted(train_ids & val_ids)
    df["split_overlap_patient"] = df["patient_id"].isin(overlap)
    df.to_csv(out_dir / "split_audit.csv", index=False)
    (out_dir / "split_overlap.txt").write_text(
        "No train/validation patient overlap detected.\n" if not overlap else "\n".join(overlap) + "\n",
        encoding="utf-8",
    )
    return df


def _write_label_distribution(train_files: Sequence[Path], val_files: Sequence[Path], cfg: Config, out_dir: Path) -> pd.DataFrame:
    """
    Description
    -----------
    Write a diagnostic or summary artifact to disk. This function implements the write label distribution step.
    
    Parameters
    ----------
    train_files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    val_files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    out_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    
    Returns
    -------
    pd.DataFrame
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    rows = []
    for split, files in [("train", train_files), ("val", val_files)]:
        for path in files:
            rows.append(_summarize_npz(path, split, cfg))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "case_label_distribution.csv", index=False)

    count_cols = [f"voxels_{name}" for name in CLASS_NAMES]
    summary = df.groupby("split")[count_cols + ["foreground_voxels", "background_fraction"]].agg(["mean", "min", "max", "sum"])
    summary.to_csv(out_dir / "split_label_distribution_summary.csv")
    return df


def _write_intensity_audit(files: Sequence[Path], out_dir: Path, max_cases: int = 5) -> pd.DataFrame:
    """
    Description
    -----------
    Write a diagnostic or summary artifact to disk. This function implements the write intensity audit step.
    
    Parameters
    ----------
    files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    out_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    max_cases : int (input)
        Internal dataset identifier for a patient or case.
    
    Returns
    -------
    pd.DataFrame
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    rows = []
    for path in list(files)[:max_cases]:
        with np.load(path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32)
            source_image = Path(str(data["source_image"])) if "source_image" in data.files else None
            case_id = str(data["case_id"]) if "case_id" in data.files else path.stem

        row = {
            "case_id": case_id,
            "normalized_min": float(np.nanmin(image)),
            "normalized_max": float(np.nanmax(image)),
            "normalized_mean": float(np.nanmean(image)),
            "normalized_std": float(np.nanstd(image)),
        }
        if source_image and source_image.exists():
            raw, meta = load_volume_any(source_image)
            row.update(
                {
                    "raw_min": float(np.nanmin(raw)),
                    "raw_max": float(np.nanmax(raw)),
                    "raw_mean": float(np.nanmean(raw)),
                    "raw_std": float(np.nanstd(raw)),
                    "raw_shape": "x".join(str(v) for v in raw.shape),
                    "raw_spacing_dhw": ",".join(f"{float(v):.4g}" for v in meta["spacing_dhw"]),
                }
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "intensity_audit.csv", index=False)
    return df


def _save_label_overlay(path: Path, split: str, out_dir: Path) -> Path:
    """
    Description
    -----------
    Save a project artifact such as a plot, table, checkpoint, or report. This function implements the save label overlay step.
    
    Parameters
    ----------
    path : Path (input)
        Filesystem path used by this step.
    split : str (input)
        The split value supplied to this function.
    out_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    
    Returns
    -------
    Path
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    with np.load(path, allow_pickle=False) as data:
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.int64)
        case_id = str(data["case_id"]) if "case_id" in data.files else path.stem

    fg_by_slice = (label > 0).sum(axis=(1, 2))
    z = int(np.argmax(fg_by_slice)) if fg_by_slice.max() > 0 else image.shape[0] // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image[z], cmap="gray")
    masked = np.ma.masked_where(label[z] == 0, label[z])
    ax.imshow(masked, cmap="tab10", alpha=0.65, vmin=0, vmax=9)
    ax.set_title(f"{split} {case_id} z={z}")
    ax.axis("off")
    fig.tight_layout()
    out_path = out_dir / f"{split}_{case_id}_label_overlay.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _write_overlays(train_files: Sequence[Path], val_files: Sequence[Path], out_dir: Path, max_cases: int) -> List[Path]:
    """
    Description
    -----------
    Write a diagnostic or summary artifact to disk. This function implements the write overlays step.
    
    Parameters
    ----------
    train_files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    val_files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    out_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    max_cases : int (input)
        Internal dataset identifier for a patient or case.
    
    Returns
    -------
    List[Path]
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    overlay_dir = out_dir / "label_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for split, files in [("train", train_files), ("val", val_files)]:
        for path in list(files)[:max_cases]:
            saved.append(_save_label_overlay(path, split, overlay_dir))
    return saved


def _write_batch_and_model_debug(train_files: Sequence[Path], cfg: Config, out_dir: Path, device: torch.device) -> Dict[str, object]:
    """
    Description
    -----------
    Write a diagnostic or summary artifact to disk. This function implements the write batch and model debug step.
    
    Parameters
    ----------
    train_files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    out_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    device : torch.device (input)
        Torch device used for tensor and model computation.
    
    Returns
    -------
    Dict[str, object]
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    repo_root = _repo_root()
    debug_cfg = deepcopy(cfg)
    debug_cfg.num_workers = 0
    debug_cfg.train_batch_size = 1
    loader = create_train_loader(list(train_files)[:1], [False], debug_cfg)
    batch = next(iter(loader))

    images = batch["image"].to(device)
    labels = batch["label"].to(device).long()
    model = build_model(debug_cfg).to(device)
    model.eval()
    class_weights = compute_class_weights(list(train_files)[: max(1, min(3, len(train_files)))], debug_cfg).to(device)
    loss_fn = build_loss_fn(class_weights, debug_cfg)

    loaded_checkpoint = None
    checkpoint_candidates = [
        _resolve_path(debug_cfg.warm_start_checkpoint, repo_root),
        Path(debug_cfg.work_dir) / "weights" / "best_supervised_model.pth",
    ]
    for checkpoint_path in checkpoint_candidates:
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            loaded_checkpoint = str(checkpoint_path)
            break

    with torch.no_grad():
        logits = model(images)
        targets = labels_to_multichannel_targets(labels, debug_cfg)
        loss = loss_fn(logits, targets)
        probs = torch.sigmoid(logits)

    pred_positive = (probs >= debug_cfg.prediction_threshold).sum(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    target_positive = targets.sum(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    prob_rows = []
    for channel_idx, class_name in enumerate(CLASS_NAMES[1:]):
        channel = probs[:, channel_idx]
        prob_rows.append(
            {
                "class_id": channel_idx + 1,
                "class_name": class_name,
                "target_positive_voxels": int(target_positive[channel_idx]),
                "pred_positive_voxels_at_threshold": int(pred_positive[channel_idx]),
                "prob_min": float(channel.min().item()),
                "prob_max": float(channel.max().item()),
                "prob_mean": float(channel.mean().item()),
            }
        )
    pd.DataFrame(prob_rows).to_csv(out_dir / "one_forward_prediction_distribution.csv", index=False)

    unique_labels = torch.unique(labels).detach().cpu().numpy().astype(int).tolist()
    debug = {
        "case_id": batch["case_id"][0],
        "image_tensor_shape": list(images.shape),
        "label_tensor_shape": list(labels.shape),
        "logits_shape": list(logits.shape),
        "target_shape": list(targets.shape),
        "loss_value": float(loss.item()),
        "checkpoint_loaded_for_forward_debug": loaded_checkpoint,
        "unique_label_values": unique_labels,
        "class_weights": [float(v) for v in class_weights.detach().cpu().numpy()],
        "image_min": float(images.min().item()),
        "image_max": float(images.max().item()),
        "image_mean": float(images.mean().item()),
        "image_std": float(images.std().item()),
    }
    (out_dir / "one_batch_model_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
    return debug


def _plot_history_comparison(repo_root: Path, out_dir: Path) -> None:
    """
    Description
    -----------
    Create a visualization and save or populate the requested figure. This function implements the plot history comparison step.
    
    Parameters
    ----------
    repo_root : Path (input)
        The repo root value supplied to this function.
    out_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    
    Returns
    -------
    None
        No value is returned; the function is executed for orchestration, mutation of supplied objects, or file output.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    history_paths = sorted((repo_root / "runs").glob("*/history_supervised.csv"))
    rows = []
    for path in history_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["run"] = path.parent.name
        rows.append(df)
    if not rows:
        return

    history = pd.concat(rows, ignore_index=True)
    history.to_csv(out_dir / "all_runs_history_supervised.csv", index=False)

    metrics = [
        ("val_dice", "Validation Dice"),
        ("val_centroid_dist", "Validation centroid distance"),
        ("val_focus_centroid_dist", "Focused class centroid distance"),
        ("train_loss", "Training loss"),
        ("val_loss", "Validation loss"),
    ]
    for col, title in metrics:
        if col not in history.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for run_name, run_df in history.groupby("run"):
            ax.scatter(run_df["epoch"], run_df[col], label=run_name, s=18)
            ax.plot(run_df["epoch"], run_df[col], linewidth=1, alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"{title} across runs")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"all_runs_{col}.png", dpi=180)
        plt.close(fig)


def run_diagnostics(cfg: Config, max_overlay_cases: int = 2, run_tiny_overfit: bool = False, tiny_cases: int = 2, tiny_epochs: int = 8) -> Path:
    """
    Description
    -----------
    Run one pipeline stage or orchestration workflow. This function implements the run diagnostics step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    max_overlay_cases : int (input)
        Internal dataset identifier for a patient or case.
    run_tiny_overfit : bool (input)
        The run tiny overfit value supplied to this function.
    tiny_cases : int (input)
        Internal dataset identifier for a patient or case.
    tiny_epochs : int (input)
        The tiny epochs value supplied to this function. Units: epochs.
    
    Returns
    -------
    Path
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: May create directories, write files, print progress, or update checkpoint/model state as part of the pipeline.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    repo_root = _repo_root()
    if not Path(cfg.work_dir).is_absolute():
        cfg.work_dir = str(repo_root / cfg.work_dir)

    seed_everything(cfg.seed)
    ensure_dirs(cfg)
    out_dir = Path(cfg.work_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = build_preprocessed_cache(cfg)
    labeled_files = cache["labeled"]
    train_files, val_files = create_train_val_split(labeled_files, seed=cfg.seed, val_fraction=0.20)

    raw_df = _inspect_raw_pairing(cfg, out_dir)
    split_df = _write_split_audit(train_files, val_files, out_dir)
    label_df = _write_label_distribution(train_files, val_files, cfg, out_dir)
    _write_intensity_audit(train_files + val_files, out_dir)
    overlays = _write_overlays(train_files, val_files, out_dir, max_overlay_cases)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_debug = _write_batch_and_model_debug(train_files, cfg, out_dir, device)
    _plot_history_comparison(repo_root, out_dir)

    train_ids = set(split_df.loc[split_df["split"] == "train", "patient_id"])
    val_ids = set(split_df.loc[split_df["split"] == "val", "patient_id"])
    missing_labels_by_class = {
        class_name: int(label_df[f"missing_{class_name}"].sum())
        for class_name in CLASS_NAMES[1:]
        if f"missing_{class_name}" in label_df
    }
    summary = {
        "train_cases": len(train_files),
        "val_cases": len(val_files),
        "train_val_patient_overlap": sorted(train_ids & val_ids),
        "raw_pair_rows": len(raw_df),
        "unmatched_raw_rows": int((~(raw_df["image_exists"] & raw_df["label_exists"])).sum()) if not raw_df.empty else 0,
        "shape_mismatch_cases": int((~label_df["shape_match"]).sum()) if not label_df.empty else 0,
        "missing_labels_by_class": missing_labels_by_class,
        "one_batch_debug": batch_debug,
        "label_overlay_files": [str(path) for path in overlays],
    }

    if run_tiny_overfit:
        tiny_dir = out_dir / "tiny_overfit"
        tiny_dir.mkdir(parents=True, exist_ok=True)
        tiny_cfg = deepcopy(cfg)
        tiny_cfg.work_dir = str(tiny_dir)
        tiny_cfg.num_workers = 0
        tiny_cfg.train_batch_size = 1
        tiny_cfg.val_batch_size = 1
        tiny_cfg.samples_per_volume = 2
        tiny_cfg.patch_size_3d = (96, 96, 96)
        tiny_cfg.channels = (8, 16, 32, 64)
        tiny_cfg.strides = (2, 2, 2)
        tiny_cfg.supervised_epochs = tiny_epochs
        tiny_cfg.early_stopping_patience = tiny_epochs + 1
        tiny_cfg.use_reduce_lr_on_plateau = False
        tiny_cfg.warm_start_checkpoint = None
        tiny_cfg.enable_spatial_augmentation = False
        tiny_cfg.enable_intensity_augmentation = False
        ensure_dirs(tiny_cfg)

        tiny_files = list(train_files)[: max(1, tiny_cases)]
        tiny_train_loader = create_train_loader(tiny_files, [False] * len(tiny_files), tiny_cfg)
        tiny_val_loader = create_val_loader(tiny_files, tiny_cfg)
        model = build_model(tiny_cfg).to(device)
        class_weights = compute_class_weights(tiny_files, tiny_cfg).to(device)
        loss_fn = build_loss_fn(class_weights, tiny_cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=max(tiny_cfg.learning_rate, 3e-4), weight_decay=0.0)
        history = fit_model(
            model=model,
            train_loader=tiny_train_loader,
            val_loader=tiny_val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            cfg=tiny_cfg,
            epochs=tiny_epochs,
            checkpoint_path=tiny_cfg.weights_dir / "best_tiny_overfit_model.pth",
            stage_name="tiny_overfit",
            gpu_aug=None,
        )
        history.to_csv(tiny_dir / "tiny_overfit_history.csv", index=False)
        summary["tiny_overfit_history"] = str(tiny_dir / "tiny_overfit_history.csv")
        if "train_loss" in history:
            train_losses = history["train_loss"].dropna().astype(float)
            if len(train_losses) >= 2:
                summary["tiny_overfit_train_loss_first"] = float(train_losses.iloc[0])
                summary["tiny_overfit_train_loss_last"] = float(train_losses.iloc[-1])
                summary["tiny_overfit_train_loss_decrease_fraction"] = float(
                    (train_losses.iloc[0] - train_losses.iloc[-1]) / max(abs(train_losses.iloc[0]), 1e-8)
                )

    (out_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved diagnostics to: {out_dir}")
    print(json.dumps(summary, indent=2))
    return out_dir


def parse_args() -> argparse.Namespace:
    """
    Description
    -----------
    Build or parse command-line arguments for S14_Training_Diagnostics.py.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    argparse.Namespace
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    parser = argparse.ArgumentParser(description="Audit training data, labels, loaders, and optional tiny-overfit sanity check.")
    parser.add_argument("--work-dir", default=None, help="Override Config.work_dir for diagnostics outputs.")
    parser.add_argument("--max-overlay-cases", type=int, default=2)
    parser.add_argument("--tiny-overfit", action="store_true", help="Train on 1-3 cases and validate on the same cases.")
    parser.add_argument("--tiny-cases", type=int, default=2)
    parser.add_argument("--tiny-epochs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """
    Description
    -----------
    Run the command-line workflow implemented by S14_Training_Diagnostics.py.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    None
        No value is returned; the function is executed for orchestration, mutation of supplied objects, or file output.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: May create directories, write files, print progress, or update checkpoint/model state as part of the pipeline.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    args = parse_args()
    cfg = Config()
    if args.work_dir:
        cfg.work_dir = args.work_dir
    run_diagnostics(
        cfg=cfg,
        max_overlay_cases=args.max_overlay_cases,
        run_tiny_overfit=args.tiny_overfit,
        tiny_cases=args.tiny_cases,
        tiny_epochs=args.tiny_epochs,
    )


if __name__ == "__main__":
    main()
