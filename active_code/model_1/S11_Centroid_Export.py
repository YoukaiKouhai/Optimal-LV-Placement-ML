from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from S1_DataLoading_Preprocessing import CLASS_NAMES, Config, seed_everything
from S2_DatasetPreparation_Augmentation import create_train_val_split
from S3_ModelDefintion import build_model, infer_logits
from S6_S7_Metrics_Quantitative_Plots import logits_to_label_map


TUPLE_CONFIG_FIELDS = {
    "raw_dataset_roots",
    "target_spacing_dhw",
    "patch_size_3d",
    "class_sampling_ratios",
    "channels",
    "strides",
    "focus_class_ids",
}


def repo_root_from_here() -> Path:
    """
    Description
    -----------
    Derive repo root from here for downstream CRT lead localization steps.
    
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


def resolve_path(path_like: Optional[str | Path], base: Optional[Path] = None) -> Optional[Path]:
    """
    Description
    -----------
    Resolve paths or configuration references into concrete runtime values. This function implements the resolve path step.
    
    Parameters
    ----------
    path_like : Optional[str | Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    base : Optional[Path] (input)
        The base value supplied to this function.
    
    Returns
    -------
    Optional[Path]
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
    if path_like is None:
        return None
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    base_candidate = (base / path).resolve() if base is not None else None
    candidates = [cwd_candidate] if ".." in path.parts else []
    if base_candidate is not None:
        candidates.append(base_candidate)
    candidates.append(cwd_candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_default_run_dir(repo_root: Optional[Path] = None) -> Path:
    """
    Description
    -----------
    Find files or records matching the requested criteria. This function implements the find default run dir step.
    
    Parameters
    ----------
    repo_root : Optional[Path] (input)
        The repo root value supplied to this function.
    
    Returns
    -------
    Path
        Loaded object, parsed value, or collection of discovered records.
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
    repo_root = repo_root or repo_root_from_here()
    runs_dir = repo_root / "runs"
    candidates = sorted(
        [p for p in runs_dir.glob("*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / "weights" / "best_supervised_model.pth").exists():
            return candidate
    raise FileNotFoundError(f"No completed run with weights/best_supervised_model.pth found in {runs_dir}")


def load_config_for_run(run_dir: Path) -> Config:
    """
    Description
    -----------
    Load data, configuration, weights, or metadata from disk. This function implements the load config for run step.
    
    Parameters
    ----------
    run_dir : Path (input)
        Saved run directory.
    
    Returns
    -------
    Config
        Loaded object, parsed value, or collection of discovered records.
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
    cfg = Config()
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        valid_names = {field.name for field in fields(Config)}
        for key, value in saved.items():
            if key not in valid_names:
                continue
            if key in TUPLE_CONFIG_FIELDS and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)

    cfg.work_dir = str(run_dir)
    return cfg


def get_validation_npz_paths(run_dir: Path, cfg: Config) -> List[Path]:
    """
    Description
    -----------
    Implement the get validation npz paths helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    run_dir : Path (input)
        Saved run directory.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
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
    labeled_dir = run_dir / "cache" / "labeled"
    labeled_npz_paths = sorted(labeled_dir.glob("*.npz"))
    if not labeled_npz_paths:
        raise FileNotFoundError(f"No labeled cache .npz files found in {labeled_dir}")
    _, val_files = create_train_val_split(labeled_npz_paths, seed=cfg.seed, val_fraction=0.20)
    return val_files


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """
    Description
    -----------
    Load data, configuration, weights, or metadata from disk. This function implements the load checkpoint weights step.
    
    Parameters
    ----------
    model : torch.nn.Module (input)
        PyTorch model used by this step.
    checkpoint_path : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    device : torch.device (input)
        Torch device used for tensor and model computation.
    
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def centroid_from_mask(mask: np.ndarray, class_id: int) -> Tuple[float, float, float, int, bool]:
    """
    Description
    -----------
    Derive centroid from mask for downstream CRT lead localization steps.
    
    Parameters
    ----------
    mask : np.ndarray (input)
        Segmentation or binary mask used for metrics, visualization, or target extraction.
    class_id : int (input)
        Class identifier, class name, or number of modeled classes.
    
    Returns
    -------
    Tuple[float, float, float, int, bool]
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
    coords = np.argwhere(mask == class_id)
    if coords.size == 0:
        return np.nan, np.nan, np.nan, 0, True
    centroid_zyx = coords.mean(axis=0)
    return float(centroid_zyx[0]), float(centroid_zyx[1]), float(centroid_zyx[2]), int(coords.shape[0]), False


def centroid_rows_for_mask(
    mask: np.ndarray,
    patient_id: str,
    source: str,
    spacing_dhw: Sequence[float],
    source_npz: Path,
    num_classes: int,
) -> List[Dict[str, object]]:
    """
    Description
    -----------
    Implement the centroid rows for mask helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    mask : np.ndarray (input)
        Segmentation or binary mask used for metrics, visualization, or target extraction.
    patient_id : str (input)
        Internal dataset identifier for a patient or case.
    source : str (input)
        The source value supplied to this function.
    spacing_dhw : Sequence[float] (input)
        The spacing dhw value supplied to this function.
    source_npz : Path (input)
        The source npz value supplied to this function.
    num_classes : int (input)
        Class identifier, class name, or number of modeled classes. Units: count.
    
    Returns
    -------
    List[Dict[str, object]]
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
    for class_id in range(1, num_classes):
        centroid_z, centroid_y, centroid_x, voxel_count, missing = centroid_from_mask(mask, class_id)
        rows.append(
            {
                "patient_id": patient_id,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                "source": source,
                "centroid_z": centroid_z,
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "voxel_count": voxel_count,
                "peak_probability": np.nan,
                "missing": bool(missing),
                "spacing_d": float(spacing_dhw[0]) if len(spacing_dhw) > 0 else np.nan,
                "spacing_h": float(spacing_dhw[1]) if len(spacing_dhw) > 1 else np.nan,
                "spacing_w": float(spacing_dhw[2]) if len(spacing_dhw) > 2 else np.nan,
                "source_npz": str(source_npz),
            }
        )
    return rows


def centroid_rows_for_prediction_probabilities(
    probs: np.ndarray,
    patient_id: str,
    spacing_dhw: Sequence[float],
    source_npz: Path,
    cfg: Config,
) -> List[Dict[str, object]]:
    """
    Description
    -----------
    Implement the centroid rows for prediction probabilities helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    probs : np.ndarray (input)
        Probability tensor derived from model outputs. Units: dimensionless.
    patient_id : str (input)
        Internal dataset identifier for a patient or case.
    spacing_dhw : Sequence[float] (input)
        The spacing dhw value supplied to this function.
    source_npz : Path (input)
        The source npz value supplied to this function.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    List[Dict[str, object]]
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
    for class_id in range(1, cfg.num_classes):
        channel_idx = class_id - 1
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        if channel_idx < 0 or channel_idx >= probs.shape[0]:
            centroid_z = centroid_y = centroid_x = np.nan
            voxel_count = 0
            peak_probability = np.nan
            missing = True
        else:
            channel = probs[channel_idx]
            flat_idx = int(np.argmax(channel))
            centroid_zyx = np.asarray(np.unravel_index(flat_idx, channel.shape), dtype=np.float64)
            centroid_z, centroid_y, centroid_x = map(float, centroid_zyx)
            voxel_count = int((channel >= cfg.prediction_threshold).sum())
            peak_probability = float(channel.reshape(-1)[flat_idx])
            missing = False

        rows.append(
            {
                "patient_id": patient_id,
                "class_id": class_id,
                "class_name": class_name,
                "source": "Prediction",
                "centroid_z": centroid_z,
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "voxel_count": voxel_count,
                "peak_probability": peak_probability,
                "missing": bool(missing),
                "spacing_d": float(spacing_dhw[0]) if len(spacing_dhw) > 0 else np.nan,
                "spacing_h": float(spacing_dhw[1]) if len(spacing_dhw) > 1 else np.nan,
                "spacing_w": float(spacing_dhw[2]) if len(spacing_dhw) > 2 else np.nan,
                "source_npz": str(source_npz),
            }
        )
    return rows


def compute_error_rows(coordinates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Description
    -----------
    Compute metrics, weights, coordinates, or summary values. This function implements the compute error rows step.
    
    Parameters
    ----------
    coordinates_df : pd.DataFrame (input)
        Coordinate value or coordinate collection in the convention required by the caller.
    
    Returns
    -------
    pd.DataFrame
        Metric value, summary table, dictionary, or collection of result records.
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
    error_rows: List[Dict[str, object]] = []
    grouped = coordinates_df.groupby(["patient_id", "class_id"], sort=True)
    for (patient_id, class_id), group in grouped:
        gt_rows = group[group["source"] == "GT"]
        pred_rows = group[group["source"] == "Prediction"]
        if gt_rows.empty or pred_rows.empty:
            continue

        gt = gt_rows.iloc[0]
        pred = pred_rows.iloc[0]
        gt_missing = bool(gt["missing"])
        pred_missing = bool(pred["missing"])
        class_name = str(gt["class_name"])

        if gt_missing or pred_missing:
            distance_voxels = np.nan
            distance_mm = np.nan
        else:
            gt_zyx = np.array([gt["centroid_z"], gt["centroid_y"], gt["centroid_x"]], dtype=np.float64)
            pred_zyx = np.array([pred["centroid_z"], pred["centroid_y"], pred["centroid_x"]], dtype=np.float64)
            delta_zyx = pred_zyx - gt_zyx
            spacing_dhw = np.array([gt["spacing_d"], gt["spacing_h"], gt["spacing_w"]], dtype=np.float64)
            distance_voxels = float(np.linalg.norm(delta_zyx))
            distance_mm = float(np.linalg.norm(delta_zyx * spacing_dhw)) if np.isfinite(spacing_dhw).all() else np.nan

        error_rows.append(
            {
                "patient_id": patient_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "distance_voxels": distance_voxels,
                "distance_mm": distance_mm,
                "gt_missing": gt_missing,
                "prediction_missing": pred_missing,
                "gt_voxel_count": int(gt["voxel_count"]),
                "prediction_voxel_count": int(pred["voxel_count"]),
            }
        )
    return pd.DataFrame(error_rows)


@torch.no_grad()
def predict_label_map_for_case(
    model: torch.nn.Module,
    image_np: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    """
    Description
    -----------
    Implement the predict label map for case helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    model : torch.nn.Module (input)
        PyTorch model used by this step.
    image_np : np.ndarray (input)
        The image np value supplied to this function.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    device : torch.device (input)
        Torch device used for tensor and model computation.
    
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
    image_tensor = torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    logits = infer_logits(model, image_tensor, cfg)
    preds, _ = logits_to_label_map(logits, cfg)
    return preds[0, 0].detach().cpu().numpy().astype(np.uint8)


@torch.no_grad()
def predict_probabilities_for_case(
    model: torch.nn.Module,
    image_np: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    """
    Description
    -----------
    Implement the predict probabilities for case helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    model : torch.nn.Module (input)
        PyTorch model used by this step.
    image_np : np.ndarray (input)
        The image np value supplied to this function.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    device : torch.device (input)
        Torch device used for tensor and model computation.
    
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
    image_tensor = torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    logits = infer_logits(model, image_tensor, cfg)
    probs = torch.sigmoid(logits)
    return probs[0].detach().cpu().numpy().astype(np.float32)


def export_centroids(
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
    force: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[Path, Path]:
    """
    Description
    -----------
    Export computed outputs into reusable files. This function implements the export centroids step.
    
    Parameters
    ----------
    run_dir : Path (input)
        Saved run directory.
    checkpoint_path : Optional[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    force : bool (input)
        The force value supplied to this function.
    device : Optional[torch.device] (input)
        Torch device used for tensor and model computation.
    
    Returns
    -------
    Tuple[Path, Path]
        Generated artifact path, summary object, or status value produced by the workflow branch.
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
    run_dir = run_dir.resolve()
    cfg = load_config_for_run(run_dir)
    seed_everything(cfg.seed)

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = metrics_dir / "centroid_coordinates.csv"
    errors_path = metrics_dir / "centroid_errors.csv"
    if not force and coordinates_path.exists() and errors_path.exists():
        print(f"Reusing centroid CSVs: {coordinates_path} and {errors_path}")
        return coordinates_path, errors_path

    checkpoint_path = checkpoint_path or (run_dir / "weights" / "best_supervised_model.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting centroids with device: {device}")
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    val_files = get_validation_npz_paths(run_dir, cfg)
    model = build_model(cfg).to(device)
    load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()

    all_rows: List[Dict[str, object]] = []
    for npz_path in tqdm(val_files, desc="Centroid export"):
        with np.load(npz_path, allow_pickle=False) as data:
            image_np = data["image"].astype(np.float32)
            label_np = data["label"].astype(np.uint8)
            patient_id = str(data["case_id"]) if "case_id" in data.files else npz_path.stem
            spacing_dhw = data["spacing_dhw"].astype(np.float32) if "spacing_dhw" in data.files else np.array([np.nan, np.nan, np.nan])

        pred_probs = predict_probabilities_for_case(model, image_np, cfg, device)
        all_rows.extend(
            centroid_rows_for_mask(
                mask=label_np,
                patient_id=patient_id,
                source="GT",
                spacing_dhw=spacing_dhw,
                source_npz=npz_path,
                num_classes=cfg.num_classes,
            )
        )
        all_rows.extend(
            centroid_rows_for_prediction_probabilities(
                probs=pred_probs,
                patient_id=patient_id,
                spacing_dhw=spacing_dhw,
                source_npz=npz_path,
                cfg=cfg,
            )
        )

    coordinates_df = pd.DataFrame(all_rows)
    errors_df = compute_error_rows(coordinates_df)
    coordinates_df.to_csv(coordinates_path, index=False)
    errors_df.to_csv(errors_path, index=False)
    print(f"Saved centroid coordinates: {coordinates_path}")
    print(f"Saved centroid errors: {errors_path}")
    return coordinates_path, errors_path


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Description
    -----------
    Build or parse command-line arguments for S11_Centroid_Export.py.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    argparse.ArgumentParser
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
    parser = argparse.ArgumentParser(description="Export GT and prediction centroid coordinates for validation cases.")
    parser.add_argument("--run-dir", type=str, default=None, help="Completed run directory. Defaults to latest run with a best checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to run-dir/weights/best_supervised_model.pth.")
    parser.add_argument("--force", action="store_true", help="Recompute even if centroid CSVs already exist.")
    return parser


def main() -> None:
    """
    Description
    -----------
    Run the command-line workflow implemented by S11_Centroid_Export.py.
    
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
    args = build_arg_parser().parse_args()
    repo_root = repo_root_from_here()
    run_dir = resolve_path(args.run_dir, repo_root) if args.run_dir else find_default_run_dir(repo_root)
    checkpoint = resolve_path(args.checkpoint, repo_root) if args.checkpoint else None
    export_centroids(run_dir=run_dir, checkpoint_path=checkpoint, force=args.force)


if __name__ == "__main__":
    main()
