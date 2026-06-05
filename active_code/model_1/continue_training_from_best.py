"""
Continue ML training from the current best validated run.

The best reported result is the weighted ensemble in
``runs/cardiac_leads_ensemble_v3_v6``. That run is an evaluation ensemble and
does not itself contain a trainable checkpoint. This script therefore resolves
the source checkpoints listed in the ensemble config and creates a weighted
average warm-start checkpoint in the new output run folder.

Use --check first. It verifies dataset paths, split overlap, checkpoint loading,
one DataLoader batch, and one model forward pass without launching training.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from S1_DataLoading_Preprocessing import CLASS_NAMES, RAW_TO_CONTIG, Config, build_preprocessed_cache, ensure_dirs, seed_everything
from S2_DatasetPreparation_Augmentation import build_gpu_train_augment, create_train_loader, create_train_val_split, create_val_loader
from S3_ModelDefintion import build_model
from S6_S7_Metrics_Quantitative_Plots import (
    evaluate_model,
    plot_confusion_matrix_heatmap,
    plot_dice_boxplot,
    plot_per_class_dice,
    plot_precision_recall_curve,
    plot_training_history,
    save_metrics_to_csv,
)
from S8_Visualization_of_Predictions import save_prediction_overlays
from S9_Output_Main_Execution_Block import build_loss_fn, compute_class_weights, fit_model, load_checkpoint_weights, run_presentation_outputs


TUPLE_CONFIG_FIELDS = {
    "raw_dataset_roots",
    "target_spacing_dhw",
    "patch_size_3d",
    "class_sampling_ratios",
    "channels",
    "strides",
    "focus_class_ids",
}


def repo_root() -> Path:
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


def resolve_path(path_value: str | Path, base: Path) -> Path:
    """
    Description
    -----------
    Resolve paths or configuration references into concrete runtime values. This function implements the resolve path step.
    
    Parameters
    ----------
    path_value : str | Path (input)
        Filesystem location used for reading inputs or writing outputs.
    base : Path (input)
        The base value supplied to this function.
    
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
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def load_config_from_run(run_dir: Path) -> Config:
    """
    Description
    -----------
    Derive load config from run for downstream CRT lead localization steps.
    
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
    if not config_path.exists():
        return cfg
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    valid_fields = {field.name for field in fields(Config)}
    for key, value in payload.items():
        if key not in valid_fields:
            continue
        if key in TUPLE_CONFIG_FIELDS and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)
    return cfg


def repair_dataset_roots_for_local_machine(cfg: Config) -> None:
    """
    Description
    -----------
    Repair saved paths or checkpoint references for the current machine. This function implements the repair dataset roots for local machine step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
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
    roots = tuple(str(path) for path in cfg.raw_dataset_roots)
    roots_missing = any(not Path(path).exists() for path in roots)
    roots_sanitized = any("<USER>" in path for path in roots)
    if roots_missing and roots_sanitized:
        local_defaults = Config().raw_dataset_roots
        print("Saved run config contains sanitized dataset paths; using local default dataset roots.")
        cfg.raw_dataset_roots = local_defaults


def ensure_output_is_safe(output_dir: Path, allow_existing: bool) -> None:
    """
    Description
    -----------
    Ensure required directories or invariants exist before continuing. This function implements the ensure output is safe step.
    
    Parameters
    ----------
    output_dir : Path (input)
        Directory where outputs are written.
    allow_existing : bool (input)
        The allow existing value supplied to this function.
    
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
    if output_dir.exists() and not allow_existing:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}\n"
            "Choose a new --output-dir or pass --allow-existing-output intentionally."
        )


def check_dataset_roots(cfg: Config) -> None:
    """
    Description
    -----------
    Check runtime assumptions and report problems early. This function implements the check dataset roots step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
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
    missing = [Path(path) for path in cfg.raw_dataset_roots if not Path(path).exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing configured dataset root(s):\n{missing_text}")
    print("Dataset roots exist:")
    for path in cfg.raw_dataset_roots:
        print(f"  {path}")


def read_case_id(npz_path: Path) -> str:
    """
    Description
    -----------
    Read and parse a value from disk. This function implements the read case id step.
    
    Parameters
    ----------
    npz_path : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    
    Returns
    -------
    str
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
    try:
        import numpy as np

        with np.load(npz_path, allow_pickle=False) as data:
            if "case_id" in data.files:
                return str(data["case_id"])
    except Exception:
        pass
    return npz_path.stem


def assert_no_split_overlap(train_files: Sequence[Path], val_files: Sequence[Path]) -> None:
    """
    Description
    -----------
    Assert a required invariant and fail if it is violated. This function implements the assert no split overlap step.
    
    Parameters
    ----------
    train_files : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    val_files : Sequence[Path] (input)
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
    train_ids = {read_case_id(path) for path in train_files}
    val_ids = {read_case_id(path) for path in val_files}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise RuntimeError(f"Train/validation overlap detected: {overlap}")
    print(f"Train/validation split OK: {len(train_ids)} train, {len(val_ids)} val, 0 overlap.")


def normalize_weights(weights: Sequence[float], count: int) -> List[float]:
    """
    Description
    -----------
    Normalize values into the representation expected by downstream code. This function implements the normalize weights step.
    
    Parameters
    ----------
    weights : Sequence[float] (input)
        Numeric hyperparameter controlling weighting, filtering, or loss behavior. Units: dimensionless.
    count : int (input)
        The count value supplied to this function. Units: count.
    
    Returns
    -------
    List[float]
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
    if len(weights) != count:
        raise ValueError(f"Expected {count} ensemble weights, got {len(weights)}.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def repair_checkpoint_path_for_local_repo(path: Path) -> Path:
    """
    Description
    -----------
    Repair saved paths or checkpoint references for the current machine. This function implements the repair checkpoint path for local repo step.
    
    Parameters
    ----------
    path : Path (input)
        Filesystem path used by this step.
    
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
    if path.exists():
        return path
    text = str(path)
    parts = path.parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        candidate = repo_root().joinpath(*parts[runs_idx:])
        if candidate.exists():
            return candidate
    if "<USER>" in text:
        candidate = Path(text.replace(r"C:\Users\<USER>\Desktop\BENG 280C Project\Optimal-LV-Placement-ML", str(repo_root())))
        if candidate.exists():
            return candidate
    return path


def average_checkpoint_state_dicts(checkpoint_paths: Sequence[Path], weights: Sequence[float], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Description
    -----------
    Implement the average checkpoint state dicts helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    checkpoint_paths : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    weights : Sequence[float] (input)
        Numeric hyperparameter controlling weighting, filtering, or loss behavior. Units: dimensionless.
    device : torch.device (input)
        Torch device used for tensor and model computation.
    
    Returns
    -------
    Dict[str, torch.Tensor]
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
    averaged: Dict[str, torch.Tensor] = {}
    reference_keys: Optional[set[str]] = None
    for checkpoint_idx, (checkpoint_path, weight) in enumerate(zip(checkpoint_paths, weights)):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        keys = set(state_dict.keys())
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise RuntimeError(f"Checkpoint state_dict keys do not match for {checkpoint_path}")

        for key, tensor in state_dict.items():
            tensor = tensor.detach().to(device)
            if not torch.is_floating_point(tensor):
                if checkpoint_idx == 0:
                    averaged[key] = tensor.clone()
                continue
            weighted_tensor = tensor.float() * float(weight)
            averaged[key] = weighted_tensor if checkpoint_idx == 0 else averaged[key] + weighted_tensor
    return averaged


def resolve_warm_start_checkpoint(
    best_run_dir: Path,
    output_dir: Path,
    explicit_checkpoint: Optional[Path],
    device: torch.device,
) -> Path:
    """
    Description
    -----------
    Resolve paths or configuration references into concrete runtime values. This function implements the resolve warm start checkpoint step.
    
    Parameters
    ----------
    best_run_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    output_dir : Path (input)
        Directory where outputs are written.
    explicit_checkpoint : Optional[Path] (input)
        The explicit checkpoint value supplied to this function.
    device : torch.device (input)
        Torch device used for tensor and model computation.
    
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
    if explicit_checkpoint is not None:
        if not explicit_checkpoint.exists():
            raise FileNotFoundError(f"Explicit checkpoint does not exist: {explicit_checkpoint}")
        return explicit_checkpoint

    direct_checkpoint = best_run_dir / "weights" / "best_supervised_model.pth"
    if direct_checkpoint.exists():
        return direct_checkpoint

    config_path = best_run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No direct checkpoint and no config.json found in {best_run_dir}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_paths = [repair_checkpoint_path_for_local_repo(Path(path)) for path in payload.get("ensemble_checkpoints", [])]
    weights = payload.get("ensemble_weights", [])
    if not checkpoint_paths:
        raise FileNotFoundError(f"No ensemble_checkpoints listed in {config_path}")
    missing = [path for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ensemble source checkpoint(s):\n" + "\n".join(str(path) for path in missing))

    weights = normalize_weights(weights if weights else [1.0] * len(checkpoint_paths), len(checkpoint_paths))
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    averaged_path = weights_dir / "warm_start_from_ensemble_average.pth"
    averaged_state = average_checkpoint_state_dicts(checkpoint_paths, weights, device=device)
    torch.save(
        {
            "model_state_dict": averaged_state,
            "source_run": str(best_run_dir),
            "source_checkpoints": [str(path) for path in checkpoint_paths],
            "source_weights": weights,
            "note": "Weighted parameter average of ensemble source checkpoints for continuation warm-start.",
        },
        averaged_path,
    )
    print(f"Created averaged ensemble warm-start checkpoint: {averaged_path}")
    return averaged_path


def configure_continuation(args: argparse.Namespace) -> Tuple[Config, Path, Path]:
    """
    Description
    -----------
    Implement the configure continuation helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    args : argparse.Namespace (input)
        Parsed command-line argument namespace.
    
    Returns
    -------
    Tuple[Config, Path, Path]
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
    root = repo_root()
    best_run_dir = resolve_path(args.best_run_dir, root)
    output_dir = resolve_path(args.output_dir, root)
    explicit_checkpoint = resolve_path(args.checkpoint, root) if args.checkpoint else None

    ensure_output_is_safe(output_dir, allow_existing=args.allow_existing_output)
    cfg = load_config_from_run(best_run_dir)
    repair_dataset_roots_for_local_machine(cfg)
    cfg.work_dir = str(output_dir)
    cfg.eval_only = False
    cfg.warm_start_checkpoint = None
    cfg.supervised_epochs = int(args.epochs)
    cfg.finetune_epochs = 0
    cfg.enable_pseudo_labeling = False
    cfg.learning_rate = float(args.learning_rate)
    cfg.early_stopping_patience = int(args.early_stopping_patience)
    cfg.early_stopping_min_delta = float(args.early_stopping_min_delta)
    cfg.gradient_clip_norm = float(args.gradient_clip_norm)
    cfg.use_reduce_lr_on_plateau = True
    cfg.lr_reduce_factor = float(args.lr_reduce_factor)
    cfg.lr_reduce_patience = int(args.lr_reduce_patience)
    cfg.min_learning_rate = float(args.min_learning_rate)
    cfg.save_overlays = not args.no_overlays
    cfg.max_overlay_cases = int(args.max_overlay_cases)
    cfg.target_set = args.target_set
    cfg.electrode_loss_weight = float(args.electrode_loss_weight)
    cfg.anatomy_loss_weight = float(args.anatomy_loss_weight)
    if args.checkpoint_metric:
        cfg.checkpoint_metric = args.checkpoint_metric
        cfg.checkpoint_mode = args.checkpoint_mode
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.check:
        cfg.num_workers = 0
        cfg.max_overlay_cases = 0
    return cfg, best_run_dir, explicit_checkpoint


def build_data_and_model(cfg: Config, warm_start_checkpoint: Path, device: torch.device):
    """
    Description
    -----------
    Construct a configured object used by the pipeline. This function implements the build data and model step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    warm_start_checkpoint : Path (input)
        The warm start checkpoint value supplied to this function.
    device : torch.device (input)
        Torch device used for tensor and model computation.
    
    Returns
    -------
    tuple
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
    seed_everything(cfg.seed)
    ensure_dirs(cfg)
    cache = build_preprocessed_cache(cfg)
    labeled_files = cache["labeled"]
    train_files, val_files = create_train_val_split(labeled_files, seed=cfg.seed, val_fraction=0.20)
    assert_no_split_overlap(train_files, val_files)

    train_loader = create_train_loader(train_files, pseudo_flags=[False] * len(train_files), cfg=cfg)
    val_loader = create_val_loader(val_files, cfg)
    model = build_model(cfg).to(device)
    load_checkpoint_weights(model, warm_start_checkpoint, device)
    print(f"Loaded warm-start checkpoint: {warm_start_checkpoint}")
    return model, train_files, val_files, train_loader, val_loader


def run_check(cfg: Config, best_run_dir: Path, explicit_checkpoint: Optional[Path]) -> None:
    """
    Description
    -----------
    Run one pipeline stage or orchestration workflow. This function implements the run check step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    best_run_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    explicit_checkpoint : Optional[Path] (input)
        The explicit checkpoint value supplied to this function.
    
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
    print("Running continuation dry check...")
    check_dataset_roots(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dirs(cfg)
    warm_start = resolve_warm_start_checkpoint(best_run_dir, Path(cfg.work_dir), explicit_checkpoint, device)
    model, _train_files, _val_files, train_loader, _val_loader = build_data_and_model(cfg, warm_start, device)
    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    with torch.no_grad():
        logits = model(images)
    print(f"One batch loaded: image tensor shape={tuple(images.shape)}")
    print(f"Model forward OK: logits shape={tuple(logits.shape)}")
    print(f"Output directory is separate from best run: {Path(cfg.work_dir)}")
    print("Dry check complete. No training was launched.")


def run_training(cfg: Config, best_run_dir: Path, explicit_checkpoint: Optional[Path]) -> None:
    """
    Description
    -----------
    Run one pipeline stage or orchestration workflow. This function implements the run training step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    best_run_dir : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    explicit_checkpoint : Optional[Path] (input)
        The explicit checkpoint value supplied to this function.
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Continuation output dir: {Path(cfg.work_dir)}")
    print(f"Epoch cap: {cfg.supervised_epochs}; early stopping patience: {cfg.early_stopping_patience}")
    print(
        "Target set/loss weights: "
        f"target_set={cfg.target_set}, "
        f"electrode_loss_weight={cfg.electrode_loss_weight}, "
        f"anatomy_loss_weight={cfg.anatomy_loss_weight}"
    )
    print(f"Primary checkpoint metric: {cfg.checkpoint_metric} ({cfg.checkpoint_mode})")
    check_dataset_roots(cfg)
    ensure_dirs(cfg)
    warm_start = resolve_warm_start_checkpoint(best_run_dir, Path(cfg.work_dir), explicit_checkpoint, device)
    model, train_files, _val_files, train_loader, val_loader = build_data_and_model(cfg, warm_start, device)

    class_weights = compute_class_weights(train_files, cfg).to(device)
    loss_fn = build_loss_fn(class_weights=class_weights, cfg=cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    gpu_aug = build_gpu_train_augment(cfg)
    checkpoint_path = cfg.weights_dir / "best_supervised_model.pth"

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        cfg=cfg,
        epochs=cfg.supervised_epochs,
        checkpoint_path=checkpoint_path,
        stage_name="continued_supervised",
        gpu_aug=gpu_aug,
    )
    history.to_csv(Path(cfg.work_dir) / "training_history_all_stages.csv", index=False)
    load_checkpoint_weights(model, checkpoint_path, device)

    print("Starting final evaluation for continuation run...")
    metrics = evaluate_model(model=model, val_loader=val_loader, device=device, cfg=cfg)
    save_metrics_to_csv(metrics, cfg)
    plot_training_history(history, cfg)
    plot_dice_boxplot(metrics["per_sample_df"], cfg)
    plot_per_class_dice(metrics["per_class_df"], cfg)
    plot_precision_recall_curve(metrics["pr_curve_df"], cfg)
    plot_confusion_matrix_heatmap(metrics["confusion_df"], cfg)
    if cfg.save_overlays:
        save_prediction_overlays(model=model, val_loader=val_loader, device=device, cfg=cfg, max_cases=cfg.max_overlay_cases)
    run_presentation_outputs(cfg, checkpoint_path)

    final_model_path = cfg.weights_dir / "final_model_weights.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "raw_to_contig": RAW_TO_CONTIG,
            "config": asdict(cfg),
            "warm_start_checkpoint": str(warm_start),
        },
        final_model_path,
    )
    print(f"Saved continuation checkpoint: {checkpoint_path}")
    print(f"Saved best electrode-centroid checkpoint: {checkpoint_path.with_name('best_electrode_centroid_model.pth')}")
    print(f"Saved final continuation weights: {final_model_path}")
    print(f"Saved metrics in: {cfg.metrics_dir}")
    print(f"Saved plots in: {cfg.plots_dir}")


def parse_args() -> argparse.Namespace:
    """
    Description
    -----------
    Build or parse command-line arguments for continue_training_from_best.py.
    
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
    parser = argparse.ArgumentParser(description="Continue training from the current best CRT lead localization run.")
    parser.add_argument("--best-run-dir", default="runs/cardiac_leads_ensemble_v3_v6")
    parser.add_argument("--output-dir", default="runs/cardiac_leads_continued_from_v3_v6_long")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit warm-start checkpoint. Overrides ensemble resolution.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--target-set", choices=("all", "electrodes", "anatomy"), default="all")
    parser.add_argument("--electrode-loss-weight", type=float, default=1.0)
    parser.add_argument("--anatomy-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-metric",
        default=None,
        help="Optional metric to drive early stopping/checkpointing, e.g. val_centroid_dist_electrodes.",
    )
    parser.add_argument("--checkpoint-mode", choices=("min", "max"), default="min")
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--lr-reduce-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-7)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-overlay-cases", type=int, default=3)
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--check", action="store_true", help="Run dry checks only; do not train.")
    return parser.parse_args()


def main() -> None:
    """
    Description
    -----------
    Run the command-line workflow implemented by continue_training_from_best.py.
    
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
    cfg, best_run_dir, explicit_checkpoint = configure_continuation(args)
    if args.check:
        run_check(cfg, best_run_dir, explicit_checkpoint)
    else:
        run_training(cfg, best_run_dir, explicit_checkpoint)


if __name__ == "__main__":
    main()
