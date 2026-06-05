"""
Analyze electrode-vs-anatomy performance and inspect NIfTI orientation metadata.

This script does not train. It reads an existing run directory, exports group
metrics for electrodes and anatomy landmarks, optionally inspects source NIfTI
orientation, and writes a short experiment recommendation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from S1_DataLoading_Preprocessing import (
    ANATOMY_CLASSES,
    CLASS_NAMES,
    ELECTRODE_CLASSES,
    Config,
    discover_cases_from_dataset_roots,
    nifti_orientation_codes,
)


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


def load_required_csv(path: Path) -> pd.DataFrame:
    """
    Description
    -----------
    Load data, configuration, weights, or metadata from disk. This function implements the load required csv step.
    
    Parameters
    ----------
    path : Path (input)
        Filesystem path used by this step.
    
    Returns
    -------
    pd.DataFrame
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
    if not path.exists():
        raise FileNotFoundError(f"Missing required metrics file: {path}")
    return pd.read_csv(path)


def class_group(class_id: int) -> str:
    """
    Description
    -----------
    Implement the class group helper for the CRT lead localization pipeline.
    
    Parameters
    ----------
    class_id : int (input)
        Class identifier, class name, or number of modeled classes.
    
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
    if class_id in ELECTRODE_CLASSES:
        return "electrode"
    if class_id in ANATOMY_CLASSES:
        return "anatomy"
    return "background"


def summarize_run(run_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Description
    -----------
    Summarize run outputs or records into compact metrics. This function implements the summarize run step.
    
    Parameters
    ----------
    run_dir : Path (input)
        Saved run directory.
    
    Returns
    -------
    dict[str, pd.DataFrame]
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
    metrics_dir = run_dir / "metrics"
    per_class = load_required_csv(metrics_dir / "per_class_metrics.csv")
    summary = load_required_csv(metrics_dir / "summary_metrics.csv")
    centroid_errors = load_required_csv(metrics_dir / "centroid_errors.csv")

    per_class["group"] = per_class["class_id"].map(class_group)
    foreground = per_class[per_class["class_id"] > 0].copy()
    grouped = (
        foreground.groupby("group", as_index=False)
        .agg(
            n_classes=("class_id", "count"),
            mean_centroid_error_voxels=("centroid_dist_mean", "mean"),
            mean_dice=("dice_mean", "mean"),
            mean_within_5vox=("accuracy_within_5vox", "mean"),
            mean_within_10vox=("accuracy_within_10vox", "mean"),
            mean_within_15vox=("accuracy_within_15vox", "mean"),
        )
        .sort_values("group")
    )

    per_class_out = foreground[
        [
            "class_id",
            "class_name",
            "group",
            "centroid_dist_mean",
            "accuracy_within_5vox",
            "accuracy_within_10vox",
            "accuracy_within_15vox",
            "dice_mean",
            "precision",
            "recall",
            "f1",
        ]
    ].copy()

    metrics_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(metrics_dir / "target_group_summary.csv", index=False)
    per_class_out.to_csv(metrics_dir / "target_set_per_class_summary.csv", index=False)

    return {
        "summary": summary,
        "per_class": per_class_out,
        "grouped": grouped,
        "centroid_errors": centroid_errors,
    }


def plot_per_class_error(per_class: pd.DataFrame, run_dir: Path) -> Path:
    """
    Description
    -----------
    Create a visualization and save or populate the requested figure. This function implements the plot per class error step.
    
    Parameters
    ----------
    per_class : pd.DataFrame (input)
        Class identifier, class name, or number of modeled classes.
    run_dir : Path (input)
        Saved run directory.
    
    Returns
    -------
    Path
        Generated artifact path, summary object, or status value produced by the workflow branch.
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
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_df = per_class.sort_values("class_id")
    colors = ["#2563eb" if group == "electrode" else "#16a34a" for group in plot_df["group"]]

    plt.figure(figsize=(9, 4.8))
    plt.bar(plot_df["class_name"], plot_df["centroid_dist_mean"], color=colors)
    plt.axhline(10.0, color="#444444", linestyle="--", linewidth=1, label="10 voxel target")
    plt.ylabel("Mean centroid error (voxels)")
    plt.xlabel("Class")
    plt.title("Per-Class Centroid Error: Electrodes vs Anatomy")
    plt.legend()
    plt.tight_layout()
    out_path = plot_dir / "target_set_per_class_centroid_error.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def find_history_files(run_dirs: Sequence[Path]) -> list[Path]:
    """
    Description
    -----------
    Find files or records matching the requested criteria. This function implements the find history files step.
    
    Parameters
    ----------
    run_dirs : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    
    Returns
    -------
    list[Path]
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
    history_files: list[Path] = []
    for run_dir in run_dirs:
        for name in ("history_supervised.csv", "training_history_all_stages.csv", "training_history.csv"):
            path = run_dir / name
            if path.exists():
                history_files.append(path)
                break
    return history_files


def summarize_history(history_files: Iterable[Path], output_dir: Path) -> Optional[Path]:
    """
    Description
    -----------
    Summarize run outputs or records into compact metrics. This function implements the summarize history step.
    
    Parameters
    ----------
    history_files : Iterable[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    output_dir : Path (input)
        Directory where outputs are written.
    
    Returns
    -------
    Optional[Path]
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
    rows = []
    for path in history_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        for metric in ("val_centroid_dist_electrodes", "val_centroid_dist_anatomy", "val_centroid_dist", "val_dice_electrodes", "val_dice"):
            if metric not in df.columns:
                continue
            series = pd.to_numeric(df[metric], errors="coerce")
            if series.notna().any():
                idx = series.idxmax() if "dice" in metric else series.idxmin()
                rows.append(
                    {
                        "run": path.parent.name,
                        "history_file": path.name,
                        "metric": metric,
                        "best_epoch": int(df.loc[idx, "epoch"]) if "epoch" in df.columns and pd.notna(df.loc[idx, "epoch"]) else idx,
                        "best_value": float(series.loc[idx]),
                        "last_value": float(series.dropna().iloc[-1]),
                    }
                )

    if not rows:
        return None

    history_summary = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "history_target_metric_summary.csv"
    history_summary.to_csv(out_csv, index=False)

    plot_df = history_summary[history_summary["metric"].isin(["val_centroid_dist_electrodes", "val_centroid_dist"])]
    if not plot_df.empty:
        plt.figure(figsize=(9, 4.8))
        for metric, metric_df in plot_df.groupby("metric"):
            plt.scatter(metric_df["run"], metric_df["best_value"], label=f"best {metric}", s=45)
            plt.scatter(metric_df["run"], metric_df["last_value"], marker="x", label=f"last {metric}", s=55)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Centroid error (voxels)")
        plt.title("Best vs Last Validation Centroid Metrics by Run")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "history_target_metric_summary.png", dpi=220)
        plt.close()

    return out_csv


def inspect_orientation(nifti_path: Path) -> dict[str, object]:
    """
    Description
    -----------
    Inspect data or metadata for diagnostics. This function implements the inspect orientation step.
    
    Parameters
    ----------
    nifti_path : Path (input)
        Filesystem location used for reading inputs or writing outputs.
    
    Returns
    -------
    dict[str, object]
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
    nii = nib.load(str(nifti_path))
    return {
        "path": str(nifti_path),
        "shape_xyz": tuple(int(x) for x in nii.shape[:3]),
        "zooms_xyz_mm": tuple(float(x) for x in nii.header.get_zooms()[:3]),
        "orientation_codes": "".join(nifti_orientation_codes(nifti_path)),
        "affine": np.asarray(nii.affine, dtype=float),
    }


def first_labeled_image_from_config(cfg: Config) -> Optional[Path]:
    """
    Description
    -----------
    Derive first labeled image from config for downstream CRT lead localization steps.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
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
    try:
        discovered = discover_cases_from_dataset_roots(cfg.raw_dataset_roots)
    except Exception as exc:
        print(f"Orientation scan skipped: could not discover raw cases ({exc}).")
        return None
    pairs = discovered.get("labeled_pairs", [])
    if not pairs:
        return None
    return Path(pairs[0][0])


def write_recommendation(run_dir: Path, tables: dict[str, pd.DataFrame], history_csv: Optional[Path], orientation_info: Optional[dict[str, object]]) -> Path:
    """
    Description
    -----------
    Write a diagnostic or summary artifact to disk. This function implements the write recommendation step.
    
    Parameters
    ----------
    run_dir : Path (input)
        Saved run directory.
    tables : dict[str, pd.DataFrame] (input)
        The tables value supplied to this function.
    history_csv : Optional[Path] (input)
        The history csv value supplied to this function.
    orientation_info : Optional[dict[str, object]] (input)
        The orientation info value supplied to this function.
    
    Returns
    -------
    Path
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
    grouped = tables["grouped"]
    per_class = tables["per_class"]
    summary = tables["summary"].set_index("metric")["value"].to_dict()

    electrode_error = float(summary.get("mean_centroid_dist_electrodes", np.nan))
    anatomy_error = float(summary.get("mean_centroid_dist_anatomy", np.nan))
    elec_within_10 = float(summary.get("landmark_accuracy_within_10vox_electrodes", np.nan)) * 100.0
    anatomy_within_10 = float(summary.get("landmark_accuracy_within_10vox_anatomy", np.nan)) * 100.0
    worst_row = per_class.sort_values("centroid_dist_mean", ascending=False).iloc[0]

    lines = [
        "# Target-Set Experiment Recommendation",
        "",
        "## Current Best Run Evidence",
        f"- Run: `{run_dir}`",
        f"- Mean electrode centroid error: {electrode_error:.2f} voxels",
        f"- Mean anatomy landmark centroid error: {anatomy_error:.2f} voxels",
        f"- Electrodes within 10 voxels: {elec_within_10:.2f}%",
        f"- Anatomy landmarks within 10 voxels: {anatomy_within_10:.2f}%",
        f"- Worst class by centroid error: {worst_row['class_name']} ({float(worst_row['centroid_dist_mean']):.2f} voxels)",
        "",
        "## Decision",
        "Long continuation of the same all-class objective is not the first choice unless a new run shows electrode-only centroid error improving.",
        "The safer next experiment is electrode-focused training from the best available weights, using electrode-only centroid localization as the primary model-selection metric.",
        "",
        "## Why",
        "Electrodes are bright metallic structures and are performing better than ANT/APEX/BASE. Anatomy landmarks are harder sparse anatomical targets and can pull the shared loss/selection metric away from electrode localization.",
        "",
        "## Orientation Note",
        "The NIfTI affine can identify patient/world orientation directions such as left/right, anterior/posterior, and superior/inferior. It cannot by itself identify the cardiac apex/base or the heart long axis. For cardiac top/bottom and bullseye normalization, this project still needs APEX/BASE/ANT landmarks, a heart segmentation, an atlas, or a separate anatomical detector.",
    ]
    if orientation_info:
        lines.extend(
            [
                "",
                "## Example NIfTI Orientation",
                f"- File: `{orientation_info['path']}`",
                f"- Orientation codes: `{orientation_info['orientation_codes']}`",
                f"- Shape XYZ: `{orientation_info['shape_xyz']}`",
                f"- Zooms XYZ mm: `{orientation_info['zooms_xyz_mm']}`",
            ]
        )
    if history_csv:
        lines.extend(["", "## History Summary", f"- Saved history trend summary: `{history_csv}`"])
    lines.extend(
        [
            "",
            "## Next Commands",
            "```powershell",
            "python active_code\\model_1\\S16_TargetSet_Analysis_Orientation.py --run-dir runs\\cardiac_leads_ensemble_v3_v6",
            "python active_code\\model_1\\continue_training_from_best.py --check --target-set electrodes --anatomy-loss-weight 0.0 --checkpoint-metric val_centroid_dist_electrodes --output-dir runs\\cardiac_leads_electrode_only_from_v3_v6",
            "python active_code\\model_1\\continue_training_from_best.py --target-set electrodes --anatomy-loss-weight 0.0 --checkpoint-metric val_centroid_dist_electrodes --epochs 150 --learning-rate 1e-6 --output-dir runs\\cardiac_leads_electrode_only_from_v3_v6",
            "python active_code\\model_1\\continue_training_from_best.py --target-set all --electrode-loss-weight 1.0 --anatomy-loss-weight 0.2 --checkpoint-metric val_centroid_dist_electrodes --epochs 150 --learning-rate 1e-6 --output-dir runs\\cardiac_leads_anatomy_downweighted_from_v3_v6",
            "```",
            "",
            "Primary decision metric: electrode-only centroid localization, especially percent of electrodes within 10 voxels and mean electrode centroid error.",
        ]
    )

    out_path = run_dir / "metrics" / "target_set_experiment_recommendation.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out_path


def parse_args() -> argparse.Namespace:
    """
    Description
    -----------
    Build or parse command-line arguments for S16_TargetSet_Analysis_Orientation.py.
    
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
    parser = argparse.ArgumentParser(description="Analyze electrode vs anatomy metrics and NIfTI orientation.")
    parser.add_argument("--run-dir", default="runs/cardiac_leads_ensemble_v3_v6")
    parser.add_argument(
        "--history-runs",
        nargs="*",
        default=[
            "runs/cardiac_leads_no_spatial_aug_v6",
            "runs/cardiac_leads_stabilized_selection_v7",
            "runs/cardiac_leads_warmstart_guard_v8",
        ],
    )
    parser.add_argument("--orientation-nifti", default=None, help="Optional raw NIfTI to inspect.")
    parser.add_argument("--skip-orientation", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Description
    -----------
    Run the command-line workflow implemented by S16_TargetSet_Analysis_Orientation.py.
    
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
    root = repo_root()
    args = parse_args()
    run_dir = resolve_path(args.run_dir, root)
    history_dirs = [resolve_path(path, root) for path in args.history_runs]

    tables = summarize_run(run_dir)
    plot_path = plot_per_class_error(tables["per_class"], run_dir)
    history_csv = summarize_history(find_history_files(history_dirs), run_dir / "plots")

    orientation_info = None
    if not args.skip_orientation:
        nifti_path = resolve_path(args.orientation_nifti, root) if args.orientation_nifti else first_labeled_image_from_config(Config())
        if nifti_path and nifti_path.exists():
            orientation_info = inspect_orientation(nifti_path)
            orientation_csv = run_dir / "metrics" / "orientation_example.csv"
            pd.DataFrame(
                [
                    {
                        "path": orientation_info["path"],
                        "shape_xyz": orientation_info["shape_xyz"],
                        "zooms_xyz_mm": orientation_info["zooms_xyz_mm"],
                        "orientation_codes": orientation_info["orientation_codes"],
                    }
                ]
            ).to_csv(orientation_csv, index=False)
            print(f"Saved orientation example: {orientation_csv}")

    recommendation_path = write_recommendation(run_dir, tables, history_csv, orientation_info)
    print(f"Saved group summary: {run_dir / 'metrics' / 'target_group_summary.csv'}")
    print(f"Saved per-class target summary: {run_dir / 'metrics' / 'target_set_per_class_summary.csv'}")
    print(f"Saved per-class plot: {plot_path}")
    print(f"Saved recommendation: {recommendation_path}")


if __name__ == "__main__":
    main()
