from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from S1_DataLoading_Preprocessing import ANATOMY_CLASSES, CLASS_NAMES, ELECTRODE_CLASSES
from S10_Bullseye_Lead_Visualization import generate_bullseye_plots
from S11_Centroid_Export import export_centroids, find_default_run_dir, repo_root_from_here, resolve_path


CLASS_ORDER = list(range(1, 10))


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def mean_sem(values: pd.Series) -> Tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan, np.nan
    sem = float(clean.std(ddof=1) / math.sqrt(len(clean))) if len(clean) > 1 else 0.0
    return float(clean.mean()), sem


def save_pipeline_overview(output_dir: Path) -> Path:
    output_path = output_dir / "pipeline_overview.png"
    stages = [
        "CT input",
        "Preprocessing",
        "3D U-Net\nMONAI",
        "Segmentation /\nlandmark prediction",
        "Bullseye\nmapping",
        "CRT\ninterpretation",
    ]
    colors = ["#f4f1de", "#e7ecef", "#d8e2dc", "#dbeafe", "#fee2e2", "#ede9fe"]

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axis("off")
    x_positions = np.linspace(0.08, 0.92, len(stages))
    width = 0.135
    height = 0.44

    for idx, (x, label) in enumerate(zip(x_positions, stages)):
        rect = plt.Rectangle((x - width / 2, 0.35), width, height, facecolor=colors[idx], edgecolor="#263238", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, 0.57, label, ha="center", va="center", fontsize=12, weight="bold")
        if idx < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - width / 2 - 0.012, 0.57),
                xytext=(x + width / 2 + 0.012, 0.57),
                arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": "#263238"},
            )

    ax.text(0.5, 0.15, "Sparse 3D landmark masks are evaluated as centroid localization and shown in CRT-relevant bullseye space.", ha="center", fontsize=10)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_per_class_centroid_error(errors_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "per_class_centroid_error.png"
    rows = []
    metric = "distance_mm" if "distance_mm" in errors_df.columns and errors_df["distance_mm"].notna().any() else "distance_voxels"
    for class_id in CLASS_ORDER:
        subset = errors_df[errors_df["class_id"] == class_id]
        mean, sem = mean_sem(subset[metric])
        rows.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], "mean": mean, "sem": sem})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = ["#1f77b4" if cid in ELECTRODE_CLASSES[:4] else "#d62728" if cid in ELECTRODE_CLASSES else "#4b5563" for cid in plot_df["class_id"]]
    ax.bar(plot_df["class_name"], plot_df["mean"], yerr=plot_df["sem"], color=colors, capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Centroid distance (mm)" if metric == "distance_mm" else "Centroid distance (voxels)")
    ax.set_xlabel("Class")
    ax.set_title("Per-Class Centroid Localization Error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_per_class_dice(run_dir: Path, output_dir: Path) -> Optional[Path]:
    metrics_path = run_dir / "metrics" / "per_class_metrics.csv"
    per_class_df = safe_read_csv(metrics_path)
    if per_class_df is None or "dice_mean" not in per_class_df.columns:
        print(f"Skipping per-class Dice figure; missing {metrics_path}")
        return None

    plot_df = per_class_df[per_class_df["class_id"].isin(CLASS_ORDER)].copy()
    plot_df["class_name"] = plot_df["class_id"].map(lambda cid: CLASS_NAMES[int(cid)])
    output_path = output_dir / "per_class_dice.png"

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = ["#1f77b4" if cid in ELECTRODE_CLASSES[:4] else "#d62728" if cid in ELECTRODE_CLASSES else "#4b5563" for cid in plot_df["class_id"]]
    ax.bar(plot_df["class_name"], plot_df["dice_mean"], color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Dice")
    ax.set_xlabel("Class")
    ax.set_title("Per-Class Dice Score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def overlay_path_for_patient(run_dir: Path, patient_id: str) -> Optional[Path]:
    exact = run_dir / "overlays" / f"{patient_id}_overlay.png"
    if exact.exists():
        return exact
    matches = sorted((run_dir / "overlays").glob(f"{patient_id}*_overlay.png"))
    return matches[0] if matches else None


def bullseye_path_for_patient(run_dir: Path, patient_id: str) -> Optional[Path]:
    path = run_dir / "bullseye_plots" / f"{patient_id}_bullseye.png"
    return path if path.exists() else None


def draw_image_or_placeholder(ax: plt.Axes, path: Optional[Path], title: str) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    if path is None or not path.exists():
        ax.text(0.5, 0.5, "Not available", ha="center", va="center", fontsize=11, color="0.35")
        return
    image = mpimg.imread(path)
    ax.imshow(image)


def save_example_patient_panel(run_dir: Path, coordinates_df: pd.DataFrame, output_dir: Path, max_patients: int = 3) -> Optional[Path]:
    overlay_patients = [p.stem.replace("_overlay", "") for p in sorted((run_dir / "overlays").glob("*_overlay.png"))]
    all_patients = sorted(coordinates_df["patient_id"].dropna().unique().tolist())
    patients = overlay_patients[:max_patients] or all_patients[:max_patients]
    if not patients:
        print("Skipping example patient panel; no patients available.")
        return None

    output_path = output_dir / "example_patient_panel.png"
    fig, axes = plt.subplots(len(patients), 2, figsize=(13, 4.4 * len(patients)))
    axes = np.atleast_2d(axes)

    for row_idx, patient_id in enumerate(patients):
        draw_image_or_placeholder(axes[row_idx, 0], overlay_path_for_patient(run_dir, patient_id), f"{patient_id}: CT, GT, Prediction")
        draw_image_or_placeholder(axes[row_idx, 1], bullseye_path_for_patient(run_dir, patient_id), f"{patient_id}: Bullseye")

    fig.suptitle("Representative Validation Examples", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_failure_case_panel(run_dir: Path, errors_df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    valid = errors_df[np.isfinite(pd.to_numeric(errors_df["distance_voxels"], errors="coerce"))].copy()
    valid = valid[valid["class_id"].isin(ELECTRODE_CLASSES)]
    if valid.empty:
        print("Skipping failure case panel; no finite centroid errors available.")
        return None
    overlay_patients = {p.stem.replace("_overlay", "") for p in (run_dir / "overlays").glob("*_overlay.png")}
    valid_with_overlay = valid[valid["patient_id"].isin(overlay_patients)]
    selection_df = valid_with_overlay if not valid_with_overlay.empty else valid
    worst = selection_df.sort_values("distance_voxels", ascending=False).iloc[0]
    patient_id = str(worst["patient_id"])
    class_name = str(worst["class_name"])
    error = float(worst["distance_voxels"])

    output_path = output_dir / "failure_case_panel.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    draw_image_or_placeholder(axes[0], overlay_path_for_patient(run_dir, patient_id), f"Worst case overlay: {patient_id}")
    draw_image_or_placeholder(axes[1], bullseye_path_for_patient(run_dir, patient_id), f"Worst bullseye: {patient_id}")
    fig.suptitle(f"Failure Case: {class_name}, centroid error {error:.1f} voxels", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_training_curves(run_dir: Path, output_dir: Path) -> Optional[Path]:
    history_path = run_dir / "training_history_all_stages.csv"
    if not history_path.exists():
        history_path = run_dir / "history_supervised.csv"
    history_df = safe_read_csv(history_path)
    if history_df is None or history_df.empty:
        print(f"Skipping training curves; missing history CSV in {run_dir}")
        return None

    output_path = output_dir / "training_curves.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    curve_specs = [("train_loss", "Train Loss"), ("val_loss", "Validation Loss"), ("val_dice", "Validation Dice")]
    for ax, (column, title) in zip(axes, curve_specs):
        if column not in history_df.columns:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{column} not found", ha="center", va="center")
            continue
        for stage, stage_df in history_df.groupby("stage") if "stage" in history_df.columns else [("training", history_df)]:
            ax.plot(stage_df["epoch"], stage_df[column], marker="o", linewidth=1.8, label=str(stage))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_training_validation_scatter_summary(run_dir: Path, output_dir: Path) -> Optional[Path]:
    history_path = run_dir / "training_history_all_stages.csv"
    if not history_path.exists():
        history_path = run_dir / "history_supervised.csv"
    history_df = safe_read_csv(history_path)
    if history_df is None or history_df.empty:
        print(f"Skipping scatter training summary; missing history CSV in {run_dir}")
        return None

    output_path = output_dir / "training_validation_scatter_summary.png"
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    axes = axes.ravel()

    color_map = {
        "train_loss": "#1f77b4",
        "val_loss": "#d62728",
        "val_dice": "#2ca02c",
        "val_centroid_dist": "#9467bd",
        "val_focus_centroid_dist": "#ff7f0e",
    }

    def scatter_metric(ax: plt.Axes, column: str, label: str, color: str, marker: str = "o") -> None:
        if column not in history_df.columns:
            ax.text(0.5, 0.5, f"{column} not found", ha="center", va="center")
            return
        ax.scatter(history_df["epoch"], history_df[column], s=58, color=color, marker=marker, label=label, edgecolors="black", linewidths=0.45)
        ax.plot(history_df["epoch"], history_df[column], color=color, linewidth=1.2, alpha=0.65)

    scatter_metric(axes[0], "train_loss", "Training loss", color_map["train_loss"], marker="o")
    scatter_metric(axes[0], "val_loss", "Validation loss", color_map["val_loss"], marker="s")
    axes[0].set_title("Training and Validation Loss vs Epoch")
    axes[0].set_ylabel("Loss")

    scatter_metric(axes[1], "val_dice", "Validation Dice", color_map["val_dice"], marker="D")
    axes[1].set_title("Validation Dice Score vs Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0.0, max(1.0, float(pd.to_numeric(history_df.get("val_dice"), errors="coerce").max()) + 0.05))

    scatter_metric(axes[2], "val_centroid_dist", "Validation centroid distance", color_map["val_centroid_dist"], marker="^")
    axes[2].set_title("Validation Centroid Distance vs Epoch")
    axes[2].set_ylabel("Distance (voxels)")

    scatter_metric(axes[3], "val_focus_centroid_dist", "Focused-class centroid distance", color_map["val_focus_centroid_dist"], marker="P")
    focus_columns = [col for col in history_df.columns if col.startswith("val_centroid_dist_") and col != "val_centroid_dist"]
    focus_colors = ["#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#1f77b4"]
    for idx, column in enumerate(focus_columns):
        label = column.replace("val_centroid_dist_", "")
        color = focus_colors[idx % len(focus_colors)]
        axes[3].scatter(history_df["epoch"], history_df[column], s=36, color=color, marker="x", label=label)
        axes[3].plot(history_df["epoch"], history_df[column], color=color, linewidth=0.9, alpha=0.45)
    axes[3].set_title("Focused Class Centroid Distance vs Epoch")
    axes[3].set_ylabel("Distance (voxels)")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Training and Validation Metrics Across Epochs", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_clinical_interpretation_summary(errors_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "clinical_interpretation_summary.md"
    valid = errors_df.copy()
    valid["distance_voxels"] = pd.to_numeric(valid["distance_voxels"], errors="coerce")
    lv_mean = valid[valid["class_id"].isin([1, 2, 3, 4])]["distance_voxels"].mean()
    rv_mean = valid[valid["class_id"].isin([5, 6])]["distance_voxels"].mean()
    class_means = valid.groupby(["class_id", "class_name"], as_index=False)["distance_voxels"].mean().dropna()
    worst = class_means.sort_values("distance_voxels", ascending=False).iloc[0] if not class_means.empty else None
    best = class_means.sort_values("distance_voxels", ascending=True).iloc[0] if not class_means.empty else None

    lines = [
        "# Clinical Interpretation Summary",
        "",
        f"- Mean centroid error for LV electrodes: {lv_mean:.2f} voxels" if np.isfinite(lv_mean) else "- Mean centroid error for LV electrodes: unavailable",
        f"- Mean centroid error for RV electrodes: {rv_mean:.2f} voxels" if np.isfinite(rv_mean) else "- Mean centroid error for RV electrodes: unavailable",
        f"- Worst class: {worst['class_name']} ({worst['distance_voxels']:.2f} voxels)" if worst is not None else "- Worst class: unavailable",
        f"- Best class: {best['class_name']} ({best['distance_voxels']:.2f} voxels)" if best is not None else "- Best class: unavailable",
        "",
        "Because the labels are sparse landmark-style electrode regions, centroid localization error is more clinically meaningful than Dice alone. Bullseye plots translate 3D electrode predictions into CRT-relevant spatial views.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def generate_presentation_figures(
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
    force_centroids: bool = False,
) -> List[Path]:
    run_dir = run_dir.resolve()
    coordinates_path, errors_path = export_centroids(run_dir=run_dir, checkpoint_path=checkpoint_path, force=force_centroids)
    generate_bullseye_plots(run_dir=run_dir, checkpoint_path=checkpoint_path, force_centroids=False)

    coordinates_df = pd.read_csv(coordinates_path)
    errors_df = pd.read_csv(errors_path)
    output_dir = run_dir / "presentation_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Optional[Path]] = [
        save_pipeline_overview(output_dir),
        save_per_class_centroid_error(errors_df, output_dir),
        save_per_class_dice(run_dir, output_dir),
        save_example_patient_panel(run_dir, coordinates_df, output_dir),
        save_failure_case_panel(run_dir, errors_df, output_dir),
        save_training_curves(run_dir, output_dir),
        save_training_validation_scatter_summary(run_dir, output_dir),
        save_clinical_interpretation_summary(errors_df, output_dir),
    ]
    saved_paths = [path for path in saved if path is not None]
    print(f"Saved presentation figures to: {output_dir}")
    return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate final presentation figures from existing model outputs.")
    parser.add_argument("--run-dir", type=str, default=None, help="Completed run directory. Defaults to latest run with a best checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path if centroid CSVs must be generated.")
    parser.add_argument("--force-centroids", action="store_true", help="Recompute centroid CSVs before plotting.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = repo_root_from_here()
    run_dir = resolve_path(args.run_dir, repo_root) if args.run_dir else find_default_run_dir(repo_root)
    checkpoint = resolve_path(args.checkpoint, repo_root) if args.checkpoint else None
    generate_presentation_figures(run_dir=run_dir, checkpoint_path=checkpoint, force_centroids=args.force_centroids)


if __name__ == "__main__":
    main()
