from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from S1_DataLoading_Preprocessing import CLASS_NAMES
from S11_Centroid_Export import export_centroids, find_default_run_dir, repo_root_from_here, resolve_path


ELECTRODE_CLASS_IDS = (1, 2, 3, 4, 5, 6)
LANDMARK_CLASS_IDS = {"ANT": 7, "Apex": 8, "Base": 9}


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    return vec / norm


def convert_xyz_to_bullseye(
    xyz: Sequence[float],
    apex_xyz: Optional[Sequence[float]],
    base_xyz: Optional[Sequence[float]],
    ant_xyz: Optional[Sequence[float]],
) -> Tuple[float, float]:
    """
    Convert x-y-z coordinates into a patient-specific bullseye angle/radius.

    Apex and Base define the long axis. ANT defines the angular zero direction
    after projection perpendicular to that long axis. If any landmark is absent,
    this falls back to normalized x-y projection and emits a warning.
    """
    point = np.asarray(xyz, dtype=np.float64)
    if not np.isfinite(point).all():
        return np.nan, np.nan

    if apex_xyz is None or base_xyz is None or ant_xyz is None:
        warnings.warn("Missing Apex/Base/ANT landmarks; using normalized x-y fallback for bullseye mapping.")
        theta = math.degrees(math.atan2(point[1], point[0])) % 360.0
        radius = min(float(np.linalg.norm(point[:2]) / max(np.linalg.norm(point), 1.0)), 1.2)
        return theta, radius

    apex = np.asarray(apex_xyz, dtype=np.float64)
    base = np.asarray(base_xyz, dtype=np.float64)
    ant = np.asarray(ant_xyz, dtype=np.float64)
    if not (np.isfinite(apex).all() and np.isfinite(base).all() and np.isfinite(ant).all()):
        warnings.warn("Invalid Apex/Base/ANT landmarks; using normalized x-y fallback for bullseye mapping.")
        theta = math.degrees(math.atan2(point[1], point[0])) % 360.0
        radius = min(float(np.linalg.norm(point[:2]) / max(np.linalg.norm(point), 1.0)), 1.2)
        return theta, radius

    long_vec = apex - base
    axis_len = float(np.linalg.norm(long_vec))
    long_axis = _normalize(long_vec)
    if long_axis is None:
        warnings.warn("Degenerate Apex/Base axis; using normalized x-y fallback for bullseye mapping.")
        theta = math.degrees(math.atan2(point[1], point[0])) % 360.0
        radius = min(float(np.linalg.norm(point[:2]) / max(np.linalg.norm(point), 1.0)), 1.2)
        return theta, radius

    ant_vec = ant - base
    ref_vec = ant_vec - np.dot(ant_vec, long_axis) * long_axis
    ref_axis = _normalize(ref_vec)
    if ref_axis is None:
        warnings.warn("ANT landmark lies on the long axis; using normalized x-y fallback for bullseye mapping.")
        theta = math.degrees(math.atan2(point[1], point[0])) % 360.0
        radius = min(float(np.linalg.norm(point[:2]) / max(np.linalg.norm(point), 1.0)), 1.2)
        return theta, radius

    side_axis = _normalize(np.cross(long_axis, ref_axis))
    if side_axis is None:
        warnings.warn("Could not build orthogonal bullseye frame; using normalized x-y fallback.")
        theta = math.degrees(math.atan2(point[1], point[0])) % 360.0
        radius = min(float(np.linalg.norm(point[:2]) / max(np.linalg.norm(point), 1.0)), 1.2)
        return theta, radius

    point_vec = point - base
    axial_fraction = float(np.dot(point_vec, long_axis) / max(axis_len, 1e-6))
    axis_point = base + axial_fraction * long_vec
    radial_vec = point - axis_point
    x_component = float(np.dot(radial_vec, ref_axis))
    y_component = float(np.dot(radial_vec, side_axis))
    theta = math.degrees(math.atan2(y_component, x_component)) % 360.0
    radius = min(float(np.linalg.norm(radial_vec) / max(axis_len * 0.50, 1.0)), 1.2)
    return theta, radius


def zyx_row_to_xyz(row: pd.Series) -> Optional[np.ndarray]:
    values = np.array([row["centroid_x"], row["centroid_y"], row["centroid_z"]], dtype=np.float64)
    return values if np.isfinite(values).all() else None


def patient_source_points(coordinates_df: pd.DataFrame, patient_id: str, source: str) -> Dict[int, Optional[np.ndarray]]:
    points: Dict[int, Optional[np.ndarray]] = {}
    subset = coordinates_df[(coordinates_df["patient_id"] == patient_id) & (coordinates_df["source"] == source)]
    for _, row in subset.iterrows():
        points[int(row["class_id"])] = None if bool(row["missing"]) else zyx_row_to_xyz(row)
    return points


def frame_landmarks(gt_points: Dict[int, Optional[np.ndarray]], pred_points: Dict[int, Optional[np.ndarray]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    apex = gt_points.get(LANDMARK_CLASS_IDS["Apex"])
    if apex is None:
        apex = pred_points.get(LANDMARK_CLASS_IDS["Apex"])
    base = gt_points.get(LANDMARK_CLASS_IDS["Base"])
    if base is None:
        base = pred_points.get(LANDMARK_CLASS_IDS["Base"])
    ant = gt_points.get(LANDMARK_CLASS_IDS["ANT"])
    if ant is None:
        ant = pred_points.get(LANDMARK_CLASS_IDS["ANT"])
    return apex, base, ant


def marker_style(class_id: int) -> Tuple[str, str]:
    if class_id in (1, 2, 3, 4):
        color = "#1f77b4"
    else:
        color = "#d62728"
    marker = "*" if class_id in (1, 5) else "o"
    return color, marker


def setup_polar_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=18)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0.0, 1.2)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_rlabel_position(18)
    theta_ticks = np.arange(10, 360, 20)
    ax.set_thetagrids(theta_ticks, labels=[f"{int(deg)}deg" for deg in theta_ticks])
    ax.grid(True, color="0.72", alpha=0.75, linewidth=0.8)
    ax.spines["polar"].set_color("0.25")
    ax.spines["polar"].set_linewidth(1.1)


def centroid_error_label(patient_errors: pd.DataFrame, class_id: int) -> Tuple[float, str]:
    err = patient_errors.loc[class_id, "distance_mm"]
    if not np.isfinite(err):
        err = patient_errors.loc[class_id, "distance_voxels"]
        return float(err), "vox"
    return float(err), "mm"


def plot_single_source_electrodes(
    ax: plt.Axes,
    points: Dict[int, Optional[np.ndarray]],
    apex: Optional[np.ndarray],
    base: Optional[np.ndarray],
    ant: Optional[np.ndarray],
    patient_errors: pd.DataFrame,
    source: str,
    compact: bool = False,
    annotate_errors: bool = False,
) -> None:
    legend_handles = []
    legend_labels = []
    is_prediction = source == "Prediction"

    for class_id in ELECTRODE_CLASS_IDS:
        xyz = points.get(class_id)
        if xyz is None:
            continue

        class_name = CLASS_NAMES[class_id]
        color, marker = marker_style(class_id)
        theta, radius = convert_xyz_to_bullseye(xyz, apex, base, ant)
        if not np.isfinite([theta, radius]).all():
            continue

        if is_prediction:
            handle = ax.scatter(
                np.deg2rad(theta),
                radius,
                marker=marker,
                s=135 if marker == "*" else 82,
                c=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
        else:
            handle = ax.scatter(
                np.deg2rad(theta),
                radius,
                marker=marker,
                s=135 if marker == "*" else 82,
                facecolors="none",
                edgecolors="black",
                linewidths=1.7,
                zorder=4,
            )

        if not compact:
            legend_handles.append(handle)
            legend_labels.append(class_name)

        if annotate_errors and class_id in patient_errors.index:
            err, suffix = centroid_error_label(patient_errors, class_id)
            ax.text(
                np.deg2rad(theta),
                min(radius + 0.08, 1.18),
                f"{class_name}\n{err:.1f} {suffix}",
                fontsize=8,
                ha="center",
                va="center",
            )
        elif not compact:
            ax.text(np.deg2rad(theta), min(radius + 0.06, 1.18), class_name, fontsize=8, ha="center", va="center")

    if legend_handles and not compact:
        ax.legend(legend_handles, legend_labels, loc="upper right", bbox_to_anchor=(1.22, 1.12), frameon=False, fontsize=9)


def plot_patient_bullseye(
    patient_id: str,
    coordinates_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    output_path: Path,
    compact: bool = False,
    ax: Optional[plt.Axes] = None,
) -> Path:
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure

    gt_points = patient_source_points(coordinates_df, patient_id, "GT")
    pred_points = patient_source_points(coordinates_df, patient_id, "Prediction")
    apex, base, ant = frame_landmarks(gt_points, pred_points)
    setup_polar_axis(ax, patient_id)

    patient_errors = errors_df[errors_df["patient_id"] == patient_id].set_index("class_id")
    legend_handles = []
    legend_labels = []

    for class_id in ELECTRODE_CLASS_IDS:
        class_name = CLASS_NAMES[class_id]
        color, marker = marker_style(class_id)
        gt_xyz = gt_points.get(class_id)
        pred_xyz = pred_points.get(class_id)

        gt_theta = gt_radius = pred_theta = pred_radius = np.nan
        if gt_xyz is not None:
            gt_theta, gt_radius = convert_xyz_to_bullseye(gt_xyz, apex, base, ant)
            handle = ax.scatter(
                np.deg2rad(gt_theta),
                gt_radius,
                marker=marker,
                s=105 if marker == "*" else 70,
                facecolors="none",
                edgecolors="black",
                linewidths=1.3,
                zorder=3,
            )
            if not compact and "GT" not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append("GT")

        if pred_xyz is not None:
            pred_theta, pred_radius = convert_xyz_to_bullseye(pred_xyz, apex, base, ant)
            handle = ax.scatter(
                np.deg2rad(pred_theta),
                pred_radius,
                marker=marker,
                s=125 if marker == "*" else 78,
                c=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
            if not compact and class_name not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(class_name)

        if np.isfinite([gt_theta, gt_radius, pred_theta, pred_radius]).all():
            ax.plot(
                np.deg2rad([gt_theta, pred_theta]),
                [gt_radius, pred_radius],
                color="0.25",
                linewidth=0.8,
                alpha=0.65,
                zorder=2,
            )
            if not compact and class_id in patient_errors.index:
                err, suffix = centroid_error_label(patient_errors, class_id)
                label_theta = np.deg2rad(pred_theta)
                label_radius = min(pred_radius + 0.08, 1.18)
                ax.text(label_theta, label_radius, f"{class_name}\n{err:.1f} {suffix}", fontsize=8, ha="center", va="center")
        elif pred_xyz is not None and not compact:
            ax.text(np.deg2rad(pred_theta), min(pred_radius + 0.08, 1.18), class_name, fontsize=8, ha="center")

    if not compact:
        ax.legend(legend_handles, legend_labels, loc="upper right", bbox_to_anchor=(1.28, 1.12), frameon=False, fontsize=9)
    fig.tight_layout()
    if created_fig:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    return output_path


def plot_patient_gt_vs_prediction_bullseye(
    patient_id: str,
    coordinates_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    gt_points = patient_source_points(coordinates_df, patient_id, "GT")
    pred_points = patient_source_points(coordinates_df, patient_id, "Prediction")
    apex, base, ant = frame_landmarks(gt_points, pred_points)
    patient_errors = errors_df[errors_df["patient_id"] == patient_id].set_index("class_id")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), subplot_kw={"projection": "polar"})
    setup_polar_axis(axes[0], "Ground Truth / Existing Bullseye")
    setup_polar_axis(axes[1], "Model Prediction")
    plot_single_source_electrodes(
        axes[0],
        points=gt_points,
        apex=apex,
        base=base,
        ant=ant,
        patient_errors=patient_errors,
        source="GT",
        annotate_errors=False,
    )
    plot_single_source_electrodes(
        axes[1],
        points=pred_points,
        apex=apex,
        base=base,
        ant=ant,
        patient_errors=patient_errors,
        source="Prediction",
        annotate_errors=True,
    )
    fig.suptitle(f"Lead Locations for Patient {patient_id}", fontsize=15, y=0.99)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_combined_gt_vs_prediction_summary(
    coordinates_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    output_path: Path,
    max_patients: int = 6,
) -> Path:
    patients = sorted(coordinates_df["patient_id"].dropna().unique().tolist())[:max_patients]
    if not patients:
        raise ValueError("No patients found for GT-vs-prediction bullseye summary.")

    fig, axes = plt.subplots(len(patients), 2, figsize=(8.5, 4.0 * len(patients)), subplot_kw={"projection": "polar"})
    axes = np.atleast_2d(axes)
    for row_idx, patient_id in enumerate(patients):
        gt_points = patient_source_points(coordinates_df, patient_id, "GT")
        pred_points = patient_source_points(coordinates_df, patient_id, "Prediction")
        apex, base, ant = frame_landmarks(gt_points, pred_points)
        patient_errors = errors_df[errors_df["patient_id"] == patient_id].set_index("class_id")

        setup_polar_axis(axes[row_idx, 0], f"{patient_id} GT")
        setup_polar_axis(axes[row_idx, 1], f"{patient_id} Prediction")
        plot_single_source_electrodes(
            axes[row_idx, 0],
            points=gt_points,
            apex=apex,
            base=base,
            ant=ant,
            patient_errors=patient_errors,
            source="GT",
            compact=True,
        )
        plot_single_source_electrodes(
            axes[row_idx, 1],
            points=pred_points,
            apex=apex,
            base=base,
            ant=ant,
            patient_errors=patient_errors,
            source="Prediction",
            compact=True,
        )

    fig.suptitle("Ground Truth vs Model Prediction Bullseyes", fontsize=16, y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_combined_bullseye_summary(coordinates_df: pd.DataFrame, errors_df: pd.DataFrame, output_path: Path, max_patients: int = 12) -> Path:
    patients = sorted(coordinates_df["patient_id"].dropna().unique().tolist())[:max_patients]
    if not patients:
        raise ValueError("No patients found for combined bullseye summary.")

    cols = min(4, len(patients))
    rows = int(math.ceil(len(patients) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.0 * rows), subplot_kw={"projection": "polar"})
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, patient_id in zip(axes_arr, patients):
        plot_patient_bullseye(patient_id, coordinates_df, errors_df, output_path, compact=True, ax=ax)
    for ax in axes_arr[len(patients):]:
        ax.set_visible(False)

    fig.suptitle("Validation Patient Bullseye Summary", fontsize=16, y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_bullseye_plots(
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
    force_centroids: bool = False,
) -> List[Path]:
    run_dir = run_dir.resolve()
    coordinates_path, errors_path = export_centroids(run_dir=run_dir, checkpoint_path=checkpoint_path, force=force_centroids)
    coordinates_df = pd.read_csv(coordinates_path)
    errors_df = pd.read_csv(errors_path)

    output_dir = run_dir / "bullseye_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for patient_id in sorted(coordinates_df["patient_id"].dropna().unique().tolist()):
        output_path = output_dir / f"{patient_id}_bullseye.png"
        plot_patient_bullseye(patient_id, coordinates_df, errors_df, output_path)
        saved_paths.append(output_path)
        comparison_path = output_dir / f"{patient_id}_bullseye_gt_vs_prediction.png"
        plot_patient_gt_vs_prediction_bullseye(patient_id, coordinates_df, errors_df, comparison_path)
        saved_paths.append(comparison_path)

    combined_path = output_dir / "combined_bullseye_summary.png"
    create_combined_bullseye_summary(coordinates_df, errors_df, combined_path)
    saved_paths.append(combined_path)
    comparison_summary_path = output_dir / "combined_gt_vs_prediction_bullseye_summary.png"
    create_combined_gt_vs_prediction_summary(coordinates_df, errors_df, comparison_summary_path)
    saved_paths.append(comparison_summary_path)
    print(f"Saved bullseye plots to: {output_dir}")
    return saved_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate validation bullseye plots for electrode placement.")
    parser.add_argument("--run-dir", type=str, default=None, help="Completed run directory. Defaults to latest run with a best checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path if centroid CSVs must be generated.")
    parser.add_argument("--force-centroids", action="store_true", help="Recompute centroid CSVs before plotting.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = repo_root_from_here()
    run_dir = resolve_path(args.run_dir, repo_root) if args.run_dir else find_default_run_dir(repo_root)
    checkpoint = resolve_path(args.checkpoint, repo_root) if args.checkpoint else None
    generate_bullseye_plots(run_dir=run_dir, checkpoint_path=checkpoint, force_centroids=args.force_centroids)


if __name__ == "__main__":
    main()
