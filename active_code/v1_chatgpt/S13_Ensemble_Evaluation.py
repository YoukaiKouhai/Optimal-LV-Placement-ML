from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from S1_DataLoading_Preprocessing import ANATOMY_CLASSES, CLASS_NAMES, ELECTRODE_CLASSES, Config
from S3_ModelDefintion import build_model, infer_logits
from S6_S7_Metrics_Quantitative_Plots import (
    ACCURACY_DISTANCE_THRESHOLDS_VOXELS,
    PRCurveAccumulator,
    channel_peak_label_map,
    nanmean_or_nan,
    per_sample_overlap_metrics,
    plot_confusion_matrix_heatmap,
    plot_dice_boxplot,
    plot_per_class_dice,
    plot_precision_recall_curve,
    save_metrics_to_csv,
)
from S10_Bullseye_Lead_Visualization import (
    create_combined_bullseye_summary,
    create_combined_gt_vs_prediction_summary,
    plot_patient_bullseye,
    plot_patient_gt_vs_prediction_bullseye,
)
from S11_Centroid_Export import (
    centroid_rows_for_mask,
    centroid_rows_for_prediction_probabilities,
    compute_error_rows,
    get_validation_npz_paths,
    load_checkpoint_weights,
    load_config_for_run,
    repo_root_from_here,
    resolve_path,
)
from S12_Presentation_Figures import (
    save_clinical_interpretation_summary,
    save_example_patient_panel,
    save_failure_case_panel,
    save_per_class_centroid_error,
    save_per_class_dice,
    save_pipeline_overview,
)


DEFAULT_ENSEMBLE = (
    "runs/cardiac_leads_apex_recovery_v3/weights/best_supervised_model.pth",
    "runs/cardiac_leads_no_spatial_aug_v6/weights/best_supervised_model.pth",
)
DEFAULT_WEIGHTS = (0.65, 0.35)


def normalize_weights(weights: Sequence[float], count: int) -> List[float]:
    if not weights:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} ensemble weights, got {len(weights)}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Ensemble weights must have positive sum.")
    return [float(w) / total for w in weights]


def load_ensemble_models(checkpoint_paths: Sequence[Path], cfg: Config, device: torch.device) -> List[torch.nn.Module]:
    models: List[torch.nn.Module] = []
    for checkpoint_path in checkpoint_paths:
        model = build_model(cfg).to(device)
        load_checkpoint_weights(model, checkpoint_path, device)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def ensemble_probabilities(
    models: Sequence[torch.nn.Module],
    weights: Sequence[float],
    image_np: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    image = torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    probs_sum: Optional[torch.Tensor] = None
    for model, weight in zip(models, weights):
        with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
            logits = infer_logits(model, image, cfg)
            probs = torch.sigmoid(logits)
        probs_sum = probs * weight if probs_sum is None else probs_sum + (probs * weight)
    if probs_sum is None:
        raise ValueError("No ensemble models were provided.")
    return probs_sum[0].detach().cpu().numpy().astype(np.float32)


def centroid_distances_for_probs(prob_np: np.ndarray, label_np: np.ndarray, cfg: Config) -> Dict[int, float]:
    distances: Dict[int, float] = {}
    for class_id in range(1, cfg.num_classes):
        true_coords = np.argwhere(label_np == class_id)
        if true_coords.size == 0:
            distances[class_id] = np.nan
            continue
        channel = prob_np[class_id - 1]
        pred_centroid = np.asarray(np.unravel_index(int(np.argmax(channel)), channel.shape), dtype=np.float32)
        distances[class_id] = float(np.linalg.norm(pred_centroid - true_coords.mean(axis=0)))
    return distances


def metrics_from_rows(per_sample_rows: List[Dict[str, object]], cm: np.ndarray, pr_acc: PRCurveAccumulator, cfg: Config) -> Dict[str, object]:
    per_sample_df = pd.DataFrame(per_sample_rows)
    per_class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_row = {
            "class_id": class_id,
            "class_name": class_name,
            "dice_mean": nanmean_or_nan(per_sample_df[f"dice_{class_name}"]),
            "iou_mean": nanmean_or_nan(per_sample_df[f"iou_{class_name}"]),
        }
        if class_id >= 1:
            centroid_col = per_sample_df[f"centroid_dist_{class_name}"]
            class_row["centroid_dist_mean"] = nanmean_or_nan(centroid_col)
            for threshold in ACCURACY_DISTANCE_THRESHOLDS_VOXELS:
                class_row[f"accuracy_within_{int(threshold)}vox"] = float(
                    np.nanmean(np.asarray(centroid_col, dtype=np.float64) <= threshold)
                )
        else:
            class_row["centroid_dist_mean"] = np.nan
            for threshold in ACCURACY_DISTANCE_THRESHOLDS_VOXELS:
                class_row[f"accuracy_within_{int(threshold)}vox"] = np.nan

        tp = cm[class_id, class_id]
        fp = cm[:, class_id].sum() - tp
        fn = cm[class_id, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
        class_row["precision"] = float(precision)
        class_row["recall"] = float(recall)
        class_row["f1"] = float(f1)
        per_class_rows.append(class_row)

    per_class_df = pd.DataFrame(per_class_rows)
    electrode_df = per_class_df[per_class_df["class_id"].isin(ELECTRODE_CLASSES)]
    non_bg_centroid_values = per_sample_df[[f"centroid_dist_{name}" for name in CLASS_NAMES[1:]]].to_numpy(dtype=np.float64)
    electrode_centroid_values = per_sample_df[
        [f"centroid_dist_{CLASS_NAMES[class_id]}" for class_id in ELECTRODE_CLASSES]
    ].to_numpy(dtype=np.float64)
    anatomy_centroid_values = per_sample_df[
        [f"centroid_dist_{CLASS_NAMES[class_id]}" for class_id in ANATOMY_CLASSES]
    ].to_numpy(dtype=np.float64)

    summary_rows = [
        {"metric": "mean_dice_non_bg", "value": float(per_sample_df["mean_dice_non_bg"].mean())},
        {"metric": "mean_iou_non_bg", "value": float(per_sample_df["mean_iou_non_bg"].mean())},
        {"metric": "mean_dice_electrodes", "value": float(per_sample_df["mean_dice_electrodes"].mean())},
        {"metric": "mean_dice_anatomy", "value": float(per_sample_df["mean_dice_anatomy"].mean())},
        {"metric": "mean_centroid_dist_non_bg", "value": float(per_sample_df["mean_centroid_dist_non_bg"].mean())},
        {"metric": "mean_centroid_dist_electrodes", "value": float(per_sample_df["mean_centroid_dist_electrodes"].mean())},
        {"metric": "mean_centroid_dist_anatomy", "value": float(per_sample_df["mean_centroid_dist_anatomy"].mean())},
        {"metric": "electrode_macro_precision", "value": float(electrode_df["precision"].mean())},
        {"metric": "electrode_macro_recall", "value": float(electrode_df["recall"].mean())},
        {"metric": "electrode_macro_f1", "value": float(electrode_df["f1"].mean())},
    ]
    for threshold in ACCURACY_DISTANCE_THRESHOLDS_VOXELS:
        threshold_label = int(threshold)
        summary_rows.extend(
            [
                {"metric": f"landmark_accuracy_within_{threshold_label}vox_non_bg", "value": float(np.nanmean(non_bg_centroid_values <= threshold))},
                {"metric": f"landmark_accuracy_within_{threshold_label}vox_electrodes", "value": float(np.nanmean(electrode_centroid_values <= threshold))},
                {"metric": f"landmark_accuracy_within_{threshold_label}vox_anatomy", "value": float(np.nanmean(anatomy_centroid_values <= threshold))},
            ]
        )

    return {
        "summary_df": pd.DataFrame(summary_rows),
        "per_class_df": per_class_df,
        "per_sample_df": per_sample_df,
        "confusion_df": pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES),
        "pr_curve_df": pr_acc.as_dataframe(),
    }


def save_overlay(image_np: np.ndarray, label_np: np.ndarray, pred_np: np.ndarray, case_id: str, cfg: Config, saved_idx: int) -> None:
    cmap = plt.get_cmap("tab10", cfg.num_classes)
    union_fg = ((label_np > 0) | (pred_np > 0)).sum(axis=(1, 2))
    slice_idx = int(np.argmax(union_fg)) if np.max(union_fg) > 0 else image_np.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(image_np[slice_idx], cmap="gray")
    axes[0].set_title(f"{case_id} | Input slice {slice_idx}")
    axes[1].imshow(image_np[slice_idx], cmap="gray")
    axes[1].imshow(np.ma.masked_where(label_np[slice_idx] == 0, label_np[slice_idx]), cmap=cmap, alpha=0.40, vmin=0, vmax=cfg.num_classes - 1)
    axes[1].set_title("Ground truth overlay")
    axes[2].imshow(image_np[slice_idx], cmap="gray")
    axes[2].imshow(np.ma.masked_where(pred_np[slice_idx] == 0, pred_np[slice_idx]), cmap=cmap, alpha=0.40, vmin=0, vmax=cfg.num_classes - 1)
    axes[2].set_title("Ensemble prediction overlay")
    out_path = cfg.overlays_dir / f"{case_id}_overlay.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ensemble overlay {saved_idx}: {out_path}")


def evaluate_ensemble(
    base_run_dir: Path,
    output_run_dir: Path,
    checkpoint_paths: Sequence[Path],
    weights: Sequence[float],
) -> None:
    cfg = load_config_for_run(base_run_dir)
    cfg.work_dir = str(output_run_dir)
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    cfg.overlays_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "bullseye_plots").mkdir(parents=True, exist_ok=True)
    (output_run_dir / "presentation_figures").mkdir(parents=True, exist_ok=True)

    with (output_run_dir / "config.json").open("w", encoding="utf-8") as f:
        payload = asdict(cfg)
        payload["ensemble_checkpoints"] = [str(path) for path in checkpoint_paths]
        payload["ensemble_weights"] = list(weights)
        json.dump(payload, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble_models(checkpoint_paths, cfg, device)
    val_files = get_validation_npz_paths(base_run_dir, cfg)
    cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    pr_acc = PRCurveAccumulator(thresholds=np.linspace(0.0, 1.0, 101))
    per_sample_rows: List[Dict[str, object]] = []
    coordinate_rows: List[Dict[str, object]] = []

    for case_idx, npz_path in enumerate(tqdm(val_files, desc="Ensemble evaluation"), start=1):
        with np.load(npz_path, allow_pickle=False) as data:
            image_np = data["image"].astype(np.float32)
            label_np = data["label"].astype(np.int64)
            patient_id = str(data["case_id"]) if "case_id" in data.files else npz_path.stem
            spacing_dhw = data["spacing_dhw"].astype(np.float32) if "spacing_dhw" in data.files else np.array([np.nan, np.nan, np.nan])

        prob_np = ensemble_probabilities(models, weights, image_np, cfg, device)
        probs = torch.from_numpy(prob_np).unsqueeze(0)
        labels = torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0).long()
        preds = channel_peak_label_map(probs, cfg)
        pred_np = preds[0, 0].numpy().astype(np.int64)

        dice_np, iou_np = per_sample_overlap_metrics(probs, labels, cfg)
        centroid_by_class = centroid_distances_for_probs(prob_np, label_np, cfg)

        y_true_flat = label_np.ravel()
        y_pred_flat = pred_np.ravel()
        cm += confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(cfg.num_classes))
        y_score_elec = prob_np[[class_id - 1 for class_id in ELECTRODE_CLASSES]].max(axis=0).ravel()
        y_true_elec = np.isin(y_true_flat, ELECTRODE_CLASSES)
        pr_acc.update(y_true_binary=y_true_elec, y_score=y_score_elec)

        row = {"case_id": patient_id}
        for class_id, class_name in enumerate(CLASS_NAMES):
            row[f"dice_{class_name}"] = float(dice_np[0, class_id]) if not np.isnan(dice_np[0, class_id]) else np.nan
            row[f"iou_{class_name}"] = float(iou_np[0, class_id]) if not np.isnan(iou_np[0, class_id]) else np.nan
            if class_id >= 1:
                row[f"centroid_dist_{class_name}"] = centroid_by_class.get(class_id, np.nan)
        row["mean_dice_non_bg"] = nanmean_or_nan(dice_np[0, 1:])
        row["mean_iou_non_bg"] = nanmean_or_nan(iou_np[0, 1:])
        row["mean_dice_electrodes"] = nanmean_or_nan(dice_np[0, ELECTRODE_CLASSES])
        row["mean_dice_anatomy"] = nanmean_or_nan(dice_np[0, ANATOMY_CLASSES])
        row["mean_centroid_dist_non_bg"] = nanmean_or_nan(list(centroid_by_class.values()))
        row["mean_centroid_dist_electrodes"] = nanmean_or_nan([centroid_by_class.get(class_id, np.nan) for class_id in ELECTRODE_CLASSES])
        row["mean_centroid_dist_anatomy"] = nanmean_or_nan([centroid_by_class.get(class_id, np.nan) for class_id in ANATOMY_CLASSES])
        per_sample_rows.append(row)

        coordinate_rows.extend(
            centroid_rows_for_mask(label_np, patient_id, "GT", spacing_dhw, npz_path, cfg.num_classes)
        )
        coordinate_rows.extend(
            centroid_rows_for_prediction_probabilities(prob_np, patient_id, spacing_dhw, npz_path, cfg)
        )
        if case_idx <= cfg.max_overlay_cases:
            save_overlay(image_np, label_np, pred_np, patient_id, cfg, case_idx)

    metrics = metrics_from_rows(per_sample_rows, cm, pr_acc, cfg)
    save_metrics_to_csv(metrics, cfg)
    plot_dice_boxplot(metrics["per_sample_df"], cfg)
    plot_per_class_dice(metrics["per_class_df"], cfg)
    plot_precision_recall_curve(metrics["pr_curve_df"], cfg)
    plot_confusion_matrix_heatmap(metrics["confusion_df"], cfg)

    coordinates_df = pd.DataFrame(coordinate_rows)
    errors_df = compute_error_rows(coordinates_df)
    coordinates_path = cfg.metrics_dir / "centroid_coordinates.csv"
    errors_path = cfg.metrics_dir / "centroid_errors.csv"
    coordinates_df.to_csv(coordinates_path, index=False)
    errors_df.to_csv(errors_path, index=False)

    bullseye_dir = output_run_dir / "bullseye_plots"
    for patient_id in sorted(coordinates_df["patient_id"].dropna().unique().tolist()):
        plot_patient_bullseye(patient_id, coordinates_df, errors_df, bullseye_dir / f"{patient_id}_bullseye.png")
        plot_patient_gt_vs_prediction_bullseye(
            patient_id,
            coordinates_df,
            errors_df,
            bullseye_dir / f"{patient_id}_bullseye_gt_vs_prediction.png",
        )
    create_combined_bullseye_summary(coordinates_df, errors_df, bullseye_dir / "combined_bullseye_summary.png")
    create_combined_gt_vs_prediction_summary(coordinates_df, errors_df, bullseye_dir / "combined_gt_vs_prediction_bullseye_summary.png")

    presentation_dir = output_run_dir / "presentation_figures"
    save_pipeline_overview(presentation_dir)
    save_per_class_centroid_error(errors_df, presentation_dir)
    save_per_class_dice(output_run_dir, presentation_dir)
    save_example_patient_panel(output_run_dir, coordinates_df, presentation_dir)
    save_failure_case_panel(output_run_dir, errors_df, presentation_dir)
    save_clinical_interpretation_summary(errors_df, presentation_dir)

    print(f"Saved ensemble metrics: {cfg.metrics_dir}")
    print(f"Saved ensemble plots: {cfg.plots_dir}")
    print(f"Saved ensemble bullseyes: {bullseye_dir}")
    print(f"Saved ensemble presentation figures: {presentation_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a weighted checkpoint ensemble on the validation split.")
    parser.add_argument("--base-run-dir", type=str, default="runs/cardiac_leads_no_spatial_aug_v6")
    parser.add_argument("--output-run-dir", type=str, default="runs/cardiac_leads_ensemble_v3_v6")
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_ENSEMBLE))
    parser.add_argument("--weights", nargs="*", type=float, default=list(DEFAULT_WEIGHTS))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = repo_root_from_here()
    base_run_dir = resolve_path(args.base_run_dir, repo_root)
    output_run_dir = resolve_path(args.output_run_dir, repo_root)
    checkpoint_paths = [resolve_path(path, repo_root) for path in args.checkpoints]
    if base_run_dir is None or output_run_dir is None or any(path is None for path in checkpoint_paths):
        raise ValueError("Could not resolve run/checkpoint paths.")
    weights = normalize_weights(args.weights, len(checkpoint_paths))
    evaluate_ensemble(base_run_dir, output_run_dir, checkpoint_paths, weights)


if __name__ == "__main__":
    main()
