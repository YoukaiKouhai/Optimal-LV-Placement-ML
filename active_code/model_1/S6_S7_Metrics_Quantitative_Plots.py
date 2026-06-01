# ==============================
# STEPS 6 + 7: METRICS + QUANTITATIVE PLOTS
# ==============================
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from S1_DataLoading_Preprocessing import ANATOMY_CLASSES, CLASS_NAMES, ELECTRODE_CLASSES, Config
from S3_ModelDefintion import infer_logits

ACCURACY_DISTANCE_THRESHOLDS_VOXELS = (5.0, 10.0, 15.0)


def _load_training_module():
    module_path = Path(__file__).with_name("S4_S5_Training_Semi-Supervised_Pseudo-Lableing.py")
    spec = importlib.util.spec_from_file_location("S4_S5_Training_Semi_Supervised_Pseudo_Lableing", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_training_module = _load_training_module()
labels_to_one_hot = _training_module.labels_to_one_hot


def nanmean_or_nan(values: object) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    return float(np.nanmean(arr)) if finite.any() else np.nan


def logits_to_label_map(logits: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = channel_peak_label_map(probs, cfg)
    return preds, probs


def channel_peak_label_map(probs: torch.Tensor, cfg: Config) -> torch.Tensor:
    preds = torch.zeros(
        (probs.shape[0], 1, *probs.shape[2:]),
        dtype=torch.long,
        device=probs.device,
    )
    radius = int(cfg.peak_prediction_blob_radius_voxels)
    spatial_shape = tuple(probs.shape[2:])

    for batch_idx in range(probs.shape[0]):
        for channel_idx in range(probs.shape[1]):
            class_id = channel_idx + 1
            flat_idx = int(torch.argmax(probs[batch_idx, channel_idx]).item())
            center = np.asarray(np.unravel_index(flat_idx, spatial_shape), dtype=np.int64)
            slices = []
            grids = []
            for axis, coord in enumerate(center):
                start = max(0, int(coord) - radius)
                stop = min(spatial_shape[axis], int(coord) + radius + 1)
                slices.append(slice(start, stop))
                grids.append(np.arange(start, stop) - int(coord))

            mesh = np.meshgrid(*grids, indexing="ij")
            ball = sum(axis_grid ** 2 for axis_grid in mesh) <= radius ** 2
            region = preds[(batch_idx, 0, *slices)]
            region[torch.as_tensor(ball, device=probs.device)] = class_id

    return preds

def per_sample_overlap_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_oh = (probs >= cfg.prediction_threshold).float()
    true_oh = torch.stack(
        [(labels.squeeze(1).long() == class_id).float() for class_id in range(1, cfg.num_classes)],
        dim=1,
    )

    reduce_dims = tuple(range(2, pred_oh.ndim))
    inter = (pred_oh * true_oh).sum(dim=reduce_dims).cpu().numpy()
    pred_sum = pred_oh.sum(dim=reduce_dims).cpu().numpy()
    true_sum = true_oh.sum(dim=reduce_dims).cpu().numpy()

    dice_denom = pred_sum + true_sum
    iou_denom = pred_sum + true_sum - inter

    dice = np.where(
        dice_denom > 0,
        (2.0 * inter + 1e-6) / (dice_denom + 1e-6),
        np.nan,
    )
    iou = np.where(
        iou_denom > 0,
        (inter + 1e-6) / (iou_denom + 1e-6),
        np.nan,
    )

    batch_size = probs.shape[0]
    dice_with_bg = np.full((batch_size, cfg.num_classes), np.nan, dtype=np.float64)
    iou_with_bg = np.full((batch_size, cfg.num_classes), np.nan, dtype=np.float64)
    dice_with_bg[:, 1:] = dice
    iou_with_bg[:, 1:] = iou
    return dice_with_bg, iou_with_bg


def centroid_distances_for_landmarks(
    preds: torch.Tensor,
    labels: torch.Tensor,
    probs: torch.Tensor,
    cfg: Config,
) -> Dict[int, float]:
    label_np = labels[0, 0].detach().cpu().numpy().astype(np.int64)
    prob_np = probs[0].detach().cpu().numpy()

    distances: Dict[int, float] = {}
    for class_id in range(1, cfg.num_classes):
        true_coords = np.argwhere(label_np == class_id)
        if true_coords.size == 0:
            distances[class_id] = np.nan
            continue

        channel_idx = class_id - 1
        if channel_idx < 0 or channel_idx >= prob_np.shape[0]:
            distances[class_id] = np.nan
            continue
        flat_idx = int(np.argmax(prob_np[channel_idx]))
        pred_centroid = np.asarray(np.unravel_index(flat_idx, prob_np[channel_idx].shape), dtype=np.float32)

        true_centroid = true_coords.mean(axis=0)
        distances[class_id] = float(np.linalg.norm(pred_centroid - true_centroid))

    return distances


class PRCurveAccumulator:
    def __init__(self, thresholds: np.ndarray):
        self.thresholds = thresholds
        self.tp = np.zeros_like(thresholds, dtype=np.int64)
        self.fp = np.zeros_like(thresholds, dtype=np.int64)
        self.fn = np.zeros_like(thresholds, dtype=np.int64)

    def update(self, y_true_binary: np.ndarray, y_score: np.ndarray) -> None:
        y_true_binary = y_true_binary.astype(bool)
        y_score = y_score.astype(np.float32)

        for i, thr in enumerate(self.thresholds):
            y_pred_binary = y_score >= thr
            self.tp[i] += int(np.logical_and(y_pred_binary, y_true_binary).sum())
            self.fp[i] += int(np.logical_and(y_pred_binary, ~y_true_binary).sum())
            self.fn[i] += int(np.logical_and(~y_pred_binary, y_true_binary).sum())

    def as_dataframe(self) -> pd.DataFrame:
        precision = self.tp / np.maximum(self.tp + self.fp, 1)
        recall = self.tp / np.maximum(self.tp + self.fn, 1)
        return pd.DataFrame(
            {
                "threshold": self.thresholds,
                "precision": precision,
                "recall": recall,
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
            }
        )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Dict[str, object]:
    model.eval()

    cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    pr_acc = PRCurveAccumulator(thresholds=np.linspace(0.0, 1.0, 101))

    per_sample_rows: List[Dict] = []

    for batch in tqdm(val_loader, desc="Final evaluation"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        case_id = batch["case_id"][0]

        logits = infer_logits(model, images, cfg)
        preds, probs = logits_to_label_map(logits, cfg)

        dice_np, iou_np = per_sample_overlap_metrics(probs, labels, cfg)
        centroid_by_class = centroid_distances_for_landmarks(preds, labels, probs, cfg)

        # Confusion matrix update
        y_true_flat = labels.squeeze(1).cpu().numpy().astype(np.int64).ravel()
        y_pred_flat = preds.squeeze(1).cpu().numpy().astype(np.int64).ravel()
        cm += confusion_matrix(y_true_flat, y_pred_flat, labels=np.arange(cfg.num_classes))

        # Precision-recall for "electrode vs not-electrode"
        # score = sum of probabilities for classes 1..6
        y_score_elec = probs[:, [class_id - 1 for class_id in ELECTRODE_CLASSES]].max(dim=1).values.cpu().numpy().ravel()
        y_true_elec = np.isin(y_true_flat, ELECTRODE_CLASSES)
        pr_acc.update(y_true_binary=y_true_elec, y_score=y_score_elec)

        # Per-sample records
        row = {"case_id": case_id}
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

    per_sample_df = pd.DataFrame(per_sample_rows)

    # Per-class summary
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

        # Precision / recall / F1 from confusion matrix
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

    # Electrode-only metrics
    electrode_df = per_class_df[per_class_df["class_id"].isin(ELECTRODE_CLASSES)]
    non_bg_centroid_values = per_sample_df[
        [f"centroid_dist_{class_name}" for class_name in CLASS_NAMES[1:]]
    ].to_numpy(dtype=np.float64)
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
                {
                    "metric": f"landmark_accuracy_within_{threshold_label}vox_non_bg",
                    "value": float(np.nanmean(non_bg_centroid_values <= threshold)),
                },
                {
                    "metric": f"landmark_accuracy_within_{threshold_label}vox_electrodes",
                    "value": float(np.nanmean(electrode_centroid_values <= threshold)),
                },
                {
                    "metric": f"landmark_accuracy_within_{threshold_label}vox_anatomy",
                    "value": float(np.nanmean(anatomy_centroid_values <= threshold)),
                },
            ]
        )
    summary_df = pd.DataFrame(
        summary_rows
    )

    confusion_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    pr_curve_df = pr_acc.as_dataframe()

    return {
        "summary_df": summary_df,
        "per_class_df": per_class_df,
        "per_sample_df": per_sample_df,
        "confusion_df": confusion_df,
        "pr_curve_df": pr_curve_df,
    }


def save_metrics_to_csv(metrics: Dict[str, object], cfg: Config) -> None:
    metrics["summary_df"].to_csv(cfg.metrics_dir / "summary_metrics.csv", index=False)
    metrics["per_class_df"].to_csv(cfg.metrics_dir / "per_class_metrics.csv", index=False)
    metrics["per_sample_df"].to_csv(cfg.metrics_dir / "per_sample_metrics.csv", index=False)
    metrics["confusion_df"].to_csv(cfg.metrics_dir / "confusion_matrix.csv")
    metrics["pr_curve_df"].to_csv(cfg.metrics_dir / "precision_recall_curve.csv", index=False)


def plot_training_history(history_df: pd.DataFrame, cfg: Config) -> None:
    # Loss plot
    plt.figure(figsize=(8, 5))
    for stage_name, stage_df in history_df.groupby("stage"):
        plt.plot(stage_df["epoch"], stage_df["train_loss"], label=f"{stage_name} train loss")
        plt.plot(stage_df["epoch"], stage_df["val_loss"], label=f"{stage_name} val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss vs epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "loss_vs_epochs.png", dpi=200)
    plt.close()

    # Val Dice plot
    plt.figure(figsize=(8, 5))
    for stage_name, stage_df in history_df.groupby("stage"):
        plt.plot(stage_df["epoch"], stage_df["val_dice"], label=f"{stage_name} val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Mean validation Dice")
    plt.title("Validation Dice score vs epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "val_dice_vs_epochs.png", dpi=200)
    plt.close()

    if "val_centroid_dist" in history_df.columns:
        plt.figure(figsize=(8, 5))
        for stage_name, stage_df in history_df.groupby("stage"):
            plt.plot(stage_df["epoch"], stage_df["val_centroid_dist"], label=f"{stage_name} centroid distance")
        plt.xlabel("Epoch")
        plt.ylabel("Mean validation centroid distance (voxels)")
        plt.title("Validation centroid distance vs epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / "val_centroid_dist_vs_epochs.png", dpi=200)
        plt.close()

    focus_cols = [
        col for col in history_df.columns
        if col.startswith("val_centroid_dist_") and col != "val_centroid_dist"
    ]
    if focus_cols:
        plt.figure(figsize=(9, 5))
        for col in focus_cols:
            plt.plot(history_df["epoch"], history_df[col], label=col.replace("val_centroid_dist_", ""))
        plt.xlabel("Epoch")
        plt.ylabel("Validation centroid distance (voxels)")
        plt.title("Focused class centroid distance vs epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / "focus_centroid_dist_vs_epochs.png", dpi=200)
        plt.close()

    if "grad_norm" in history_df.columns:
        plt.figure(figsize=(8, 5))
        for stage_name, stage_df in history_df.groupby("stage"):
            plt.plot(stage_df["epoch"], stage_df["grad_norm"], label=f"{stage_name} grad norm")
        plt.xlabel("Epoch")
        plt.ylabel("Mean gradient norm before clipping")
        plt.title("Gradient norm vs epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / "grad_norm_vs_epochs.png", dpi=200)
        plt.close()

    if "learning_rate" in history_df.columns:
        plt.figure(figsize=(8, 5))
        for stage_name, stage_df in history_df.groupby("stage"):
            plt.plot(stage_df["epoch"], stage_df["learning_rate"], label=f"{stage_name} learning rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.yscale("log")
        plt.title("Learning rate vs epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cfg.plots_dir / "learning_rate_vs_epochs.png", dpi=200)
        plt.close()


def plot_dice_boxplot(per_sample_df: pd.DataFrame, cfg: Config) -> None:
    plt.figure(figsize=(7, 5))
    plt.boxplot(per_sample_df["mean_dice_non_bg"].dropna().values, vert=True)
    plt.ylabel("Mean non-background Dice per validation sample")
    plt.title("Boxplot of Dice scores across validation samples")
    plt.xticks([1], ["Validation set"])
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "dice_boxplot.png", dpi=200)
    plt.close()


def plot_per_class_dice(per_class_df: pd.DataFrame, cfg: Config) -> None:
    plot_df = per_class_df[per_class_df["class_id"] > 0].copy()
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["class_name"], plot_df["dice_mean"])
    plt.xlabel("Class")
    plt.ylabel("Mean Dice")
    plt.title("Per-class Dice scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "per_class_dice.png", dpi=200)
    plt.close()


def plot_precision_recall_curve(pr_curve_df: pd.DataFrame, cfg: Config) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(pr_curve_df["recall"], pr_curve_df["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve for electrode detection")
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "precision_recall_curve_electrodes.png", dpi=200)
    plt.close()


def plot_confusion_matrix_heatmap(confusion_df: pd.DataFrame, cfg: Config) -> None:
    cm = confusion_df.values.astype(np.float64)

    # Row-normalized view is more readable than raw counts because background dominates.
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    cm_norm = cm / row_sums

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Pixel-wise confusion matrix heatmap")
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "confusion_matrix_heatmap.png", dpi=200)
    plt.close()
