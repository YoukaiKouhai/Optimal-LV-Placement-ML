# ==============================
# STEP 15: GAUSSIAN HEATMAP LANDMARK REGRESSION
# ==============================
from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from S1_DataLoading_Preprocessing import (
    ANATOMY_CLASSES,
    CLASS_NAMES,
    ELECTRODE_CLASSES,
    Config,
    build_preprocessed_cache,
    ensure_dirs,
    seed_everything,
)
from S2_DatasetPreparation_Augmentation import create_train_val_split, create_val_loader
from S3_ModelDefintion import build_model, infer_logits


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def class_centers_from_labels(labels: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert integer landmark labels to per-class centers.

    Returns:
        centers: [B, 9, 3] in z/y/x voxel coordinates
        present: [B, 9] true when that landmark exists in the tensor
    """
    squeezed = labels.squeeze(1).long()
    batch_size = squeezed.shape[0]
    centers = torch.full((batch_size, cfg.num_landmark_classes, 3), torch.nan, device=labels.device)
    present = torch.zeros((batch_size, cfg.num_landmark_classes), dtype=torch.bool, device=labels.device)

    for batch_idx in range(batch_size):
        label_volume = squeezed[batch_idx]
        for class_id in range(1, cfg.num_classes):
            coords = torch.nonzero(label_volume == class_id, as_tuple=False)
            if coords.numel() == 0:
                continue
            channel_idx = class_id - 1
            centers[batch_idx, channel_idx] = coords.float().mean(dim=0)
            present[batch_idx, channel_idx] = True
    return centers, present


def gaussian_heatmaps_from_labels(labels: torch.Tensor, cfg: Config, sigma_voxels: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build Gaussian targets with one channel per non-background landmark.

    Labels may be sparse single voxels or small regions; the center is computed
    from the class centroid, then a normalized 3D Gaussian is drawn at that point.
    """
    centers, present = class_centers_from_labels(labels, cfg)
    spatial_shape = tuple(int(v) for v in labels.shape[2:])
    z_grid, y_grid, x_grid = torch.meshgrid(
        torch.arange(spatial_shape[0], device=labels.device, dtype=torch.float32),
        torch.arange(spatial_shape[1], device=labels.device, dtype=torch.float32),
        torch.arange(spatial_shape[2], device=labels.device, dtype=torch.float32),
        indexing="ij",
    )
    heatmaps = torch.zeros(
        (labels.shape[0], cfg.num_landmark_classes, *spatial_shape),
        device=labels.device,
        dtype=torch.float32,
    )
    sigma_sq = max(float(sigma_voxels) ** 2, 1e-6)

    for batch_idx in range(labels.shape[0]):
        for channel_idx in range(cfg.num_landmark_classes):
            if not bool(present[batch_idx, channel_idx]):
                continue
            zc, yc, xc = centers[batch_idx, channel_idx]
            dist_sq = (z_grid - zc) ** 2 + (y_grid - yc) ** 2 + (x_grid - xc) ** 2
            heatmaps[batch_idx, channel_idx] = torch.exp(-0.5 * dist_sq / sigma_sq)

    return heatmaps, centers, present


def foreground_weighted_heatmap_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
    foreground_alpha: float,
    active_channel_indices: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Foreground-weighted MSE/SmoothL1 heatmap loss.

    A plain mean over the full 3D volume is dominated by zero-valued voxels and
    can improve while the predicted argmax stays wrong. The target Gaussian is
    used as a soft foreground weight so voxels near the true peak matter more.
    """
    if loss_name == "smoothl1":
        loss_map = F.smooth_l1_loss(predictions, targets, reduction="none")
    elif loss_name == "mse":
        loss_map = F.mse_loss(predictions, targets, reduction="none")
    else:
        raise ValueError(f"Unsupported heatmap loss: {loss_name}")

    reduce_dims = tuple(range(2, predictions.ndim))
    weights = 1.0 + (float(foreground_alpha) * targets)
    weighted_loss_by_channel = (loss_map * weights).mean(dim=reduce_dims)
    unweighted_mse_by_channel = F.mse_loss(predictions, targets, reduction="none").mean(dim=reduce_dims)

    if active_channel_indices is not None:
        channel_idx = torch.as_tensor(active_channel_indices, device=predictions.device, dtype=torch.long)
        weighted_mean = weighted_loss_by_channel.index_select(1, channel_idx).mean()
        unweighted_mean = unweighted_mse_by_channel.index_select(1, channel_idx).mean()
    else:
        weighted_mean = weighted_loss_by_channel.mean()
        unweighted_mean = unweighted_mse_by_channel.mean()
    return weighted_mean, weighted_loss_by_channel, unweighted_mean


def softargmax_centers_from_outputs(outputs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Differentiably convert each channel heatmap to an expected z/y/x coordinate.
    Lower temperatures make the softmax behave more like an argmax.
    """
    batch_size, channels = outputs.shape[:2]
    spatial_shape = tuple(int(v) for v in outputs.shape[2:])
    flat = outputs.reshape(batch_size, channels, -1)
    weights = torch.softmax(flat / max(float(temperature), 1e-6), dim=-1)
    z_grid, y_grid, x_grid = torch.meshgrid(
        torch.arange(spatial_shape[0], device=outputs.device, dtype=torch.float32),
        torch.arange(spatial_shape[1], device=outputs.device, dtype=torch.float32),
        torch.arange(spatial_shape[2], device=outputs.device, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack([z_grid.reshape(-1), y_grid.reshape(-1), x_grid.reshape(-1)], dim=1)
    return torch.matmul(weights, coords)


def coordinate_loss_from_softargmax(
    pred_centers: torch.Tensor,
    target_centers: torch.Tensor,
    present: torch.Tensor,
    active_channel_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    valid = present.clone()
    if active_channel_indices is not None:
        active = torch.zeros(valid.shape[1], dtype=torch.bool, device=valid.device)
        active[torch.as_tensor(active_channel_indices, device=valid.device, dtype=torch.long)] = True
        valid = valid & active.unsqueeze(0)
    if not bool(valid.any()):
        return pred_centers.sum() * 0.0
    return F.smooth_l1_loss(pred_centers[valid], target_centers[valid], reduction="mean")


def heatmap_objective(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    target_centers: torch.Tensor,
    present: torch.Tensor,
    loss_name: str,
    foreground_alpha: float,
    lambda_coord: float,
    softargmax_temperature: float,
    active_channel_indices: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    heatmap_loss, loss_by_channel, unweighted_mse = foreground_weighted_heatmap_loss(
        predictions=outputs,
        targets=targets,
        loss_name=loss_name,
        foreground_alpha=foreground_alpha,
        active_channel_indices=active_channel_indices,
    )
    soft_centers = softargmax_centers_from_outputs(outputs, temperature=softargmax_temperature)
    coord_loss = coordinate_loss_from_softargmax(
        pred_centers=soft_centers,
        target_centers=target_centers,
        present=present,
        active_channel_indices=active_channel_indices,
    )
    total_loss = heatmap_loss + (float(lambda_coord) * coord_loss)
    components = {
        "total_loss": total_loss.detach(),
        "weighted_heatmap_loss": heatmap_loss.detach(),
        "unweighted_mse": unweighted_mse.detach(),
        "coord_loss": coord_loss.detach(),
    }
    return total_loss, components, loss_by_channel, soft_centers


def apply_output_activation(logits: torch.Tensor, output_activation: str) -> torch.Tensor:
    if output_activation == "raw":
        return logits
    if output_activation == "sigmoid":
        return torch.sigmoid(logits)
    raise ValueError(f"Unsupported output activation: {output_activation}")


def argmax_centers_from_outputs(outputs: torch.Tensor, cfg: Config) -> torch.Tensor:
    batch_size, channels = outputs.shape[:2]
    spatial_shape = tuple(int(v) for v in outputs.shape[2:])
    centers = torch.zeros((batch_size, channels, 3), device=outputs.device, dtype=torch.float32)
    for batch_idx in range(batch_size):
        for channel_idx in range(channels):
            flat_idx = int(torch.argmax(outputs[batch_idx, channel_idx]).item())
            centers[batch_idx, channel_idx] = torch.tensor(
                np.unravel_index(flat_idx, spatial_shape),
                device=outputs.device,
                dtype=torch.float32,
            )
    return centers


def centroid_distances(pred_centers: torch.Tensor, target_centers: torch.Tensor, present: torch.Tensor) -> Dict[int, List[float]]:
    distances: Dict[int, List[float]] = {class_id: [] for class_id in range(1, len(CLASS_NAMES))}
    for batch_idx in range(pred_centers.shape[0]):
        for channel_idx in range(pred_centers.shape[1]):
            if not bool(present[batch_idx, channel_idx]):
                continue
            dist = torch.linalg.vector_norm(pred_centers[batch_idx, channel_idx] - target_centers[batch_idx, channel_idx])
            distances[channel_idx + 1].append(float(dist.item()))
    return distances


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def finite_threshold_mean(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite <= threshold)) if finite.size else np.nan


def centers_from_label_np(label: np.ndarray, cfg: Config) -> Dict[int, np.ndarray]:
    centers: Dict[int, np.ndarray] = {}
    for class_id in range(1, cfg.num_classes):
        coords = np.argwhere(label == class_id)
        if coords.size == 0:
            continue
        centers[class_id] = coords.astype(np.float32).mean(axis=0)
    return centers


def crop_or_pad_3d(volume: np.ndarray, center_zyx: Sequence[float], patch_size: Sequence[int], pad_value: float) -> np.ndarray:
    patch_size = np.asarray(patch_size, dtype=np.int64)
    center = np.rint(np.asarray(center_zyx, dtype=np.float32)).astype(np.int64)
    start = center - (patch_size // 2)
    stop = start + patch_size

    src_slices = []
    dst_slices = []
    for axis, dim in enumerate(volume.shape):
        src_start = max(0, int(start[axis]))
        src_stop = min(dim, int(stop[axis]))
        dst_start = src_start - int(start[axis])
        dst_stop = dst_start + (src_stop - src_start)
        src_slices.append(slice(src_start, src_stop))
        dst_slices.append(slice(dst_start, dst_stop))

    out = np.full(tuple(int(v) for v in patch_size), pad_value, dtype=volume.dtype)
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


class LandmarkCenteredPatchDataset(Dataset):
    """
    Heatmap-specific training dataset.

    Each sample is a patch centered on one present landmark. This avoids the
    old hard-mask random crop behavior where many epochs may barely show a
    difficult class such as Base or ANT.
    """
    def __init__(
        self,
        npz_paths: Sequence[Path],
        cfg: Config,
        samples_per_landmark: int = 1,
        jitter_voxels: int = 8,
        class_filter: Optional[Sequence[int]] = None,
        debug_crops: bool = False,
    ) -> None:
        self.npz_paths = [Path(p) for p in npz_paths]
        self.cfg = cfg
        self.samples_per_landmark = max(1, int(samples_per_landmark))
        self.jitter_voxels = max(0, int(jitter_voxels))
        self.class_filter = set(int(v) for v in class_filter) if class_filter is not None else None
        self.debug_crops = debug_crops
        self._debug_prints = 0
        self.entries: List[Tuple[Path, int, np.ndarray]] = []

        for path in self.npz_paths:
            with np.load(path, allow_pickle=False) as data:
                label = data["label"].astype(np.int64)
            for class_id, center in centers_from_label_np(label, cfg).items():
                if self.class_filter is not None and class_id not in self.class_filter:
                    continue
                for _ in range(self.samples_per_landmark):
                    self.entries.append((path, class_id, center))
        if not self.entries:
            raise RuntimeError("No landmark-centered training patches could be created.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path, class_id, center = self.entries[index]
        with np.load(path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32)
            label = data["label"].astype(np.int64)
            case_id = str(data["case_id"]) if "case_id" in data.files else path.stem

        if self.jitter_voxels > 0:
            jitter = np.random.randint(-self.jitter_voxels, self.jitter_voxels + 1, size=3)
            crop_center = center + jitter
        else:
            crop_center = center

        patch_size = np.asarray(self.cfg.patch_size_3d, dtype=np.int64)
        requested_origin = np.rint(crop_center).astype(np.int64) - (patch_size // 2)
        center_in_crop = center - requested_origin.astype(np.float32)
        image_patch = crop_or_pad_3d(image, crop_center, self.cfg.patch_size_3d, pad_value=0.0)
        label_patch = crop_or_pad_3d(label, crop_center, self.cfg.patch_size_3d, pad_value=0)
        contains_center_class = bool(np.any(label_patch == class_id))
        if self.debug_crops and self._debug_prints < 12:
            print(
                "[heatmap crop] "
                f"case={case_id} class={class_id}:{CLASS_NAMES[class_id]} "
                f"origin_zyx={requested_origin.tolist()} "
                f"landmark_full_zyx={[round(float(v), 2) for v in center]} "
                f"landmark_crop_zyx={[round(float(v), 2) for v in center_in_crop]} "
                f"contains_class={contains_center_class}"
            )
            self._debug_prints += 1
        if not contains_center_class:
            print(
                "WARNING: landmark-centered crop does not contain its target class. "
                f"case={case_id} class={class_id}:{CLASS_NAMES[class_id]}"
            )
        return {
            "image": torch.from_numpy(image_patch[None].astype(np.float32)),
            "label": torch.from_numpy(label_patch[None].astype(np.int64)),
            "case_id": case_id,
            "npz_path": str(path),
            "center_class_id": int(class_id),
            "crop_origin_zyx": torch.as_tensor(requested_origin, dtype=torch.int64),
            "landmark_full_zyx": torch.as_tensor(center.astype(np.float32), dtype=torch.float32),
            "landmark_crop_zyx": torch.as_tensor(center_in_crop.astype(np.float32), dtype=torch.float32),
        }


def train_one_epoch_heatmap(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    sigma_voxels: float,
    loss_name: str,
    foreground_alpha: float,
    lambda_coord: float,
    softargmax_temperature: float,
    output_activation: str,
    active_channel_indices: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))
    losses: List[float] = []
    weighted_heatmap_losses: List[float] = []
    unweighted_mses: List[float] = []
    coord_losses: List[float] = []
    grad_norms: List[float] = []

    for batch in tqdm(loader, desc="Heatmap train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        targets, target_centers, present = gaussian_heatmaps_from_labels(labels, cfg, sigma_voxels=sigma_voxels)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
            logits = model(images)
            outputs = apply_output_activation(logits, output_activation)
            loss, components, _, _ = heatmap_objective(
                outputs=outputs,
                targets=targets,
                target_centers=target_centers,
                present=present,
                loss_name=loss_name,
                foreground_alpha=foreground_alpha,
                lambda_coord=lambda_coord,
                softargmax_temperature=softargmax_temperature,
                active_channel_indices=active_channel_indices,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))
        weighted_heatmap_losses.append(float(components["weighted_heatmap_loss"].item()))
        unweighted_mses.append(float(components["unweighted_mse"].item()))
        coord_losses.append(float(components["coord_loss"].item()))
        grad_norms.append(float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm))

    return {
        "train_loss": float(np.mean(losses)) if losses else np.nan,
        "train_weighted_heatmap_loss": float(np.mean(weighted_heatmap_losses)) if weighted_heatmap_losses else np.nan,
        "train_unweighted_mse": float(np.mean(unweighted_mses)) if unweighted_mses else np.nan,
        "train_coord_loss": float(np.mean(coord_losses)) if coord_losses else np.nan,
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else np.nan,
    }


@torch.no_grad()
def validate_heatmap(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Config,
    sigma_voxels: float,
    loss_name: str,
    foreground_alpha: float,
    lambda_coord: float,
    softargmax_temperature: float,
    output_activation: str,
    active_channel_indices: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    model.eval()
    losses: List[float] = []
    weighted_heatmap_losses: List[float] = []
    unweighted_mses: List[float] = []
    coord_losses: List[float] = []
    rows: List[Dict[str, object]] = []
    per_class_distances: Dict[int, List[float]] = {class_id: [] for class_id in range(1, cfg.num_classes)}

    for batch in tqdm(loader, desc="Heatmap validate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        case_id = str(batch["case_id"][0])

        logits = infer_logits(model, images, cfg)
        outputs = apply_output_activation(logits, output_activation)
        targets, target_centers, present = gaussian_heatmaps_from_labels(labels, cfg, sigma_voxels=sigma_voxels)
        loss, components, loss_by_channel, soft_centers = heatmap_objective(
            outputs=outputs,
            targets=targets,
            target_centers=target_centers,
            present=present,
            loss_name=loss_name,
            foreground_alpha=foreground_alpha,
            lambda_coord=lambda_coord,
            softargmax_temperature=softargmax_temperature,
            active_channel_indices=active_channel_indices,
        )
        losses.append(float(loss.item()))
        weighted_heatmap_losses.append(float(components["weighted_heatmap_loss"].item()))
        unweighted_mses.append(float(components["unweighted_mse"].item()))
        coord_losses.append(float(components["coord_loss"].item()))

        pred_centers = argmax_centers_from_outputs(outputs, cfg)
        target_argmax_centers = argmax_centers_from_outputs(targets, cfg)
        present_for_metrics = present.clone()
        if active_channel_indices is not None:
            active = torch.zeros(present.shape[1], dtype=torch.bool, device=present.device)
            active[torch.as_tensor(active_channel_indices, device=present.device, dtype=torch.long)] = True
            present_for_metrics = present_for_metrics & active.unsqueeze(0)
        distances = centroid_distances(pred_centers, target_argmax_centers, present_for_metrics)
        for class_id, values in distances.items():
            per_class_distances[class_id].extend(values)

        row: Dict[str, object] = {
            "case_id": case_id,
            "val_loss": float(loss.item()),
            "val_weighted_heatmap_loss": float(components["weighted_heatmap_loss"].item()),
            "val_unweighted_mse": float(components["unweighted_mse"].item()),
            "val_coord_loss": float(components["coord_loss"].item()),
        }
        for class_id in range(1, cfg.num_classes):
            class_name = CLASS_NAMES[class_id]
            channel_idx = class_id - 1
            if bool(present_for_metrics[0, channel_idx]):
                pred = pred_centers[0, channel_idx].detach().cpu().numpy()
                true = target_argmax_centers[0, channel_idx].detach().cpu().numpy()
                true_center = target_centers[0, channel_idx].detach().cpu().numpy()
                soft = soft_centers[0, channel_idx].detach().cpu().numpy()
                dist = float(np.linalg.norm(pred - true))
                row[f"pred_z_{class_name}"] = float(pred[0])
                row[f"pred_y_{class_name}"] = float(pred[1])
                row[f"pred_x_{class_name}"] = float(pred[2])
                row[f"target_argmax_z_{class_name}"] = float(true[0])
                row[f"target_argmax_y_{class_name}"] = float(true[1])
                row[f"target_argmax_x_{class_name}"] = float(true[2])
                row[f"target_center_z_{class_name}"] = float(true_center[0])
                row[f"target_center_y_{class_name}"] = float(true_center[1])
                row[f"target_center_x_{class_name}"] = float(true_center[2])
                row[f"softargmax_z_{class_name}"] = float(soft[0])
                row[f"softargmax_y_{class_name}"] = float(soft[1])
                row[f"softargmax_x_{class_name}"] = float(soft[2])
                row[f"centroid_dist_{class_name}"] = dist
                row[f"target_heatmap_max_{class_name}"] = float(targets[0, channel_idx].max().item())
                row[f"pred_heatmap_max_{class_name}"] = float(outputs[0, channel_idx].max().item())
                row[f"pred_peak_in_neighborhood_{class_name}"] = bool(dist <= max(2.0, float(sigma_voxels)))
                row[f"loss_{class_name}"] = float(loss_by_channel[0, channel_idx].item())
            else:
                row[f"centroid_dist_{class_name}"] = np.nan
                row[f"loss_{class_name}"] = float(loss_by_channel[0, channel_idx].item())
        rows.append(row)

    per_sample_df = pd.DataFrame(rows)
    per_class_rows = []
    for class_id in range(1, cfg.num_classes):
        values = np.asarray(per_class_distances[class_id], dtype=np.float64)
        class_name = CLASS_NAMES[class_id]
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "centroid_dist_mean": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                "accuracy_within_5vox": float(np.nanmean(values <= 5.0)) if values.size else np.nan,
                "accuracy_within_10vox": float(np.nanmean(values <= 10.0)) if values.size else np.nan,
                "accuracy_within_15vox": float(np.nanmean(values <= 15.0)) if values.size else np.nan,
            }
        )
    per_class_df = pd.DataFrame(per_class_rows)

    non_bg_cols = [f"centroid_dist_{name}" for name in CLASS_NAMES[1:]]
    electrode_cols = [f"centroid_dist_{CLASS_NAMES[class_id]}" for class_id in ELECTRODE_CLASSES]
    anatomy_cols = [f"centroid_dist_{CLASS_NAMES[class_id]}" for class_id in ANATOMY_CLASSES]
    non_bg = per_sample_df[non_bg_cols].to_numpy(dtype=np.float64)
    electrode = per_sample_df[electrode_cols].to_numpy(dtype=np.float64)
    anatomy = per_sample_df[anatomy_cols].to_numpy(dtype=np.float64)
    stats = {
        "val_loss": float(np.mean(losses)) if losses else np.nan,
        "val_weighted_heatmap_loss": float(np.mean(weighted_heatmap_losses)) if weighted_heatmap_losses else np.nan,
        "val_unweighted_mse": float(np.mean(unweighted_mses)) if unweighted_mses else np.nan,
        "val_coord_loss": float(np.mean(coord_losses)) if coord_losses else np.nan,
        "mean_centroid_dist_non_bg": finite_mean(non_bg),
        "mean_centroid_dist_electrodes": finite_mean(electrode),
        "mean_centroid_dist_anatomy": finite_mean(anatomy),
        "landmark_accuracy_within_5vox_non_bg": finite_threshold_mean(non_bg, 5.0),
        "landmark_accuracy_within_10vox_non_bg": finite_threshold_mean(non_bg, 10.0),
        "landmark_accuracy_within_15vox_non_bg": finite_threshold_mean(non_bg, 15.0),
        "landmark_accuracy_within_10vox_electrodes": finite_threshold_mean(electrode, 10.0),
        "landmark_accuracy_within_15vox_electrodes": finite_threshold_mean(electrode, 15.0),
    }
    return stats, per_sample_df, per_class_df


def save_heatmap_peak_overlays(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Config,
    sigma_voxels: float,
    output_activation: str,
    max_cases: int = 3,
) -> List[Path]:
    out_dir = Path(cfg.work_dir) / "heatmap_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    model.eval()

    with torch.no_grad():
        for case_idx, batch in enumerate(loader):
            if case_idx >= max_cases:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).long()
            case_id = str(batch["case_id"][0])
            logits = infer_logits(model, images, cfg)
            outputs = apply_output_activation(logits, output_activation)
            _, target_centers, present = gaussian_heatmaps_from_labels(labels, cfg, sigma_voxels=sigma_voxels)
            pred_centers = argmax_centers_from_outputs(outputs, cfg)

            image_np = images[0, 0].detach().cpu().numpy()
            target_np = target_centers[0].detach().cpu().numpy()
            pred_np = pred_centers[0].detach().cpu().numpy()
            present_np = present[0].detach().cpu().numpy().astype(bool)
            valid_z = [int(round(target_np[i, 0])) for i, ok in enumerate(present_np) if ok]
            if not valid_z:
                valid_z = [image_np.shape[0] // 2]
            slice_ids = sorted(set(valid_z))
            if len(slice_ids) > 4:
                idx = np.linspace(0, len(slice_ids) - 1, 4).round().astype(int)
                slice_ids = [slice_ids[i] for i in idx]

            fig, axes = plt.subplots(1, len(slice_ids), figsize=(4 * len(slice_ids), 4), squeeze=False)
            for ax, z_idx in zip(axes[0], slice_ids):
                ax.imshow(image_np[z_idx], cmap="gray")
                for channel_idx, ok in enumerate(present_np):
                    if not ok:
                        continue
                    true_z, true_y, true_x = target_np[channel_idx]
                    pred_z, pred_y, pred_x = pred_np[channel_idx]
                    if int(round(true_z)) == z_idx:
                        ax.scatter(true_x, true_y, c="lime", marker="x", s=35, linewidths=1.5)
                        ax.text(true_x + 2, true_y + 2, CLASS_NAMES[channel_idx + 1], color="lime", fontsize=7)
                    if int(round(pred_z)) == z_idx:
                        ax.scatter(pred_x, pred_y, c="red", marker="+", s=35, linewidths=1.5)
                ax.set_title(f"z={z_idx}")
                ax.axis("off")
            fig.suptitle(f"{case_id}: target x, prediction +")
            fig.tight_layout()
            out_path = out_dir / f"{case_id}_heatmap_peak_overlay.png"
            fig.savefig(out_path, dpi=180)
            plt.close(fig)
            saved.append(out_path)
    return saved


def save_history_plots(history_df: pd.DataFrame, cfg: Config) -> None:
    plots_dir = Path(cfg.work_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for col, ylabel in [
        ("train_loss", "Train loss"),
        ("train_weighted_heatmap_loss", "Train weighted heatmap loss"),
        ("train_unweighted_mse", "Train unweighted MSE"),
        ("train_coord_loss", "Train soft-argmax coordinate loss"),
        ("val_loss", "Validation loss"),
        ("val_weighted_heatmap_loss", "Validation weighted heatmap loss"),
        ("val_unweighted_mse", "Validation unweighted MSE"),
        ("val_coord_loss", "Validation soft-argmax coordinate loss"),
        ("mean_centroid_dist_non_bg", "Mean centroid distance"),
        ("mean_centroid_dist_electrodes", "Electrode centroid distance"),
        ("grad_norm", "Gradient norm"),
        ("learning_rate", "Learning rate"),
    ]:
        if col not in history_df:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(history_df["epoch"], history_df[col], marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Heatmap {ylabel} vs epoch")
        if col == "learning_rate":
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(plots_dir / f"heatmap_{col}_vs_epoch.png", dpi=180)
        plt.close(fig)


def save_metrics(
    history_df: pd.DataFrame,
    per_sample_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    final_stats: Dict[str, float],
    cfg: Config,
) -> None:
    metrics_dir = Path(cfg.work_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(Path(cfg.work_dir) / "history_heatmap.csv", index=False)
    per_sample_df.to_csv(metrics_dir / "heatmap_per_sample_metrics.csv", index=False)
    per_class_df.to_csv(metrics_dir / "heatmap_per_class_metrics.csv", index=False)
    pd.DataFrame([{"metric": key, "value": value} for key, value in final_stats.items()]).to_csv(
        metrics_dir / "heatmap_summary_metrics.csv",
        index=False,
    )


def write_debug_snapshot(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Config,
    sigma_voxels: float,
    loss_name: str,
    foreground_alpha: float,
    lambda_coord: float,
    softargmax_temperature: float,
    output_activation: str,
    active_channel_indices: Optional[Sequence[int]] = None,
) -> Path:
    out_path = Path(cfg.work_dir) / "heatmap_debug_snapshot.json"
    model.eval()
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True).long()
    with torch.no_grad():
        logits = model(images)
        outputs = apply_output_activation(logits, output_activation)
        targets, target_centers, present = gaussian_heatmaps_from_labels(labels, cfg, sigma_voxels)
        loss, components, loss_by_channel, soft_centers = heatmap_objective(
            outputs=outputs,
            targets=targets,
            target_centers=target_centers,
            present=present,
            loss_name=loss_name,
            foreground_alpha=foreground_alpha,
            lambda_coord=lambda_coord,
            softargmax_temperature=softargmax_temperature,
            active_channel_indices=active_channel_indices,
        )
        pred_centers = argmax_centers_from_outputs(outputs, cfg)
        target_argmax_centers = argmax_centers_from_outputs(targets, cfg)

    rows = []
    for channel_idx, class_name in enumerate(CLASS_NAMES[1:]):
        pred = pred_centers[0, channel_idx].detach().cpu().numpy()
        target_argmax = target_argmax_centers[0, channel_idx].detach().cpu().numpy()
        target_center = target_centers[0, channel_idx].detach().cpu().numpy()
        soft_center = soft_centers[0, channel_idx].detach().cpu().numpy()
        dist = float(np.linalg.norm(pred - target_argmax)) if bool(present[0, channel_idx]) else np.nan
        rows.append(
            {
                "class_id": channel_idx + 1,
                "class_name": class_name,
                "present": bool(present[0, channel_idx]),
                "target_center_zyx": [float(v) for v in target_center],
                "target_argmax_zyx": [float(v) for v in target_argmax],
                "pred_argmax_zyx": [float(v) for v in pred],
                "softargmax_zyx": [float(v) for v in soft_center],
                "argmax_distance_vox": dist,
                "pred_peak_in_neighborhood": bool(np.isfinite(dist) and dist <= max(2.0, float(sigma_voxels))),
                "target_heatmap_max": float(targets[0, channel_idx].max().item()),
                "pred_output_max": float(outputs[0, channel_idx].max().item()),
                "per_channel_loss": float(loss_by_channel[0, channel_idx].item()),
            }
        )
    payload = {
        "case_id": str(batch["case_id"][0]),
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "logits_shape": list(logits.shape),
        "target_heatmap_shape": list(targets.shape),
        "loss": float(loss.item()),
        "weighted_heatmap_loss": float(components["weighted_heatmap_loss"].item()),
        "unweighted_mse": float(components["unweighted_mse"].item()),
        "coord_loss": float(components["coord_loss"].item()),
        "sigma_voxels": sigma_voxels,
        "foreground_alpha": foreground_alpha,
        "lambda_coord": lambda_coord,
        "softargmax_temperature": softargmax_temperature,
        "output_activation": output_activation,
        "active_channel_indices": list(active_channel_indices) if active_channel_indices is not None else None,
        "crop_origin_zyx": batch.get("crop_origin_zyx", torch.empty(0)).detach().cpu().tolist() if isinstance(batch.get("crop_origin_zyx"), torch.Tensor) else None,
        "landmark_crop_zyx": batch.get("landmark_crop_zyx", torch.empty(0)).detach().cpu().tolist() if isinstance(batch.get("landmark_crop_zyx"), torch.Tensor) else None,
        "channels": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved heatmap debug snapshot: {out_path}")
    return out_path


def compare_to_previous_methods(cfg: Config, final_stats: Dict[str, float]) -> Path:
    rows = [
        {
            "method": "Gaussian heatmap ML",
            "metric": "mean_centroid_dist_electrodes_vox",
            "value": final_stats.get("mean_centroid_dist_electrodes", np.nan),
        },
        {
            "method": "Gaussian heatmap ML",
            "metric": "electrode_accuracy_within_10vox",
            "value": final_stats.get("landmark_accuracy_within_10vox_electrodes", np.nan),
        },
        {
            "method": "Gaussian heatmap ML",
            "metric": "electrode_accuracy_within_15vox",
            "value": final_stats.get("landmark_accuracy_within_15vox_electrodes", np.nan),
        },
    ]

    previous_summary = repo_root() / "runs" / "cardiac_leads_ensemble_v3_v6" / "metrics" / "summary_metrics.csv"
    if previous_summary.exists():
        prev_df = pd.read_csv(previous_summary)
        prev = dict(zip(prev_df["metric"], prev_df["value"]))
        rows.extend(
            [
                {
                    "method": "Previous BCE/Dice ensemble v3+v6",
                    "metric": "mean_centroid_dist_electrodes_vox",
                    "value": prev.get("mean_centroid_dist_electrodes", np.nan),
                },
                {
                    "method": "Previous BCE/Dice ensemble v3+v6",
                    "metric": "electrode_accuracy_within_10vox",
                    "value": prev.get("landmark_accuracy_within_10vox_electrodes", np.nan),
                },
                {
                    "method": "Previous BCE/Dice ensemble v3+v6",
                    "metric": "electrode_accuracy_within_15vox",
                    "value": prev.get("landmark_accuracy_within_15vox_electrodes", np.nan),
                },
            ]
        )

    threshold_report = repo_root() / "legacy_code" / "claude" / "threshold_report.txt"
    if threshold_report.exists():
        text = threshold_report.read_text(encoding="utf-8", errors="ignore")
        detection = re.search(r"GT electrode detection\s*:\s*\d+/\d+\s*\(([\d.]+)%\)", text)
        mean_error = re.search(r"Mean position error\s*:\s*([\d.]+)\s*mm", text)
        if detection:
            rows.append({"method": "Classical CV / PointNet Step 6b", "metric": "electrode_detection_percent", "value": float(detection.group(1))})
        if mean_error:
            rows.append({"method": "Classical CV / PointNet Step 6b", "metric": "mean_position_error_mm", "value": float(mean_error.group(1))})

    cv_report = repo_root() / "legacy_code" / "claude" / "cv_report.txt"
    if cv_report.exists():
        text = cv_report.read_text(encoding="utf-8", errors="ignore")
        detection = re.search(r"Detected.*?\(([\d.]+)%\)", text)
        mean_error = re.search(r"Mean position error\s*:\s*([\d.]+)\s*mm", text)
        if detection:
            rows.append({"method": "Classical CV Step 5", "metric": "electrode_detection_percent", "value": float(detection.group(1))})
        if mean_error:
            rows.append({"method": "Classical CV Step 5", "metric": "mean_position_error_mm", "value": float(mean_error.group(1))})

    out_path = Path(cfg.work_dir) / "metrics" / "heatmap_method_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def parse_one_landmark(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return 1
    if text.isdigit():
        class_id = int(text)
        if 1 <= class_id < len(CLASS_NAMES):
            return class_id
        raise ValueError(f"--one-landmark must be 1-9, got {value!r}")
    normalized = text.lower().replace("-", "_")
    for class_id, class_name in enumerate(CLASS_NAMES):
        if class_id == 0:
            continue
        aliases = {class_name.lower(), class_name.lower().replace("-", "_")}
        if class_id == 1:
            aliases.update({"ll1", "lv_distal"})
        elif class_id == 2:
            aliases.update({"ll2", "lv_2"})
        elif class_id == 3:
            aliases.update({"ll3", "lv_3"})
        elif class_id == 4:
            aliases.update({"ll4", "lv_proximal"})
        elif class_id == 5:
            aliases.update({"rl1", "rv_distal"})
        elif class_id == 6:
            aliases.update({"rl2", "rv_proximal"})
        if normalized in aliases:
            return class_id
    raise ValueError(f"Unknown landmark for --one-landmark: {value!r}")


def fit_heatmap_model(
    cfg: Config,
    sigma_voxels: float,
    epochs: int,
    loss_name: str,
    foreground_alpha: float,
    lambda_coord: float,
    softargmax_temperature: float,
    output_activation: str,
    tiny_overfit: bool,
    tiny_cases: int,
    one_landmark_class_id: Optional[int] = None,
    landmark_centered_patches: bool = False,
) -> Dict[str, object]:
    seed_everything(cfg.seed)
    ensure_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Heatmap target sigma: {sigma_voxels} voxels")
    print(f"Heatmap output activation: {output_activation}")
    print(f"Foreground loss alpha: {foreground_alpha}")
    print(f"Coordinate loss lambda: {lambda_coord}")
    print(f"Soft-argmax temperature: {softargmax_temperature}")
    active_channel_indices = [one_landmark_class_id - 1] if one_landmark_class_id is not None else None
    class_filter = [one_landmark_class_id] if one_landmark_class_id is not None else None
    if one_landmark_class_id is not None:
        print(f"One-landmark sanity mode: class {one_landmark_class_id} ({CLASS_NAMES[one_landmark_class_id]})")

    cache = build_preprocessed_cache(cfg)
    labeled_files = cache["labeled"]
    train_files, val_files = create_train_val_split(labeled_files, seed=cfg.seed, val_fraction=0.20)
    if tiny_overfit:
        train_files = train_files[: max(1, tiny_cases)]
        val_files = train_files
        cfg.num_workers = 0
        cfg.samples_per_volume = max(cfg.samples_per_volume, 2)
        print(f"Tiny-overfit mode: train and validate on {len(train_files)} case(s).")

    train_dataset = LandmarkCenteredPatchDataset(
        train_files,
        cfg=cfg,
        samples_per_landmark=2 if tiny_overfit else 1,
        jitter_voxels=4 if tiny_overfit else 8,
        class_filter=class_filter,
        debug_crops=bool(tiny_overfit and landmark_centered_patches),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    if tiny_overfit and landmark_centered_patches:
        val_dataset = LandmarkCenteredPatchDataset(
            val_files,
            cfg=cfg,
            samples_per_landmark=1,
            jitter_voxels=0,
            class_filter=class_filter,
            debug_crops=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=cfg.pin_memory,
            persistent_workers=False,
        )
        print("Tiny-overfit validation uses landmark-centered patches, not full-volume inference.")
    else:
        val_loader = create_val_loader(val_files, cfg)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = None
    if not tiny_overfit:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_reduce_factor,
            patience=cfg.lr_reduce_patience,
            min_lr=cfg.min_learning_rate,
        )

    history_rows: List[Dict[str, float]] = []
    best_metric = np.inf
    best_path = cfg.weights_dir / "best_heatmap_model.pth"
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch_heatmap(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            sigma_voxels=sigma_voxels,
            loss_name=loss_name,
            foreground_alpha=foreground_alpha,
            lambda_coord=lambda_coord,
            softargmax_temperature=softargmax_temperature,
            output_activation=output_activation,
            active_channel_indices=active_channel_indices,
        )
        val_stats, per_sample_df, per_class_df = validate_heatmap(
            model=model,
            loader=val_loader,
            device=device,
            cfg=cfg,
            sigma_voxels=sigma_voxels,
            loss_name=loss_name,
            foreground_alpha=foreground_alpha,
            lambda_coord=lambda_coord,
            softargmax_temperature=softargmax_temperature,
            output_activation=output_activation,
            active_channel_indices=active_channel_indices,
        )
        metric = val_stats["mean_centroid_dist_non_bg"]
        if scheduler is not None:
            scheduler.step(metric)

        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **train_stats,
            **val_stats,
        }
        history_rows.append(row)
        print(
            f"[heatmap] epoch={epoch:03d} train_loss={train_stats['train_loss']:.5f} "
            f"heatmap={train_stats['train_weighted_heatmap_loss']:.5f} "
            f"coord={train_stats['train_coord_loss']:.2f} "
            f"val_loss={val_stats['val_loss']:.5f} "
            f"val_heatmap={val_stats['val_weighted_heatmap_loss']:.5f} "
            f"val_coord={val_stats['val_coord_loss']:.2f} "
            f"centroid={val_stats['mean_centroid_dist_non_bg']:.2f} "
            f"electrode_centroid={val_stats['mean_centroid_dist_electrodes']:.2f} "
            f"acc10_elec={val_stats['landmark_accuracy_within_10vox_electrodes']:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} grad_norm={train_stats['grad_norm']:.3f}"
        )

        if np.isfinite(metric) and metric < best_metric - cfg.early_stopping_min_delta:
            best_metric = metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "config": asdict(cfg),
                    "sigma_voxels": sigma_voxels,
                    "loss_name": loss_name,
                    "foreground_alpha": foreground_alpha,
                    "lambda_coord": lambda_coord,
                    "softargmax_temperature": softargmax_temperature,
                    "active_channel_indices": active_channel_indices,
                    "output_activation": output_activation,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if (not tiny_overfit) and epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}; best centroid distance={best_metric:.3f}")
                break

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best heatmap checkpoint: {best_path}")

    final_stats, per_sample_df, per_class_df = validate_heatmap(
        model=model,
        loader=val_loader,
        device=device,
        cfg=cfg,
        sigma_voxels=sigma_voxels,
        loss_name=loss_name,
        foreground_alpha=foreground_alpha,
        lambda_coord=lambda_coord,
        softargmax_temperature=softargmax_temperature,
        output_activation=output_activation,
        active_channel_indices=active_channel_indices,
    )
    history_df = pd.DataFrame(history_rows)
    save_metrics(history_df, per_sample_df, per_class_df, final_stats, cfg)
    save_history_plots(history_df, cfg)
    overlays = save_heatmap_peak_overlays(
        model,
        val_loader,
        device,
        cfg,
        sigma_voxels=sigma_voxels,
        output_activation=output_activation,
        max_cases=cfg.max_overlay_cases,
    )
    comparison_path = compare_to_previous_methods(cfg, final_stats)

    if tiny_overfit:
        train_losses = history_df["train_loss"].dropna().to_numpy(dtype=np.float64)
        decrease_fraction = (
            float((train_losses[0] - train_losses[-1]) / max(abs(train_losses[0]), 1e-8))
            if len(train_losses) >= 2
            else np.nan
        )
        centroid_ok = bool(np.isfinite(final_stats["mean_centroid_dist_non_bg"]) and final_stats["mean_centroid_dist_non_bg"] <= 2.0)
        loss_ok = bool(np.isfinite(decrease_fraction) and decrease_fraction >= 0.5)
        if centroid_ok:
            print(
                "Tiny-overfit sanity check PASSED: "
                f"mean argmax centroid error={final_stats['mean_centroid_dist_non_bg']:.2f} vox."
            )
        elif loss_ok and final_stats["mean_centroid_dist_non_bg"] <= 10.0:
            print(
                "Tiny-overfit sanity check partially passed: loss dropped strongly and "
                f"mean argmax centroid error={final_stats['mean_centroid_dist_non_bg']:.2f} vox."
            )
        else:
            print("WARNING: tiny-overfit did not strongly reduce loss/centroid error. Writing debug snapshot.")
            write_debug_snapshot(
                model=model,
                loader=train_loader,
                device=device,
                cfg=cfg,
                sigma_voxels=sigma_voxels,
                loss_name=loss_name,
                foreground_alpha=foreground_alpha,
                lambda_coord=lambda_coord,
                softargmax_temperature=softargmax_temperature,
                output_activation=output_activation,
                active_channel_indices=active_channel_indices,
            )

    summary = {
        "work_dir": str(Path(cfg.work_dir)),
        "best_checkpoint": str(best_path),
        "history": str(Path(cfg.work_dir) / "history_heatmap.csv"),
        "summary_metrics": str(Path(cfg.work_dir) / "metrics" / "heatmap_summary_metrics.csv"),
        "comparison": str(comparison_path),
        "overlays": [str(path) for path in overlays],
        "final_stats": final_stats,
    }
    (Path(cfg.work_dir) / "heatmap_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate Gaussian heatmap landmark regression for CRT lead localization.")
    parser.add_argument("--work-dir", default="runs/cardiac_leads_heatmap_v1")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian target sigma in voxels. Try 2 to 4.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--loss", choices=["mse", "smoothl1"], default="mse")
    parser.add_argument("--output-activation", choices=["sigmoid", "raw"], default="sigmoid")
    parser.add_argument("--foreground-loss-alpha", type=float, default=100.0)
    parser.add_argument("--lambda-coord", type=float, default=0.1)
    parser.add_argument("--softargmax-temperature", type=float, default=0.05)
    parser.add_argument(
        "--one-landmark",
        nargs="?",
        const="1",
        default=None,
        help="Restrict loss/metrics to one landmark class, e.g. LL1, Apex, or 1. Bare flag defaults to LL1.",
    )
    parser.add_argument("--landmark-centered-patches", action="store_true")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--tiny-cases", type=int, default=1)
    parser.add_argument("--small-model", action="store_true")
    parser.add_argument("--max-overlay-cases", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    cfg = Config()
    cfg.work_dir = str(resolve_path(args.work_dir, root))
    cfg.label_dilation_radius_voxels = 0
    cfg.learning_rate = args.learning_rate
    cfg.supervised_epochs = args.epochs
    cfg.warm_start_checkpoint = None
    cfg.amp = False
    cfg.enable_spatial_augmentation = False
    cfg.enable_intensity_augmentation = False
    if args.disable_augmentation:
        cfg.enable_spatial_augmentation = False
        cfg.enable_intensity_augmentation = False
    cfg.patch_size_3d = (args.patch_size, args.patch_size, args.patch_size)
    cfg.max_overlay_cases = args.max_overlay_cases
    if args.small_model or args.tiny_overfit:
        cfg.channels = (8, 16, 32, 64)
        cfg.strides = (2, 2, 2)
    one_landmark_class_id = parse_one_landmark(args.one_landmark)
    fit_heatmap_model(
        cfg=cfg,
        sigma_voxels=args.sigma,
        epochs=args.epochs,
        loss_name=args.loss,
        foreground_alpha=args.foreground_loss_alpha,
        lambda_coord=args.lambda_coord,
        softargmax_temperature=args.softargmax_temperature,
        output_activation=args.output_activation,
        tiny_overfit=args.tiny_overfit,
        tiny_cases=args.tiny_cases,
        one_landmark_class_id=one_landmark_class_id,
        landmark_centered_patches=bool(args.landmark_centered_patches or args.tiny_overfit),
    )


if __name__ == "__main__":
    main()
