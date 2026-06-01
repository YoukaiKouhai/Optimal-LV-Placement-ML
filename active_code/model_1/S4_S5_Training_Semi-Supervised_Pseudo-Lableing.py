# ==============================
# STEPS 4 + 5: TRAINING + SEMI-SUPERVISED PSEUDO-LABELING
# ==============================
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from monai.losses import DiceCELoss
from monai.transforms import Compose

from S1_DataLoading_Preprocessing import ANATOMY_CLASSES, CLASS_NAMES, ELECTRODE_CLASSES, Config
from S2_DatasetPreparation_Augmentation import apply_gpu_augment
from S3_ModelDefintion import infer_logits, prepare_training_batch_for_model

def compute_class_weights(
    train_npz_paths: Sequence[Path],
    cfg: Config,
) -> torch.Tensor:
    num_classes = cfg.num_classes
    counts = np.zeros(num_classes, dtype=np.float64)
    total_voxels = 0
    for path in train_npz_paths:
        with np.load(path, allow_pickle=False) as data:
            y = data["label"].astype(np.int64)
        total_voxels += y.size
        unique, freq = np.unique(y, return_counts=True)
        counts[unique] += freq

    positives = np.maximum(counts[1:], 1.0)
    negatives = np.maximum(total_voxels - positives, 1.0)
    pos_weight = np.minimum(negatives / positives, cfg.bce_pos_weight_max)
    return torch.tensor(pos_weight, dtype=torch.float32)


def build_channel_loss_weights(cfg: Config) -> torch.Tensor:
    weights = torch.ones(cfg.num_landmark_classes, dtype=torch.float32)
    for class_id in cfg.focus_class_ids:
        channel_idx = class_id - 1
        if 0 <= channel_idx < cfg.num_landmark_classes:
            weights[channel_idx] = cfg.focus_class_loss_multiplier
    return weights


def labels_to_multichannel_targets(labels: torch.Tensor, cfg: Config) -> torch.Tensor:
    squeezed = labels.squeeze(1).long()
    channels = [(squeezed == class_id).float() for class_id in range(1, cfg.num_classes)]
    return torch.stack(channels, dim=1)


class LandmarkMaskLoss(torch.nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        channel_weight: torch.Tensor,
        lambda_dice: float,
        lambda_bce: float,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("channel_weight", channel_weight)
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = self.pos_weight.view(1, -1, *([1] * (targets.ndim - 2)))
        channel_weight = self.channel_weight.view(1, -1, *([1] * (targets.ndim - 2)))
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
        bce_loss = (bce * channel_weight).sum() / (
            targets.shape[0] * self.channel_weight.sum() * np.prod(targets.shape[2:])
        )

        probs = torch.sigmoid(logits)
        reduce_dims = tuple(range(2, probs.ndim))
        intersection = (probs * targets).sum(dim=reduce_dims)
        denom = probs.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice_loss_by_channel = 1.0 - ((2.0 * intersection + 1e-5) / (denom + 1e-5))
        flat_channel_weight = self.channel_weight.view(1, -1)
        dice_loss = (dice_loss_by_channel * flat_channel_weight).sum() / (
            targets.shape[0] * self.channel_weight.sum()
        )
        return (self.lambda_bce * bce_loss) + (self.lambda_dice * dice_loss)


def build_loss_fn(class_weights: torch.Tensor, cfg: Config) -> LandmarkMaskLoss:
    return LandmarkMaskLoss(
        pos_weight=class_weights,
        channel_weight=build_channel_loss_weights(cfg).to(class_weights.device),
        lambda_dice=cfg.lambda_dice,
        lambda_bce=cfg.lambda_ce,
    )


def labels_to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B, 1, D, H, W] or [B, 1, H, W]
    squeezed = labels.squeeze(1).long()
    onehot = F.one_hot(squeezed, num_classes=num_classes)
    dims = list(range(onehot.ndim))
    onehot = onehot.permute(0, dims[-1], *dims[1:-1]).float()
    return onehot


@torch.no_grad()
def mean_dice_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    threshold: float = 0.5,
) -> float:
    probs = torch.sigmoid(logits)
    pred_oh = (probs >= threshold).float()
    true_oh = torch.stack(
        [(labels.squeeze(1).long() == class_id).float() for class_id in range(1, num_classes)],
        dim=1,
    )

    reduce_dims = tuple(range(2, pred_oh.ndim))
    inter = (pred_oh * true_oh).sum(dim=reduce_dims)
    denom = pred_oh.sum(dim=reduce_dims) + true_oh.sum(dim=reduce_dims)

    dice = torch.where(
        denom > 0,
        (2.0 * inter + 1e-6) / (denom + 1e-6),
        torch.full_like(denom, torch.nan),
    )

    return float(torch.nanmean(dice).item())


@torch.no_grad()
def logits_to_label_map(logits: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    best_prob, best_channel = probs.max(dim=1, keepdim=True)
    preds = best_channel.long() + 1
    preds = torch.where(best_prob >= threshold, preds, torch.zeros_like(preds))
    return preds, probs


@torch.no_grad()
def mean_centroid_distance_from_channel_peaks(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int],
    threshold: float = 0.5,
) -> float:
    _, probs = logits_to_label_map(logits, threshold)
    distances: List[float] = []

    for batch_idx in range(probs.shape[0]):
        for class_id in class_ids:
            channel_idx = class_id - 1
            if channel_idx < 0 or channel_idx >= probs.shape[1]:
                continue
            flat_idx = int(torch.argmax(probs[batch_idx, channel_idx]).item())
            pred_centroid = torch.tensor(
                np.unravel_index(flat_idx, tuple(probs.shape[2:])),
                device=probs.device,
                dtype=torch.float32,
            )
            true_coords = torch.nonzero(labels[batch_idx, 0] == class_id, as_tuple=False).float()
            if true_coords.numel() == 0:
                continue

            true_centroid = true_coords.mean(dim=0)
            distances.append(float(torch.linalg.vector_norm(pred_centroid - true_centroid).item()))

    return float(np.mean(distances)) if distances else np.nan


@torch.no_grad()
def mean_centroid_distance_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int],
    threshold: float = 0.5,
) -> float:
    return mean_centroid_distance_from_channel_peaks(
        logits=logits,
        labels=labels,
        class_ids=class_ids,
        threshold=threshold,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    cfg: Config,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "checkpoint_metric": cfg.checkpoint_metric,
        "config": asdict(cfg),
    }
    torch.save(payload, path)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: Config,
    gpu_aug: Optional[Compose] = None,
) -> Tuple[float, float]:
    model.train()
    epoch_losses: List[float] = []
    grad_norms: List[float] = []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()

        if gpu_aug is not None:
            images, labels = apply_gpu_augment(images, labels, gpu_aug)

        model_inputs, model_labels = prepare_training_batch_for_model(
            images=images,
            labels=labels,
            cfg=cfg,
            training=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
            logits = model(model_inputs)
            targets = labels_to_multichannel_targets(model_labels, cfg)
            loss = loss_fn(logits, targets)

            # If this batch came from pseudo-labeled data, downweight it.
            # With train_batch_size=1, all patches in the batch are from the same source case.
            batch_is_pseudo = bool(batch["is_pseudo"][0].item()) if "is_pseudo" in batch else False
            if batch_is_pseudo:
                loss = cfg.pseudo_weight * loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        epoch_losses.append(float(loss.item()))
        grad_norms.append(float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm))

    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else np.nan
    mean_grad_norm = float(np.nanmean(grad_norms)) if grad_norms else np.nan
    return mean_loss, mean_grad_norm


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    cfg: Config,
) -> Dict[str, float]:
    model.eval()

    losses: List[float] = []
    dices: List[float] = []
    centroid_distances: List[float] = []
    focus_centroid_distances: Dict[int, List[float]] = {class_id: [] for class_id in cfg.focus_class_ids}

    for batch in tqdm(loader, desc="Validate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()

        with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
            logits = infer_logits(model, images, cfg)
            targets = labels_to_multichannel_targets(labels, cfg)
            loss = loss_fn(logits, targets)

        losses.append(float(loss.item()))
        dices.append(mean_dice_from_logits(logits, labels, cfg.num_classes, include_background=False, threshold=cfg.prediction_threshold))
        centroid_distances.append(mean_centroid_distance_from_logits(logits, labels, range(1, cfg.num_classes), threshold=cfg.prediction_threshold))
        for class_id in cfg.focus_class_ids:
            focus_centroid_distances[class_id].append(
                mean_centroid_distance_from_logits(
                    logits,
                    labels,
                    [class_id],
                    threshold=cfg.prediction_threshold,
                )
            )

    val_stats = {
        "val_loss": float(np.mean(losses)) if losses else np.nan,
        "val_dice": float(np.mean(dices)) if dices else np.nan,
        "val_centroid_dist": float(np.nanmean(centroid_distances)) if centroid_distances else np.nan,
    }
    focus_epoch_distances: List[float] = []
    for class_id, distances in focus_centroid_distances.items():
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        class_distance = float(np.nanmean(distances)) if distances else np.nan
        val_stats[f"val_centroid_dist_{class_name}"] = class_distance
        if np.isfinite(class_distance):
            focus_epoch_distances.append(class_distance)
    val_stats["val_focus_centroid_dist"] = float(np.mean(focus_epoch_distances)) if focus_epoch_distances else np.nan
    val_stats["val_worst_focus_centroid_dist"] = float(np.max(focus_epoch_distances)) if focus_epoch_distances else np.nan
    if np.isfinite(val_stats["val_centroid_dist"]) and np.isfinite(val_stats["val_dice"]):
        focus_penalty = (
            cfg.selection_score_focus_weight * val_stats["val_focus_centroid_dist"]
            if np.isfinite(val_stats["val_focus_centroid_dist"])
            else 0.0
        )
        worst_focus_penalty = (
            cfg.selection_score_worst_focus_weight * val_stats["val_worst_focus_centroid_dist"]
            if np.isfinite(val_stats["val_worst_focus_centroid_dist"])
            else 0.0
        )
        dice_reward = cfg.selection_score_dice_weight * val_stats["val_dice"]
        val_stats["val_selection_score"] = val_stats["val_centroid_dist"] + focus_penalty + worst_focus_penalty - dice_reward
    else:
        val_stats["val_selection_score"] = np.nan
    return val_stats


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    cfg: Config,
    epochs: int,
    checkpoint_path: Path,
    stage_name: str,
    gpu_aug: Optional[Compose] = None,
) -> pd.DataFrame:
    if cfg.checkpoint_mode not in {"min", "max"}:
        raise ValueError(f"checkpoint_mode must be 'min' or 'max', got {cfg.checkpoint_mode!r}")

    best_metric = np.inf if cfg.checkpoint_mode == "min" else -np.inf
    epochs_without_improvement = 0
    history_rows: List[Dict] = []
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))
    scheduler = None
    if cfg.use_reduce_lr_on_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.checkpoint_mode,
            factor=cfg.lr_reduce_factor,
            patience=cfg.lr_reduce_patience,
            min_lr=cfg.min_learning_rate,
        )

    initial_val_stats = validate_one_epoch(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        cfg=cfg,
    )
    initial_metric = initial_val_stats.get(cfg.checkpoint_metric)
    if initial_metric is None or not np.isfinite(initial_metric):
        raise ValueError(f"Initial checkpoint metric {cfg.checkpoint_metric!r} is missing or non-finite.")

    best_metric = float(initial_metric)
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        best_metric=best_metric,
        cfg=cfg,
    )
    initial_row = {
        "stage": stage_name,
        "epoch": 0,
        "train_loss": np.nan,
        "val_loss": initial_val_stats["val_loss"],
        "val_dice": initial_val_stats["val_dice"],
        "val_centroid_dist": initial_val_stats["val_centroid_dist"],
        "val_focus_centroid_dist": initial_val_stats["val_focus_centroid_dist"],
        "val_worst_focus_centroid_dist": initial_val_stats["val_worst_focus_centroid_dist"],
        "val_selection_score": initial_val_stats["val_selection_score"],
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "grad_norm": np.nan,
    }
    for class_id in cfg.focus_class_ids:
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        metric_key = f"val_centroid_dist_{class_name}"
        initial_row[metric_key] = initial_val_stats.get(metric_key, np.nan)
    history_rows.append(initial_row)

    print(
        f"[{stage_name}] epoch=000 "
        f"train_loss=nan "
        f"val_loss={initial_val_stats['val_loss']:.5f} "
        f"val_dice={initial_val_stats['val_dice']:.5f} "
        f"val_centroid_dist={initial_val_stats['val_centroid_dist']:.2f} "
        f"val_focus_centroid_dist={initial_val_stats['val_focus_centroid_dist']:.2f} "
        f"val_worst_focus_centroid_dist={initial_val_stats['val_worst_focus_centroid_dist']:.2f} "
        f"val_selection_score={initial_val_stats['val_selection_score']:.5f} "
        f"lr={optimizer.param_groups[0]['lr']:.2e} "
        f"grad_norm=nan "
        f"best_{cfg.checkpoint_metric}={best_metric:.5f}"
    )
    for class_id in cfg.focus_class_ids:
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        metric_key = f"val_centroid_dist_{class_name}"
        print(f"    focus {class_name} centroid_dist={initial_val_stats.get(metric_key, np.nan):.2f}")

    for epoch in range(1, epochs + 1):
        train_loss, grad_norm = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            cfg=cfg,
            gpu_aug=gpu_aug,
        )

        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            cfg=cfg,
        )

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_stats["val_loss"],
            "val_dice": val_stats["val_dice"],
            "val_centroid_dist": val_stats["val_centroid_dist"],
            "val_focus_centroid_dist": val_stats["val_focus_centroid_dist"],
            "val_worst_focus_centroid_dist": val_stats["val_worst_focus_centroid_dist"],
            "val_selection_score": val_stats["val_selection_score"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": grad_norm,
        }
        for class_id in cfg.focus_class_ids:
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            metric_key = f"val_centroid_dist_{class_name}"
            row[metric_key] = val_stats.get(metric_key, np.nan)

        metric_value = val_stats.get(cfg.checkpoint_metric)
        if metric_value is None or not np.isfinite(metric_value):
            raise ValueError(f"Checkpoint metric {cfg.checkpoint_metric!r} is missing or non-finite.")
        if scheduler is not None:
            scheduler.step(metric_value)
        row["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history_rows.append(row)
        if cfg.checkpoint_mode == "min":
            improved = metric_value < (best_metric - cfg.early_stopping_min_delta)
        else:
            improved = metric_value > (best_metric + cfg.early_stopping_min_delta)

        if improved:
            best_metric = metric_value
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                cfg=cfg,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[{stage_name}] epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_stats['val_loss']:.5f} "
            f"val_dice={val_stats['val_dice']:.5f} "
            f"val_centroid_dist={val_stats['val_centroid_dist']:.2f} "
            f"val_focus_centroid_dist={val_stats['val_focus_centroid_dist']:.2f} "
            f"val_worst_focus_centroid_dist={val_stats['val_worst_focus_centroid_dist']:.2f} "
            f"val_selection_score={val_stats['val_selection_score']:.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"grad_norm={grad_norm:.3f} "
            f"best_{cfg.checkpoint_metric}={best_metric:.5f}"
        )
        for class_id in cfg.focus_class_ids:
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
            metric_key = f"val_centroid_dist_{class_name}"
            print(f"    focus {class_name} centroid_dist={val_stats.get(metric_key, np.nan):.2f}")

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"[{stage_name}] early stopping after {epoch:03d} epochs; "
                f"best_{cfg.checkpoint_metric}={best_metric:.5f}"
            )
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(Path(cfg.work_dir) / f"history_{stage_name}.csv", index=False)
    return history_df


@torch.no_grad()
def tta_mean_probabilities(
    model: torch.nn.Module,
    image: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        mean_probs: [B, C, D, H, W]
        normalized_entropy: [B, D, H, W]
    """
    flip_sets = [
        tuple(),
        (2,),  # D
        (3,),  # H
        (4,),  # W
    ]

    probs_all = []
    for dims in flip_sets:
        aug_img = torch.flip(image, dims=dims) if dims else image
        logits = infer_logits(model, aug_img, cfg)
        probs = torch.softmax(logits, dim=1)
        if dims:
            probs = torch.flip(probs, dims=dims)
        probs_all.append(probs)

    prob_stack = torch.stack(probs_all, dim=0)
    mean_probs = prob_stack.mean(dim=0)

    entropy = -(mean_probs.clamp_min(1e-8) * mean_probs.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / math.log(cfg.num_classes)  # normalize to [0, 1] approximately
    return mean_probs, entropy


def class_threshold_tensor(device: torch.device, cfg: Config) -> torch.Tensor:
    thresholds = torch.ones(cfg.num_classes, device=device) * cfg.pseudo_min_prob_anatomy
    thresholds[0] = 1.0
    for c in ELECTRODE_CLASSES:
        thresholds[c] = cfg.pseudo_min_prob_electrode
    for c in ANATOMY_CLASSES:
        thresholds[c] = cfg.pseudo_min_prob_anatomy
    return thresholds


@torch.no_grad()
def generate_pseudo_labels(
    model: torch.nn.Module,
    unlabeled_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> List[Path]:
    model.eval()
    accepted_pseudo_files: List[Path] = []
    thresholds = class_threshold_tensor(device, cfg)
    stats_rows: List[Dict] = []

    cfg.pseudo_cache_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(unlabeled_loader, desc="Pseudo-labeling"):
        image_cpu = batch["image"]        # keep CPU copy for saving
        image = batch["image"].to(device, non_blocking=True)
        case_id = batch["case_id"][0]
        source_npz_path = Path(batch["npz_path"][0])

        mean_probs, entropy = tta_mean_probabilities(model, image, cfg)
        conf, pred = mean_probs.max(dim=1)  # [B, D, H, W]

        per_voxel_threshold = thresholds[pred]
        confident = (conf >= per_voxel_threshold) & (entropy <= cfg.pseudo_max_entropy)

        pseudo = torch.where(confident, pred, torch.zeros_like(pred))
        fg_mask = pseudo > 0

        fg_voxels = int(fg_mask.sum().item())
        case_mean_conf = float(conf[fg_mask].mean().item()) if fg_voxels > 0 else 0.0

        accept = (
            fg_voxels >= cfg.pseudo_min_foreground_voxels
            and case_mean_conf >= cfg.pseudo_min_case_mean_conf
        )

        stats_rows.append(
            {
                "case_id": case_id,
                "fg_voxels": fg_voxels,
                "pseudo_mean_conf": case_mean_conf,
                "accepted": int(accept),
                "source_npz": str(source_npz_path),
            }
        )

        if not accept:
            continue

        out_path = cfg.pseudo_cache_dir / f"{case_id}_pseudo.npz"
        np.savez_compressed(
            out_path,
            image=image_cpu[0, 0].cpu().numpy().astype(np.float32),
            label=pseudo[0].cpu().numpy().astype(np.uint8),
            case_id=case_id,
            source_image_npz=str(source_npz_path),
            pseudo_mean_conf=np.float32(case_mean_conf),
        )
        accepted_pseudo_files.append(out_path)

    pd.DataFrame(stats_rows).to_csv(Path(cfg.work_dir) / "pseudo_label_stats.csv", index=False)
    return sorted(accepted_pseudo_files)
