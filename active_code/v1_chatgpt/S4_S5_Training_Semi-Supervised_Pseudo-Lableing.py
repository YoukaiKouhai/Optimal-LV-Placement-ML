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

from S1_DataLoading_Preprocessing import ANATOMY_CLASSES, ELECTRODE_CLASSES, Config
from S2_DatasetPreparation_Augmentation import apply_gpu_augment
from S3_ModelDefintion import infer_logits, prepare_training_batch_for_model

def compute_class_weights(
    train_npz_paths: Sequence[Path],
    num_classes: int,
) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for path in train_npz_paths:
        with np.load(path, allow_pickle=False) as data:
            y = data["label"].astype(np.int64)
        unique, freq = np.unique(y, return_counts=True)
        counts[unique] += freq

    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()

    # So background does not dominate the CE term
    weights[0] *= 0.25
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_fn(class_weights: torch.Tensor, cfg: Config) -> DiceCELoss:
    return DiceCELoss(
        include_background=False,   # for Dice only
        to_onehot_y=True,
        softmax=True,
        weight=class_weights,
        lambda_dice=cfg.lambda_dice,
        lambda_ce=cfg.lambda_ce,
        squared_pred=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
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
) -> float:
    preds = torch.argmax(logits, dim=1, keepdim=True)
    pred_oh = labels_to_one_hot(preds, num_classes)
    true_oh = labels_to_one_hot(labels, num_classes)

    reduce_dims = tuple(range(2, pred_oh.ndim))
    inter = (pred_oh * true_oh).sum(dim=reduce_dims)
    denom = pred_oh.sum(dim=reduce_dims) + true_oh.sum(dim=reduce_dims)

    dice = torch.where(
        denom > 0,
        (2.0 * inter + 1e-6) / (denom + 1e-6),
        torch.full_like(denom, torch.nan),
    )

    if not include_background:
        dice = dice[:, 1:]

    return float(torch.nanmean(dice).item())


@torch.no_grad()
def mean_centroid_distance_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int],
) -> float:
    preds = torch.argmax(logits, dim=1, keepdim=True)
    distances: List[float] = []

    for batch_idx in range(preds.shape[0]):
        for class_id in class_ids:
            pred_coords = torch.nonzero(preds[batch_idx, 0] == class_id, as_tuple=False).float()
            true_coords = torch.nonzero(labels[batch_idx, 0] == class_id, as_tuple=False).float()
            if pred_coords.numel() == 0 or true_coords.numel() == 0:
                continue

            pred_centroid = pred_coords.mean(dim=0)
            true_centroid = true_coords.mean(dim=0)
            distances.append(float(torch.linalg.vector_norm(pred_centroid - true_centroid).item()))

    return float(np.mean(distances)) if distances else np.nan


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
        "config": asdict(cfg),
    }
    torch.save(payload, path)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DiceCELoss,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: Config,
    gpu_aug: Optional[Compose] = None,
) -> float:
    model.train()
    epoch_losses: List[float] = []

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
            loss = loss_fn(logits, model_labels)

            # If this batch came from pseudo-labeled data, downweight it.
            # With train_batch_size=1, all patches in the batch are from the same source case.
            batch_is_pseudo = bool(batch["is_pseudo"][0].item()) if "is_pseudo" in batch else False
            if batch_is_pseudo:
                loss = cfg.pseudo_weight * loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_losses.append(float(loss.item()))

    return float(np.mean(epoch_losses)) if epoch_losses else np.nan


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: DiceCELoss,
    device: torch.device,
    cfg: Config,
) -> Dict[str, float]:
    model.eval()

    losses: List[float] = []
    dices: List[float] = []
    centroid_distances: List[float] = []

    for batch in tqdm(loader, desc="Validate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()

        with torch.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
            logits = infer_logits(model, images, cfg)
            loss = loss_fn(logits, labels)

        losses.append(float(loss.item()))
        dices.append(mean_dice_from_logits(logits, labels, cfg.num_classes, include_background=False))
        centroid_distances.append(mean_centroid_distance_from_logits(logits, labels, range(1, cfg.num_classes)))

    return {
        "val_loss": float(np.mean(losses)) if losses else np.nan,
        "val_dice": float(np.mean(dices)) if dices else np.nan,
        "val_centroid_dist": float(np.nanmean(centroid_distances)) if centroid_distances else np.nan,
    }


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DiceCELoss,
    device: torch.device,
    cfg: Config,
    epochs: int,
    checkpoint_path: Path,
    stage_name: str,
    gpu_aug: Optional[Compose] = None,
) -> pd.DataFrame:
    best_dice = -np.inf
    epochs_without_improvement = 0
    history_rows: List[Dict] = []
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
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
        }
        history_rows.append(row)

        if val_stats["val_dice"] > best_dice:
            improved = val_stats["val_dice"] > (best_dice + cfg.early_stopping_min_delta)
            best_dice = val_stats["val_dice"]
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_dice,
                cfg=cfg,
            )
            epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        else:
            epochs_without_improvement += 1

        print(
            f"[{stage_name}] epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_stats['val_loss']:.5f} "
            f"val_dice={val_stats['val_dice']:.5f} "
            f"val_centroid_dist={val_stats['val_centroid_dist']:.2f}"
        )

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"[{stage_name}] early stopping after {epoch:03d} epochs; "
                f"best_val_dice={best_dice:.5f}"
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
