# ==============================
# STEP 2: DATASET PREPARATION + AUGMENTATION
# ==============================
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from monai.data import list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Rand3DElasticd,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)

from S1_DataLoading_Preprocessing import Config

class NPZSegmentationDataset(Dataset):
    """
    Loads preprocessed .npz cases.
    Supports:
      - image only
      - image + label
      - metadata fields for later saving/evaluation
    """
    def __init__(
        self,
        npz_paths: Sequence[Path],
        pseudo_flags: Optional[Sequence[bool]] = None,
        transform: Optional[Compose] = None,
        with_label: bool = True,
    ) -> None:
        self.npz_paths = [Path(p) for p in npz_paths]
        self.pseudo_flags = list(pseudo_flags) if pseudo_flags is not None else [False] * len(self.npz_paths)
        if len(self.pseudo_flags) != len(self.npz_paths):
            raise ValueError("pseudo_flags must have same length as npz_paths")
        self.transform = transform
        self.with_label = with_label

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, index: int):
        path = self.npz_paths[index]
        with np.load(path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32)
            case_id = str(data["case_id"]) if "case_id" in data.files else path.stem
            sample = {"image": image}
            if self.with_label:
                if "label" not in data.files:
                    raise KeyError(f"{path} does not contain label")
                sample["label"] = data["label"].astype(np.int64)

        base_meta = {
            "case_id": case_id,
            "npz_path": str(path),
            "is_pseudo": torch.tensor(float(self.pseudo_flags[index]), dtype=torch.float32),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        if isinstance(sample, list):
            out = []
            for item in sample:
                item.update(base_meta)
                out.append(item)
            return out

        sample.update(base_meta)
        return sample


def build_train_transform(cfg: Config) -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            SpatialPadd(keys=["image", "label"], spatial_size=cfg.patch_size_3d, mode=("constant", "constant")),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.patch_size_3d,
                ratios=cfg.class_sampling_ratios,
                num_classes=cfg.num_classes,
                num_samples=cfg.samples_per_volume,
            ),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64), track_meta=False),
        ]
    )


def build_val_transform() -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64), track_meta=False),
        ]
    )


def build_unlabeled_transform() -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
        ]
    )


def build_gpu_train_augment(cfg: Config) -> Compose:
    """
    These MONAI transforms operate on torch tensors.
    They are applied after batches are moved to CUDA.
    """
    return Compose(
        [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3, spatial_axes=(1, 2)),
            RandAffined(
                keys=["image", "label"],
                prob=0.25,
                rotate_range=(0.12, 0.12, 0.12),
                scale_range=(0.10, 0.10, 0.10),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            Rand3DElasticd(
                keys=["image", "label"],
                prob=0.10,
                sigma_range=(4, 7),
                magnitude_range=(20, 50),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.20),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        ]
    )


def apply_gpu_augment(
    images: torch.Tensor,
    labels: torch.Tensor,
    gpu_aug: Optional[Compose],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if gpu_aug is None:
        return images, labels

    aug_images = []
    aug_labels = []
    for i in range(images.shape[0]):
        sample = {
            "image": images[i],
            "label": labels[i],
        }
        sample = gpu_aug(sample)
        aug_images.append(sample["image"])
        aug_labels.append(sample["label"])

    return torch.stack(aug_images, dim=0), torch.stack(aug_labels, dim=0)


def create_train_val_split(
    labeled_npz_paths: Sequence[Path],
    seed: int,
    val_fraction: float = 0.20,
) -> Tuple[List[Path], List[Path]]:
    train_files, val_files = train_test_split(
        list(labeled_npz_paths),
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )
    return sorted(train_files), sorted(val_files)


def create_train_loader(
    npz_paths: Sequence[Path],
    pseudo_flags: Sequence[bool],
    cfg: Config,
) -> DataLoader:
    dataset = NPZSegmentationDataset(
        npz_paths=npz_paths,
        pseudo_flags=pseudo_flags,
        transform=build_train_transform(cfg),
        with_label=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=list_data_collate,   # needed because RandCropByLabelClassesd emits lists
    )


def create_val_loader(
    npz_paths: Sequence[Path],
    cfg: Config,
) -> DataLoader:
    dataset = NPZSegmentationDataset(
        npz_paths=npz_paths,
        pseudo_flags=[False] * len(npz_paths),
        transform=build_val_transform(),
        with_label=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=cfg.pin_memory,
        persistent_workers=(max(1, cfg.num_workers // 2) > 0),
    )


def create_unlabeled_loader(
    npz_paths: Sequence[Path],
    cfg: Config,
) -> DataLoader:
    dataset = NPZSegmentationDataset(
        npz_paths=npz_paths,
        pseudo_flags=[False] * len(npz_paths),
        transform=build_unlabeled_transform(),
        with_label=False,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=cfg.pin_memory,
        persistent_workers=(max(1, cfg.num_workers // 2) > 0),
    )
