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
    RandCropByPosNegLabeld,
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
        """
        Description
        -----------
        Initialize the object and store the inputs needed by later method calls.
        
        Parameters
        ----------
        self : Any (both)
            Instance receiving this method call.
        npz_paths : Sequence[Path] (input)
            Filesystem location used for reading inputs or writing outputs.
        pseudo_flags : Optional[Sequence[bool]] (input)
            Boolean option controlling whether the associated behavior is enabled.
        transform : Optional[Compose] (input)
            The transform value supplied to this function.
        with_label : bool (input)
            Boolean option controlling whether the associated behavior is enabled.
        
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
        self.npz_paths = [Path(p) for p in npz_paths]
        self.pseudo_flags = list(pseudo_flags) if pseudo_flags is not None else [False] * len(self.npz_paths)
        if len(self.pseudo_flags) != len(self.npz_paths):
            raise ValueError("pseudo_flags must have same length as npz_paths")
        self.transform = transform
        self.with_label = with_label

    def __len__(self) -> int:
        """
        Description
        -----------
        Return the number of records available from this dataset or collection.
        
        Parameters
        ----------
        self : Any (input)
            Instance receiving this method call.
        
        Returns
        -------
        int
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
        return len(self.npz_paths)

    def __getitem__(self, index: int):
        """
        Description
        -----------
        Load one indexed record and return it in the format expected by the DataLoader.
        
        Parameters
        ----------
        self : Any (input)
            Instance receiving this method call.
        index : int (input)
            Zero-based index selecting an item.
        
        Returns
        -------
        Any
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
    """
    Description
    -----------
    Construct a configured object used by the pipeline. This function implements the build train transform step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    Compose
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
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            SpatialPadd(keys=["image", "label"], spatial_size=cfg.patch_size_3d, mode=("constant", "constant")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.patch_size_3d,
                pos=1.0,
                neg=0.0,
                num_samples=cfg.samples_per_volume,
            ),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64), track_meta=False),
        ]
    )


def build_val_transform() -> Compose:
    """
    Description
    -----------
    Construct a configured object used by the pipeline. This function implements the build val transform step.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    Compose
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
    return Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64), track_meta=False),
        ]
    )


def build_unlabeled_transform() -> Compose:
    """
    Description
    -----------
    Construct a configured object used by the pipeline. This function implements the build unlabeled transform step.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    Compose
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
    return Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
        ]
    )


def build_gpu_train_augment(cfg: Config) -> Optional[Compose]:
    """
    Description
    -----------
    Construct a configured object used by the pipeline. This function implements the build gpu train augment step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    Optional[Compose]
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
    transforms = []
    if cfg.enable_spatial_augmentation:
        transforms.extend(
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
            ]
        )
    if cfg.enable_intensity_augmentation:
        transforms.extend(
            [
                RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.20),
                RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
            ]
        )
    return Compose(transforms) if transforms else None


def apply_gpu_augment(
    images: torch.Tensor,
    labels: torch.Tensor,
    gpu_aug: Optional[Compose],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description
    -----------
    Apply a transform or post-processing step to supplied data. This function implements the apply gpu augment step.
    
    Parameters
    ----------
    images : torch.Tensor (input)
        Batch of input image volumes or tensors.
    labels : torch.Tensor (input)
        Ground-truth label maps or target tensors.
    gpu_aug : Optional[Compose] (input)
        The gpu aug value supplied to this function.
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
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
    """
    Description
    -----------
    Create a split, loader, artifact, or derived data object. This function implements the create train val split step.
    
    Parameters
    ----------
    labeled_npz_paths : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    seed : int (input)
        The seed value supplied to this function. Units: count.
    val_fraction : float (input)
        The val fraction value supplied to this function.
    
    Returns
    -------
    Tuple[List[Path], List[Path]]
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
    """
    Description
    -----------
    Create a split, loader, artifact, or derived data object. This function implements the create train loader step.
    
    Parameters
    ----------
    npz_paths : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    pseudo_flags : Sequence[bool] (input)
        Boolean option controlling whether the associated behavior is enabled.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    DataLoader
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
        collate_fn=list_data_collate,   # needed because random crop transforms emit lists
    )


def create_val_loader(
    npz_paths: Sequence[Path],
    cfg: Config,
) -> DataLoader:
    """
    Description
    -----------
    Create a split, loader, artifact, or derived data object. This function implements the create val loader step.
    
    Parameters
    ----------
    npz_paths : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    DataLoader
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
    val_num_workers = max(0, cfg.num_workers // 2)
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
        num_workers=val_num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=val_num_workers > 0,
    )


def create_unlabeled_loader(
    npz_paths: Sequence[Path],
    cfg: Config,
) -> DataLoader:
    """
    Description
    -----------
    Create a split, loader, artifact, or derived data object. This function implements the create unlabeled loader step.
    
    Parameters
    ----------
    npz_paths : Sequence[Path] (input)
        Filesystem location used for reading inputs or writing outputs.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    DataLoader
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
    unlabeled_num_workers = max(0, cfg.num_workers // 2)
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
        num_workers=unlabeled_num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=unlabeled_num_workers > 0,
    )
