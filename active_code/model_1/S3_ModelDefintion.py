# ==============================
# STEP 3: MODEL DEFINITION
# ==============================
from __future__ import annotations

from typing import Tuple

import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from S1_DataLoading_Preprocessing import Config

def build_model(cfg: Config) -> torch.nn.Module:
    """
    Description
    -----------
    Construct a configured object used by the pipeline. This function implements the build model step.
    
    Parameters
    ----------
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    torch.nn.Module
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
    if cfg.spatial_dims not in (2, 3):
        raise ValueError("cfg.spatial_dims must be 2 or 3")

    model = UNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=1,
        out_channels=cfg.num_landmark_classes,
        channels=cfg.channels,
        strides=cfg.strides,
        num_res_units=cfg.num_res_units,
        dropout=cfg.dropout,
        norm="INSTANCE",
    )
    return model


def prepare_training_batch_for_model(
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: Config,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description
    -----------
    Prepare tensors or metadata for training or inference. This function implements the prepare training batch for model step.
    
    Parameters
    ----------
    images : torch.Tensor (input)
        Batch of input image volumes or tensors.
    labels : torch.Tensor (input)
        Ground-truth label maps or target tensors.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    training : bool (input)
        The training value supplied to this function.
    
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
    if cfg.spatial_dims == 3:
        return images, labels

    # Slice-wise 2D mode
    selected_images = []
    selected_labels = []

    for img, lbl in zip(images, labels):
        # img/lbl shapes: [1, D, H, W]
        fg_per_slice = (lbl[0] > 0).sum(dim=(1, 2))  # [D]

        if fg_per_slice.max() > 0:
            valid_z = torch.where(fg_per_slice > 0)[0]
            if training:
                z_idx = valid_z[torch.randint(0, len(valid_z), (1,), device=valid_z.device)].item()
            else:
                z_idx = int(valid_z[len(valid_z) // 2].item())
        else:
            z_idx = img.shape[1] // 2

        selected_images.append(img[:, z_idx, :, :])  # [1, H, W]
        selected_labels.append(lbl[:, z_idx, :, :])  # [1, H, W]

    return torch.stack(selected_images, dim=0), torch.stack(selected_labels, dim=0)


@torch.no_grad()
def infer_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """
    Description
    -----------
    Run model inference and return output tensors. This function implements the infer logits step.
    
    Parameters
    ----------
    model : torch.nn.Module (input)
        PyTorch model used by this step.
    images : torch.Tensor (input)
        Batch of input image volumes or tensors.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    
    Returns
    -------
    torch.Tensor
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
    if cfg.spatial_dims == 3:
        return sliding_window_inference(
            inputs=images,
            roi_size=cfg.patch_size_3d,
            sw_batch_size=cfg.sw_batch_size,
            predictor=model,
            overlap=cfg.infer_overlap,
        )

    # 2D model: reconstruct 3D volume by running every axial slice
    b, c, d, h, w = images.shape
    logits_slices = []
    for z in range(d):
        logits_2d = model(images[:, :, z, :, :])  # [B, C, H, W]
        logits_slices.append(logits_2d.unsqueeze(2))  # [B, C, 1, H, W]
    return torch.cat(logits_slices, dim=2)
