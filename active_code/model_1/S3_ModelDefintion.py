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
    Flexible MONAI U-Net that can toggle between 2D and 3D.
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
    3D mode:
        input remains [B, 1, D, H, W]
    2D mode:
        select one informative axial slice from each 3D patch
        output becomes [B, 1, H, W]
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
    3D mode:
        sliding-window inference over the full volume
    2D mode:
        infer slice-by-slice and reconstruct a 3D logits volume
    Returns:
        logits of shape [B, C, D, H, W]
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
