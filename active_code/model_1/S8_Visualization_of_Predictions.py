# ==============================
# STEP 8: VISUALIZATION OF PREDICTIONS
# ==============================
from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from S1_DataLoading_Preprocessing import Config
from S6_S7_Metrics_Quantitative_Plots import logits_to_label_map
from S3_ModelDefintion import infer_logits


@torch.no_grad()
def save_prediction_overlays(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    max_cases: Optional[int] = None,
) -> None:
    """
    Description
    -----------
    Save a project artifact such as a plot, table, checkpoint, or report. This function implements the save prediction overlays step.
    
    Parameters
    ----------
    model : torch.nn.Module (input)
        PyTorch model used by this step.
    val_loader : DataLoader (input)
        The val loader value supplied to this function.
    device : torch.device (input)
        Torch device used for tensor and model computation.
    cfg : Config (input)
        Configuration object containing project paths, model settings, and hyperparameters.
    max_cases : Optional[int] (input)
        Internal dataset identifier for a patient or case.
    
    Returns
    -------
    None
        No value is returned; the function is executed for orchestration, mutation of supplied objects, or file output.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: May create directories, write files, print progress, or update checkpoint/model state as part of the pipeline.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    model.eval()
    max_cases = cfg.max_overlay_cases if max_cases is None else max_cases
    if max_cases <= 0:
        print("Skipping overlays because max_overlay_cases <= 0.")
        return

    cfg.overlays_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    cmap = plt.get_cmap("tab10", cfg.num_classes)

    for batch in tqdm(val_loader, desc="Saving overlays"):
        if saved >= max_cases:
            break

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        case_id = batch["case_id"][0]

        logits = infer_logits(model, images, cfg)
        preds, _ = logits_to_label_map(logits, cfg)

        image_np = images[0, 0].cpu().numpy()  # [D,H,W]
        label_np = labels[0, 0].cpu().numpy().astype(np.int32)
        pred_np = preds[0, 0].cpu().numpy().astype(np.int32)

        union_fg = ((label_np > 0) | (pred_np > 0)).sum(axis=(1, 2))
        slice_idx = int(np.argmax(union_fg)) if np.max(union_fg) > 0 else image_np.shape[0] // 2

        img2d = image_np[slice_idx]
        gt2d = label_np[slice_idx]
        pr2d = pred_np[slice_idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        axes[0].imshow(img2d, cmap="gray")
        axes[0].set_title(f"{case_id} | Input slice {slice_idx}")
        axes[0].axis("off")

        axes[1].imshow(img2d, cmap="gray")
        axes[1].imshow(np.ma.masked_where(gt2d == 0, gt2d), cmap=cmap, alpha=0.40, vmin=0, vmax=cfg.num_classes - 1)
        axes[1].set_title("Ground truth overlay")
        axes[1].axis("off")

        axes[2].imshow(img2d, cmap="gray")
        axes[2].imshow(np.ma.masked_where(pr2d == 0, pr2d), cmap=cmap, alpha=0.40, vmin=0, vmax=cfg.num_classes - 1)
        axes[2].set_title("Prediction overlay")
        axes[2].axis("off")

        out_path = cfg.overlays_dir / f"{case_id}_overlay.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        saved += 1
        print(f"Saved overlay {saved}/{max_cases}: {out_path}")

    print(f"Finished saving {saved} overlay image(s).")
