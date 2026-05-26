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
