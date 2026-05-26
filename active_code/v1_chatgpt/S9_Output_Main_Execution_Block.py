# ==============================
# STEP 9: OUTPUT + SINGLE MAIN EXECUTION BLOCK
# ==============================
from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from S1_DataLoading_Preprocessing import (
    CLASS_NAMES,
    RAW_TO_CONTIG,
    Config,
    build_preprocessed_cache,
    ensure_dirs,
    seed_everything,
)
from S2_DatasetPreparation_Augmentation import (
    build_gpu_train_augment,
    create_train_loader,
    create_train_val_split,
    create_unlabeled_loader,
    create_val_loader,
)
from S3_ModelDefintion import build_model
from S6_S7_Metrics_Quantitative_Plots import (
    evaluate_model,
    plot_confusion_matrix_heatmap,
    plot_dice_boxplot,
    plot_per_class_dice,
    plot_precision_recall_curve,
    plot_training_history,
    save_metrics_to_csv,
)
from S8_Visualization_of_Predictions import save_prediction_overlays


def _load_training_module():
    module_path = Path(__file__).with_name("S4_S5_Training_Semi-Supervised_Pseudo-Lableing.py")
    spec = importlib.util.spec_from_file_location("S4_S5_Training_Semi_Supervised_Pseudo_Lableing", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_training_module = _load_training_module()
build_loss_fn = _training_module.build_loss_fn
compute_class_weights = _training_module.compute_class_weights
fit_model = _training_module.fit_model
generate_pseudo_labels = _training_module.generate_pseudo_labels


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])


def load_history_if_available(cfg: Config) -> pd.DataFrame:
    all_history_path = Path(cfg.work_dir) / "training_history_all_stages.csv"
    supervised_history_path = Path(cfg.work_dir) / "history_supervised.csv"
    if all_history_path.exists():
        return pd.read_csv(all_history_path)
    if supervised_history_path.exists():
        return pd.read_csv(supervised_history_path)
    return pd.DataFrame()


def main() -> None:
    cfg = Config()
    seed_everything(cfg.seed)
    ensure_dirs(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------------
    # Step 1: preprocess and cache
    # -----------------------------------
    cache = build_preprocessed_cache(cfg)
    labeled_files = cache["labeled"]
    unlabeled_files = cache["unlabeled"]

    # -----------------------------------
    # Step 2: train/val split
    # -----------------------------------
    train_files, val_files = create_train_val_split(
        labeled_npz_paths=labeled_files,
        seed=cfg.seed,
        val_fraction=0.20,
    )

    print(f"Labeled train cases: {len(train_files)}")
    print(f"Labeled val cases:   {len(val_files)}")
    print(f"Unlabeled cases:     {len(unlabeled_files)}")

    train_loader = create_train_loader(
        npz_paths=train_files,
        pseudo_flags=[False] * len(train_files),
        cfg=cfg,
    )
    val_loader = create_val_loader(val_files, cfg)
    unlabeled_loader = create_unlabeled_loader(unlabeled_files, cfg) if unlabeled_files else None

    # -----------------------------------
    # Step 3: model
    # -----------------------------------
    model = build_model(cfg).to(device)

    # -----------------------------------
    # Step 4: supervised training
    # -----------------------------------
    supervised_ckpt = cfg.weights_dir / "best_supervised_model.pth"
    supervised_history = pd.DataFrame()
    finetune_history = pd.DataFrame()
    supervised_best_dice = float("nan")

    if cfg.eval_only and supervised_ckpt.exists():
        print(f"Eval-only mode: loading checkpoint without retraining: {supervised_ckpt}")
        load_checkpoint_weights(model, supervised_ckpt, device)
        supervised_history = load_history_if_available(cfg)
        if not supervised_history.empty and "val_dice" in supervised_history:
            supervised_best_dice = float(supervised_history["val_dice"].max())
    else:
        if cfg.eval_only:
            print(f"Eval-only requested, but no checkpoint was found at {supervised_ckpt}; starting training.")

        class_weights = compute_class_weights(train_files, cfg).to(device)
        loss_fn = build_loss_fn(class_weights=class_weights, cfg=cfg)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        gpu_aug = build_gpu_train_augment(cfg)

        supervised_history = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            cfg=cfg,
            epochs=cfg.supervised_epochs,
            checkpoint_path=supervised_ckpt,
            stage_name="supervised",
            gpu_aug=gpu_aug,
        )

        load_checkpoint_weights(model, supervised_ckpt, device)
        supervised_best_dice = float(supervised_history["val_dice"].max())

        # -----------------------------------
        # Step 5: pseudo-label generation + fine-tuning
        # -----------------------------------
        should_pseudo_label = (
            cfg.enable_pseudo_labeling
            and cfg.finetune_epochs > 0
            and unlabeled_loader is not None
            and supervised_best_dice >= cfg.min_supervised_dice_for_pseudo
        )

        if should_pseudo_label:
            pseudo_files = generate_pseudo_labels(
                model=model,
                unlabeled_loader=unlabeled_loader,
                device=device,
                cfg=cfg,
            )
            print(f"Accepted pseudo-labeled cases: {len(pseudo_files)}")

            combined_train_files = train_files + pseudo_files
            combined_flags = ([False] * len(train_files)) + ([True] * len(pseudo_files))

            finetune_loader = create_train_loader(
                npz_paths=combined_train_files,
                pseudo_flags=combined_flags,
                cfg=cfg,
            )

            finetune_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate * 0.5,
                weight_decay=cfg.weight_decay,
            )

            finetune_ckpt = cfg.weights_dir / "best_finetuned_model.pth"
            finetune_history = fit_model(
                model=model,
                train_loader=finetune_loader,
                val_loader=val_loader,
                optimizer=finetune_optimizer,
                loss_fn=loss_fn,
                device=device,
                cfg=cfg,
                epochs=cfg.finetune_epochs,
                checkpoint_path=finetune_ckpt,
                stage_name="finetune",
                gpu_aug=gpu_aug,
            )

            load_checkpoint_weights(model, finetune_ckpt, device)
        else:
            print(
                "Skipping pseudo-label fine-tuning. "
                f"enable_pseudo_labeling={cfg.enable_pseudo_labeling}, "
                f"finetune_epochs={cfg.finetune_epochs}, "
                f"best_supervised_dice={supervised_best_dice:.5f}, "
                f"required={cfg.min_supervised_dice_for_pseudo:.5f}"
            )

    # -----------------------------------
    # Steps 6 + 7: final evaluation + plots
    # -----------------------------------
    print("Starting final evaluation...")
    metrics = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
    )
    save_metrics_to_csv(metrics, cfg)
    print(f"Saved metrics CSVs in: {cfg.metrics_dir}")

    history_parts = [supervised_history]
    if not finetune_history.empty:
        history_parts.append(finetune_history)
    non_empty_history = [df for df in history_parts if not df.empty]
    if non_empty_history:
        full_history = pd.concat(non_empty_history, ignore_index=True)
        full_history.to_csv(Path(cfg.work_dir) / "training_history_all_stages.csv", index=False)
        plot_training_history(full_history, cfg)
    else:
        print("No training history CSV found; skipping training-history plots.")

    plot_dice_boxplot(metrics["per_sample_df"], cfg)
    plot_per_class_dice(metrics["per_class_df"], cfg)
    plot_precision_recall_curve(metrics["pr_curve_df"], cfg)
    plot_confusion_matrix_heatmap(metrics["confusion_df"], cfg)
    print(f"Saved plots in: {cfg.plots_dir}")

    # -----------------------------------
    # Step 8: overlays
    # -----------------------------------
    if cfg.save_overlays:
        print("Starting prediction overlays...")
        save_prediction_overlays(
            model=model,
            val_loader=val_loader,
            device=device,
            cfg=cfg,
            max_cases=cfg.max_overlay_cases,
        )
    else:
        print("Skipping overlays because cfg.save_overlays=False.")

    # -----------------------------------
    # Step 9: final model save
    # -----------------------------------
    final_model_path = cfg.weights_dir / "final_model_weights.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "raw_to_contig": RAW_TO_CONTIG,
            "config": asdict(cfg),
        },
        final_model_path,
    )

    print(f"Saved final weights to: {final_model_path}")
    print(f"Metrics CSVs saved in:   {cfg.metrics_dir}")
    print(f"Plots saved in:          {cfg.plots_dir}")
    print(f"Overlays saved in:       {cfg.overlays_dir}")


if __name__ == "__main__":
    main()
