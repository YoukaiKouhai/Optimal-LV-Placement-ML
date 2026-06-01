# Optimal LV Placement ML

UCSD Bioengineering (Spring 2026) BENG 280C - Imaging Cardiovascular Disease Course Project: Medical imaging pipeline for automatic cardiac resynchronization therapy (CRT) lead/electrode localization from 3D cardiac CT scans.

> **Special thanks** to Dr. Elliot McVeigh for their guidance, mentorship, and support throughout this project.

## Contributors
| Name |
|------|
| Austin Wong | 
| Deepta Bharadwaj |
| Nausheen Nakhawa |

## Objective

This project localizes CRT lead/electrode contacts and anatomical landmarks from cardiac CT. The pipeline supports:

- organizing raw CT, segmentation, and manual ROI files by internal patient ID
- extracting electrode and landmark centroids from segmentation masks
- validating centroids against Horos/manual annotation CSV coordinates
- normalizing lead coordinates with patient-specific APEX, BASE, and ANT landmarks
- evaluating classical computer vision lead detection baselines
- training and evaluating 3D MONAI/PyTorch ML models for landmark/electrode localization
- generating presentation outputs, including overlays, bullseye plots, centroid metrics, and summary figures

The clinical/research motivation is to translate 3D CT-visible CRT lead locations into interpretable spatial coordinates, including bullseye-style views that are relevant to CRT lead placement analysis.

## Data Expected

The repository expects de-identified data stored locally. Do not commit protected health information or raw patient data unless sharing has been explicitly approved.

Expected input data types:

- raw CT NIfTI volumes: `.nii` or `.nii.gz`
- segmentation NIfTI masks with integer labels
- Horos/manual ROI CSV files, when available
- optional PNG overlays or ground-truth slice images for visual checking

Current default local data roots are defined in [S1_DataLoading_Preprocessing.py](active_code/model_1/S1_DataLoading_Preprocessing.py):

```text
C:\...\BENG280C_pacing_lead_data_1st20
C:\...\HCT2_lead_segmentation_training
```

Users on another machine should update `DEFAULT_DATASET_ROOTS` or pass equivalent local paths in their own config/script.

## Label Map

The active ML code remaps raw segmentation labels into a contiguous training label space:

| Class | Name |
|---:|---|
| 0 | Background |
| 1 | LL1 / LV distal |
| 2 | LL2 / LV 2 |
| 3 | LL3 / LV 3 |
| 4 | LL4 / LV proximal |
| 5 | RL1 / RV distal |
| 6 | RL2 / RV proximal |
| 7 | ANT |
| 8 | Apex |
| 9 | Base |

## Expected Folder Structure

```text
Optimal-LV-Placement-ML/
  active_code/
    deep-research-report.md
    model_1/
      S1_DataLoading_Preprocessing.py
      S2_DatasetPreparation_Augmentation.py
      S3_ModelDefintion.py
      S4_S5_Training_Semi-Supervised_Pseudo-Lableing.py
      S6_S7_Metrics_Quantitative_Plots.py
      S8_Visualization_of_Predictions.py
      S9_Output_Main_Execution_Block.py
      S10_Bullseye_Lead_Visualization.py
      S11_Centroid_Export.py
      S12_Presentation_Figures.py
      S13_Ensemble_Evaluation.py
      S14_Training_Diagnostics.py
      S15_Heatmap_Landmark_Regression.py
      S16_TargetSet_Analysis_Orientation.py
      continue_training_from_best.py
      generate_presentation_outputs.py
      make_bullseye_video.py
      requirements.txt
  legacy_code/
    prototype_pipeline/
    research_baselines/
    indexing_workbench/
  runs/
    cardiac_leads_ensemble_v3_v6/
      metrics/
      plots/
      overlays/
      bullseye_plots/
      presentation_figures/
      config.json
```

Raw data should live outside the repository or in an ignored local data folder. Preprocessed `.npz` caches are regenerated and should not be treated as source data.

## Major Active Code Files

- `S1_DataLoading_Preprocessing.py`: configuration, dataset roots, patient matching, NIfTI loading, label remapping, resampling, CT normalization, and preprocessed cache generation.
- `S2_DatasetPreparation_Augmentation.py`: PyTorch datasets, train/validation split, patch sampling, and optional augmentations.
- `S3_ModelDefintion.py`: 3D MONAI U-Net model definition and sliding-window inference helper.
- `S4_S5_Training_Semi-Supervised_Pseudo-Lableing.py`: loss function, class weighting, training loop, checkpointing, gradient clipping, ReduceLROnPlateau, pseudo-label utilities.
- `S6_S7_Metrics_Quantitative_Plots.py`: Dice, IoU, centroid distance, confusion matrix, precision/recall, training plots, and CSV metric export.
- `S8_Visualization_of_Predictions.py`: CT/label/prediction overlay generation.
- `S9_Output_Main_Execution_Block.py`: main training/evaluation entry point for the hard-mask landmark model.
- `S10_Bullseye_Lead_Visualization.py`: patient-specific CRT bullseye plots from GT and predicted centroids.
- `S11_Centroid_Export.py`: exports GT and predicted centroid coordinate CSVs and centroid error CSVs.
- `S12_Presentation_Figures.py`: final presentation figures from saved run outputs.
- `S13_Ensemble_Evaluation.py`: evaluates a weighted ensemble of saved checkpoints. This produced the current best run.
- `S14_Training_Diagnostics.py`: data and training diagnostics, including split checks, label distribution, loader checks, and tiny-overfit mode.
- `S15_Heatmap_Landmark_Regression.py`: experimental Gaussian heatmap landmark regression with foreground-weighted loss and soft-argmax coordinate loss.
- `S16_TargetSet_Analysis_Orientation.py`: electrode-vs-anatomy metric summary, target-set experiment recommendation, and NIfTI orientation checks.
- `continue_training_from_best.py`: safe continuation training script for a new long run starting from the best available checkpoint sources.
- `generate_presentation_outputs.py`: runs centroid export, bullseye plot generation, and presentation figure generation without retraining.
- `make_bullseye_video.py`: standalone utility to make a 1080p MP4 from bullseye plot PNGs.

## Legacy Code

The `legacy_code/` folder contains earlier pipeline stages and experiments:

- `legacy_code/research_baselines/stage1_build_data_inventory.py`: patient/file inventory.
- `legacy_code/research_baselines/stage2_extract_centroids.py`: centroid extraction from segmentation masks.
- `legacy_code/research_baselines/stage3_normalize_coords.py`: anatomical coordinate normalization using APEX/BASE/ANT.
- `legacy_code/research_baselines/stage5a_cv_baseline.py`, `stage5b_ransac_sweep.py`, `stage6b_threshold.py`: classical CV lead detection experiments.
- `legacy_code/prototype_pipeline/phase1_data_registry.py`: earlier file loading and data organization script.

These scripts are useful for provenance and baseline comparison. The active ML workflow is in `active_code/model_1/`.

## Setup

Install dependencies in a Python environment with PyTorch, MONAI, nibabel, pandas, numpy, matplotlib, scikit-image, scipy, pillow, imageio, and OpenCV.

```powershell
cd "C:\...\Optimal-LV-Placement-ML"
python -m pip install -r active_code\model_1\requirements.txt
```

The current machine has additional packages installed locally. If a dependency is missing, install it into the active Python environment.

## Build or Check the Patient Registry

For the current active ML pipeline, registry-style matching and cache creation happen in `S1_DataLoading_Preprocessing.py` through `build_preprocessed_cache(cfg)`.

Run a diagnostic audit:

```powershell
cd "C:\...\Optimal-LV-Placement-ML"
python active_code\model_1\S14_Training_Diagnostics.py
```

Legacy inventory command:

```powershell
python legacy_code\research_baselines\stage1_build_data_inventory.py
```

## Centroid Extraction

From an existing completed ML run:

```powershell
python active_code\model_1\S11_Centroid_Export.py --run-dir runs\cardiac_leads_no_spatial_aug_v6 --force
```

Outputs:

```text
runs/<run_name>/metrics/centroid_coordinates.csv
runs/<run_name>/metrics/centroid_errors.csv
```

For the ensemble run, centroid CSVs are already present in:

```text
runs/cardiac_leads_ensemble_v3_v6/metrics/
```

## Validate Centroids Against Horos CSV Coordinates

Use the legacy centroid/validation scripts for manual Horos-coordinate comparison:

```powershell
python legacy_code\research_baselines\stage2_extract_centroids.py
```

Relevant outputs include `legacy_code/research_baselines/centroids_report.txt` and `legacy_code/research_baselines/centroids_results.json`.

Earlier validation found segmentation centroids matched manual/Horos coordinates after coordinate-convention correction; use centroid distance in millimeters as the main validation quantity.

## Anatomical Normalization

Normalize lead coordinates into a patient-specific coordinate frame using APEX, BASE, and ANT:

```powershell
python legacy_code\research_baselines\stage3_normalize_coords.py
```

The same coordinate-frame idea is used by `S10_Bullseye_Lead_Visualization.py` for bullseye plots:

- Base to Apex defines the long axis.
- ANT defines the angular reference direction.
- Electrodes are projected onto the short-axis plane.

## Classical CV Detection

Classical CV baselines are in `legacy_code/research_baselines/`.

```powershell
python legacy_code\research_baselines\stage5a_cv_baseline.py
python legacy_code\research_baselines\stage5b_ransac_sweep.py
python legacy_code\research_baselines\stage6b_threshold.py
```

Saved reports include:

- `legacy_code/research_baselines/cv_report.txt`
- `legacy_code/research_baselines/threshold_report.txt`
- `legacy_code/research_baselines/pointnet_report.txt`

These are useful baselines when deciding whether the ML model is improving over threshold/connected-component methods.

## Train the ML Model

The main hard-mask 3D U-Net training entry point is:

```powershell
python active_code\model_1\S9_Output_Main_Execution_Block.py
```

Training settings live in the `Config` dataclass in `S1_DataLoading_Preprocessing.py`.

The training loop includes:

- best-checkpoint saving
- early stopping
- gradient clipping
- ReduceLROnPlateau
- learning-rate logging
- gradient-norm logging
- train/validation loss logging
- Dice and centroid metrics

## Continue Training From the Current Best Result

The best evaluation result is the weighted ensemble in:

```text
runs/cardiac_leads_ensemble_v3_v6
```

Important: this run is an ensemble evaluation folder and does not contain a single trainable `weights/best_supervised_model.pth`. Its `config.json` lists the source checkpoints:

```text
runs/cardiac_leads_apex_recovery_v3/weights/best_supervised_model.pth
runs/cardiac_leads_no_spatial_aug_v6/weights/best_supervised_model.pth
```

The continuation script safely creates a weighted-average warm-start checkpoint in the new output folder and trains a single model from that initialization.

Dry check before training:

```powershell
python active_code\model_1\continue_training_from_best.py --check --output-dir runs\cardiac_leads_continued_from_v3_v6_long_check
```

Launch the long continuation run:

```powershell
python active_code\model_1\continue_training_from_best.py --output-dir runs\cardiac_leads_continued_from_v3_v6_long --epochs 150 --learning-rate 1e-6 --early-stopping-patience 12
```

This does not overwrite `runs/cardiac_leads_ensemble_v3_v6`. It refuses to reuse an existing output directory unless `--allow-existing-output` is supplied.

## Electrode-Focused Next Experiment

The best current model performs better on metallic electrodes than on anatomical landmarks. Use the target-set analysis script before choosing the next model:

```powershell
python active_code\model_1\S16_TargetSet_Analysis_Orientation.py --run-dir runs\cardiac_leads_ensemble_v3_v6
```

Recommended electrode-only dry check:

```powershell
python active_code\model_1\continue_training_from_best.py --check --target-set electrodes --anatomy-loss-weight 0.0 --checkpoint-metric val_centroid_dist_electrodes --output-dir runs\cardiac_leads_electrode_only_from_v3_v6_check
```

Recommended electrode-only training run:

```powershell
python active_code\model_1\continue_training_from_best.py --target-set electrodes --anatomy-loss-weight 0.0 --checkpoint-metric val_centroid_dist_electrodes --epochs 150 --learning-rate 1e-6 --early-stopping-patience 12 --output-dir runs\cardiac_leads_electrode_only_from_v3_v6
```

Alternative all-class run with anatomy downweighted:

```powershell
python active_code\model_1\continue_training_from_best.py --target-set all --electrode-loss-weight 1.0 --anatomy-loss-weight 0.2 --checkpoint-metric val_centroid_dist_electrodes --epochs 150 --learning-rate 1e-6 --early-stopping-patience 12 --output-dir runs\cardiac_leads_anatomy_downweighted_from_v3_v6
```

For these experiments, choose the next best model by electrode-only centroid localization, especially `landmark_accuracy_within_10vox_electrodes` and `mean_centroid_dist_electrodes`, not voxel accuracy.

## Evaluate Saved Runs

Evaluate the weighted ensemble:

```powershell
python active_code\model_1\S13_Ensemble_Evaluation.py --base-run-dir runs\cardiac_leads_no_spatial_aug_v6 --output-run-dir runs\cardiac_leads_ensemble_v3_v6
```

Generate centroid CSVs, bullseye plots, and presentation figures without retraining:

```powershell
python active_code\model_1\generate_presentation_outputs.py --run-dir runs\cardiac_leads_ensemble_v3_v6 --eval-only --force-centroids
```

Generate bullseye plots only:

```powershell
python active_code\model_1\S10_Bullseye_Lead_Visualization.py --run-dir runs\cardiac_leads_ensemble_v3_v6
```

Create a 1080p bullseye video:

```powershell
python active_code\model_1\make_bullseye_video.py --input-dir runs\cardiac_leads_ensemble_v3_v6\bullseye_plots --output runs\cardiac_leads_ensemble_v3_v6\bullseye_plots\bullseye_gt_vs_prediction_1080p.mp4
```

## Outputs Produced

Each run folder can contain:

- `config.json`: saved configuration
- `weights/`: model checkpoints, if the run trained a model
- `metrics/summary_metrics.csv`: summary metrics
- `metrics/per_class_metrics.csv`: per-class Dice, centroid, precision, recall, F1
- `metrics/per_sample_metrics.csv`: per-case metrics
- `metrics/confusion_matrix.csv`: voxel confusion matrix
- `metrics/centroid_coordinates.csv`: GT/predicted centroid coordinates
- `metrics/centroid_errors.csv`: centroid error per patient/class
- `plots/`: training curves, Dice plots, PR curves, confusion matrix
- `overlays/`: CT/GT/prediction image overlays
- `bullseye_plots/`: patient-specific CRT bullseye plots
- `presentation_figures/`: final project figures

## Current Best Results

Best current run:

```text
runs/cardiac_leads_ensemble_v3_v6
```

This is a weighted ensemble of:

- `runs/cardiac_leads_apex_recovery_v3/weights/best_supervised_model.pth`
- `runs/cardiac_leads_no_spatial_aug_v6/weights/best_supervised_model.pth`

Key validation metrics:

| Metric | Value |
|---|---:|
| Foreground vs background voxel accuracy | 99.65% |
| Foreground recall / sensitivity | 26.70% |
| Foreground precision / PPV | 70.14% |
| All landmarks within 10 voxels | 83.33% |
| All landmarks within 15 voxels | 93.52% |
| Electrodes within 10 voxels | 86.11% |
| Electrodes within 15 voxels | 94.44% |
| Mean electrode centroid error | about 6.51 voxels |

Important interpretation: voxel accuracy, specificity, and NPV are inflated because most voxels are background. This task is sparse landmark/electrode localization, so centroid distance and within-threshold centroid accuracy are more clinically meaningful than voxel accuracy alone.

Electrode/anatomy split for the best run:

| Group | Mean centroid error | Within 10 voxels |
|---|---:|---:|
| Electrodes: LL1-LL4, RL1-RL2 | about 6.51 voxels | 86.11% |
| Anatomy: ANT, Apex, Base | about 9.80 voxels | 77.78% |

## Experimental Gaussian Heatmap Regression

`S15_Heatmap_Landmark_Regression.py` implements a next-step landmark regression experiment:

- 9-channel Gaussian heatmap targets
- foreground-weighted heatmap loss
- soft-argmax coordinate loss
- one-landmark and tiny-overfit modes
- landmark-centered patch training

One-patient, one-landmark sanity testing reached about 1.41 voxel LL1 argmax error in a centered patch, but this is not yet a replacement for the best ensemble run. Do not launch full heatmap training until the all-9-channel and three-patient centered-patch sanity checks pass.

Example sanity command:

```powershell
python active_code\model_1\S15_Heatmap_Landmark_Regression.py --work-dir runs\cardiac_leads_heatmap_debug_A_ll1_12ep --tiny-overfit --tiny-cases 1 --one-landmark LL1 --landmark-centered-patches --disable-augmentation --epochs 12 --patch-size 32 --sigma 3 --foreground-loss-alpha 150 --lambda-coord 0.1 --softargmax-temperature 0.05 --learning-rate 0.001 --small-model
```

## Limitations

- Labels are sparse landmark-style targets, so voxel-level metrics can be misleading.
- Background dominates the volume and inflates voxel accuracy, specificity, and NPV.
- The labeled dataset is small for 3D deep learning.
- Some classes are missing in some cases.
- Centroid localization is more meaningful than Dice for this project.
- Current hard-mask segmentation training can reduce loss without improving difficult centroid classes.
- Ensemble performance is better than the individual single checkpoints but is not itself a single trainable model.
- The NIfTI affine gives patient/world orientation, but it does not identify cardiac apex/base or the heart long axis by itself.
- APEX, BASE, and ANT are currently needed to define the patient-specific cardiac coordinate frame used for bullseye plots.
- External users must provide their own de-identified data paths.

## Future Work

Recommended next research steps:

- Gaussian heatmap regression instead of hard cubic masks.
- Soft-argmax coordinate loss for direct localization pressure.
- Landmark-centered patch training before full-volume inference.
- Better heart/lead ROI masking to reduce background dominance.
- Larger labeled dataset and/or carefully vetted pseudo-labels.
- Stronger comparison against the classical CV baseline.
- Coarse-to-fine model: first localize heart/lead ROI, then localize electrodes inside the crop.
- Separate anatomy landmark model or heart segmentation/atlas to estimate APEX, BASE, and ANT independently from metallic electrode detection.

## Privacy and Handoff Notes

This repository should avoid storing PHI. Patient IDs in filenames are treated only as internal dataset identifiers. Raw CT volumes, segmentation masks, and Horos files may be large and sensitive; do not share them unless the dataset is confirmed de-identified and approved for the recipient.

See [HANDOFF.md](HANDOFF.md) for the short final handoff checklist.
