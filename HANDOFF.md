# Final Handoff Checklist

## What Works Now

- Data loading and preprocessing for local de-identified CT NIfTI volumes and segmentation masks.
- Patient/file matching by internal dataset ID.
- Label remapping into a contiguous 10-class space.
- Train/validation splitting with diagnostics for overlap.
- 3D MONAI U-Net training and evaluation for landmark/electrode masks.
- Gradient clipping, early stopping, ReduceLROnPlateau, learning-rate logging, and gradient-norm logging.
- Centroid coordinate export for GT and predictions.
- Per-class/per-patient centroid error metrics.
- Confusion matrix, Dice, precision/recall, and centroid plots.
- Prediction overlays.
- Patient-specific bullseye plots.
- Presentation figure generation.
- Weighted ensemble evaluation.
- Standalone bullseye video generation.
- Experimental Gaussian heatmap regression sanity-test code.

## First Files/Scripts To Run

From repository root:

```powershell
cd "C:\...\Optimal-LV-Placement-ML"
```

Run diagnostics:

```powershell
python active_code\model_1\S14_Training_Diagnostics.py
```

Regenerate presentation outputs from the best completed run:

```powershell
python active_code\model_1\generate_presentation_outputs.py --run-dir runs\cardiac_leads_ensemble_v3_v6 --eval-only --force-centroids
```

Create bullseye video:

```powershell
python active_code\model_1\make_bullseye_video.py --input-dir runs\cardiac_leads_ensemble_v3_v6\bullseye_plots --output runs\cardiac_leads_ensemble_v3_v6\bullseye_plots\bullseye_gt_vs_prediction_1080p.mp4
```

Dry-check continuation training:

```powershell
python active_code\model_1\continue_training_from_best.py --check --output-dir runs\cardiac_leads_continued_from_v3_v6_long_check
```

Launch continuation training:

```powershell
python active_code\model_1\continue_training_from_best.py --output-dir runs\cardiac_leads_continued_from_v3_v6_long --epochs 150 --learning-rate 1e-6 --early-stopping-patience 12
```

## Best Current Model/Run

Best current result:

```text
runs/cardiac_leads_ensemble_v3_v6
```

This is a weighted ensemble, not a single trainable checkpoint. It uses:

```text
runs/cardiac_leads_apex_recovery_v3/weights/best_supervised_model.pth
runs/cardiac_leads_no_spatial_aug_v6/weights/best_supervised_model.pth
```

Best summary metrics:

- Foreground vs background voxel accuracy: 99.65%
- Foreground recall: 26.70%
- Foreground precision: 70.14%
- All landmarks within 10 voxels: 83.33%
- All landmarks within 15 voxels: 93.52%
- Electrodes within 10 voxels: 86.11%
- Electrodes within 15 voxels: 94.44%
- Mean electrode centroid error: about 6.51 voxels

Use centroid localization metrics as the primary result because voxel accuracy is inflated by background class imbalance.

## Important Run Outputs To Preserve

Preserve these folders and files for handoff:

```text
runs/cardiac_leads_ensemble_v3_v6/config.json
runs/cardiac_leads_ensemble_v3_v6/metrics/
runs/cardiac_leads_ensemble_v3_v6/plots/
runs/cardiac_leads_ensemble_v3_v6/overlays/
runs/cardiac_leads_ensemble_v3_v6/bullseye_plots/
runs/cardiac_leads_ensemble_v3_v6/presentation_figures/
runs/cardiac_leads_apex_recovery_v3/weights/best_supervised_model.pth
runs/cardiac_leads_no_spatial_aug_v6/weights/best_supervised_model.pth
active_code/model_1/
legacy_code/research_baselines/*.txt
legacy_code/research_baselines/*.json
```

Cache folders can be regenerated and do not need to be uploaded:

```text
runs/*/cache/
__pycache__/
*.pyc
```

## Known Limitations

- Sparse landmark labels make Dice and voxel accuracy less clinically meaningful.
- Background voxels dominate the volume.
- Small labeled dataset for 3D deep learning.
- Some labels are absent in some patients.
- The best run is an ensemble and does not directly provide one trainable checkpoint.
- Current ML results should be compared against the classical CV baseline rather than interpreted alone.
- Raw data paths are local and must be updated on other machines.

## Recommended Next Experiment

Continue the Gaussian heatmap landmark-regression direction before another full hard-mask run:

1. Confirm one-patient, all-9-channel centered-patch overfit.
2. Confirm three-patient, all-9-channel centered-patch overfit.
3. Add or tune direct coordinate loss if argmax localization fails.
4. Only then move to full-volume or heart-ROI heatmap training.

Suggested command for the next sanity check:

```powershell
python active_code\model_1\S15_Heatmap_Landmark_Regression.py --work-dir runs\cardiac_leads_heatmap_debug_B_all9_1pt --tiny-overfit --tiny-cases 1 --landmark-centered-patches --disable-augmentation --epochs 12 --patch-size 32 --sigma 3 --foreground-loss-alpha 150 --lambda-coord 0.1 --softargmax-temperature 0.05 --learning-rate 0.001 --small-model
```

## Manual Attention Needed

- Confirm whether raw CT/segmentation/ROI files are allowed to be shared. If not, provide them separately through the approved course or lab mechanism.
- Confirm whether model checkpoint files are acceptable for GitHub upload; each main checkpoint is about 58 MB.
- Update local data paths on any new machine before running training.
- Run `continue_training_from_best.py --check` before launching the long continuation run.
