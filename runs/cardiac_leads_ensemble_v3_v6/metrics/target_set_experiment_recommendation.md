# Target-Set Experiment Recommendation

## Current Best Run Evidence
- Run: `C:\Users\...\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\runs\cardiac_leads_ensemble_v3_v6`
- Mean electrode centroid error: 6.51 voxels
- Mean anatomy landmark centroid error: 9.80 voxels
- Electrodes within 10 voxels: 86.11%
- Anatomy landmarks within 10 voxels: 77.78%
- Worst class by centroid error: Base (14.31 voxels)

## Decision
Long continuation of the same all-class objective is not the first choice unless a new run shows electrode-only centroid error improving.
The safer next experiment is electrode-focused training from the best available weights, using electrode-only centroid localization as the primary model-selection metric.

## Why
Electrodes are bright metallic structures and are performing better than ANT/APEX/BASE. Anatomy landmarks are harder sparse anatomical targets and can pull the shared loss/selection metric away from electrode localization.

## Orientation Note
The NIfTI affine can identify patient/world orientation directions such as left/right, anterior/posterior, and superior/inferior. It cannot by itself identify the cardiac apex/base or the heart long axis. For cardiac top/bottom and bullseye normalization, this project still needs APEX/BASE/ANT landmarks, a heart segmentation, an atlas, or a separate anatomical detector.

## Example NIfTI Orientation
- File: `C:\Users\...\Desktop\BENG 280C Project\HCT2_lead_segmentation_training\HCT2_img_nii\10001_HCT2_img.nii.gz`
- Orientation codes: `PLS`
- Shape XYZ: `(512, 512, 189)`
- Zooms XYZ mm: `(0.337890625, 0.337890625, 0.699999988079071)`

## History Summary
- Saved history trend summary: `C:\Users\...\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\runs\cardiac_leads_ensemble_v3_v6\plots\history_target_metric_summary.csv`

## Next Commands
```powershell
python active_code\model_1\S16_TargetSet_Analysis_Orientation.py --run-dir runs\cardiac_leads_ensemble_v3_v6
python active_code\model_1\continue_training_from_best.py --check --target-set electrodes --anatomy-loss-weight 0.0 --checkpoint-metric val_centroid_dist_electrodes --output-dir runs\cardiac_leads_electrode_only_from_v3_v6
python active_code\model_1\continue_training_from_best.py --target-set electrodes --anatomy-loss-weight 0.0 --checkpoint-metric val_centroid_dist_electrodes --epochs 150 --learning-rate 1e-6 --output-dir runs\cardiac_leads_electrode_only_from_v3_v6
python active_code\model_1\continue_training_from_best.py --target-set all --electrode-loss-weight 1.0 --anatomy-loss-weight 0.2 --checkpoint-metric val_centroid_dist_electrodes --epochs 150 --learning-rate 1e-6 --output-dir runs\cardiac_leads_anatomy_downweighted_from_v3_v6
```

Primary decision metric: electrode-only centroid localization, especially percent of electrodes within 10 voxels and mean electrode centroid error.
