# Electrode-Only Continuation Diagnosis

## Run Status

The electrode-only continuation run completed normally and early-stopped after epoch 16.

## Main Finding

The run briefly improved electrode centroid distance, then plateaued and degraded relative to the best checkpoint.

| Epoch | Electrode Centroid Distance | Notes |
|---:|---:|---|
| 0 | 6.85 voxels | Warm-start baseline for this single averaged model |
| 1 | 6.68 voxels | Initial improvement |
| 4 | 6.55 voxels | Best saved electrode-centroid checkpoint |
| 16 | 6.58 voxels | Early stopping after no better electrode metric |

The final evaluation reported:

- Mean electrode centroid error: 6.55 voxels
- Electrodes within 10 voxels: 86.11%
- Mean anatomy centroid error: 9.81 voxels

Compared with the weighted ensemble run, the electrode-only continuation did not improve the primary electrode metric.

## Likely Causes

1. The warm-start model was already close to the best validation point, so there was little room for low-learning-rate continuation to help.
2. Electrode-only loss still updates the shared model output backbone, and small updates can shift all class peaks even when anatomy channels are not directly weighted.
3. The LR scheduler reduced the learning rate after the validation metric stopped improving, which indicates plateau rather than useful continued learning.
4. The validation set is small, so centroid metrics can jump when one difficult class or patient changes peak location.
5. The run used a weighted average of ensemble checkpoints as a single-model warm start. A single averaged model is not guaranteed to outperform the original ensemble.

## Recommendation

Do not use the final epoch as the best model. If comparing this experiment, use:

`weights/best_electrode_centroid_model.pth`

For the next experiment, prefer one of:

- keep the original weighted ensemble as the current best model,
- try anatomy-downweighted all-class training and compare only electrode centroid metrics,
- return to Gaussian heatmap/soft-argmax localization after centered-patch sanity checks pass.

Primary model-selection metric should remain `mean_centroid_dist_electrodes` and `landmark_accuracy_within_10vox_electrodes`, not voxel accuracy.
