# ============================================================
# PHASE 6 — AUTOMATIC LEAD DETECTION (Improved Tuning)
# ============================================================

import json
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import nibabel as nib
from scipy.ndimage import gaussian_filter, label, center_of_mass

# ============================================================
# LOAD REGISTRY
# ============================================================

REGISTRY_PATH = Path(
    r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\chatgpt\patient_registry.json"
)

with open(REGISTRY_PATH, "r") as f:
    registry = json.load(f)

# ============================================================
# CHOOSE PATIENT
# ============================================================

patient_id = "10001"
patient = registry[patient_id]

# ============================================================
# LOAD RAW CT
# ============================================================

raw_nii = nib.load(patient["raw_img"])
raw_data = raw_nii.get_fdata()
affine = raw_nii.affine

# ============================================================
# LOAD SEGMENTATION (if available)
# ============================================================

seg_available = patient["segmentation"] is not None
if seg_available:
    seg_nii = nib.load(patient["segmentation"])
    seg_data = seg_nii.get_fdata()

# ============================================================
# LABEL DEFINITIONS
# ============================================================

LABEL_MAP = {
    4001: "LL1", 4002: "LL2", 4003: "LL3", 4004: "LL4",
    4005: "RL1", 4006: "RL2",
    4007: "APEX", 4008: "BASE", 4009: "ANT"
}

# ============================================================
# HELPER FUNCTIONS (unchanged except minor improvements)
# ============================================================

def voxel_to_world(voxel_coord, affine):
    return nib.affines.apply_affine(affine, voxel_coord)

def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_candidate_blobs(smoothed_data, affine, HU_THRESHOLD, min_blob_size, max_blob_size):
    metal_mask = smoothed_data > HU_THRESHOLD
    labeled_array, num_features = label(metal_mask)

    candidate_blobs = []
    for blob_id in range(1, num_features + 1):
        blob_mask = labeled_array == blob_id
        blob_size = np.sum(blob_mask)

        if blob_size < min_blob_size or blob_size > max_blob_size:
            continue

        centroid_voxel = center_of_mass(blob_mask)
        centroid_world = voxel_to_world(centroid_voxel, affine)

        candidate_blobs.append({
            "blob_id": int(blob_id),
            "blob_size": int(blob_size),
            "voxel_centroid": [float(v) for v in centroid_voxel],
            "world_centroid_mm": [float(v) for v in centroid_world]
        })

    return candidate_blobs, num_features

def validate_detection(candidate_blobs, seg_data, affine):
    segmentation_centroids = {}
    for label_value, label_name in LABEL_MAP.items():
        coords = np.argwhere(seg_data == label_value)
        if coords.size == 0:
            continue
        voxel_centroid = coords.mean(axis=0)
        world_centroid = voxel_to_world(voxel_centroid, affine)
        segmentation_centroids[label_name] = world_centroid

    if not segmentation_centroids:
        return np.inf, {}

    validation_results = {}
    total_error = 0.0
    matched_count = 0

    for label_name, true_coord in segmentation_centroids.items():
        best_distance = np.inf
        best_blob = None
        for blob in candidate_blobs:
            dist = compute_distance(true_coord, blob["world_centroid_mm"])
            if dist < best_distance:
                best_distance = dist
                best_blob = blob

        if best_blob:
            validation_results[label_name] = {
                "best_blob_id": best_blob["blob_id"],
                "distance_mm": float(best_distance),
                "detected_coord": best_blob["world_centroid_mm"],
                "true_coord": true_coord.tolist()
            }
            total_error += best_distance
            matched_count += 1

    mean_error = np.inf if matched_count == 0 else total_error / matched_count
    return mean_error, validation_results

def evaluate_parameters(HU_THRESHOLD, sigma, min_blob_size, max_blob_size):
    smoothed_data = gaussian_filter(raw_data, sigma=sigma)
    candidate_blobs, _ = extract_candidate_blobs(
        smoothed_data, affine, HU_THRESHOLD, min_blob_size, max_blob_size
    )

    if not seg_available:
        return np.inf, candidate_blobs, {}, 0

    mean_error, validation_results = validate_detection(candidate_blobs, seg_data, affine)
    return mean_error, candidate_blobs, validation_results, len(candidate_blobs)

# ============================================================
# TWO-STAGE RANDOM SEARCH (Coarse → Fine)
# ============================================================

print("\n" + "="*70)
print("STARTING TWO-STAGE PARAMETER TUNING (Coarse → Fine)")
print("="*70)

if not seg_available:
    print("No segmentation available → using default parameters")
    best_params = (2900, 0.5, 5, 400)
else:
    # Stage 1: Coarse Search
    print("Stage 1: Coarse Random Search...")
    n_coarse = 400

    coarse_space = []
    for _ in range(n_coarse):
        hu = random.randint(2400, 3600)
        sigma = round(random.uniform(0.0, 2.5), 1)
        minsz = random.randint(3, 25)
        maxsz = random.randint(150, 700)
        coarse_space.append((hu, sigma, minsz, maxsz))

    best_error = np.inf
    best_params = None
    best_blobs = None
    best_validation = None
    errors = []  # for visualization

    for i, params in enumerate(coarse_space):
        if i % 50 == 0:
            print(f"  Coarse {i}/{n_coarse} | Best error: {best_error:.4f} mm")

        error, blobs, val, _ = evaluate_parameters(*params)
        errors.append((params, error))

        if error < best_error:
            best_error = error
            best_params = params
            best_blobs = blobs
            best_validation = val
            print(f"  → New best: HU={params[0]}, σ={params[1]}, min={params[2]}, max={params[3]} | Error={error:.4f} mm")

    # Stage 2: Fine Search around best parameters
    print("\nStage 2: Fine Search around best parameters...")
    hu, sigma, minsz, maxsz = best_params

    fine_space = []
    for dh in [-100, -50, -20, 0, 20, 50, 100]:
        for ds in [-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5]:
            for dm in [-5, -3, -1, 0, 1, 3, 5]:
                for dM in [-50, -30, -10, 0, 10, 30, 50]:
                    fine_space.append((
                        int(hu + dh),
                        round(sigma + ds, 1),
                        max(3, minsz + dm),
                        maxsz + dM
                    ))

    random.shuffle(fine_space)
    fine_space = fine_space[:300]  # limit evaluations

    for i, params in enumerate(fine_space):
        if i % 50 == 0:
            print(f"  Fine {i}/{len(fine_space)} | Best error: {best_error:.4f} mm")

        error, blobs, val, _ = evaluate_parameters(*params)
        errors.append((params, error))

        if error < best_error:
            best_error = error
            best_params = params
            best_blobs = blobs
            best_validation = val
            print(f"  → New best: HU={params[0]}, σ={params[1]}, min={params[2]}, max={params[3]} | Error={error:.4f} mm")

    HU_THRESHOLD, SIGMA, MIN_BLOB_SIZE, MAX_BLOB_SIZE = best_params

    print("\n" + "="*60)
    print("TUNING COMPLETE - BEST PARAMETERS")
    print("="*60)
    print(f"HU_THRESHOLD  : {HU_THRESHOLD}")
    print(f"Sigma         : {SIGMA}")
    print(f"Min Blob Size : {MIN_BLOB_SIZE}")
    print(f"Max Blob Size : {MAX_BLOB_SIZE}")
    print(f"Best Avg Error: {best_error:.4f} mm")

# ============================================================
# APPLY BEST PARAMETERS
# ============================================================

print("\nApplying best parameters to final detection...")
smoothed_data = gaussian_filter(raw_data, sigma=SIGMA)
metal_mask = smoothed_data > HU_THRESHOLD

candidate_blobs, _ = extract_candidate_blobs(
    smoothed_data, affine, HU_THRESHOLD, MIN_BLOB_SIZE, MAX_BLOB_SIZE
)

# ============================================================
# VALIDATION
# ============================================================

if seg_available:
    validation_error, validation_results = validate_detection(candidate_blobs, seg_data, affine)
    print(f"\nFinal validation error: {validation_error:.4f} mm")

# ============================================================
# SAVE BEST PARAMETERS TO REGISTRY
# ============================================================

patient["best_detection_params"] = {
    "HU_THRESHOLD": int(HU_THRESHOLD),
    "sigma": float(SIGMA),
    "min_blob_size": int(MIN_BLOB_SIZE),
    "max_blob_size": int(MAX_BLOB_SIZE),
    "avg_error_mm": float(best_error) if 'best_error' in locals() else None
}

with open(REGISTRY_PATH, "w") as f:
    json.dump(registry, f, indent=4)

print(f"✅ Best parameters saved to registry for patient {patient_id}")

# ============================================================
# VISUALIZATION
# ============================================================

if seg_available and len(errors) > 0:
    plt.figure(figsize=(12, 8))

    # Plot error distribution
    error_values = [e for _, e in errors if e < 50]  # filter outliers
    plt.hist(error_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(best_error, color='red', linestyle='--', linewidth=2, label=f'Best Error = {best_error:.4f} mm')
    plt.title(f'Parameter Search Error Distribution - Patient {patient_id}')
    plt.xlabel('Mean Distance Error (mm)')
    plt.ylabel('Number of Parameter Combinations')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = f"patient_{patient_id}_tuning_histogram.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    print(f"📊 Tuning visualization saved as: {viz_path}")
    # plt.show()   # uncomment if you want to see it immediately

# ============================================================
# SAVE DETECTION RESULTS
# ============================================================

output_data = {
    "patient_id": patient_id,
    "best_params": patient["best_detection_params"],
    "num_detected_blobs": len(candidate_blobs),
    "candidate_blobs": candidate_blobs,
    "validation_error_mm": float(validation_error) if seg_available else None
}

output_path = Path(f"patient_{patient_id}_phase6_a_detection.json")
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"\n🎉 PHASE 6 COMPLETE - Results saved to: {output_path.resolve()}")