# ============================================================
# PHASE 6B — HEART REGION MASKING
# Anatomical ROI Constrained Lead Detection
# ============================================================

import json
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import (
    gaussian_filter,
    label,
    center_of_mass
)

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

shape = raw_data.shape

# ============================================================
# LOAD SEGMENTATION
# ============================================================

seg_available = patient["segmentation"] is not None

if seg_available:

    seg_nii = nib.load(
        patient["segmentation"]
    )

    seg_data = seg_nii.get_fdata()

# ============================================================
# LABEL MAP
# ============================================================

LABEL_MAP = {

    4001: "LL1",
    4002: "LL2",
    4003: "LL3",
    4004: "LL4",

    4005: "RL1",
    4006: "RL2",

    4007: "APEX",
    4008: "BASE",
    4009: "ANT"
}

# ============================================================
# LOAD BEST PARAMETERS FROM PHASE 6A
# ============================================================

best_params = patient["best_detection_params"]

HU_THRESHOLD = best_params["HU_THRESHOLD"]

SIGMA = best_params["sigma"]

MIN_BLOB_SIZE = best_params["min_blob_size"]

MAX_BLOB_SIZE = best_params["max_blob_size"]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def voxel_to_world(voxel_coord, affine):

    return nib.affines.apply_affine(
        affine,
        voxel_coord
    )

def world_to_voxel(world_coord, affine):

    inv_affine = np.linalg.inv(affine)

    return nib.affines.apply_affine(
        inv_affine,
        world_coord
    )

def compute_distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )

# ============================================================
# EXTRACT SEGMENTATION CENTROIDS
# ============================================================

segmentation_centroids = {}

for label_value, label_name in LABEL_MAP.items():

    coords = np.argwhere(
        seg_data == label_value
    )

    if len(coords) == 0:
        continue

    voxel_centroid = coords.mean(axis=0)

    world_centroid = voxel_to_world(
        voxel_centroid,
        affine
    )

    segmentation_centroids[label_name] = {

        "voxel": voxel_centroid,

        "world": world_centroid
    }

# ============================================================
# BUILD HEART ROI
# ============================================================

print("\n" + "=" * 60)
print("BUILDING HEART ROI")
print("=" * 60)

# ------------------------------------------------------------
# KEY LANDMARKS
# ------------------------------------------------------------

apex = segmentation_centroids["APEX"]["voxel"]

base = segmentation_centroids["BASE"]["voxel"]

ant = segmentation_centroids["ANT"]["voxel"]

# ------------------------------------------------------------
# HEART CENTER
# ------------------------------------------------------------

heart_center = (

    apex +
    base +
    ant

) / 3.0

print(f"\nHeart Center (voxel):")
print(heart_center)

# ------------------------------------------------------------
# HEART LONG AXIS
# ------------------------------------------------------------

long_axis = base - apex

long_axis_norm = long_axis / np.linalg.norm(
    long_axis
)

# ------------------------------------------------------------
# ESTIMATE HEART SIZE
# ------------------------------------------------------------

heart_length = np.linalg.norm(
    base - apex
)

print(f"\nEstimated Heart Length:")
print(f"{heart_length:.2f} voxels")

# ------------------------------------------------------------
# ROI RADII
# ------------------------------------------------------------

# Long axis radius
rz = heart_length * 0.75

# Transverse radii
rx = heart_length * 0.60
ry = heart_length * 0.60

print(f"\nROI Radii:")
print(f"rx = {rx:.2f}")
print(f"ry = {ry:.2f}")
print(f"rz = {rz:.2f}")

# ============================================================
# CREATE ELLIPSOID HEART MASK
# ============================================================

print("\nCreating 3D ellipsoid mask...")

X, Y, Z = np.indices(shape)

cx, cy, cz = heart_center

heart_mask = (

    ((X - cx) / rx) ** 2 +
    ((Y - cy) / ry) ** 2 +
    ((Z - cz) / rz) ** 2

) <= 1

print(f"\nHeart ROI voxels:")
print(np.sum(heart_mask))

# ============================================================
# APPLY HEART MASK
# ============================================================

print("\nApplying anatomical ROI constraint...")

smoothed_data = gaussian_filter(
    raw_data,
    sigma=SIGMA
)

metal_mask = (

    (smoothed_data > HU_THRESHOLD)

    &

    heart_mask

)

print(f"\nMetal voxels inside heart ROI:")
print(np.sum(metal_mask))

# ============================================================
# CONNECTED COMPONENTS
# ============================================================

labeled_array, num_features = label(
    metal_mask
)

print(f"\nConnected blobs found:")
print(num_features)

# ============================================================
# EXTRACT BLOBS
# ============================================================

candidate_blobs = []

for blob_id in range(1, num_features + 1):

    blob_mask = labeled_array == blob_id

    blob_size = np.sum(blob_mask)

    if blob_size < MIN_BLOB_SIZE:
        continue

    if blob_size > MAX_BLOB_SIZE:
        continue

    centroid_voxel = center_of_mass(
        blob_mask
    )

    centroid_world = voxel_to_world(
        centroid_voxel,
        affine
    )

    candidate_blobs.append({

        "blob_id": int(blob_id),

        "blob_size": int(blob_size),

        "voxel_centroid": [
            float(v)
            for v in centroid_voxel
        ],

        "world_centroid_mm": [
            float(v)
            for v in centroid_world
        ]
    })

# ============================================================
# DISPLAY BLOBS
# ============================================================

print("\n" + "=" * 60)
print("DETECTED HEART-CONSTRAINED BLOBS")
print("=" * 60)

for blob in candidate_blobs:

    print(f"\nBlob ID: {blob['blob_id']}")

    print(f"Blob Size: {blob['blob_size']}")

    print(
        f"Voxel Centroid: "
        f"{blob['voxel_centroid']}"
    )

    print(
        f"World Centroid: "
        f"{blob['world_centroid_mm']}"
    )

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION AGAINST SEGMENTATION")
print("=" * 60)

validation_results = {}

total_error = 0.0

matched_count = 0

for label_name, true_data in segmentation_centroids.items():

    true_coord = true_data["world"]

    best_distance = np.inf

    best_blob = None

    for blob in candidate_blobs:

        detected_coord = blob["world_centroid_mm"]

        dist = compute_distance(
            true_coord,
            detected_coord
        )

        if dist < best_distance:

            best_distance = dist

            best_blob = blob

    if best_blob is not None:

        validation_results[label_name] = {

            "best_blob_id": best_blob["blob_id"],

            "distance_mm": float(best_distance)
        }

        total_error += best_distance

        matched_count += 1

        print(f"\n{label_name}")

        print(
            f"Best Blob ID: "
            f"{best_blob['blob_id']}"
        )

        print(
            f"Distance Error (mm): "
            f"{best_distance:.4f}"
        )

# ============================================================
# FINAL ERROR
# ============================================================

if matched_count > 0:

    mean_error = total_error / matched_count

else:

    mean_error = np.inf

print("\n" + "=" * 60)
print("FINAL HEART-ROI VALIDATION")
print("=" * 60)

print(f"\nAverage Error:")
print(f"{mean_error:.4f} mm")

# ============================================================
# SAVE RESULTS
# ============================================================

output_data = {

    "patient_id": patient_id,

    "heart_center_voxel": [
        float(v)
        for v in heart_center
    ],

    "heart_radii_voxel": {

        "rx": float(rx),
        "ry": float(ry),
        "rz": float(rz)
    },

    "best_detection_params": best_params,

    "num_detected_blobs": len(candidate_blobs),

    "candidate_blobs": candidate_blobs,

    "validation_results": validation_results,

    "mean_validation_error_mm": float(mean_error)
}

output_path = Path(
    f"patient_{patient_id}_phase6b_heart_roi.json"
)

with open(output_path, "w") as f:

    json.dump(
        output_data,
        f,
        indent=4
    )

print(f"\nResults saved:")
print(output_path.resolve())

# ============================================================
# VISUALIZATION
# ============================================================

print("\nGenerating visualization...")

z_slice = int(round(cz))

raw_slice = raw_data[:, :, z_slice]

heart_slice = heart_mask[:, :, z_slice]

metal_slice = metal_mask[:, :, z_slice]

plt.figure(figsize=(10, 10))

plt.imshow(
    raw_slice,
    cmap="gray",
    vmin=-300,
    vmax=1000
)

plt.imshow(
    heart_slice,
    cmap="Blues",
    alpha=0.15
)

plt.imshow(
    metal_slice,
    cmap="autumn",
    alpha=0.7
)

plt.scatter(
    cy,
    cx,
    c="cyan",
    s=200,
    marker="x"
)

plt.title(
    f"Heart ROI Constrained Detection | Patient {patient_id}"
)

plt.axis("off")

plt.show()

print("\n🎉 PHASE 6B COMPLETE")