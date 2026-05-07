import json
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LOAD REGISTRY
# ============================================================

REGISTRY_PATH = Path(
    r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\chatgpt\patient_registry.json"
)

with open(REGISTRY_PATH, "r") as f:
    registry = json.load(f)

# ============================================================
# LABEL METADATA
# ============================================================

LABEL_METADATA = {

    4001: {
        "name": "LL1",
        "lead_type": "LV",
        "electrode_order": 1,
        "description": "LV distal tip"
    },

    4002: {
        "name": "LL2",
        "lead_type": "LV",
        "electrode_order": 2,
        "description": "LV electrode 2"
    },

    4003: {
        "name": "LL3",
        "lead_type": "LV",
        "electrode_order": 3,
        "description": "LV electrode 3"
    },

    4004: {
        "name": "LL4",
        "lead_type": "LV",
        "electrode_order": 4,
        "description": "LV proximal electrode"
    },

    4005: {
        "name": "RL1",
        "lead_type": "RV",
        "electrode_order": 1,
        "description": "RV distal tip"
    },

    4006: {
        "name": "RL2",
        "lead_type": "RV",
        "electrode_order": 2,
        "description": "RV proximal electrode"
    },

    4007: {
        "name": "APEX",
        "lead_type": "LANDMARK",
        "electrode_order": None,
        "description": "Cardiac apex"
    },

    4008: {
        "name": "BASE",
        "lead_type": "LANDMARK",
        "electrode_order": None,
        "description": "Mitral valve/base"
    },

    4009: {
        "name": "ANT",
        "lead_type": "LANDMARK",
        "electrode_order": None,
        "description": "Anterior wall reference"
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_label_centroid(
    seg_data,
    affine,
    label_value
):
    """
    Extract centroid information for a label.
    """

    coords = np.argwhere(
        seg_data == label_value
    )

    if len(coords) == 0:
        return None

    # --------------------------------------------------------
    # VOXEL CENTROID
    # --------------------------------------------------------

    voxel_centroid = coords.mean(axis=0)

    # --------------------------------------------------------
    # WORLD COORDINATES
    # --------------------------------------------------------

    world_centroid = (
        nib.affines.apply_affine(
            affine,
            voxel_centroid
        )
    )

    return {

        "num_voxels": int(len(coords)),

        "voxel_centroid": (
            voxel_centroid.tolist()
        ),

        "world_centroid_mm": (
            world_centroid.tolist()
        )
    }

# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def visualize_centroid(
    raw_data,
    seg_data,
    label_value,
    voxel_centroid
):
    """
    Display centroid overlay for validation.
    """

    x, y, z = voxel_centroid

    z = int(round(z))

    raw_slice = raw_data[:, :, z]

    seg_slice = seg_data[:, :, z]

    plt.figure(figsize=(8, 8))

    plt.imshow(
        raw_slice,
        cmap="gray",
        vmin=-300,
        vmax=1000
    )

    plt.imshow(
        seg_slice > 0,
        cmap="Reds",
        alpha=0.5
    )

    plt.scatter(
        y,
        x,
        c="cyan",
        s=100
    )

    plt.title(
        f"Centroid Validation | Label {label_value}"
    )

    plt.axis("off")

    plt.show()

# ============================================================
# PROCESS PATIENT
# ============================================================

patient_id = "10001"

patient = registry[patient_id]

# ------------------------------------------------------------
# LOAD IMAGES
# ------------------------------------------------------------

raw_nii = nib.load(
    patient["raw_img"]
)

seg_nii = nib.load(
    patient["segmentation"]
)

raw_data = raw_nii.get_fdata()

seg_data = seg_nii.get_fdata()

affine = seg_nii.affine

# ============================================================
# BUILD PATIENT RESULTS
# ============================================================

patient_results = {

    "patient_id": patient_id,

    "raw_image_path": patient["raw_img"],

    "segmentation_path": patient["segmentation"],

    "image_shape": list(
        raw_data.shape
    ),

    "affine_matrix": affine.tolist(),

    "labels": {}
}

# ============================================================
# EXTRACT ALL LABELS
# ============================================================

for label_value, metadata in LABEL_METADATA.items():

    centroid_data = (
        extract_label_centroid(
            seg_data,
            affine,
            label_value
        )
    )

    if centroid_data is None:

        print(
            f"{metadata['name']}: NOT FOUND"
        )

        continue

    label_name = metadata["name"]

    # --------------------------------------------------------
    # MERGE METADATA + RESULTS
    # --------------------------------------------------------

    patient_results["labels"][
        label_name
    ] = {

        "label_value": label_value,

        "lead_type": metadata[
            "lead_type"
        ],

        "electrode_order": metadata[
            "electrode_order"
        ],

        "description": metadata[
            "description"
        ],

        **centroid_data
    }

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print(f"CENTROIDS FOR PATIENT {patient_id}")
print("=" * 60)

for label_name, data in (
    patient_results["labels"].items()
):

    print(f"\n{label_name}")

    print("-" * 40)

    print(
        f"Voxel Centroid: "
        f"{data['voxel_centroid']}"
    )

    print(
        f"World Coordinates (mm): "
        f"{data['world_centroid_mm']}"
    )

# ============================================================
# SAVE JSON
# ============================================================

output_path = Path(
    rf"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\chatgpt\patient_centroids_json\patient_{patient_id}_centroids.json"
)

with open(output_path, "w") as f:

    json.dump(
        patient_results,
        f,
        indent=4
    )

print("\n" + "=" * 60)
print(f"Saved results to:")
print(output_path.resolve())
print("=" * 60)

# ============================================================
# OPTIONAL VISUAL VALIDATION
# ============================================================

label_to_visualize = "LL2"

voxel_centroid = (
    patient_results["labels"][
        label_to_visualize
    ]["voxel_centroid"]
)

label_value = (
    patient_results["labels"][
        label_to_visualize
    ]["label_value"]
)

visualize_centroid(
    raw_data,
    seg_data,
    label_value,
    voxel_centroid
)