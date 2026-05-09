# ============================================================
# PHASE 5 — ANATOMICAL NORMALIZATION
# ============================================================

import json
from pathlib import Path

import nibabel as nib
import numpy as np

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
        "name": "ANT",
        "lead_type": "LANDMARK",
        "electrode_order": None,
        "description": "Anterior wall reference"
    },

    4002: {
        "name": "APEX",
        "lead_type": "LANDMARK",
        "electrode_order": None,
        "description": "Cardiac apex"
    },

    4003: {
        "name": "BASE",
        "lead_type": "LANDMARK",
        "electrode_order": None,
        "description": "Mitral valve/base"
    },

    4004: {
        "name": "LL1",
        "lead_type": "LV",
        "electrode_order": 1,
        "description": "LV distal tip"
    },

    4005: {
        "name": "LL2",
        "lead_type": "LV",
        "electrode_order": 2,
        "description": "LV electrode 2"
    },

    4006: {
        "name": "LL3",
        "lead_type": "LV",
        "electrode_order": 3,
        "description": "LV electrode 3"
    },

    4007: {
        "name": "LL4",
        "lead_type": "LV",
        "electrode_order": 4,
        "description": "LV proximal electrode"
    },

    4008: {
        "name": "RL1",
        "lead_type": "RV",
        "electrode_order": 1,
        "description": "RV distal tip"
    },

    4009: {
        "name": "RL2",
        "lead_type": "RV",
        "electrode_order": 2,
        "description": "RV proximal electrode"
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_vector(v):

    norm = np.linalg.norm(v)

    if norm == 0:
        return v

    return v / norm


def extract_label_centroid(
    seg_data,
    affine,
    label_value
):

    coords = np.argwhere(
        seg_data == label_value
    )

    if len(coords) == 0:
        return None

    voxel_centroid = coords.mean(axis=0)

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


def convert_lps_to_ras(coord):
    """
    Convert LPS -> RAS
    """

    x, y, z = coord

    return np.array([
        -x,
        -y,
        z
    ])


# ============================================================
# LONGITUDINAL POSITION
# ============================================================

def compute_longitudinal_position(
    point,
    apex,
    base
):

    axis = base - apex

    t = np.dot(
        point - apex,
        axis
    ) / np.dot(axis, axis)

    return t


# ============================================================
# RADIAL DISTANCE
# ============================================================

def compute_radial_distance(
    point,
    apex,
    base
):

    axis = base - apex

    t = np.dot(
        point - apex,
        axis
    ) / np.dot(axis, axis)

    projection = apex + t * axis

    radial_vector = point - projection

    radial_distance = np.linalg.norm(
        radial_vector
    )

    return radial_distance


# ============================================================
# ROTATION ANGLE
# ============================================================

def compute_rotation_angle(
    point,
    apex,
    base,
    ant
):

    axis = normalize_vector(
        base - apex
    )

    # --------------------------------------------------------
    # ANTERIOR REFERENCE VECTOR
    # --------------------------------------------------------

    ant_vector = ant - apex

    ant_projection = (
        np.dot(
            ant_vector,
            axis
        ) * axis
    )

    ant_reference = (
        ant_vector - ant_projection
    )

    ant_reference = normalize_vector(
        ant_reference
    )

    # --------------------------------------------------------
    # LEAD RADIAL VECTOR
    # --------------------------------------------------------

    lead_vector = point - apex

    lead_projection = (
        np.dot(
            lead_vector,
            axis
        ) * axis
    )

    lead_radial = (
        lead_vector - lead_projection
    )

    lead_radial = normalize_vector(
        lead_radial
    )

    # --------------------------------------------------------
    # ANGLE
    # --------------------------------------------------------

    dot_product = np.clip(

        np.dot(
            ant_reference,
            lead_radial
        ),

        -1.0,
        1.0
    )

    angle_rad = np.arccos(
        dot_product
    )

    angle_deg = np.degrees(
        angle_rad
    )

    return angle_deg


# ============================================================
# CHOOSE PATIENT
# ============================================================

patient_id = "10001"

patient = registry[patient_id]

# ============================================================
# LOAD SEGMENTATION
# ============================================================

seg_nii = nib.load(
    patient["segmentation"]
)

seg_data = seg_nii.get_fdata()

affine = seg_nii.affine

# ============================================================
# BUILD PATIENT RESULTS
# ============================================================

patient_results = {

    "patient_id": patient_id,

    "labels": {}
}

# ============================================================
# EXTRACT CENTROIDS
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

    # --------------------------------------------------------
    # CONVERT TO RAS
    # --------------------------------------------------------

    ras_coord = convert_lps_to_ras(

        centroid_data[
            "world_centroid_mm"
        ]
    )

    patient_results["labels"][
        metadata["name"]
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

        "num_voxels": centroid_data[
            "num_voxels"
        ],

        "voxel_centroid": centroid_data[
            "voxel_centroid"
        ],

        "world_centroid_mm": (
            ras_coord.tolist()
        )
    }

# ============================================================
# DEFINE ANATOMICAL LANDMARKS
# ============================================================

apex = np.array(
    patient_results["labels"]["APEX"][
        "world_centroid_mm"
    ]
)

base = np.array(
    patient_results["labels"]["BASE"][
        "world_centroid_mm"
    ]
)

ant = np.array(
    patient_results["labels"]["ANT"][
        "world_centroid_mm"
    ]
)

# ============================================================
# HEART AXIS
# ============================================================

heart_axis = base - apex

heart_axis_unit = normalize_vector(
    heart_axis
)

# ============================================================
# LV LEADS
# ============================================================

LV_LABELS = [
    "LL1",
    "LL2",
    "LL3",
    "LL4"
]

# ============================================================
# COMPUTE NORMALIZED POSITIONS
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZED LV POSITIONS")
print("=" * 60)

patient_results[
    "normalized_positions"
] = {}

for label in LV_LABELS:

    point = np.array(

        patient_results["labels"][
            label
        ]["world_centroid_mm"]
    )

    # --------------------------------------------------------
    # LONGITUDINAL
    # --------------------------------------------------------

    longitudinal = (
        compute_longitudinal_position(
            point,
            apex,
            base
        )
    )

    # --------------------------------------------------------
    # RADIAL DISTANCE
    # --------------------------------------------------------

    radial_distance = (
        compute_radial_distance(
            point,
            apex,
            base
        )
    )

    # --------------------------------------------------------
    # ROTATION ANGLE
    # --------------------------------------------------------

    rotation_angle = (
        compute_rotation_angle(
            point,
            apex,
            base,
            ant
        )
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    patient_results[
        "normalized_positions"
    ][label] = {

        "longitudinal_position": float(
            longitudinal
        ),

        "radial_distance_mm": float(
            radial_distance
        ),

        "rotation_angle_deg": float(
            rotation_angle
        )
    }

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(f"\n{label}")

    print("-" * 40)

    print(
        f"Longitudinal Position: "
        f"{longitudinal:.4f}"
    )

    print(
        f"Radial Distance (mm): "
        f"{radial_distance:.4f}"
    )

    print(
        f"Rotation Angle (deg): "
        f"{rotation_angle:.4f}"
    )

# ============================================================
# SAVE RESULTS
# ============================================================

output_path = Path(
    f"patient_{patient_id}_normalized.json"
)

with open(output_path, "w") as f:

    json.dump(
        patient_results,
        f,
        indent=4
    )

print("\n" + "=" * 60)
print("NORMALIZATION COMPLETE")
print("=" * 60)

print(f"\nSaved to:\n{output_path.resolve()}")