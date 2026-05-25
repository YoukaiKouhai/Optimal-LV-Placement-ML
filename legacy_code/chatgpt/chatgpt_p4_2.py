import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import json
from pathlib import Path


from chatgpt_p3_0_EXTRACT_CENTROIDS import  patient_results
from chatgpt_p4_1 import load_horos_csv, convert_lps_to_ras, compute_error_mm

def world_to_voxel(
    world_coord,
    affine
):
    """
    Convert world coordinates (mm)
    back to voxel coordinates.
    """

    inverse_affine = np.linalg.inv(
        affine
    )

    voxel_coord = nib.affines.apply_affine(
        inverse_affine,
        world_coord
    )

    return voxel_coord

# ============================================================
# LOAD REGISTRY
# ============================================================

REGISTRY_PATH = Path(
    r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\chatgpt\patient_registry.json"
)

with open(REGISTRY_PATH, "r") as f:
    registry = json.load(f)

# ============================================================
# PROCESS PATIENT
# ============================================================

patient_id = "10001"

patient = registry[patient_id]

# ============================================================
# LOAD MANUAL CSV
# ============================================================

manual_df = load_horos_csv(
    patient["roi_csv"]
)

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
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

for _, row in manual_df.iterrows():

    label_name = row["label"]

    if (
        label_name
        not in patient_results["labels"]
    ):
        continue

    # --------------------------------------------------------
    # AUTOMATED COORD
    # --------------------------------------------------------

    automated_coord = (
        patient_results["labels"][
            label_name
        ]["world_centroid_mm"]
    )

    # --------------------------------------------------------
    # CONVERT LPS -> RAS
    # --------------------------------------------------------

    automated_coord = (
        convert_lps_to_ras(
            automated_coord
        )
    )

    # --------------------------------------------------------
    # MANUAL COORD
    # --------------------------------------------------------

    manual_coord = np.array([

        row["x"],
        row["y"],
        row["z"]

    ])

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    error_mm = compute_error_mm(

        automated_coord,
        manual_coord

    )

    print(f"\n{label_name}")

    print(
        f"Automated: "
        f"{automated_coord}"
    )

    print(
        f"Manual: "
        f"{manual_coord}"
    )

    print(
        f"Error (mm): "
        f"{error_mm:.4f}"
    )


manual_world_coord = np.array([
    row["x"],
    row["y"],
    row["z"]
])

# RAS -> LPS
manual_world_coord = np.array([
    -manual_world_coord[0],
    -manual_world_coord[1],
     manual_world_coord[2]
])

manual_voxel = world_to_voxel(
    manual_world_coord,
    affine
)

mx, my, mz = manual_voxel

plt.scatter(
    my,
    mx,
    c="cyan",
    s=40,
    label="Automated"
)

plt.scatter(
    my,
    mx,
    c="magenta",
    s=40,
    label="Manual"
)

plt.legend()