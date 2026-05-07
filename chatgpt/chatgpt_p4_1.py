import pandas as pd
import json
import numpy as np
from pathlib import Path

from chatgpt_p3_0_EXTRACT_CENTROIDS import patient_results

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

def load_horos_csv(csv_path):

    df = pd.read_csv(
        csv_path,
        header=None
    )

    df.columns = [
        "label",
        "x",
        "y",
        "z"
    ]

    return df

def compute_error_mm(
    automated_coord,
    manual_coord
):

    automated_coord = np.array(
        automated_coord
    )

    manual_coord = np.array(
        manual_coord
    )

    return np.linalg.norm(
        automated_coord - manual_coord
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