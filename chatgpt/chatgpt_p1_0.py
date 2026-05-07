import os
import re
import json
from pathlib import Path

# ============================================================
# ROOT DATASETS
# ============================================================

DATASET1_ROOT = Path(
    r"C:\Users\ayw005\Desktop\BENG 280C Project\BENG280C_pacing_lead_data_1st20"
)

DATASET2_ROOT = Path(
    r"C:\Users\ayw005\Desktop\BENG 280C Project\HCT2_lead_segmentation_training"
)

# ============================================================
# REGEX PATTERNS
# ============================================================

PATIENT_REGEX = re.compile(r"(\d{5})")

# ============================================================
# MAIN REGISTRY
# ============================================================

registry = {}

# ============================================================
# HELPER FUNCTION
# ============================================================

def initialize_patient(patient_id, dataset_source):

    if patient_id not in registry:

        registry[patient_id] = {

            "dataset_source": dataset_source,

            "raw_img": None,

            "segmentation": None,

            "png_folder": None,

            "roi_csv": None,

            "bullseye_csv": None,

            "bullseye_png": None,

            "rois_series": None,

            "has_segmentation": False,

            "has_ground_truth": False,

            "has_png_slices": False
        }

# ============================================================
# PROCESS DATASET 1
# ============================================================

def process_dataset1():

    root = DATASET1_ROOT

    # --------------------------------------------------------
    # RAW IMAGES
    # --------------------------------------------------------

    raw_folder = root / "HCT2_img_nii"

    for file in raw_folder.glob("*.nii.gz"):

        match = PATIENT_REGEX.search(file.name)

        if match:

            patient_id = match.group(1)

            initialize_patient(
                patient_id,
                "dataset1"
            )

            registry[patient_id]["raw_img"] = str(
                file.resolve()
            )

    # --------------------------------------------------------
    # SEGMENTATIONS
    # --------------------------------------------------------

    seg_folder = root / "HCT2_leads_seg_nii"

    for file in seg_folder.glob("*.nii.gz"):

        match = PATIENT_REGEX.search(file.name)

        if match:

            patient_id = match.group(1)

            initialize_patient(
                patient_id,
                "dataset1"
            )

            registry[patient_id]["segmentation"] = str(
                file.resolve()
            )

            registry[patient_id][
                "has_segmentation"
            ] = True

    # --------------------------------------------------------
    # PNG FOLDERS
    # --------------------------------------------------------

    png_folder = root / "HCT2_leads_png"

    for folder in png_folder.iterdir():

        if folder.is_dir():

            match = PATIENT_REGEX.search(folder.name)

            if match:

                patient_id = match.group(1)

                initialize_patient(
                    patient_id,
                    "dataset1"
                )

                registry[patient_id]["png_folder"] = str(
                    folder.resolve()
                )

                registry[patient_id][
                    "has_png_slices"
                ] = True

    # --------------------------------------------------------
    # ROI FILES
    # --------------------------------------------------------

    roi_folder = root / "AUH-2024-HCT2-rois"

    for file in roi_folder.iterdir():

        match = PATIENT_REGEX.search(file.name)

        if not match:
            continue

        patient_id = match.group(1)

        initialize_patient(
            patient_id,
            "dataset1"
        )

        filename = file.name.lower()

        if filename.endswith("_leads.csv"):

            registry[patient_id]["roi_csv"] = str(
                file.resolve()
            )

            registry[patient_id][
                "has_ground_truth"
            ] = True

        elif filename.endswith(
            "_leads_bullseye.csv"
        ):

            registry[patient_id][
                "bullseye_csv"
            ] = str(file.resolve())

        elif filename.endswith(
            "_leads_bullseye.png"
        ):

            registry[patient_id][
                "bullseye_png"
            ] = str(file.resolve())

        elif filename.endswith(
            ".rois_series"
        ):

            registry[patient_id][
                "rois_series"
            ] = str(file.resolve())

# ============================================================
# PROCESS DATASET 2
# ============================================================

def process_dataset2():

    root = DATASET2_ROOT

    # --------------------------------------------------------
    # RAW IMAGES
    # --------------------------------------------------------

    raw_folder = root / "HCT2_img_nii"

    for file in raw_folder.glob("*.nii.gz"):

        match = PATIENT_REGEX.search(file.name)

        if match:

            patient_id = match.group(1)

            initialize_patient(
                patient_id,
                "dataset2"
            )

            registry[patient_id]["raw_img"] = str(
                file.resolve()
            )

    # --------------------------------------------------------
    # SEGMENTATIONS
    # --------------------------------------------------------

    seg_folder = root / "HCT2_leads_seg_nii"

    for file in seg_folder.glob("*.nii.gz"):

        match = PATIENT_REGEX.search(file.name)

        if match:

            patient_id = match.group(1)

            initialize_patient(
                patient_id,
                "dataset2"
            )

            registry[patient_id]["segmentation"] = str(
                file.resolve()
            )

            registry[patient_id][
                "has_segmentation"
            ] = True

    # --------------------------------------------------------
    # PNG FOLDERS
    # --------------------------------------------------------

    png_root = root / "HCT2_leads_groundtruth_png"

    for folder in png_root.iterdir():

        if folder.is_dir():

            match = PATIENT_REGEX.search(folder.name)

            if match:

                patient_id = match.group(1)

                initialize_patient(
                    patient_id,
                    "dataset2"
                )

                registry[patient_id]["png_folder"] = str(
                    folder.resolve()
                )

                registry[patient_id][
                    "has_png_slices"
                ] = True

    # --------------------------------------------------------
    # ROI FILES
    # --------------------------------------------------------

    roi_folder = root / "AUH-2024-HCT2-rois"

    for file in roi_folder.iterdir():

        match = PATIENT_REGEX.search(file.name)

        if not match:
            continue

        patient_id = match.group(1)

        initialize_patient(
            patient_id,
            "dataset2"
        )

        filename = file.name.lower()

        if filename.endswith("_leads.csv"):

            registry[patient_id]["roi_csv"] = str(
                file.resolve()
            )

            registry[patient_id][
                "has_ground_truth"
            ] = True

        elif filename.endswith(
            "_leads_bullseye.csv"
        ):

            registry[patient_id][
                "bullseye_csv"
            ] = str(file.resolve())

        elif filename.endswith(
            "_leads_bullseye.png"
        ):

            registry[patient_id][
                "bullseye_png"
            ] = str(file.resolve())

        elif filename.endswith(
            ".rois_series"
        ):

            registry[patient_id][
                "rois_series"
            ] = str(file.resolve())

# ============================================================
# RUN EVERYTHING
# ============================================================

process_dataset1()
process_dataset2()

# ============================================================
# SAVE JSON
# ============================================================

output_json = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\chatgpt\patient_registry.json")

with open(output_json, "w") as f:

    json.dump(
        registry,
        f,
        indent=4
    )

# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print(f"Total Patients: {len(registry)}")

num_seg = sum(
    p["has_segmentation"]
    for p in registry.values()
)

num_gt = sum(
    p["has_ground_truth"]
    for p in registry.values()
)

print(f"Patients with segmentations: {num_seg}")
print(f"Patients with ground truth: {num_gt}")

print("=" * 60)