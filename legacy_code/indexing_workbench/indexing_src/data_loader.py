# src/data_loader.py
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

def build_patient_dictionary(root_paths: List[str]) -> Dict:
    """
    Scan two root folders, match raw CT, segmentation, PNGs, and manual CSVs.
    Returns a dict keyed by patient ID with all paths and a 'has_ground_truth' flag.
    """
    patient_dict = {}

    for base_path in root_paths:
        base = Path(base_path)

        img_dir = base / "HCT2_img_nii"
        seg_dir = base / "HCT2_leads_seg_nii"
        png_dir = base / "HCT2_leads_png"
        png_dir_alt = base / "HCT2_leads_groundtruth_png"
        rois_dir = base / "AUH-2024-HCT2-rois"

        if not img_dir.exists():
            print(f"Warning: {img_dir} not found")
            continue

        # Match both naming patterns
        img_files = list(img_dir.glob("*_HCT2.nii.gz")) + list(img_dir.glob("*_HCT2_img.nii.gz"))

        for img_file in img_files:
            patient_id = re.match(r"^(\d+)", img_file.name).group(1)
            if patient_id in patient_dict:
                continue

            patient_dict[patient_id] = {
                "raw_nii": str(img_file),
                "seg_nii": None,
                "png_folder": None,
                "rois_csv": None,
                "rois_bullseye_csv": None,
                "rois_bullseye_png": None,
                "has_ground_truth": False,
                "source_folder": str(base)
            }

            # Segmentation (two possible patterns)
            if seg_dir.exists():
                seg_file = seg_dir / f"{patient_id}_HCT2_leads_seg.nii.gz"
                if seg_file.exists():
                    patient_dict[patient_id]["seg_nii"] = str(seg_file)
                    patient_dict[patient_id]["has_ground_truth"] = True
                else:
                    seg_file_alt = seg_dir / f"{patient_id}_HCT2_HCT2_leads_seg.nii.gz"
                    if seg_file_alt.exists():
                        patient_dict[patient_id]["seg_nii"] = str(seg_file_alt)
                        patient_dict[patient_id]["has_ground_truth"] = True

            # PNG folders
            if png_dir.exists():
                png_sub = png_dir / f"{patient_id}_HCT2"
                if png_sub.exists():
                    patient_dict[patient_id]["png_folder"] = str(png_sub)
            if png_dir_alt.exists() and not patient_dict[patient_id]["png_folder"]:
                png_sub = png_dir_alt / f"{patient_id}_HCT2"
                if png_sub.exists():
                    patient_dict[patient_id]["png_folder"] = str(png_sub)

            # Manual CSVs (with or without leading dot)
            if rois_dir.exists():
                # main CSV
                csv_file = rois_dir / f"{patient_id}_leads.csv"
                if csv_file.exists():
                    patient_dict[patient_id]["rois_csv"] = str(csv_file)
                    patient_dict[patient_id]["has_ground_truth"] = True
                else:
                    csv_alt = rois_dir / f"._{patient_id}_leads.csv"
                    if csv_alt.exists():
                        patient_dict[patient_id]["rois_csv"] = str(csv_alt)
                        patient_dict[patient_id]["has_ground_truth"] = True

                # bullseye CSV
                bull_csv = rois_dir / f"{patient_id}_leads_bullseye.csv"
                if bull_csv.exists():
                    patient_dict[patient_id]["rois_bullseye_csv"] = str(bull_csv)
                else:
                    bull_alt = rois_dir / f"._{patient_id}_leads_bullseye.csv"
                    if bull_alt.exists():
                        patient_dict[patient_id]["rois_bullseye_csv"] = str(bull_alt)

                # bullseye PNG
                bull_png = rois_dir / f"{patient_id}_leads_bullseye.png"
                if bull_png.exists():
                    patient_dict[patient_id]["rois_bullseye_png"] = str(bull_png)
                else:
                    bull_png_alt = rois_dir / f"._{patient_id}_leads_bullseye.png"
                    if bull_png_alt.exists():
                        patient_dict[patient_id]["rois_bullseye_png"] = str(bull_png_alt)

    return patient_dict


def load_patient_index(json_path: str = "data/patient_data_index.json") -> Dict:
    """Load the previously saved patient index JSON."""
    with open(json_path, "r") as f:
        return json.load(f)