#!/usr/bin/env python
from src.data_loader import build_patient_dictionary
import json

if __name__ == "__main__":
    root_folders = [
        r"C:\Users\ayw005\Desktop\BENG 280C Project\BENG280C_pacing_lead_data_1st20",
        r"C:\Users\ayw005\Desktop\BENG 280C Project\HCT2_lead_segmentation_training"
    ]
    patient_dict = build_patient_dictionary(root_folders)

    with open("data/patient_data_index.json", "w") as f:
        json.dump(patient_dict, f, indent=2)

    gt_count = sum(1 for v in patient_dict.values() if v["has_ground_truth"])
    print(f"Total patients: {len(patient_dict)}")
    print(f"With ground truth: {gt_count}")
    print(f"Without ground truth: {len(patient_dict) - gt_count}")