#!/usr/bin/env python
from src.data_loader import load_patient_index
from src.extract import extract_centroids
from src.validate import load_manual_csv, compute_errors, LABEL_TO_NAME
import numpy as np

if __name__ == "__main__":
    index = load_patient_index("data/patient_data_index.json")
    label_list = list(LABEL_TO_NAME.keys())

    for pid, info in index.items():
        if info["seg_nii"] and info["rois_csv"]:
            print(f"\nProcessing {pid}...")
            centroids_world, _, _ = extract_centroids(info["seg_nii"], label_list)
            manual = load_manual_csv(info["rois_csv"])
            errors = compute_errors(centroids_world, manual, LABEL_TO_NAME)
            if errors:
                for name, err in errors.items():
                    print(f"  {name}: {err:.2f} mm")
                mean_err = np.mean(list(errors.values()))
                print(f"  Mean error: {mean_err:.2f} mm")
                if mean_err > 2.0:
                    print("  WARNING: error > 2 mm – check affine/coordinate order")
            else:
                print("  No matching labels found – check label map")