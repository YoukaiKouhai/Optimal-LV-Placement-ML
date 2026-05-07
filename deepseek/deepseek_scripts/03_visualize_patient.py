#!/usr/bin/env python
import sys
from src.data_loader import load_patient_index
from src.extract import extract_centroids
from src.visualize import interactive_centroid_viewer

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 03_visualize_patient.py <patient_id>")
        sys.exit(1)
    pid = sys.argv[1]

    index = load_patient_index("data/patient_data_index.json")
    if pid not in index:
        print(f"Patient {pid} not found in index")
        sys.exit(1)

    info = index[pid]
    if not info["seg_nii"]:
        print(f"No segmentation for {pid}, cannot visualise centroids")
        sys.exit(1)

    centroids_world, _, _ = extract_centroids(info["seg_nii"])
    interactive_centroid_viewer(info["raw_nii"], info["seg_nii"], centroids_world)