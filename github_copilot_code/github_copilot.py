"""Starter script for CRT lead dataset discovery and centroid extraction.

Usage examples:
    python github_copilot.py scan --root . --output dataset_metadata.json
    python github_copilot.py centroids --seg path/to/segmentation.nii.gz --output centroids.json
    python github_copilot.py batch-centroids --metadata dataset_metadata.json --output centroids_batch.json
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

PATIENT_ID_RE = re.compile(r"(\d{4,6})")
KNOWN_DATASETS = {
    "data1": {
        "name": "BENG280C first 20",
        "root_names": ["BENG280C_pacing_lead_data_1st20"],
        "subfolders": {
            "raw": "HCT2_img_nii",
            "seg": "HCT2_leads_seg_nii",
            "png": "HCT2_leads_png",
            "rois": "AUH-2024-HCT2-rois",
        },
        "patterns": {
            "raw": ["*_HCT2.nii.gz"],
            "seg": ["*_HCT2_HCT2_leads_seg.nii.gz"],
            "roi_csv": ["*_leads.csv"],
        },
    },
    "data2": {
        "name": "HCT2 lead segmentation training",
        "root_names": ["HCT2_lead_segmentation_training"],
        "subfolders": {
            "raw": "HCT2_img_nii",
            "seg": "HCT2_leads_seg_nii",
            "groundtruth_png": "HCT2_leads_groundtruth_png",
            "rois": "AUH-2024-HCT2-rois",
        },
        "patterns": {
            "raw": ["*_HCT2_img.nii.gz"],
            "seg": ["*_HCT2_leads_seg.nii.gz"],
            "roi_csv": ["*_leads.csv"],
        },
    },
}
DEFAULT_LABELS = [4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008]

# Configuration - Edit these to customize behavior
CONFIG = {
    "root_path": r"C:\Users\ayw005\Desktop\BENG 280C Project",  # Change this to your data root directory
    "output_dir": r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\github_copilot_code/output",  # Directory to save JSON files
    "scan_datasets": True,  # Scan datasets and create metadata
    "compute_centroids": True,  # Compute centroids for all segmentations
}


def extract_patient_id(name: str) -> Optional[str]:
    """Extract a 4-6 digit patient ID from a file or folder name."""
    match = PATIENT_ID_RE.search(name)
    return match.group(1) if match else None


def make_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)) if path.exists() else str(path)


def scan_folder(folder: Path, root: Path, key_name: str) -> Dict[str, Any]:
    """Scan a dataset folder and categorize files by patient ID."""
    metadata: Dict[str, Any] = {}
    if not folder.exists():
        return metadata

    for item in folder.rglob("*"):
        if not item.is_file():
            continue

        patient_id = extract_patient_id(item.name) or extract_patient_id(str(item.parent.name))
        if patient_id is None:
            continue

        patient = metadata.setdefault(patient_id, {
            "patient_id": patient_id,
            "raw_files": [],
            "seg_files": [],
            "png_files": [],
            "groundtruth_png_files": [],
            "roi_csv_files": [],
            "rois_series_files": [],
            "other_files": [],
        })

        rel_path = make_relative(item, root)
        suffix = item.suffix.lower()

        if key_name == "raw" and suffix in {".nii", ".gz"}:
            patient["raw_files"].append(rel_path)
        elif key_name == "seg" and suffix in {".nii", ".gz"}:
            patient["seg_files"].append(rel_path)
        elif key_name == "png" and suffix == ".png":
            patient["png_files"].append(rel_path)
        elif key_name == "groundtruth_png" and suffix == ".png":
            patient["groundtruth_png_files"].append(rel_path)
        elif key_name == "rois":
            if suffix == ".csv":
                patient["roi_csv_files"].append(rel_path)
            elif suffix == ".png":
                patient["png_files"].append(rel_path)
            elif suffix == ".rois_series":
                patient["rois_series_files"].append(rel_path)
            else:
                patient["other_files"].append(rel_path)
        else:
            patient["other_files"].append(rel_path)

    return metadata


def scan_datasets(root: Path) -> Dict[str, Any]:
    """Scan supported dataset folders under a root path."""
    result: Dict[str, Any] = {
        "root": str(root),
        "datasets": {},
    }

    for dataset_key, dataset_spec in KNOWN_DATASETS.items():
        dataset_root: Optional[Path] = None
        for folder_name in dataset_spec["root_names"]:
            candidate = root / folder_name
            if candidate.exists():
                dataset_root = candidate
                break

        if dataset_root is None:
            continue

        dataset_meta: Dict[str, Any] = {
            "dataset_name": dataset_spec["name"],
            "root": str(dataset_root),
            "patients": {},
            "scan_folders": {},
        }

        for key_name, folder_name in dataset_spec["subfolders"].items():
            folder = dataset_root / folder_name
            dataset_meta["scan_folders"][key_name] = str(folder)
            scanned = scan_folder(folder, dataset_root, key_name)
            for patient_id, patient_meta in scanned.items():
                patient = dataset_meta["patients"].setdefault(patient_id, {
                    "patient_id": patient_id,
                    "raw_files": [],
                    "seg_files": [],
                    "png_files": [],
                    "groundtruth_png_files": [],
                    "roi_csv_files": [],
                    "rois_series_files": [],
                    "other_files": [],
                })
                for field_name, values in patient_meta.items():
                    if field_name == "patient_id":
                        continue
                    patient[field_name].extend(values)

        result["datasets"][dataset_key] = dataset_meta

    return result


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def require_nibabel() -> None:
    if nib is None:
        raise RuntimeError(
            "Please install nibabel before using centroid computation. "
            "Run: pip install nibabel"
        )


def compute_centroids(seg_path: Path, labels: Optional[Iterable[int]] = None) -> Dict[int, Dict[str, Any]]:
    """Compute voxel and world centroids for label masks in a segmentation file."""
    require_nibabel()
    labels = list(labels or DEFAULT_LABELS)

    seg_img = nib.load(str(seg_path))
    affine = seg_img.affine
    data = seg_img.get_fdata(dtype=np.float32)

    results: Dict[int, Dict[str, Any]] = {}
    for label in labels:
        coords = np.argwhere(data == label)
        if coords.size == 0:
            results[label] = {
                "label": label,
                "voxel_centroid": None,
                "world_centroid": None,
                "count": 0,
            }
            continue

        voxel_centroid = coords.mean(axis=0).tolist()
        world_centroid = (affine @ np.array([*voxel_centroid, 1.0]))[:3].tolist()
        results[label] = {
            "label": label,
            "count": int(coords.shape[0]),
            "voxel_centroid": [float(x) for x in voxel_centroid],
            "world_centroid": [float(x) for x in world_centroid],
        }

    return results


def build_batch_centroids(metadata: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Compute centroids for every patient in scanned metadata that has segmentation files."""
    batch: Dict[str, Any] = {
        "root": str(root),
        "files": [],
    }

    datasets = metadata.get("datasets", {})
    for dataset_key, dataset_meta in datasets.items():
        dataset_root = Path(dataset_meta["root"])
        for patient_id, patient_meta in dataset_meta.get("patients", {}).items():
            for seg_rel in patient_meta.get("seg_files", []):
                seg_path = dataset_root / seg_rel
                if not seg_path.exists():
                    batch["files"].append({
                        "dataset": dataset_key,
                        "patient_id": patient_id,
                        "seg_file": seg_rel,
                        "error": "missing segmentation file",
                    })
                    continue

                try:
                    centroids = compute_centroids(seg_path)
                    batch["files"].append({
                        "dataset": dataset_key,
                        "patient_id": patient_id,
                        "seg_file": str(seg_path),
                        "centroids": centroids,
                    })
                except Exception as exc:
                    batch["files"].append({
                        "dataset": dataset_key,
                        "patient_id": patient_id,
                        "seg_file": str(seg_path),
                        "error": str(exc),
                    })

    return batch


def run_pipeline() -> None:
    """Run the full pipeline: scan datasets and compute centroids."""
    try:
        root = Path(CONFIG["root_path"]).expanduser().resolve()
        output_dir = Path(CONFIG["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using root directory: {root}")
        print(f"Output directory: {output_dir}")

        metadata_path = output_dir / "dataset_metadata.json"
        centroids_batch_path = output_dir / "centroids_batch.json"

        # Step 1: Scan datasets
        if CONFIG["scan_datasets"]:
            print("\n[Step 1] Scanning datasets...")
            metadata = scan_datasets(root)
            save_json(metadata, metadata_path)
            print(f"✓ Wrote metadata for {len(metadata['datasets'])} dataset(s)")
            print(f"  Saved to: {metadata_path}")
        else:
            print("\n[Step 1] Loading existing metadata...")
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            print(f"✓ Loaded metadata from: {metadata_path}")

        # Step 2: Compute centroids
        if CONFIG["compute_centroids"]:
            print("\n[Step 2] Computing centroids for all segmentations...")
            batch = build_batch_centroids(metadata, root)
            save_json(batch, centroids_batch_path)
            print(f"✓ Computed centroids for {len(batch['files'])} files")
            print(f"  Saved to: {centroids_batch_path}")

        print("\n✓ Pipeline complete!")
        print(f"\nOutput files:")
        print(f"  - {metadata_path}")
        if CONFIG["compute_centroids"]:
            print(f"  - {centroids_batch_path}")
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting script...")
    run_pipeline()
