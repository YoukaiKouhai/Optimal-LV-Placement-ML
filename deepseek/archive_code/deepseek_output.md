Step 0: Setup & Understanding Your Data
First, install required libraries (run in terminal or colab):
```bash
pip install nibabel numpy matplotlib scipy scikit-image pandas SimpleITK opencv-python-headless
```
For ML later, you'll also need:

```bash
pip install torch torchvision monai  # if you want deep learning
```
But don't rush to ML yet. First, understand what you have.

Your Two Folder Structures
From your description:

Folder	                                What's inside
BENG280C_pacing_lead_data_1st20/	    17 patients? Contains seg_nii, png, img_nii, and AUH-2024-HCT2-rois (manual CSVs/bullseye). This is your ground truth set.

HCT2_lead_segmentation_training/	    More patients, similar subfolders, but maybe without manual CSVs. This is unprocessed raw set.

Ground truth = you have the _seg.nii.gz files (expert labeled leads/landmarks) AND/OR the manual CSV coordinates.
Unprocessed = only raw img.nii.gz, no segmentation or manual clicks.

You said ~20% ground truth, 80% unprocessed. That's ~4-5 patients with labels, the rest without. That's very little for deep learning from scratch. But we can still try classical methods or use the labeled ones as a testbed.

---

Step 1: Build a Patient Dictionary & Save to JSON
This script will scan both root folders, match files per patient, and create a structured dictionary. It will also mark if ground truth (segmentation or manual CSV) exists.

```python
import os
import json
import re
from pathlib import Path

def build_patient_dictionary(root_paths):
    """
    root_paths: list of two folder paths, e.g. 
        ['/path/to/BENG280C_pacing_lead_data_1st20',
         '/path/to/HCT2_lead_segmentation_training']
    Returns a dictionary keyed by patient ID, with file paths and ground_truth flag.
    """
    patient_dict = {}
    
    for base_path in root_paths:
        base = Path(base_path)
        
        # Find raw image nii files
        img_dir = base / "HCT2_img_nii"
        seg_dir = base / "HCT2_leads_seg_nii"
        png_dir = base / "HCT2_leads_png"
        rois_dir = base / "AUH-2024-HCT2-rois"
        
        # Some folders might not exist in the second structure, handle gracefully
        if not img_dir.exists():
            print(f"Warning: {img_dir} not found")
            continue
            
        for img_file in img_dir.glob("*_HCT2.nii.gz"):
            # Extract patient ID: e.g., "10001_HCT2.nii.gz" -> "10001"
            patient_id = img_file.stem.replace("_HCT2", "")
            
            # Initialize entry
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
            
            # Find matching segmentation
            seg_file = seg_dir / f"{patient_id}_HCT2_leads_seg.nii.gz" if seg_dir.exists() else None
            if seg_file and seg_file.exists():
                patient_dict[patient_id]["seg_nii"] = str(seg_file)
                patient_dict[patient_id]["has_ground_truth"] = True
            
            # Find PNG folder (might be named "10001_HCT2" inside HCT2_leads_png)
            if png_dir.exists():
                png_subdir = png_dir / f"{patient_id}_HCT2"
                if png_subdir.exists():
                    patient_dict[patient_id]["png_folder"] = str(png_subdir)
            
            # Find ROI files (manual ground truth)
            if rois_dir.exists():
                csv_file = rois_dir / f"._{patient_id}_leads.csv"
                if csv_file.exists():
                    patient_dict[patient_id]["rois_csv"] = str(csv_file)
                    patient_dict[patient_id]["has_ground_truth"] = True
                
                bull_csv = rois_dir / f"._{patient_id}_leads_bullseye.csv"
                if bull_csv.exists():
                    patient_dict[patient_id]["rois_bullseye_csv"] = str(bull_csv)
                
                bull_png = rois_dir / f"._{patient_id}_leads_bullseye.png"
                if bull_png.exists():
                    patient_dict[patient_id]["rois_bullseye_png"] = str(bull_png)
    
    return patient_dict

# Usage
root_folders = [
    "/path/to/BENG280C_pacing_lead_data_1st20",
    "/path/to/HCT2_lead_segmentation_training"
]   # <--- CHANGE TO YOUR ACTUAL PATHS

patient_dict = build_patient_dictionary(root_folders)

# Save to JSON for later use
with open("patient_data_index.json", "w") as f:
    json.dump(patient_dict, f, indent=2)

print(f"Total patients indexed: {len(patient_dict)}")
ground_truth_count = sum(1 for v in patient_dict.values() if v["has_ground_truth"])
print(f"With ground truth: {ground_truth_count}")
print(f"Without ground truth: {len(patient_dict) - ground_truth_count}")
```
This gives you a single source of truth about all your data. Any other script can load this JSON and know exactly where to find each patient's files.

---

Step 2: Extract 3D Coordinates from Segmentation Masks
Now let's write a function that takes a segmentation NIfTI and returns real-world (X, Y, Z) coordinates for each label (4001-4008). This is critical for comparing with manual CSVs.

```python
import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass

def get_label_centroids(seg_nii_path, label_list=None):
    """
    Load segmentation NIfTI, compute centroid (voxel) then transform to world coordinates (mm).
    
    Args:
        seg_nii_path: path to .nii.gz segmentation file
        label_list: list of integers, e.g. [4001,4002,4003,4004,4005,4006,4007,4008]
                   If None, returns all unique labels present.
    
    Returns:
        centroids_world: dict {label: (x,y,z) in mm}
        affine: the 4x4 affine matrix
    """
    img = nib.load(seg_nii_path)
    data = img.get_fdata()
    affine = img.affine
    
    if label_list is None:
        label_list = np.unique(data).astype(int).tolist()
        # remove background 0
        label_list = [l for l in label_list if l != 0]
    
    centroids_world = {}
    for label in label_list:
        # Find all voxels with this label
        mask = (data == label)
        if not np.any(mask):
            print(f"Label {label} not found in {seg_nii_path}")
            continue
        
        # Voxel centroid (i,j,k)
        com_vox = center_of_mass(mask)  # returns (z,y,x)? Actually center_of_mass returns order as array dimensions: (dim0, dim1, dim2)
        # Nibabel uses (x,y,z) for affine but data is (z,y,x)? Let's be safe: convert to (i,j,k) indexes
        # center_of_mass returns (r,c,s) where r is first dimension (usually z in medical images)
        # We'll just apply affine to the voxel coordinate as (i,j,k,1)
        # Standard: affine transforms (x,y,z,1) where x = column, y = row, z = slice
        # But our com_vox order depends on how data was loaded. Let's use the proper method:
        # Use nibabel's apply_affine after converting to RAS+ world coordinates.
        # Simpler: get the centroid in voxel space as (x,y,z) by reordering.
        # Data shape is (slices, rows, cols) usually. So:
        #   dim0 = slice (z), dim1 = row (y), dim2 = col (x)
        z, y, x = com_vox
        vox_coord = np.array([x, y, z, 1])   # (x,y,z,1) for affine multiplication
        world_coord = affine @ vox_coord
        centroids_world[label] = world_coord[:3]   # (x,y,z) in mm
    
    return centroids_world, affine

# Example usage on a ground truth patient
example_patient = list(patient_dict.keys())[0]
seg_path = patient_dict[example_patient]["seg_nii"]
if seg_path:
    centroids, _ = get_label_centroids(seg_path)
    print(f"Patient {example_patient} centroids (mm):")
    for label, coord in centroids.items():
        print(f"  Label {label}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
```

---

Step 3: Compare with Manual CSV (Validation)
Manual CSVs (from Horos) contain points like LL1 (LV lead distal), RV1, APEX, BASE, ANT. You need to map these to your label IDs.

Create a mapping:

Manual point	Likely label
LL1	            4001 (LV distal)
LL2	            4002
LL3	            4003
LL4	            4004 (LV proximal)
RV1	            4005 (RV distal)
RV2	            4006
APEX	        4007
BASE	        4008
ANT	            (not in seg labels, used for rotation)
Now compute Euclidean distance between your extracted centroid and the manual CSV coordinate.

```python
import pandas as pd

def load_manual_csv(csv_path):
    """Load the manual CSV (with columns like LL1_x, LL1_y, LL1_z)"""
    df = pd.read_csv(csv_path)
    # The format may vary. Usually they have columns: Name, X, Y, Z
    # Or separate columns per coordinate. We'll inspect.
    # Let's assume typical Horos ROI export has rows for each point.
    # If not, you'll need to adapt.
    # I'll write a generic parser based on typical structure.
    points = {}
    # Try to read as long format (Name, X, Y, Z)
    if 'Name' in df.columns and 'X' in df.columns:
        for _, row in df.iterrows():
            points[row['Name']] = (row['X'], row['Y'], row['Z'])
    else:
        # Try wide format: LL1_x, LL1_y, LL1_z
        for col in df.columns:
            if col.endswith('_x'):
                base = col[:-2]
                if f"{base}_y" in df.columns and f"{base}_z" in df.columns:
                    points[base] = (df[col].values[0], df[f"{base}_y"].values[0], df[f"{base}_z"].values[0])
    return points

def calculate_error(extracted_centroids, manual_points, label_to_name_map):
    """
    label_to_name_map: e.g., {4001: 'LL1', 4002: 'LL2', ..., 4007: 'APEX', 4008: 'BASE'}
    """
    errors = {}
    for label, centroid in extracted_centroids.items():
        name = label_to_name_map.get(label)
        if name and name in manual_points:
            manual = manual_points[name]
            dist = np.linalg.norm(np.array(centroid) - np.array(manual))
            errors[name] = dist
    return errors

# Example mapping
label_to_name = {
    4001: 'LL1', 4002: 'LL2', 4003: 'LL3', 4004: 'LL4',
    4005: 'RV1', 4006: 'RV2',
    4007: 'APEX', 4008: 'BASE'
}

# For a patient with both seg and manual CSV
pid = example_patient
if patient_dict[pid]["seg_nii"] and patient_dict[pid]["rois_csv"]:
    centroids, _ = get_label_centroids(patient_dict[pid]["seg_nii"], label_list=list(label_to_name.keys()))
    manual = load_manual_csv(patient_dict[pid]["rois_csv"])
    errors = calculate_error(centroids, manual, label_to_name)
    print(f"Errors for {pid} (mm): {errors}")
    mean_error = np.mean(list(errors.values()))
    print(f"Mean error: {mean_error:.2f} mm")
```
If mean error is < 2 mm, your extraction pipeline is correct. If larger, check affine transformation or CSV coordinate system (RAS vs LPS).

---

Step 4: Visualize in 3D (Stacked Slices)
You wanted a 3D graph with slices stacked. Here's how to visualize raw CT with segmentation overlay and centroids as markers.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_slices(ct_nii_path, seg_nii_path=None, centroids=None, slice_indices=[50,60,70]):
    """
    Show three orthogonal slices through the 3D volume with optional segmentation overlay.
    """
    ct_img = nib.load(ct_nii_path)
    ct_data = ct_img.get_fdata()
    
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    titles = ['Axial (Z)', 'Coronal (Y)', 'Sagittal (X)']
    
    # Get center slices if indices not provided
    if slice_indices is None:
        slice_indices = [ct_data.shape[0]//2, ct_data.shape[1]//2, ct_data.shape[2]//2]
    
    # Axial (z slice)
    axes[0].imshow(ct_data[slice_indices[0], :, :], cmap='gray')
    axes[0].set_title(f'Axial at z={slice_indices[0]}')
    # Coronal (y slice)
    axes[1].imshow(ct_data[:, slice_indices[1], :], cmap='gray')
    axes[1].set_title(f'Coronal at y={slice_indices[1]}')
    # Sagittal (x slice)
    axes[2].imshow(ct_data[:, :, slice_indices[2]], cmap='gray')
    axes[2].set_title(f'Sagittal at x={slice_indices[2]}')
    
    # Overlay segmentation if provided
    if seg_nii_path:
        seg_img = nib.load(seg_nii_path)
        seg_data = seg_img.get_fdata()
        # Create a mask for any non-zero label
        mask = seg_data > 0
        # Overlay with transparency (alpha)
        for i, ax in enumerate(axes):
            if i == 0:
                overlay = mask[slice_indices[0], :, :]
            elif i == 1:
                overlay = mask[:, slice_indices[1], :]
            else:
                overlay = mask[:, :, slice_indices[2]]
            ax.imshow(overlay, cmap='jet', alpha=0.4)
    
    # Plot centroids if provided (convert to voxel coordinates first)
    if centroids:
        # We need the inverse affine to map world -> voxel
        inv_aff = np.linalg.inv(ct_img.affine)
        for label, world_xyz in centroids.items():
            vox = inv_aff @ [world_xyz[0], world_xyz[1], world_xyz[2], 1]
            xv, yv, zv = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
            # Plot in each slice if it passes through
            # For simplicity, we just mark in all three views
            axes[0].scatter(yv, xv, color='red', s=50, marker='x')  # yv is column, xv is row? Adjust based on orientation
            axes[1].scatter(zv, xv, color='red', s=50, marker='x')
            axes[2].scatter(zv, yv, color='red', s=50, marker='x')
    
    plt.tight_layout()
    plt.show()

# Example: visual check for one patient
pid = example_patient
raw_path = patient_dict[pid]["raw_nii"]
seg_path = patient_dict[pid]["seg_nii"]
if seg_path:
    centroids, _ = get_label_centroids(seg_path)
    plot_3d_slices(raw_path, seg_path, centroids)
```
For a true 3D scatter plot of leads inside the heart volume:

```python
def plot_3d_scatter(ct_nii_path, centroids):
    ct_img = nib.load(ct_nii_path)
    data = ct_img.get_fdata()
    # Create a meshgrid for the bounding box (optional)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the centroid points
    for label, (x,y,z) in centroids.items():
        ax.scatter(x, y, z, s=50, label=f'Label {label}')
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()
    plt.title('Lead Centroids in World Coordinates')
    plt.show()

# Usage
plot_3d_scatter(raw_path, centroids)
```

---

Step 5: Machine Learning – Is It Feasible?
You have ~5 labeled patients (if 20% of 25 = 5). That is far too few to train a deep learning model from scratch. Even overfitting a tiny model on 5 patients tells you almost nothing about generalization – it will simply memorize those 5.

Realistic options:
Classical computer vision (rule-based) – Might work because leads are very bright in CT (>2000 HU). You can segment them using thresholding + connected components + shape filtering. No ML needed.

Use segmentation masks as ground truth – But you already have them. So why ML? You'd be training a model to replicate what you already have. That's only useful if you want to apply it to the 80% unlabeled data. But training on 5 patients will likely fail on new unseen patients.

Transfer learning / fine-tuning a pretrained 3D U-Net – Still needs at least 50-100 labeled cases. You could try data augmentation (rotation, scaling, elastic deformations) to artificially increase dataset size, but the underlying anatomy variability is huge.

Active learning – Manually label a few more of your unprocessed patients (maybe 10-20) using your extraction code as a starting point, then train.

What your friend meant by "overfit a model"
They likely suggest: take your 5 labeled patients, train a small model (e.g., a 3D CNN to predict lead coordinates directly) and see if it can achieve near-zero training error. If it can't even memorize 5 patients, then the problem is too hard. If it can, it doesn't guarantee generalization, but at least the data contains the signal.

Here's a minimal overfitting test using a simple regression network (Monai + PyTorch). Don't expect it to work for real – it's just a diagnostic.

```python
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from monai.transforms import LoadImage, EnsureChannelFirst, Compose, ScaleIntensity, RandRotate, RandZoom
from monai.data import Dataset, DataLoader

# This is just a sketch – you'll need to adapt to your actual data size
# But I strongly advise you to first try classical methods before ML.

# Classical method for lead detection (no training):
def detect_leads_by_threshold(raw_nii_path, hu_threshold=2000, min_volume_mm3=5):
    """
    Simple threshold-based lead detection.
    Returns list of centroids in mm for all blobs above threshold.
    """
    import nibabel as nib
    from skimage import measure
    img = nib.load(raw_nii_path)
    data = img.get_fdata()
    affine = img.affine
    
    # Threshold for metal
    binary = (data > hu_threshold).astype(np.uint8)
    # Connected components
    labeled = measure.label(binary, connectivity=2)
    props = measure.regionprops(labeled)
    
    # Filter by volume (convert voxels to mm^3)
    vox_vol = np.prod(img.header.get_zooms())  # voxel volume in mm^3
    centroids = []
    for prop in props:
        if prop.area * vox_vol >= min_volume_mm3:
            # Centroid in voxel coordinates
            cy, cx, cz = prop.centroid  # careful: order might be (z,y,x)
            vox_coord = np.array([cx, cy, cz, 1])
            world = affine @ vox_coord
            centroids.append(world[:3])
    return centroids

# Try this on a patient and compare to ground truth segmentation centroids
# Often this works surprisingly well for bright leads.
```
Recommendation for your project:
First, manually verify that the segmentation masks are correct. Use the visualization above to check.

Extract all ground truth coordinates from the 20% labeled patients using the centroid method. Save as CSV for each patient.

Implement a threshold+connected components pipeline to detect leads automatically on raw images. Compare its detections to the ground truth. If it works decently (error < 5mm), you already have an automatic method without ML.

Only if threshold method fails badly, consider supervised ML. Then you'll need to label at least 50 more patients manually (or use your segmentation masks as labels and artificially augment).

For the unprocessed 80%, if you cannot label them, you cannot train a model. But you can apply your threshold method or use registration to a template heart.

Step 6: Putting It All Together – Your Action Plan
```text
Week 1: Data organization
- Run Step 1 to build patient_data_index.json
- Verify that all ground truth patients have both seg_nii and rois_csv

Week 2: Coordinate extraction & validation
- Run Step 2 to get centroids from seg_nii
- Run Step 3 to compare with manual CSV → compute mean error
- If error > 2mm, fix affine or CSV coordinate interpretation

Week 3: Visualization & threshold method
- Run Step 4 to visually confirm centroids align with actual leads
- Implement threshold+connected components on raw images
- Compare threshold results to ground truth (centroids from seg)

Week 4: Decide on ML
- If threshold method achieves < 3mm error → use it, no ML needed
- Else, if you have > 50 labeled patients, try a 3D U-Net (Monai)
- Else, accept that ML is not feasible and focus on improving classical method

Bonus: Normalization (heart axis & bullseye map)
- Use apex (4007) and base (4008) to define long axis
- Project each LV lead onto that axis to get height (0=apex,1=base)
- Compute radial angle relative to RV lead or anterior marker (ANT from CSV)
```