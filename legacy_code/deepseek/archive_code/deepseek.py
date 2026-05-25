import os
import json
import re
from pathlib import Path

def build_patient_dictionary(root_paths):

    #root_paths: list of two folder paths, e.g. 
    #    [r"C:\Users\ayw005\Desktop\BENG 280C Project\BENG280C_pacing_lead_data_1st20",
    #     r"C:\Users\ayw005\Desktop\BENG 280C Project\HCT2_lead_segmentation_training"]
    #Returns a dictionary keyed by patient ID, with file paths and ground_truth flag.

    patient_dict = {}
    
    for base_path in root_paths:
        base = Path(base_path)
        
        # Find raw image nii files
        img_dir = base / "HCT2_img_nii"
        seg_dir = base / "HCT2_leads_seg_nii"
        png_dir = base / "HCT2_leads_png"
        png_dir_alt = base / "HCT2_leads_groundtruth_png"  # Alternative name in second folder
        rois_dir = base / "AUH-2024-HCT2-rois"
        
        # Some folders might not exist in the second structure, handle gracefully
        if not img_dir.exists():
            print(f"Warning: {img_dir} not found")
            continue
        
        # Handle both naming patterns: *_HCT2.nii.gz and *_HCT2_img.nii.gz
        img_files = list(img_dir.glob("*_HCT2.nii.gz")) + list(img_dir.glob("*_HCT2_img.nii.gz"))
            
        for img_file in img_files:
            # Extract patient ID from either pattern
            # "10001_HCT2.nii.gz" -> "10001" or "10001_HCT2_img.nii.gz" -> "10001"
            patient_id = re.match(r"^(\d+)", img_file.name).group(1)
            
            # Skip if already added (shouldn't happen, but safety check)
            if patient_id in patient_dict:
                continue
            
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
            
            # Find matching segmentation - handle both naming patterns
            # Folder 1: 10001_HCT2_HCT2_leads_seg.nii.gz, Folder 2: 10001_HCT2_leads_seg.nii.gz
            if seg_dir.exists():
                seg_file = seg_dir / f"{patient_id}_HCT2_leads_seg.nii.gz"
                if seg_file.exists():
                    patient_dict[patient_id]["seg_nii"] = str(seg_file)
                    patient_dict[patient_id]["has_ground_truth"] = True
                else:
                    # Try folder 1 pattern with extra _HCT2
                    seg_file_alt = seg_dir / f"{patient_id}_HCT2_HCT2_leads_seg.nii.gz"
                    if seg_file_alt.exists():
                        patient_dict[patient_id]["seg_nii"] = str(seg_file_alt)
                        patient_dict[patient_id]["has_ground_truth"] = True
            
            # Find PNG folder (might be named "10001_HCT2" inside HCT2_leads_png or HCT2_leads_groundtruth_png)
            if png_dir.exists():
                png_subdir = png_dir / f"{patient_id}_HCT2"
                if png_subdir.exists():
                    patient_dict[patient_id]["png_folder"] = str(png_subdir)
            if png_dir_alt.exists() and not patient_dict[patient_id]["png_folder"]:
                png_subdir = png_dir_alt / f"{patient_id}_HCT2"
                if png_subdir.exists():
                    patient_dict[patient_id]["png_folder"] = str(png_subdir)
            
            # Find ROI files (manual ground truth) - handle both with and without leading dot
            if rois_dir.exists():
                # Try pattern without leading dot first (10001_leads.csv)
                csv_file = rois_dir / f"{patient_id}_leads.csv"
                if csv_file.exists():
                    patient_dict[patient_id]["rois_csv"] = str(csv_file)
                    patient_dict[patient_id]["has_ground_truth"] = True
                else:
                    # Try with leading dot (._10001_leads.csv)
                    csv_file_alt = rois_dir / f"._{patient_id}_leads.csv"
                    if csv_file_alt.exists():
                        patient_dict[patient_id]["rois_csv"] = str(csv_file_alt)
                        patient_dict[patient_id]["has_ground_truth"] = True
                
                # Bullseye CSV - try both patterns
                bull_csv = rois_dir / f"{patient_id}_leads_bullseye.csv"
                if bull_csv.exists():
                    patient_dict[patient_id]["rois_bullseye_csv"] = str(bull_csv)
                else:
                    bull_csv_alt = rois_dir / f"._{patient_id}_leads_bullseye.csv"
                    if bull_csv_alt.exists():
                        patient_dict[patient_id]["rois_bullseye_csv"] = str(bull_csv_alt)
                
                # Bullseye PNG - try both patterns
                bull_png = rois_dir / f"{patient_id}_leads_bullseye.png"
                if bull_png.exists():
                    patient_dict[patient_id]["rois_bullseye_png"] = str(bull_png)
                else:
                    bull_png_alt = rois_dir / f"._{patient_id}_leads_bullseye.png"
                    if bull_png_alt.exists():
                        patient_dict[patient_id]["rois_bullseye_png"] = str(bull_png_alt)
    
    return patient_dict

# Usage
root_folders = [r"C:\Users\ayw005\Desktop\BENG 280C Project\BENG280C_pacing_lead_data_1st20",  r"C:\Users\ayw005\Desktop\BENG 280C Project\HCT2_lead_segmentation_training"]   # <--- CHANGE TO YOUR ACTUAL PATHS

patient_dict = build_patient_dictionary(root_folders)

# Save to JSON for later use
with open("patient_data_index.json", "w") as f:
    json.dump(patient_dict, f, indent=2)

print(f"Total patients indexed: {len(patient_dict)}")
ground_truth_count = sum(1 for v in patient_dict.values() if v["has_ground_truth"])
print(f"With ground truth: {ground_truth_count}")
print(f"Without ground truth: {len(patient_dict) - ground_truth_count}")

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
    # Load segmentation
    seg_img = nib.load(seg_nii_path)
    seg_data = seg_img.get_fdata()
    affine = seg_img.affine
    
    # Get unique labels if not provided
    if label_list is None:
        label_list = np.unique(seg_data).astype(int).tolist()
        label_list = [l for l in label_list if l != 0]
    
    centroids_world = {}
    centroids_voxel = {}
    
    for label in label_list:
        # Find all voxels with this label
        mask = (seg_data == label)
        if not np.any(mask):
            print(f"Warning: Label {label} not found in {seg_nii_path}")
            continue
        
        # Calculate centroid in voxel space
        # center_of_mass returns (z, y, x) in that order
        com_z, com_y, com_x = center_of_mass(mask)
        
        # Store voxel centroid as (x, y, z) for easier use
        voxel_centroid = (com_x, com_y, com_z)
        centroids_voxel[label] = voxel_centroid
        
        # Prepare for affine transformation: NIfTI expects (x, y, z, 1)
        voxel_homog = np.array([com_x, com_y, com_z, 1])
        
        # Apply affine to get world coordinates
        world_coords = affine @ voxel_homog
        
        # Store world coordinates (x, y, z)
        centroids_world[label] = tuple(world_coords[:3])
    
    return centroids_world, centroids_voxel, affine


# Alternative: Using regionprops if labels are connected components
def get_label_centroids_regionprops(seg_nii_path, label_list=None):
    """
    Alternative method using regionprops (better for disconnected components).
    """
    from skimage import measure
    
    seg_img = nib.load(seg_nii_path)
    seg_data = seg_img.get_fdata().astype(int)
    affine = seg_img.affine
    
    # Label connected components if needed
    labeled = measure.label(seg_data)
    props = measure.regionprops(labeled)
    
    centroids_world = {}
    centroids_voxel = {}
    
    for prop in props:
        label = seg_data[prop.coords[0][0], prop.coords[0][1], prop.coords[0][2]]
        
        if label_list and label not in label_list:
            continue
        
        # Get centroid in voxel coordinates (z, y, x)
        cz, cy, cx = prop.centroid
        
        # Convert to (x, y, z, 1) for affine
        voxel_coords = np.array([cx, cy, cz, 1])
        world_coords = affine @ voxel_coords
        
        centroids_voxel[label] = (int(round(cx)), int(round(cy)), int(round(cz)))
        centroids_world[label] = tuple(world_coords[:3])
    
    return centroids_world, centroids_voxel, affine

# List all patient IDs that have both segmentation and manual CSV
valid_patients = []
for pid, info in patient_dict.items():
    if info["seg_nii"] and info["rois_csv"]:
        valid_patients.append(pid)

print("Patients with ground truth (seg + manual CSV):", valid_patients)

# Choose the first one as example
if valid_patients:
    example_patient = valid_patients[0]
    print(f"\nUsing patient: {example_patient}")
else:
    print("No patient with both seg_nii and rois_csv found.")

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

# Label mapping
label_to_name = {
    4001: 'LL1', 4002: 'LL2', 4003: 'LL3', 4004: 'LL4',
    4005: 'RV1', 4006: 'RV2', 4007: 'APEX', 4008: 'BASE'
}

# Find patients with both seg and manual CSV
valid_patients = [pid for pid, info in patient_dict.items() 
                  if info["seg_nii"] and info["rois_csv"]]

if not valid_patients:
    print("No valid patient found.")
else:
    example_patient = valid_patients[0]
    print(f"Processing {example_patient}...")
    
    # Extract centroids
    centroids_world, centroids_voxel, affine = get_label_centroids(
        patient_dict[example_patient]["seg_nii"],
        label_list=list(label_to_name.keys())
    )
    
    # Load manual CSV
    manual = load_manual_csv(patient_dict[example_patient]["rois_csv"])
    
    # Calculate errors
    errors = calculate_error(centroids_world, manual, label_to_name)
    
    print("\nErrors (mm):")
    for name, err in errors.items():
        print(f"  {name}: {err:.2f}")
    print(f"Mean error: {np.mean(list(errors.values())):.2f} mm")

# For the chosen patient
if example_patient and patient_dict[example_patient]["seg_nii"] and patient_dict[example_patient]["rois_csv"]:
    # Unpack all three returned values
    centroids_world, centroids_voxel, affine = get_label_centroids(
        patient_dict[example_patient]["seg_nii"], 
        label_list=list(label_to_name.keys())
    )
    
    manual = load_manual_csv(patient_dict[example_patient]["rois_csv"])
    errors = calculate_error(centroids_world, manual, label_to_name)
    
    print(f"\nErrors for {example_patient} (mm):")
    for name, err in errors.items():
        print(f"  {name}: {err:.2f} mm")
    
    mean_error = np.mean(list(errors.values()))
    print(f"Mean error: {mean_error:.2f} mm")

# Check what labels exist in the segmentation file
import nibabel as nib

seg_path = patient_dict[example_patient]["seg_nii"]
seg_img = nib.load(seg_path)
seg_data = seg_img.get_fdata()
unique_labels = np.unique(seg_data)
print(f"Unique labels in segmentation: {unique_labels}")
print(f"Labels we're looking for: {list(label_to_name.keys())}")

# Check which labels from our list are actually present
found_labels = [l for l in label_to_name.keys() if l in unique_labels]
missing_labels = [l for l in label_to_name.keys() if l not in unique_labels]
print(f"\nFound labels: {found_labels}")
print(f"Missing labels: {missing_labels}")

# Check what's in the manual CSV
import pandas as pd
csv_path = patient_dict[example_patient]["rois_csv"]
df = pd.read_csv(csv_path, header=None)  # No header in your file!
print(f"\nManual CSV content (first few rows):")
print(df.head())

# Let's properly parse your CSV (it has no header)
def load_manual_csv_corrected(csv_path):
    """Parse your specific CSV format (no header, columns: Name, X, Y, Z)"""
    df = pd.read_csv(csv_path, header=None)
    points = {}
    for _, row in df.iterrows():
        name = row[0].strip()  # First column is the label name
        x, y, z = float(row[1]), float(row[2]), float(row[3])
        points[name] = (x, y, z)
    return points

manual_points = load_manual_csv_corrected(csv_path)
print(f"\nManual points loaded: {list(manual_points.keys())}")

# Now check the mapping between segmentation labels and manual names
print("\nExpected mapping:")
for seg_label, manual_name in label_to_name.items():
    if manual_name in manual_points:
        print(f"  Label {seg_label} -> {manual_name}: FOUND in manual CSV")
    else:
        print(f"  Label {seg_label} -> {manual_name}: NOT found in manual CSV")

import nibabel as nib
import numpy as np

def diagnose_nifti_coordinates(ct_nii_path, seg_nii_path):
    """Diagnose coordinate system issues"""
    
    # Load images
    ct_img = nib.load(ct_nii_path)
    seg_img = nib.load(seg_nii_path)
    
    print("=" * 60)
    print("DIAGNOSTIC INFORMATION")
    print("=" * 60)
    
    # 1. Image dimensions
    print(f"\nCT Image dimensions (z,y,x): {ct_img.shape}")
    print(f"Segmentation dimensions (z,y,x): {seg_img.shape}")
    
    # 2. Affine matrix
    print(f"\nAffine matrix:\n{ct_img.affine}")
    
    # 3. Header info
    print(f"\nVoxel sizes (mm): {ct_img.header.get_zooms()}")
    
    # 4. Coordinate system
    print(f"\nCoordinate system: {ct_img.header.get_sform()}")
    
    # 5. Check a sample voxel to world transformation
    # Take the center voxel
    center_vox = np.array([ct_img.shape[0]//2, ct_img.shape[1]//2, ct_img.shape[2]//2, 1])
    center_world = ct_img.affine @ center_vox
    print(f"\nCenter voxel {center_vox[:3]} -> World coordinates: ({center_world[0]:.1f}, {center_world[1]:.1f}, {center_world[2]:.1f})")
    
    # 6. Check the range of world coordinates for all corners
    corners = [
        [0, 0, 0, 1],
        [0, 0, ct_img.shape[2]-1, 1],
        [0, ct_img.shape[1]-1, 0, 1],
        [0, ct_img.shape[1]-1, ct_img.shape[2]-1, 1],
        [ct_img.shape[0]-1, 0, 0, 1],
        [ct_img.shape[0]-1, 0, ct_img.shape[2]-1, 1],
        [ct_img.shape[0]-1, ct_img.shape[1]-1, 0, 1],
        [ct_img.shape[0]-1, ct_img.shape[1]-1, ct_img.shape[2]-1, 1],
    ]
    
    print("\nWorld coordinate ranges:")
    world_x = []
    world_y = []
    world_z = []
    for corner in corners:
        world = ct_img.affine @ corner
        world_x.append(world[0])
        world_y.append(world[1])
        world_z.append(world[2])
    
    print(f"X (world): {min(world_x):.1f} to {max(world_x):.1f} mm")
    print(f"Y (world): {min(world_y):.1f} to {max(world_y):.1f} mm")
    print(f"Z (world): {min(world_z):.1f} to {max(world_z):.1f} mm")
    
    # 7. Check your centroid world coordinates against this range
    print("\n" + "=" * 60)
    print("CHECKING YOUR CENTROIDS AGAINST WORLD BOUNDS")
    print("=" * 60)
    
    # Get a few centroids from segmentation (first few labels)
    seg_data = seg_img.get_fdata()
    unique_labels = np.unique(seg_data)
    unique_labels = unique_labels[unique_labels > 0][:3]  # first 3 non-zero labels
    
    for label in unique_labels:
        mask = seg_data == label
        if np.any(mask):
            # Get voxel coordinates of this label
            voxels = np.argwhere(mask)
            vox_center = np.mean(voxels, axis=0)
            
            # Add homogeneous coordinate
            vox_homog = np.append(vox_center, 1)
            world_from_vox = ct_img.affine @ vox_homog
            
            print(f"\nLabel {label}:")
            print(f"  Voxel center from segmentation: ({vox_center[0]:.1f}, {vox_center[1]:.1f}, {vox_center[2]:.1f})")
            print(f"  World coordinates from affine: ({world_from_vox[0]:.1f}, {world_from_vox[1]:.1f}, {world_from_vox[2]:.1f})")
    
    return ct_img, seg_img

# Run diagnostic on your patient
pid = example_patient
raw_path = patient_dict[pid]["raw_nii"]
seg_path = patient_dict[pid]["seg_nii"]

if seg_path:
    ct_img, seg_img = diagnose_nifti_coordinates(raw_path, seg_path)

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def plot_3d_slices_fixed(ct_nii_path, seg_nii_path=None, centroids=None):
    """
    CORRECTED VERSION: Shows three orthogonal slices with proper coordinate mapping.
    
    Args:
        ct_nii_path: path to raw CT NIfTI
        seg_nii_path: path to segmentation NIfTI (optional)
        centroids: dict of {label: (x,y,z)} in world coordinates (mm)
    """
    # Load data
    ct_img = nib.load(ct_nii_path)
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine
    
    # Get approximate center slices (indices in voxel space)
    z_center = ct_data.shape[0] // 2
    y_center = ct_data.shape[1] // 2  
    x_center = ct_data.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- 1. Axial view (looking down from top) - shows X (horizontal) vs Y (vertical) ---
    # Data is [z, y, x] so axial_slice = ct_data[z, :, :]
    axial_slice = ct_data[z_center, :, :]
    axes[0].imshow(axial_slice.T, cmap='gray', origin='lower')  # .T to swap x/y, origin='lower' for proper orientation
    axes[0].set_title(f'Axial View (Z = {z_center}) - looking from top')
    axes[0].set_xlabel('X (mm, left->right)')
    axes[0].set_ylabel('Y (mm, posterior->anterior)')
    
    # --- 2. Coronal view (looking from front) - shows X vs Z ---
    # coronal_slice = ct_data[:, y_center, :]
    coronal_slice = ct_data[:, y_center, :]
    axes[1].imshow(coronal_slice.T, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal View (Y = {y_center}) - looking from front')
    axes[1].set_xlabel('X (mm, left->right)')
    axes[1].set_ylabel('Z (mm, inferior->superior)')
    
    # --- 3. Sagittal view (looking from left side) - shows Y vs Z ---
    # sagittal_slice = ct_data[:, :, x_center]
    sagittal_slice = ct_data[:, :, x_center]
    axes[2].imshow(sagittal_slice, cmap='gray', origin='lower')
    axes[2].set_title(f'Sagittal View (X = {x_center}) - looking from left')
    axes[2].set_xlabel('Y (mm, posterior->anterior)')
    axes[2].set_ylabel('Z (mm, inferior->superior)')
    
    # Overlay segmentation if provided
    if seg_nii_path:
        seg_img = nib.load(seg_nii_path)
        seg_data = seg_img.get_fdata()
        mask = seg_data > 0
        
        # Overlay on each view
        # Axial overlay
        axial_mask = mask[z_center, :, :]
        axes[0].imshow(axial_mask.T, cmap='jet', alpha=0.3, origin='lower')
        
        # Coronal overlay
        coronal_mask = mask[:, y_center, :]
        axes[1].imshow(coronal_mask.T, cmap='jet', alpha=0.3, origin='lower')
        
        # Sagittal overlay
        sagittal_mask = mask[:, :, x_center]
        axes[2].imshow(sagittal_mask, cmap='jet', alpha=0.3, origin='lower')
    
    # Plot centroids if provided
    if centroids:
        for label, world_xyz in centroids.items():
            # Convert world coordinates (mm) to voxel coordinates (i,j,k)
            # Need to solve: world = affine @ [vox_x, vox_y, vox_z, 1]
            # So: vox = inv_affine @ [world_x, world_y, world_z, 1]
            inv_affine = np.linalg.inv(affine)
            vox_homog = inv_affine @ [world_xyz[0], world_xyz[1], world_xyz[2], 1]
            vox_x, vox_y, vox_z = int(round(vox_homog[0])), int(round(vox_homog[1])), int(round(vox_homog[2]))
            
            # Plot on axial view (z = vox_z, coordinates: x=vox_x, y=vox_y)
            axes[0].scatter(vox_x, vox_y, color='red', s=100, marker='o', linewidth=2, 
                           edgecolors='white', label=f'Label {label}' if label == list(centroids.keys())[0] else "")
            axes[0].text(vox_x, vox_y, f' {label}', color='red', fontsize=8)
            
            # Plot on coronal view (y = vox_y, coordinates: x=vox_x, z=vox_z)
            axes[1].scatter(vox_x, vox_z, color='red', s=100, marker='o', linewidth=2, edgecolors='white')
            axes[1].text(vox_x, vox_z, f' {label}', color='red', fontsize=8)
            
            # Plot on sagittal view (x = vox_x, coordinates: y=vox_y, z=vox_z)
            axes[2].scatter(vox_y, vox_z, color='red', s=100, marker='o', linewidth=2, edgecolors='white')
            axes[2].text(vox_y, vox_z, f' {label}', color='red', fontsize=8)
    
    # Add legends
    if centroids:
        axes[0].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


# Alternative: Simple 2D slice viewer with scroll (much more useful for debugging)
def view_all_slices_with_centroids(ct_nii_path, seg_nii_path=None, centroids=None):
    """
    Interactive slice viewer - allows you to scroll through all slices.
    BETTER for finding where your leads actually are!
    """
    ct_img = nib.load(ct_nii_path)
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine
    
    if seg_nii_path:
        seg_img = nib.load(seg_nii_path)
        seg_data = seg_img.get_fdata()
    
    # Convert centroids to voxel coordinates once
    centroids_vox = {}
    if centroids:
        inv_affine = np.linalg.inv(affine)
        for label, world_xyz in centroids.items():
            vox_homog = inv_affine @ [world_xyz[0], world_xyz[1], world_xyz[2], 1]
            centroids_vox[label] = (int(round(vox_homog[0])), int(round(vox_homog[1])), int(round(vox_homog[2])))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initial slice indices
    z_idx = ct_data.shape[0] // 2
    y_idx = ct_data.shape[1] // 2
    x_idx = ct_data.shape[2] // 2
    
    def update_axial():
        axes[0].clear()
        # Axial: show slice at z_idx
        slice_data = ct_data[z_idx, :, :]
        axes[0].imshow(slice_data.T, cmap='gray', origin='lower')
        
        if seg_nii_path:
            seg_slice = seg_data[z_idx, :, :]
            axes[0].imshow(seg_slice.T, cmap='jet', alpha=0.3, origin='lower')
        
        # Plot centroids that fall near this z slice (within 2 slices)
        for label, (vx, vy, vz) in centroids_vox.items():
            if abs(vz - z_idx) <= 2:
                axes[0].scatter(vx, vy, color='red', s=100, marker='o', linewidth=2, edgecolors='white')
                axes[0].text(vx, vy, f' {label}', color='red', fontsize=8)
        
        axes[0].set_title(f'Axial (Z={z_idx}) - Use scroll on this plot to change Z')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
    
    def update_coronal():
        axes[1].clear()
        # Coronal: show slice at y_idx
        slice_data = ct_data[:, y_idx, :]
        axes[1].imshow(slice_data.T, cmap='gray', origin='lower')
        
        if seg_nii_path:
            seg_slice = seg_data[:, y_idx, :]
            axes[1].imshow(seg_slice.T, cmap='jet', alpha=0.3, origin='lower')
        
        for label, (vx, vy, vz) in centroids_vox.items():
            if abs(vy - y_idx) <= 2:
                axes[1].scatter(vx, vz, color='red', s=100, marker='o', linewidth=2, edgecolors='white')
                axes[1].text(vx, vz, f' {label}', color='red', fontsize=8)
        
        axes[1].set_title(f'Coronal (Y={y_idx}) - Use scroll on this plot to change Y')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
    
    def update_sagittal():
        axes[2].clear()
        # Sagittal: show slice at x_idx
        slice_data = ct_data[:, :, x_idx]
        axes[2].imshow(slice_data, cmap='gray', origin='lower')
        
        if seg_nii_path:
            seg_slice = seg_data[:, :, x_idx]
            axes[2].imshow(seg_slice, cmap='jet', alpha=0.3, origin='lower')
        
        for label, (vx, vy, vz) in centroids_vox.items():
            if abs(vx - x_idx) <= 2:
                axes[2].scatter(vy, vz, color='red', s=100, marker='o', linewidth=2, edgecolors='white')
                axes[2].text(vy, vz, f' {label}', color='red', fontsize=8)
        
        axes[2].set_title(f'Sagittal (X={x_idx}) - Use scroll on this plot to change X')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
    
    def on_scroll(event, ax_name):
        nonlocal z_idx, y_idx, x_idx
        if ax_name == 'axial':
            z_idx = max(0, min(ct_data.shape[0]-1, z_idx + int(event.step)))
            update_axial()
        elif ax_name == 'coronal':
            y_idx = max(0, min(ct_data.shape[1]-1, y_idx + int(event.step)))
            update_coronal()
        elif ax_name == 'sagittal':
            x_idx = max(0, min(ct_data.shape[2]-1, x_idx + int(event.step)))
            update_sagittal()
        fig.canvas.draw_idle()
    
    # Connect scroll events
    def on_scroll_factory(ax_name):
        def handler(event):
            if event.inaxes == axes[0] and ax_name == 'axial':
                on_scroll(event, 'axial')
            elif event.inaxes == axes[1] and ax_name == 'coronal':
                on_scroll(event, 'coronal')
            elif event.inaxes == axes[2] and ax_name == 'sagittal':
                on_scroll(event, 'sagittal')
        return handler
    
    fig.canvas.mpl_connect('scroll_event', on_scroll_factory('axial'))
    fig.canvas.mpl_connect('scroll_event', on_scroll_factory('coronal'))
    fig.canvas.mpl_connect('scroll_event', on_scroll_factory('sagittal'))
    
    # Initial render
    update_axial()
    update_coronal()
    update_sagittal()
    
    plt.tight_layout()
    plt.show()


# Usage example:
print("Visualizing centroids on slices for patient:", example_patient)
pid = example_patient
raw_path = patient_dict[pid]["raw_nii"]
seg_path = patient_dict[pid]["seg_nii"]
if seg_path:
    # Unpack correctly: centroids_world, centroids_voxel, affine
    centroids_world, _, affine = get_label_centroids(seg_path)
    
    # Option 1: Static view
    plot_3d_slices_fixed(raw_path, seg_path, centroids_world)
    
    # Option 2: Interactive viewer
    view_all_slices_with_centroids(raw_path, seg_path, centroids_world)

# Verification
if seg_path:
    centroids_world, _, affine = get_label_centroids(seg_path)
    inv_aff = np.linalg.inv(affine)
    print("\nCentroid verification (world -> voxel):")
    for label, world_xyz in centroids_world.items():
        vox = inv_aff @ [world_xyz[0], world_xyz[1], world_xyz[2], 1]
        print(f"Label {label}: World=({world_xyz[0]:.1f}, {world_xyz[1]:.1f}, {world_xyz[2]:.1f}) -> Voxel=({int(vox[0])}, {int(vox[1])}, {int(vox[2])})")

print("Centroid verification (world to voxel coordinates):")

if seg_path:
    centroids, affine = get_label_centroids(seg_path)
    inv_aff = np.linalg.inv(affine)
    
    print("Centroid verification:")
    for label, world_xyz in centroids.items():
        vox = inv_aff @ [world_xyz[0], world_xyz[1], world_xyz[2], 1]
        print(f"Label {label}: World=({world_xyz[0]:.1f}, {world_xyz[1]:.1f}, {world_xyz[2]:.1f}) -> Voxel=({int(vox[0])}, {int(vox[1])}, {int(vox[2])})")

# Test the corrected function
if seg_path:
    print("\n" + "=" * 60)
    print("TESTING CORRECTED CENTROID EXTRACTION")
    print("=" * 60)
    
    # Method 1: center_of_mass
    centroids_world, centroids_voxel, affine = get_label_centroids(seg_path)
    
    print("\nCentroids from corrected method:")
    for label in sorted(centroids_world.keys()):
        world = centroids_world[label]
        vox = centroids_voxel[label]
        print(f"Label {label}: Voxel=({vox[0]}, {vox[1]}, {vox[2]}) -> World=({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f})")
    
    # Now visualize with corrected centroids
    print("\n" + "=" * 60)
    print("VISUALIZING WITH CORRECTED CENTROIDS")
    print("=" * 60)
    
    # Use the interactive viewer with corrected centroids
    view_all_slices_with_centroids(raw_path, seg_path, centroids_world)

# Test the fixed function
if seg_path:
    print("=" * 60)
    print("TESTING FIXED CENTROID EXTRACTION")
    print("=" * 60)
    
    centroids_world, centroids_voxel, affine = get_label_centroids(seg_path)
    
    print("\nCorrected centroids:")
    for label in sorted(centroids_world.keys()):
        world = centroids_world[label]
        vox = centroids_voxel[label]
        print(f"Label {int(label)}: Voxel(x={vox[0]:.0f}, y={vox[1]:.0f}, z={vox[2]:.0f}) -> World(x={world[0]:.1f}, y={world[1]:.1f}, z={world[2]:.1f})")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def visualize_centroids_correct(ct_nii_path, seg_nii_path=None, centroids_world=None):
    """
    CORRECT visualization that properly handles the coordinate systems.
    """
    # Load CT data
    ct_img = nib.load(ct_nii_path)
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine
    inv_affine = np.linalg.inv(affine)
    
    # Convert world centroids to voxel coordinates if provided
    centroids_voxel = {}
    if centroids_world:
        for label, world_xyz in centroids_world.items():
            # Convert world to voxel
            world_homog = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1])
            vox_homog = inv_affine @ world_homog
            centroids_voxel[label] = (int(round(vox_homog[0])), 
                                      int(round(vox_homog[1])), 
                                      int(round(vox_homog[2])))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get center slices
    z_center = ct_data.shape[0] // 2
    y_center = ct_data.shape[1] // 2
    x_center = ct_data.shape[2] // 2
    
    # 1. Axial view (looking from top) - shows X vs Y at a fixed Z
    axial_data = ct_data[z_center, :, :]
    axes[0].imshow(axial_data, cmap='gray', origin='lower')
    axes[0].set_title(f'Axial View (Z-slice = {z_center})')
    axes[0].set_xlabel('X (voxel index)')
    axes[0].set_ylabel('Y (voxel index)')
    
    # Overlay segmentation if provided
    if seg_nii_path:
        seg_img = nib.load(seg_nii_path)
        seg_data = seg_img.get_fdata()
        seg_slice = seg_data[z_center, :, :]
        axes[0].imshow(seg_slice, cmap='jet', alpha=0.3, origin='lower')
    
    # Plot centroids on axial view
    for label, (vx, vy, vz) in centroids_voxel.items():
        if abs(vz - z_center) < 10:  # Show if close to current slice
            axes[0].scatter(vx, vy, color='red', s=100, marker='o', 
                          linewidth=2, edgecolors='white')
            axes[0].text(vx, vy, f' {int(label)}', color='red', fontsize=8)
    
    # 2. Coronal view (looking from front) - shows X vs Z at fixed Y
    coronal_data = ct_data[:, y_center, :]
    axes[1].imshow(coronal_data, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal View (Y-slice = {y_center})')
    axes[1].set_xlabel('X (voxel index)')
    axes[1].set_ylabel('Z (voxel index)')
    
    if seg_nii_path:
        seg_slice = seg_data[:, y_center, :]
        axes[1].imshow(seg_slice, cmap='jet', alpha=0.3, origin='lower')
    
    for label, (vx, vy, vz) in centroids_voxel.items():
        if abs(vy - y_center) < 10:
            axes[1].scatter(vx, vz, color='red', s=100, marker='o', 
                          linewidth=2, edgecolors='white')
            axes[1].text(vx, vz, f' {int(label)}', color='red', fontsize=8)
    
    # 3. Sagittal view (looking from side) - shows Y vs Z at fixed X
    sagittal_data = ct_data[:, :, x_center]
    axes[2].imshow(sagittal_data, cmap='gray', origin='lower')
    axes[2].set_title(f'Sagittal View (X-slice = {x_center})')
    axes[2].set_xlabel('Y (voxel index)')
    axes[2].set_ylabel('Z (voxel index)')
    
    if seg_nii_path:
        seg_slice = seg_data[:, :, x_center]
        axes[2].imshow(seg_slice, cmap='jet', alpha=0.3, origin='lower')
    
    for label, (vx, vy, vz) in centroids_voxel.items():
        if abs(vx - x_center) < 10:
            axes[2].scatter(vy, vz, color='red', s=100, marker='o', 
                          linewidth=2, edgecolors='white')
            axes[2].text(vy, vz, f' {int(label)}', color='red', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return centroids_voxel


# Interactive viewer with scrolling
def interactive_centroid_viewer(ct_nii_path, seg_nii_path=None, centroids_world=None):
    """
    Interactive viewer - scroll through slices to see centroids.
    """
    ct_img = nib.load(ct_nii_path)
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine
    inv_affine = np.linalg.inv(affine)
    
    if seg_nii_path:
        seg_img = nib.load(seg_nii_path)
        seg_data = seg_img.get_fdata()
    
    # Convert world centroids to voxel
    centroids_voxel = {}
    if centroids_world:
        for label, world_xyz in centroids_world.items():
            world_homog = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1])
            vox_homog = inv_affine @ world_homog
            centroids_voxel[label] = (int(round(vox_homog[0])), 
                                      int(round(vox_homog[1])), 
                                      int(round(vox_homog[2])))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initial slice indices
    z_idx = ct_data.shape[0] // 2
    y_idx = ct_data.shape[1] // 2
    x_idx = ct_data.shape[2] // 2
    
    def update_views():
        # Clear axes
        for ax in axes:
            ax.clear()
        
        # Axial view
        axial_data = ct_data[z_idx, :, :]
        axes[0].imshow(axial_data, cmap='gray', origin='lower')
        if seg_nii_path:
            axial_seg = seg_data[z_idx, :, :]
            axes[0].imshow(axial_seg, cmap='jet', alpha=0.3, origin='lower')
        for label, (vx, vy, vz) in centroids_voxel.items():
            if vz == z_idx:
                axes[0].scatter(vx, vy, color='red', s=100, marker='o', linewidth=2, edgecolors='white')
                axes[0].text(vx, vy, f' {int(label)}', color='red', fontsize=8)
        axes[0].set_title(f'Axial Z={z_idx} (scroll to change)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Coronal view
        coronal_data = ct_data[:, y_idx, :]
        axes[1].imshow(coronal_data, cmap='gray', origin='lower')
        if seg_nii_path:
            coronal_seg = seg_data[:, y_idx, :]
            axes[1].imshow(coronal_seg, cmap='jet', alpha=0.3, origin='lower')
        for label, (vx, vy, vz) in centroids_voxel.items():
            if vy == y_idx:
                axes[1].scatter(vx, vz, color='red', s=100, marker='o', linewidth=2, edgecolors='white')
                axes[1].text(vx, vz, f' {int(label)}', color='red', fontsize=8)
        axes[1].set_title(f'Coronal Y={y_idx}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        
        # Sagittal view
        sagittal_data = ct_data[:, :, x_idx]
        axes[2].imshow(sagittal_data, cmap='gray', origin='lower')
        if seg_nii_path:
            sagittal_seg = seg_data[:, :, x_idx]
            axes[2].imshow(sagittal_seg, cmap='jet', alpha=0.3, origin='lower')
        for label, (vx, vy, vz) in centroids_voxel.items():
            if vx == x_idx:
                axes[2].scatter(vy, vz, color='red', s=100, marker='o', linewidth=2, edgecolors='white')
                axes[2].text(vy, vz, f' {int(label)}', color='red', fontsize=8)
        axes[2].set_title(f'Sagittal X={x_idx}')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        
        fig.canvas.draw_idle()
    
    def on_scroll(event):
        nonlocal z_idx, y_idx, x_idx
        if event.inaxes == axes[0]:
            z_idx = max(0, min(ct_data.shape[0]-1, z_idx + int(event.step)))
            update_views()
        elif event.inaxes == axes[1]:
            y_idx = max(0, min(ct_data.shape[1]-1, y_idx + int(event.step)))
            update_views()
        elif event.inaxes == axes[2]:
            x_idx = max(0, min(ct_data.shape[2]-1, x_idx + int(event.step)))
            update_views()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    update_views()
    plt.tight_layout()
    plt.show()


# Now test everything
if seg_path:
    print("=" * 60)
    print("FINAL TEST WITH CORRECTED FUNCTIONS")
    print("=" * 60)
    
    # Get corrected centroids
    centroids_world, centroids_voxel, affine = get_label_centroids_fixed(seg_path)
    
    print("\nFinal corrected centroids:")
    for label in sorted(centroids_world.keys()):
        world = centroids_world[label]
        vox = centroids_voxel[label]
        print(f"Label {int(label)}: Voxel(x={vox[0]:.0f}, y={vox[1]:.0f}, z={vox[2]:.0f}) -> World(x={world[0]:.1f}, y={world[1]:.1f}, z={world[2]:.1f})")
    
    # Visualize
    print("\nOpening interactive viewer...")
    print("Scroll on each view to change slices. Red circles should be on bright spots (leads).")
    interactive_centroid_viewer(raw_path, seg_path, centroids_world)

def simple_centroid_check(ct_nii_path, seg_nii_path, label=4001):
    """Simple check: show the slice where label 4001 should be"""
    
    ct_img = nib.load(ct_nii_path)
    seg_img = nib.load(seg_nii_path)
    
    # Get centroid for specific label
    centroids_world, centroids_voxel, _ = get_label_centroids(seg_nii_path, [label])
    
    if label not in centroids_voxel:
        print(f"Label {label} not found")
        return
    
    vx, vy, vz = centroids_voxel[label]
    
    # Create a figure with 3 views centered on this centroid
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial view at slice z=vz
    axial = ct_img.get_fdata()[vz, :, :]
    axes[0].imshow(axial.T, cmap='gray', origin='lower')
    axes[0].scatter(vx, vy, color='red', s=100, marker='o')
    axes[0].set_title(f'Axial at Z={vz} - Lead should be at red circle')
    
    # Coronal view at slice y=vy
    coronal = ct_img.get_fdata()[:, vy, :]
    axes[1].imshow(coronal.T, cmap='gray', origin='lower')
    axes[1].scatter(vx, vz, color='red', s=100, marker='o')
    axes[1].set_title(f'Coronal at Y={vy}')
    
    # Sagittal view at slice x=vx
    sagittal = ct_img.get_fdata()[:, :, vx]
    axes[2].imshow(sagittal, cmap='gray', origin='lower')
    axes[2].scatter(vy, vz, color='red', s=100, marker='o')
    axes[2].set_title(f'Sagittal at X={vx}')
    
    plt.suptitle(f'Label {label} centroid verification')
    plt.tight_layout()
    plt.show()
    
    return centroids_voxel[label]

# Test on your patient
if seg_path:
    voxel_coords = simple_centroid_check(raw_path, seg_path, 4001)
    print(f"Label 4001 should be at voxel coordinates: {voxel_coords}")

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