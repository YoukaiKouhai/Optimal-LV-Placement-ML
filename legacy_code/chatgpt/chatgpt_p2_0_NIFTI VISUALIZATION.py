import json
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import random

# ============================================================
# LOAD REGISTRY
# ============================================================

with open(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\chatgpt\patient_registry.json", "r") as f:
    registry = json.load(f)

# ============================================================
# CHOOSE PATIENT
# ============================================================

patient_id = "10001"

patient = registry[patient_id]

# ============================================================
# LOAD RAW CT
# ============================================================

raw_nii = nib.load(patient["raw_img"])

raw_data = raw_nii.get_fdata()

print("=" * 60)
print("RAW IMAGE")
print("=" * 60)

print(f"Shape: {raw_data.shape}")
print(f"Data type: {raw_data.dtype}")

print("\nAffine Matrix:")
print(raw_nii.affine)

# ============================================================
# LOAD SEGMENTATION
# ============================================================

seg_nii = nib.load(patient["segmentation"])

seg_data = seg_nii.get_fdata()

print("\n" + "=" * 60)
print("SEGMENTATION")
print("=" * 60)

print(f"Shape: {seg_data.shape}")
print(f"Unique Labels: {np.unique(seg_data)}")

# ============================================================
# ANALYZE SEGMENTATION
# ============================================================

label_voxels = np.argwhere(seg_data > 0)

print(label_voxels[:10])

z_values = label_voxels[:, 2]

print("\nSlices containing labels:")
print(np.unique(z_values))

# ============================================================
# PICK a SLICE
# ============================================================

slice_idx = random.choice(np.unique(z_values))

raw_slice = raw_data[:, :, slice_idx]

# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(8, 8))

"""
plt.imshow(
    raw_slice,
    cmap="gray"
)
"""

plt.imshow(
    raw_slice,
    cmap="gray",
    vmin=-300,
    vmax=1000
)

plt.title(
    f"Patient {patient_id} | Slice {slice_idx}"
)

plt.axis("off")

plt.show()

"""
seg_slice = seg_data[:, :, slice_idx]

plt.figure(figsize=(8, 8))

# Base CT image
plt.imshow(
    raw_slice,
    cmap="gray"
)

# Segmentation overlay
plt.imshow(
    seg_slice,
    cmap="jet",
    alpha=0.5
)

plt.title(
    f"Overlay | Patient {patient_id}"
)

plt.axis("off")

plt.show()
"""

seg_slice = seg_data[:, :, slice_idx]

binary_mask = seg_slice > 0

print("np.unique(seg_slice):")
print(np.unique(seg_slice))
print("np.sum(binary_mask):")
print(np.sum(binary_mask))

coords = np.argwhere(binary_mask)

print("coords.min(axis=0):")
print(coords.min(axis=0))
print("coords.max(axis=0):")
print(coords.max(axis=0))

plt.figure(figsize=(8, 8))

plt.imshow(binary_mask)

plt.title(f"Segmentation Only | Slice {slice_idx}")

plt.show()

plt.figure(figsize=(8, 8))

plt.imshow(
    raw_slice,
    cmap="gray"
)

plt.imshow(
    binary_mask,
    cmap="Reds",
    alpha=0.8
)

plt.title(f"Overlay Slice {slice_idx}")

plt.axis("off")

plt.show()

plt.figure(figsize=(8,8))

plt.imshow(
    raw_slice,
    cmap="gray",
    vmin=-300,
    vmax=1000
)

plt.imshow(
    binary_mask,
    cmap="Reds",
    alpha=0.8
)

plt.xlim(350, 390)
plt.ylim(140, 100)

plt.show()

print(np.unique(seg_data))