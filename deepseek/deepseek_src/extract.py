# src/extract.py
import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass
from typing import List, Dict, Tuple, Optional

def extract_centroids(seg_nii_path: str,
                      label_list: Optional[List[int]] = None
                      ) -> Tuple[Dict[int, Tuple[float, float, float]],
                                 Dict[int, Tuple[float, float, float]],
                                 np.ndarray]:
    """
    Extract centroids of given labels from a segmentation NIfTI file.

    Args:
        seg_nii_path: path to the .nii.gz segmentation file.
        label_list: list of integer labels to extract. If None, use all non-zero labels.

    Returns:
        centroids_world: dict {label: (x, y, z)} in world coordinates (mm).
        centroids_voxel: dict {label: (x, y, z)} in voxel coordinates.
        affine: the 4x4 affine matrix of the segmentation.
    """
    seg_img = nib.load(seg_nii_path)
    seg_data = seg_img.get_fdata()
    affine = seg_img.affine

    if label_list is None:
        label_list = np.unique(seg_data).astype(int).tolist()
        label_list = [l for l in label_list if l != 0]

    centroids_world = {}
    centroids_voxel = {}

    for label in label_list:
        mask = (seg_data == label)
        if not np.any(mask):
            print(f"Warning: Label {label} not found in {seg_nii_path}")
            continue

        # center_of_mass returns (z, y, x)
        com_z, com_y, com_x = center_of_mass(mask)
        voxel_centroid = (com_x, com_y, com_z)
        centroids_voxel[label] = voxel_centroid

        # affine expects (x, y, z, 1)
        voxel_homog = np.array([com_x, com_y, com_z, 1])
        world_coords = affine @ voxel_homog
        centroids_world[label] = tuple(world_coords[:3])

    return centroids_world, centroids_voxel, affine