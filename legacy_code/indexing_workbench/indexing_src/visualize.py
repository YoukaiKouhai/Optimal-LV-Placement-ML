# src/visualize.py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from typing import Dict, Tuple, Optional

def interactive_centroid_viewer(ct_nii_path: str,
                                seg_nii_path: Optional[str] = None,
                                centroids_world: Optional[Dict[int, Tuple[float, float, float]]] = None):
    """
    Interactive 3‑panel viewer. Scroll on each panel to change the slice.
    Red circles mark centroids (only shown when the slice contains the centroid).
    """
    ct_img = nib.load(ct_nii_path)
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine
    inv_affine = np.linalg.inv(affine)

    seg_data = None
    if seg_nii_path:
        seg_img = nib.load(seg_nii_path)
        seg_data = seg_img.get_fdata()

    # Convert world centroids to voxel coordinates
    centroids_vox = {}
    if centroids_world:
        for label, world_xyz in centroids_world.items():
            world_homog = np.array([world_xyz[0], world_xyz[1], world_xyz[2], 1])
            vox_homog = inv_affine @ world_homog
            centroids_vox[label] = (int(round(vox_homog[0])),
                                    int(round(vox_homog[1])),
                                    int(round(vox_homog[2])))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    z_idx = ct_data.shape[0] // 2
    y_idx = ct_data.shape[1] // 2
    x_idx = ct_data.shape[2] // 2

    def update_views():
        for ax in axes:
            ax.clear()

        # Axial
        axial = ct_data[z_idx, :, :]
        axes[0].imshow(axial, cmap='gray', origin='lower')
        if seg_data is not None:
            axes[0].imshow(seg_data[z_idx, :, :], cmap='jet', alpha=0.3, origin='lower')
        for label, (vx, vy, vz) in centroids_vox.items():
            if vz == z_idx:
                axes[0].scatter(vx, vy, color='red', s=100, marker='o', edgecolors='white')
                axes[0].text(vx, vy, f' {label}', color='red', fontsize=8)
        axes[0].set_title(f'Axial Z={z_idx} (scroll)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')

        # Coronal
        coronal = ct_data[:, y_idx, :]
        axes[1].imshow(coronal, cmap='gray', origin='lower')
        if seg_data is not None:
            axes[1].imshow(seg_data[:, y_idx, :], cmap='jet', alpha=0.3, origin='lower')
        for label, (vx, vy, vz) in centroids_vox.items():
            if vy == y_idx:
                axes[1].scatter(vx, vz, color='red', s=100, marker='o', edgecolors='white')
                axes[1].text(vx, vz, f' {label}', color='red', fontsize=8)
        axes[1].set_title(f'Coronal Y={y_idx}')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')

        # Sagittal
        sagittal = ct_data[:, :, x_idx]
        axes[2].imshow(sagittal, cmap='gray', origin='lower')
        if seg_data is not None:
            axes[2].imshow(seg_data[:, :, x_idx], cmap='jet', alpha=0.3, origin='lower')
        for label, (vx, vy, vz) in centroids_vox.items():
            if vx == x_idx:
                axes[2].scatter(vy, vz, color='red', s=100, marker='o', edgecolors='white')
                axes[2].text(vy, vz, f' {label}', color='red', fontsize=8)
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


def static_slice_view(ct_nii_path: str,
                      seg_nii_path: Optional[str] = None,
                      centroids_world: Optional[Dict[int, Tuple[float, float, float]]] = None):
    """Simpler static view through the centre slices."""
    ct_img = nib.load(ct_nii_path)
    ct_data = ct_img.get_fdata()
    affine = ct_img.affine
    inv_affine = np.linalg.inv(affine)

    centroids_vox = {}
    if centroids_world:
        for label, world_xyz in centroids_world.items():
            vox = inv_affine @ [world_xyz[0], world_xyz[1], world_xyz[2], 1]
            centroids_vox[label] = (int(round(vox[0])), int(round(vox[1])), int(round(vox[2])))

    z_center = ct_data.shape[0] // 2
    y_center = ct_data.shape[1] // 2
    x_center = ct_data.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial
    axes[0].imshow(ct_data[z_center, :, :], cmap='gray', origin='lower')
    if seg_nii_path:
        seg = nib.load(seg_nii_path).get_fdata()
        axes[0].imshow(seg[z_center, :, :], cmap='jet', alpha=0.3, origin='lower')
    for (vx, vy, vz) in centroids_vox.values():
        if abs(vz - z_center) < 5:
            axes[0].scatter(vx, vy, color='red', s=80, marker='o')
    axes[0].set_title(f'Axial Z={z_center}')

    # Coronal
    axes[1].imshow(ct_data[:, y_center, :], cmap='gray', origin='lower')
    if seg_nii_path:
        axes[1].imshow(seg[:, y_center, :], cmap='jet', alpha=0.3, origin='lower')
    for (vx, vy, vz) in centroids_vox.values():
        if abs(vy - y_center) < 5:
            axes[1].scatter(vx, vz, color='red', s=80, marker='o')
    axes[1].set_title(f'Coronal Y={y_center}')

    # Sagittal
    axes[2].imshow(ct_data[:, :, x_center], cmap='gray', origin='lower')
    if seg_nii_path:
        axes[2].imshow(seg[:, :, x_center], cmap='jet', alpha=0.3, origin='lower')
    for (vx, vy, vz) in centroids_vox.values():
        if abs(vx - x_center) < 5:
            axes[2].scatter(vy, vz, color='red', s=80, marker='o')
    axes[2].set_title(f'Sagittal X={x_center}')

    plt.tight_layout()
    plt.show()