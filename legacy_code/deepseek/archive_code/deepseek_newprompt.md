Based on everything we discovered and debugged together, here is a **revised, production‑ready project prompt** that incorporates all the hard‑won lessons. If you were to start fresh, this is the specification you would give to an LLM or use as your personal roadmap.

---

# Complete Project Prompt: CRT Lead Localization from CT

## 1. Project Goals (The “Why”)

**Primary Objective**  
Automatically identify and spatially localize Cardiac Resynchronization Therapy (CRT) leads from 3D CT images.

**Secondary Objective**  
Build a structured dataset (ground truth) to later train a machine learning model that suggests optimal lead placement.

**Anatomical Context**  
We need the 3D coordinates of each electrode relative to the heart’s apex (bottom) and base (top) to compute:
- Longitudinal position (height from apex)
- Radial angle (clock‑face orientation)

**Success Criteria**  
- Extracted lead coordinates (in mm) must match manual expert clicks with a **mean Euclidean error < 2 mm**.
- The pipeline must handle both segmentation masks (when available) and raw CT (when masks are missing).
- All data must be indexed in a single JSON file for easy access by later ML scripts.

---

## 2. Data Structure Specification (Two Folder Hierarchies)

You have two separate root folders. Their naming conventions are **not identical**. You must handle both.

### Root Folder A: `BENG280C_pacing_lead_data_1st20/`  
This contains **~18 patients** with ground truth (both segmentation masks and manual CSVs).

| Subfolder | Content | File pattern |
|-----------|---------|---------------|
| `HCT2_img_nii/` | Raw CT volumes | `[PatientID]_HCT2.nii.gz` (e.g. `10001_HCT2.nii.gz`) |
| `HCT2_leads_seg_nii/` | Segmentation masks (integer labels) | `[PatientID]_HCT2_HCT2_leads_seg.nii.gz` |
| `HCT2_leads_png/` | 2D PNG slices (for quick preview) | folders/files named after patient |
| `AUH-2024-HCT2-rois/` | Manual CSV coordinates (Horos exports) | `[PatientID]_leads.csv` (headerless), `[PatientID]_leads_bullseye.csv`, etc. |

### Root Folder B: `HCT2_lead_segmentation_training/`  
This contains **~281 patients**, mostly unlabeled (only raw CT + sometimes segmentation, rarely CSVs).

| Subfolder | Content | File pattern |
|-----------|---------|---------------|
| `HCT2_img_nii/` | Raw CT volumes | `[PatientID]_HCT2_img.nii.gz` (note extra `_img`) |
| `HCT2_leads_seg_nii/` | Segmentation masks (when available) | `[PatientID]_HCT2_leads_seg.nii.gz` |
| `HCT2_leads_groundtruth_png/` | PNG slices (one folder per patient) | `[PatientID]_HCT2/` containing PNGs |
| `AUH-2024-HCT2-rois/` | Manual CSVs (rare) | same pattern as folder A |

### Segmentation Label Map (from expert masks)

| Label | Description |
|-------|-------------|
| 4001‑4004 | LV lead electrodes (distal → proximal) |
| 4005‑4006 | RV lead electrodes (distal → proximal) |
| 4007 | Cardiac Apex |
| 4008 | Base center |
| 4009 | (optional additional marker – ignore if present) |

### Manual CSV Format (Ground Truth for Validation)

The `[PatientID]_leads.csv` files are **headerless** (no column names). Each line:  
`Name, X, Y, Z`  
where X, Y, Z are **world coordinates in millimeters** (same physical space as NIfTI affine).

Example:
```
ANT,75.27,-199.199,224.9
APEX,96.388,-211.36,178.0
BASE,48.432,-139.447,214.4
LL1,100.614,-136.582,203.9
LL2,89.903,-124.121,208.8
LL3,81.542,-119.918,208.1
LL4,69.985,-114.853,216.5
RL1,76.868,-222.524,203.9
RL2,67.96,-222.169,200.4
```

> **Critical note** – Right ventricular leads are named **RL1, RL2** (not RV1, RV2). The anterior wall marker is `ANT` (not a lead, used for rotation).

---

## 3. Critical Technical Notes (Read Before Coding)

### 3.1 NIfTI coordinate systems
- NIfTI stores data as `(z, y, x)` – first dimension is slice index (Z), second is row (Y), third is column (X).
- The **affine matrix** maps voxel coordinates `(x, y, z, 1)` to world `(X, Y, Z, 1)` in mm.
- `center_of_mass` from `scipy.ndimage` returns `(z, y, x)`. You **must reorder** to `(x, y, z)` before applying the affine.

### 3.2 Correct centroid extraction (pseudocode)
```python
com_z, com_y, com_x = center_of_mass(mask)
voxel_homog = [com_x, com_y, com_z, 1]
world = affine @ voxel_homog   # -> (X, Y, Z)
```

### 3.3 CSV parsing
- **No headers** – use `header=None` in `pd.read_csv()`.
- Coordinate order is `X, Y, Z` (already in mm).

### 3.4 Return values of functions
Write functions that return a **consistent number of values**. The main extraction function should return:
```python
def extract_centroids(seg_path, label_list):
    return centroids_world_dict, centroids_voxel_dict, affine
```
Do **not** change the return signature mid‑project – all callers must expect three values.

### 3.5 Visualization pitfalls
- Use `origin='lower'` in `imshow` to put (0,0) at bottom‑left (matches radiological orientation).
- Do **not** transpose slices arbitrarily; instead plot the slice as is, and when overlaying centroids, use the same (x,y) order as the slice array dimensions.
- Interactive scrolling is highly recommended for validation.

---

## 4. Pipeline Stages (Build in Order)

### Stage 0: Environment & Diagnostics (One‑time)
- Install: `nibabel`, `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-image`, `SimpleITK` (optional).
- Run a diagnostic function that prints:
  - Image dimensions
  - Affine matrix
  - Voxel sizes
  - Unique labels in segmentation
  - First 5 rows of any manual CSV
- This prevents silent coordinate errors.

### Stage I: File Orchestrator (Build Once, Use Everywhere)
Create a script that:
1. Walks through both root folders.
2. For each patient ID, matches:
   - Raw CT (handling both `_HCT2.nii.gz` and `_HCT2_img.nii.gz`)
   - Segmentation mask (if exists)
   - PNG folder (if exists)
   - Manual CSV (if exists)
3. Builds a **single Python dictionary** with all paths and a flag `has_ground_truth` (True if seg_nii or CSV exists).
4. Saves the dictionary as `patient_data_index.json`.

> This JSON becomes the **single source of truth** for all subsequent scripts.

### Stage II: Centroid Extraction (Spatial Feature Extraction)
Implement a function `extract_centroids(seg_nii_path, label_list)` that:
- Loads the segmentation NIfTI.
- For each requested label:
  - Finds voxels where `data == label`.
  - Computes the centroid in voxel space using `center_of_mass` (correct reordering as per 3.2).
  - Transforms to world mm using the affine.
- Returns three dictionaries: `centroids_world`, `centroids_voxel`, and the `affine`.

### Stage III: Validation Against Manual CSVs
For every patient that has both a segmentation mask **and** a manual CSV:
- Extract centroids (world).
- Parse the CSV into a dictionary `{name: (X,Y,Z)}`.
- Use a **corrected label map**:
  ```python
  label_to_name = {
      4001: 'LL1', 4002: 'LL2', 4003: 'LL3', 4004: 'LL4',
      4005: 'RL1', 4006: 'RL2',   # note RL not RV
      4007: 'APEX', 4008: 'BASE'
  }
  ```
- Compute Euclidean distance per point and the mean error.
- If mean error > 2 mm, stop and debug the affine or centroid ordering.

### Stage IV: Visualization (Static & Interactive)
Write two visualization functions:

**A. Static 3‑panel view**  
Show axial, coronal, sagittal slices through the center of the volume. Overlay:
- The segmentation mask (semi‑transparent)
- Red markers at the centroids (world → voxel via inverse affine)

**B. Interactive slice viewer**  
Allow scrolling through each orthogonal plane. Update the markers in real time.  
Use mouse scroll events to change slice index. Markers appear only when the current slice is within 2 voxels of the centroid’s position.

### Stage V: Computer Vision Fallback (For Raw CT, No Segmentation)
When no segmentation exists, use intensity thresholding:
- Threshold > 2000 HU (metal).
- Run 3D connected components (`skimage.measure.label`).
- Filter components by volume (typical lead electrode is 5‑30 mm³).
- Compute centroid of each surviving component.
- Return a list of candidate points (in world mm).  
This is not as accurate as segmentation but allows processing of the 80% unlabeled data.

### Stage VI: Normalization (Heart Axis & Bullseye Map)
After coordinates are extracted, compute:
1. **Long axis**: vector from `APEX` → `BASE`.
2. **Relative height** for each LV electrode:  
   `height = ( (coord - APEX) · axis ) / ( (BASE - APEX) · axis )`  
   (0 = apex, 1 = base)
3. **Radial angle** using the anterior marker (`ANT`) as reference:  
   Project leads onto the axial plane (X‑Y), compute angle between the vector (lead – heart center) and the anterior direction.

---

## 5. Machine Learning Feasibility (What We Learned)

You have only **~18 patients with full ground truth** (segmentation + CSV).  
**Deep learning from scratch is not feasible** with so few examples.

### Recommended approach
1. **First**, prove that the classical thresholding + connected components (Stage V) works on the 18 ground‑truth patients.  
   - If it achieves error < 5 mm, use that as your automatic method – no ML needed.
2. **If thresholding fails**, consider **transfer learning** with a pretrained 3D U‑Net (e.g., MONAI). But you will need to artificially augment your 18 patients (rotation, scaling, elastic deformations) to at least 200–300 training samples.
3. **Overfitting test** (as your friend suggested):  
   Train a small model (e.g., 3D CNN regressor) on your 18 patients.  
   - If it cannot even memorize the training set (loss stays high), the problem is too hard.  
   - If it does memorize but fails on a held‑out patient (leave‑one‑out cross‑validation), you lack data.

**Practical advice**: Do **not** spend weeks on ML. First get the classical pipeline working. Then, if you need to process the 80% unlabeled data, apply the thresholding method. Only if that fails badly, consider investing in more manual labels (at least 50–100).

---

## 6. Code Quality Requirements

- All functions must have **docstrings** (purpose, args, returns, side effects).
- Use **type hints** where possible.
- Write **small, single‑purpose functions** (e.g., `load_nifti`, `voxel_to_world`, `compute_centroids`, `parse_manual_csv`).
- Include a `if __name__ == "__main__":` block that runs the full pipeline on the first ground‑truth patient and prints validation errors.
- Save the final normalized coordinates (height, angle) as a new JSON file (e.g., `patient_10001_normalized.json`).

---

## 7. Deliverables (What “Success” Looks Like)

- **`patient_data_index.json`** – indexes all patients across both root folders.
- **`centroid_extraction.py`** – exports `extract_centroids()` and `extract_centroids_threshold()`.
- **`validate.py`** – computes errors against manual CSVs for all ground‑truth patients.
- **`visualize.py`** – produces both static and interactive plots.
- **`normalize.py`** – computes heart‑normalized coordinates.
- A **short report** (or log) showing the mean error for each ground‑truth patient (should be < 2 mm).

---

## 8. Common Pitfalls & Debugging Checklist

| Symptom | Likely Fix |
|---------|-------------|
| `too many values to unpack (expected 2)` | Your extraction function returns 3 values; unpack as `centroids_world, _, affine` |
| `nan` errors in validation | Label‑name mapping mismatch (e.g., RV1 vs RL1) or CSV parsing without `header=None` |
| Markers appear far from bright spots | Wrong coordinate order in affine transformation; check that `center_of_mass` order is reversed correctly |
| CT image looks upside down | Set `origin='lower'` in `imshow` |
| Segmentation overlay not aligning | Do not transpose slices – use the native array shape; apply same origin/lower |
| `get_label_centroids` finds no labels | Segmentation file may have a different naming pattern; verify file actually loads and contains those labels (run diagnostic) |

---

## 9. Starting Fresh – The First Three Commands

```bash
# 1. Install required libraries
pip install nibabel numpy scipy matplotlib pandas scikit-image

# 2. Run diagnostic on one patient (from your early code)
python diagnose_nifti.py --raw path/to/raw.nii.gz --seg path/to/seg.nii.gz

# 3. Build the patient index JSON
python build_index.py --root1 BENG280C_pacing_lead_data_1st20 --root2 HCT2_lead_segmentation_training
```

After these steps, proceed to Stage II and III. Do **not** touch ML until validation errors are < 2 mm.

---

## 10. Final Advice

- **Baby steps** – get one patient working perfectly before scaling to all.
- **Always visualize** – you cannot trust numbers alone.
- **Save intermediate results** (world coordinates, voxel coordinates) to CSV for manual inspection.
- **Do not change the affine or coordinate convention** – NIfTI’s affine is correct; errors come from code, not the data.

If you follow this prompt, you will have a robust, validated pipeline that can later be extended to machine learning. Good luck!