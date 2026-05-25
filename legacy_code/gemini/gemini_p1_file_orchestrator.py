import os
import json
import re

def create_data_inventory(root_paths):
    inventory = {
        "labeled": {},
        "unlabeled": {}
    }

    for root in root_paths:
        # Define the specific folders based on your structure
        img_dir = os.path.join(root, "HCT2_img_nii")
        seg_dir = os.path.join(root, "HCT2_leads_seg_nii")
        roi_dir = os.path.join(root, "AUH-2024-HCT2-rois")

        if not os.path.exists(img_dir):
            continue

        for filename in os.listdir(img_dir):
            if filename.endswith(".nii.gz"):
                # Extract Patient ID (e.g., 10001)
                match = re.search(r'(\d+)', filename)
                if not match: continue
                p_id = match.group(1)

                # Look for matching files
                seg_file = os.path.join(seg_dir, f"{p_id}_HCT2_leads_seg.nii.gz") # Adjust name pattern as needed
                roi_file = os.path.join(roi_dir, f"._{p_id}_leads.csv")

                data_entry = {
                    "image_path": os.path.abspath(os.path.join(img_dir, filename)),
                    "segmentation_path": os.path.abspath(seg_file) if os.path.exists(seg_file) else None,
                    "csv_path": os.path.abspath(roi_file) if os.path.exists(roi_file) else None
                }

                # Categorize: If it has a segmentation file, it's Ground Truth
                if data_entry["segmentation_path"]:
                    inventory["labeled"][p_id] = data_entry
                else:
                    inventory["unlabeled"][p_id] = data_entry

    return inventory

# Usage
paths_to_search = ["/BENG280C_pacing_lead_data_1st20/", "/HCT2_lead_segmentation_training/"]
full_inventory = create_data_inventory(paths_to_search)

# Save to JSON
with open("heart_data_map.json", "w") as f:
    json.dump(full_inventory, f, indent=4)

print(f"Inventory created! Labeled: {len(full_inventory['labeled'])}, Unlabeled: {len(full_inventory['unlabeled'])}")