# ── In any downstream script ─────────────────────────────────────────────────
import json

with open(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\data_inventory.json") as f:
    inv = json.load(f)

# Loop over only patients with ground truth (for training / validation)
for pid, record in inv["ground_truth_patients"].items():
    img_path = record["img_nii"]      # path to raw CT .nii.gz
    seg_path = record["seg_nii"]      # path to label mask .nii.gz
    csv_path = record["leads_csv"]    # path to clicked X,Y,Z coords

    print(f"Patient {pid}: img={img_path}, seg={seg_path}")

# Loop over raw-only patients (future inference targets)
for pid, record in inv["raw_only_patients"].items():
    img_path = record["img_nii"]
    print(f"Unlabeled patient {pid}: {img_path}")

# Check overall stats
print(inv["summary"])