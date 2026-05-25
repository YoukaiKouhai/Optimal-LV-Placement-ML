# ── In any downstream script ─────────────────────────────────────────────────
import csv
import json
from pathlib import Path


def load_leads_csv(csv_path: str | Path) -> list[dict[str, float | str]]:
    """Read a headerless leads CSV with rows: Name, X, Y, Z."""
    points: list[dict[str, float | str]] = []
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 4:
                continue
            try:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
            except ValueError:
                continue
            points.append({"name": row[0].strip(), "x": x, "y": y, "z": z})
    return points


with open(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\data_inventory.json") as f:
    inv = json.load(f)

# Loop over only patients with ground truth (for training / validation)
for pid, record in inv["ground_truth_patients"].items():
    img_path = record["img_nii"]      # path to raw CT .nii.gz
    seg_path = record["seg_nii"]      # path to label mask .nii.gz
    csv_path = record["leads_csv"]    # headerless file with labeled world coords

    points = load_leads_csv(csv_path) if csv_path else []
    print(f"Patient {pid}: img={img_path}, seg={seg_path}, leads={points}")

# Loop over raw-only patients (future inference targets)
for pid, record in inv["raw_only_patients"].items():
    img_path = record["img_nii"]
    print(f"Unlabeled patient {pid}: {img_path}")

# Check overall stats
print(inv["summary"])