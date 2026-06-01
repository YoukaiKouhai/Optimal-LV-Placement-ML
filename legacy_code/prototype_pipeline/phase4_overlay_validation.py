import pandas as pd
import json
from pathlib import Path

# ============================================================
# LOAD REGISTRY
# ============================================================

REGISTRY_PATH = Path(
    r"C:\Users\<USER>\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\prototype_pipeline\patient_registry.json"
)

with open(REGISTRY_PATH, "r") as f:
    registry = json.load(f)

# ============================================================
# PROCESS PATIENT
# ============================================================

patient_id = "10001"

patient = registry[patient_id]

# ============================================================

csv_path = patient["roi_csv"]

df = pd.read_csv(
    csv_path,
    header=None
)

print(df.head(20))