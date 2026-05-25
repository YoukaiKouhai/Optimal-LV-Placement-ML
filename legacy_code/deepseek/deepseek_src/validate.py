# src/validate.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Fixed label mapping for your data
LABEL_TO_NAME = {
    4001: 'LL1', 4002: 'LL2', 4003: 'LL3', 4004: 'LL4',
    4005: 'RL1', 4006: 'RL2',   # note RL, not RV
    4007: 'APEX', 4008: 'BASE'
}

def load_manual_csv(csv_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Load headerless CSV with format: Name, X, Y, Z.
    Returns dict {name: (x, y, z)}.
    """
    df = pd.read_csv(csv_path, header=None)
    points = {}
    for _, row in df.iterrows():
        name = row[0].strip()
        x, y, z = float(row[1]), float(row[2]), float(row[3])
        points[name] = (x, y, z)
    return points

def compute_errors(extracted_world: Dict[int, Tuple[float, float, float]],
                   manual_points: Dict[str, Tuple[float, float, float]],
                   label_map: Dict[int, str] = LABEL_TO_NAME) -> Dict[str, float]:
    """
    Compute Euclidean distance (mm) between extracted centroids and manual points.
    Returns dict {manual_name: error_mm}.
    """
    errors = {}
    for label, world_coord in extracted_world.items():
        name = label_map.get(label)
        if name and name in manual_points:
            manual_coord = manual_points[name]
            dist = np.linalg.norm(np.array(world_coord) - np.array(manual_coord))
            errors[name] = dist
    return errors