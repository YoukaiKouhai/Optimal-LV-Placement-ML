"""
CRT Lead Detection Project — Step 1: Data Inventory Builder
============================================================
Scans both dataset folders, matches files per patient,
flags which patients have ground truth, and saves everything
to a JSON file for use by downstream scripts.

Usage:
    python build_data_inventory.py

Output:
    data_inventory.json  — full matched inventory of both datasets
    inventory_report.txt — human-readable summary
"""

import os
import re
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURE YOUR PATHS HERE
# ─────────────────────────────────────────────────────────────────────────────

DATASET_1_ROOT = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\BENG280C_pacing_lead_data_1st20")
DATASET_2_ROOT = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\HCT2_lead_segmentation_training")

OUTPUT_JSON    = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\data_inventory.json")
OUTPUT_REPORT  = Path(r"C:\Users\ayw005\Desktop\BENG 280C Project\Optimal-LV-Placement-ML\claude\inventory_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
#  DATA MODEL — one record per patient
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientRecord:
    patient_id:      str
    dataset:         str                   # "dataset_1" or "dataset_2"

    # raw CT scan
    img_nii:         Optional[str] = None  # path to .nii.gz

    # segmentation mask (ground truth labels 4001–4008)
    seg_nii:         Optional[str] = None

    # PNG slices folder
    png_folder:      Optional[str] = None

    # ROI / clinical files
    leads_csv:       Optional[str] = None  # path to headerless CSV with rows: Name, X, Y, Z in mm
    bullseye_csv:    Optional[str] = None
    bullseye_png:    Optional[str] = None
    rois_series:     Optional[str] = None

    # derived flags (set automatically below)
    has_ground_truth: bool = False   # True when seg_nii AND leads_csv exist
    missing_files:    list = field(default_factory=list)

    def compute_flags(self):
        """Mark ground truth status and list missing key files."""
        self.has_ground_truth = bool(self.seg_nii and self.leads_csv)

        # Warn about critical missing pieces
        if not self.img_nii:
            self.missing_files.append("img_nii")
        if not self.seg_nii:
            self.missing_files.append("seg_nii")
        if not self.leads_csv:
            self.missing_files.append("leads_csv")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER — extract patient ID from any filename
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_ID_RE = re.compile(r'(\d{5})')   # e.g. "10001"

def extract_patient_id(filename: str) -> Optional[str]:
    """Pull the 5-digit patient ID out of a filename."""
    match = PATIENT_ID_RE.search(filename)
    return match.group(1) if match else None


def parse_leads_csv(csv_path: Path) -> list[dict[str, float | str]]:
    """Read a headerless lead CSV and return labeled world coordinates.

    The CSV format is expected to be headerless, one line per point:
    Name, X, Y, Z
    where X, Y, Z are world coordinates in millimeters.
    """
    points: list[dict[str, float | str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
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


# ─────────────────────────────────────────────────────────────────────────────
#  SCANNER — walk a folder and collect files by patient
# ─────────────────────────────────────────────────────────────────────────────

def is_hidden_file(path: Path) -> bool:
    return path.name.startswith('.') or path.name.startswith('._')


def first_match(folder: Path, patterns: list[str] | str) -> Optional[str]:
    """Return the first non-hidden file in *folder* that matches one or more glob patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]
    if not folder.exists():
        return None

    for pattern in patterns:
        hits = [p for p in folder.glob(pattern) if not is_hidden_file(p)]
        if hits:
            return str(sorted(hits)[0])
    return None


def scan_dataset(root: Path, dataset_label: str) -> dict[str, PatientRecord]:
    """
    Scan one dataset root folder and return a dict of
    { patient_id → PatientRecord }.
    """
    records: dict[str, PatientRecord] = {}

    # ── sub-folder layout ────────────────────────────────────────────────────
    img_dir       = root / "HCT2_img_nii"
    seg_dir       = root / "HCT2_leads_seg_nii"
    png_base_dir  = root / ("HCT2_leads_png"
                            if dataset_label == "dataset_1"
                            else "HCT2_leads_groundtruth_png")
    rois_dir      = root / "AUH-2024-HCT2-rois"

    # support both common PNG folder names in case directory names vary
    if not png_base_dir.exists():
        for alt in ("HCT2_leads_png", "HCT2_leads_groundtruth_png"):
            candidate = root / alt
            if candidate.exists():
                png_base_dir = candidate
                break

    # ── discover all patient IDs from the raw image folder ──────────────────
    if img_dir.exists():
        for f in img_dir.iterdir():
            if is_hidden_file(f):
                continue
            pid = extract_patient_id(f.name)
            if pid and f.suffix in ('.gz', '.nii') and pid not in records:
                records[pid] = PatientRecord(patient_id=pid, dataset=dataset_label)
                records[pid].img_nii = str(f)

    # ── also sweep seg folder in case img is missing for some patients ───────
    if seg_dir.exists():
        for f in seg_dir.iterdir():
            if is_hidden_file(f):
                continue
            pid = extract_patient_id(f.name)
            if pid and f.suffix in ('.gz', '.nii'):
                if pid not in records:
                    records[pid] = PatientRecord(patient_id=pid, dataset=dataset_label)
                records[pid].seg_nii = str(f)

    # ── fill in remaining fields ─────────────────────────────────────────────
    for pid, rec in records.items():

        # img (in case we found patient via seg first)
        if rec.img_nii is None and img_dir.exists():
            rec.img_nii = first_match(
                img_dir,
                [
                    f"*{pid}*_HCT2.nii.gz",
                    f"*{pid}*_HCT2_img.nii.gz",
                    f"*{pid}*_HCT2_HCT2.nii.gz",
                    f"*{pid}*.nii*",
                ],
            )

        # seg (in case we found patient via img first)
        if rec.seg_nii is None and seg_dir.exists():
            rec.seg_nii = first_match(
                seg_dir,
                [
                    f"*{pid}*_HCT2_leads_seg.nii.gz",
                    f"*{pid}*_HCT2_HCT2_leads_seg.nii.gz",
                    f"*{pid}*seg.nii*",
                ],
            )

        # PNG slices folder — dataset_1 uses flat folder, dataset_2 uses subdir
        if png_base_dir.exists():
            if dataset_label == "dataset_1":
                rec.png_folder = str(png_base_dir)
            else:
                patient_png = png_base_dir / f"{pid}_HCT2"
                rec.png_folder = str(patient_png) if patient_png.exists() else None

        # ROI / clinical files
        if rois_dir.exists():
            rec.leads_csv     = first_match(rois_dir, f"{pid}_leads.csv")
            rec.bullseye_csv  = first_match(rois_dir, f"{pid}_leads_bullseye.csv")
            rec.bullseye_png  = first_match(rois_dir, f"{pid}_leads_bullseye.png")
            rec.rois_series   = first_match(rois_dir, f"{pid}.rois_series")

        rec.compute_flags()

    return records


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_inventory():
    print("=" * 60)
    print("  CRT Lead Inventory Builder")
    print("=" * 60)

    # scan both datasets
    print(f"\n[1/2] Scanning Dataset 1: {DATASET_1_ROOT}")
    ds1 = scan_dataset(DATASET_1_ROOT, "dataset_1")
    print(f"      Found {len(ds1)} patients.")

    print(f"\n[2/2] Scanning Dataset 2: {DATASET_2_ROOT}")
    ds2 = scan_dataset(DATASET_2_ROOT, "dataset_2")
    print(f"      Found {len(ds2)} patients.")

    # ── split by ground truth availability ───────────────────────────────────
    all_records = {**ds1, **ds2}  # merge (dataset_2 wins if same ID in both)

    with_gt  = {pid: r for pid, r in all_records.items() if r.has_ground_truth}
    raw_only = {pid: r for pid, r in all_records.items() if not r.has_ground_truth}

    # ── serialise to JSON ────────────────────────────────────────────────────
    payload = {
        "summary": {
            "total_patients":         len(all_records),
            "with_ground_truth":      len(with_gt),
            "raw_only":               len(raw_only),
            "ground_truth_patient_ids": sorted(with_gt.keys()),
            "raw_only_patient_ids":     sorted(raw_only.keys()),
        },
        "dataset_1": {pid: asdict(r) for pid, r in ds1.items()},
        "dataset_2": {pid: asdict(r) for pid, r in ds2.items()},
        "ground_truth_patients": {pid: asdict(r) for pid, r in with_gt.items()},
        "raw_only_patients":     {pid: asdict(r) for pid, r in raw_only.items()},
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✅  Inventory saved → {OUTPUT_JSON}")

    # ── human-readable report ────────────────────────────────────────────────
    lines = []
    lines.append("CRT Lead Detection — Data Inventory Report")
    lines.append("=" * 60)
    lines.append(f"Total patients found : {len(all_records)}")
    lines.append(f"  With ground truth  : {len(with_gt)}  ← use for supervised training/validation")
    lines.append(f"  Raw only           : {len(raw_only)} ← use for inference after model is trained")
    lines.append("")

    for label, group in [("GROUND TRUTH PATIENTS", with_gt),
                          ("RAW-ONLY PATIENTS",     raw_only)]:
        lines.append(f"── {label} ──────────────────────────")
        for pid in sorted(group.keys()):
            r = group[pid]
            status_parts = []
            if r.img_nii:     status_parts.append("img✓")
            if r.seg_nii:     status_parts.append("seg✓")
            if r.leads_csv:   status_parts.append("csv✓")
            if r.bullseye_png:status_parts.append("bull✓")
            missing_str = f"  ⚠ missing: {r.missing_files}" if r.missing_files else ""
            lines.append(f"  {pid}  [{', '.join(status_parts)}]{missing_str}")

    report_text = "\n".join(lines)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✅  Report saved      → {OUTPUT_REPORT}")
    print()
    print(report_text)

if __name__ == "__main__":
    build_inventory()