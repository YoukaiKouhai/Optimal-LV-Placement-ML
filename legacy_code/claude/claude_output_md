Here's what this script does and how to use it:
Setup — edit just 2 lines at the top:
pythonDATASET_1_ROOT = Path("/BENG280C_pacing_lead_data_1st20")
DATASET_2_ROOT = Path("/HCT2_lead_segmentation_training")
Change those to the actual paths on your computer where those folders live.
Then run it:
bashpython build_data_inventory.py
What comes out:
data_inventory.json — a structured file with 4 sections: dataset_1, dataset_2, ground_truth_patients, and raw_only_patients. Every patient entry has paths to their img, seg, csv, bullseye, etc. — or null if the file is missing.
inventory_report.txt — a readable summary printed to screen and saved, showing something like:
Total patients found : 20
  With ground truth  : 4   ← use for supervised training/validation
  Raw only           : 16  ← use for inference after model is trained
```
── GROUND TRUTH PATIENTS ──────────
  10001  [img✓, seg✓, csv✓, bull✓]
  10002  [img✓, seg✓, csv✓]  ⚠ missing: ['bullseye_csv']
```
Why this matters for your ML plan: Once you run this, you'll know your exact training set size. Your friend's suggestion to overfit first is smart — if you can't overfit on 4 patients you definitely can't generalize. The JSON also means Step 2 (extracting centroids from the seg files) can just import json and loop, no hardcoding paths anywhere.
Next baby steps in order:

✅ This script — map your data
Step 2 — load a seg .nii.gz, extract centroids for labels 4001–4008, apply affine matrix → get X,Y,Z in mm
Step 3 — compare those X,Y,Z to the CSV clicked points (Euclidean distance check)
Step 4 — only then think about the ML model