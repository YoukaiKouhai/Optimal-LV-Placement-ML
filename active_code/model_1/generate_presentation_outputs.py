from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

from S10_Bullseye_Lead_Visualization import generate_bullseye_plots
from S11_Centroid_Export import export_centroids, find_default_run_dir, repo_root_from_here, resolve_path
from S12_Presentation_Figures import generate_presentation_figures


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate centroid CSVs, bullseye plots, and presentation figures without retraining."
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Completed run directory. Defaults to latest run with a best checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to run-dir/weights/best_supervised_model.pth.")
    parser.add_argument("--eval-only", action="store_true", help="Accepted for clarity; this script never trains.")
    parser.add_argument("--force-centroids", action="store_true", help="Recompute centroid CSVs even if they already exist.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = repo_root_from_here()
    run_dir = resolve_path(args.run_dir, repo_root) if args.run_dir else find_default_run_dir(repo_root)
    checkpoint: Optional[Path] = resolve_path(args.checkpoint, repo_root) if args.checkpoint else None

    print("Presentation output generation only; no training will be run.")
    print(f"Using run directory: {run_dir}")
    if checkpoint is not None:
        print(f"Using checkpoint: {checkpoint}")

    export_centroids(run_dir=run_dir, checkpoint_path=checkpoint, force=args.force_centroids)
    generate_bullseye_plots(run_dir=run_dir, checkpoint_path=checkpoint, force_centroids=False)
    generate_presentation_figures(run_dir=run_dir, checkpoint_path=checkpoint, force_centroids=False)

    print("Done generating presentation outputs.")


if __name__ == "__main__":
    main()
