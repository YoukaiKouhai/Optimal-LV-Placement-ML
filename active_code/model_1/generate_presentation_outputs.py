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
    """
    Description
    -----------
    Build or parse command-line arguments for generate_presentation_outputs.py.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    argparse.ArgumentParser
        Result produced by the function.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: Does not intentionally modify external state except through mutable objects provided by the caller.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
    parser = argparse.ArgumentParser(
        description="Generate centroid CSVs, bullseye plots, and presentation figures without retraining."
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Completed run directory. Defaults to latest run with a best checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to run-dir/weights/best_supervised_model.pth.")
    parser.add_argument("--eval-only", action="store_true", help="Accepted for clarity; this script never trains.")
    parser.add_argument("--force-centroids", action="store_true", help="Recompute centroid CSVs even if they already exist.")
    return parser


def main() -> None:
    """
    Description
    -----------
    Run the command-line workflow implemented by generate_presentation_outputs.py.
    
    Parameters
    ----------
    None
        This function does not take input parameters.
    
    Returns
    -------
    None
        No value is returned; the function is executed for orchestration, mutation of supplied objects, or file output.
        Raises: Propagates validation, I/O, shape, or runtime exceptions from underlying libraries when inputs are invalid or unavailable.
        Side effects: May create directories, write files, print progress, or update checkpoint/model state as part of the pipeline.
    
    Comments
    --------
    - Preconditions: Inputs must satisfy the path, tensor shape, dtype, and configuration assumptions of the surrounding pipeline.
    - Postconditions: Returned values or written artifacts follow the conventions used by downstream project scripts.
    - Usage constraints: Intended for the CRT lead localization research pipeline; validate assumptions before reuse with another dataset.
    - Performance considerations: Large 3D volumes and model inference can be memory- and GPU-intensive.
    - Thread safety: No explicit locking is used; avoid sharing mutable models, tensors, or output paths across concurrent calls.
    """
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
