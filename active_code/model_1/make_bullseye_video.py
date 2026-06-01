"""
Create a 1080p video from patient bullseye PNGs.

This is a standalone utility and is intentionally not imported by the main
training/evaluation pipeline.

Example:
    python make_bullseye_video.py ^
        --input-dir runs/cardiac_leads_ensemble_v3_v6/bullseye_plots ^
        --output runs/cardiac_leads_ensemble_v3_v6/bullseye_plots/bullseye_gt_vs_prediction_1080p.mp4
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PATTERN = "*_HCT2_img_bullseye_gt_vs_prediction.png"
DEFAULT_SIZE = (1920, 1080)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def patient_sort_key(path: Path) -> Tuple[int, str]:
    """
    Sort filenames numerically by the leading patient/case number.

    Example:
        10001_HCT2_img_bullseye_gt_vs_prediction.png -> 10001
    """
    match = re.match(r"^(\d+)_", path.name)
    if match:
        return int(match.group(1)), path.name
    fallback = re.search(r"(\d+)", path.stem)
    if fallback:
        return int(fallback.group(1)), path.name
    return 10**12, path.name


def find_bullseye_images(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    iterator: Iterable[Path] = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    images = sorted((path for path in iterator if path.is_file()), key=patient_sort_key)
    if not images:
        mode = "recursively " if recursive else ""
        raise FileNotFoundError(f"No PNG files found {mode}in {input_dir} with pattern {pattern!r}")
    return images


def fit_image_to_canvas(image: Image.Image, size: Tuple[int, int], background: Tuple[int, int, int]) -> Image.Image:
    target_w, target_h = size
    image = image.convert("RGB")
    scale = min(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, background)
    x = (target_w - resized.width) // 2
    y = (target_h - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def draw_patient_label(frame: Image.Image, image_path: Path, enabled: bool) -> Image.Image:
    if not enabled:
        return frame
    draw = ImageDraw.Draw(frame)
    label = image_path.stem.replace("_bullseye_gt_vs_prediction", "")
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except OSError:
        font = ImageFont.load_default()

    margin = 28
    bbox = draw.textbbox((0, 0), label, font=font)
    box_w = bbox[2] - bbox[0] + 2 * margin
    box_h = bbox[3] - bbox[1] + 2 * margin
    draw.rounded_rectangle((24, 24, 24 + box_w, 24 + box_h), radius=12, fill=(255, 255, 255), outline=(210, 210, 210))
    draw.text((24 + margin, 24 + margin), label, fill=(20, 20, 20), font=font)
    return frame


def pil_to_bgr(frame: Image.Image) -> np.ndarray:
    rgb = np.asarray(frame, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def create_video(
    image_paths: List[Path],
    output_path: Path,
    size: Tuple[int, int],
    fps: float,
    seconds_per_image: float,
    background: Tuple[int, int, int],
    show_label: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_repeat = max(1, int(round(fps * seconds_per_image)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as image:
                frame = fit_image_to_canvas(image, size=size, background=background)
            frame = draw_patient_label(frame, image_path, enabled=show_label)
            bgr = pil_to_bgr(frame)
            for _ in range(frame_repeat):
                writer.write(bgr)
            print(f"[{index:03d}/{len(image_paths):03d}] added {image_path.name}")
    finally:
        writer.release()


def parse_rgb(value: str) -> Tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("RGB color must be three comma-separated values from 0 to 255.")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 1080p MP4 from bullseye GT-vs-prediction PNGs.")
    parser.add_argument(
        "--input-dir",
        default="runs/cardiac_leads_ensemble_v3_v6/bullseye_plots",
        help="Folder containing *_HCT2_img_bullseye_gt_vs_prediction.png files.",
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--output",
        default="runs/cardiac_leads_ensemble_v3_v6/bullseye_plots/bullseye_gt_vs_prediction_1080p.mp4",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds-per-image", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1])
    parser.add_argument("--background", type=parse_rgb, default=(255, 255, 255), help="Canvas RGB color, e.g. 255,255,255.")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders under input-dir.")
    parser.add_argument("--no-label", action="store_true", help="Do not draw the patient ID in the video frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    input_dir = resolve_path(args.input_dir, root)
    output_path = resolve_path(args.output, root)
    size = (int(args.width), int(args.height))

    image_paths = find_bullseye_images(input_dir=input_dir, pattern=args.pattern, recursive=args.recursive)
    print(f"Found {len(image_paths)} bullseye PNGs in numeric order.")
    print(f"First frame: {image_paths[0].name}")
    print(f"Last frame:  {image_paths[-1].name}")
    create_video(
        image_paths=image_paths,
        output_path=output_path,
        size=size,
        fps=float(args.fps),
        seconds_per_image=float(args.seconds_per_image),
        background=args.background,
        show_label=not args.no_label,
    )
    print(f"Saved 1080p bullseye video: {output_path}")


if __name__ == "__main__":
    main()
