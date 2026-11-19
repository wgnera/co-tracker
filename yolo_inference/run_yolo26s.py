"""Run YOLO26s inference on the holdout epee video frames.

Usage: python scripts/yolo26s.py
Note: Add your own data to use the model 
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "best.pt"
DEFAULT_SOURCE = REPO_ROOT / "sample_data" 
DEFAULT_OUTPUT = REPO_ROOT / "sample_predictions"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Model checkpoint (default: models/best.pt)")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Image, video, or directory (default: holdout epee frames)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory (numbered if it already exists)")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Minimum detection confidence (default: 0.5)")
    parser.add_argument("--device", default="cpu", help="cpu or GPU index, e.g. 0")
    parser.add_argument("--save-txt", action="store_true",
                        help="Also save YOLO labels with confidence scores")
    args = parser.parse_args(argv)
    if not args.weights.is_file():
        parser.error(f"Checkpoint not found: {args.weights}")
    if not args.source.exists():
        parser.error(f"Input not found: {args.source}")
    if args.image_size < 32:
        parser.error("image-size must be >= 32")
    if not 0 <= args.conf <= 1:
        parser.error("conf must be between 0 and 1")
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("device must be cpu or one GPU index")
    return args


def predict(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
    from ultralytics import YOLO

    output = args.output_dir.resolve()
    model = YOLO(str(args.weights.resolve()))
    results = model.predict(
        source=str(args.source.resolve()),
        imgsz=args.image_size,
        conf=args.conf,
        device=args.device,
        batch=1,
        stream=True,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_txt,
        project=str(output.parent),
        name=output.name,
        exist_ok=False,
    )
    # Consume the generator so inference runs without retaining every frame in RAM.
    frames = 0
    detections = 0
    for result in results:
        frames += 1
        if result.boxes is not None:
            detections += len(result.boxes)
    print(f"Processed {frames} frames; found {detections} detections.")
    print(f"Results saved to: {model.predictor.save_dir}")


def main(argv=None):
    args = parse_args(argv)
    try:
        predict(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
