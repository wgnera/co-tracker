"""Fine-tune YOLO26s on a detection dataset described by data.yaml."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET = REPO_ROOT / "dataset" 


def default_output_dir(dataset_dir: Path) -> Path:
    """Name the run after the dataset, excluding preparation folder suffixes."""
    return REPO_ROOT / "runs" / "yolo26s" / dataset_dir.name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                        help="Dataset root containing data.yaml (default: dataset)")
    parser.add_argument("--output-dir", type=Path,
                        help="Exact run directory; default: runs/yolo26s/<dataset-name>")
    parser.add_argument("--model", default="yolo26s.pt", help="Pretrained detection checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Images per batch (default: 2 to limit CPU RAM use)")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="One GPU index, or cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("amp", "fp32"), default="amp")
    parser.add_argument("--resume", action="store_true", help="Resume output-dir/weights/last.pt")

    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate best.pt on test after training")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.image_size < 32:
        parser.error("epochs, patience, batch-size must be positive; image-size must be >=32")
    if args.workers < 0 or not 0 <= args.seed < 2**32:
        parser.error("workers must be nonnegative and seed must be in [0, 2**32)")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be positive and finite")
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("device must be one GPU index or cpu")
    if not args.model.endswith(".pt"):
        parser.error("Fine-tuning requires a pretrained .pt checkpoint, not a model YAML")
    if args.output_dir is None:
        try:
            args.output_dir = default_output_dir(args.dataset_dir)
        except ValueError as error:
            parser.error(str(error))
    return args


def train(args):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
    import torch
    from ultralytics import YOLO

    output = args.output_dir.resolve()
    if not args.resume and output.exists():
        raise ValueError(f"Run directory already exists; use --resume or another --output-dir: {output}")
    if args.resume and not (output / "weights" / "last.pt").is_file():
        raise ValueError(f"No checkpoint to resume in {output}")
    data_yaml = (args.dataset_dir / "data.yaml").resolve()
    if not args.dataset_dir.is_dir() or not data_yaml.is_file():
        raise ValueError(f"Dataset directory must contain data.yaml: {args.dataset_dir}")

    if args.device != "cpu":
        try:
            torch.cuda.init()
        except (RuntimeError, AssertionError) as error:
            raise ValueError(
                f"Cannot use GPU {args.device}: PyTorch {torch.__version__} "
                f"was built for CUDA {torch.version.cuda}. {error} "
                "Check nvidia-smi and install a PyTorch CUDA build compatible with "
                "the NVIDIA driver, or use --device cpu --batch-size 2."
            ) from error
    else:
        print(
            f"CPU training: batch={args.batch_size}, image-size={args.image_size}. "
            "AMP is disabled on CPU. If the process is killed, try "
            "--batch-size 1; reducing --image-size also lowers RAM use.",
            flush=True,
        )

    overrides = {
        "model": args.model,
        "data": str(data_yaml),
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": args.resume,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch_size,
        "imgsz": args.image_size,
        "optimizer": "AdamW",
        "lr0": args.lr,
        "freeze": 0,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "amp": args.precision == "amp" and args.device != "cpu",
        "cache": False,
    }
    if args.resume:
        overrides["resume"] = str(output / "weights" / "last.pt")

    model = YOLO(args.model)
    model.train(**overrides)
    if args.evaluate_test:
        YOLO(str(model.best)).val(
            data=str(data_yaml), split="test",
            imgsz=model.args.imgsz, batch=model.args.batch, workers=args.workers,
            device=args.device, project=str(output), name="test", exist_ok=False,
        )
    return model


def main(argv=None):
    args = parse_args(argv)
    try:
        train(args)
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
