#!/usr/bin/env python3
"""Step 4: Ultralytics YOLOv8s-seg training, validation & static FP32 ONNX export."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import (
    DEFAULT_CONFIG_PATH,
    atomic_write_json,
    get_pipeline_paths,
    load_config,
    setup_logging,
)

LOGGER = setup_logging("step4.train")


def resolve_training_device(requested: str) -> str | int:
    """Resolve `auto` to CUDA GPU 0, Apple Silicon MPS, or CPU."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _extract_val_metrics(val_results: Any) -> dict[str, float | None]:
    """Safely extract box and mask detection/segmentation metrics from Ultralytics Val."""
    metrics: dict[str, float | None] = {
        "box_map50": None,
        "box_map50_95": None,
        "box_precision": None,
        "box_recall": None,
        "mask_map50": None,
        "mask_map50_95": None,
        "mask_precision": None,
        "mask_recall": None,
    }
    box = getattr(val_results, "box", None)
    if box is not None:
        metrics["box_map50"] = _to_float(getattr(box, "map50", None))
        metrics["box_map50_95"] = _to_float(getattr(box, "map", None))
        metrics["box_precision"] = _to_float(getattr(box, "mp", None))
        metrics["box_recall"] = _to_float(getattr(box, "mr", None))

    seg = getattr(val_results, "seg", None)
    if seg is not None:
        metrics["mask_map50"] = _to_float(getattr(seg, "map50", None))
        metrics["mask_map50_95"] = _to_float(getattr(seg, "map", None))
        metrics["mask_precision"] = _to_float(getattr(seg, "mp", None))
        metrics["mask_recall"] = _to_float(getattr(seg, "mr", None))
    return metrics


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return round(float(val), 5)
    except (TypeError, ValueError):
        return None


def validate_dataset_ready(dataset_dir: Path) -> Path:
    """Verify `data.yaml` and non-empty train/val splits exist before launching training."""
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}. Run Step 3 first.")

    train_imgs = list((dataset_dir / "images" / "train").glob("*.png"))
    val_imgs = list((dataset_dir / "images" / "val").glob("*.png"))
    if not train_imgs:
        raise ValueError(f"No training PNGs found under {dataset_dir / 'images' / 'train'}.")
    if not val_imgs:
        raise ValueError(f"No validation PNGs found under {dataset_dir / 'images' / 'val'}.")
    return data_yaml


def train_and_validate(
    cfg: dict[str, Any],
    dataset_dir: Path,
    runs_dir: Path,
    weights_override: str | None = None,
    epochs_override: int | None = None,
    batch_override: int | None = None,
    device_override: str | None = None,
    resume: bool = False,
    skip_onnx_export: bool = False,
) -> dict[str, Any]:
    """Train YOLOv8s-seg, validate best checkpoint, and export static FP32 ONNX."""
    data_yaml = validate_dataset_ready(dataset_dir)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install with `pip install -r requirements.txt`."
        ) from exc

    tr_cfg = cfg["training"]
    hailo_cfg = cfg["hailo"]

    weights = weights_override or str(tr_cfg.get("model", "yolov8s-seg.pt"))
    run_name = str(tr_cfg.get("run_name", "storm_yolov8s_seg"))
    imgsz = int(tr_cfg.get("imgsz", cfg["region"]["image_size"]))
    epochs = epochs_override if epochs_override is not None else int(tr_cfg.get("epochs", 80))
    batch = batch_override if batch_override is not None else int(tr_cfg.get("batch", 16))
    device = resolve_training_device(device_override or str(tr_cfg.get("device", "auto")))
    aug = tr_cfg.get("augmentations", {})

    runs_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Starting YOLOv8s-seg training: weights=%s, imgsz=%d, epochs=%d, batch=%d, device=%s",
        weights,
        imgsz,
        epochs,
        batch,
        device,
    )

    model = YOLO(weights)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": int(tr_cfg.get("patience", 20)),
        "device": device,
        "workers": int(tr_cfg.get("workers", 4)),
        "seed": int(tr_cfg.get("seed", 42)),
        "optimizer": str(tr_cfg.get("optimizer", "auto")),
        "lr0": float(tr_cfg.get("lr0", 0.01)),
        "project": str(runs_dir),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "resume": resume,
        "hsv_h": float(aug.get("hsv_h", 0.0)),
        "hsv_s": float(aug.get("hsv_s", 0.0)),
        "hsv_v": float(aug.get("hsv_v", 0.10)),
        "degrees": float(aug.get("degrees", 0.0)),
        "translate": float(aug.get("translate", 0.05)),
        "scale": float(aug.get("scale", 0.20)),
        "fliplr": float(aug.get("fliplr", 0.0)),
        "flipud": float(aug.get("flipud", 0.0)),
        "mosaic": float(aug.get("mosaic", 0.50)),
        "mixup": float(aug.get("mixup", 0.0)),
    }
    model.train(**train_kwargs)

    run_dir = runs_dir / run_name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Trained checkpoint not found under {run_dir / 'weights'}")

    LOGGER.info("Validating best checkpoint: %s", best_pt)
    best_model = YOLO(str(best_pt))
    val_results = best_model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(runs_dir),
        name=f"{run_name}_val",
        exist_ok=True,
    )
    metrics = _extract_val_metrics(val_results)

    onnx_path: Path | None = None
    if not skip_onnx_export:
        LOGGER.info("Exporting static FP32 ONNX for Hailo-8 DFC (imgsz=%d, dynamic=False)...", imgsz)
        exported = best_model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False,
            opset=int(hailo_cfg.get("opset", 12)),
            simplify=bool(hailo_cfg.get("simplify_onnx", True)),
        )
        onnx_path = Path(str(exported)).resolve()
        LOGGER.info("Static ONNX exported -> %s", onnx_path)

    report = {
        "weights_initial": weights,
        "best_pt": str(best_pt.resolve()),
        "best_onnx": str(onnx_path) if onnx_path else None,
        "data_yaml": str(data_yaml.resolve()),
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "device": str(device),
        "metrics": metrics,
    }
    report_path = run_dir / "metrics.json"
    atomic_write_json(report_path, report)
    LOGGER.info(
        "Training & validation complete | Box mAP50=%s Mask mAP50=%s | Report: %s",
        metrics["box_map50"],
        metrics["mask_map50"],
        report_path,
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: Train and validate YOLOv8s-seg storm detector and export static ONNX."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Initial weights (default: yolov8s-seg.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device: auto | cpu | mps | 0")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip post-training ONNX export")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_pipeline_paths(cfg)
    train_and_validate(
        cfg=cfg,
        dataset_dir=paths["dataset_dir"],
        runs_dir=paths["runs_dir"],
        weights_override=args.weights,
        epochs_override=args.epochs,
        batch_override=args.batch,
        device_override=args.device,
        resume=args.resume,
        skip_onnx_export=args.skip_onnx,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
