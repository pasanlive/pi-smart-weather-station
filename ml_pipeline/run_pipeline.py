#!/usr/bin/env python3
"""Master orchestrator for the Himawari-9 -> YOLOv8s-seg -> Hailo-8 ML pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from utils import DEFAULT_CONFIG_PATH, PIPELINE_ROOT, setup_logging

LOGGER = setup_logging("pipeline.orchestrator")

STEPS: list[tuple[str, str]] = [
    ("harvest", "1_harvest_himawari.py"),
    ("autolabel", "2_autolabel_seg.py"),
    ("prepare", "3_prepare_dataset.py"),
    ("train", "4_train_yolov8_seg.py"),
    ("compile", "5_compile_hailo8.py"),
]
STEP_NAMES = [name for name, _ in STEPS]


def select_steps(
    explicit_steps: list[str] | None = None,
    from_step: str | None = None,
    to_step: str | None = None,
) -> list[tuple[str, str]]:
    """Resolve the ordered subset of pipeline steps to execute."""
    if explicit_steps:
        step_map = dict(STEPS)
        unknown = [s for s in explicit_steps if s not in step_map]
        if unknown:
            raise ValueError(f"Unknown steps: {unknown}. Valid steps: {STEP_NAMES}")
        return [(name, step_map[name]) for name in explicit_steps]

    start_idx = STEP_NAMES.index(from_step) if from_step else 0
    end_idx = STEP_NAMES.index(to_step) if to_step else len(STEPS) - 1
    if start_idx > end_idx:
        raise ValueError(f"--from-step ({from_step}) cannot come after --to-step ({to_step})")
    return STEPS[start_idx : end_idx + 1]


def build_step_command(
    step_name: str,
    script_file: str,
    config_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Assemble the CLI invocation for an individual pipeline module."""
    cmd = [sys.executable, str(PIPELINE_ROOT / script_file), "--config", str(config_path)]

    if step_name == "harvest":
        if args.start_date:
            cmd += ["--start-date", args.start_date]
        if args.end_date:
            cmd += ["--end-date", args.end_date]
        if args.step_hours is not None:
            cmd += ["--step-hours", str(args.step_hours)]
        if args.max_frames is not None:
            cmd += ["--max-frames", str(args.max_frames)]
    elif step_name == "prepare":
        if args.calib_frames is not None:
            cmd += ["--calib-frames", str(args.calib_frames)]
    elif step_name == "train":
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.batch is not None:
            cmd += ["--batch", str(args.batch)]
        if args.device:
            cmd += ["--device", args.device]
    elif step_name == "compile":
        if args.compile_mode:
            cmd += ["--mode", args.compile_mode]

    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end Himawari-9 storm segmentation & Hailo-8 compilation pipeline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEP_NAMES,
        default=None,
        help="Explicit list of steps to run (overrides --from-step/--to-step)",
    )
    parser.add_argument("--from-step", choices=STEP_NAMES, default=None, help="Start at step")
    parser.add_argument("--to-step", choices=STEP_NAMES, default=None, help="Stop after step")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    # Optional passthrough overrides for common knobs
    parser.add_argument("--start-date", type=str, default=None, help="[harvest] Start UTC date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="[harvest] End UTC date YYYY-MM-DD")
    parser.add_argument("--step-hours", type=int, default=None, help="[harvest] Interval hours")
    parser.add_argument("--max-frames", type=int, default=None, help="[harvest] Max frames")
    parser.add_argument("--calib-frames", type=int, default=None, help="[prepare] Calibration frames")
    parser.add_argument("--epochs", type=int, default=None, help="[train] Epochs")
    parser.add_argument("--batch", type=int, default=None, help="[train] Batch size")
    parser.add_argument("--device", type=str, default=None, help="[train] Device (auto|cpu|mps|0)")
    parser.add_argument(
        "--compile-mode",
        choices=["auto", "ultralytics", "dfc", "onnx-only"],
        default=None,
        help="[compile] Compilation mode",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    chosen = select_steps(args.steps, args.from_step, args.to_step)

    LOGGER.info("Pipeline plan (%d steps): %s", len(chosen), " -> ".join(name for name, _ in chosen))
    t0_total = time.monotonic()

    for step_name, script_file in chosen:
        cmd = build_step_command(step_name, script_file, config_path, args)
        if args.dry_run:
            LOGGER.info("[DRY-RUN] %s: %s", step_name, " ".join(cmd))
            continue

        LOGGER.info("=== Running step: %s ===", step_name)
        t0 = time.monotonic()
        completed = subprocess.run(cmd, check=False)
        elapsed = time.monotonic() - t0
        if completed.returncode != 0:
            LOGGER.error(
                "Step '%s' failed with exit code %d after %.1fs.",
                step_name,
                completed.returncode,
                elapsed,
            )
            return completed.returncode
        LOGGER.info("=== Completed step: %s (%.1fs) ===", step_name, elapsed)

    LOGGER.info("Pipeline finished successfully in %.1fs.", time.monotonic() - t0_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
