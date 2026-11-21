#!/usr/bin/env python3
"""Step 5: Dual-mode Hailo-8 INT8 .hef compiler (Ultralytics native + Direct Hailo DFC fallback)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    DEFAULT_CONFIG_PATH,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    get_pipeline_paths,
    load_config,
    setup_logging,
)

LOGGER = setup_logging("step5.compile")


def detect_environment() -> dict[str, Any]:
    """Probe host architecture and Hailo DFC / Ultralytics availability."""
    system = platform.system()
    machine = platform.machine().lower()
    is_linux_x86_64 = system == "Linux" and machine in {"x86_64", "amd64"}
    has_dfc = importlib.util.find_spec("hailo_sdk_client") is not None
    has_ultralytics = importlib.util.find_spec("ultralytics") is not None
    has_onnx = importlib.util.find_spec("onnx") is not None
    return {
        "system": system,
        "machine": machine,
        "is_linux_x86_64": is_linux_x86_64,
        "has_hailo_dfc": has_dfc,
        "has_ultralytics": has_ultralytics,
        "has_onnx": has_onnx,
    }


def generate_alls_script(cfg: dict[str, Any], output_path: Path) -> Path:
    """Write a Hailo Model Script (.alls) embedding uint8->float normalization and INT8 flavor."""
    hailo_cfg = cfg["hailo"]
    mean = hailo_cfg.get("normalization_mean", [0.0, 0.0, 0.0])
    std = hailo_cfg.get("normalization_std", [255.0, 255.0, 255.0])
    opt_lvl = int(hailo_cfg.get("optimization_level", 2))
    comp_lvl = int(hailo_cfg.get("compression_level", 1))
    compiler_opt = str(hailo_cfg.get("compiler_optimization_level", "max"))

    lines = [
        "# Auto-generated Hailo-8 Model Script for Himawari-9 YOLOv8s-seg",
        f"normalization1 = normalization({[float(x) for x in mean]}, {[float(x) for x in std]})",
        f"model_optimization_flavor(optimization_level={opt_lvl}, compression_level={comp_lvl})",
        f"performance_param(compiler_optimization_level={compiler_opt})",
    ]
    atomic_write_text(output_path, "\n".join(lines) + "\n")
    LOGGER.info("Generated Hailo model script -> %s", output_path)
    return output_path


def inspect_onnx_nodes(onnx_path: Path, expected_end_nodes: list[str]) -> tuple[str, list[str]]:
    """Inspect ONNX input tensor name and verify configured YOLOv8s-seg end node names."""
    input_name = "images"
    try:
        import onnx
    except ImportError:
        LOGGER.warning("`onnx` package not installed; skipping static end-node verification.")
        return input_name, expected_end_nodes

    model = onnx.load(str(onnx_path))
    if model.graph.input:
        input_name = model.graph.input[0].name

    available_nodes = {node.name for node in model.graph.node if node.name}
    available_outputs = {out for node in model.graph.node for out in node.output}
    missing = [
        node_name
        for node_name in expected_end_nodes
        if node_name not in available_nodes and node_name not in available_outputs
    ]
    if missing:
        candidate_heads = sorted(
            n for n in available_nodes if "/model.22/" in n and ("Conv" in n or "proto" in n)
        )
        LOGGER.warning(
            "Some configured end_node_names were not found in %s: %s. "
            "Candidate /model.22/ nodes in this ONNX: %s",
            onnx_path.name,
            missing,
            candidate_heads[:20],
        )
    else:
        LOGGER.info(
            "Verified all %d YOLOv8s-seg end nodes in %s (input=%s).",
            len(expected_end_nodes),
            onnx_path.name,
            input_name,
        )
    return input_name, expected_end_nodes


def ensure_static_onnx(
    weights_pt: Path | None,
    onnx_path: Path | None,
    compiled_dir: Path,
    imgsz: int,
    opset: int = 12,
    simplify: bool = True,
) -> Path:
    """Return an existing static FP32 ONNX file or export one from `best.pt`."""
    if onnx_path and onnx_path.exists():
        return onnx_path.resolve()

    if weights_pt is None or not weights_pt.exists():
        raise FileNotFoundError(
            f"Neither ONNX ({onnx_path}) nor PyTorch weights ({weights_pt}) exist. Run Step 4 first."
        )

    from ultralytics import YOLO

    LOGGER.info("Exporting static FP32 ONNX from %s...", weights_pt)
    model = YOLO(str(weights_pt))
    exported = Path(
        str(
            model.export(
                format="onnx",
                imgsz=imgsz,
                dynamic=False,
                opset=opset,
                simplify=simplify,
            )
        )
    ).resolve()

    compiled_dir.mkdir(parents=True, exist_ok=True)
    target_onnx = compiled_dir / exported.name
    if exported != target_onnx:
        shutil.copy2(exported, target_onnx)
    return target_onnx


def compile_via_ultralytics(
    weights_pt: Path,
    data_yaml: Path,
    output_hef: Path,
    hw_arch: str,
    imgsz: int,
) -> Path:
    """Mode 1: Ultralytics native `model.export(format='hailo')`."""
    from ultralytics import YOLO

    if not weights_pt.exists():
        raise FileNotFoundError(f"Trained weights not found: {weights_pt}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    LOGGER.info(
        "Attempting Ultralytics native Hailo export (format='hailo', name=%s, imgsz=%d)...",
        hw_arch,
        imgsz,
    )
    model = YOLO(str(weights_pt))
    export_result = model.export(
        format="hailo",
        name=hw_arch,
        imgsz=imgsz,
        data=str(data_yaml),
    )

    exported_path = Path(str(export_result)).resolve()
    candidate_hef: Path | None = None
    if exported_path.is_file() and exported_path.suffix == ".hef":
        candidate_hef = exported_path
    elif exported_path.is_dir():
        hefs = sorted(exported_path.rglob("*.hef"))
        if hefs:
            candidate_hef = hefs[0]
    else:
        sibling_hefs = sorted(weights_pt.parent.rglob("*.hef"))
        if sibling_hefs:
            candidate_hef = sibling_hefs[-1]

    if candidate_hef is None or not candidate_hef.exists() or candidate_hef.stat().st_size == 0:
        raise RuntimeError(
            f"Ultralytics Hailo export completed without producing a valid .hef (got: {export_result})."
        )

    output_hef.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_hef, output_hef)
    LOGGER.info("Ultralytics native compilation succeeded -> %s", output_hef)
    return output_hef


def compile_via_dfc(
    cfg: dict[str, Any],
    onnx_path: Path,
    calib_npy: Path,
    alls_path: Path,
    compiled_dir: Path,
    output_hef: Path,
) -> Path:
    """Mode 2: Direct Hailo Dataflow Compiler (`hailo_sdk_client.ClientRunner`)."""
    from hailo_sdk_client import ClientRunner

    if not calib_npy.exists():
        raise FileNotFoundError(f"Calibration .npy not found: {calib_npy}. Run Step 3 first.")

    calib_data = np.load(calib_npy)
    if calib_data.ndim != 4 or calib_data.shape[-1] != 3:
        raise ValueError(
            f"Expected calibration array shape (N, H, W, 3), got {calib_data.shape}"
        )
    if calib_data.dtype != np.uint8:
        LOGGER.warning("Casting calibration array from %s to uint8.", calib_data.dtype)
        calib_data = np.clip(calib_data, 0, 255).astype(np.uint8)

    hailo_cfg = cfg["hailo"]
    hw_arch = str(hailo_cfg.get("hw_arch", "hailo8"))
    net_name = str(hailo_cfg.get("net_name", "storm_yolov8s_seg"))
    imgsz = int(cfg["region"]["image_size"])
    end_nodes = list(hailo_cfg["end_node_names"])
    input_name, end_nodes = inspect_onnx_nodes(onnx_path, end_nodes)

    LOGGER.info(
        "Running Hailo DFC ClientRunner(hw_arch=%s) on %s with %d calibration frames...",
        hw_arch,
        onnx_path.name,
        calib_data.shape[0],
    )
    runner = ClientRunner(hw_arch=hw_arch)

    # 1. Translate ONNX -> HAR
    runner.translate_onnx_model(
        str(onnx_path),
        net_name,
        start_node_names=[input_name],
        end_node_names=end_nodes,
        net_input_shapes={input_name: [1, 3, imgsz, imgsz]},
    )
    parsed_har = compiled_dir / f"{net_name}_parsed.har"
    runner.save_har(str(parsed_har))
    LOGGER.info("Saved parsed HAR -> %s", parsed_har)

    # 2. Load embedded normalization & optimization script, then run INT8 calibration
    runner.load_model_script(str(alls_path))
    runner.optimize(calib_data)
    quantized_har = compiled_dir / f"{net_name}_quantized.har"
    runner.save_har(str(quantized_har))
    LOGGER.info("Saved INT8 quantized HAR -> %s", quantized_har)

    # 3. Compile quantized HAR -> Hailo-8 HEF binary
    hef_bytes = runner.compile()
    if not hef_bytes:
        raise RuntimeError("Hailo DFC runner.compile() returned empty HEF payload.")

    atomic_write_bytes(output_hef, hef_bytes)
    LOGGER.info(
        "Direct Hailo DFC compilation succeeded -> %s (%.2f MB)",
        output_hef,
        output_hef.stat().st_size / (1024.0 * 1024.0),
    )
    return output_hef


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_hailo8(
    cfg: dict[str, Any],
    mode: str = "auto",
    weights_pt: Path | None = None,
    onnx_path: Path | None = None,
    calib_npy: Path | None = None,
    data_yaml: Path | None = None,
    output_hef: Path | None = None,
    skip_deploy_copy: bool = False,
) -> dict[str, Any]:
    """Compile trained YOLOv8s-seg model to Hailo-8 INT8 .hef with dual-mode fallback."""
    paths = get_pipeline_paths(cfg)
    hailo_cfg = cfg["hailo"]
    tr_cfg = cfg["training"]

    compiled_dir = paths["compiled_dir"]
    compiled_dir.mkdir(parents=True, exist_ok=True)

    run_name = str(tr_cfg.get("run_name", "storm_yolov8s_seg"))
    default_weights = paths["runs_dir"] / run_name / "weights" / "best.pt"
    default_onnx = paths["runs_dir"] / run_name / "weights" / "best.onnx"
    resolved_weights = weights_pt.resolve() if weights_pt else default_weights
    resolved_onnx = onnx_path.resolve() if onnx_path else (default_onnx if default_onnx.exists() else None)
    resolved_calib = calib_npy.resolve() if calib_npy else (paths["calibration_dir"] / "calib_set.npy")
    resolved_data_yaml = data_yaml.resolve() if data_yaml else (paths["dataset_dir"] / "data.yaml")
    resolved_hef = (
        output_hef.resolve()
        if output_hef
        else (compiled_dir / str(hailo_cfg.get("output_hef_name", "storm_yolov8s_seg.hef")))
    )

    hw_arch = str(hailo_cfg.get("hw_arch", "hailo8"))
    imgsz = int(cfg["region"]["image_size"])
    alls_path = generate_alls_script(cfg, compiled_dir / f"{hailo_cfg.get('net_name', 'storm_yolov8s_seg')}.alls")

    env = detect_environment()
    LOGGER.info(
        "Environment probe: os=%s arch=%s hailo_dfc=%s ultralytics=%s",
        env["system"],
        env["machine"],
        env["has_hailo_dfc"],
        env["has_ultralytics"],
    )

    if mode == "onnx-only":
        static_onnx = ensure_static_onnx(
            weights_pt=resolved_weights,
            onnx_path=resolved_onnx,
            compiled_dir=compiled_dir,
            imgsz=imgsz,
            opset=int(hailo_cfg.get("opset", 12)),
            simplify=bool(hailo_cfg.get("simplify_onnx", True)),
        )
        inspect_onnx_nodes(static_onnx, list(hailo_cfg["end_node_names"]))
        meta = {
            "status": "onnx_ready",
            "mode_used": "onnx-only",
            "onnx_path": str(static_onnx),
            "alls_path": str(alls_path),
            "calib_npy": str(resolved_calib),
            "hw_arch": hw_arch,
        }
        atomic_write_json(compiled_dir / "compile_metadata.json", meta)
        LOGGER.info("ONNX + .alls prepared. Run `./docker/compile_in_docker.sh` on x86_64 Linux to build .hef.")
        return meta

    if not env["has_hailo_dfc"]:
        hint = (
            "Hailo Dataflow Compiler (`hailo_sdk_client`) is not available in this Python environment "
            f"(host: {env['system']} {env['machine']}).\n"
            "Hailo DFC requires x86_64 Linux (Hailo AI SW Suite).\n"
            "Options:\n"
            "  1. Run with `--mode onnx-only` here to prepare static ONNX + .alls.\n"
            "  2. Run `ml_pipeline/docker/compile_in_docker.sh` on an x86_64 Linux host "
            "(see `ml_pipeline/docker/README.md`)."
        )
        raise RuntimeError(hint)

    mode_used: str
    if mode == "ultralytics":
        compile_via_ultralytics(resolved_weights, resolved_data_yaml, resolved_hef, hw_arch, imgsz)
        mode_used = "ultralytics"
    elif mode == "dfc":
        static_onnx = ensure_static_onnx(
            weights_pt=resolved_weights,
            onnx_path=resolved_onnx,
            compiled_dir=compiled_dir,
            imgsz=imgsz,
            opset=int(hailo_cfg.get("opset", 12)),
            simplify=bool(hailo_cfg.get("simplify_onnx", True)),
        )
        compile_via_dfc(cfg, static_onnx, resolved_calib, alls_path, compiled_dir, resolved_hef)
        mode_used = "dfc"
    elif mode == "auto":
        try:
            if not env["has_ultralytics"] or not resolved_weights.exists():
                raise RuntimeError(
                    "Ultralytics or best.pt unavailable; falling back directly to Hailo DFC ClientRunner."
                )
            compile_via_ultralytics(resolved_weights, resolved_data_yaml, resolved_hef, hw_arch, imgsz)
            mode_used = "ultralytics"
        except Exception as exc:
            LOGGER.warning(
                "Ultralytics native `format='hailo'` failed (%s). Falling back to direct Hailo DFC ClientRunner...",
                exc,
            )
            static_onnx = ensure_static_onnx(
                weights_pt=resolved_weights,
                onnx_path=resolved_onnx,
                compiled_dir=compiled_dir,
                imgsz=imgsz,
                opset=int(hailo_cfg.get("opset", 12)),
                simplify=bool(hailo_cfg.get("simplify_onnx", True)),
            )
            compile_via_dfc(cfg, static_onnx, resolved_calib, alls_path, compiled_dir, resolved_hef)
            mode_used = "dfc"
    else:
        raise ValueError(f"Unsupported compilation mode: {mode}")

    deployed_path: Path | None = None
    deploy_target = paths.get("deploy_hef_path")
    if not skip_deploy_copy and deploy_target is not None:
        deploy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_hef, deploy_target)
        deployed_path = deploy_target
        LOGGER.info("Copied compiled HEF for Pi 5 deployment -> %s", deploy_target)

    calib_frames = None
    if resolved_calib.exists():
        calib_frames = int(np.load(resolved_calib, mmap_mode="r").shape[0])

    metadata = {
        "status": "compiled",
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode_used": mode_used,
        "hw_arch": hw_arch,
        "imgsz": imgsz,
        "hef_path": str(resolved_hef),
        "deploy_hef_path": str(deployed_path) if deployed_path else None,
        "hef_size_bytes": resolved_hef.stat().st_size,
        "hef_sha256": sha256_file(resolved_hef),
        "alls_path": str(alls_path),
        "calib_npy": str(resolved_calib),
        "calib_num_frames": calib_frames,
        "classes": {
            str(int(k)): v["name"]
            for k, v in sorted(cfg["autolabel"]["classes"].items(), key=lambda kv: int(kv[0]))
        },
    }
    meta_path = compiled_dir / "compile_metadata.json"
    atomic_write_json(meta_path, metadata)
    LOGGER.info("Compilation metadata written -> %s", meta_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5: Compile YOLOv8s-seg to Hailo-8 INT8 .hef (Ultralytics native + DFC fallback)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument(
        "--mode",
        choices=["auto", "ultralytics", "dfc", "onnx-only"],
        default=None,
        help="Compilation mode (default from config.yaml: auto)",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Path to trained best.pt")
    parser.add_argument("--onnx", type=Path, default=None, help="Path to static FP32 best.onnx")
    parser.add_argument("--calib-npy", type=Path, default=None, help="Path to calibration calib_set.npy")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Path to dataset data.yaml")
    parser.add_argument("--output-hef", type=Path, default=None, help="Output .hef file path")
    parser.add_argument("--skip-deploy-copy", action="store_true", help="Do not copy .hef to ../models/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    mode = args.mode or str(cfg["hailo"].get("mode", "auto"))
    compile_hailo8(
        cfg=cfg,
        mode=mode,
        weights_pt=args.weights,
        onnx_path=args.onnx,
        calib_npy=args.calib_npy,
        data_yaml=args.data_yaml,
        output_hef=args.output_hef,
        skip_deploy_copy=args.skip_deploy_copy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
