"""Automated verification suite for the Himawari-9 -> YOLOv8s-seg -> Hailo-8 ML pipeline."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest
import yaml

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from utils import format_frame_filename, load_config, parse_frame_timestamp  # noqa: E402


def _load_step_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, PIPELINE_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


step1 = _load_step_module("1_harvest_himawari.py", "step1_harvest_himawari")
step2 = _load_step_module("2_autolabel_seg.py", "step2_autolabel_seg")
step3 = _load_step_module("3_prepare_dataset.py", "step3_prepare_dataset")
step4 = _load_step_module("4_train_yolov8_seg.py", "step4_train_yolov8_seg")
step5 = _load_step_module("5_compile_hailo8.py", "step5_compile_hailo8")
orchestrator = _load_step_module("run_pipeline.py", "pipeline_orchestrator")


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _make_synthetic_ir_frame(
    size: int = 640,
    with_deep: bool = True,
    with_shield: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Create a realistic synthetic 640x640 grayscale Band 13 IR frame."""
    rng = np.random.default_rng(seed)
    # Warm ocean/land background (intensity ~35..75)
    img = rng.integers(35, 75, size=(size, size), dtype=np.uint8)

    if with_shield:
        cv2.circle(img, (320, 320), 75, 190, -1)
        cv2.ellipse(img, (210, 410), (55, 35), 25, 0, 360, 185, -1)
    if with_deep:
        cv2.circle(img, (320, 320), 28, 238, -1)
        # Tiny 2x2 speckle noise that morphological open + min_area must suppress
        img[10:12, 10:12] = 245

    return img


def _make_temp_config(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    cfg = load_config(PIPELINE_DIR / "config.yaml")
    cfg["_base_dir"] = str(tmp_path)
    cfg["paths"] = {
        "work_dir": str(tmp_path / "work"),
        "raw_frames_dir": str(tmp_path / "work" / "raw_frames"),
        "autolabel_dir": str(tmp_path / "work" / "autolabeled"),
        "dataset_dir": str(tmp_path / "work" / "dataset"),
        "calibration_dir": str(tmp_path / "work" / "calibration"),
        "runs_dir": str(tmp_path / "work" / "runs"),
        "compiled_dir": str(tmp_path / "work" / "compiled"),
        "deploy_hef_path": str(tmp_path / "models" / "storm_yolov8s_seg.hef"),
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg, cfg_path


def test_config_and_orchestrator_dry_run():
    cfg = load_config(PIPELINE_DIR / "config.yaml")
    assert cfg["region"]["bbox"] == [75.0, 3.0, 86.0, 12.0]
    assert cfg["region"]["image_size"] == 640
    assert cfg["autolabel"]["classes"][0]["name"] == "deep_convective"
    assert cfg["autolabel"]["classes"][0]["min_intensity"] == 215
    assert cfg["autolabel"]["classes"][0]["min_area_px"] == 120
    assert cfg["autolabel"]["classes"][1]["name"] == "storm_shield"
    assert cfg["autolabel"]["classes"][1]["min_intensity"] == 165
    assert cfg["autolabel"]["classes"][1]["min_area_px"] == 400
    assert cfg["hailo"]["hw_arch"] == "hailo8"

    rc = orchestrator.main(["--dry-run", "--from-step", "harvest", "--to-step", "compile"])
    assert rc == 0


def test_step1_wms_params_validation_and_resumable_harvest(tmp_path: Path):
    cfg, _ = _make_temp_config(tmp_path)
    dt = datetime(2026, 10, 15, 6, 0, tzinfo=timezone.utc)
    params = step1.build_wms_params(cfg, dt)
    assert params["LAYERS"] == "Himawari_AHI_Band13_Clean_Infrared"
    assert params["SRS"] == "EPSG:4326"
    assert params["BBOX"] == "75.0,3.0,86.0,12.0"
    assert params["WIDTH"] == "640"
    assert params["HEIGHT"] == "640"
    assert params["TIME"] == "2026-10-15T06:00:00Z"

    valid_png = _encode_png(_make_synthetic_ir_frame(seed=1))
    black_png = _encode_png(np.zeros((640, 640), dtype=np.uint8))
    flat_png = _encode_png(np.full((640, 640), 120, dtype=np.uint8))

    assert step1.validate_png_bytes(valid_png, expected_size=640)[0] is True
    assert step1.validate_png_bytes(black_png, expected_size=640)[0] is False
    assert step1.validate_png_bytes(flat_png, expected_size=640)[0] is False

    # Mock NASA GIBS WMS responses:
    # 00:00Z -> valid PNG, 02:00Z -> all-black PNG (discarded), 04:00Z -> XML ServiceException (failed)
    call_log: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ts = request.url.params.get("TIME", "")
        call_log.append(ts)
        if ts.endswith("00:00:00Z"):
            return httpx.Response(200, content=valid_png, headers={"content-type": "image/png"})
        if ts.endswith("10:00:00Z"):
            return httpx.Response(200, content=black_png, headers={"content-type": "image/png"})
        return httpx.Response(
            200,
            content=b"<ServiceExceptionReport>Invalid TIME</ServiceExceptionReport>",
            headers={"content-type": "application/vnd.ogc.se_xml"},
        )

    transport = httpx.MockTransport(handler)
    out_dir = Path(cfg["paths"]["raw_frames_dir"])

    manifest1 = asyncio.run(
        step1.harvest_himawari_async(
            cfg=cfg,
            start_date=date(2026, 10, 15),
            end_date=date(2026, 10, 15),
            step_hours=2,
            max_frames=3,
            concurrency=2,
            out_dir=out_dir,
            force=False,
            transport=transport,
        )
    )
    assert manifest1["counts"]["downloaded"] == 1
    assert manifest1["counts"]["discarded"] == 1
    assert manifest1["counts"]["failed"] == 1
    assert (out_dir / "harvest_manifest.json").exists()

    # Second run must resume the valid 00:00Z frame without issuing an HTTP request for it
    call_log.clear()
    manifest2 = asyncio.run(
        step1.harvest_himawari_async(
            cfg=cfg,
            start_date=date(2026, 10, 15),
            end_date=date(2026, 10, 15),
            step_hours=2,
            max_frames=3,
            concurrency=2,
            out_dir=out_dir,
            force=False,
            transport=transport,
        )
    )
    assert manifest2["counts"]["resumed"] == 1
    assert not any(ts.endswith("00:00:00Z") for ts in call_log)


def test_step2_autolabel_and_step3_temporal_split_and_calibration(tmp_path: Path):
    cfg, _ = _make_temp_config(tmp_path)
    raw_dir = Path(cfg["paths"]["raw_frames_dir"])
    autolabel_dir = Path(cfg["paths"]["autolabel_dir"])
    dataset_dir = Path(cfg["paths"]["dataset_dir"])
    calib_dir = Path(cfg["paths"]["calibration_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create 48 frames across 6 days (every 3 hours):
    # 20 deep+shield, 12 shield-only, 16 clear-sky
    base_dt = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
    for i in range(48):
        dt = base_dt + timedelta(hours=3 * i)
        if i % 3 == 0:
            frame = _make_synthetic_ir_frame(with_deep=True, with_shield=True, seed=100 + i)
        elif i % 3 == 1:
            frame = _make_synthetic_ir_frame(with_deep=False, with_shield=True, seed=200 + i)
        else:
            frame = _make_synthetic_ir_frame(with_deep=False, with_shield=False, seed=300 + i)
        cv2.imwrite(str(raw_dir / format_frame_filename(dt)), frame)

    # Step 2: Auto-label
    al_manifest = step2.autolabel_frames(cfg, raw_dir, autolabel_dir)
    summary = al_manifest["summary"]
    assert summary["positive_frames"] == 32
    assert summary["total_clear_sky_frames"] == 16
    # ~10% negative retention -> 3 or 4 clear-sky frames retained out of 35-36 total
    assert 2 <= summary["retained_clear_sky_frames"] <= 5
    assert summary["instances_by_class"]["0"] == 16  # 16 deep convective cores (noise speck ignored!)
    assert summary["instances_by_class"]["1"] >= 32  # shield polygons present

    # Validate normalized polygon coordinate bounds in every generated .txt file
    labels_dir = Path(al_manifest["labels_dir"])
    for rec in al_manifest["retained_frames"]:
        lbl_path = labels_dir / f"{rec['stem']}.txt"
        assert lbl_path.exists()
        content = lbl_path.read_text(encoding="utf-8").strip()
        if not rec["is_positive"]:
            assert content == ""
            continue
        for line in content.splitlines():
            parts = line.split()
            cid = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            assert cid in {0, 1}
            assert len(coords) >= 6 and len(coords) % 2 == 0
            assert all(0.0 <= c <= 1.0 for c in coords)

    qa_img = cv2.imread(al_manifest["qa_preview"])
    assert qa_img is not None and qa_img.shape[0] > 320 and qa_img.shape[1] > 320

    # Step 3: Temporally grouped 80/20 split + data.yaml + .npy calibration set
    split_summary = step3.prepare_dataset(
        cfg=cfg,
        raw_frames_dir=raw_dir,
        autolabel_dir=autolabel_dir,
        dataset_dir=dataset_dir,
        calibration_dir=calib_dir,
        calib_num_frames_override=20,
    )
    assert split_summary["train_count"] > 0
    assert split_summary["val_count"] > 0

    train_pngs = sorted((dataset_dir / "images" / "train").glob("*.png"))
    val_pngs = sorted((dataset_dir / "images" / "val").glob("*.png"))
    assert len(train_pngs) == split_summary["train_count"]
    assert len(val_pngs) == split_summary["val_count"]

    # Verify zero temporal leakage across train <-> val (min_temporal_gap_hours = 4.0h)
    train_times = [parse_frame_timestamp(p.name) for p in train_pngs]
    val_times = [parse_frame_timestamp(p.name) for p in val_pngs]
    for t_tr in train_times:
        for t_va in val_times:
            assert t_tr is not None and t_va is not None
            diff_hours = abs((t_tr - t_va).total_seconds()) / 3600.0
            assert diff_hours >= 4.0, f"Temporal leakage detected: {t_tr} vs {t_va}"

    # Verify data.yaml
    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == {0: "deep_convective", 1: "storm_shield"}
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/val"

    # Verify uncropped RGB uint8 calibration .npy array
    calib_arr = np.load(calib_dir / "calib_set.npy")
    assert calib_arr.shape == (20, 640, 640, 3)
    assert calib_arr.dtype == np.uint8


def test_step4_train_and_validate_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, _ = _make_temp_config(tmp_path)
    dataset_dir = Path(cfg["paths"]["dataset_dir"])
    runs_dir = Path(cfg["paths"]["runs_dir"])
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    cv2.imwrite(str(dataset_dir / "images" / "train" / "a.png"), _make_synthetic_ir_frame(seed=1))
    cv2.imwrite(str(dataset_dir / "images" / "val" / "b.png"), _make_synthetic_ir_frame(seed=2))
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    class FakeYOLO:
        def __init__(self, weights: str):
            self.weights = weights

        def train(self, **kwargs):
            captured["train_kwargs"] = kwargs
            w_dir = Path(kwargs["project"]) / kwargs["name"] / "weights"
            w_dir.mkdir(parents=True, exist_ok=True)
            (w_dir / "best.pt").write_bytes(b"FAKE_PT_WEIGHTS")
            return types.SimpleNamespace()

        def val(self, **kwargs):
            captured["val_kwargs"] = kwargs
            return types.SimpleNamespace(
                box=types.SimpleNamespace(map50=0.88, map=0.64, mp=0.86, mr=0.82),
                seg=types.SimpleNamespace(map50=0.84, map=0.59, mp=0.83, mr=0.79),
            )

        def export(self, **kwargs):
            captured["export_kwargs"] = kwargs
            out = Path(self.weights).with_suffix(".onnx")
            out.write_bytes(b"FAKE_ONNX")
            return str(out)

    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    report = step4.train_and_validate(cfg, dataset_dir, runs_dir, epochs_override=5, device_override="cpu")
    assert captured["train_kwargs"]["epochs"] == 5
    assert captured["train_kwargs"]["hsv_h"] == 0.0
    assert captured["export_kwargs"]["format"] == "onnx"
    assert captured["export_kwargs"]["dynamic"] is False
    assert report["metrics"]["box_map50"] == 0.88
    assert report["metrics"]["mask_map50"] == 0.84
    assert (runs_dir / "storm_yolov8s_seg" / "metrics.json").exists()


def test_step5_dual_mode_ultralytics_fallback_to_dfc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg, _ = _make_temp_config(tmp_path)
    runs_weights = Path(cfg["paths"]["runs_dir"]) / "storm_yolov8s_seg" / "weights"
    runs_weights.mkdir(parents=True)
    best_pt = runs_weights / "best.pt"
    best_onnx = runs_weights / "best.onnx"
    best_pt.write_bytes(b"FAKE_PT")
    best_onnx.write_bytes(b"FAKE_ONNX")

    ds_dir = Path(cfg["paths"]["dataset_dir"])
    ds_dir.mkdir(parents=True)
    (ds_dir / "data.yaml").write_text("path: .\n", encoding="utf-8")

    calib_dir = Path(cfg["paths"]["calibration_dir"])
    calib_dir.mkdir(parents=True)
    calib_arr = np.random.default_rng(7).integers(0, 255, size=(16, 640, 640, 3), dtype=np.uint8)
    np.save(calib_dir / "calib_set.npy", calib_arr)

    dfc_calls: list[str] = []

    class FailingHailoYOLO:
        def __init__(self, weights: str):
            self.weights = weights

        def export(self, **kwargs):
            if kwargs.get("format") == "hailo":
                raise RuntimeError("Simulated Ultralytics hailo export failure")
            return str(best_onnx)

    class FakeClientRunner:
        def __init__(self, hw_arch: str):
            assert hw_arch == "hailo8"
            dfc_calls.append("init")

        def translate_onnx_model(self, onnx_path, net_name, **kwargs):
            assert len(kwargs["end_node_names"]) == 10
            dfc_calls.append("translate")

        def save_har(self, path: str):
            Path(path).write_bytes(b"FAKE_HAR")
            dfc_calls.append("save_har")

        def load_model_script(self, path: str):
            script_text = Path(path).read_text(encoding="utf-8")
            assert "normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])" in script_text
            dfc_calls.append("load_script")

        def optimize(self, calib_data: np.ndarray):
            assert calib_data.shape == (16, 640, 640, 3)
            assert calib_data.dtype == np.uint8
            dfc_calls.append("optimize")

        def compile(self) -> bytes:
            dfc_calls.append("compile")
            return b"FAKE_HAILO8_HEF_BINARY_DATA"

    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.YOLO = FailingHailoYOLO  # type: ignore[attr-defined]
    fake_dfc = types.ModuleType("hailo_sdk_client")
    fake_dfc.ClientRunner = FakeClientRunner  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    monkeypatch.setitem(sys.modules, "hailo_sdk_client", fake_dfc)
    monkeypatch.setattr(
        step5,
        "detect_environment",
        lambda: {
            "system": "Linux",
            "machine": "x86_64",
            "is_linux_x86_64": True,
            "has_hailo_dfc": True,
            "has_ultralytics": True,
            "has_onnx": False,
        },
    )
    monkeypatch.setattr(
        step5,
        "inspect_onnx_nodes",
        lambda onnx_path, end_nodes: ("images", end_nodes),
    )

    meta = step5.compile_hailo8(cfg=cfg, mode="auto")
    assert meta["mode_used"] == "dfc"
    assert meta["hw_arch"] == "hailo8"
    assert meta["calib_num_frames"] == 16
    assert dfc_calls == ["init", "translate", "save_har", "load_script", "optimize", "save_har", "compile"]
    assert Path(meta["hef_path"]).read_bytes() == b"FAKE_HAILO8_HEF_BINARY_DATA"
    assert Path(meta["deploy_hef_path"]).read_bytes() == b"FAKE_HAILO8_HEF_BINARY_DATA"
    assert len(meta["hef_sha256"]) == 64

    # Also verify Ultralytics native happy path (`mode="ultralytics"`)
    class NativeHailoYOLO:
        def __init__(self, weights: str):
            self.weights = weights

        def export(self, **kwargs):
            assert kwargs["format"] == "hailo"
            assert kwargs["name"] == "hailo8"
            out = Path(self.weights).with_suffix(".hef")
            out.write_bytes(b"NATIVE_ULTRALYTICS_HEF")
            return str(out)

    fake_ultra.YOLO = NativeHailoYOLO  # type: ignore[attr-defined]
    meta_native = step5.compile_hailo8(cfg=cfg, mode="ultralytics")
    assert meta_native["mode_used"] == "ultralytics"
    assert Path(meta_native["hef_path"]).read_bytes() == b"NATIVE_ULTRALYTICS_HEF"

