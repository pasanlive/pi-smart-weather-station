from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from active_learning.pseudo_label import detections_to_yolo_seg, parse_yolo_seg, validate_yolo_seg
from active_learning.sampler import ActiveLearningSampler
from api.active_learning import create_active_learning_router
from config import Settings
from models import RiverBasinRisk, RiverGaugeReading, StormDetection
from storage.db import Database


def _make_ir_frame(with_bright_storm: bool = True) -> np.ndarray:
    """Generates a realistic synthetic 640x640 Himawari-9 Band 13 Clean IR frame."""
    rng = np.random.default_rng(42)
    frame = rng.integers(30, 75, size=(640, 640, 3), dtype=np.uint8)
    if with_bright_storm:
        # Bright cold cloud top over Kalu/Kelani region
        cv2.circle(frame, (310, 370), 55, (185, 185, 185), -1)
        cv2.circle(frame, (315, 375), 26, (238, 238, 238), -1)
    return frame


def _make_detection(
    *,
    cell_id: str = "cell_1",
    label: str = "deep_convective",
    confidence: float = 0.42,
    px_poly: list[list[float]] | None = None,
    geo_poly: list[list[float]] | None = None,
) -> StormDetection:
    poly = px_poly or [[280.0, 340.0], [340.0, 340.0], [340.0, 400.0], [280.0, 400.0]]
    norm = [[round(x / 640.0, 5), round(y / 640.0, 5)] for x, y in poly]
    # Default geo polygon sits over ocean west of Sri Lanka unless overridden
    gpoly = geo_poly or [
        [77.0, 6.0],
        [77.4, 6.0],
        [77.4, 6.4],
        [77.0, 6.4],
        [77.0, 6.0],
    ]
    return StormDetection(
        cell_id=cell_id,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        pixel_polygon=poly,
        normalized_polygon=norm,
        geo_polygon=gpoly,
        centroid_px=[310.0, 370.0],
        centroid_geo=[77.2, 6.2],
        area_px=3600.0,
        peak_intensity=232.0,
    )


def _make_kalu_basin_risk(deep_convective_pct: float = 0.0) -> RiverBasinRisk:
    return RiverBasinRisk(
        basin_id="kalu",
        name="Kalu Ganga",
        risk_level="WATCH",
        deep_convective_pct=deep_convective_pct,
        storm_coverage_pct=deep_convective_pct,
        rainfall_24h_mm=78.5,
    )


def _make_kalu_gauge(trend: str = "RISING", rate_ft_hr: float = 0.45) -> RiverGaugeReading:
    now = datetime.now(timezone.utc).isoformat()
    return RiverGaugeReading(
        station_id="ratnapura",
        station_name="Ratnapura",
        river_name="Kalu Ganga",
        basin_id="kalu",
        role="upstream",
        lat=6.6828,
        lon=80.3992,
        water_level_ft=18.2,
        alert_level_ft=17.0,
        minor_flood_ft=24.0,
        major_flood_ft=31.0,
        rainfall_24h_mm=85.0,
        reading_timestamp=now,
        fetched_at=now,
        rate_of_rise_ft_hr=rate_ft_hr,
        trend=trend,  # type: ignore[arg-type]
        status="ALERT",
    )


def _make_sampler(tmp_path: Path, **overrides) -> ActiveLearningSampler:
    cfg_kwargs = {
        "data_dir": str(tmp_path / "data"),
        "al_enabled": True,
        "al_conf_min": 0.25,
        "al_conf_max": 0.55,
        "al_heavy_rain_mm_hr": 20.0,
        "al_gauge_rise_ft_hr": 0.3,
        "al_capture_boundary_artifacts": True,
        "al_max_daily_captures": 48,
        "al_max_storage_mb": 2048.0,
    }
    cfg_kwargs.update(overrides)
    settings = Settings(**cfg_kwargs)
    db = Database(settings.db_path)
    return ActiveLearningSampler(settings=settings, db=db)


# -----------------------------------------------------------------------------
# 1. Trigger Rules: low_conf, sensor_mismatch, boundary_artifact
# -----------------------------------------------------------------------------
def test_low_conf_trigger_band(tmp_path: Path):
    sampler = _make_sampler(tmp_path)
    frame = _make_ir_frame()

    # High confidence (> 0.55) should NOT capture
    high_det = _make_detection(confidence=0.82)
    res_high = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T01:00:00Z",
        detections=[high_det],
        backend_name="hailo-8",
    )
    assert res_high is None

    # Too low (< 0.25) should NOT capture
    very_low_det = _make_detection(confidence=0.12)
    res_very_low = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T01:10:00Z",
        detections=[very_low_det],
        backend_name="hailo-8",
    )
    assert res_very_low is None

    # Inside [0.25, 0.55] MUST capture as low_conf
    uncertain_det = _make_detection(confidence=0.38)
    sample = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T01:20:00Z",
        detections=[uncertain_det],
        backend_name="hailo-8",
    )
    assert sample is not None
    assert sample.trigger_type == "low_conf"
    assert "0.38" in sample.trigger_detail
    assert sample.status == "pending"

    # Verify all 3 files exist on disk under /pending/{timestamp}_{trigger}/
    sample_dir = sampler.pending_dir / sample.id
    assert (sample_dir / "frame.png").exists()
    assert (sample_dir / "label.txt").exists()
    assert (sample_dir / "meta.json").exists()

    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["trigger_type"] == "low_conf"
    assert meta["confidences"] == [0.38]
    assert "storm_yolov8s_seg.hef" in meta["model_filename"]


def test_sensor_mismatch_trigger_rain_and_gauge(tmp_path: Path):
    sampler = _make_sampler(tmp_path)
    frame = _make_ir_frame(with_bright_storm=True)

    # Case A: Open-Meteo heavy rainfall >= 20 mm/hr in Kalu with 0 deep_convective over Kalu -> MUST capture
    sample_rain = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T02:00:00Z",
        detections=[],
        backend_name="hailo-8",
        basin_risks=[_make_kalu_basin_risk(deep_convective_pct=0.0)],
        basin_rainfall_1h_mm={"kalu": 24.5},
        basin_rainfall_24h_mm={"kalu": 92.0},
        gauges=[],
    )
    assert sample_rain is not None
    assert sample_rain.trigger_type == "sensor_mismatch"
    assert "Kalu" in sample_rain.trigger_detail
    assert sample_rain.meta["pseudo_label_source"] == "cpu_ir_fallback"
    # CPUIRStormDetector fallback should have extracted pseudo-label polygons from the bright IR core
    assert len(sample_rain.polygons) >= 1
    assert any(p.label == "deep_convective" for p in sample_rain.polygons)

    # Case B: Irrigation ArcGIS gauge RISING >= 0.3 ft/hr in Kalu with 0 deep_convective -> MUST capture
    sample_gauge = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T02:10:00Z",
        detections=[],
        backend_name="hailo-8",
        basin_risks=[_make_kalu_basin_risk(deep_convective_pct=0.0)],
        basin_rainfall_1h_mm={"kalu": 5.0},
        gauges=[_make_kalu_gauge(trend="RISING", rate_ft_hr=0.45)],
    )
    assert sample_gauge is not None
    assert sample_gauge.trigger_type == "sensor_mismatch"
    assert sample_gauge.meta["basins_snapshot"]["kalu"]["gauges"][0]["station_name"] == "Ratnapura"

    # Case C: Heavy rain + rising gauge, BUT vision model DID detect deep_convective over Kalu -> NO capture
    kalu_poly_geo = [
        [80.1, 6.5],
        [80.5, 6.5],
        [80.5, 6.75],
        [80.1, 6.75],
        [80.1, 6.5],
    ]
    confident_kalu_det = _make_detection(confidence=0.88, geo_poly=kalu_poly_geo)
    no_mismatch = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T02:20:00Z",
        detections=[confident_kalu_det],
        backend_name="hailo-8",
        basin_risks=[_make_kalu_basin_risk(deep_convective_pct=42.0)],
        basin_rainfall_1h_mm={"kalu": 28.0},
        gauges=[_make_kalu_gauge(trend="RISING", rate_ft_hr=0.6)],
    )
    assert no_mismatch is None


def test_boundary_artifact_optional_trigger(tmp_path: Path):
    frame = _make_ir_frame()
    # Polygon touches left edge (x=1) with confidence below AL_CONF_MIN (0.20 <= 0.55)
    edge_det = _make_detection(
        confidence=0.20,
        px_poly=[[1.0, 200.0], [50.0, 200.0], [50.0, 260.0], [1.0, 260.0]],
    )

    sampler_on = _make_sampler(tmp_path / "on", al_capture_boundary_artifacts=True)
    captured = sampler_on.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T03:00:00Z",
        detections=[edge_det],
    )
    assert captured is not None
    assert captured.trigger_type == "boundary_artifact"

    sampler_off = _make_sampler(tmp_path / "off", al_capture_boundary_artifacts=False)
    skipped = sampler_off.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T03:00:00Z",
        detections=[edge_det],
    )
    assert skipped is None


# -----------------------------------------------------------------------------
# 2. Daily Capture Cap & Storage Quota Pruning
# -----------------------------------------------------------------------------
def test_daily_capture_cap_enforced(tmp_path: Path):
    sampler = _make_sampler(tmp_path, al_max_daily_captures=2)
    frame = _make_ir_frame()
    det = _make_detection(confidence=0.40)
    base_day = datetime(2026, 11, 22, 4, 0, tzinfo=timezone.utc)

    s1 = sampler.maybe_capture(
        frame_bgr=frame, frame_timestamp="2026-11-22T04:00:00Z", detections=[det], now_utc=base_day
    )
    s2 = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T04:10:00Z",
        detections=[det],
        now_utc=base_day + timedelta(minutes=10),
    )
    s3 = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T04:20:00Z",
        detections=[det],
        now_utc=base_day + timedelta(minutes=20),
    )

    assert s1 is not None
    assert s2 is not None
    assert s3 is None  # Blocked by AL_MAX_DAILY_CAPTURES=2
    assert sampler.get_stats(now_utc=base_day).captures_today == 2

    # Next UTC day should be allowed again
    next_day = base_day + timedelta(days=1)
    s4 = sampler.maybe_capture(
        frame_bgr=frame, frame_timestamp="2026-11-23T04:00:00Z", detections=[det], now_utc=next_day
    )
    assert s4 is not None


def test_storage_quota_prunes_rejected_then_pending_protects_approved(tmp_path: Path):
    # Each 640x640 noisy PNG is ~350-450 KB; set quota to 0.75 MB so at most ~1-2 samples fit
    sampler = _make_sampler(tmp_path, al_max_storage_mb=2048.0, al_max_daily_captures=20)
    det = _make_detection(confidence=0.41)
    t0 = datetime(2026, 11, 22, 5, 0, tzinfo=timezone.utc)

    # Create 4 distinct noisy frames (~400KB each)
    samples = []
    for i in range(4):
        rng = np.random.default_rng(100 + i)
        noisy = rng.integers(0, 255, size=(640, 640, 3), dtype=np.uint8)
        s = sampler.maybe_capture(
            frame_bgr=noisy,
            frame_timestamp=f"2026-11-22T05:{i * 10:02d}:00Z",
            detections=[det],
            now_utc=t0 + timedelta(minutes=i * 10),
        )
        assert s is not None
        samples.append(s)

    # Mark sample[0] approved, sample[1] rejected, sample[2] & sample[3] pending
    sampler.db.update_al_decision(
        sample_id=samples[0].id,
        status="approved",
        reviewed_at=(t0 + timedelta(minutes=1)).isoformat(),
    )
    sampler.db.update_al_decision(
        sample_id=samples[1].id,
        status="rejected",
        reviewed_at=(t0 + timedelta(minutes=2)).isoformat(),
    )

    # Tighten quota so exactly ~2 samples fit (should prune rejected sample[1] first, then oldest pending sample[2])
    two_sample_bytes = samples[0].size_bytes + samples[3].size_bytes + 4096
    sampler.settings.al_max_storage_mb = two_sample_bytes / (1024.0 * 1024.0)

    pruned_ids = sampler.enforce_storage_quota()
    assert len(pruned_ids) >= 2
    assert pruned_ids[0] == samples[1].id  # Oldest rejected pruned FIRST
    assert pruned_ids[1] == samples[2].id  # Oldest pending pruned SECOND

    # Approved sample[0] MUST be preserved on disk and in DB
    assert sampler.db.get_al_sample(samples[0].id) is not None
    assert (sampler.pending_dir / samples[0].id / "frame.png").exists()
    # Pruned directories MUST be deleted from disk
    assert not (sampler.pending_dir / samples[1].id).exists()
    assert not (sampler.pending_dir / samples[2].id).exists()


# -----------------------------------------------------------------------------
# 3. YOLOv8-seg Formatting & Review/Export API (.zip structure)
# -----------------------------------------------------------------------------
def test_yolo_seg_formatting_and_validation():
    det_deep = _make_detection(
        label="deep_convective",
        px_poly=[[64.0, 64.0], [192.0, 64.0], [192.0, 192.0], [64.0, 192.0]],
    )
    det_band = _make_detection(
        cell_id="cell_2",
        label="cyclonic_band",
        px_poly=[[320.0, 320.0], [480.0, 320.0], [400.0, 480.0]],
    )
    txt = detections_to_yolo_seg([det_deep, det_band], width=640, height=640)
    ok, err = validate_yolo_seg(txt)
    assert ok is True, f"Validation unexpectedly failed: {err}"

    lines = [l for l in txt.strip().splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("1 ")

    parsed = parse_yolo_seg(txt)
    assert len(parsed) == 2
    assert parsed[0].class_id == 0
    assert parsed[0].label == "deep_convective"
    assert len(parsed[0].points) == 4
    for x, y in parsed[0].points:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0

    # Invalid coordinates (> 1.0) or odd coordinates must fail validation
    bad_ok, _ = validate_yolo_seg("0 0.1 0.2 1.5 0.3 0.2 0.4\n")
    assert bad_ok is False
    odd_ok, _ = validate_yolo_seg("0 0.1 0.2 0.3 0.4 0.5\n")
    assert odd_ok is False


def test_api_review_workflow_and_zip_export(tmp_path: Path):
    sampler = _make_sampler(tmp_path)
    app = FastAPI()
    app.include_router(create_active_learning_router(sampler))
    client = TestClient(app)

    frame = _make_ir_frame(with_bright_storm=True)
    s1 = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T06:00:00Z",
        detections=[_make_detection(confidence=0.36)],
        backend_name="hailo-8",
    )
    s2 = sampler.maybe_capture(
        frame_bgr=frame,
        frame_timestamp="2026-11-22T06:10:00Z",
        detections=[],
        backend_name="hailo-8",
        basin_risks=[_make_kalu_basin_risk(deep_convective_pct=0.0)],
        basin_rainfall_1h_mm={"kalu": 25.0},
    )
    assert s1 is not None and s2 is not None

    # Stats check
    stats_resp = client.get("/api/active-learning/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["pending"] == 2
    assert stats["approved"] == 0
    assert stats["captures_today"] == 2

    # Queue check
    q_resp = client.get("/api/active-learning/queue?status=pending&limit=20")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert q_data["count"] == 2
    assert q_data["samples"][0]["frame_url"].endswith("/frame")
    assert len(q_data["samples"][0]["polygons"]) >= 1

    # Raw frame serving check
    img_resp = client.get(f"/api/active-learning/{s1.id}/frame")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"
    arr = cv2.imdecode(np.frombuffer(img_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None and arr.shape == (640, 640, 3)

    # Export with 0 approved -> HTTP 400
    empty_export = client.get("/api/active-learning/export")
    assert empty_export.status_code == 400

    # Approve s1 with updated YOLOv8-seg label_txt
    corrected_label = "0 0.400000 0.500000 0.550000 0.500000 0.550000 0.650000 0.400000 0.650000\n"
    dec1 = client.post(
        f"/api/active-learning/{s1.id}/decision",
        json={"status": "approved", "updated_label_txt": corrected_label},
    )
    assert dec1.status_code == 200
    assert dec1.json()["sample"]["status"] == "approved"
    assert dec1.json()["sample"]["label_txt"] == corrected_label
    assert dec1.json()["stats"]["approved"] == 1
    assert dec1.json()["stats"]["pending"] == 1

    # Reject s2
    dec2 = client.post(
        f"/api/active-learning/{s2.id}/decision",
        json={"status": "rejected"},
    )
    assert dec2.status_code == 200
    assert dec2.json()["stats"]["rejected"] == 1
    assert dec2.json()["stats"]["pending"] == 0

    # Export approved ZIP and verify exact structure for ml_pipeline/3_prepare_dataset.py
    export_resp = client.get("/api/active-learning/export")
    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"] == "application/zip"
    assert "active_learning_batch_" in export_resp.headers["content-disposition"]

    zf = zipfile.ZipFile(io.BytesIO(export_resp.content))
    names = set(zf.namelist())
    assert f"images/{s1.id}.png" in names
    assert f"labels/{s1.id}.txt" in names
    assert "manifest.json" in names
    # Rejected s2 must NOT be in the export bundle
    assert f"images/{s2.id}.png" not in names

    assert zf.read(f"labels/{s1.id}.txt").decode("utf-8") == corrected_label
    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    assert manifest["sample_count"] == 1
    assert manifest["format"] == "yolov8-seg"
    assert manifest["ml_pipeline_target"] == "3_prepare_dataset.py"
    assert manifest["retained_frames"][0]["filename"] == f"{s1.id}.png"
    assert manifest["retained_frames"][0]["stem"] == s1.id
    assert manifest["retained_frames"][0]["class_counts"]["0"] == 1
