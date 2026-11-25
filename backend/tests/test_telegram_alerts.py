from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from alerts.snapshot import render_alert_snapshot
from alerts.telegram_notifier import (
    TELEGRAM_CAPTION_LIMIT,
    TelegramAlertDispatcher,
    build_hourly_summary_caption,
    build_safe_html_caption,
)
from config import Settings
from models import (
    DistrictThreat,
    ProjectedStormCell,
    RiverBasinRisk,
    RiverGaugeReading,
    StormDetection,
    UpstreamSurgeWarning,
)
from storage.db import Database

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASINS_GEOJSON = json.loads((DATA_DIR / "sri_lanka_flood_basins.geojson").read_text(encoding="utf-8"))


def _sample_frame() -> np.ndarray:
    img = np.full((640, 640, 3), 42, dtype=np.uint8)
    cv2.circle(img, (320, 340), 45, (235, 235, 235), -1)
    cv2.ellipse(img, (260, 290), (50, 25), 20, 0, 360, (175, 175, 175), -1)
    return img


def _sample_detections_and_motions() -> tuple[list[StormDetection], list[ProjectedStormCell]]:
    det = StormDetection(
        cell_id="cell_1",
        label="deep_convective",
        confidence=0.91,
        pixel_polygon=[[290.0, 310.0], [350.0, 310.0], [350.0, 370.0], [290.0, 370.0]],
        normalized_polygon=[[0.45, 0.48], [0.55, 0.48], [0.55, 0.58], [0.45, 0.58]],
        geo_polygon=[[80.0, 7.6], [80.8, 7.6], [80.8, 6.9], [80.0, 6.9], [80.0, 7.6]],
        centroid_px=[320.0, 340.0],
        centroid_geo=[80.4, 7.25],
        area_px=3600.0,
        peak_intensity=240.0,
    )
    motion = ProjectedStormCell(
        cell_id="cell_1",
        label="deep_convective",
        confidence=0.91,
        centroid_lon=80.4,
        centroid_lat=7.25,
        dx_kmh=24.0,
        dy_kmh=-10.0,
        speed_kmh=26.0,
        bearing_deg=112.6,
        compass_dir="ESE",
        geometry_now={"type": "Polygon", "coordinates": [det.geo_polygon]},
        geometry_projected={},
    )
    return [det], [motion]


def _basin(basin_id: str, name: str, risk_level: str) -> RiverBasinRisk:
    return RiverBasinRisk(
        basin_id=basin_id,
        name=name,
        risk_level=risk_level,  # type: ignore[arg-type]
        eta_min=30,
        storm_coverage_pct=38.0,
        projected_60m_pct=46.0,
        rainfall_24h_mm=112.5,
        downstream_stations=["Hanwella", "Nagalagam Street"],
        geometry={"type": "Polygon", "coordinates": [[[79.9, 6.9], [80.4, 6.9], [80.4, 7.1], [79.9, 6.9]]]},
    )


def _gauge() -> RiverGaugeReading:
    now = datetime.now(timezone.utc).isoformat()
    return RiverGaugeReading(
        station_id="kitulgala",
        station_name="Kitulgala <Upper>",
        river_name="Kelani Ganga",
        basin_id="kelani",
        role="upstream",
        lat=6.99,
        lon=80.42,
        water_level_ft=14.2,
        alert_level_ft=9.8,
        minor_flood_ft=13.1,
        major_flood_ft=16.4,
        reading_timestamp=now,
        fetched_at=now,
        rate_of_rise_ft_hr=0.65,
        trend="RISING",
        status="MINOR_FLOOD",
    )


def test_render_alert_snapshot_outputs_valid_jpeg():
    dets, motions = _sample_detections_and_motions()
    jpeg_bytes = render_alert_snapshot(
        frame_bgr=_sample_frame(),
        detections=dets,
        motions=motions,
        basins_geojson=BASINS_GEOJSON,
        timestamp_str="2026-11-22T02:00:00Z",
        alert_title="CRITICAL FLOOD ALERT - KELANI GANGA",
        engine_label="Hailo-8",
    )

    assert isinstance(jpeg_bytes, bytes)
    assert len(jpeg_bytes) > 5_000
    assert len(jpeg_bytes) < 10 * 1024 * 1024  # well under Telegram 10 MB limit
    assert jpeg_bytes[:2] == b"\xff\xd8"  # valid JPEG SOI marker

    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[0] >= 640 and decoded.shape[1] >= 640 and decoded.shape[2] == 3


def test_html_escaping_and_1024_caption_cap():
    long_advisory = ("Heavy rain > 120mm & flash flooding near <script>alert(1)</script> basins. " * 35).strip()
    caption = build_safe_html_caption(
        header_prefix="FLOOD ALERT",
        severity="CRITICAL",
        zone_name="Kelani & Kalu <Wet-Zone>",
        eta_min=30,
        storm_coverage_pct=44.2,
        projected_60m_pct=58.0,
        rainfall_24h_mm=134.8,
        gauge_line="Kitulgala <Upper> > 14.2 ft & RISING",
        downstream_stations=["Hanwella & Glencourse", "Nagalagam <Colombo>"],
        advisory_text=long_advisory,
    )

    assert len(caption) <= TELEGRAM_CAPTION_LIMIT
    assert "<script>" not in caption
    assert "&lt;script&gt;" in caption
    assert "Kelani &amp; Kalu &lt;Wet-Zone&gt;" in caption
    assert "&gt; 14.2 ft &amp; RISING" in caption
    # Check balanced bold tags
    assert caption.count("<b>") == caption.count("</b>")


@pytest.mark.asyncio
async def test_cooldown_suppresses_duplicates_and_escalation_bypasses(tmp_path: Path):
    db = Database(tmp_path / "tg_cooldown.db")
    settings = Settings(
        data_dir=str(tmp_path),
        telegram_enabled=True,
        telegram_bot_token="123456:TEST_TOKEN",
        telegram_chat_id="-100999888777",
        telegram_alert_cooldown_min=30,
    )
    dispatcher = TelegramAlertDispatcher(settings, db)
    dets, motions = _sample_detections_and_motions()
    sent_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(sent_requests)}})

    t0 = datetime(2026, 11, 22, 4, 0, tzinfo=timezone.utc)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # 1. Initial WARNING -> sent
        r1 = await dispatcher.dispatch_nowcast_alerts(
            frame_bgr=_sample_frame(),
            frame_timestamp="2026-11-22T04:00:00Z",
            backend_name="cpu-fallback",
            detections=dets,
            storm_cells=motions,
            districts=[],
            basins=[_basin("kelani", "Kelani Ganga", "WARNING")],
            basins_geojson=BASINS_GEOJSON,
            gauges=[_gauge()],
            advisory_text="Kelani catchment warning.",
            client=client,
            now=t0,
            base_backoff_sec=0.01,
        )
        assert len(r1) == 1 and r1[0]["ok"] is True
        assert len(sent_requests) == 1

        # 2. Duplicate WARNING 10 min later -> suppressed by 30-min cooldown
        r2 = await dispatcher.dispatch_nowcast_alerts(
            frame_bgr=_sample_frame(),
            frame_timestamp="2026-11-22T04:10:00Z",
            backend_name="cpu-fallback",
            detections=dets,
            storm_cells=motions,
            districts=[],
            basins=[_basin("kelani", "Kelani Ganga", "WARNING")],
            basins_geojson=BASINS_GEOJSON,
            gauges=[_gauge()],
            advisory_text="Still warning.",
            client=client,
            now=t0 + timedelta(minutes=10),
            base_backoff_sec=0.01,
        )
        assert r2 == []
        assert len(sent_requests) == 1

        # 3. Escalation WARNING -> CRITICAL 15 min later -> bypasses cooldown immediately!
        r3 = await dispatcher.dispatch_nowcast_alerts(
            frame_bgr=_sample_frame(),
            frame_timestamp="2026-11-22T04:15:00Z",
            backend_name="hailo-8",
            detections=dets,
            storm_cells=motions,
            districts=[],
            basins=[_basin("kelani", "Kelani Ganga", "CRITICAL")],
            basins_geojson=BASINS_GEOJSON,
            gauges=[_gauge()],
            advisory_text="Escalated to critical flood risk.",
            client=client,
            now=t0 + timedelta(minutes=15),
            base_backoff_sec=0.01,
        )
        assert len(r3) == 1 and r3[0]["ok"] is True
        assert "escalation_bypass" in r3[0]["reason"]
        assert len(sent_requests) == 2

        # 4. After 31 min -> cooldown expired, allowed again
        r4 = await dispatcher.dispatch_nowcast_alerts(
            frame_bgr=_sample_frame(),
            frame_timestamp="2026-11-22T04:46:00Z",
            backend_name="hailo-8",
            detections=dets,
            storm_cells=motions,
            districts=[],
            basins=[_basin("kelani", "Kelani Ganga", "CRITICAL")],
            basins_geojson=BASINS_GEOJSON,
            gauges=[_gauge()],
            advisory_text="Continued critical status.",
            client=client,
            now=t0 + timedelta(minutes=46),
            base_backoff_sec=0.01,
        )
        assert len(r4) == 1 and r4[0]["ok"] is True
        assert len(sent_requests) == 3

    history = db.list_alert_history(limit=10)
    assert len(history) == 3
    assert history[0]["level"] == "CRITICAL"


@pytest.mark.asyncio
async def test_missing_credentials_degrades_gracefully(tmp_path: Path):
    db = Database(tmp_path / "tg_unconfigured.db")
    settings = Settings(
        data_dir=str(tmp_path),
        telegram_enabled=True,
        telegram_bot_token="",
        telegram_chat_id="",
    )
    dispatcher = TelegramAlertDispatcher(settings, db)
    assert dispatcher.is_ready is False

    dets, motions = _sample_detections_and_motions()
    dispatched = await dispatcher.dispatch_nowcast_alerts(
        frame_bgr=_sample_frame(),
        frame_timestamp="2026-11-22T04:00:00Z",
        backend_name="cpu-fallback",
        detections=dets,
        storm_cells=motions,
        districts=[DistrictThreat(district_id="colombo", name="Colombo", threat_level="SEVERE", eta_min=0)],
        basins=[_basin("kelani", "Kelani Ganga", "CRITICAL")],
        basins_geojson=BASINS_GEOJSON,
        gauges=[],
        advisory_text="Severe storm.",
    )
    assert dispatched == []

    test_res = await dispatcher.send_test_alert(
        frame_bgr=_sample_frame(),
        frame_timestamp="2026-11-22T04:00:00Z",
        backend_name="cpu-fallback",
        detections=dets,
        storm_cells=motions,
        basins_geojson=BASINS_GEOJSON,
        advisory_text="Test advisory.",
    )
    assert test_res["ok"] is False
    assert test_res["status"] == "unconfigured"
    assert test_res["snapshot_bytes"] > 1000


@pytest.mark.asyncio
async def test_retry_on_http_429_succeeds_and_logs_history(tmp_path: Path):
    db = Database(tmp_path / "tg_retry.db")
    settings = Settings(
        data_dir=str(tmp_path),
        telegram_enabled=True,
        telegram_bot_token="123456:TEST_TOKEN",
        telegram_chat_id="987654321",
    )
    dispatcher = TelegramAlertDispatcher(settings, db)
    dets, motions = _sample_detections_and_motions()
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 0.01}})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await dispatcher.send_test_alert(
            frame_bgr=_sample_frame(),
            frame_timestamp="2026-11-22T04:00:00Z",
            backend_name="hailo-8",
            detections=dets,
            storm_cells=motions,
            basins_geojson=BASINS_GEOJSON,
            advisory_text="Retry test.",
            client=client,
            base_backoff_sec=0.01,
        )

    assert res["ok"] is True
    assert res["status"] == "sent_telegram_200"
    assert attempts["count"] == 2
    hist = db.list_alert_history(limit=5)
    assert len(hist) == 1
    assert hist[0]["entity_key"] == "telegram:test:system"


@pytest.mark.asyncio
async def test_hourly_summary_caption_and_dispatch(tmp_path: Path):
    db = Database(tmp_path / "tg_summary.db")
    settings = Settings(
        data_dir=str(tmp_path),
        telegram_enabled=True,
        telegram_bot_token="123456:TEST_TOKEN",
        telegram_chat_id="-1003972685096",
        telegram_hourly_summary_enabled=True,
    )
    dispatcher = TelegramAlertDispatcher(settings, db)
    dets, motions = _sample_detections_and_motions()
    basins = [
        _basin("kelani", "Kelani Ganga", "WARNING"),
        _basin("kalu", "Kalu Ganga", "NORMAL"),
    ]
    surge = UpstreamSurgeWarning(
        basin_id="kelani",
        basin_name="Kelani Ganga",
        upstream_stations=["Kitulgala"],
        downstream_stations=["Hanwella"],
        max_rate_of_rise_ft_hr=0.55,
        drainage_lag_hours=20,
        drainage_lag_window_hours=[18, 24],
        estimated_crest_window="18–24 hours",
        message="Surge at Kitulgala.",
    )

    caption = build_hourly_summary_caption(
        frame_timestamp="2026-11-22T02:00:00Z",
        backend_name="hailo-8",
        inference_ms=7.2,
        storm_cells=motions,
        districts=[DistrictThreat(district_id="colombo", name="Colombo", threat_level="HIGH", eta_min=30)],
        basins=basins,
        surges=[surge],
        hydrology_source_status="arcgis_live",
        advisory_text="Rain > 50mm & <alert> across western slopes.",
    )
    assert len(caption) <= TELEGRAM_CAPTION_LIMIT
    assert "&lt;alert&gt;" in caption
    assert "SL HOURLY WEATHER" in caption
    assert "Kelani" in caption
    assert caption.count("<b>") == caption.count("</b>")

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 99}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await dispatcher.send_hourly_summary(
            frame_bgr=_sample_frame(),
            frame_timestamp="2026-11-22T02:00:00Z",
            backend_name="hailo-8",
            inference_ms=7.2,
            detections=dets,
            storm_cells=motions,
            districts=[DistrictThreat(district_id="colombo", name="Colombo", threat_level="HIGH", eta_min=30)],
            basins=basins,
            basins_geojson=BASINS_GEOJSON,
            surges=[surge],
            hydrology_source_status="arcgis_live",
            advisory_text="Active rain band moving west.",
            client=client,
            base_backoff_sec=0.01,
        )

    assert res["ok"] is True
    assert res["status"] == "sent_telegram_200"
    assert len(sent) == 1
    hist = db.list_alert_history(limit=5)
    assert hist[0]["entity_key"] == "telegram:summary:hourly"
    assert hist[0]["level"] == "INFO"

