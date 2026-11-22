from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pytest

from alerts.hourly_digest import (
    HourlyDigestDispatcher,
    filter_qualifying_zones,
    format_hourly_digest_messages,
)
from alerts.telegram_notifier import TELEGRAM_CAPTION_LIMIT, TelegramAlertDispatcher
from config import Settings
from models import DistrictThreat, HazardProbabilities, SeaZoneCoverage, SeaZoneHazard
from nowcast.hazard_probabilities import (
    HazardProbabilityEngine,
    compute_high_wind_prob,
    compute_zone_hazard_probabilities,
)
from storage.db import Database

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DISTRICTS_PATH = DATA_DIR / "sri_lanka_districts.geojson"
SEA_ZONES_PATH = DATA_DIR / "sri_lanka_sea_zones.geojson"
BASINS_GEOJSON = json.loads((DATA_DIR / "sri_lanka_flood_basins.geojson").read_text(encoding="utf-8"))


def _make_district(did: str, name: str, rain: int, ts: int, wind: int, gusts: float = 35.0) -> DistrictThreat:
    max_p = max(rain, ts, wind)
    return DistrictThreat(
        district_id=did,
        name=name,
        threat_level="HIGH" if max_p > 60 else "LOW",
        hazards=HazardProbabilities(
            rainfall_prob=rain,
            thunderstorm_prob=ts,
            high_wind_prob=wind,
            max_hazard_prob=max_p,
            dominant_hazard="thunderstorm" if ts == max_p else ("high_wind" if wind == max_p else "rainfall"),
            wind_gusts_kmh=gusts,
            exceeds_alert_threshold=max_p > 50,
        ),
    )


def _make_sea_zone(zid: str, name: str, rain: int, ts: int, wind: int, gusts: float = 40.0, wave: float = 2.1) -> SeaZoneHazard:
    max_p = max(rain, ts, wind)
    return SeaZoneHazard(
        zone_id=zid,
        name=name,
        centroid=[79.5, 6.5],
        hazards=HazardProbabilities(
            rainfall_prob=rain,
            thunderstorm_prob=ts,
            high_wind_prob=wind,
            max_hazard_prob=max_p,
            dominant_hazard="thunderstorm" if ts == max_p else ("high_wind" if wind == max_p else "rainfall"),
            wind_gusts_kmh=gusts,
            wave_height_m=wave,
            exceeds_alert_threshold=max_p > 50,
        ),
        geometry={"type": "Polygon", "coordinates": [[[79.1, 6.1], [79.8, 6.1], [79.8, 6.8], [79.1, 6.1]]]},
    )


@pytest.mark.asyncio
async def test_batched_open_meteo_33_centroids_and_clamping():
    engine = HazardProbabilityEngine(DISTRICTS_PATH, SEA_ZONES_PATH, alert_threshold_pct=50)
    assert len(engine.all_zone_points) == 33  # 25 districts + 8 sea zones

    now_hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
    request_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_urls.append(str(request.url))
        if "marine-api.open-meteo.com" in str(request.url):
            assert len(request.url.params["latitude"].split(",")) == 8
            return httpx.Response(
                200,
                json=[{"hourly": {"time": [now_hour], "wave_height": [1.85 + i * 0.1]}} for i in range(8)],
            )
        # Single batched Forecast call for all 33 centroids
        assert len(request.url.params["latitude"].split(",")) == 33
        items = []
        for i in range(33):
            items.append(
                {
                    "hourly": {
                        "time": [now_hour],
                        "precipitation_probability": [120 if i == 0 else 40],  # test clamping >100
                        "cape": [3500.0 if i == 0 else 400.0],
                        "weather_code": [95 if i == 1 else 3],  # test WMO 95 >= 80% thunderstorm floor
                        "wind_speed_10m": [25.0],
                        "wind_gusts_10m": [85.0 if i == 0 else 28.0],
                    }
                }
            )
        return httpx.Response(200, json=items)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        metrics = await engine.poll(client=client)

    assert len(request_urls) == 2  # 1 Forecast batch + 1 Marine batch
    assert len(metrics) == 33

    # Check probability clamping & WMO thunderstorm override
    clamped = compute_zone_hazard_probabilities(
        om_precip_prob=120.0,
        max_storm_coverage_pct=110.0,
        deep_convective_pct=100.0,
        cape_jkg=4000.0,
        weather_code=3,
        wind_speed_kmh=40.0,
        wind_gusts_kmh=95.0,
        is_sea_zone=False,
    )
    assert clamped.rainfall_prob == 100
    assert clamped.thunderstorm_prob == 100
    assert clamped.high_wind_prob == 100

    wmo_ts = compute_zone_hazard_probabilities(
        om_precip_prob=20.0,
        max_storm_coverage_pct=0.0,
        deep_convective_pct=0.0,
        cape_jkg=100.0,
        weather_code=95,
        wind_speed_kmh=10.0,
        wind_gusts_kmh=15.0,
        is_sea_zone=False,
    )
    assert wmo_ts.thunderstorm_prob >= 80
    assert wmo_ts.exceeds_alert_threshold is True

    # Verify sea_zones wave_height populated via compute_hazard_probabilities
    sea_cov = {
        "colombo_galle": SeaZoneCoverage(
            zone_id="colombo_galle",
            name="Colombo to Galle",
            storm_coverage_pct=50.0,
            projected_60m_pct=60.0,
            deep_convective_pct=30.0,
        )
    }
    _, sea_zones, sea_fc = engine.compute_hazard_probabilities([], sea_cov)
    assert len(sea_zones) == 8
    assert len(sea_fc["features"]) == 8
    assert all(sz.hazards.wave_height_m is not None for sz in sea_zones)


def test_stricter_sea_zone_wind_scaling_vs_land():
    # Below floor -> 0%
    assert compute_high_wind_prob(30.0, is_sea_zone=False) == 0
    assert compute_high_wind_prob(25.0, is_sea_zone=True) == 0
    # At / above ceiling -> 100%
    assert compute_high_wind_prob(65.0, is_sea_zone=False) == 100
    assert compute_high_wind_prob(55.0, is_sea_zone=True) == 100
    # Mid-range 40 km/h: land = (40-30)/35 = 29%, sea = (40-25)/30 = 50%
    land_40 = compute_high_wind_prob(40.0, is_sea_zone=False)
    sea_40 = compute_high_wind_prob(40.0, is_sea_zone=True)
    assert land_40 == 29
    assert sea_40 == 50
    assert sea_40 > land_40


@pytest.mark.asyncio
async def test_hourly_digest_silent_when_calm_and_grouped_when_over_50(tmp_path: Path):
    db = Database(tmp_path / "digest.db")
    settings = Settings(
        data_dir=str(tmp_path),
        telegram_enabled=True,
        telegram_bot_token="123456:TEST_TOKEN",
        telegram_chat_id="-1003972685096",
        hourly_digest_enabled=True,
        hourly_alert_prob_threshold=50,
    )
    tg = TelegramAlertDispatcher(settings, db)
    digest = HourlyDigestDispatcher(settings, db, tg)

    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 500 + len(calls)}})

    frame = np.full((640, 640, 3), 35, dtype=np.uint8)

    # 1. All zones <= 50% -> skipped silently, 0 Telegram HTTP calls
    calm_districts = [_make_district("colombo", "Colombo", 50, 45, 30)]
    calm_sea = [_make_sea_zone("colombo_galle", "Colombo to Galle", 48, 20, 50)]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res_calm = await digest.dispatch(
            frame_bgr=frame,
            frame_timestamp="2026-11-22T04:00:00Z",
            backend_name="hailo-8",
            detections=[],
            storm_cells=[],
            districts=calm_districts,
            sea_zones=calm_sea,
            basins_geojson=BASINS_GEOJSON,
            client=client,
            base_backoff_sec=0.01,
        )
    assert res_calm["status"] == "skipped_calm"
    assert len(calls) == 0

    # 2. Qualifying >50% zones -> grouped DISTRICTS + SEA AREAS, sorted descending
    hot_districts = [
        _make_district("galle", "Galle", 62, 40, 20, gusts=38.0),
        _make_district("colombo", "Colombo", 70, 86, 55, gusts=52.0),
        _make_district("jaffna", "Jaffna", 20, 10, 15, gusts=22.0),
    ]
    hot_sea = [
        _make_sea_zone("colombo_galle", "Colombo to Galle", 65, 82, 74, gusts=48.0, wave=2.6),
        _make_sea_zone("gulf_of_mannar", "Gulf of Mannar", 30, 20, 45, gusts=38.0, wave=1.4),
    ]
    q_d, q_s = filter_qualifying_zones(hot_districts, hot_sea, threshold_pct=50)
    assert [d.name for d in q_d] == ["Colombo", "Galle"]
    assert [s.name for s in q_s] == ["Colombo to Galle"]

    caption, overflow = format_hourly_digest_messages(
        qualifying_districts=q_d,
        qualifying_sea_zones=q_s,
        threshold_pct=50,
        timestamp_str="2026-11-22T04:00:00Z",
    )
    assert "🏝️ <b>DISTRICTS</b>" in caption
    assert "🌊 <b>SEA AREAS</b>" in caption
    assert len(caption) <= TELEGRAM_CAPTION_LIMIT
    assert overflow is None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res_hot = await digest.dispatch(
            frame_bgr=frame,
            frame_timestamp="2026-11-22T04:00:00Z",
            backend_name="hailo-8",
            detections=[],
            storm_cells=[],
            districts=hot_districts,
            sea_zones=hot_sea,
            basins_geojson=BASINS_GEOJSON,
            client=client,
            base_backoff_sec=0.01,
        )
    assert res_hot["ok"] is True
    assert res_hot["status"] == "sent_telegram_200"
    assert len(calls) == 1
    assert calls[0][1].endswith("/sendPhoto")

    # 3. Overflow split when 20+ zones qualify simultaneously -> sendPhoto + sendMessage reply
    many_districts = [_make_district(f"d_{i}", f"District {i} & Sector", 70 + (i % 20), 65, 60) for i in range(22)]
    many_sea = [_make_sea_zone(f"s_{i}", f"Sea Zone {i}", 68, 72, 80, wave=2.8) for i in range(8)]
    calls.clear()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res_many = await digest.dispatch(
            frame_bgr=frame,
            frame_timestamp="2026-11-22T05:00:00Z",
            backend_name="cpu-fallback",
            detections=[],
            storm_cells=[],
            districts=many_districts,
            sea_zones=many_sea,
            basins_geojson=BASINS_GEOJSON,
            client=client,
            base_backoff_sec=0.01,
        )
    assert res_many["ok"] is True
    assert len(calls) == 2
    assert calls[0][1].endswith("/sendPhoto")
    assert calls[1][1].endswith("/sendMessage")

    hist = db.list_alert_history(limit=5)
    assert len(hist) == 2
    assert hist[0]["entity_key"] == "telegram:digest:hourly"
