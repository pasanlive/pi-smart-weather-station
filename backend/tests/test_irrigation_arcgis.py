from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from config import Settings
from ingestion.irrigation_arcgis import IrrigationArcGISPoller
from storage.db import Database


def _mock_arcgis_gauges_payload(ts_ms: int, kitulgala_level: float, hanwella_level: float) -> dict:
    return {
        "features": [
            {
                "attributes": {
                    "OBJECTID": 1,
                    "station_name": "Kitulgala",
                    "river_name": "Kelani Ganga",
                    "water_level_ft": kitulgala_level,
                    "alert_level_ft": 9.8,
                    "minor_flood_ft": 13.1,
                    "major_flood_ft": 16.4,
                    "reading_timestamp": ts_ms,
                },
                "geometry": {"x": 80.4178, "y": 6.9896},
            },
            {
                "attributes": {
                    "OBJECTID": 2,
                    "station_name": "Hanwella",
                    "river_name": "Kelani Ganga",
                    "water_level_ft": hanwella_level,
                    "alert_level_ft": 23.0,
                    "minor_flood_ft": 26.2,
                    "major_flood_ft": 32.8,
                    "reading_timestamp": ts_ms,
                },
                "geometry": {"x": 80.0816, "y": 6.9097},
            },
        ]
    }


def _mock_arcgis_rainfall_payload() -> dict:
    return {
        "features": [
            {
                "attributes": {"station_name": "Kitulgala", "rainfall_24hr": 84.5},
                "geometry": {"x": 80.4178, "y": 6.9896},
            },
            {
                "attributes": {"station_name": "Hanwella", "rainfall_24hr": 42.0},
                "geometry": {"x": 80.0816, "y": 6.9097},
            },
        ]
    }


@pytest.mark.asyncio
async def test_arcgis_parsing_and_sqlite_rate_of_rise(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    settings = Settings(
        data_dir=str(tmp_path),
        irrigation_arcgis_gauges_url="https://mock.arcgis.local/gauges/0/query",
        irrigation_arcgis_rainfall_url="https://mock.arcgis.local/rain/0/query",
        irrigation_fallback_enabled=False,
    )
    poller = IrrigationArcGISPoller(settings, db)

    t1 = datetime.now(timezone.utc) - timedelta(hours=2)
    t2 = datetime.now(timezone.utc)

    # Seed t1 reading (2 hours ago)
    seed_rows = IrrigationArcGISPoller.parse_arcgis_gauges(
        _mock_arcgis_gauges_payload(int(t1.timestamp() * 1000), kitulgala_level=8.5, hanwella_level=16.0),
        fetched_at=t1,
    )
    db.upsert_gauge_readings(seed_rows)

    # Live poll at t2 via MockTransport
    def handler(request: httpx.Request) -> httpx.Response:
        if "rain" in str(request.url):
            return httpx.Response(200, json=_mock_arcgis_rainfall_payload())
        return httpx.Response(
            200,
            json=_mock_arcgis_gauges_payload(int(t2.timestamp() * 1000), kitulgala_level=10.5, hanwella_level=16.0),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await poller.poll(client=client)

    assert resp.source_status == "arcgis_live"
    assert len(resp.stations) == 2

    kitulgala = next(s for s in resp.stations if s.station_name == "Kitulgala")
    hanwella = next(s for s in resp.stations if s.station_name == "Hanwella")

    assert kitulgala.basin_id == "kelani"
    assert kitulgala.role == "upstream"
    assert kitulgala.status == "ALERT"  # 10.5 >= 9.8
    assert kitulgala.trend == "RISING"
    assert 0.9 <= kitulgala.rate_of_rise_ft_hr <= 1.1  # +2.0 ft over 2.0 hrs = +1.0 ft/hr
    assert kitulgala.rainfall_24h_mm == 84.5
    assert len(kitulgala.history_24h) == 2

    assert hanwella.role == "downstream"
    assert hanwella.status == "NORMAL"
    assert hanwella.trend == "STEADY"
    assert hanwella.rainfall_24h_mm == 42.0


@pytest.mark.asyncio
async def test_arcgis_http_500_degrades_to_unavailable_and_stale(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    settings = Settings(
        data_dir=str(tmp_path),
        irrigation_arcgis_gauges_url="https://mock.arcgis.local/gauges/0/query",
        irrigation_fallback_enabled=False,
    )
    poller = IrrigationArcGISPoller(settings, db)

    # 1. HTTP 500 with empty DB -> must report 'unavailable' (never 'normal')
    def fail_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail_500)) as client:
        resp = await poller.poll(client=client)
    assert resp.source_status == "unavailable"
    assert resp.stations == []
    assert resp.error_detail is not None

    # 2. Seed one reading, then simulate timeout -> must report 'stale'
    now = datetime.now(timezone.utc)
    rows = IrrigationArcGISPoller.parse_arcgis_gauges(
        _mock_arcgis_gauges_payload(int(now.timestamp() * 1000), 6.0, 14.0),
        fetched_at=now,
    )
    db.upsert_gauge_readings(rows)

    def fail_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("ArcGIS timed out")

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail_timeout)) as client:
        resp_stale = await poller.poll(client=client)
    assert resp_stale.source_status == "stale"
    assert len(resp_stale.stations) == 2


@pytest.mark.asyncio
async def test_arcgis_falls_back_to_waterlevels_lk_when_enabled(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    settings = Settings(
        data_dir=str(tmp_path),
        irrigation_arcgis_gauges_url="https://mock.arcgis.local/gauges/0/query",
        irrigation_fallback_enabled=True,
        irrigation_fallback_url="https://waterlevels.lk/api/levels",
    )
    poller = IrrigationArcGISPoller(settings, db)

    def handler(request: httpx.Request) -> httpx.Response:
        if "arcgis" in str(request.url):
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(
            200,
            json=[
                {
                    "station_name": "Ratnapura",
                    "river_name": "Kalu Ganga",
                    "lat": 6.6828,
                    "lon": 80.3992,
                    "water_level_ft": 18.2,
                    "alert_level_ft": 17.0,
                    "minor_flood_ft": 24.6,
                    "major_flood_ft": 31.2,
                }
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await poller.poll(client=client)

    assert resp.source_status == "fallback"
    assert len(resp.stations) == 1
    assert resp.stations[0].basin_id == "kalu"
    assert resp.stations[0].status == "ALERT"
