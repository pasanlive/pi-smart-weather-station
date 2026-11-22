from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from config import Settings
from models import (
    DistrictThreat,
    ProjectedStormCell,
    RiverBasinRisk,
    RiverGaugeReading,
    UpstreamSurgeWarning,
)
from nowcast.advisory_generator import WeatherAdvisoryGenerator


def _storm_cell() -> ProjectedStormCell:
    return ProjectedStormCell(
        cell_id="cell_1",
        label="deep_convective",
        confidence=0.92,
        centroid_lon=80.2,
        centroid_lat=6.9,
        dx_kmh=-28.0,
        dy_kmh=12.0,
        speed_kmh=30.5,
        bearing_deg=293.2,
        compass_dir="WNW",
        geometry_now={"type": "Polygon", "coordinates": [[[80.0, 6.8], [80.3, 6.8], [80.3, 7.0], [80.0, 6.8]]]},
        geometry_projected={},
    )


def _district(name: str, threat: str, eta: int | None = 30) -> DistrictThreat:
    return DistrictThreat(
        district_id=name.lower(),
        name=name,
        threat_level=threat,  # type: ignore[arg-type]
        eta_min=eta,
        affected_area_pct=22.0 if eta == 0 else 0.0,
        peak_coverage_pct=46.0 if threat != "LOW" else 3.0,
        precipitation_mm=14.2 if threat != "LOW" else 0.0,
        wind_gusts_kmh=42.0 if threat != "LOW" else 12.0,
    )


def _basin(basin_id: str, name: str, risk: str, rain_24h: float = 12.0) -> RiverBasinRisk:
    return RiverBasinRisk(
        basin_id=basin_id,
        name=name,
        risk_level=risk,  # type: ignore[arg-type]
        eta_min=60 if risk != "NORMAL" else None,
        storm_coverage_pct=28.0 if risk != "NORMAL" else 0.0,
        projected_60m_pct=42.0 if risk != "NORMAL" else 0.0,
        rainfall_24h_mm=rain_24h,
        downstream_stations=["Hanwella", "Nagalagam Street"],
        geometry={"type": "Polygon", "coordinates": [[[79.9, 6.9], [80.4, 6.9], [80.4, 7.1], [79.9, 6.9]]]},
    )


def _rising_gauge() -> RiverGaugeReading:
    now = datetime.now(timezone.utc).isoformat()
    return RiverGaugeReading(
        station_id="kitulgala",
        station_name="Kitulgala",
        river_name="Kelani Ganga",
        basin_id="kelani",
        role="upstream",
        lat=6.99,
        lon=80.42,
        water_level_ft=11.4,
        alert_level_ft=9.8,
        minor_flood_ft=13.1,
        major_flood_ft=16.4,
        reading_timestamp=now,
        fetched_at=now,
        rate_of_rise_ft_hr=0.45,
        trend="RISING",
        status="ALERT",
    )


@pytest.mark.asyncio
async def test_skips_ollama_and_returns_template_when_normal_and_low():
    settings = Settings(
        ollama_enabled=True,
        ollama_url="http://localhost:11434",
        ollama_model="qwen2.5:3b",
        ollama_min_severity="WATCH",
    )
    gen = WeatherAdvisoryGenerator(settings)
    called = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        called["count"] += 1
        return httpx.Response(200, json={"response": "Should not be called."})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await gen.generate_advisory(
            storm_cells=[],
            districts=[_district("Colombo", "LOW", eta=None), _district("Kandy", "LOW", eta=None)],
            basins=[_basin("kelani", "Kelani Ganga", "NORMAL", rain_24h=8.5)],
            gauges=[],
            surges=[],
            client=client,
        )

    assert called["count"] == 0
    assert res.source == "template"
    assert res.model is None
    assert "LOW" in res.text and "NORMAL" in res.text


@pytest.mark.asyncio
async def test_ollama_success_verifies_cpu_pinned_options():
    settings = Settings(
        ollama_enabled=True,
        ollama_url="http://localhost:11434",
        ollama_model="qwen2.5:3b",
        ollama_timeout_sec=30,
        ollama_num_ctx=2048,
        ollama_num_thread=2,
        ollama_num_predict=120,
        ollama_min_severity="WATCH",
    )
    gen = WeatherAdvisoryGenerator(settings)
    captured_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        captured_body.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "response": (
                    "Deep convective storms moving WNW at 31 km/h will impact Colombo within +30m. "
                    "Kelani Ganga is at WARNING with 88 mm 24h rainfall and rising levels at Kitulgala."
                )
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await gen.generate_advisory(
            storm_cells=[_storm_cell()],
            districts=[_district("Colombo", "HIGH", eta=30)],
            basins=[_basin("kelani", "Kelani Ganga", "WARNING", rain_24h=88.0)],
            gauges=[_rising_gauge()],
            surges=[],
            client=client,
        )

    assert res.source == "ollama"
    assert res.model == "qwen2.5:3b"
    assert res.latency_ms >= 0.0
    assert "Colombo" in res.text
    assert captured_body["model"] == "qwen2.5:3b"
    assert captured_body["stream"] is False
    assert captured_body["keep_alive"] == "15m"
    assert captured_body["options"]["num_ctx"] == 2048
    assert captured_body["options"]["num_thread"] == 2
    assert captured_body["options"]["num_predict"] == 120
    assert captured_body["options"]["temperature"] == 0.2


@pytest.mark.asyncio
async def test_fallback_to_template_on_timeout_and_connect_error():
    settings = Settings(
        ollama_enabled=True,
        ollama_url="http://localhost:11434",
        ollama_model="qwen2.5:3b",
        ollama_timeout_sec=5,
        ollama_min_severity="WATCH",
    )
    gen = WeatherAdvisoryGenerator(settings)

    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Ollama inference timed out")

    def raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(raise_timeout)) as client:
        r_timeout = await gen.generate_advisory(
            storm_cells=[_storm_cell()],
            districts=[_district("Colombo", "HIGH", eta=30)],
            basins=[_basin("kelani", "Kelani Ganga", "WARNING", rain_24h=72.0)],
            gauges=[_rising_gauge()],
            client=client,
        )

    assert r_timeout.source == "template_fallback"
    assert "Colombo" in r_timeout.text
    assert "+30m" in r_timeout.text
    assert "WNW" in r_timeout.text
    assert "Kelani Ganga" in r_timeout.text
    assert "Kitulgala" in r_timeout.text

    async with httpx.AsyncClient(transport=httpx.MockTransport(raise_connect)) as client:
        r_conn = await gen.generate_advisory(
            storm_cells=[_storm_cell()],
            districts=[_district("Ratnapura", "MODERATE", eta=60)],
            basins=[_basin("kalu", "Kalu Ganga", "WATCH", rain_24h=55.0)],
            gauges=[],
            surges=[
                UpstreamSurgeWarning(
                    basin_id="kalu",
                    basin_name="Kalu Ganga",
                    upstream_stations=["Ratnapura"],
                    downstream_stations=["Ellagawa", "Putupaula"],
                    max_rate_of_rise_ft_hr=0.62,
                    drainage_lag_hours=15,
                    drainage_lag_window_hours=[12, 18],
                    estimated_crest_window="12–18 hours",
                    message="Surge at Ratnapura.",
                )
            ],
            client=client,
        )

    assert r_conn.source == "template_fallback"
    assert "Kalu Ganga" in r_conn.text
    assert "12–18 hours" in r_conn.text
