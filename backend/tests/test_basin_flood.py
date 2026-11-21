from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from config import Settings
from models import BasinZoneCoverage, DistrictThreat, RiverGaugeReading
from nowcast.alerting import AlertDispatcher
from nowcast.basin_flood import BasinFloodAnalyzer
from storage.db import Database

BASINS_PATH = Path(__file__).resolve().parent.parent / "data" / "sri_lanka_flood_basins.geojson"


def _cov(bid: str, curr: float, proj60: float, deep: float = 0.0) -> BasinZoneCoverage:
    return BasinZoneCoverage(
        basin_id=bid,
        name=bid.title(),
        eta_min=0 if curr >= 10.0 else (60 if proj60 >= 10.0 else None),
        coverage_by_horizon={"0m": curr, "+30m": curr, "+60m": proj60, "+120m": proj60},
        deep_convective_by_horizon={"0m": deep, "+60m": deep},
        storm_coverage_pct=curr,
        projected_60m_pct=proj60,
        deep_convective_pct=deep,
        deep_convective_60m_pct=deep,
        peak_coverage_pct=max(curr, proj60),
    )


def _gauge(
    station_name: str,
    basin_id: str,
    role: str,
    status: str = "NORMAL",
    trend: str = "STEADY",
    rate: float = 0.0,
) -> RiverGaugeReading:
    now = datetime.now(timezone.utc).isoformat()
    return RiverGaugeReading(
        station_id=station_name.lower().replace(" ", "_"),
        station_name=station_name,
        river_name=f"{basin_id.title()} Ganga",
        basin_id=basin_id,
        role=role,  # type: ignore[arg-type]
        lat=6.9,
        lon=80.2,
        water_level_ft=15.0,
        alert_level_ft=14.0,
        minor_flood_ft=18.0,
        major_flood_ft=24.0,
        reading_timestamp=now,
        fetched_at=now,
        rate_of_rise_ft_hr=rate,
        trend=trend,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
    )


def test_basin_flood_all_four_satellite_rainfall_tiers():
    analyzer = BasinFloodAnalyzer(BASINS_PATH)
    coverages = {
        "kelani": _cov("kelani", curr=40.0, proj60=45.0, deep=30.0),  # CRITICAL with 115mm
        "kalu": _cov("kalu", curr=26.0, proj60=34.0, deep=10.0),      # WARNING with 65mm
        "gin": _cov("gin", curr=18.0, proj60=15.0, deep=0.0),         # WATCH (curr >= 15%)
        "nilwala": _cov("nilwala", curr=5.0, proj60=8.0, deep=0.0),   # NORMAL
    }
    rain = {"kelani": 115.0, "kalu": 65.0, "gin": 20.0, "nilwala": 10.0}

    basins, fc, surges = analyzer.analyze(coverages, rain, gauges=[])
    by_id = {b.basin_id: b for b in basins}

    assert by_id["kelani"].risk_level == "CRITICAL"
    assert by_id["kalu"].risk_level == "WARNING"
    assert by_id["gin"].risk_level == "WATCH"
    assert by_id["nilwala"].risk_level == "NORMAL"
    assert len(fc["features"]) == 4
    assert surges == []


def test_basin_flood_gauge_fusion_and_upstream_surge():
    analyzer = BasinFloodAnalyzer(BASINS_PATH)
    coverages = {
        # Kelani: minor storm overhead + rapidly rising upstream Kitulgala while Hanwella is NORMAL -> CRITICAL + Upstream Surge!
        "kelani": _cov("kelani", curr=28.0, proj60=32.0, deep=22.0),
        # Kalu: gauge at ALERT and RISING -> WARNING
        "kalu": _cov("kalu", curr=5.0, proj60=5.0),
        # Gin: gauge at MAJOR_FLOOD -> CRITICAL even without storm
        "gin": _cov("gin", curr=0.0, proj60=0.0),
        # Nilwala: gauge at ALERT steady -> WATCH
        "nilwala": _cov("nilwala", curr=0.0, proj60=0.0),
    }
    gauges = [
        _gauge("Kitulgala", "kelani", "upstream", status="ALERT", trend="RISING", rate=0.72),
        _gauge("Hanwella", "kelani", "downstream", status="NORMAL", trend="STEADY", rate=0.01),
        _gauge("Ratnapura", "kalu", "upstream", status="ALERT", trend="RISING", rate=0.25),
        _gauge("Putupaula", "kalu", "downstream", status="NORMAL", trend="STEADY", rate=0.0),
        _gauge("Baddegama", "gin", "downstream", status="MAJOR_FLOOD", trend="RISING", rate=0.4),
        _gauge("Thalgahagoda", "nilwala", "downstream", status="ALERT", trend="STEADY", rate=0.0),
    ]

    basins, _, surges = analyzer.analyze(coverages, {"kelani": 30.0, "kalu": 20.0, "gin": 10.0, "nilwala": 10.0}, gauges=gauges)
    by_id = {b.basin_id: b for b in basins}

    assert by_id["kelani"].risk_level == "CRITICAL"
    assert by_id["kelani"].upstream_surge is True
    assert by_id["kalu"].risk_level == "WARNING"
    assert by_id["kalu"].upstream_surge is True
    assert by_id["gin"].risk_level == "CRITICAL"
    assert by_id["nilwala"].risk_level == "WATCH"

    surge_ids = {s.basin_id for s in surges}
    assert {"kelani", "kalu"} == surge_ids
    kelani_surge = next(s for s in surges if s.basin_id == "kelani")
    assert "18–24 hours" in kelani_surge.estimated_crest_window
    assert "Hanwella" in kelani_surge.downstream_stations


@pytest.mark.asyncio
async def test_webhook_alert_cooldown_for_districts_and_basins(tmp_path: Path):
    db = Database(tmp_path / "alerts.db")
    settings = Settings(
        data_dir=str(tmp_path),
        alert_webhook_url="https://mock.webhook.local/notify",
        alert_cooldown_min=30,
    )
    dispatcher = AlertDispatcher(settings, db)
    sent_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_payloads.append({"url": str(request.url)})
        return httpx.Response(200, json={"ok": True})

    analyzer = BasinFloodAnalyzer(BASINS_PATH)
    basins, _, _ = analyzer.analyze({"kelani": _cov("kelani", 40.0, 45.0)}, {"kelani": 120.0}, gauges=[])
    districts = [
        DistrictThreat(district_id="colombo", name="Colombo", threat_level="HIGH", eta_min=30, peak_coverage_pct=42.0)
    ]

    t0 = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await dispatcher.dispatch_if_needed(districts, basins, client=client, now=t0)
        assert len(first) == 2  # 1 district + 1 basin
        assert len(sent_payloads) == 2

        # 10 minutes later -> suppressed by 30-min cooldown
        second = await dispatcher.dispatch_if_needed(districts, basins, client=client, now=t0 + timedelta(minutes=10))
        assert second == []
        assert len(sent_payloads) == 2

        # 31 minutes later -> allowed again
        third = await dispatcher.dispatch_if_needed(districts, basins, client=client, now=t0 + timedelta(minutes=31))
        assert len(third) == 2
        assert len(sent_payloads) == 4
