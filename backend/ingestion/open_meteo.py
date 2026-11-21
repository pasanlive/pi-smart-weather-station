from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from config import Settings
from models import SurfaceStationTelemetry

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

MONITORING_STATIONS: list[dict[str, Any]] = [
    {"name": "Colombo", "lat": 6.9271, "lon": 79.8612},
    {"name": "Kandy", "lat": 7.2906, "lon": 80.6337},
    {"name": "Galle", "lat": 6.0535, "lon": 80.2210},
    {"name": "Ratnapura", "lat": 6.6828, "lon": 80.3992},
    {"name": "Trincomalee", "lat": 8.5874, "lon": 81.2152},
    {"name": "Jaffna", "lat": 9.6615, "lon": 80.0255},
    {"name": "Batticaloa", "lat": 7.7310, "lon": 81.6747},
]


class OpenMeteoPoller:
    """
    Polls Open-Meteo every 30 minutes for:
      1. 7 key Sri Lankan surface weather monitoring points (precipitation, wind gusts, 3h pressure drop)
      2. Rolling 24-hour rainfall accumulation (mm) at each wet-zone river basin's upper-catchment centroid.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stations_cache: list[SurfaceStationTelemetry] = []
        self.basin_rainfall_24h_mm: dict[str, float] = {}
        self.basin_rainfall_1h_mm: dict[str, float] = {}
        self.last_polled_at: str | None = None
        self._basin_centroids = self._load_basin_centroids(settings.basins_geojson_path)

    @staticmethod
    def _load_basin_centroids(geojson_path: Path) -> list[dict[str, Any]]:
        if not geojson_path.exists():
            return []
        data = json.loads(geojson_path.read_text(encoding="utf-8"))
        out: list[dict[str, Any]] = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            uc = props.get("upper_catchment", {})
            if "lat" in uc and "lon" in uc and "basin_id" in props:
                out.append(
                    {
                        "basin_id": props["basin_id"],
                        "name": props.get("name", props["basin_id"]),
                        "lat": float(uc["lat"]),
                        "lon": float(uc["lon"]),
                    }
                )
        return out

    @staticmethod
    def _extract_last_24h_rain(hourly: dict[str, Any], now_utc: datetime) -> float:
        times = hourly.get("time") or []
        precip = hourly.get("precipitation") or []
        if not times or not precip:
            return 0.0
        now_key = now_utc.strftime("%Y-%m-%dT%H:00")
        idx = len(times) - 1
        for i, t in enumerate(times):
            if t <= now_key:
                idx = i
        start = max(0, idx - 23)
        total = sum(float(v or 0.0) for v in precip[start : idx + 1])
        return round(total, 1)

    @staticmethod
    def _extract_3h_pressure_drop(hourly: dict[str, Any], current_pressure: float, now_utc: datetime) -> float:
        times = hourly.get("time") or []
        pressures = hourly.get("surface_pressure") or []
        if not times or not pressures:
            return 0.0
        now_key = now_utc.strftime("%Y-%m-%dT%H:00")
        idx = len(times) - 1
        for i, t in enumerate(times):
            if t <= now_key:
                idx = i
        idx_3h_ago = max(0, idx - 3)
        past_pressure = pressures[idx_3h_ago]
        if past_pressure is None:
            return 0.0
        # Positive value means pressure dropped over the last 3 hours
        return round(float(past_pressure) - float(current_pressure), 2)

    async def poll(self) -> tuple[list[SurfaceStationTelemetry], dict[str, float]]:
        now_utc = datetime.now(timezone.utc)
        all_points = [
            {"kind": "station", "key": s["name"], "lat": s["lat"], "lon": s["lon"]}
            for s in MONITORING_STATIONS
        ] + [
            {"kind": "basin", "key": b["basin_id"], "lat": b["lat"], "lon": b["lon"]}
            for b in self._basin_centroids
        ]

        lats = ",".join(str(p["lat"]) for p in all_points)
        lons = ",".join(str(p["lon"]) for p in all_points)
        params = {
            "latitude": lats,
            "longitude": lons,
            "current": "precipitation,wind_gusts_10m,surface_pressure",
            "hourly": "precipitation,surface_pressure",
            "past_days": 1,
            "forecast_days": 1,
            "timezone": "UTC",
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(OPEN_METEO_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()

            items = payload if isinstance(payload, list) else [payload]
            stations: list[SurfaceStationTelemetry] = []
            basin_rain: dict[str, float] = {}
            basin_rain_1h: dict[str, float] = {}

            for point, data in zip(all_points, items):
                current = data.get("current", {}) or {}
                hourly = data.get("hourly", {}) or {}
                if point["kind"] == "station":
                    pressure = float(current.get("surface_pressure") or 1010.0)
                    drop_3h = self._extract_3h_pressure_drop(hourly, pressure, now_utc)
                    stations.append(
                        SurfaceStationTelemetry(
                            name=point["key"],
                            lat=point["lat"],
                            lon=point["lon"],
                            precipitation_mm=round(float(current.get("precipitation") or 0.0), 2),
                            wind_gusts_kmh=round(float(current.get("wind_gusts_10m") or 0.0), 1),
                            surface_pressure_hpa=round(pressure, 1),
                            pressure_drop_3h_hpa=drop_3h,
                            observed_at=str(current.get("time") or now_utc.isoformat()),
                        )
                    )
                else:
                    basin_rain[point["key"]] = self._extract_last_24h_rain(hourly, now_utc)
                    current_precip = current.get("precipitation")
                    if current_precip is None:
                        times = hourly.get("time") or []
                        precip_series = hourly.get("precipitation") or []
                        now_key = now_utc.strftime("%Y-%m-%dT%H:00")
                        latest_val = 0.0
                        for t, v in zip(times, precip_series):
                            if t <= now_key and v is not None:
                                latest_val = float(v)
                        current_precip = latest_val
                    basin_rain_1h[point["key"]] = round(float(current_precip or 0.0), 2)

            self.stations_cache = stations
            self.basin_rainfall_24h_mm = basin_rain
            self.basin_rainfall_1h_mm = basin_rain_1h
            self.last_polled_at = now_utc.isoformat()
            logger.info(
                "Open-Meteo polled: %d surface stations, %d basin catchments (24h rain: %s)",
                len(stations),
                len(basin_rain),
                basin_rain,
            )
            return stations, basin_rain
        except Exception as exc:
            logger.warning("Open-Meteo poll failed: %s", exc)
            return self.stations_cache, self.basin_rainfall_24h_mm
