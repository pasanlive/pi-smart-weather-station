from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx

from models import (
    DistrictThreat,
    HazardProbabilities,
    HazardType,
    SeaZoneCoverage,
    SeaZoneHazard,
)

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
THUNDERSTORM_WMO_CODES = {95, 96, 99}


@dataclass
class ZoneForecastMetrics:
    zone_id: str
    zone_type: str  # "district" | "sea_zone"
    om_precip_prob: float = 0.0
    cape_jkg: float = 0.0
    weather_code: int = 0
    wind_speed_kmh: float = 0.0
    wind_gusts_kmh: float = 0.0
    wind_direction_deg: float = 0.0
    wave_height_m: Optional[float] = None


_COMPASS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def degrees_to_compass(deg: float) -> str:
    idx = int(((float(deg) % 360.0) + 22.5) // 45.0) % 8
    return _COMPASS_8[idx]


def _clamp_pct(val: float) -> int:
    return int(max(0, min(100, round(val))))


def compute_high_wind_prob(wind_gusts_kmh: float, is_sea_zone: bool = False) -> int:
    """
    Linear 0-100% high-wind probability scale based on wind_gusts_10m:
      - Land districts: 30 km/h (0%) -> 65 km/h (100%)
      - Offshore sea zones: stricter 25 km/h (0%) -> 55 km/h (100%)
    """
    low, high = (25.0, 55.0) if is_sea_zone else (30.0, 65.0)
    if wind_gusts_kmh <= low:
        return 0
    if wind_gusts_kmh >= high:
        return 100
    return _clamp_pct(((wind_gusts_kmh - low) / (high - low)) * 100.0)


def compute_zone_hazard_probabilities(
    *,
    om_precip_prob: float,
    max_storm_coverage_pct: float,
    deep_convective_pct: float,
    cape_jkg: float,
    weather_code: int,
    wind_speed_kmh: float,
    wind_gusts_kmh: float,
    wind_direction_deg: float = 0.0,
    is_sea_zone: bool,
    wave_height_m: Optional[float] = None,
    alert_threshold_pct: int = 50,
) -> HazardProbabilities:
    """Fuses Open-Meteo hourly forecast variables with Himawari-9 storm detections into 0-100% scores."""
    rainfall_prob = _clamp_pct(0.6 * float(om_precip_prob) + 0.4 * float(max_storm_coverage_pct))

    cape_score = min(100.0, max(0.0, float(cape_jkg) / 20.0))
    ts_base = _clamp_pct(0.55 * float(deep_convective_pct) + 0.45 * cape_score)
    thunderstorm_prob = max(80, ts_base) if int(weather_code) in THUNDERSTORM_WMO_CODES else ts_base

    high_wind_prob = compute_high_wind_prob(float(wind_gusts_kmh), is_sea_zone=is_sea_zone)

    max_prob = max(rainfall_prob, thunderstorm_prob, high_wind_prob)
    # Deterministic tie-break: thunderstorm > high_wind > rainfall
    if thunderstorm_prob == max_prob and thunderstorm_prob > 0:
        dominant: HazardType = "thunderstorm"
    elif high_wind_prob == max_prob and high_wind_prob > 0:
        dominant = "high_wind"
    else:
        dominant = "rainfall"

    return HazardProbabilities(
        rainfall_prob=rainfall_prob,
        thunderstorm_prob=thunderstorm_prob,
        high_wind_prob=high_wind_prob,
        max_hazard_prob=max_prob,
        dominant_hazard=dominant,
        wind_speed_kmh=round(float(wind_speed_kmh), 1),
        wind_gusts_kmh=round(float(wind_gusts_kmh), 1),
        wind_direction_deg=round(float(wind_direction_deg) % 360.0, 1),
        wind_compass_dir=degrees_to_compass(wind_direction_deg),
        cape_jkg=round(float(cape_jkg), 1),
        weather_code=int(weather_code),
        wave_height_m=round(float(wave_height_m), 2) if wave_height_m is not None else None,
        exceeds_alert_threshold=max_prob > int(alert_threshold_pct),
    )


def _select_nowcast_window_index(times: list[str], now_utc: datetime) -> int:
    if not times:
        return 0
    now_key = now_utc.strftime("%Y-%m-%dT%H:00")
    idx = 0
    for i, t in enumerate(times):
        if t <= now_key:
            idx = i
        else:
            break
    return idx


def _window_max(values: Optional[list[Any]], start_idx: int, hours: int = 3) -> float:
    if not values:
        return 0.0
    window = [float(v) for v in values[start_idx : start_idx + hours] if v is not None]
    return max(window) if window else 0.0


def _window_weather_code(values: Optional[list[Any]], start_idx: int, hours: int = 3) -> int:
    if not values:
        return 0
    window = [int(v) for v in values[start_idx : start_idx + hours] if v is not None]
    if not window:
        return 0
    for code in window:
        if code in THUNDERSTORM_WMO_CODES:
            return code
    return max(window)


class HazardProbabilityEngine:
    """
    Polls Open-Meteo Forecast (single batched request for all 33 centroids: 25 districts + 8 sea zones)
    and Open-Meteo Marine (single batched request for 8 sea zones) every 30 minutes, then fuses them
    with Himawari-9 satellite storm coverages from NowcastEngine.
    """

    def __init__(
        self,
        districts_geojson_path: Path | str,
        sea_zones_geojson_path: Path | str,
        alert_threshold_pct: int = 50,
    ):
        self.districts_geojson_path = Path(districts_geojson_path)
        self.sea_zones_geojson_path = Path(sea_zones_geojson_path)
        self.alert_threshold_pct = alert_threshold_pct

        self._district_points: list[dict[str, Any]] = self._load_district_centroids(self.districts_geojson_path)
        self._sea_features: list[dict[str, Any]] = self._load_sea_features(self.sea_zones_geojson_path)
        self.forecast_cache: dict[str, ZoneForecastMetrics] = {}
        self.last_polled_at: Optional[str] = None

    @staticmethod
    def _load_district_centroids(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: list[dict[str, Any]] = []
        for feat in raw.get("features", []):
            props = feat.get("properties") or {}
            did = str(props.get("district_id") or props.get("name", "").lower().replace(" ", "_"))
            out.append(
                {
                    "zone_id": did,
                    "name": str(props.get("name") or did),
                    "zone_type": "district",
                    "lat": float(props.get("centroid_lat", 7.87)),
                    "lon": float(props.get("centroid_lon", 80.77)),
                }
            )
        return out

    @staticmethod
    def _load_sea_features(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        return list(raw.get("features", []))

    @property
    def all_zone_points(self) -> list[dict[str, Any]]:
        sea_pts = []
        for feat in self._sea_features:
            props = feat.get("properties") or {}
            centroid = props.get("centroid") or [80.0, 6.0]
            sea_pts.append(
                {
                    "zone_id": str(props["zone_id"]),
                    "name": str(props.get("name") or props["zone_id"]),
                    "zone_type": "sea_zone",
                    "lon": float(centroid[0]),
                    "lat": float(centroid[1]),
                }
            )
        return self._district_points + sea_pts

    @classmethod
    def parse_batched_open_meteo(
        cls,
        points: Sequence[dict[str, Any]],
        forecast_payload: Any,
        marine_payload: Any = None,
        now_utc: Optional[datetime] = None,
    ) -> dict[str, ZoneForecastMetrics]:
        """Parses Open-Meteo Forecast (33 points) + Marine (8 sea zones) batched responses."""
        ref_now = now_utc or datetime.now(timezone.utc)
        forecast_items = forecast_payload if isinstance(forecast_payload, list) else [forecast_payload]

        metrics: dict[str, ZoneForecastMetrics] = {}
        for pt, data in zip(points, forecast_items):
            hourly = (data or {}).get("hourly") or {}
            idx = _select_nowcast_window_index(hourly.get("time") or [], ref_now)
            metrics[pt["zone_id"]] = ZoneForecastMetrics(
                zone_id=pt["zone_id"],
                zone_type=pt["zone_type"],
                om_precip_prob=_window_max(hourly.get("precipitation_probability"), idx, hours=3),
                cape_jkg=_window_max(hourly.get("cape"), idx, hours=3),
                weather_code=_window_weather_code(hourly.get("weather_code"), idx, hours=3),
                wind_speed_kmh=_window_max(hourly.get("wind_speed_10m"), idx, hours=3),
                wind_gusts_kmh=_window_max(hourly.get("wind_gusts_10m"), idx, hours=3),
                wind_direction_deg=float((hourly.get("wind_direction_10m") or [0.0])[min(idx, max(0, len(hourly.get("wind_direction_10m") or [0.0]) - 1))] or 0.0),
            )

        if marine_payload is not None:
            sea_pts = [p for p in points if p["zone_type"] == "sea_zone"]
            marine_items = marine_payload if isinstance(marine_payload, list) else [marine_payload]
            for pt, mdata in zip(sea_pts, marine_items):
                hourly = (mdata or {}).get("hourly") or {}
                idx = _select_nowcast_window_index(hourly.get("time") or [], ref_now)
                wave = _window_max(hourly.get("wave_height"), idx, hours=3)
                if pt["zone_id"] in metrics:
                    metrics[pt["zone_id"]].wave_height_m = round(wave, 2) if wave > 0 else 0.0

        return metrics

    async def poll(self, client: Optional[httpx.AsyncClient] = None) -> dict[str, ZoneForecastMetrics]:
        """Executes single-request batched Open-Meteo Forecast (33 points) + Marine (8 sea zones) polls."""
        points = self.all_zone_points
        if not points:
            return self.forecast_cache

        now_utc = datetime.now(timezone.utc)
        lats = ",".join(f"{p['lat']:.4f}" for p in points)
        lons = ",".join(f"{p['lon']:.4f}" for p in points)

        sea_pts = [p for p in points if p["zone_type"] == "sea_zone"]
        sea_lats = ",".join(f"{p['lat']:.4f}" for p in sea_pts)
        sea_lons = ",".join(f"{p['lon']:.4f}" for p in sea_pts)

        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=25.0)
            close_client = True

        try:
            f_resp = await client.get(
                OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": lats,
                    "longitude": lons,
                    "hourly": "precipitation_probability,cape,weather_code,wind_speed_10m,wind_gusts_10m,wind_direction_10m",
                    "forecast_days": 1,
                    "timezone": "UTC",
                },
            )
            f_resp.raise_for_status()
            forecast_payload = f_resp.json()

            marine_payload = None
            if sea_pts:
                try:
                    m_resp = await client.get(
                        OPEN_METEO_MARINE_URL,
                        params={
                            "latitude": sea_lats,
                            "longitude": sea_lons,
                            "hourly": "wave_height",
                            "forecast_days": 1,
                            "timezone": "UTC",
                        },
                    )
                    m_resp.raise_for_status()
                    marine_payload = m_resp.json()
                except Exception as marine_exc:
                    logger.warning("Open-Meteo Marine batch poll failed (continuing without wave height): %s", marine_exc)

            self.forecast_cache = self.parse_batched_open_meteo(
                points=points,
                forecast_payload=forecast_payload,
                marine_payload=marine_payload,
                now_utc=now_utc,
            )
            self.last_polled_at = now_utc.isoformat()
            logger.info(
                "Hazard Open-Meteo batch polled: %d total zones (%d districts + %d sea zones)",
                len(self.forecast_cache),
                len(self._district_points),
                len(sea_pts),
            )
            return self.forecast_cache
        except Exception as exc:
            logger.warning("Hazard Open-Meteo batch poll failed: %s", exc)
            return self.forecast_cache
        finally:
            if close_client:
                await client.aclose()

    def compute_hazard_probabilities(
        self,
        districts: Sequence[DistrictThreat],
        sea_zone_coverages: dict[str, SeaZoneCoverage],
        forecast_metrics: Optional[dict[str, ZoneForecastMetrics]] = None,
        alert_threshold_pct: Optional[int] = None,
    ) -> tuple[list[DistrictThreat], list[SeaZoneHazard], dict[str, Any]]:
        """
        Fuses Himawari-9 storm coverages (0m & +60m) with batched Open-Meteo forecasts to attach
        0-100% hazard probabilities to every land district and offshore sea zone.
        """
        metrics_map = forecast_metrics if forecast_metrics is not None else self.forecast_cache
        thresh = self.alert_threshold_pct if alert_threshold_pct is None else int(alert_threshold_pct)

        enriched_districts: list[DistrictThreat] = []
        for d in districts:
            m = metrics_map.get(d.district_id) or ZoneForecastMetrics(zone_id=d.district_id, zone_type="district")
            cov_0 = float(d.coverage_by_horizon.get("0m", d.affected_area_pct))
            cov_60 = float(d.coverage_by_horizon.get("+60m", 0.0))
            max_storm = max(cov_0, cov_60)
            deep_0 = float(d.deep_convective_by_horizon.get("0m", 0.0))
            deep_60 = float(d.deep_convective_by_horizon.get("+60m", 0.0))
            deep_pct = max(deep_0, deep_60)

            hz = compute_zone_hazard_probabilities(
                om_precip_prob=m.om_precip_prob,
                max_storm_coverage_pct=max_storm,
                deep_convective_pct=deep_pct,
                cape_jkg=m.cape_jkg,
                weather_code=m.weather_code,
                wind_speed_kmh=m.wind_speed_kmh,
                wind_gusts_kmh=m.wind_gusts_kmh or d.wind_gusts_kmh,
                wind_direction_deg=m.wind_direction_deg,
                is_sea_zone=False,
                wave_height_m=None,
                alert_threshold_pct=thresh,
            )
            enriched_districts.append(d.model_copy(update={"hazards": hz}))

        sea_zones: list[SeaZoneHazard] = []
        sea_features_out: list[dict[str, Any]] = []
        for feat in self._sea_features:
            props = feat.get("properties") or {}
            zid = str(props["zone_id"])
            name = str(props.get("name") or zid)
            centroid = list(props.get("centroid") or [80.0, 6.0])
            cov = sea_zone_coverages.get(zid) or SeaZoneCoverage(zone_id=zid, name=name)
            m = metrics_map.get(zid) or ZoneForecastMetrics(zone_id=zid, zone_type="sea_zone")

            max_storm = max(cov.storm_coverage_pct, cov.projected_60m_pct)
            deep_pct = max(cov.deep_convective_pct, cov.deep_convective_60m_pct)

            hz = compute_zone_hazard_probabilities(
                om_precip_prob=m.om_precip_prob,
                max_storm_coverage_pct=max_storm,
                deep_convective_pct=deep_pct,
                cape_jkg=m.cape_jkg,
                weather_code=m.weather_code,
                wind_speed_kmh=m.wind_speed_kmh,
                wind_gusts_kmh=m.wind_gusts_kmh,
                wind_direction_deg=m.wind_direction_deg,
                is_sea_zone=True,
                wave_height_m=m.wave_height_m,
                alert_threshold_pct=thresh,
            )

            sz_obj = SeaZoneHazard(
                zone_id=zid,
                name=name,
                zone_type="sea_zone",
                centroid=centroid,
                eta_min=cov.eta_min,
                storm_coverage_pct=cov.storm_coverage_pct,
                projected_60m_pct=cov.projected_60m_pct,
                deep_convective_pct=cov.deep_convective_pct,
                deep_convective_60m_pct=cov.deep_convective_60m_pct,
                peak_coverage_pct=cov.peak_coverage_pct,
                hazards=hz,
                geometry=feat["geometry"],
            )
            sea_zones.append(sz_obj)
            sea_features_out.append(
                {
                    "type": "Feature",
                    "properties": {
                        "zone_id": zid,
                        "name": name,
                        "zone_type": "sea_zone",
                        "centroid": centroid,
                        "eta_min": cov.eta_min,
                        "storm_coverage_pct": cov.storm_coverage_pct,
                        "projected_60m_pct": cov.projected_60m_pct,
                        "hazards": hz.model_dump(),
                    },
                    "geometry": feat["geometry"],
                }
            )

        sea_zones.sort(key=lambda z: (-z.hazards.max_hazard_prob, z.name))
        sea_geojson = {"type": "FeatureCollection", "features": sea_features_out}
        return enriched_districts, sea_zones, sea_geojson
