from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

import httpx

from config import Settings
from models import (
    GaugeHistoryPoint,
    GaugesResponse,
    GaugeStatus,
    GaugeTrend,
    HydrologySourceStatus,
    RiverGaugeReading,
)
from storage.db import Database

logger = logging.getLogger(__name__)

# Canonical upstream & downstream routing for Sri Lanka's four primary wet-zone river basins
BASIN_STATION_ROLES: dict[str, dict[str, list[str]]] = {
    "kelani": {
        "upstream": ["kitulgala", "deraniyagala", "kithulgala"],
        "downstream": ["glencourse", "hanwella", "nagalagam street", "nagalagam"],
    },
    "kalu": {
        "upstream": ["ratnapura", "pelmadulla"],
        "downstream": ["ellagawa", "millakanda", "putupaula"],
    },
    "gin": {
        "upstream": ["tawalama", "thawalama"],
        "downstream": ["baddegama"],
    },
    "nilwala": {
        "upstream": ["pitabeddara", "panadugama"],
        "downstream": ["thalgahagoda", "talgahagoda"],
    },
}

RIVER_TO_BASIN: dict[str, str] = {
    "kelani": "kelani",
    "kalu": "kalu",
    "gin": "gin",
    "nilwala": "nilwala",
}

STALE_AFTER_HOURS = 6.0


def _pick(attrs: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    lower_map = {k.lower(): v for k, v in attrs.items()}
    for key in candidates:
        val = lower_map.get(key.lower())
        if val is not None and val != "":
            return val
    return default


def _to_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _parse_timestamp(raw: Any, fallback: datetime) -> str:
    if raw is None or raw == "":
        return fallback.isoformat()
    # ArcGIS FeatureServer timestamps are often epoch milliseconds
    if isinstance(raw, (int, float)):
        ts_sec = raw / 1000.0 if raw > 1e11 else float(raw)
        return datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat()
    if isinstance(raw, str):
        s = raw.strip()
        if s.isdigit():
            num = float(s)
            ts_sec = num / 1000.0 if num > 1e11 else num
            return datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    return fallback.isoformat()


def infer_basin_and_role(
    station_name: str, river_name: str
) -> tuple[str, Literal["upstream", "downstream", "other"]]:
    s_norm = re.sub(r"\s+", " ", station_name.strip().lower())
    r_norm = re.sub(r"\s+", " ", river_name.strip().lower())

    for basin_id, roles in BASIN_STATION_ROLES.items():
        if any(alias in s_norm for alias in roles["upstream"]):
            return basin_id, "upstream"
        if any(alias in s_norm for alias in roles["downstream"]):
            return basin_id, "downstream"

    for keyword, basin_id in RIVER_TO_BASIN.items():
        if keyword in r_norm or keyword in s_norm:
            return basin_id, "other"

    return "other", "other"


def classify_gauge_status(
    water_level_ft: float,
    alert_level_ft: Optional[float],
    minor_flood_ft: Optional[float],
    major_flood_ft: Optional[float],
) -> GaugeStatus:
    if major_flood_ft is not None and water_level_ft >= major_flood_ft:
        return "MAJOR_FLOOD"
    if minor_flood_ft is not None and water_level_ft >= minor_flood_ft:
        return "MINOR_FLOOD"
    if alert_level_ft is not None and water_level_ft >= alert_level_ft:
        return "ALERT"
    return "NORMAL"


def compute_rate_and_trend(history: list[dict[str, Any]]) -> tuple[float, GaugeTrend]:
    """
    Computes rate of rise (ft/hr) over the last 1-3 hours from chronological readings
    stored in SQLite, returning (rate_of_rise_ft_hr, trend).
    """
    if len(history) < 2:
        return 0.0, "STEADY"

    latest = history[-1]
    latest_dt = datetime.fromisoformat(latest["reading_timestamp"].replace("Z", "+00:00"))
    latest_val = float(latest["water_level_ft"])

    # Prefer a baseline reading between 1 and 3 hours before latest; fall back to earliest within 3.5h
    baseline = None
    for row in reversed(history[:-1]):
        row_dt = datetime.fromisoformat(row["reading_timestamp"].replace("Z", "+00:00"))
        delta_hr = (latest_dt - row_dt).total_seconds() / 3600.0
        if 0.75 <= delta_hr <= 3.25:
            baseline = (row, delta_hr)
            # Keep looking slightly further back up to ~2-3h for stability, or stop once >= 1h
            if delta_hr >= 1.0:
                break
        elif 0.1 <= delta_hr < 0.75 and baseline is None:
            baseline = (row, delta_hr)

    if baseline is None:
        return 0.0, "STEADY"

    base_row, delta_hr = baseline
    if delta_hr <= 0.01:
        return 0.0, "STEADY"

    rate = round((latest_val - float(base_row["water_level_ft"])) / delta_hr, 3)
    if rate >= 0.08:
        return rate, "RISING"
    if rate <= -0.08:
        return rate, "FALLING"
    return rate, "STEADY"


def _slug_id(name: str, fallback_idx: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or f"station_{fallback_idx}"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class IrrigationArcGISPoller:
    """
    Polls Sri Lanka Department of Irrigation ArcGIS Online REST Feature Services
    (gauges + 24hr rainfall), normalizes readings, persists 72-hour history in SQLite,
    and falls back gracefully without ever silently masking stale/unavailable telemetry.
    """

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.source_status: HydrologySourceStatus = "unavailable"
        self.last_updated: Optional[str] = None
        self.last_error: Optional[str] = None

    @staticmethod
    def parse_arcgis_rainfall(payload: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for feat in payload.get("features", []) or []:
            attrs = feat.get("attributes") or {}
            geom = feat.get("geometry") or {}
            name = str(
                _pick(
                    attrs,
                    ["station_name", "station", "name", "gauging_station", "location"],
                    "",
                )
            ).strip()
            rain = _to_float(
                _pick(
                    attrs,
                    [
                        "rainfall_24h_mm",
                        "rainfall_24hr",
                        "rain_24h",
                        "rain_24hr",
                        "rainfall",
                        "rf_24h",
                        "value",
                    ],
                )
            )
            lat = _to_float(_pick(attrs, ["lat", "latitude", "y"], geom.get("y")))
            lon = _to_float(_pick(attrs, ["lon", "lng", "longitude", "x"], geom.get("x")))
            if rain is not None:
                entries.append({"name": name.lower(), "lat": lat, "lon": lon, "rainfall_24h_mm": rain})
        return entries

    @staticmethod
    def _match_rainfall(
        station_name: str,
        lat: float,
        lon: float,
        rainfall_entries: list[dict[str, Any]],
    ) -> Optional[float]:
        if not rainfall_entries:
            return None
        s_low = station_name.strip().lower()
        for entry in rainfall_entries:
            if entry["name"] and (entry["name"] == s_low or entry["name"] in s_low or s_low in entry["name"]):
                return round(float(entry["rainfall_24h_mm"]), 1)
        # Spatial proximity fallback within 12 km
        best: tuple[float, float] | None = None
        for entry in rainfall_entries:
            if entry["lat"] is not None and entry["lon"] is not None:
                dist = _haversine_km(lat, lon, entry["lat"], entry["lon"])
                if dist <= 12.0 and (best is None or dist < best[0]):
                    best = (dist, float(entry["rainfall_24h_mm"]))
        return round(best[1], 1) if best is not None else None

    @classmethod
    def parse_arcgis_gauges(
        cls,
        payload: dict[str, Any],
        rainfall_entries: Optional[list[dict[str, Any]]] = None,
        fetched_at: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        if "error" in payload:
            raise ValueError(f"ArcGIS REST error: {payload['error']}")
        now_utc = fetched_at or datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        rain_list = rainfall_entries or []
        parsed: list[dict[str, Any]] = []

        for idx, feat in enumerate(payload.get("features", []) or []):
            attrs = feat.get("attributes") or {}
            geom = feat.get("geometry") or {}

            station_name = str(
                _pick(
                    attrs,
                    ["station_name", "gauging_station", "station", "name", "location"],
                    f"Station {idx + 1}",
                )
            ).strip()
            river_name = str(
                _pick(attrs, ["river_name", "river", "basin", "basin_name"], "Unknown River")
            ).strip()
            raw_id = _pick(attrs, ["station_id", "code", "objectid", "fid", "id"])
            station_id = str(raw_id).strip() if raw_id is not None else _slug_id(station_name, idx)
            # Prefer stable slug when objectid is just an integer row id
            if station_id.isdigit():
                station_id = _slug_id(station_name, idx)

            lat = _to_float(_pick(attrs, ["lat", "latitude", "y"], geom.get("y")))
            lon = _to_float(_pick(attrs, ["lon", "lng", "longitude", "x"], geom.get("x")))
            level = _to_float(
                _pick(
                    attrs,
                    [
                        "water_level_ft",
                        "water_level",
                        "waterlevel",
                        "level_ft",
                        "level",
                        "current_level",
                        "wl_ft",
                        "gauge_height",
                    ],
                )
            )
            if lat is None or lon is None or level is None:
                continue

            alert_ft = _to_float(_pick(attrs, ["alert_level_ft", "alert_level", "alert", "alert_ft"]))
            minor_ft = _to_float(_pick(attrs, ["minor_flood_ft", "minor_flood_level", "minor_flood", "minor_ft"]))
            major_ft = _to_float(_pick(attrs, ["major_flood_ft", "major_flood_level", "major_flood", "major_ft"]))
            rain_mm = _to_float(
                _pick(attrs, ["rainfall_24h_mm", "rainfall_24hr", "rain_24h", "rainfall"])
            )
            if rain_mm is None:
                rain_mm = cls._match_rainfall(station_name, lat, lon, rain_list)

            ts_raw = _pick(
                attrs,
                [
                    "reading_timestamp",
                    "reading_time",
                    "timestamp",
                    "last_updated",
                    "obs_time",
                    "date_time",
                    "editdate",
                ],
            )
            reading_ts = _parse_timestamp(ts_raw, now_utc)
            basin_id, _ = infer_basin_and_role(station_name, river_name)

            parsed.append(
                {
                    "station_id": station_id,
                    "station_name": station_name,
                    "river_name": river_name,
                    "basin_id": basin_id,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "water_level_ft": round(level, 2),
                    "alert_level_ft": round(alert_ft, 2) if alert_ft is not None else None,
                    "minor_flood_ft": round(minor_ft, 2) if minor_ft is not None else None,
                    "major_flood_ft": round(major_ft, 2) if major_ft is not None else None,
                    "rainfall_24h_mm": round(rain_mm, 1) if rain_mm is not None else None,
                    "reading_timestamp": reading_ts,
                    "fetched_at": now_iso,
                }
            )
        return parsed

    @classmethod
    def parse_fallback_waterlevels(cls, payload: Any, fetched_at: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Normalizes fallback JSON from waterlevels.lk/api/levels (list or {data/stations: [...]})"""
        now_utc = fetched_at or datetime.now(timezone.utc)
        now_iso = now_utc.isoformat()
        if isinstance(payload, dict):
            rows = payload.get("data") or payload.get("stations") or payload.get("levels") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        parsed: list[dict[str, Any]] = []
        for idx, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            station_name = str(
                _pick(item, ["station_name", "station", "name", "gauging_station"], f"Station {idx + 1}")
            ).strip()
            river_name = str(_pick(item, ["river_name", "river", "basin"], "Unknown River")).strip()
            lat = _to_float(_pick(item, ["lat", "latitude"]))
            lon = _to_float(_pick(item, ["lon", "lng", "longitude"]))
            level = _to_float(_pick(item, ["water_level_ft", "water_level", "level", "current_level"]))
            if lat is None or lon is None or level is None:
                continue
            basin_id, _ = infer_basin_and_role(station_name, river_name)
            parsed.append(
                {
                    "station_id": _slug_id(station_name, idx),
                    "station_name": station_name,
                    "river_name": river_name,
                    "basin_id": basin_id,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    "water_level_ft": round(level, 2),
                    "alert_level_ft": _to_float(_pick(item, ["alert_level_ft", "alert_level", "alert"])),
                    "minor_flood_ft": _to_float(_pick(item, ["minor_flood_ft", "minor_flood_level", "minor_flood"])),
                    "major_flood_ft": _to_float(_pick(item, ["major_flood_ft", "major_flood_level", "major_flood"])),
                    "rainfall_24h_mm": _to_float(_pick(item, ["rainfall_24h_mm", "rainfall", "rain_24h"])),
                    "reading_timestamp": _parse_timestamp(_pick(item, ["timestamp", "updated_at", "reading_time"]), now_utc),
                    "fetched_at": now_iso,
                }
            )
        return parsed

    async def poll(self, client: Optional[httpx.AsyncClient] = None) -> GaugesResponse:
        """
        Polls configured ArcGIS REST Feature Services (gauges + optional 24h rainfall).
        On failure or missing URL, attempts waterlevels.lk fallback if enabled, else marks
        cached data as `stale` or `unavailable` (never silently reporting normal).
        """
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=15.0)
            close_client = True

        arcgis_params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "json",
        }
        now_utc = datetime.now(timezone.utc)
        errors: list[str] = []

        try:
            # 1. Primary: Irrigation Dept ArcGIS Online REST Feature Service
            if self.settings.irrigation_arcgis_gauges_url.strip():
                try:
                    rain_entries: list[dict[str, Any]] = []
                    if self.settings.irrigation_arcgis_rainfall_url.strip():
                        try:
                            r_resp = await client.get(
                                self.settings.irrigation_arcgis_rainfall_url.strip(),
                                params=arcgis_params,
                            )
                            r_resp.raise_for_status()
                            rain_entries = self.parse_arcgis_rainfall(r_resp.json())
                        except Exception as rain_exc:
                            logger.warning("ArcGIS 24hr rainfall layer failed (continuing with gauges): %s", rain_exc)

                    g_resp = await client.get(
                        self.settings.irrigation_arcgis_gauges_url.strip(),
                        params=arcgis_params,
                    )
                    g_resp.raise_for_status()
                    rows = self.parse_arcgis_gauges(g_resp.json(), rainfall_entries=rain_entries, fetched_at=now_utc)
                    if not rows:
                        raise ValueError("ArcGIS FeatureServer returned 0 valid gauge features")

                    self.db.upsert_gauge_readings(rows)
                    self.db.prune_gauge_readings(max_age_hours=72)
                    self.source_status = "arcgis_live"
                    self.last_updated = now_utc.isoformat()
                    self.last_error = None
                    logger.info("Irrigation ArcGIS polled: %d river gauges persisted", len(rows))
                    return self.get_gauges_response()
                except Exception as exc:
                    msg = f"ArcGIS poll failed: {exc}"
                    logger.warning(msg)
                    errors.append(msg)
            else:
                errors.append("IRRIGATION_ARCGIS_GAUGES_URL not configured")

            # 2. Optional Fallback: waterlevels.lk
            if self.settings.irrigation_fallback_enabled and self.settings.irrigation_fallback_url.strip():
                try:
                    fb_resp = await client.get(self.settings.irrigation_fallback_url.strip())
                    fb_resp.raise_for_status()
                    rows = self.parse_fallback_waterlevels(fb_resp.json(), fetched_at=now_utc)
                    if rows:
                        self.db.upsert_gauge_readings(rows)
                        self.db.prune_gauge_readings(max_age_hours=72)
                        self.source_status = "fallback"
                        self.last_updated = now_utc.isoformat()
                        self.last_error = "; ".join(errors)
                        logger.info("Irrigation fallback (%s) returned %d stations", self.settings.irrigation_fallback_url, len(rows))
                        return self.get_gauges_response()
                    errors.append("Fallback endpoint returned 0 valid gauge stations")
                except Exception as fb_exc:
                    msg = f"Fallback poll failed: {fb_exc}"
                    logger.warning(msg)
                    errors.append(msg)

            # 3. Graceful degradation: check cached SQLite readings -> stale vs unavailable
            cached = self.db.get_latest_gauges(max_age_hours=72)
            self.last_error = "; ".join(errors)
            if cached:
                self.source_status = "stale"
                self.last_updated = max(r["fetched_at"] for r in cached)
            else:
                self.source_status = "unavailable"
                self.last_updated = None
            return self.get_gauges_response()
        finally:
            if close_client:
                await client.aclose()

    def get_gauges_response(self) -> GaugesResponse:
        """Builds normalized RiverGaugeReading objects with SQLite rate-of-rise, trend, status, and 24h history."""
        latest_rows = self.db.get_latest_gauges(max_age_hours=72)
        now_utc = datetime.now(timezone.utc)
        stations: list[RiverGaugeReading] = []

        # If data was marked live/fallback earlier, verify it hasn't aged past STALE_AFTER_HOURS
        effective_status: HydrologySourceStatus = self.source_status
        if not latest_rows:
            effective_status = "unavailable"
        elif self.last_updated:
            try:
                age_hr = (now_utc - datetime.fromisoformat(self.last_updated.replace("Z", "+00:00"))).total_seconds() / 3600.0
                if age_hr > STALE_AFTER_HOURS:
                    effective_status = "stale"
            except ValueError:
                pass

        for row in latest_rows:
            history_rows = self.db.get_gauge_history(row["station_id"], hours=24)
            rate_ft_hr, trend = compute_rate_and_trend(history_rows)
            status = classify_gauge_status(
                row["water_level_ft"],
                row["alert_level_ft"],
                row["minor_flood_ft"],
                row["major_flood_ft"],
            )
            basin_id, role = infer_basin_and_role(row["station_name"], row["river_name"])
            history_pts = [
                GaugeHistoryPoint(timestamp=h["reading_timestamp"], water_level_ft=float(h["water_level_ft"]))
                for h in history_rows
            ]
            stations.append(
                RiverGaugeReading(
                    station_id=row["station_id"],
                    station_name=row["station_name"],
                    river_name=row["river_name"],
                    basin_id=basin_id,
                    role=role,
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    water_level_ft=float(row["water_level_ft"]),
                    alert_level_ft=row["alert_level_ft"],
                    minor_flood_ft=row["minor_flood_ft"],
                    major_flood_ft=row["major_flood_ft"],
                    rainfall_24h_mm=row["rainfall_24h_mm"],
                    reading_timestamp=row["reading_timestamp"],
                    fetched_at=row["fetched_at"],
                    rate_of_rise_ft_hr=rate_ft_hr,
                    trend=trend,
                    status=status,
                    history_24h=history_pts,
                )
            )

        return GaugesResponse(
            source_status=effective_status,
            last_updated=self.last_updated,
            error_detail=self.last_error,
            stations=stations,
        )
