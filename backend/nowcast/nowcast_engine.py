from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

from models import (
    BasinZoneCoverage,
    DistrictThreat,
    ProjectedStormCell,
    StormDetection,
    SurfaceStationTelemetry,
    ThreatLevel,
)

COMPASS_16 = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]

HORIZONS_MIN = (0, 30, 60, 120)
ETA_COVERAGE_THRESHOLD_PCT = 10.0
MAX_STORM_SPEED_KMH = 120.0

THREAT_RANK: dict[ThreatLevel, int] = {
    "SEVERE": 4,
    "HIGH": 3,
    "MODERATE": 2,
    "LOW": 1,
}


def bearing_to_compass_16(bearing_deg: float) -> str:
    idx = int(((bearing_deg % 360.0) + 11.25) // 22.5) % 16
    return COMPASS_16[idx]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


@dataclass
class _ZoneRecord:
    zone_id: str
    name: str
    centroid_lat: float
    centroid_lon: float
    geom: Any


@dataclass
class NowcastEngineOutput:
    storm_cells: list[ProjectedStormCell]
    storm_cells_geojson: dict[str, Any]
    projected_geojson: dict[str, dict[str, Any]]
    districts: list[DistrictThreat]
    basin_coverages: dict[str, BasinZoneCoverage]


class NowcastEngine:
    """
    Optical flow trajectory & geospatial ETA projection engine.
    Tracks Himawari-9 storm detections via averaged Farneback optical flow across the
    last 2-3 buffered frames, projects storm polygons forward to +30/+60/+120 minutes,
    and intersects horizon unions with Sri Lanka's 25 districts and 4 wet-zone river basins.
    """

    def __init__(
        self,
        districts_geojson_path: Path | str,
        basins_geojson_path: Path | str,
        bbox: tuple[float, float, float, float] = (75.0, 3.0, 86.0, 12.0),
        frame_size: tuple[int, int] = (640, 640),
    ):
        self.bbox = bbox
        self.frame_width, self.frame_height = frame_size
        self.districts = self._load_zones(Path(districts_geojson_path), id_key="district_id")
        self.basins = self._load_zones(Path(basins_geojson_path), id_key="basin_id")

    @staticmethod
    def _load_zones(path: Path, id_key: str) -> list[_ZoneRecord]:
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        records: list[_ZoneRecord] = []
        for feat in raw.get("features", []):
            props = feat.get("properties") or {}
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            c = geom.centroid
            name = str(props.get("name") or props.get(id_key) or "Unknown")
            zid = str(props.get(id_key) or name.lower().replace(" ", "_"))
            records.append(
                _ZoneRecord(
                    zone_id=zid,
                    name=name,
                    centroid_lat=float(props.get("centroid_lat", c.y)),
                    centroid_lon=float(props.get("centroid_lon", c.x)),
                    geom=geom,
                )
            )
        return records

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def compute_mean_flow(
        self,
        frames: list[np.ndarray] | list[tuple[str, np.ndarray]],
        default_dt_min: float = 10.0,
    ) -> np.ndarray:
        """
        Computes averaged dense OpenCV Farneback Optical Flow across the last 2-3 buffered
        frames, normalized to pixel displacement per minute: shape (H, W, 2) float32.
        """
        if len(frames) < 2:
            return np.zeros((self.frame_height, self.frame_width, 2), dtype=np.float32)

        recent = frames[-3:]
        parsed: list[tuple[Optional[datetime], np.ndarray]] = []
        for item in recent:
            if isinstance(item, tuple):
                ts_str, img = item
                dt_obj = None
                try:
                    dt_obj = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    pass
                parsed.append((dt_obj, self._to_gray(img)))
            else:
                parsed.append((None, self._to_gray(item)))

        flows_per_min: list[np.ndarray] = []
        for i in range(len(parsed) - 1):
            t0, g0 = parsed[i]
            t1, g1 = parsed[i + 1]
            if g0.shape != (self.frame_height, self.frame_width):
                g0 = cv2.resize(g0, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
            if g1.shape != (self.frame_height, self.frame_width):
                g1 = cv2.resize(g1, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)

            dt_min = default_dt_min
            if t0 is not None and t1 is not None:
                diff = (t1 - t0).total_seconds() / 60.0
                if 1.0 <= diff <= 60.0:
                    dt_min = diff

            flow = cv2.calcOpticalFlowFarneback(
                g0,
                g1,
                None,
                pyr_scale=0.5,
                levels=4,
                winsize=19,
                iterations=4,
                poly_n=7,
                poly_sigma=1.5,
                flags=0,
            )
            flows_per_min.append(flow.astype(np.float32) / float(dt_min))

        if not flows_per_min:
            return np.zeros((self.frame_height, self.frame_width, 2), dtype=np.float32)
        return np.mean(np.stack(flows_per_min, axis=0), axis=0).astype(np.float32)

    def project_storms(
        self,
        detections: list[StormDetection],
        mean_flow_per_min: np.ndarray,
    ) -> tuple[list[ProjectedStormCell], dict[int, list[tuple[str, Any]]]]:
        """
        Rasterizes each detected storm polygon, takes median (dx, dy) flow inside the polygon,
        converts to geographic drift (dlon_per_min, dlat_per_min), ground speed (capped at 120 km/h),
        bearing (0-360°), 16-point compass direction, and projects polygons to +30/+60/+120 min
        via shapely.affinity.translate.
        """
        lon_min, lat_min, lon_max, lat_max = self.bbox
        lon_span = lon_max - lon_min
        lat_span = lat_max - lat_min
        h, w = mean_flow_per_min.shape[:2]

        projected_cells: list[ProjectedStormCell] = []
        horizon_polys: dict[int, list[tuple[str, Any]]] = {hr: [] for hr in HORIZONS_MIN}

        for det in detections:
            if len(det.geo_polygon) < 3 or len(det.pixel_polygon) < 3:
                continue

            poly_now = Polygon(det.geo_polygon)
            if not poly_now.is_valid:
                poly_now = poly_now.buffer(0)
            if poly_now.is_empty:
                continue

            # Rasterize pixel polygon and sample median flow inside to reject cloud-edge noise
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.round(np.asarray(det.pixel_polygon, dtype=np.float32)).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(mask, [pts], 255)

            inside = mask > 0
            if np.any(inside):
                dx_px_min = float(np.median(mean_flow_per_min[inside, 0]))
                dy_px_min = float(np.median(mean_flow_per_min[inside, 1]))
            else:
                dx_px_min, dy_px_min = 0.0, 0.0

            # Convert pixel velocity to geographic drift (note: +y in image is south -> negative dlat)
            dlon_per_min = dx_px_min * (lon_span / float(w))
            dlat_per_min = -dy_px_min * (lat_span / float(h))

            clon, clat = float(det.centroid_geo[0]), float(det.centroid_geo[1])
            km_per_deg_lat = 111.32
            km_per_deg_lon = 111.32 * max(0.2, math.cos(math.radians(clat)))

            dx_kmh = dlon_per_min * 60.0 * km_per_deg_lon
            dy_kmh = dlat_per_min * 60.0 * km_per_deg_lat
            speed_kmh = math.hypot(dx_kmh, dy_kmh)

            if speed_kmh > MAX_STORM_SPEED_KMH:
                scale = MAX_STORM_SPEED_KMH / speed_kmh
                dx_kmh *= scale
                dy_kmh *= scale
                dlon_per_min *= scale
                dlat_per_min *= scale
                speed_kmh = MAX_STORM_SPEED_KMH

            if speed_kmh < 0.5:
                bearing_deg = 0.0
                compass_dir = "STAT"
            else:
                bearing_deg = (math.degrees(math.atan2(dx_kmh, dy_kmh)) + 360.0) % 360.0
                compass_dir = bearing_to_compass_16(bearing_deg)

            horizon_polys[0].append((det.label, poly_now))
            geom_proj: dict[str, dict[str, Any]] = {}
            for horizon in (30, 60, 120):
                shifted = translate(
                    poly_now,
                    xoff=dlon_per_min * float(horizon),
                    yoff=dlat_per_min * float(horizon),
                )
                if not shifted.is_valid:
                    shifted = shifted.buffer(0)
                horizon_polys[horizon].append((det.label, shifted))
                geom_proj[f"+{horizon}m"] = mapping(shifted)

            projected_cells.append(
                ProjectedStormCell(
                    cell_id=det.cell_id,
                    label=det.label,
                    confidence=det.confidence,
                    centroid_lon=round(clon, 5),
                    centroid_lat=round(clat, 5),
                    dx_kmh=round(dx_kmh, 2),
                    dy_kmh=round(dy_kmh, 2),
                    speed_kmh=round(speed_kmh, 1),
                    bearing_deg=round(bearing_deg, 1),
                    compass_dir=compass_dir,
                    geometry_now=mapping(poly_now),
                    geometry_projected=geom_proj,
                )
            )

        return projected_cells, horizon_polys

    @staticmethod
    def _build_horizon_unions(
        horizon_polys: dict[int, list[tuple[str, Any]]]
    ) -> tuple[dict[int, Any], dict[int, Any]]:
        all_unions: dict[int, Any] = {}
        deep_unions: dict[int, Any] = {}
        for hr in HORIZONS_MIN:
            items = horizon_polys.get(hr, [])
            all_list = [g for _, g in items if g is not None and not g.is_empty]
            deep_list = [g for lbl, g in items if lbl == "deep_convective" and g is not None and not g.is_empty]
            all_unions[hr] = unary_union(all_list) if all_list else None
            deep_unions[hr] = unary_union(deep_list) if deep_list else None
        return all_unions, deep_unions

    @staticmethod
    def _classify_district_threat(
        peak_all: float,
        peak_deep: float,
        eta_min: Optional[int],
        station: Optional[SurfaceStationTelemetry],
    ) -> ThreatLevel:
        rain = station.precipitation_mm if station else 0.0
        gust = station.wind_gusts_kmh if station else 0.0
        pdrop = station.pressure_drop_3h_hpa if station else 0.0

        # SEVERE: heavy deep convection overhead/imminent or severe surface telemetry fused with storm cell
        if (
            (peak_deep >= 35.0 and (eta_min is not None and eta_min <= 60))
            or (peak_all >= 55.0 and (rain >= 12.0 or gust >= 55.0 or pdrop >= 3.0))
            or (rain >= 25.0 or gust >= 70.0)
        ):
            return "SEVERE"

        if (
            peak_deep >= 15.0
            or peak_all >= 35.0
            or (peak_all >= 18.0 and (rain >= 5.0 or gust >= 40.0 or pdrop >= 2.0))
            or rain >= 10.0
            or gust >= 50.0
        ):
            return "HIGH"

        if (
            peak_all >= ETA_COVERAGE_THRESHOLD_PCT
            or rain >= 2.0
            or gust >= 32.0
            or pdrop >= 1.5
        ):
            return "MODERATE"

        return "LOW"

    def _evaluate_zones(
        self,
        horizon_polys: dict[int, list[tuple[str, Any]]],
        surface_stations: list[SurfaceStationTelemetry],
    ) -> tuple[list[DistrictThreat], dict[str, BasinZoneCoverage]]:
        """
        Unions storm polygons per horizon via unary_union (separately for all storms and
        deep_convective cores), intersects against all 25 Sri Lanka districts and 4 wet-zone
        river basins, and calculates coverage %, earliest arrival ETA (>=10%), and threat level.
        """
        all_unions, deep_unions = self._build_horizon_unions(horizon_polys)

        def eval_zone(zone: _ZoneRecord) -> tuple[dict[str, float], dict[str, float], Optional[int], float]:
            zone_area = zone.geom.area
            cov_all: dict[str, float] = {}
            cov_deep: dict[str, float] = {}
            eta: Optional[int] = None

            for hr in HORIZONS_MIN:
                key = "0m" if hr == 0 else f"+{hr}m"
                u_all = all_unions.get(hr)
                u_deep = deep_unions.get(hr)
                pct_all = (
                    round(min(100.0, (zone.geom.intersection(u_all).area / zone_area) * 100.0), 1)
                    if (u_all is not None and zone_area > 0)
                    else 0.0
                )
                pct_deep = (
                    round(min(100.0, (zone.geom.intersection(u_deep).area / zone_area) * 100.0), 1)
                    if (u_deep is not None and zone_area > 0)
                    else 0.0
                )
                cov_all[key] = pct_all
                cov_deep[key] = pct_deep
                if eta is None and pct_all >= ETA_COVERAGE_THRESHOLD_PCT:
                    eta = hr

            peak = max(cov_all.values()) if cov_all else 0.0
            return cov_all, cov_deep, eta, peak

        # 1. Districts
        district_threats: list[DistrictThreat] = []
        for dist in self.districts:
            cov_all, cov_deep, eta, peak_all = eval_zone(dist)
            peak_deep = max(cov_deep.values()) if cov_deep else 0.0

            nearest: Optional[SurfaceStationTelemetry] = None
            if surface_stations:
                nearest = min(
                    surface_stations,
                    key=lambda s: _haversine_km(dist.centroid_lat, dist.centroid_lon, s.lat, s.lon),
                )

            level = self._classify_district_threat(peak_all, peak_deep, eta, nearest)
            district_threats.append(
                DistrictThreat(
                    district_id=dist.zone_id,
                    name=dist.name,
                    threat_level=level,
                    eta_min=eta,
                    coverage_by_horizon=cov_all,
                    deep_convective_by_horizon=cov_deep,
                    affected_area_pct=cov_all.get("0m", 0.0),
                    peak_coverage_pct=peak_all,
                    nearest_station=nearest.name if nearest else "",
                    precipitation_mm=nearest.precipitation_mm if nearest else 0.0,
                    wind_gusts_kmh=nearest.wind_gusts_kmh if nearest else 0.0,
                    pressure_drop_3h_hpa=nearest.pressure_drop_3h_hpa if nearest else 0.0,
                )
            )

        # Sort by highest threat, soonest ETA, highest peak coverage, then name
        district_threats.sort(
            key=lambda d: (
                -THREAT_RANK[d.threat_level],
                d.eta_min if d.eta_min is not None else 999,
                -d.peak_coverage_pct,
                d.name,
            )
        )

        # 2. River Basins
        basin_coverages: dict[str, BasinZoneCoverage] = {}
        for basin in self.basins:
            cov_all, cov_deep, eta, peak_all = eval_zone(basin)
            basin_coverages[basin.zone_id] = BasinZoneCoverage(
                basin_id=basin.zone_id,
                name=basin.name,
                eta_min=eta,
                coverage_by_horizon=cov_all,
                deep_convective_by_horizon=cov_deep,
                storm_coverage_pct=cov_all.get("0m", 0.0),
                projected_60m_pct=cov_all.get("+60m", 0.0),
                deep_convective_pct=cov_deep.get("0m", 0.0),
                deep_convective_60m_pct=cov_deep.get("+60m", 0.0),
                peak_coverage_pct=peak_all,
            )

        return district_threats, basin_coverages

    def run(
        self,
        frames: list[np.ndarray] | list[tuple[str, np.ndarray]],
        detections: list[StormDetection],
        surface_stations: Optional[list[SurfaceStationTelemetry]] = None,
        default_dt_min: float = 10.0,
    ) -> NowcastEngineOutput:
        mean_flow = self.compute_mean_flow(frames, default_dt_min=default_dt_min)
        projected_cells, horizon_polys = self.project_storms(detections, mean_flow)
        districts, basin_coverages = self._evaluate_zones(horizon_polys, surface_stations or [])

        storm_features = [
            {
                "type": "Feature",
                "properties": {
                    "cell_id": c.cell_id,
                    "label": c.label,
                    "confidence": c.confidence,
                    "speed_kmh": c.speed_kmh,
                    "bearing_deg": c.bearing_deg,
                    "compass_dir": c.compass_dir,
                    "centroid_lon": c.centroid_lon,
                    "centroid_lat": c.centroid_lat,
                },
                "geometry": c.geometry_now,
            }
            for c in projected_cells
        ]
        storm_geojson = {"type": "FeatureCollection", "features": storm_features}

        projected_geojson: dict[str, dict[str, Any]] = {}
        for key in ("+30m", "+60m", "+120m"):
            projected_geojson[key] = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "cell_id": c.cell_id,
                            "label": c.label,
                            "horizon": key,
                            "speed_kmh": c.speed_kmh,
                            "compass_dir": c.compass_dir,
                        },
                        "geometry": c.geometry_projected[key],
                    }
                    for c in projected_cells
                    if key in c.geometry_projected
                ],
            }

        return NowcastEngineOutput(
            storm_cells=projected_cells,
            storm_cells_geojson=storm_geojson,
            projected_geojson=projected_geojson,
            districts=districts,
            basin_coverages=basin_coverages,
        )
