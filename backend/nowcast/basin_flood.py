from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from shapely.geometry import shape
from shapely.ops import unary_union

from models import (
    BasinRiskLevel,
    BasinZoneCoverage,
    RiverBasinRisk,
    RiverGaugeReading,
    UpstreamSurgeWarning,
)

BASIN_RISK_RANK: dict[BasinRiskLevel, int] = {
    "CRITICAL": 4,
    "WARNING": 3,
    "WATCH": 2,
    "NORMAL": 1,
}

UPSTREAM_SURGE_RAPID_RATE_FT_HR = 0.5


class BasinFloodAnalyzer:
    """
    Fuses Himawari-9 storm trajectories (from NowcastEngine) with Open-Meteo 24h upper-catchment
    rainfall and live Sri Lanka Dept. of Irrigation ArcGIS river gauge telemetry across the
    four primary wet-zone river basins: Kelani, Kalu, Gin, and Nilwala.
    """

    def __init__(self, basins_geojson_path: Path | str):
        self.basins_geojson_path = Path(basins_geojson_path)
        self._raw_geojson = json.loads(self.basins_geojson_path.read_text(encoding="utf-8"))
        self._basin_features: list[dict[str, Any]] = self._raw_geojson.get("features", [])
        self._basin_shapes: dict[str, Any] = {}
        for feat in self._basin_features:
            bid = feat["properties"]["basin_id"]
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            self._basin_shapes[bid] = geom

    def compute_coverage_from_polygons(
        self,
        current_polygons: list[tuple[str, Any]],
        projected_60m_polygons: list[tuple[str, Any]],
    ) -> dict[str, BasinZoneCoverage]:
        """
        Helper allowing direct Shapely intersection when calling BasinFloodAnalyzer standalone
        (outside NowcastEngine.run()).
        """
        u_curr_all = unary_union([g for _, g in current_polygons]) if current_polygons else None
        u_curr_deep = (
            unary_union([g for lbl, g in current_polygons if lbl == "deep_convective"])
            if any(lbl == "deep_convective" for lbl, _ in current_polygons)
            else None
        )
        u_60_all = unary_union([g for _, g in projected_60m_polygons]) if projected_60m_polygons else None
        u_60_deep = (
            unary_union([g for lbl, g in projected_60m_polygons if lbl == "deep_convective"])
            if any(lbl == "deep_convective" for lbl, _ in projected_60m_polygons)
            else None
        )

        out: dict[str, BasinZoneCoverage] = {}
        for feat in self._basin_features:
            props = feat["properties"]
            bid = props["basin_id"]
            geom = self._basin_shapes[bid]
            area = geom.area or 1.0
            c_all = round(min(100.0, (geom.intersection(u_curr_all).area / area) * 100.0), 1) if u_curr_all else 0.0
            c_deep = round(min(100.0, (geom.intersection(u_curr_deep).area / area) * 100.0), 1) if u_curr_deep else 0.0
            p60_all = round(min(100.0, (geom.intersection(u_60_all).area / area) * 100.0), 1) if u_60_all else 0.0
            p60_deep = round(min(100.0, (geom.intersection(u_60_deep).area / area) * 100.0), 1) if u_60_deep else 0.0
            eta = 0 if c_all >= 10.0 else (60 if p60_all >= 10.0 else None)
            out[bid] = BasinZoneCoverage(
                basin_id=bid,
                name=props["name"],
                eta_min=eta,
                coverage_by_horizon={"0m": c_all, "+30m": round((c_all + p60_all) / 2, 1), "+60m": p60_all, "+120m": p60_all},
                deep_convective_by_horizon={"0m": c_deep, "+60m": p60_deep},
                storm_coverage_pct=c_all,
                projected_60m_pct=p60_all,
                deep_convective_pct=c_deep,
                deep_convective_60m_pct=p60_deep,
                peak_coverage_pct=max(c_all, p60_all),
            )
        return out

    def analyze(
        self,
        basin_coverages: dict[str, BasinZoneCoverage],
        open_meteo_rainfall_24h: dict[str, float],
        gauges: Optional[list[RiverGaugeReading]] = None,
    ) -> tuple[list[RiverBasinRisk], dict[str, Any], list[UpstreamSurgeWarning]]:
        gauges_list = gauges or []
        results: list[RiverBasinRisk] = []
        surge_warnings: list[UpstreamSurgeWarning] = []
        geojson_features: list[dict[str, Any]] = []

        for feat in self._basin_features:
            props = feat["properties"]
            bid = props["basin_id"]
            name = props["name"]
            uc = props.get("upper_catchment") or {}
            uc_stations = list(uc.get("stations") or [])
            ds_stations = list(props.get("downstream_stations") or [])
            lag_hours = int(props.get("drainage_lag_hours", 12))
            lag_window = list(props.get("drainage_lag_window_hours") or [max(4, lag_hours - 3), lag_hours + 3])

            cov = basin_coverages.get(bid) or BasinZoneCoverage(basin_id=bid, name=name)
            curr_pct = cov.storm_coverage_pct
            proj60_pct = cov.projected_60m_pct
            deep_pct = max(cov.deep_convective_pct, cov.deep_convective_60m_pct)

            basin_gauges = [g for g in gauges_list if g.basin_id == bid]
            upstream_gauges = [g for g in basin_gauges if g.role == "upstream"]
            downstream_gauges = [g for g in basin_gauges if g.role == "downstream"]

            # Rainfall fusion: prefer ArcGIS observed 24h rain when available & higher, else Open-Meteo
            om_rain = round(float(open_meteo_rainfall_24h.get(bid, 0.0)), 1)
            arcgis_rains = [g.rainfall_24h_mm for g in basin_gauges if g.rainfall_24h_mm is not None]
            if arcgis_rains:
                max_arcgis_rain = round(max(arcgis_rains), 1)
                if max_arcgis_rain >= om_rain:
                    rain_24h = max_arcgis_rain
                    rain_source = "irrigation_arcgis"
                else:
                    rain_24h = om_rain
                    rain_source = "open_meteo_fused"
            else:
                rain_24h = om_rain
                rain_source = "open_meteo"

            # Inspect gauge states
            any_major = any(g.status == "MAJOR_FLOOD" for g in basin_gauges)
            any_minor = any(g.status == "MINOR_FLOOD" for g in basin_gauges)
            any_minor_rising = any(g.status == "MINOR_FLOOD" and g.trend == "RISING" for g in basin_gauges)
            any_alert = any(g.status == "ALERT" for g in basin_gauges)
            any_alert_rising = any(g.status == "ALERT" and g.trend == "RISING" for g in basin_gauges)
            upstream_rapid = [
                g for g in upstream_gauges if g.rate_of_rise_ft_hr >= UPSTREAM_SURGE_RAPID_RATE_FT_HR
            ]
            active_heavy_storm = deep_pct >= 20.0 or curr_pct >= 25.0 or proj60_pct >= 30.0

            # Combined Satellite + Hydrological Risk Escalation
            reasons: list[str] = []
            risk_level: BasinRiskLevel = "NORMAL"

            # CRITICAL tier
            if (
                any_major
                or any_minor_rising
                or (any_minor and deep_pct >= 25.0)
                or (bool(upstream_rapid) and active_heavy_storm)
                or ((curr_pct >= 35.0 or proj60_pct >= 40.0) and rain_24h >= 100.0)
            ):
                risk_level = "CRITICAL"
                if any_major:
                    reasons.append("Gauge at MAJOR FLOOD level")
                if any_minor_rising:
                    reasons.append("Gauge at MINOR FLOOD and RISING")
                if any_minor and deep_pct >= 25.0:
                    reasons.append(f"MINOR FLOOD gauge under {deep_pct:.0f}% deep convective cover")
                if upstream_rapid and active_heavy_storm:
                    reasons.append(
                        f"Upstream rapid rise ({max(g.rate_of_rise_ft_hr for g in upstream_rapid):.2f} ft/hr) under active storm cells"
                    )
                if (curr_pct >= 35.0 or proj60_pct >= 40.0) and rain_24h >= 100.0:
                    reasons.append(f"Heavy catchment rain ({rain_24h:.0f} mm/24h) + storm coverage ({max(curr_pct, proj60_pct):.0f}%)")
            # WARNING tier
            elif (
                any_minor
                or any_alert_rising
                or (any_alert and max(curr_pct, proj60_pct) >= 20.0)
                or ((curr_pct >= 25.0 or proj60_pct >= 30.0) and rain_24h >= 50.0)
            ):
                risk_level = "WARNING"
                if any_minor:
                    reasons.append("Gauge at MINOR FLOOD level")
                if any_alert_rising:
                    reasons.append("Gauge at ALERT level and RISING")
                if any_alert and max(curr_pct, proj60_pct) >= 20.0:
                    reasons.append(f"ALERT gauge with {max(curr_pct, proj60_pct):.0f}% storm coverage")
                if (curr_pct >= 25.0 or proj60_pct >= 30.0) and rain_24h >= 50.0:
                    reasons.append(f"Projected/current storm ({max(curr_pct, proj60_pct):.0f}%) with {rain_24h:.0f} mm/24h rain")
            # WATCH tier
            elif (
                any_alert
                or curr_pct >= 15.0
                or proj60_pct >= 20.0
                or rain_24h >= 50.0
            ):
                risk_level = "WATCH"
                if any_alert:
                    reasons.append("Gauge at ALERT level")
                if curr_pct >= 15.0 or proj60_pct >= 20.0:
                    reasons.append(f"Satellite storm coverage {curr_pct:.0f}% now / {proj60_pct:.0f}% at +60m")
                if rain_24h >= 50.0:
                    reasons.append(f"Elevated 24h catchment rainfall ({rain_24h:.0f} mm)")
            else:
                reasons.append("Catchment rainfall, satellite coverage, and river levels within normal bounds")

            # Upstream Surge detection: upstream RISING while downstream stations are still NORMAL
            rising_up = [g for g in upstream_gauges if g.trend == "RISING"]
            downstream_normal = bool(downstream_gauges) and all(g.status == "NORMAL" for g in downstream_gauges)
            # Also allow surge detection if downstream gauges exist OR configured downstream stations haven't flooded yet
            has_upstream_surge = bool(rising_up) and (downstream_normal or (not downstream_gauges and len(ds_stations) > 0))
            if has_upstream_surge:
                max_rate = round(max(g.rate_of_rise_ft_hr for g in rising_up), 2)
                up_names = [g.station_name for g in rising_up]
                crest_window = f"{lag_window[0]}–{lag_window[1]} hours"
                target_ds = [g.station_name for g in downstream_gauges] or ds_stations
                msg = (
                    f"{name}: Upstream surge at {', '.join(up_names)} (+{max_rate:.2f} ft/hr) "
                    f"while downstream ({', '.join(target_ds)}) remains NORMAL. "
                    f"Estimated crest arrival in {crest_window}."
                )
                surge_warnings.append(
                    UpstreamSurgeWarning(
                        basin_id=bid,
                        basin_name=name,
                        upstream_stations=up_names,
                        downstream_stations=target_ds,
                        max_rate_of_rise_ft_hr=max_rate,
                        drainage_lag_hours=lag_hours,
                        drainage_lag_window_hours=lag_window,
                        estimated_crest_window=crest_window,
                        message=msg,
                    )
                )

            gauge_summary = {
                "total_gauges": len(basin_gauges),
                "highest_status": (
                    max(
                        basin_gauges,
                        key=lambda g: ["NORMAL", "ALERT", "MINOR_FLOOD", "MAJOR_FLOOD"].index(g.status),
                    ).status
                    if basin_gauges
                    else "NO_DATA"
                ),
                "max_rate_of_rise_ft_hr": round(max((g.rate_of_rise_ft_hr for g in basin_gauges), default=0.0), 2),
                "rising_count": sum(1 for g in basin_gauges if g.trend == "RISING"),
            }

            risk = RiverBasinRisk(
                basin_id=bid,
                name=name,
                risk_level=risk_level,
                eta_min=cov.eta_min,
                storm_coverage_pct=curr_pct,
                projected_60m_pct=proj60_pct,
                deep_convective_pct=cov.deep_convective_pct,
                deep_convective_60m_pct=cov.deep_convective_60m_pct,
                coverage_by_horizon=cov.coverage_by_horizon,
                peak_coverage_pct=cov.peak_coverage_pct,
                rainfall_24h_mm=rain_24h,
                rainfall_source=rain_source,
                drainage_lag_hours=lag_hours,
                drainage_lag_window_hours=lag_window,
                upper_catchment_stations=uc_stations,
                downstream_stations=ds_stations,
                upstream_surge=has_upstream_surge,
                gauge_summary=gauge_summary,
                reasons=reasons,
                geometry=feat["geometry"],
            )
            results.append(risk)

        results.sort(
            key=lambda b: (
                -BASIN_RISK_RANK[b.risk_level],
                b.eta_min if b.eta_min is not None else 999,
                -b.peak_coverage_pct,
                b.name,
            )
        )

        for r in results:
            geojson_features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "basin_id": r.basin_id,
                        "name": r.name,
                        "risk_level": r.risk_level,
                        "eta_min": r.eta_min,
                        "storm_coverage_pct": r.storm_coverage_pct,
                        "projected_60m_pct": r.projected_60m_pct,
                        "peak_coverage_pct": r.peak_coverage_pct,
                        "rainfall_24h_mm": r.rainfall_24h_mm,
                        "rainfall_source": r.rainfall_source,
                        "drainage_lag_hours": r.drainage_lag_hours,
                        "drainage_lag_window_hours": r.drainage_lag_window_hours,
                        "upper_catchment_stations": r.upper_catchment_stations,
                        "downstream_stations": r.downstream_stations,
                        "upstream_surge": r.upstream_surge,
                        "gauge_summary": r.gauge_summary,
                        "reasons": r.reasons,
                    },
                    "geometry": r.geometry,
                }
            )

        geojson_fc = {"type": "FeatureCollection", "features": geojson_features}
        return results, geojson_fc, surge_warnings
