from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

StormClass = Literal["deep_convective", "cyclonic_band", "monsoon_shield"]
ThreatLevel = Literal["LOW", "MODERATE", "HIGH", "SEVERE"]
BasinRiskLevel = Literal["NORMAL", "WATCH", "WARNING", "CRITICAL"]
GaugeStatus = Literal["NORMAL", "ALERT", "MINOR_FLOOD", "MAJOR_FLOOD"]
GaugeTrend = Literal["RISING", "STEADY", "FALLING"]
HydrologySourceStatus = Literal["arcgis_live", "fallback", "stale", "unavailable"]
ActiveLearningTrigger = Literal["low_conf", "sensor_mismatch", "boundary_artifact"]
ActiveLearningStatus = Literal["pending", "approved", "rejected"]

STORM_CLASS_TO_ID: dict[StormClass, int] = {
    "deep_convective": 0,
    "cyclonic_band": 1,
    "monsoon_shield": 2,
}
ID_TO_STORM_CLASS: dict[int, StormClass] = {v: k for k, v in STORM_CLASS_TO_ID.items()}


class FrameMetadata(BaseModel):
    timestamp: str
    filename: str
    image_url: str
    width: int = 640
    height: int = 640
    mean_intensity: float = 0.0
    fetched_at: str


class StormDetection(BaseModel):
    cell_id: str
    label: StormClass
    confidence: float = Field(ge=0.0, le=1.0)
    pixel_polygon: list[list[float]]
    normalized_polygon: list[list[float]]
    geo_polygon: list[list[float]]
    centroid_px: list[float]
    centroid_geo: list[float]
    area_px: float
    peak_intensity: float = 0.0


class DetectionResult(BaseModel):
    backend: str
    inference_ms: float
    detections: list[StormDetection] = Field(default_factory=list)


class ProjectedStormCell(BaseModel):
    cell_id: str
    label: StormClass
    confidence: float
    centroid_lon: float
    centroid_lat: float
    dx_kmh: float
    dy_kmh: float
    speed_kmh: float
    bearing_deg: float
    compass_dir: str
    geometry_now: dict[str, Any]
    geometry_projected: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SurfaceStationTelemetry(BaseModel):
    name: str
    lat: float
    lon: float
    precipitation_mm: float = 0.0
    wind_gusts_kmh: float = 0.0
    surface_pressure_hpa: float = 1010.0
    pressure_drop_3h_hpa: float = 0.0
    observed_at: str = ""


class DistrictThreat(BaseModel):
    district_id: str
    name: str
    threat_level: ThreatLevel
    eta_min: Optional[int] = None
    coverage_by_horizon: dict[str, float] = Field(default_factory=dict)
    deep_convective_by_horizon: dict[str, float] = Field(default_factory=dict)
    affected_area_pct: float = 0.0
    peak_coverage_pct: float = 0.0
    nearest_station: str = ""
    precipitation_mm: float = 0.0
    wind_gusts_kmh: float = 0.0
    pressure_drop_3h_hpa: float = 0.0


class GaugeHistoryPoint(BaseModel):
    timestamp: str
    water_level_ft: float


class RiverGaugeReading(BaseModel):
    station_id: str
    station_name: str
    river_name: str
    basin_id: str = "other"
    role: Literal["upstream", "downstream", "other"] = "other"
    lat: float
    lon: float
    water_level_ft: float
    alert_level_ft: Optional[float] = None
    minor_flood_ft: Optional[float] = None
    major_flood_ft: Optional[float] = None
    rainfall_24h_mm: Optional[float] = None
    reading_timestamp: str
    fetched_at: str
    rate_of_rise_ft_hr: float = 0.0
    trend: GaugeTrend = "STEADY"
    status: GaugeStatus = "NORMAL"
    history_24h: list[GaugeHistoryPoint] = Field(default_factory=list)


class UpstreamSurgeWarning(BaseModel):
    basin_id: str
    basin_name: str
    upstream_stations: list[str]
    downstream_stations: list[str]
    max_rate_of_rise_ft_hr: float
    drainage_lag_hours: int
    drainage_lag_window_hours: list[int]
    estimated_crest_window: str
    message: str


class BasinZoneCoverage(BaseModel):
    basin_id: str
    name: str
    eta_min: Optional[int] = None
    coverage_by_horizon: dict[str, float] = Field(default_factory=dict)
    deep_convective_by_horizon: dict[str, float] = Field(default_factory=dict)
    storm_coverage_pct: float = 0.0
    projected_60m_pct: float = 0.0
    deep_convective_pct: float = 0.0
    deep_convective_60m_pct: float = 0.0
    peak_coverage_pct: float = 0.0


class RiverBasinRisk(BaseModel):
    basin_id: str
    name: str
    risk_level: BasinRiskLevel
    eta_min: Optional[int] = None
    storm_coverage_pct: float = 0.0
    projected_60m_pct: float = 0.0
    deep_convective_pct: float = 0.0
    deep_convective_60m_pct: float = 0.0
    coverage_by_horizon: dict[str, float] = Field(default_factory=dict)
    peak_coverage_pct: float = 0.0
    rainfall_24h_mm: float = 0.0
    rainfall_source: str = "open_meteo"
    drainage_lag_hours: int = 12
    drainage_lag_window_hours: list[int] = Field(default_factory=lambda: [8, 16])
    upper_catchment_stations: list[str] = Field(default_factory=list)
    downstream_stations: list[str] = Field(default_factory=list)
    upstream_surge: bool = False
    gauge_summary: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    geometry: dict[str, Any] = Field(default_factory=dict)


class NowcastPayload(BaseModel):
    generated_at: str
    frame_timestamp: Optional[str] = None
    inference_backend: str
    inference_ms: float = 0.0
    storm_cells: list[ProjectedStormCell] = Field(default_factory=list)
    storm_cells_geojson: dict[str, Any] = Field(default_factory=lambda: {"type": "FeatureCollection", "features": []})
    projected_geojson: dict[str, dict[str, Any]] = Field(default_factory=dict)
    districts: list[DistrictThreat] = Field(default_factory=list)
    river_basins: list[RiverBasinRisk] = Field(default_factory=list)
    river_basins_geojson: dict[str, Any] = Field(default_factory=lambda: {"type": "FeatureCollection", "features": []})
    gauges: list[RiverGaugeReading] = Field(default_factory=list)
    upstream_surge_warning: list[UpstreamSurgeWarning] = Field(default_factory=list)
    hydrology_source_status: HydrologySourceStatus = "unavailable"
    hydrology_last_updated: Optional[str] = None
    surface_stations: list[SurfaceStationTelemetry] = Field(default_factory=list)
    advisory_text: str = ""
    advisory_source: str = "template"


class GaugesResponse(BaseModel):
    source_status: HydrologySourceStatus
    last_updated: Optional[str] = None
    error_detail: Optional[str] = None
    stations: list[RiverGaugeReading] = Field(default_factory=list)


class SystemStatusResponse(BaseModel):
    status: str = "ok"
    inference_backend: str
    hailo_available: bool
    model_hef_exists: bool
    last_inference_ms: float
    last_frame_timestamp: Optional[str] = None
    buffered_frames: int
    hydrology_source_status: HydrologySourceStatus
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    cpu_temp_c: Optional[float] = None
    uptime_seconds: float


class ActiveLearningPolygon(BaseModel):
    class_id: int
    label: str
    points: list[list[float]]  # Normalized [x, y] pairs in [0.0, 1.0]


class ActiveLearningSample(BaseModel):
    id: str
    frame_timestamp: str
    captured_at: str
    trigger_type: ActiveLearningTrigger
    trigger_detail: str
    status: ActiveLearningStatus
    model_filename: str
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    detection_count: int = 0
    size_bytes: int = 0
    reviewed_at: Optional[str] = None
    frame_url: str
    label_txt: str = ""
    polygons: list[ActiveLearningPolygon] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class ActiveLearningStats(BaseModel):
    pending: int = 0
    approved: int = 0
    rejected: int = 0
    total: int = 0
    storage_used_mb: float = 0.0
    max_storage_mb: float = 2048.0
    storage_pct: float = 0.0
    captures_today: int = 0
    max_daily_captures: int = 48


class ActiveLearningQueueResponse(BaseModel):
    status_filter: str
    count: int
    stats: ActiveLearningStats
    samples: list[ActiveLearningSample] = Field(default_factory=list)


class ActiveLearningDecisionRequest(BaseModel):
    status: Literal["approved", "rejected"]
    updated_label_txt: Optional[str] = None


class ActiveLearningDecisionResponse(BaseModel):
    sample: ActiveLearningSample
    stats: ActiveLearningStats
