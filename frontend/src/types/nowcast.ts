export type StormClass = "deep_convective" | "storm_shield" | "cyclonic_band" | "monsoon_shield";
export type ThreatLevel = "LOW" | "MODERATE" | "HIGH" | "SEVERE";
export type BasinRiskLevel = "NORMAL" | "WATCH" | "WARNING" | "CRITICAL";
export type GaugeStatus = "NORMAL" | "ALERT" | "MINOR_FLOOD" | "MAJOR_FLOOD";
export type GaugeTrend = "RISING" | "STEADY" | "FALLING";
export type HydrologySourceStatus = "arcgis_live" | "fallback" | "stale" | "unavailable";
export type AdvisorySource = "ollama" | "template" | "template_fallback";
export type HazardType = "rainfall" | "thunderstorm" | "high_wind";
export type HazardMapMode = "combined" | "rainfall" | "thunderstorm" | "high_wind" | "threat";

export interface HazardProbabilities {
  rainfall_prob: number;
  thunderstorm_prob: number;
  high_wind_prob: number;
  max_hazard_prob: number;
  dominant_hazard: HazardType;
  wind_speed_kmh: number;
  wind_gusts_kmh: number;
  wind_direction_deg?: number;
  wind_compass_dir?: string;
  cape_jkg: number;
  weather_code: number;
  wave_height_m: number | null;
  exceeds_alert_threshold: boolean;
}

export interface FrameMetadata {
  timestamp: string;
  filename: string;
  image_url: string;
  width: number;
  height: number;
  mean_intensity: number;
  fetched_at: string;
}

export interface ProjectedStormCell {
  cell_id: string;
  label: StormClass;
  confidence: number;
  centroid_lon: number;
  centroid_lat: number;
  dx_kmh: number;
  dy_kmh: number;
  speed_kmh: number;
  bearing_deg: number;
  compass_dir: string;
  geometry_now: GeoJSON.Geometry;
  geometry_projected: Record<string, GeoJSON.Geometry>;
}

export interface SurfaceStationTelemetry {
  name: string;
  lat: number;
  lon: number;
  precipitation_mm: number;
  wind_gusts_kmh: number;
  surface_pressure_hpa: number;
  pressure_drop_3h_hpa: number;
  observed_at: string;
}

export interface DistrictThreat {
  district_id: string;
  name: string;
  threat_level: ThreatLevel;
  eta_min: number | null;
  coverage_by_horizon: Record<string, number>;
  deep_convective_by_horizon: Record<string, number>;
  affected_area_pct: number;
  peak_coverage_pct: number;
  nearest_station: string;
  precipitation_mm: number;
  wind_gusts_kmh: number;
  pressure_drop_3h_hpa: number;
  hazards?: HazardProbabilities | null;
}

export interface SeaZoneHazard {
  zone_id: string;
  name: string;
  zone_type: "sea_zone";
  centroid: [number, number];
  eta_min: number | null;
  storm_coverage_pct: number;
  projected_60m_pct: number;
  deep_convective_pct: number;
  deep_convective_60m_pct: number;
  peak_coverage_pct: number;
  hazards: HazardProbabilities;
  geometry: GeoJSON.Geometry;
}

export interface GaugeHistoryPoint {
  timestamp: string;
  water_level_ft: number;
}

export interface RiverGaugeReading {
  station_id: string;
  station_name: string;
  river_name: string;
  basin_id: string;
  role: "upstream" | "downstream" | "other";
  lat: number;
  lon: number;
  water_level_ft: number;
  alert_level_ft: number | null;
  minor_flood_ft: number | null;
  major_flood_ft: number | null;
  rainfall_24h_mm: number | null;
  reading_timestamp: string;
  fetched_at: string;
  rate_of_rise_ft_hr: number;
  trend: GaugeTrend;
  status: GaugeStatus;
  history_24h: GaugeHistoryPoint[];
}

export interface UpstreamSurgeWarning {
  basin_id: string;
  basin_name: string;
  upstream_stations: string[];
  downstream_stations: string[];
  max_rate_of_rise_ft_hr: number;
  drainage_lag_hours: number;
  drainage_lag_window_hours: number[];
  estimated_crest_window: string;
  message: string;
}

export interface RiverBasinRisk {
  basin_id: string;
  name: string;
  risk_level: BasinRiskLevel;
  eta_min: number | null;
  storm_coverage_pct: number;
  projected_60m_pct: number;
  deep_convective_pct: number;
  deep_convective_60m_pct: number;
  coverage_by_horizon: Record<string, number>;
  peak_coverage_pct: number;
  rainfall_24h_mm: number;
  rainfall_source: string;
  drainage_lag_hours: number;
  drainage_lag_window_hours: number[];
  upper_catchment_stations: string[];
  downstream_stations: string[];
  upstream_surge: boolean;
  gauge_summary: {
    total_gauges: number;
    highest_status: string;
    max_rate_of_rise_ft_hr: number;
    rising_count: number;
  };
  reasons: string[];
  geometry: GeoJSON.Geometry;
}

export interface NowcastPayload {
  generated_at: string;
  frame_timestamp: string | null;
  inference_backend: string;
  inference_ms: number;
  storm_cells: ProjectedStormCell[];
  storm_cells_geojson: GeoJSON.FeatureCollection;
  projected_geojson: Record<string, GeoJSON.FeatureCollection>;
  districts: DistrictThreat[];
  sea_zones: SeaZoneHazard[];
  sea_zones_geojson: GeoJSON.FeatureCollection;
  hourly_digest_threshold_pct: number;
  river_basins: RiverBasinRisk[];
  river_basins_geojson: GeoJSON.FeatureCollection;
  gauges: RiverGaugeReading[];
  upstream_surge_warning: UpstreamSurgeWarning[];
  hydrology_source_status: HydrologySourceStatus;
  hydrology_last_updated: string | null;
  surface_stations: SurfaceStationTelemetry[];
  advisory_text: string;
  advisory_source: AdvisorySource | string;
  advisory_model?: string | null;
  advisory_latency_ms?: number;
}

export interface SystemStatusResponse {
  status: string;
  inference_backend: string;
  hailo_available: boolean;
  model_hef_exists: boolean;
  model_hef_name?: string | null;
  model_hef_size_mb?: number | null;
  last_inference_ms: number;
  last_frame_timestamp: string | null;
  buffered_frames: number;
  hydrology_source_status: HydrologySourceStatus;
  cpu_percent: number;
  ram_used_gb: number;
  ram_total_gb: number;
  ram_percent: number;
  cpu_temp_c: number | null;
  uptime_seconds: number;
}

export interface LayerVisibility {
  satellite: boolean;
  districts: boolean;
  seaZones: boolean;
  floodBasins: boolean;
  riverGauges: boolean;
  stormMasks: boolean;
  projectedTrajectories: boolean;
  velocityArrows: boolean;
  surfaceStations: boolean;
}

export type ActiveLearningTrigger = "low_conf" | "sensor_mismatch" | "boundary_artifact";
export type ActiveLearningStatus = "pending" | "approved" | "rejected";

export interface ActiveLearningPolygon {
  class_id: number;
  label: string;
  points: [number, number][];
}

export interface ActiveLearningBasinGaugeSnapshot {
  station_id: string;
  station_name: string;
  water_level_ft: number;
  rate_of_rise_ft_hr: number;
  trend: GaugeTrend;
  status: GaugeStatus;
}

export interface ActiveLearningBasinSnapshot {
  basin_id: string;
  name: string;
  risk_level: BasinRiskLevel;
  deep_convective_pct: number;
  storm_coverage_pct: number;
  rainfall_1h_mm: number;
  rainfall_24h_mm: number;
  gauges: ActiveLearningBasinGaugeSnapshot[];
}

export interface ActiveLearningSample {
  id: string;
  frame_timestamp: string;
  captured_at: string;
  trigger_type: ActiveLearningTrigger;
  trigger_detail: string;
  status: ActiveLearningStatus;
  model_filename: string;
  min_confidence: number | null;
  max_confidence: number | null;
  detection_count: number;
  size_bytes: number;
  reviewed_at: string | null;
  frame_url: string;
  label_txt: string;
  polygons: ActiveLearningPolygon[];
  meta: {
    pseudo_label_source?: string;
    all_triggers?: ActiveLearningTrigger[];
    confidences?: number[];
    basins_snapshot?: Record<string, ActiveLearningBasinSnapshot>;
    mismatched_basins?: Array<{
      basin_id: string;
      basin_name: string;
      rainfall_1h_mm: number;
      heavy_rain: boolean;
      rising_gauges: Array<{
        station_name: string;
        water_level_ft: number;
        rate_of_rise_ft_hr: number;
        status: string;
      }>;
    }>;
    human_edited_label?: boolean;
    [key: string]: unknown;
  };
}

export interface ActiveLearningStats {
  pending: number;
  approved: number;
  rejected: number;
  total: number;
  storage_used_mb: number;
  max_storage_mb: number;
  storage_pct: number;
  captures_today: number;
  max_daily_captures: number;
}

export interface ActiveLearningQueueResponse {
  status_filter: string;
  count: number;
  stats: ActiveLearningStats;
  samples: ActiveLearningSample[];
}

export interface ActiveLearningDecisionResponse {
  sample: ActiveLearningSample;
  stats: ActiveLearningStats;
}

export interface AlertStatusResponse {
  telegram_enabled: boolean;
  telegram_configured: boolean;
  telegram_ready: boolean;
  cooldown_min: number;
  hourly_summary_enabled: boolean;
  hourly_summary_interval_min: number;
  chat_id_masked: string | null;
  last_dispatched_at: string | null;
  last_status: string | null;
  total_logged: number;
}

export interface AlertHistoryItem {
  id: number;
  entity_key: string;
  entity_type: string;
  zone_name: string;
  level: string;
  channel: string;
  status: string;
  caption_preview: string;
  error: string | null;
  sent_at: string;
}

export interface AlertHistoryResponse {
  count: number;
  alerts: AlertHistoryItem[];
}

export interface TelegramTestResponse {
  ok: boolean;
  status: string;
  detail: string;
  sent_at: string | null;
  snapshot_bytes: number;
}

export interface HourlyDigestTestResponse {
  ok: boolean;
  status: string;
  detail: string;
  threshold_pct: number;
  qualifying_districts: Array<{
    zone_id: string;
    name: string;
    rainfall_prob: number;
    thunderstorm_prob: number;
    high_wind_prob: number;
    max_hazard_prob: number;
    dominant_hazard: HazardType;
    wind_gusts_kmh: number;
  }>;
  qualifying_sea_zones: Array<{
    zone_id: string;
    name: string;
    rainfall_prob: number;
    thunderstorm_prob: number;
    high_wind_prob: number;
    max_hazard_prob: number;
    dominant_hazard: HazardType;
    wind_gusts_kmh: number;
    wave_height_m: number | null;
  }>;
  caption_preview: string;
  sent_at: string | null;
  snapshot_bytes: number;
}
