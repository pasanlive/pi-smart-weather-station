from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_ROOT / ".env"), str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    data_dir: str = str(REPO_ROOT / "data")

    # Inference
    inference_backend: Literal["auto", "hailo", "cpu"] = "auto"
    model_hef_path: str = str(REPO_ROOT / "models" / "storm_yolov8s_seg.hef")
    confidence_threshold: float = 0.35

    # Himawari-9 Satellite Ingestion
    satellite_poll_interval_min: int = 10
    satellite_ring_buffer_size: int = 12
    satellite_backfill_frames: int = 6
    satellite_bbox: str = "75.0,3.0,86.0,12.0"

    # Open-Meteo
    open_meteo_poll_interval_min: int = 30

    # Sri Lanka Dept. of Irrigation ArcGIS Online REST Feature Services
    irrigation_arcgis_gauges_url: str = ""
    irrigation_arcgis_rainfall_url: str = ""
    irrigation_poll_interval_min: int = 15
    irrigation_fallback_enabled: bool = True
    irrigation_fallback_url: str = "https://waterlevels.lk/api/levels"

    # Optional Ollama local advisory summarizer
    ollama_url: str = ""
    ollama_model: str = "llama3.2:1b"

    # Optional Webhook Alerting
    alert_webhook_url: str = ""
    alert_cooldown_min: int = 30

    # Active-Learning Data Flywheel
    al_enabled: bool = True
    al_conf_min: float = 0.25
    al_conf_max: float = 0.55
    al_heavy_rain_mm_hr: float = 20.0
    al_gauge_rise_ft_hr: float = 0.3
    al_capture_boundary_artifacts: bool = True
    al_boundary_margin_px: int = 3
    al_max_daily_captures: int = 48
    al_max_storage_mb: float = 2048.0

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p

    @property
    def frames_path(self) -> Path:
        return self.data_path / "frames"

    @property
    def active_learning_path(self) -> Path:
        return self.data_path / "active_learning"

    @property
    def db_path(self) -> Path:
        return self.data_path / "nowcast.db"

    @property
    def bbox_tuple(self) -> tuple[float, float, float, float]:
        parts = [float(x.strip()) for x in self.satellite_bbox.split(",")]
        if len(parts) != 4:
            return (75.0, 3.0, 86.0, 12.0)
        return (parts[0], parts[1], parts[2], parts[3])

    @property
    def districts_geojson_path(self) -> Path:
        return BACKEND_DIR / "data" / "sri_lanka_districts.geojson"

    @property
    def basins_geojson_path(self) -> Path:
        return BACKEND_DIR / "data" / "sri_lanka_flood_basins.geojson"

    @property
    def frontend_dist_path(self) -> Path:
        return REPO_ROOT / "frontend" / "dist"


settings = Settings()
