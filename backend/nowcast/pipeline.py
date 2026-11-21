from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import psutil

from active_learning.sampler import ActiveLearningSampler
from config import Settings
from inference.cpu_detector import CPUIRStormDetector
from inference.factory import create_storm_detector
from inference.hailo_detector import is_hailo_available
from ingestion.himawari import HimawariPoller
from ingestion.irrigation_arcgis import IrrigationArcGISPoller
from ingestion.open_meteo import OpenMeteoPoller
from models import NowcastPayload, SystemStatusResponse
from nowcast.advisory import generate_advisory
from nowcast.alerting import AlertDispatcher
from nowcast.basin_flood import BasinFloodAnalyzer
from nowcast.nowcast_engine import NowcastEngine
from storage.db import Database

logger = logging.getLogger(__name__)


def _read_pi_cpu_temp_c() -> Optional[float]:
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal.exists():
        try:
            return round(int(thermal.read_text().strip()) / 1000.0, 1)
        except Exception:
            pass
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for entries in temps.values():
                if entries:
                    return round(float(entries[0].current), 1)
    except Exception:
        pass
    return None


class NowcastPipeline:
    """Coordinates data ingestion, Vision AI inference, optical-flow nowcasting, flood fusion, and alerts."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        broadcast_cb: Optional[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = None,
    ):
        self.settings = settings
        self.db = db
        self.broadcast_cb = broadcast_cb
        self.started_at = time.monotonic()

        self.himawari = HimawariPoller(settings, db)
        self.open_meteo = OpenMeteoPoller(settings)
        self.irrigation = IrrigationArcGISPoller(settings, db)
        self.detector = create_storm_detector(settings)
        self.nowcast_engine = NowcastEngine(
            districts_geojson_path=settings.districts_geojson_path,
            basins_geojson_path=settings.basins_geojson_path,
            bbox=settings.bbox_tuple,
            frame_size=(640, 640),
        )
        self.basin_analyzer = BasinFloodAnalyzer(settings.basins_geojson_path)
        self.alert_dispatcher = AlertDispatcher(settings, db)
        self.active_learning = ActiveLearningSampler(
            settings=settings,
            db=db,
            cpu_detector=CPUIRStormDetector(confidence_threshold=min(0.25, settings.al_conf_min)),
        )

        self._lock = asyncio.Lock()
        self.latest_payload: Optional[NowcastPayload] = None
        self.last_inference_ms: float = 0.0
        self.last_frame_timestamp: Optional[str] = None

        # Restore persisted snapshot immediately if present in SQLite
        cached = self.db.get_latest_nowcast()
        if cached:
            try:
                self.latest_payload = NowcastPayload.model_validate(cached)
                self.last_inference_ms = self.latest_payload.inference_ms
                self.last_frame_timestamp = self.latest_payload.frame_timestamp
            except Exception:
                pass

    async def startup_bootstrap(self) -> NowcastPayload:
        """Runs initial 6-frame Himawari backfill, Open-Meteo poll, Irrigation ArcGIS poll, and first nowcast."""
        logger.info("Starting edge nowcast bootstrap (backfill + Open-Meteo + Irrigation ArcGIS)...")
        await asyncio.gather(
            self.himawari.backfill(self.settings.satellite_backfill_frames),
            self.open_meteo.poll(),
            self.irrigation.poll(),
            return_exceptions=True,
        )
        return await self.run_nowcast()

    async def poll_satellite_and_nowcast(self) -> NowcastPayload:
        await self.himawari.poll_latest()
        return await self.run_nowcast()

    async def poll_open_meteo(self) -> None:
        await self.open_meteo.poll()
        if self.latest_payload is not None:
            await self.run_nowcast()

    async def poll_irrigation(self) -> None:
        await self.irrigation.poll()
        if self.latest_payload is not None:
            await self.run_nowcast()

    async def refresh_all(self) -> NowcastPayload:
        await asyncio.gather(
            self.himawari.poll_latest(),
            self.open_meteo.poll(),
            self.irrigation.poll(),
            return_exceptions=True,
        )
        # Ensure we have at least 3 frames if the buffer was empty
        if len(self.himawari.get_buffered_frames()) < 3:
            await self.himawari.backfill(self.settings.satellite_backfill_frames)
        return await self.run_nowcast()

    async def run_nowcast(self) -> NowcastPayload:
        async with self._lock:
            recent_frames = self.himawari.load_recent_bgr_frames(count=3)
            gauges_resp = self.irrigation.get_gauges_response()
            stations = self.open_meteo.stations_cache
            om_basin_rain = self.open_meteo.basin_rainfall_24h_mm

            detections = []
            raw_detections = []
            latest_bgr = None
            inference_ms = 0.0
            frame_ts: Optional[str] = None
            backend_name = self.detector.backend_name

            if recent_frames:
                frame_ts, latest_bgr = recent_frames[-1]
                # When active learning is enabled, run detection down to min(conf_threshold, AL_CONF_MIN)
                # so low-confidence uncertainty cells [0.25..0.35) reach the sampler while operational
                # nowcasting still filters at settings.confidence_threshold.
                detect_thresh = (
                    min(self.settings.confidence_threshold, self.settings.al_conf_min)
                    if self.settings.al_enabled
                    else self.settings.confidence_threshold
                )
                try:
                    det_result = await asyncio.to_thread(
                        self.detector.detect,
                        latest_bgr,
                        self.settings.bbox_tuple,
                        detect_thresh,
                    )
                except TypeError:
                    det_result = await asyncio.to_thread(
                        self.detector.detect,
                        latest_bgr,
                        self.settings.bbox_tuple,
                    )
                raw_detections = det_result.detections
                detections = [
                    d for d in raw_detections if d.confidence >= self.settings.confidence_threshold
                ]
                inference_ms = det_result.inference_ms
                backend_name = det_result.backend

            engine_out = await asyncio.to_thread(
                self.nowcast_engine.run,
                recent_frames,
                detections,
                stations,
            )

            basins_list, basins_geojson, surge_warnings = self.basin_analyzer.analyze(
                basin_coverages=engine_out.basin_coverages,
                open_meteo_rainfall_24h=om_basin_rain,
                gauges=gauges_resp.stations,
            )

            # Active-learning capture hook (non-fatal so SD-card I/O never interrupts nowcasting)
            if latest_bgr is not None and frame_ts is not None:
                try:
                    await asyncio.to_thread(
                        self.active_learning.maybe_capture,
                        frame_bgr=latest_bgr,
                        frame_timestamp=frame_ts,
                        detections=raw_detections,
                        backend_name=backend_name,
                        basin_risks=basins_list,
                        basin_rainfall_1h_mm=self.open_meteo.basin_rainfall_1h_mm,
                        basin_rainfall_24h_mm=om_basin_rain,
                        gauges=gauges_resp.stations,
                        bbox=self.settings.bbox_tuple,
                    )
                except Exception as al_exc:
                    logger.warning("Active learning capture hook failed (non-fatal): %s", al_exc)

            advisory_text, advisory_source = await generate_advisory(
                settings=self.settings,
                storm_cells=engine_out.storm_cells,
                districts=engine_out.districts,
                basins=basins_list,
                surges=surge_warnings,
            )

            await self.alert_dispatcher.dispatch_if_needed(
                districts=engine_out.districts,
                basins=basins_list,
            )

            payload = NowcastPayload(
                generated_at=datetime.now(timezone.utc).isoformat(),
                frame_timestamp=frame_ts,
                inference_backend=backend_name,
                inference_ms=inference_ms,
                storm_cells=engine_out.storm_cells,
                storm_cells_geojson=engine_out.storm_cells_geojson,
                projected_geojson=engine_out.projected_geojson,
                districts=engine_out.districts,
                river_basins=basins_list,
                river_basins_geojson=basins_geojson,
                gauges=gauges_resp.stations,
                upstream_surge_warning=surge_warnings,
                hydrology_source_status=gauges_resp.source_status,
                hydrology_last_updated=gauges_resp.last_updated,
                surface_stations=stations,
                advisory_text=advisory_text,
                advisory_source=advisory_source,
            )

            self.latest_payload = payload
            self.last_inference_ms = inference_ms
            self.last_frame_timestamp = frame_ts
            self.db.save_nowcast_run(
                frame_timestamp=frame_ts,
                backend_used=backend_name,
                inference_ms=inference_ms,
                advisory_text=advisory_text,
                payload_dict=payload.model_dump(),
            )

            if self.broadcast_cb is not None:
                await self.broadcast_cb({"type": "nowcast_update", "payload": payload.model_dump()})

            logger.info(
                "Nowcast complete: frame=%s backend=%s latency=%.1fms cells=%d surges=%d hydrology=%s",
                frame_ts,
                backend_name,
                inference_ms,
                len(payload.storm_cells),
                len(payload.upstream_surge_warning),
                payload.hydrology_source_status,
            )
            return payload

    def get_status(self) -> SystemStatusResponse:
        vm = psutil.virtual_memory()
        frames = self.himawari.get_buffered_frames()
        gauges_resp = self.irrigation.get_gauges_response()
        return SystemStatusResponse(
            status="ok",
            inference_backend=self.detector.backend_name,
            hailo_available=is_hailo_available(),
            model_hef_exists=Path(self.settings.model_hef_path).exists(),
            last_inference_ms=self.last_inference_ms,
            last_frame_timestamp=self.last_frame_timestamp,
            buffered_frames=len(frames),
            hydrology_source_status=gauges_resp.source_status,
            cpu_percent=round(psutil.cpu_percent(interval=None), 1),
            ram_used_gb=round(vm.used / (1024**3), 2),
            ram_total_gb=round(vm.total / (1024**3), 2),
            ram_percent=round(vm.percent, 1),
            cpu_temp_c=_read_pi_cpu_temp_c(),
            uptime_seconds=round(time.monotonic() - self.started_at, 1),
        )
