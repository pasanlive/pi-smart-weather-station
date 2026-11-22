from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import psutil

from active_learning.sampler import ActiveLearningSampler
from alerts.hourly_digest import HourlyDigestDispatcher
from alerts.telegram_notifier import TelegramAlertDispatcher
from config import Settings
from inference.cpu_detector import CPUIRStormDetector
from inference.factory import create_storm_detector
from inference.hailo_detector import is_hailo_available
from ingestion.himawari import HimawariPoller
from ingestion.irrigation_arcgis import IrrigationArcGISPoller
from ingestion.open_meteo import OpenMeteoPoller
from models import NowcastPayload, SystemStatusResponse
from nowcast.advisory_generator import WeatherAdvisoryGenerator
from nowcast.alerting import AlertDispatcher
from nowcast.basin_flood import BasinFloodAnalyzer
from nowcast.hazard_probabilities import HazardProbabilityEngine
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
            sea_zones_geojson_path=settings.sea_zones_geojson_path,
            bbox=settings.bbox_tuple,
            frame_size=(640, 640),
        )
        self.hazard_engine = HazardProbabilityEngine(
            districts_geojson_path=settings.districts_geojson_path,
            sea_zones_geojson_path=settings.sea_zones_geojson_path,
            alert_threshold_pct=settings.hourly_alert_prob_threshold,
        )
        self.basin_analyzer = BasinFloodAnalyzer(settings.basins_geojson_path)
        self.advisory_generator = WeatherAdvisoryGenerator(settings)
        self.alert_dispatcher = AlertDispatcher(settings, db)
        self.telegram_dispatcher = TelegramAlertDispatcher(settings, db)
        self.hourly_digest = HourlyDigestDispatcher(settings, db, self.telegram_dispatcher)
        self.active_learning = ActiveLearningSampler(
            settings=settings,
            db=db,
            cpu_detector=CPUIRStormDetector(confidence_threshold=min(0.25, settings.al_conf_min)),
        )

        self._lock = asyncio.Lock()
        self._advisory_task: Optional[asyncio.Task[Any]] = None
        self._run_seq: int = 0
        self.latest_payload: Optional[NowcastPayload] = None
        self.last_inference_ms: float = 0.0
        self.last_frame_timestamp: Optional[str] = None
        self._last_detections: list[Any] = []

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
            self.hazard_engine.poll(),
            self.irrigation.poll(),
            return_exceptions=True,
        )
        return await self.run_nowcast()

    async def poll_satellite_and_nowcast(self) -> NowcastPayload:
        await self.himawari.poll_latest()
        return await self.run_nowcast()

    async def poll_open_meteo(self) -> None:
        await asyncio.gather(
            self.open_meteo.poll(),
            self.hazard_engine.poll(),
            return_exceptions=True,
        )
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
            self.hazard_engine.poll(),
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

            enriched_districts, sea_zones, sea_zones_geojson = self.hazard_engine.compute_hazard_probabilities(
                districts=engine_out.districts,
                sea_zone_coverages=engine_out.sea_zone_coverages,
                alert_threshold_pct=self.settings.hourly_alert_prob_threshold,
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

            # Instant zero-latency deterministic advisory so GET /api/nowcast/latest
            # and CRITICAL Telegram alerts are NEVER blocked waiting on local Ollama LLM.
            t_adv = time.perf_counter()
            advisory_text = self.advisory_generator.generate_template_advisory(
                storm_cells=engine_out.storm_cells,
                districts=engine_out.districts,
                basins=basins_list,
                gauges=gauges_resp.stations,
                surges=surge_warnings,
            )
            advisory_source = "template"
            advisory_latency_ms = round((time.perf_counter() - t_adv) * 1000.0, 2)

            await self.alert_dispatcher.dispatch_if_needed(
                districts=engine_out.districts,
                basins=basins_list,
            )

            try:
                await self.telegram_dispatcher.dispatch_nowcast_alerts(
                    frame_bgr=latest_bgr,
                    frame_timestamp=frame_ts,
                    backend_name=backend_name,
                    detections=detections,
                    storm_cells=engine_out.storm_cells,
                    districts=enriched_districts,
                    basins=basins_list,
                    basins_geojson=basins_geojson,
                    gauges=gauges_resp.stations,
                    advisory_text=advisory_text,
                    ir_loop_frames=[img for _, img in recent_frames],
                    sea_zones_geojson=sea_zones_geojson,
                    projected_geojson=engine_out.projected_geojson,
                )
            except Exception as tg_exc:
                logger.warning("Telegram alert dispatch hook failed (non-fatal): %s", tg_exc)

            self._last_detections = detections
            self._run_seq += 1
            current_run_seq = self._run_seq

            payload = NowcastPayload(
                generated_at=datetime.now(timezone.utc).isoformat(),
                frame_timestamp=frame_ts,
                inference_backend=backend_name,
                inference_ms=inference_ms,
                storm_cells=engine_out.storm_cells,
                storm_cells_geojson=engine_out.storm_cells_geojson,
                projected_geojson=engine_out.projected_geojson,
                districts=enriched_districts,
                sea_zones=sea_zones,
                sea_zones_geojson=sea_zones_geojson,
                hourly_digest_threshold_pct=self.settings.hourly_alert_prob_threshold,
                river_basins=basins_list,
                river_basins_geojson=basins_geojson,
                gauges=gauges_resp.stations,
                upstream_surge_warning=surge_warnings,
                hydrology_source_status=gauges_resp.source_status,
                hydrology_last_updated=gauges_resp.last_updated,
                surface_stations=stations,
                advisory_text=advisory_text,
                advisory_source=advisory_source,
                advisory_model=None,
                advisory_latency_ms=advisory_latency_ms,
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

            # Fire-and-forget local Ollama 3B refinement when severity >= OLLAMA_MIN_SEVERITY
            if self.advisory_generator.should_invoke_llm(engine_out.districts, basins_list, surge_warnings):
                if self._advisory_task is not None and not self._advisory_task.done():
                    self._advisory_task.cancel()
                self._advisory_task = asyncio.create_task(
                    self._refine_advisory_async(
                        run_seq=current_run_seq,
                        storm_cells=engine_out.storm_cells,
                        districts=engine_out.districts,
                        basins=basins_list,
                        gauges=gauges_resp.stations,
                        surges=surge_warnings,
                    )
                )

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

    async def _refine_advisory_async(
        self,
        *,
        run_seq: int,
        storm_cells: list[Any],
        districts: list[Any],
        basins: list[Any],
        gauges: list[Any],
        surges: list[Any],
    ) -> None:
        """Runs local Ollama 3B asynchronously and broadcasts the refined advisory if still current."""
        try:
            result = await self.advisory_generator.generate_advisory(
                storm_cells=storm_cells,
                districts=districts,
                basins=basins,
                gauges=gauges,
                surges=surges,
            )
            if run_seq != self._run_seq or self.latest_payload is None:
                return
            updated = self.latest_payload.model_copy(
                update={
                    "advisory_text": result.text,
                    "advisory_source": result.source,
                    "advisory_model": result.model,
                    "advisory_latency_ms": result.latency_ms,
                }
            )
            self.latest_payload = updated
            self.db.save_nowcast_run(
                frame_timestamp=updated.frame_timestamp,
                backend_used=updated.inference_backend,
                inference_ms=updated.inference_ms,
                advisory_text=updated.advisory_text,
                payload_dict=updated.model_dump(),
            )
            if self.broadcast_cb is not None:
                await self.broadcast_cb({"type": "nowcast_update", "payload": updated.model_dump()})
            logger.info(
                "Async advisory refined: source=%s model=%s latency=%.1fms",
                result.source,
                result.model,
                result.latency_ms,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("Background Ollama advisory task failed (non-fatal): %s", exc)

    async def send_telegram_test_alert(self) -> dict[str, Any]:
        """Renders a full-overlay snapshot from the latest buffered Himawari-9 IR loop and dispatches a [TEST ALERT]."""
        recent_frames = self.himawari.load_recent_bgr_frames(count=3)
        frame_ts: Optional[str] = None
        latest_bgr = None
        if recent_frames:
            frame_ts, latest_bgr = recent_frames[-1]

        payload = self.latest_payload
        storm_cells = payload.storm_cells if payload else []
        districts = payload.districts if payload else []
        basins_geojson = (
            payload.river_basins_geojson
            if payload and payload.river_basins_geojson.get("features")
            else self.basin_analyzer._raw_geojson
        )
        sea_zones_geojson = payload.sea_zones_geojson if payload else None
        projected_geojson = payload.projected_geojson if payload else None
        advisory_text = payload.advisory_text if payload else ""
        detections = self._last_detections if self._last_detections else storm_cells

        return await self.telegram_dispatcher.send_test_alert(
            frame_bgr=latest_bgr,
            frame_timestamp=frame_ts or (payload.frame_timestamp if payload else None),
            backend_name=self.detector.backend_name,
            detections=detections,
            storm_cells=storm_cells,
            basins_geojson=basins_geojson,
            advisory_text=advisory_text,
            ir_loop_frames=[img for _, img in recent_frames],
            districts=districts,
            sea_zones_geojson=sea_zones_geojson,
            projected_geojson=projected_geojson,
        )

    async def send_telegram_hourly_summary(self) -> dict[str, Any]:
        """Dispatches the scheduled or manual hourly Sri Lanka status summary photo to Telegram."""
        if self.latest_payload is None:
            await self.run_nowcast()

        recent_frames = self.himawari.load_recent_bgr_frames(count=3)
        frame_ts: Optional[str] = None
        latest_bgr = None
        if recent_frames:
            frame_ts, latest_bgr = recent_frames[-1]

        payload = self.latest_payload
        storm_cells = payload.storm_cells if payload else []
        districts = payload.districts if payload else []
        basins = payload.river_basins if payload else []
        surges = payload.upstream_surge_warning if payload else []
        basins_geojson = (
            payload.river_basins_geojson
            if payload and payload.river_basins_geojson.get("features")
            else self.basin_analyzer._raw_geojson
        )
        sea_zones_geojson = payload.sea_zones_geojson if payload else None
        projected_geojson = payload.projected_geojson if payload else None
        advisory_text = payload.advisory_text if payload else ""
        hydrology_status = payload.hydrology_source_status if payload else self.irrigation.get_gauges_response().source_status
        detections = self._last_detections if self._last_detections else storm_cells

        return await self.telegram_dispatcher.send_hourly_summary(
            frame_bgr=latest_bgr,
            frame_timestamp=frame_ts or (payload.frame_timestamp if payload else None),
            backend_name=self.detector.backend_name,
            inference_ms=self.last_inference_ms,
            detections=detections,
            storm_cells=storm_cells,
            districts=districts,
            basins=basins,
            basins_geojson=basins_geojson,
            surges=surges,
            hydrology_source_status=hydrology_status,
            advisory_text=advisory_text,
            ir_loop_frames=[img for _, img in recent_frames],
            sea_zones_geojson=sea_zones_geojson,
            projected_geojson=projected_geojson,
        )

    async def send_hourly_hazard_digest(
        self,
        *,
        send_to_telegram: bool = True,
        is_manual_test: bool = False,
    ) -> dict[str, Any]:
        """Dispatches (or previews) the hourly >50% district & sea-zone hazard probability digest."""
        if self.latest_payload is None:
            await self.run_nowcast()

        recent_frames = self.himawari.load_recent_bgr_frames(count=3)
        frame_ts: Optional[str] = None
        latest_bgr = None
        if recent_frames:
            frame_ts, latest_bgr = recent_frames[-1]

        payload = self.latest_payload
        storm_cells = payload.storm_cells if payload else []
        districts = payload.districts if payload else []
        sea_zones = payload.sea_zones if payload else []
        basins_geojson = (
            payload.river_basins_geojson
            if payload and payload.river_basins_geojson.get("features")
            else self.basin_analyzer._raw_geojson
        )
        sea_zones_geojson = payload.sea_zones_geojson if payload else None
        projected_geojson = payload.projected_geojson if payload else None
        detections = self._last_detections if self._last_detections else storm_cells

        return await self.hourly_digest.dispatch(
            frame_bgr=latest_bgr,
            frame_timestamp=frame_ts or (payload.frame_timestamp if payload else None),
            backend_name=self.detector.backend_name,
            detections=detections,
            storm_cells=storm_cells,
            districts=districts,
            sea_zones=sea_zones,
            basins_geojson=basins_geojson,
            ir_loop_frames=[img for _, img in recent_frames],
            sea_zones_geojson=sea_zones_geojson,
            projected_geojson=projected_geojson,
            threshold_pct=self.settings.hourly_alert_prob_threshold,
            send_to_telegram=send_to_telegram,
            is_manual_test=is_manual_test,
        )

    def get_status(self) -> SystemStatusResponse:
        vm = psutil.virtual_memory()
        frames = self.himawari.get_buffered_frames()
        gauges_resp = self.irrigation.get_gauges_response()
        hef_path = self.settings.model_hef_resolved_path
        hef_exists = hef_path.exists()
        hef_size_mb = round(hef_path.stat().st_size / (1024 * 1024), 2) if hef_exists else None
        return SystemStatusResponse(
            status="ok",
            inference_backend=self.detector.backend_name,
            hailo_available=is_hailo_available(),
            model_hef_exists=hef_exists,
            model_hef_name=hef_path.name if hef_exists else None,
            model_hef_size_mb=hef_size_mb,
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
