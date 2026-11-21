from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon, shape

from active_learning.pseudo_label import detections_to_yolo_seg, parse_yolo_seg
from config import Settings
from inference.cpu_detector import CPUIRStormDetector
from models import (
    ActiveLearningSample,
    ActiveLearningStats,
    ActiveLearningTrigger,
    RiverBasinRisk,
    RiverGaugeReading,
    StormDetection,
)
from storage.db import Database

logger = logging.getLogger(__name__)


@dataclass
class TriggerEvaluation:
    should_capture: bool = False
    primary_trigger: Optional[ActiveLearningTrigger] = None
    trigger_detail: str = ""
    all_triggers: list[ActiveLearningTrigger] = field(default_factory=list)
    low_conf_cells: list[dict[str, Any]] = field(default_factory=list)
    mismatched_basins: list[dict[str, Any]] = field(default_factory=list)
    boundary_cells: list[dict[str, Any]] = field(default_factory=list)


def _safe_timestamp_slug(ts: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "", ts)
    return cleaned[:20] or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


class ActiveLearningSampler:
    """
    Automatic active-learning capture engine for the Raspberry Pi 5 + Hailo-8
    Sri Lanka weather nowcasting pipeline.

    Hooks in after StormDetector and BasinFloodAnalyzer to capture:
      1. Low-Confidence Uncertainty (`low_conf`)
      2. Cross-Sensor Missed Detection (`sensor_mismatch`)
      3. Boundary / Edge-Case Artifacts (`boundary_artifact`, optional)

    Enforces daily capture caps and SD-card storage quotas with automatic pruning.
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        cpu_detector: Optional[CPUIRStormDetector] = None,
    ):
        self.settings = settings
        self.db = db
        self.cpu_detector = cpu_detector or CPUIRStormDetector(
            confidence_threshold=min(0.25, settings.al_conf_min)
        )
        self.root_dir = settings.active_learning_path
        self.pending_dir = self.root_dir / "pending"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self._basin_shapes, self._basin_names = self._load_basins(settings.basins_geojson_path)

    @staticmethod
    def _load_basins(geojson_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
        shapes: dict[str, Any] = {}
        names: dict[str, str] = {}
        if not geojson_path.exists():
            return shapes, names
        try:
            data = json.loads(geojson_path.read_text(encoding="utf-8"))
            for feat in data.get("features", []):
                props = feat.get("properties") or {}
                bid = props.get("basin_id")
                if not bid:
                    continue
                geom = shape(feat["geometry"])
                if not geom.is_valid:
                    geom = geom.buffer(0)
                shapes[bid] = geom
                names[bid] = str(props.get("name") or bid)
        except Exception as exc:
            logger.warning("Failed to load basins GeoJSON for ActiveLearningSampler: %s", exc)
        return shapes, names

    def _basin_has_deep_convective(
        self,
        basin_id: str,
        detections: list[StormDetection],
        basin_risks_by_id: dict[str, RiverBasinRisk],
    ) -> bool:
        # 1. Check BasinFloodAnalyzer / NowcastEngine current deep_convective coverage
        risk = basin_risks_by_id.get(basin_id)
        if risk is not None and risk.deep_convective_pct > 0.0:
            return True

        # 2. Cross-check raw deep_convective polygons directly against basin GeoJSON
        basin_geom = self._basin_shapes.get(basin_id)
        if basin_geom is None:
            return False

        for det in detections:
            if det.label != "deep_convective" or len(det.geo_polygon) < 3:
                continue
            try:
                poly = Polygon(det.geo_polygon)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty and basin_geom.intersects(poly):
                    return True
            except Exception:
                continue
        return False

    def _touches_image_edge(
        self,
        det: StormDetection,
        width: int = 640,
        height: int = 640,
    ) -> bool:
        margin = max(1, int(self.settings.al_boundary_margin_px))
        if det.pixel_polygon:
            for pt in det.pixel_polygon:
                if len(pt) < 2:
                    continue
                x, y = float(pt[0]), float(pt[1])
                if x <= margin or x >= (width - 1 - margin) or y <= margin or y >= (height - 1 - margin):
                    return True
            return False

        if det.normalized_polygon:
            mx = margin / float(max(1, width))
            my = margin / float(max(1, height))
            for pt in det.normalized_polygon:
                if len(pt) < 2:
                    continue
                x, y = float(pt[0]), float(pt[1])
                if x <= mx or x >= (1.0 - mx) or y <= my or y >= (1.0 - my):
                    return True
        return False

    def evaluate_triggers(
        self,
        *,
        detections: list[StormDetection],
        basin_risks: Optional[list[RiverBasinRisk]] = None,
        basin_rainfall_1h_mm: Optional[dict[str, float]] = None,
        gauges: Optional[list[RiverGaugeReading]] = None,
        frame_shape: tuple[int, int] = (640, 640),
    ) -> TriggerEvaluation:
        """Evaluates all active-learning capture conditions without performing disk I/O."""
        ev = TriggerEvaluation()
        h, w = frame_shape
        conf_min = float(self.settings.al_conf_min)
        conf_max = float(self.settings.al_conf_max)
        heavy_rain_thresh = float(self.settings.al_heavy_rain_mm_hr)
        gauge_rise_thresh = float(self.settings.al_gauge_rise_ft_hr)

        # 1. Low-Confidence Uncertainty (`low_conf`)
        for det in detections:
            if conf_min <= det.confidence <= conf_max:
                ev.low_conf_cells.append(
                    {
                        "cell_id": det.cell_id,
                        "label": det.label,
                        "confidence": round(float(det.confidence), 3),
                    }
                )

        # 2. Cross-Sensor Missed Detection (`sensor_mismatch`)
        risks_by_id = {b.basin_id: b for b in (basin_risks or [])}
        rain_1h_map = basin_rainfall_1h_mm or {}
        gauges_list = gauges or []

        candidate_basins = set(self._basin_shapes.keys()) | set(risks_by_id.keys()) | set(rain_1h_map.keys())
        for g in gauges_list:
            if g.basin_id and g.basin_id != "other":
                candidate_basins.add(g.basin_id)

        for bid in sorted(candidate_basins):
            rain_1h = float(rain_1h_map.get(bid, 0.0))
            basin_gauges = [g for g in gauges_list if g.basin_id == bid]
            rising_gauges = [
                g
                for g in basin_gauges
                if g.trend == "RISING" and float(g.rate_of_rise_ft_hr) >= gauge_rise_thresh
            ]
            heavy_rain = rain_1h >= heavy_rain_thresh

            if not (heavy_rain or rising_gauges):
                continue

            if not self._basin_has_deep_convective(bid, detections, risks_by_id):
                basin_name = (
                    risks_by_id[bid].name
                    if bid in risks_by_id
                    else self._basin_names.get(bid, bid.title())
                )
                ev.mismatched_basins.append(
                    {
                        "basin_id": bid,
                        "basin_name": basin_name,
                        "rainfall_1h_mm": round(rain_1h, 2),
                        "heavy_rain": heavy_rain,
                        "rising_gauges": [
                            {
                                "station_name": g.station_name,
                                "water_level_ft": g.water_level_ft,
                                "rate_of_rise_ft_hr": g.rate_of_rise_ft_hr,
                                "status": g.status,
                            }
                            for g in rising_gauges
                        ],
                    }
                )

        # 3. Optional Boundary / Edge-Case Capture (`boundary_artifact`)
        if self.settings.al_capture_boundary_artifacts:
            for det in detections:
                if det.confidence <= conf_max and self._touches_image_edge(det, width=w, height=h):
                    ev.boundary_cells.append(
                        {
                            "cell_id": det.cell_id,
                            "label": det.label,
                            "confidence": round(float(det.confidence), 3),
                        }
                    )

        # Assemble triggers in priority order: sensor_mismatch > low_conf > boundary_artifact
        if ev.mismatched_basins:
            ev.all_triggers.append("sensor_mismatch")
        if ev.low_conf_cells:
            ev.all_triggers.append("low_conf")
        if ev.boundary_cells:
            ev.all_triggers.append("boundary_artifact")

        if not ev.all_triggers:
            return ev

        ev.should_capture = True
        ev.primary_trigger = ev.all_triggers[0]

        if ev.primary_trigger == "sensor_mismatch":
            names = ", ".join(b["basin_name"] for b in ev.mismatched_basins[:2])
            ev.trigger_detail = f"Sensor Mismatch: {names}"
        elif ev.primary_trigger == "low_conf":
            min_c = min(c["confidence"] for c in ev.low_conf_cells)
            ev.trigger_detail = f"Low Conf: {min_c:.2f}"
        else:
            min_c = min(c["confidence"] for c in ev.boundary_cells)
            ev.trigger_detail = f"Boundary Artifact: {min_c:.2f}"

        return ev

    def _build_basin_snapshot(
        self,
        basin_risks: Optional[list[RiverBasinRisk]],
        basin_rainfall_1h_mm: Optional[dict[str, float]],
        basin_rainfall_24h_mm: Optional[dict[str, float]],
        gauges: Optional[list[RiverGaugeReading]],
    ) -> dict[str, Any]:
        rain_1h = basin_rainfall_1h_mm or {}
        rain_24h = basin_rainfall_24h_mm or {}
        gauges_list = gauges or []
        risks_by_id = {b.basin_id: b for b in (basin_risks or [])}
        all_bids = sorted(set(self._basin_names.keys()) | set(risks_by_id.keys()) | set(rain_1h.keys()))

        basins_out: dict[str, Any] = {}
        for bid in all_bids:
            risk = risks_by_id.get(bid)
            b_gauges = [g for g in gauges_list if g.basin_id == bid]
            basins_out[bid] = {
                "basin_id": bid,
                "name": risk.name if risk else self._basin_names.get(bid, bid.title()),
                "risk_level": risk.risk_level if risk else "NORMAL",
                "deep_convective_pct": risk.deep_convective_pct if risk else 0.0,
                "storm_coverage_pct": risk.storm_coverage_pct if risk else 0.0,
                "rainfall_1h_mm": round(float(rain_1h.get(bid, 0.0)), 2),
                "rainfall_24h_mm": round(
                    float(risk.rainfall_24h_mm if risk else rain_24h.get(bid, 0.0)), 1
                ),
                "gauges": [
                    {
                        "station_id": g.station_id,
                        "station_name": g.station_name,
                        "water_level_ft": g.water_level_ft,
                        "rate_of_rise_ft_hr": g.rate_of_rise_ft_hr,
                        "trend": g.trend,
                        "status": g.status,
                    }
                    for g in b_gauges
                ],
            }
        return basins_out

    def _active_model_filename(self, backend_name: str) -> str:
        hef_path = Path(self.settings.model_hef_path)
        if backend_name == "hailo-8" or hef_path.exists():
            return hef_path.name
        return f"{backend_name} ({hef_path.name} unavailable)"

    def maybe_capture(
        self,
        *,
        frame_bgr: np.ndarray,
        frame_timestamp: str,
        detections: list[StormDetection],
        backend_name: str = "cpu-fallback",
        basin_risks: Optional[list[RiverBasinRisk]] = None,
        basin_rainfall_1h_mm: Optional[dict[str, float]] = None,
        basin_rainfall_24h_mm: Optional[dict[str, float]] = None,
        gauges: Optional[list[RiverGaugeReading]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        now_utc: Optional[datetime] = None,
    ) -> Optional[ActiveLearningSample]:
        """
        Evaluates capture triggers and, if triggered and within daily cap, writes:
          - frame.png (raw unannotated 640x640 IR frame)
          - label.txt (YOLOv8-seg normalized pseudo-labels)
          - meta.json (trigger reason, confidences, model filename, hydrology snapshot)
        Records the sample in SQLite and enforces disk quota pruning.
        """
        if not self.settings.al_enabled:
            return None

        if self.db.has_al_frame_timestamp(frame_timestamp):
            logger.debug("Active learning: frame %s already captured; skipping", frame_timestamp)
            return None

        h, w = frame_bgr.shape[:2]
        ev = self.evaluate_triggers(
            detections=detections,
            basin_risks=basin_risks,
            basin_rainfall_1h_mm=basin_rainfall_1h_mm,
            gauges=gauges,
            frame_shape=(h, w),
        )
        if not ev.should_capture or ev.primary_trigger is None:
            return None

        captured_dt = now_utc or datetime.now(timezone.utc)
        today_prefix = captured_dt.strftime("%Y-%m-%d")
        captures_today = self.db.count_al_captures_on_date(today_prefix)
        if captures_today >= self.settings.al_max_daily_captures:
            logger.info(
                "Active learning daily capture cap reached (%d/%d for %s); skipping frame %s",
                captures_today,
                self.settings.al_max_daily_captures,
                today_prefix,
                frame_timestamp,
            )
            return None

        # Normalize frame to 640x640 raw PNG
        raw_frame = frame_bgr
        if raw_frame.shape[0] != 640 or raw_frame.shape[1] != 640:
            raw_frame = cv2.resize(raw_frame, (640, 640), interpolation=cv2.INTER_AREA)

        # Generate YOLOv8-seg pseudo-labels
        pseudo_source = "model_raw"
        label_detections = detections
        if ev.primary_trigger == "sensor_mismatch":
            eff_bbox = bbox or self.settings.bbox_tuple
            cpu_res = self.cpu_detector.detect(
                raw_frame,
                bbox=eff_bbox,
                confidence_threshold=min(0.25, self.settings.al_conf_min),
            )
            if cpu_res.detections:
                label_detections = cpu_res.detections
                pseudo_source = "cpu_ir_fallback"
            else:
                pseudo_source = "cpu_ir_fallback_empty"

        label_txt = detections_to_yolo_seg(label_detections, width=640, height=640)
        polygons = parse_yolo_seg(label_txt)

        safe_ts = _safe_timestamp_slug(frame_timestamp)
        sample_id = f"{safe_ts}_{ev.primary_trigger}"
        sample_dir = self.pending_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        frame_path = sample_dir / "frame.png"
        label_path = sample_dir / "label.txt"
        meta_path = sample_dir / "meta.json"

        ok = cv2.imwrite(str(frame_path), raw_frame)
        if not ok:
            logger.error("Active learning failed to write raw frame to %s", frame_path)
            shutil.rmtree(sample_dir, ignore_errors=True)
            return None

        label_path.write_text(label_txt, encoding="utf-8")

        confidences = [round(float(d.confidence), 3) for d in detections]
        min_conf = min(confidences) if confidences else None
        max_conf = max(confidences) if confidences else None
        model_filename = self._active_model_filename(backend_name)
        captured_iso = captured_dt.isoformat()

        meta: dict[str, Any] = {
            "sample_id": sample_id,
            "frame_timestamp": frame_timestamp,
            "captured_at": captured_iso,
            "trigger_type": ev.primary_trigger,
            "trigger_detail": ev.trigger_detail,
            "all_triggers": ev.all_triggers,
            "status": "pending",
            "model_filename": model_filename,
            "inference_backend": backend_name,
            "pseudo_label_source": pseudo_source,
            "confidences": confidences,
            "min_confidence": min_conf,
            "max_confidence": max_conf,
            "detection_count": len(polygons),
            "low_conf_cells": ev.low_conf_cells,
            "mismatched_basins": ev.mismatched_basins,
            "boundary_cells": ev.boundary_cells,
            "basins_snapshot": self._build_basin_snapshot(
                basin_risks=basin_risks,
                basin_rainfall_1h_mm=basin_rainfall_1h_mm,
                basin_rainfall_24h_mm=basin_rainfall_24h_mm,
                gauges=gauges,
            ),
        }
        meta_json = json.dumps(meta, indent=2)
        meta_path.write_text(meta_json, encoding="utf-8")

        size_bytes = _dir_size_bytes(sample_dir)

        self.db.insert_al_sample(
            sample_id=sample_id,
            frame_timestamp=frame_timestamp,
            captured_at=captured_iso,
            trigger_type=ev.primary_trigger,
            trigger_detail=ev.trigger_detail,
            status="pending",
            model_filename=model_filename,
            min_confidence=min_conf,
            max_confidence=max_conf,
            detection_count=len(polygons),
            dir_path=str(sample_dir),
            size_bytes=size_bytes,
            meta_json=meta_json,
        )

        pruned = self.enforce_storage_quota()
        logger.info(
            "Active learning captured sample %s (%s, polygons=%d, size=%dB, pruned=%d)",
            sample_id,
            ev.trigger_detail,
            len(polygons),
            size_bytes,
            len(pruned),
        )

        return ActiveLearningSample(
            id=sample_id,
            frame_timestamp=frame_timestamp,
            captured_at=captured_iso,
            trigger_type=ev.primary_trigger,
            trigger_detail=ev.trigger_detail,
            status="pending",
            model_filename=model_filename,
            min_confidence=min_conf,
            max_confidence=max_conf,
            detection_count=len(polygons),
            size_bytes=size_bytes,
            reviewed_at=None,
            frame_url=f"/api/active-learning/{sample_id}/frame",
            label_txt=label_txt,
            polygons=polygons,
            meta=meta,
        )

    def enforce_storage_quota(self) -> list[str]:
        """
        Prunes oldest `rejected` samples first, then oldest `pending` samples, until total
        active-learning storage fits within `AL_MAX_STORAGE_MB`. Protects `approved` samples.
        """
        max_bytes = int(float(self.settings.al_max_storage_mb) * 1024.0 * 1024.0)
        if max_bytes <= 0:
            return []

        total_bytes = _dir_size_bytes(self.root_dir)
        if total_bytes <= max_bytes:
            return []

        pruned_ids: list[str] = []
        candidates = self.db.list_prunable_al_samples()
        for row in candidates:
            if total_bytes <= max_bytes:
                break
            sample_dir = Path(row["dir_path"])
            reclaimed = _dir_size_bytes(sample_dir) if sample_dir.exists() else int(row["size_bytes"] or 0)
            if sample_dir.exists():
                shutil.rmtree(sample_dir, ignore_errors=True)
            self.db.delete_al_sample(row["id"])
            total_bytes = max(0, total_bytes - reclaimed)
            pruned_ids.append(row["id"])
            logger.info(
                "Active learning quota prune: removed %s (status=%s, reclaimed=%dB)",
                row["id"],
                row["status"],
                reclaimed,
            )

        if total_bytes > max_bytes:
            logger.warning(
                "Active learning storage (%dB) still exceeds quota (%dB) because remaining samples are approved.",
                total_bytes,
                max_bytes,
            )

        return pruned_ids

    def get_stats(self, now_utc: Optional[datetime] = None) -> ActiveLearningStats:
        counts = self.db.get_al_counts_and_bytes()
        disk_bytes = _dir_size_bytes(self.root_dir)
        used_bytes = max(disk_bytes, counts["total_bytes"]) if counts["pending"] + counts["approved"] + counts["rejected"] > 0 else disk_bytes
        used_mb = round(used_bytes / (1024.0 * 1024.0), 3)
        max_mb = float(self.settings.al_max_storage_mb)
        pct = round(min(100.0, (used_mb / max_mb) * 100.0), 1) if max_mb > 0 else 0.0
        today_prefix = (now_utc or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        captures_today = self.db.count_al_captures_on_date(today_prefix)

        return ActiveLearningStats(
            pending=counts["pending"],
            approved=counts["approved"],
            rejected=counts["rejected"],
            total=counts["pending"] + counts["approved"] + counts["rejected"],
            storage_used_mb=used_mb,
            max_storage_mb=max_mb,
            storage_pct=pct,
            captures_today=captures_today,
            max_daily_captures=self.settings.al_max_daily_captures,
        )

    def hydrate_sample(self, row: dict[str, Any]) -> ActiveLearningSample:
        sample_dir = Path(row["dir_path"])
        label_path = sample_dir / "label.txt"
        label_txt = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        polygons = parse_yolo_seg(label_txt)
        try:
            meta = json.loads(row["meta_json"]) if row.get("meta_json") else {}
        except Exception:
            meta = {}
        return ActiveLearningSample(
            id=row["id"],
            frame_timestamp=row["frame_timestamp"],
            captured_at=row["captured_at"],
            trigger_type=row["trigger_type"],
            trigger_detail=row["trigger_detail"],
            status=row["status"],
            model_filename=row["model_filename"],
            min_confidence=row["min_confidence"],
            max_confidence=row["max_confidence"],
            detection_count=len(polygons),
            size_bytes=int(row["size_bytes"] or 0),
            reviewed_at=row.get("reviewed_at"),
            frame_url=f"/api/active-learning/{row['id']}/frame",
            label_txt=label_txt,
            polygons=polygons,
            meta=meta,
        )
