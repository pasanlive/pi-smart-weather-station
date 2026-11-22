from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx
import numpy as np

from alerts.snapshot import render_alert_snapshot
from alerts.telegram_notifier import (
    TELEGRAM_CAPTION_LIMIT,
    TelegramAlertDispatcher,
    _engine_display_name,
    mask_chat_id,
)
from config import Settings
from models import DistrictThreat, ProjectedStormCell, SeaZoneHazard
from storage.db import Database

logger = logging.getLogger(__name__)
SLT_TZ = timezone(timedelta(hours=5, minutes=30))


def filter_qualifying_zones(
    districts: Sequence[DistrictThreat],
    sea_zones: Sequence[SeaZoneHazard],
    threshold_pct: int = 50,
) -> tuple[list[DistrictThreat], list[SeaZoneHazard]]:
    """Returns districts and sea zones where any hazard probability strictly exceeds `threshold_pct`."""
    qual_districts = [
        d
        for d in districts
        if d.hazards is not None
        and (
            d.hazards.rainfall_prob > threshold_pct
            or d.hazards.thunderstorm_prob > threshold_pct
            or d.hazards.high_wind_prob > threshold_pct
        )
    ]
    qual_districts.sort(key=lambda d: (-d.hazards.max_hazard_prob if d.hazards else 0, d.name))

    qual_sea = [
        s
        for s in sea_zones
        if (
            s.hazards.rainfall_prob > threshold_pct
            or s.hazards.thunderstorm_prob > threshold_pct
            or s.hazards.high_wind_prob > threshold_pct
        )
    ]
    qual_sea.sort(key=lambda s: (-s.hazards.max_hazard_prob, s.name))
    return qual_districts, qual_sea


def _prob_dot(prob: int) -> str:
    return "🔴" if prob > 75 else "🟠"


def _format_hazard_tags(rain: int, ts: int, wind: int, threshold: int) -> str:
    parts: list[str] = []
    if rain > threshold:
        parts.append(f"🌧 {rain}%")
    if ts > threshold:
        parts.append(f"⛈ {ts}%")
    if wind > threshold:
        parts.append(f"💨 {wind}%")
    return " · ".join(parts)


def format_hourly_digest_messages(
    *,
    qualifying_districts: Sequence[DistrictThreat],
    qualifying_sea_zones: Sequence[SeaZoneHazard],
    threshold_pct: int = 50,
    timestamp_str: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Formats the grouped Land/Sea >50% Telegram HTML digest.
    Returns `(photo_caption, overflow_reply_text)`:
      - If the full message fits in Telegram's 1,024-char photo caption limit, `overflow_reply_text` is None.
      - Otherwise, returns a concise `photo_caption` (<= 1,024 chars) and the full grouped breakdown
        as `overflow_reply_text` for `reply_to_message_id` `sendMessage`.
    """
    now_utc = datetime.now(timezone.utc)
    if timestamp_str:
        try:
            raw = timestamp_str[:-1] + "+00:00" if timestamp_str.endswith("Z") else timestamp_str
            ref_dt = datetime.fromisoformat(raw).astimezone(timezone.utc)
        except ValueError:
            ref_dt = now_utc
    else:
        ref_dt = now_utc

    time_lbl = f"{ref_dt.astimezone(SLT_TZ).strftime('%H:%M SLT')} ({ref_dt.strftime('%H:%M UTC')})"
    header = (
        f"⚠️ <b>SRI LANKA HOURLY HAZARD DIGEST (&gt;{threshold_pct}%)</b>\n"
        f"🕒 {html.escape(time_lbl, quote=False)}  |  "
        f"<b>{len(qualifying_districts)}</b> Districts, <b>{len(qualifying_sea_zones)}</b> Sea Areas"
    )

    body_lines: list[str] = []
    if qualifying_districts:
        body_lines.append("\n🏝️ <b>DISTRICTS</b>")
        for d in qualifying_districts:
            hz = d.hazards
            assert hz is not None
            tags = _format_hazard_tags(hz.rainfall_prob, hz.thunderstorm_prob, hz.high_wind_prob, threshold_pct)
            body_lines.append(
                f"  {_prob_dot(hz.max_hazard_prob)} <b>{html.escape(d.name, quote=False)}</b>: "
                f"{tags} (gusts {hz.wind_gusts_kmh:.0f} km/h)"
            )

    if qualifying_sea_zones:
        body_lines.append("\n🌊 <b>SEA AREAS</b>")
        for s in qualifying_sea_zones:
            hz = s.hazards
            tags = _format_hazard_tags(hz.rainfall_prob, hz.thunderstorm_prob, hz.high_wind_prob, threshold_pct)
            wave_part = f", wave {hz.wave_height_m:.1f}m" if hz.wave_height_m is not None else ""
            body_lines.append(
                f"  {_prob_dot(hz.max_hazard_prob)} <b>{html.escape(s.name, quote=False)}</b>: "
                f"{tags} (gusts {hz.wind_gusts_kmh:.0f} km/h{wave_part})"
            )

    full_text = header + "\n" + "\n".join(body_lines)
    if len(full_text) <= TELEGRAM_CAPTION_LIMIT:
        return full_text, None

    # Overflow split: concise photo caption + full grouped list as reply_to_message_id
    top_names = [d.name for d in qualifying_districts[:4]] + [s.name for s in qualifying_sea_zones[:3]]
    short_caption = (
        f"{header}\n\n"
        f"🔥 <b>Peak zones:</b> {html.escape(', '.join(top_names), quote=False)}\n"
        f"👇 Full district &amp; sea-area probability breakdown in the reply below."
    )[:TELEGRAM_CAPTION_LIMIT]
    return short_caption, full_text


class HourlyDigestDispatcher:
    """
    Hourly >50% hazard probability digest dispatcher for Sri Lanka's 25 districts & 8 sea zones.
    Stays completely silent on calm hours (0 qualifying zones) and sends one grouped Telegram
    photo digest (with overflow reply if > 1,024 chars) whenever >= 1 zone exceeds the threshold.
    """

    def __init__(self, settings: Settings, db: Database, telegram: TelegramAlertDispatcher):
        self.settings = settings
        self.db = db
        self.telegram = telegram
        self._districts_raw: dict[str, Any] = self._load_geojson(settings.districts_geojson_path)

    @staticmethod
    def _load_geojson(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"type": "FeatureCollection", "features": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_highlighted_geojson(
        self,
        qualifying_districts: Sequence[DistrictThreat],
        qualifying_sea_zones: Sequence[SeaZoneHazard],
    ) -> dict[str, Any]:
        features: list[dict[str, Any]] = []
        dist_by_id = {d.district_id: d for d in qualifying_districts}
        for feat in self._districts_raw.get("features", []):
            did = str((feat.get("properties") or {}).get("district_id", ""))
            if did in dist_by_id and dist_by_id[did].hazards is not None:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "zone_id": did,
                            "max_hazard_prob": dist_by_id[did].hazards.max_hazard_prob,
                        },
                        "geometry": feat["geometry"],
                    }
                )
        for s in qualifying_sea_zones:
            if s.geometry:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "zone_id": s.zone_id,
                            "max_hazard_prob": s.hazards.max_hazard_prob,
                        },
                        "geometry": s.geometry,
                    }
                )
        return {"type": "FeatureCollection", "features": features}

    @staticmethod
    def _serialize_qualifying(
        qualifying_districts: Sequence[DistrictThreat],
        qualifying_sea_zones: Sequence[SeaZoneHazard],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        q_dist = [
            {
                "zone_id": d.district_id,
                "name": d.name,
                "rainfall_prob": d.hazards.rainfall_prob if d.hazards else 0,
                "thunderstorm_prob": d.hazards.thunderstorm_prob if d.hazards else 0,
                "high_wind_prob": d.hazards.high_wind_prob if d.hazards else 0,
                "max_hazard_prob": d.hazards.max_hazard_prob if d.hazards else 0,
                "dominant_hazard": d.hazards.dominant_hazard if d.hazards else "rainfall",
                "wind_gusts_kmh": d.hazards.wind_gusts_kmh if d.hazards else d.wind_gusts_kmh,
            }
            for d in qualifying_districts
        ]
        q_sea = [
            {
                "zone_id": s.zone_id,
                "name": s.name,
                "rainfall_prob": s.hazards.rainfall_prob,
                "thunderstorm_prob": s.hazards.thunderstorm_prob,
                "high_wind_prob": s.hazards.high_wind_prob,
                "max_hazard_prob": s.hazards.max_hazard_prob,
                "dominant_hazard": s.hazards.dominant_hazard,
                "wind_gusts_kmh": s.hazards.wind_gusts_kmh,
                "wave_height_m": s.hazards.wave_height_m,
            }
            for s in qualifying_sea_zones
        ]
        return q_dist, q_sea

    async def dispatch(
        self,
        *,
        frame_bgr: Optional[np.ndarray],
        frame_timestamp: Optional[str],
        backend_name: str,
        detections: Sequence[Any],
        storm_cells: Sequence[ProjectedStormCell],
        districts: Sequence[DistrictThreat],
        sea_zones: Sequence[SeaZoneHazard],
        basins_geojson: Optional[dict[str, Any]],
        ir_loop_frames: Optional[Sequence[np.ndarray]] = None,
        sea_zones_geojson: Optional[dict[str, Any]] = None,
        projected_geojson: Optional[dict[str, dict[str, Any]]] = None,
        threshold_pct: Optional[int] = None,
        send_to_telegram: bool = True,
        is_manual_test: bool = False,
        client: Optional[httpx.AsyncClient] = None,
        base_backoff_sec: float = 1.0,
    ) -> dict[str, Any]:
        thresh = self.settings.hourly_alert_prob_threshold if threshold_pct is None else int(threshold_pct)
        qual_dist, qual_sea = filter_qualifying_zones(districts, sea_zones, threshold_pct=thresh)
        q_dist_json, q_sea_json = self._serialize_qualifying(qual_dist, qual_sea)

        # Calm hours -> silent skip (no channel spam!)
        if not qual_dist and not qual_sea:
            logger.info("Hourly hazard digest skipped silently: 0 districts and 0 sea zones exceed %d%%", thresh)
            return {
                "ok": True,
                "status": "skipped_calm",
                "detail": f"No districts or sea zones currently exceed {thresh}% probability; digest skipped silently.",
                "threshold_pct": thresh,
                "qualifying_districts": [],
                "qualifying_sea_zones": [],
                "caption_preview": "",
                "sent_at": None,
                "snapshot_bytes": 0,
            }

        photo_caption, overflow_reply = format_hourly_digest_messages(
            qualifying_districts=qual_dist,
            qualifying_sea_zones=qual_sea,
            threshold_pct=thresh,
            timestamp_str=frame_timestamp,
        )
        full_preview = overflow_reply or photo_caption

        highlighted = self._build_highlighted_geojson(qual_dist, qual_sea)
        engine_label = _engine_display_name(backend_name)
        title_prefix = "[TEST] " if is_manual_test else ""
        jpeg_bytes = render_alert_snapshot(
            frame_bgr=frame_bgr,
            detections=detections,
            motions=storm_cells,
            basins_geojson=basins_geojson,
            ir_loop_frames=ir_loop_frames,
            districts=districts,
            sea_zones_geojson=sea_zones_geojson,
            projected_geojson=projected_geojson,
            highlighted_zones_geojson=highlighted,
            timestamp_str=frame_timestamp,
            bbox=self.settings.bbox_tuple,
            alert_title=f"{title_prefix}HOURLY HAZARD DIGEST (>{thresh}%)",
            engine_label=engine_label,
        )

        if not send_to_telegram:
            return {
                "ok": True,
                "status": "preview_only",
                "detail": f"{len(qual_dist)} district(s) and {len(qual_sea)} sea zone(s) exceed {thresh}%.",
                "threshold_pct": thresh,
                "qualifying_districts": q_dist_json,
                "qualifying_sea_zones": q_sea_json,
                "caption_preview": full_preview,
                "sent_at": None,
                "snapshot_bytes": len(jpeg_bytes),
            }

        if not self.telegram.is_ready:
            return {
                "ok": False,
                "status": "unconfigured",
                "detail": "Telegram bot token or chat ID not configured; preview generated without dispatch.",
                "threshold_pct": thresh,
                "qualifying_districts": q_dist_json,
                "qualifying_sea_zones": q_sea_json,
                "caption_preview": full_preview,
                "sent_at": None,
                "snapshot_bytes": len(jpeg_bytes),
            }

        now_utc = datetime.now(timezone.utc)
        ok, status, err, msg_id = await self.telegram.post_photo_returning_id(
            jpeg_bytes,
            photo_caption,
            client=client,
            base_backoff_sec=base_backoff_sec,
        )

        if ok and overflow_reply is not None:
            reply_ok, reply_status, reply_err = await self.telegram.post_reply_message(
                overflow_reply,
                reply_to_message_id=msg_id,
                client=client,
                base_backoff_sec=base_backoff_sec,
            )
            if not reply_ok:
                ok = False
                status = f"{status}_reply_{reply_status}"
                err = reply_err

        peak_prob = max(
            [d.hazards.max_hazard_prob for d in qual_dist if d.hazards]
            + [s.hazards.max_hazard_prob for s in qual_sea],
            default=0,
        )
        severity = "CRITICAL" if peak_prob > 75 else "WARNING"

        self.db.record_alert(
            entity_key="telegram:digest:hourly",
            entity_type="digest",
            level=severity,
            webhook_status=status,
            payload={
                "channel": "telegram",
                "zone_name": f"Hourly >{thresh}% Digest ({len(qual_dist)}D / {len(qual_sea)}S)",
                "reason": "manual_digest_test" if is_manual_test else "hourly_digest",
                "caption": full_preview,
                "snapshot_bytes": len(jpeg_bytes),
                "qualifying_districts": len(qual_dist),
                "qualifying_sea_zones": len(qual_sea),
                "overflow_reply_sent": overflow_reply is not None,
                "error": err,
            },
            sent_at=now_utc,
        )

        return {
            "ok": ok,
            "status": status,
            "detail": (
                f"Hourly >{thresh}% digest ({len(qual_dist)} districts, {len(qual_sea)} sea areas, "
                f"{len(jpeg_bytes) // 1024} KB) delivered to {mask_chat_id(self.settings.telegram_chat_id)}."
                if ok
                else f"Hourly digest delivery failed: {err or status}"
            ),
            "threshold_pct": thresh,
            "qualifying_districts": q_dist_json,
            "qualifying_sea_zones": q_sea_json,
            "caption_preview": full_preview,
            "sent_at": now_utc.isoformat(),
            "snapshot_bytes": len(jpeg_bytes),
        }
