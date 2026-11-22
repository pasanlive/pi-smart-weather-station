from __future__ import annotations

import asyncio
import html
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import httpx
import numpy as np

from alerts.snapshot import render_alert_snapshot
from config import Settings
from models import (
    DistrictThreat,
    ProjectedStormCell,
    RiverBasinRisk,
    RiverGaugeReading,
    StormDetection,
    UpstreamSurgeWarning,
)
from storage.db import Database

logger = logging.getLogger(__name__)

TELEGRAM_CAPTION_LIMIT = 1024
TELEGRAM_API_BASE = "https://api.telegram.org"

SEVERITY_RANK: dict[str, int] = {
    "LOW": 1,
    "NORMAL": 1,
    "MODERATE": 2,
    "WATCH": 2,
    "WARNING": 3,
    "HIGH": 3,
    "CRITICAL": 4,
    "SEVERE": 4,
}


def _severity_emoji(level: str) -> str:
    lvl = level.upper()
    if lvl in ("CRITICAL", "SEVERE"):
        return "🔴"
    if lvl in ("WARNING", "HIGH"):
        return "🟠"
    if lvl == "TEST":
        return "🧪"
    return "🟡"


def _format_eta(eta_min: Optional[int]) -> str:
    if eta_min == 0:
        return "NOW"
    if eta_min is not None:
        return f"+{eta_min}m"
    return "—"


def _engine_display_name(backend_name: str) -> str:
    if "hailo" in backend_name.lower():
        return "Hailo-8"
    return "CPU Fallback"


def mask_chat_id(chat_id: str) -> Optional[str]:
    clean = chat_id.strip()
    if not clean:
        return None
    if len(clean) <= 5:
        return "*" * len(clean)
    return f"{clean[:3]}…{clean[-3:]}"


def build_safe_html_caption(
    *,
    header_prefix: str,
    severity: str,
    zone_name: str,
    eta_min: Optional[int],
    storm_coverage_pct: float,
    projected_60m_pct: Optional[float],
    rainfall_24h_mm: Optional[float],
    gauge_line: Optional[str],
    downstream_stations: Sequence[str],
    advisory_text: str,
) -> str:
    """
    Formats a Telegram `parse_mode="HTML"` photo caption escaping all dynamic values
    with `html.escape()` and strictly enforcing Telegram's 1,024-character limit.
    """
    emoji = _severity_emoji(severity)
    esc_prefix = html.escape(header_prefix, quote=False)
    esc_zone = html.escape(zone_name, quote=False)
    esc_sev = html.escape(severity.upper(), quote=False)
    esc_eta = html.escape(_format_eta(eta_min), quote=False)

    lines: list[str] = [
        f"{emoji} <b>{esc_prefix}: {esc_zone} ({esc_sev})</b>",
        f"⏱ <b>ETA:</b> {esc_eta}  |  <b>Storm Cover:</b> {storm_coverage_pct:.0f}%"
        + (f" (+60m: {projected_60m_pct:.0f}%)" if projected_60m_pct is not None else ""),
    ]

    if rainfall_24h_mm is not None:
        lines.append(f"🌧 <b>24h Catchment Rain:</b> {rainfall_24h_mm:.1f} mm")

    if gauge_line:
        lines.append(f"🌊 <b>River Gauge:</b> {html.escape(gauge_line, quote=False)}")

    if downstream_stations:
        ds_escaped = html.escape(", ".join(downstream_stations), quote=False)
        lines.append(f"📍 <b>Monitor Downstream:</b> {ds_escaped}")

    body_prefix = "\n".join(lines)
    advisory_clean = html.escape((advisory_text or "").strip(), quote=False)

    if advisory_clean:
        advisory_header = "\n\n📝 <b>Advisory:</b> "
        budget = TELEGRAM_CAPTION_LIMIT - len(body_prefix) - len(advisory_header)
        if budget > 24:
            if len(advisory_clean) > budget:
                # Truncate safely without breaking an HTML entity like &amp;
                truncated = advisory_clean[: max(0, budget - 1)]
                if "&" in truncated[-6:]:
                    amp_pos = truncated.rfind("&")
                    if ";" not in truncated[amp_pos:]:
                        truncated = truncated[:amp_pos]
                advisory_clean = truncated.rstrip() + "…"
            caption = body_prefix + advisory_header + advisory_clean
        else:
            caption = body_prefix
    else:
        caption = body_prefix

    # Final hard clamp to 1,024 characters (never splitting tags/entities)
    if len(caption) > TELEGRAM_CAPTION_LIMIT:
        # Rebuild minimal guaranteed-safe summary if metadata lines alone were very long
        safe_base = f"{emoji} <b>{esc_prefix}: {esc_zone} ({esc_sev})</b>\n⏱ <b>ETA:</b> {esc_eta} | <b>Cover:</b> {storm_coverage_pct:.0f}%"
        caption = safe_base[:TELEGRAM_CAPTION_LIMIT]
    return caption


def _basin_emoji(level: str) -> str:
    lvl = level.upper()
    if lvl == "CRITICAL":
        return "🔴"
    if lvl == "WARNING":
        return "🟠"
    if lvl == "WATCH":
        return "🟡"
    return "🟢"


def build_hourly_summary_caption(
    *,
    frame_timestamp: Optional[str],
    backend_name: str,
    inference_ms: float,
    storm_cells: Sequence[ProjectedStormCell],
    districts: Sequence[DistrictThreat],
    basins: Sequence[RiverBasinRisk],
    surges: Sequence[UpstreamSurgeWarning],
    hydrology_source_status: str,
    advisory_text: str,
) -> str:
    """
    Formats a concise, HTML-safe (<= 1,024 chars) hourly Sri Lanka weather & river basin
    status summary for Telegram `parse_mode="HTML"`.
    """
    slt_tz = timezone(timedelta(hours=5, minutes=30))
    now_utc = datetime.now(timezone.utc)
    if frame_timestamp:
        try:
            ts_clean = frame_timestamp[:-1] + "+00:00" if frame_timestamp.endswith("Z") else frame_timestamp
            ref_dt = datetime.fromisoformat(ts_clean).astimezone(timezone.utc)
        except ValueError:
            ref_dt = now_utc
    else:
        ref_dt = now_utc

    time_str = f"{ref_dt.astimezone(slt_tz).strftime('%H:%M SLT')} ({ref_dt.strftime('%H:%M UTC')})"
    engine = html.escape(_engine_display_name(backend_name), quote=False)
    deep_count = sum(1 for c in storm_cells if c.label == "deep_convective")

    severe_high = [d for d in districts if d.threat_level in ("SEVERE", "HIGH")]
    moderate = [d for d in districts if d.threat_level == "MODERATE"]
    if severe_high:
        top_dist = ", ".join(f"{d.name} ({d.threat_level}, {_format_eta(d.eta_min)})" for d in severe_high[:3])
        district_line = f"⚠️ <b>Districts:</b> {len(severe_high)} High/Severe — {html.escape(top_dist, quote=False)}"
    elif moderate:
        top_mod = ", ".join(d.name for d in moderate[:4])
        district_line = f"🟡 <b>Districts:</b> {len(moderate)} Moderate ({html.escape(top_mod, quote=False)}); others LOW"
    else:
        district_line = "🟢 <b>Districts:</b> All 25 districts at LOW convective threat"

    basin_lines = ["🌊 <b>Wet-Zone River Basins:</b>"]
    for b in basins:
        b_name = html.escape(b.name.replace(" Ganga", ""), quote=False)
        b_risk = html.escape(b.risk_level, quote=False)
        surge_tag = " ⚡SURGE" if b.upstream_surge else ""
        basin_lines.append(
            f"  {_basin_emoji(b.risk_level)} {b_name}: <b>{b_risk}</b> • {b.rainfall_24h_mm:.0f}mm/24h • Cover {b.storm_coverage_pct:.0f}%/{b.projected_60m_pct:.0f}%{surge_tag}"
        )

    hydro_label = html.escape(hydrology_source_status.replace("_", " "), quote=False)
    lines = [
        f"🛰 <b>SL HOURLY WEATHER &amp; FLOOD SUMMARY</b>",
        f"🕒 {html.escape(time_str, quote=False)}  |  🤖 {engine} ({inference_ms:.0f}ms)",
        f"⛈ <b>Himawari-9 IR:</b> {len(storm_cells)} tracked cells ({deep_count} deep convective)",
        district_line,
        *basin_lines,
        f"📡 <b>Hydrology Feed:</b> {hydro_label}",
    ]

    if surges:
        s0 = surges[0]
        lines.append(
            f"🚨 <b>Upstream Surge:</b> {html.escape(s0.basin_name, quote=False)} (+{s0.max_rate_of_rise_ft_hr:.2f} ft/hr, crest {html.escape(s0.estimated_crest_window, quote=False)})"
        )

    prefix = "\n".join(lines)
    adv_clean = html.escape((advisory_text or "").strip(), quote=False)
    if adv_clean:
        adv_hdr = "\n\n📝 "
        budget = TELEGRAM_CAPTION_LIMIT - len(prefix) - len(adv_hdr)
        if budget > 24:
            if len(adv_clean) > budget:
                trunc = adv_clean[: max(0, budget - 1)]
                if "&" in trunc[-6:]:
                    amp_pos = trunc.rfind("&")
                    if ";" not in trunc[amp_pos:]:
                        trunc = trunc[:amp_pos]
                adv_clean = trunc.rstrip() + "…"
            caption = prefix + adv_hdr + adv_clean
        else:
            caption = prefix
    else:
        caption = prefix

    if len(caption) > TELEGRAM_CAPTION_LIMIT:
        caption = (
            f"🛰 <b>SL HOURLY WEATHER &amp; FLOOD SUMMARY</b>\n"
            f"🕒 {html.escape(time_str, quote=False)}\n"
            f"⛈ Cells: {len(storm_cells)} | {district_line}"
        )[:TELEGRAM_CAPTION_LIMIT]
    return caption


class TelegramAlertDispatcher:
    """
    Async Telegram Bot photo-alert dispatcher using `httpx`.
    Enforces per-zone cooldown with immediate severity-escalation bypass, HTML-safe
    1,024-char captions, HTTP 429/network retry backoff, and SQLite `alert_history` persistence.
    """

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.cooldown = timedelta(minutes=settings.telegram_alert_cooldown_min)

        if not self.settings.telegram_enabled:
            logger.info("Telegram alerts disabled via TELEGRAM_ENABLED=false")
        elif not self.settings.telegram_configured:
            logger.info(
                "Telegram alerts unconfigured (TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID empty); "
                "nowcasting will run cleanly without Telegram dispatch."
            )
        else:
            logger.info(
                "Telegram alert dispatcher ready (chat=%s, cooldown=%dm)",
                mask_chat_id(self.settings.telegram_chat_id),
                self.settings.telegram_alert_cooldown_min,
            )

    @property
    def is_ready(self) -> bool:
        return self.settings.telegram_configured

    def should_dispatch(self, zone_id: str, severity: str, now_utc: datetime) -> tuple[bool, str]:
        """
        Checks SQLite `alert_history` for `zone_id`:
          - Allows immediately if no prior sent alert exists
          - Bypasses cooldown immediately if severity escalated (e.g. WARNING -> CRITICAL)
          - Suppresses duplicates within TELEGRAM_ALERT_COOLDOWN_MIN otherwise
        """
        last = self.db.last_alert_record(zone_id, only_sent=True)
        if last is None:
            return True, "first_alert"

        last_ts = datetime.fromisoformat(str(last["sent_at"]).replace("Z", "+00:00"))
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        elapsed = now_utc - last_ts
        if elapsed >= self.cooldown:
            return True, "cooldown_expired"

        prev_rank = SEVERITY_RANK.get(str(last.get("level", "")).upper(), 0)
        curr_rank = SEVERITY_RANK.get(severity.upper(), 0)
        if curr_rank > prev_rank:
            return True, f"escalation_bypass:{last.get('level')}->{severity}"

        return False, "cooldown_active"

    async def _post_photo_with_retry(
        self,
        jpeg_bytes: bytes,
        caption: str,
        *,
        client: Optional[httpx.AsyncClient] = None,
        max_attempts: int = 3,
        base_backoff_sec: float = 1.0,
    ) -> tuple[bool, str, Optional[str]]:
        ok, status, err, _ = await self.post_photo_returning_id(
            jpeg_bytes,
            caption,
            client=client,
            max_attempts=max_attempts,
            base_backoff_sec=base_backoff_sec,
        )
        return ok, status, err

    async def post_photo_returning_id(
        self,
        jpeg_bytes: bytes,
        caption: str,
        *,
        client: Optional[httpx.AsyncClient] = None,
        max_attempts: int = 3,
        base_backoff_sec: float = 1.0,
    ) -> tuple[bool, str, Optional[str], Optional[int]]:
        token = self.settings.telegram_bot_token.strip()
        chat_id = self.settings.telegram_chat_id.strip()
        url = f"{TELEGRAM_API_BASE}/bot{token}/sendPhoto"

        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=20.0)
            close_client = True

        last_err: Optional[str] = None
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = await client.post(
                        url,
                        data={
                            "chat_id": chat_id,
                            "caption": caption,
                            "parse_mode": "HTML",
                        },
                        files={"photo": ("himawari_alert.jpg", jpeg_bytes, "image/jpeg")},
                    )

                    if resp.status_code == 200:
                        msg_id = None
                        try:
                            msg_id = resp.json().get("result", {}).get("message_id")
                        except Exception:
                            pass
                        return True, "sent_telegram_200", None, msg_id

                    if resp.status_code == 429:
                        retry_after = base_backoff_sec * (2 ** (attempt - 1))
                        try:
                            body = resp.json()
                            ra = body.get("parameters", {}).get("retry_after")
                            if ra is not None:
                                retry_after = min(float(ra), 10.0)
                        except Exception:
                            hdr_ra = resp.headers.get("Retry-After")
                            if hdr_ra and hdr_ra.isdigit():
                                retry_after = min(float(hdr_ra), 10.0)
                        last_err = f"HTTP 429 Too Many Requests (attempt {attempt}/{max_attempts})"
                        logger.warning("Telegram rate limited (%s); backing off %.2fs", last_err, retry_after)
                        if attempt < max_attempts:
                            await asyncio.sleep(retry_after)
                            continue
                        return False, "failed_telegram_429", last_err, None

                    if resp.status_code >= 500:
                        last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
                        if attempt < max_attempts:
                            await asyncio.sleep(base_backoff_sec * (2 ** (attempt - 1)))
                            continue
                        return False, f"failed_telegram_{resp.status_code}", last_err, None

                    # Non-retryable 4xx error
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    return False, f"failed_telegram_{resp.status_code}", last_err, None
                except httpx.HTTPError as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    logger.warning("Telegram network error (attempt %d/%d): %s", attempt, max_attempts, last_err)
                    if attempt < max_attempts:
                        await asyncio.sleep(base_backoff_sec * (2 ** (attempt - 1)))
                        continue
                    return False, "failed_telegram_network", last_err, None
        finally:
            if close_client:
                await client.aclose()

        return False, "failed_telegram_unknown", last_err, None

    async def post_reply_message(
        self,
        text: str,
        *,
        reply_to_message_id: Optional[int] = None,
        client: Optional[httpx.AsyncClient] = None,
        max_attempts: int = 3,
        base_backoff_sec: float = 1.0,
    ) -> tuple[bool, str, Optional[str]]:
        """Sends an HTML follow-up text message (used when photo digest overflows 1,024 chars)."""
        token = self.settings.telegram_bot_token.strip()
        chat_id = self.settings.telegram_chat_id.strip()
        url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": "HTML",
        }
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id

        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=15.0)
            close_client = True

        last_err: Optional[str] = None
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        return True, "sent_telegram_200", None
                    last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                        await asyncio.sleep(base_backoff_sec * (2 ** (attempt - 1)))
                        continue
                    return False, f"failed_telegram_{resp.status_code}", last_err
                except httpx.HTTPError as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                    if attempt < max_attempts:
                        await asyncio.sleep(base_backoff_sec * (2 ** (attempt - 1)))
                        continue
                    return False, "failed_telegram_network", last_err
        finally:
            if close_client:
                await client.aclose()
        return False, "failed_telegram_unknown", last_err

    @staticmethod
    def _summarize_basin_gauge(basin_id: str, gauges: Sequence[RiverGaugeReading]) -> Optional[str]:
        b_gauges = [g for g in gauges if g.basin_id == basin_id]
        if not b_gauges:
            return None
        rank_order = ["NORMAL", "ALERT", "MINOR_FLOOD", "MAJOR_FLOOD"]
        top = max(b_gauges, key=lambda g: (rank_order.index(g.status), g.rate_of_rise_ft_hr))
        return (
            f"{top.station_name} {top.water_level_ft:.2f} ft "
            f"({top.status.replace('_', ' ')}, {top.trend}, {top.rate_of_rise_ft_hr:+.2f} ft/hr)"
        )

    async def dispatch_nowcast_alerts(
        self,
        *,
        frame_bgr: Optional[np.ndarray],
        frame_timestamp: Optional[str],
        backend_name: str,
        detections: Sequence[StormDetection],
        storm_cells: Sequence[ProjectedStormCell],
        districts: Sequence[DistrictThreat],
        basins: Sequence[RiverBasinRisk],
        basins_geojson: Optional[dict[str, Any]],
        gauges: Sequence[RiverGaugeReading],
        advisory_text: str,
        ir_loop_frames: Optional[Sequence[np.ndarray]] = None,
        sea_zones_geojson: Optional[dict[str, Any]] = None,
        projected_geojson: Optional[dict[str, dict[str, Any]]] = None,
        client: Optional[httpx.AsyncClient] = None,
        now: Optional[datetime] = None,
        base_backoff_sec: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Triggers Telegram photo alerts for any district at HIGH/SEVERE or river basin at
        WARNING/CRITICAL, enforcing per-zone cooldown with immediate escalation bypass.
        """
        if not self.is_ready:
            return []

        now_utc = now or datetime.now(timezone.utc)
        engine_label = _engine_display_name(backend_name)
        results: list[dict[str, Any]] = []

        # Build prioritized alert candidates: river basins first (flood safety), then districts
        candidates: list[dict[str, Any]] = []
        for b in basins:
            if b.risk_level in ("WARNING", "CRITICAL"):
                candidates.append(
                    {
                        "zone_id": f"telegram:basin:{b.basin_id}",
                        "entity_type": "basin",
                        "zone_name": b.name,
                        "severity": b.risk_level,
                        "title": f"{b.risk_level} FLOOD ALERT - {b.name.upper()}",
                        "caption": build_safe_html_caption(
                            header_prefix="FLOOD ALERT",
                            severity=b.risk_level,
                            zone_name=b.name,
                            eta_min=b.eta_min,
                            storm_coverage_pct=b.storm_coverage_pct,
                            projected_60m_pct=b.projected_60m_pct,
                            rainfall_24h_mm=b.rainfall_24h_mm,
                            gauge_line=self._summarize_basin_gauge(b.basin_id, gauges),
                            downstream_stations=b.downstream_stations,
                            advisory_text=advisory_text,
                        ),
                    }
                )

        for d in districts:
            if d.threat_level in ("HIGH", "SEVERE"):
                candidates.append(
                    {
                        "zone_id": f"telegram:district:{d.district_id}",
                        "entity_type": "district",
                        "zone_name": f"{d.name} District",
                        "severity": d.threat_level,
                        "title": f"{d.threat_level} STORM ALERT - {d.name.upper()}",
                        "caption": build_safe_html_caption(
                            header_prefix="STORM ALERT",
                            severity=d.threat_level,
                            zone_name=f"{d.name} District",
                            eta_min=d.eta_min,
                            storm_coverage_pct=d.affected_area_pct,
                            projected_60m_pct=d.coverage_by_horizon.get("+60m"),
                            rainfall_24h_mm=None,
                            gauge_line=(
                                f"Nearest {d.nearest_station}: {d.precipitation_mm:.1f} mm rain, "
                                f"{d.wind_gusts_kmh:.0f} km/h gusts"
                                if d.nearest_station
                                else None
                            ),
                            downstream_stations=[],
                            advisory_text=advisory_text,
                        ),
                    }
                )

        if not candidates:
            return results

        for item in candidates:
            allowed, reason = self.should_dispatch(item["zone_id"], item["severity"], now_utc)
            if not allowed:
                continue

            jpeg_bytes = render_alert_snapshot(
                frame_bgr=frame_bgr,
                detections=detections,
                motions=storm_cells,
                basins_geojson=basins_geojson,
                timestamp_str=frame_timestamp,
                ir_loop_frames=ir_loop_frames,
                districts=districts,
                sea_zones_geojson=sea_zones_geojson,
                projected_geojson=projected_geojson,
                bbox=self.settings.bbox_tuple,
                alert_title=item["title"],
                engine_label=engine_label,
            )

            ok, status, err = await self._post_photo_with_retry(
                jpeg_bytes,
                item["caption"],
                client=client,
                base_backoff_sec=base_backoff_sec,
            )

            self.db.record_alert(
                entity_key=item["zone_id"],
                entity_type=item["entity_type"],
                level=item["severity"],
                webhook_status=status,
                payload={
                    "channel": "telegram",
                    "zone_name": item["zone_name"],
                    "reason": reason,
                    "caption": item["caption"],
                    "snapshot_bytes": len(jpeg_bytes),
                    "error": err,
                },
                sent_at=now_utc,
            )
            results.append(
                {
                    "zone_id": item["zone_id"],
                    "zone_name": item["zone_name"],
                    "severity": item["severity"],
                    "ok": ok,
                    "status": status,
                    "reason": reason,
                    "error": err,
                }
            )

        return results

    async def send_test_alert(
        self,
        *,
        frame_bgr: Optional[np.ndarray],
        frame_timestamp: Optional[str],
        backend_name: str,
        detections: Sequence[StormDetection],
        storm_cells: Sequence[ProjectedStormCell],
        basins_geojson: Optional[dict[str, Any]],
        advisory_text: str,
        ir_loop_frames: Optional[Sequence[np.ndarray]] = None,
        districts: Optional[Sequence[DistrictThreat]] = None,
        sea_zones_geojson: Optional[dict[str, Any]] = None,
        projected_geojson: Optional[dict[str, dict[str, Any]]] = None,
        client: Optional[httpx.AsyncClient] = None,
        base_backoff_sec: float = 0.5,
    ) -> dict[str, Any]:
        """Renders a snapshot from the latest buffered Himawari-9 frame and sends a [TEST ALERT] photo."""
        now_utc = datetime.now(timezone.utc)
        engine_label = _engine_display_name(backend_name)

        jpeg_bytes = render_alert_snapshot(
            frame_bgr=frame_bgr,
            detections=detections,
            motions=storm_cells,
            basins_geojson=basins_geojson,
            timestamp_str=frame_timestamp,
            ir_loop_frames=ir_loop_frames,
            districts=districts,
            sea_zones_geojson=sea_zones_geojson,
            projected_geojson=projected_geojson,
            bbox=self.settings.bbox_tuple,
            alert_title="[TEST ALERT] SRI LANKA EDGE NOWCAST",
            engine_label=engine_label,
        )

        if not self.settings.telegram_enabled:
            return {
                "ok": False,
                "status": "disabled",
                "detail": "Telegram alerting is disabled (TELEGRAM_ENABLED=false).",
                "sent_at": None,
                "snapshot_bytes": len(jpeg_bytes),
            }

        if not self.settings.telegram_configured:
            return {
                "ok": False,
                "status": "unconfigured",
                "detail": "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable Telegram delivery.",
                "sent_at": None,
                "snapshot_bytes": len(jpeg_bytes),
            }

        caption = build_safe_html_caption(
            header_prefix="[TEST ALERT] SL Edge Nowcast",
            severity="TEST",
            zone_name="Sri Lanka System Check",
            eta_min=0 if storm_cells else None,
            storm_coverage_pct=0.0,
            projected_60m_pct=None,
            rainfall_24h_mm=None,
            gauge_line=f"Engine: {engine_label} | Tracked storm cells: {len(storm_cells)}",
            downstream_stations=[],
            advisory_text=advisory_text or "Diagnostic test notification from Raspberry Pi 5 weather nowcasting node.",
        )

        ok, status, err = await self._post_photo_with_retry(
            jpeg_bytes,
            caption,
            client=client,
            base_backoff_sec=base_backoff_sec,
        )

        self.db.record_alert(
            entity_key="telegram:test:system",
            entity_type="test",
            level="TEST",
            webhook_status=status,
            payload={
                "channel": "telegram",
                "zone_name": "[TEST ALERT] Sri Lanka System Check",
                "reason": "manual_test",
                "caption": caption,
                "snapshot_bytes": len(jpeg_bytes),
                "error": err,
            },
            sent_at=now_utc,
        )

        return {
            "ok": ok,
            "status": status,
            "detail": (
                f"Test snapshot ({len(jpeg_bytes) // 1024} KB) delivered to Telegram chat {mask_chat_id(self.settings.telegram_chat_id)}."
                if ok
                else f"Telegram delivery failed: {err or status}"
            ),
            "sent_at": now_utc.isoformat(),
            "snapshot_bytes": len(jpeg_bytes),
        }

    async def send_hourly_summary(
        self,
        *,
        frame_bgr: Optional[np.ndarray],
        frame_timestamp: Optional[str],
        backend_name: str,
        inference_ms: float,
        detections: Sequence[Any],
        storm_cells: Sequence[ProjectedStormCell],
        districts: Sequence[DistrictThreat],
        basins: Sequence[RiverBasinRisk],
        basins_geojson: Optional[dict[str, Any]],
        surges: Sequence[UpstreamSurgeWarning],
        hydrology_source_status: str,
        advisory_text: str,
        ir_loop_frames: Optional[Sequence[np.ndarray]] = None,
        sea_zones_geojson: Optional[dict[str, Any]] = None,
        projected_geojson: Optional[dict[str, dict[str, Any]]] = None,
        client: Optional[httpx.AsyncClient] = None,
        base_backoff_sec: float = 1.0,
    ) -> dict[str, Any]:
        """Renders an annotated Himawari-9 snapshot and sends the hourly status summary to Telegram."""
        now_utc = datetime.now(timezone.utc)
        engine_label = _engine_display_name(backend_name)

        jpeg_bytes = render_alert_snapshot(
            frame_bgr=frame_bgr,
            detections=detections,
            motions=storm_cells,
            basins_geojson=basins_geojson,
            timestamp_str=frame_timestamp,
            ir_loop_frames=ir_loop_frames,
            districts=districts,
            sea_zones_geojson=sea_zones_geojson,
            projected_geojson=projected_geojson,
            bbox=self.settings.bbox_tuple,
            alert_title="HOURLY STATUS SUMMARY - SRI LANKA",
            engine_label=engine_label,
        )

        if not self.settings.telegram_enabled:
            return {
                "ok": False,
                "status": "disabled",
                "detail": "Telegram alerting is disabled (TELEGRAM_ENABLED=false).",
                "sent_at": None,
                "snapshot_bytes": len(jpeg_bytes),
            }

        if not self.settings.telegram_configured:
            return {
                "ok": False,
                "status": "unconfigured",
                "detail": "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable hourly summaries.",
                "sent_at": None,
                "snapshot_bytes": len(jpeg_bytes),
            }

        caption = build_hourly_summary_caption(
            frame_timestamp=frame_timestamp,
            backend_name=backend_name,
            inference_ms=inference_ms,
            storm_cells=storm_cells,
            districts=districts,
            basins=basins,
            surges=surges,
            hydrology_source_status=hydrology_source_status,
            advisory_text=advisory_text,
        )

        ok, status, err = await self._post_photo_with_retry(
            jpeg_bytes,
            caption,
            client=client,
            base_backoff_sec=base_backoff_sec,
        )

        self.db.record_alert(
            entity_key="telegram:summary:hourly",
            entity_type="summary",
            level="INFO",
            webhook_status=status,
            payload={
                "channel": "telegram",
                "zone_name": "Hourly Status Summary (Sri Lanka)",
                "reason": "hourly_summary",
                "caption": caption,
                "snapshot_bytes": len(jpeg_bytes),
                "error": err,
            },
            sent_at=now_utc,
        )

        if ok:
            logger.info("Sent Telegram hourly status summary (%d KB) to %s", len(jpeg_bytes) // 1024, mask_chat_id(self.settings.telegram_chat_id))
        else:
            logger.warning("Failed to send Telegram hourly status summary: %s", err or status)

        return {
            "ok": ok,
            "status": status,
            "detail": (
                f"Hourly status summary ({len(jpeg_bytes) // 1024} KB) delivered to Telegram chat {mask_chat_id(self.settings.telegram_chat_id)}."
                if ok
                else f"Hourly summary delivery failed: {err or status}"
            ),
            "sent_at": now_utc.isoformat(),
            "snapshot_bytes": len(jpeg_bytes),
        }

