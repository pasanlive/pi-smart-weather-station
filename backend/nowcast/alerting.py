from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from config import Settings
from models import DistrictThreat, RiverBasinRisk
from storage.db import Database

logger = logging.getLogger(__name__)


class AlertDispatcher:
    """
    Sends webhook alerts (Generic JSON / Discord / Telegram-compatible) whenever:
      - Any Sri Lanka district escalates to HIGH or SEVERE
      - Any wet-zone river basin escalates to WARNING or CRITICAL
    Enforces a configurable per-entity cooldown (default 30 minutes) persisted in SQLite.
    """

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.cooldown = timedelta(minutes=settings.alert_cooldown_min)

    def _is_in_cooldown(self, entity_key: str, now_utc: datetime) -> bool:
        last_sent = self.db.last_alert_time(entity_key)
        if last_sent is None:
            return False
        if last_sent.tzinfo is None:
            last_sent = last_sent.replace(tzinfo=timezone.utc)
        return (now_utc - last_sent) < self.cooldown

    @staticmethod
    def _build_webhook_body(webhook_url: str, title: str, message: str, meta: dict[str, Any]) -> dict[str, Any]:
        url_low = webhook_url.lower()
        if "discord.com/api/webhooks" in url_low:
            return {"content": f"**{title}**\n{message}"}
        if "api.telegram.org" in url_low:
            return {"text": f"{title}\n{message}"}
        return {
            "event": "sl_nowcast_alert",
            "title": title,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": meta,
        }

    async def dispatch_if_needed(
        self,
        districts: list[DistrictThreat],
        basins: list[RiverBasinRisk],
        client: Optional[httpx.AsyncClient] = None,
        now: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        now_utc = now or datetime.now(timezone.utc)
        candidates: list[tuple[str, str, str, str, str, dict[str, Any]]] = []

        for d in districts:
            if d.threat_level in ("HIGH", "SEVERE"):
                key = f"district:{d.district_id}"
                eta_lbl = "NOW" if d.eta_min == 0 else (f"+{d.eta_min}m" if d.eta_min is not None else "N/A")
                candidates.append(
                    (
                        key,
                        "district",
                        d.threat_level,
                        f"[{d.threat_level}] {d.name} District Weather Alert",
                        f"Peak storm coverage {d.peak_coverage_pct:.0f}% (ETA {eta_lbl}), rain {d.precipitation_mm:.1f} mm, gusts {d.wind_gusts_kmh:.0f} km/h.",
                        d.model_dump(),
                    )
                )

        for b in basins:
            if b.risk_level in ("WARNING", "CRITICAL"):
                key = f"basin:{b.basin_id}"
                candidates.append(
                    (
                        key,
                        "basin",
                        b.risk_level,
                        f"[{b.risk_level}] {b.name} Flood Alert",
                        f"Storm cover {b.storm_coverage_pct:.0f}% now / {b.projected_60m_pct:.0f}% +60m, 24h rain {b.rainfall_24h_mm:.0f} mm. Monitor: {', '.join(b.downstream_stations)}.",
                        {
                            "basin_id": b.basin_id,
                            "name": b.name,
                            "risk_level": b.risk_level,
                            "storm_coverage_pct": b.storm_coverage_pct,
                            "projected_60m_pct": b.projected_60m_pct,
                            "rainfall_24h_mm": b.rainfall_24h_mm,
                            "upstream_surge": b.upstream_surge,
                            "reasons": b.reasons,
                        },
                    )
                )

        dispatched: list[dict[str, Any]] = []
        if not candidates:
            return dispatched

        webhook_url = self.settings.alert_webhook_url.strip()
        close_client = False
        if webhook_url and client is None:
            client = httpx.AsyncClient(timeout=10.0)
            close_client = True

        try:
            for key, etype, level, title, message, meta in candidates:
                if self._is_in_cooldown(key, now_utc):
                    continue
                body = self._build_webhook_body(webhook_url, title, message, meta)
                status = "recorded_no_webhook"
                if webhook_url and client is not None:
                    try:
                        resp = await client.post(webhook_url, json=body)
                        resp.raise_for_status()
                        status = f"sent_http_{resp.status_code}"
                    except Exception as exc:
                        logger.warning("Webhook delivery failed for %s: %s", key, exc)
                        status = f"failed:{exc}"

                self.db.record_alert(
                    entity_key=key,
                    entity_type=etype,
                    level=level,
                    webhook_status=status,
                    payload=body,
                    sent_at=now_utc,
                )
                dispatched.append({"entity_key": key, "entity_type": etype, "level": level, "status": status})
        finally:
            if close_client and client is not None:
                await client.aclose()

        return dispatched
