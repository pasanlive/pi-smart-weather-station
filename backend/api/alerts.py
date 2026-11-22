from __future__ import annotations

import re
from fastapi import APIRouter, Query

from alerts.telegram_notifier import mask_chat_id
from models import (
    AlertHistoryItem,
    AlertHistoryResponse,
    AlertStatusResponse,
    HourlyDigestTestResponse,
    TelegramTestResponse,
)
from nowcast.pipeline import NowcastPipeline

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(text: str) -> str:
    return _HTML_TAG_RE.sub("", text or "").strip()


def create_alerts_router(pipeline: NowcastPipeline) -> APIRouter:
    router = APIRouter(prefix="/api/alerts", tags=["alerts"])

    @router.get("/status", response_model=AlertStatusResponse)
    async def get_alerts_status() -> AlertStatusResponse:
        summary = pipeline.db.get_alert_summary(channel_prefix="telegram:")
        return AlertStatusResponse(
            telegram_enabled=pipeline.settings.telegram_enabled,
            telegram_configured=pipeline.settings.telegram_configured,
            telegram_ready=pipeline.telegram_dispatcher.is_ready,
            cooldown_min=pipeline.settings.telegram_alert_cooldown_min,
            hourly_summary_enabled=pipeline.settings.telegram_hourly_summary_enabled,
            hourly_summary_interval_min=pipeline.settings.telegram_hourly_summary_interval_min,
            chat_id_masked=mask_chat_id(pipeline.settings.telegram_chat_id),
            last_dispatched_at=summary["last_dispatched_at"],
            last_status=summary["last_status"],
            total_logged=summary["total_logged"],
        )

    @router.get("/history", response_model=AlertHistoryResponse)
    async def get_alerts_history(limit: int = Query(default=20, ge=1, le=100)) -> AlertHistoryResponse:
        rows = pipeline.db.list_alert_history(limit=limit)
        items: list[AlertHistoryItem] = []
        for r in rows:
            payload = r.get("payload") or {}
            channel = str(
                payload.get("channel")
                or ("telegram" if str(r["entity_key"]).startswith("telegram:") else "webhook")
            )
            zone_name = str(
                payload.get("zone_name")
                or payload.get("title")
                or str(r["entity_key"]).split(":")[-1].replace("_", " ").title()
            )
            raw_caption = str(payload.get("caption") or payload.get("message") or payload.get("content") or "")
            preview = _strip_tags(raw_caption)[:180]
            items.append(
                AlertHistoryItem(
                    id=int(r["id"]),
                    entity_key=str(r["entity_key"]),
                    entity_type=str(r["entity_type"]),
                    zone_name=zone_name,
                    level=str(r["level"]),
                    channel=channel,
                    status=str(r["webhook_status"]),
                    caption_preview=preview,
                    error=payload.get("error"),
                    sent_at=str(r["sent_at"]),
                )
            )
        return AlertHistoryResponse(count=len(items), alerts=items)

    @router.post("/telegram/test", response_model=TelegramTestResponse)
    async def send_telegram_test_alert() -> TelegramTestResponse:
        result = await pipeline.send_telegram_test_alert()
        return TelegramTestResponse(**result)

    @router.post("/telegram/summary", response_model=TelegramTestResponse)
    async def send_telegram_hourly_summary() -> TelegramTestResponse:
        result = await pipeline.send_telegram_hourly_summary()
        return TelegramTestResponse(**result)

    @router.post("/hourly-digest/test", response_model=HourlyDigestTestResponse)
    async def test_hourly_hazard_digest(
        dispatch: bool = Query(default=True, description="Set false to preview without sending to Telegram"),
    ) -> HourlyDigestTestResponse:
        result = await pipeline.send_hourly_hazard_digest(
            send_to_telegram=dispatch,
            is_manual_test=True,
        )
        return HourlyDigestTestResponse(**result)

    return router

