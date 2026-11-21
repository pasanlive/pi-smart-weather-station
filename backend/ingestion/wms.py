from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

GIBS_WMS_BASE_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
HIMAWARI_LAYER = "Himawari_AHI_Band13_Clean_Infrared"
DEFAULT_BBOX = (75.0, 3.0, 86.0, 12.0)


def format_gibs_timestamp(dt: datetime) -> str:
    """Formats a datetime to NASA GIBS ISO-8601 UTC timestamp rounded down to 10-minute slots."""
    utc_dt = dt.astimezone(timezone.utc)
    floored_min = (utc_dt.minute // 10) * 10
    utc_dt = utc_dt.replace(minute=floored_min, second=0, microsecond=0)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_wms_url(
    timestamp: Optional[str | datetime] = None,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    width: int = 640,
    height: int = 640,
    layer: str = HIMAWARI_LAYER,
    base_url: str = GIBS_WMS_BASE_URL,
) -> str:
    """Builds a NASA GIBS WMS 1.1.1 GetMap URL for Himawari-9 Band 13 Clean IR."""
    if isinstance(timestamp, datetime):
        time_param = format_gibs_timestamp(timestamp)
    elif isinstance(timestamp, str) and timestamp.strip():
        time_param = timestamp.strip()
    else:
        time_param = "DEFAULT"

    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "SRS": "EPSG:4326",
        "BBOX": bbox_str,
        "WIDTH": str(width),
        "HEIGHT": str(height),
        "FORMAT": "image/png",
        "TIME": time_param,
    }
    return f"{base_url}?{urlencode(params)}"


async def discover_latest_gibs_timestamp(
    client: Optional[httpx.AsyncClient] = None,
    layer: str = HIMAWARI_LAYER,
    base_url: str = GIBS_WMS_BASE_URL,
) -> Optional[datetime]:
    """
    Reads GIBS WMS GetCapabilities to discover the latest published `default` timestamp
    for Himawari-9 Band 13 Clean IR (which typically lags real-time by ~90-120 minutes).
    """
    caps_url = f"{base_url}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetCapabilities"
    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=20.0)
        close_client = True

    try:
        resp = await client.get(caps_url)
        resp.raise_for_status()
        text = resp.text
        idx = text.find(f"<Name>{layer}</Name>")
        if idx == -1:
            idx = text.find(layer)
        if idx == -1:
            return None
        snippet = text[idx : idx + 2500]
        match = re.search(r'<Extent[^>]*name="time"[^>]*default="([^"]+)"', snippet)
        if not match:
            return None
        raw = match.group(1).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except Exception as exc:
        logger.warning("Failed to query GIBS GetCapabilities for latest time: %s", exc)
        return None
    finally:
        if close_client:
            await client.aclose()
