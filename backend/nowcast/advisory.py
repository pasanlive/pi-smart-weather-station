from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from config import Settings
from models import DistrictThreat, ProjectedStormCell, RiverBasinRisk, UpstreamSurgeWarning

logger = logging.getLogger(__name__)


def _deterministic_advisory(
    storm_cells: list[ProjectedStormCell],
    districts: list[DistrictThreat],
    basins: list[RiverBasinRisk],
    surges: list[UpstreamSurgeWarning],
) -> str:
    severe_high = [d for d in districts if d.threat_level in ("SEVERE", "HIGH")]
    moderate = [d for d in districts if d.threat_level == "MODERATE"]
    elevated_basins = [b for b in basins if b.risk_level in ("CRITICAL", "WARNING", "WATCH")]

    if severe_high:
        top_names = ", ".join(d.name for d in severe_high[:4])
        soonest = severe_high[0]
        eta_str = "currently impacting" if soonest.eta_min == 0 else (
            f"projected within +{soonest.eta_min} min for" if soonest.eta_min else "affecting"
        )
        lead = (
            f"Active convective weather ({len(storm_cells)} tracked cells) is {eta_str} "
            f"{top_names} with peak threat at {soonest.threat_level}."
        )
    elif moderate:
        top_names = ", ".join(d.name for d in moderate[:4])
        lead = (
            f"Moderate cloud and shower activity ({len(storm_cells)} tracked cells) is observed near "
            f"{top_names}; localized rainfall and gusty winds remain possible over the next 2 hours."
        )
    else:
        lead = (
            f"Himawari-9 Band 13 IR shows {len(storm_cells)} regional cloud feature(s) with low immediate "
            f"convective threat across Sri Lanka's 25 districts over the next 2 hours."
        )

    if surges:
        s0 = surges[0]
        tail = (
            f"Hydrological alert: upstream surge detected in {s0.basin_name} (+{s0.max_rate_of_rise_ft_hr:.2f} ft/hr) "
            f"with downstream crest expected in {s0.estimated_crest_window}."
        )
    elif elevated_basins:
        b0 = elevated_basins[0]
        tail = (
            f"Hydrological status: {b0.name} is at {b0.risk_level} ({b0.rainfall_24h_mm:.0f} mm/24h rain, "
            f"{b0.projected_60m_pct:.0f}% +60m storm cover); monitor {', '.join(b0.downstream_stations[:2])}."
        )
    else:
        tail = "All four wet-zone river basins (Kelani, Kalu, Gin, Nilwala) are currently at NORMAL flood risk."

    return f"{lead} {tail}"


async def generate_advisory(
    settings: Settings,
    storm_cells: list[ProjectedStormCell],
    districts: list[DistrictThreat],
    basins: list[RiverBasinRisk],
    surges: list[UpstreamSurgeWarning],
) -> tuple[str, str]:
    """
    Sends the structured threat report to a local Ollama LLM if OLLAMA_URL is configured
    and reachable; otherwise returns a crisp deterministic 2-sentence weather & flood advisory.
    """
    fallback_text = _deterministic_advisory(storm_cells, districts, basins, surges)
    if not settings.ollama_url.strip():
        return fallback_text, "template"

    summary_payload: dict[str, Any] = {
        "storm_cells_count": len(storm_cells),
        "top_districts": [
            {
                "name": d.name,
                "threat": d.threat_level,
                "eta_min": d.eta_min,
                "peak_pct": d.peak_coverage_pct,
                "rain_mm": d.precipitation_mm,
                "gust_kmh": d.wind_gusts_kmh,
            }
            for d in districts[:5]
        ],
        "river_basins": [
            {
                "name": b.name,
                "risk": b.risk_level,
                "rain_24h_mm": b.rainfall_24h_mm,
                "curr_pct": b.storm_coverage_pct,
                "proj_60m_pct": b.projected_60m_pct,
                "upstream_surge": b.upstream_surge,
            }
            for b in basins
        ],
        "upstream_surges": [s.message for s in surges],
    }

    prompt = (
        "You are an operational Sri Lanka meteorology & hydrology officer. "
        "Write exactly 2 concise plain-English sentences summarizing the immediate storm threat "
        "and wet-zone river flood risk from this JSON telemetry:\n"
        f"{json.dumps(summary_payload)}"
    )

    try:
        endpoint = settings.ollama_url.rstrip("/") + "/api/generate"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                endpoint,
                json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            text = str(data.get("response") or "").strip()
            if text:
                return text, f"ollama:{settings.ollama_model}"
    except Exception as exc:
        logger.debug("Ollama advisory unavailable (%s); using deterministic template.", exc)

    return fallback_text, "template"
