from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import httpx

from config import Settings
from models import (
    DistrictThreat,
    ProjectedStormCell,
    RiverBasinRisk,
    RiverGaugeReading,
    UpstreamSurgeWarning,
)

logger = logging.getLogger(__name__)

AdvisorySourceType = Literal["ollama", "template", "template_fallback"]

SEVERITY_LEVEL_RANK: dict[str, int] = {
    "NORMAL": 1,
    "LOW": 1,
    "WATCH": 2,
    "MODERATE": 2,
    "WARNING": 3,
    "HIGH": 3,
    "CRITICAL": 4,
    "SEVERE": 4,
}


def _format_eta(eta_min: Optional[int]) -> str:
    if eta_min == 0:
        return "NOW"
    if eta_min is not None:
        return f"+{eta_min}m"
    return "no direct arrival"


@dataclass
class AdvisoryResult:
    text: str
    source: AdvisorySourceType
    model: Optional[str] = None
    latency_ms: float = 0.0


class WeatherAdvisoryGenerator:
    """
    CPU-pinned local Ollama 3B (qwen2.5:3b) weather & flood advisory generator with
    instant deterministic template fallback and severity gating.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def peak_severity_rank(
        districts: Sequence[DistrictThreat],
        basins: Sequence[RiverBasinRisk],
        surges: Sequence[UpstreamSurgeWarning] = (),
    ) -> int:
        rank = 1
        for d in districts:
            rank = max(rank, SEVERITY_LEVEL_RANK.get(d.threat_level.upper(), 1))
        for b in basins:
            rank = max(rank, SEVERITY_LEVEL_RANK.get(b.risk_level.upper(), 1))
        if surges:
            rank = max(rank, 3)  # Upstream surge warrants LLM narrative
        return rank

    def should_invoke_llm(
        self,
        districts: Sequence[DistrictThreat],
        basins: Sequence[RiverBasinRisk],
        surges: Sequence[UpstreamSurgeWarning] = (),
    ) -> bool:
        if not self.settings.ollama_enabled or not self.settings.ollama_url.strip():
            return False
        min_rank = SEVERITY_LEVEL_RANK.get(self.settings.ollama_min_severity.strip().upper(), 2)
        return self.peak_severity_rank(districts, basins, surges) >= min_rank

    @staticmethod
    def generate_template_advisory(
        storm_cells: Sequence[ProjectedStormCell],
        districts: Sequence[DistrictThreat],
        basins: Sequence[RiverBasinRisk],
        gauges: Sequence[RiverGaugeReading] = (),
        surges: Sequence[UpstreamSurgeWarning] = (),
    ) -> str:
        """
        Zero-latency deterministic 2-sentence factual summary citing the highest-risk
        river basins/districts, ETA (NOW/+30m/+60m/+120m), storm motion, 24h rainfall,
        and any rising Sri Lanka Irrigation Dept river gauges.
        """
        # Dominant storm motion summary
        moving_cells = [c for c in storm_cells if c.speed_kmh >= 2.0]
        motion_clause = ""
        if moving_cells:
            lead_cell = max(
                moving_cells,
                key=lambda c: (1 if c.label == "deep_convective" else 0, c.speed_kmh),
            )
            motion_clause = f" moving {lead_cell.compass_dir} at {lead_cell.speed_kmh:.0f} km/h"

        severe_high = [d for d in districts if d.threat_level in ("SEVERE", "HIGH")]
        moderate = [d for d in districts if d.threat_level == "MODERATE"]

        if severe_high:
            top_names = ", ".join(d.name for d in severe_high[:3])
            d0 = severe_high[0]
            eta_txt = _format_eta(d0.eta_min)
            sentence_1 = (
                f"{d0.threat_level} storm threat across {top_names} (ETA {eta_txt}, "
                f"peak cover {d0.peak_coverage_pct:.0f}%), driven by {len(storm_cells)} tracked "
                f"Himawari-9 cell(s){motion_clause}."
            )
        elif moderate:
            top_names = ", ".join(d.name for d in moderate[:3])
            d0 = moderate[0]
            eta_txt = _format_eta(d0.eta_min)
            sentence_1 = (
                f"MODERATE cloud and shower activity near {top_names} (ETA {eta_txt}) with "
                f"{len(storm_cells)} tracked cell(s){motion_clause}."
            )
        else:
            sentence_1 = (
                f"Himawari-9 Band 13 IR shows {len(storm_cells)} regional cloud feature(s)"
                f"{motion_clause} with LOW convective threat across all 25 Sri Lanka districts."
            )

        # Sentence 2: River basins + 24h rainfall + rising Irrigation Dept gauges
        elevated_basins = [b for b in basins if b.risk_level in ("CRITICAL", "WARNING", "WATCH")]
        rising_gauges = sorted(
            [g for g in gauges if g.trend == "RISING" or g.status in ("ALERT", "MINOR_FLOOD", "MAJOR_FLOOD")],
            key=lambda g: (
                ["NORMAL", "ALERT", "MINOR_FLOOD", "MAJOR_FLOOD"].index(g.status),
                g.rate_of_rise_ft_hr,
            ),
            reverse=True,
        )

        if surges:
            s0 = surges[0]
            b_match = next((b for b in basins if b.basin_id == s0.basin_id), None)
            rain_part = f" ({b_match.rainfall_24h_mm:.0f} mm/24h)" if b_match else ""
            sentence_2 = (
                f"Hydrology alert: upstream surge in {s0.basin_name}{rain_part} at "
                f"{', '.join(s0.upstream_stations)} (+{s0.max_rate_of_rise_ft_hr:.2f} ft/hr); "
                f"monitor {', '.join(s0.downstream_stations[:2])} with crest ETA {s0.estimated_crest_window}."
            )
        elif elevated_basins:
            b0 = elevated_basins[0]
            gauge_note = ""
            if rising_gauges:
                g0 = rising_gauges[0]
                gauge_note = (
                    f" and {g0.station_name} gauge at {g0.water_level_ft:.1f} ft "
                    f"({g0.trend}, {g0.rate_of_rise_ft_hr:+.2f} ft/hr)"
                )
            sentence_2 = (
                f"Hydrology {b0.risk_level} for {b0.name} (ETA {_format_eta(b0.eta_min)}, "
                f"{b0.rainfall_24h_mm:.0f} mm/24h rain, {b0.projected_60m_pct:.0f}% +60m cover)"
                f"{gauge_note}; monitor {', '.join(b0.downstream_stations[:2])}."
            )
        elif rising_gauges:
            g0 = rising_gauges[0]
            sentence_2 = (
                f"Wet-zone basins remain NORMAL, though Irrigation Dept gauge {g0.station_name} "
                f"({g0.river_name}) is {g0.status.replace('_', ' ')} at {g0.water_level_ft:.1f} ft "
                f"({g0.trend}, {g0.rate_of_rise_ft_hr:+.2f} ft/hr)."
            )
        else:
            max_rain_basin = max(basins, key=lambda b: b.rainfall_24h_mm) if basins else None
            rain_clause = (
                f" (highest 24h catchment rain: {max_rain_basin.name} {max_rain_basin.rainfall_24h_mm:.1f} mm)"
                if max_rain_basin
                else ""
            )
            sentence_2 = (
                f"All four wet-zone river basins (Kelani, Kalu, Gin, Nilwala) are at NORMAL flood risk{rain_clause}."
            )

        return f"{sentence_1} {sentence_2}"

    @staticmethod
    def _build_compact_prompt_payload(
        storm_cells: Sequence[ProjectedStormCell],
        districts: Sequence[DistrictThreat],
        basins: Sequence[RiverBasinRisk],
        gauges: Sequence[RiverGaugeReading],
        surges: Sequence[UpstreamSurgeWarning],
    ) -> dict[str, Any]:
        top_cells = sorted(
            storm_cells,
            key=lambda c: (1 if c.label == "deep_convective" else 0, c.speed_kmh),
            reverse=True,
        )[:3]
        active_gauges = [
            g for g in gauges if g.trend == "RISING" or g.status in ("ALERT", "MINOR_FLOOD", "MAJOR_FLOOD")
        ][:4]

        return {
            "storm_cells_total": len(storm_cells),
            "dominant_cells": [
                {
                    "class": c.label,
                    "speed_kmh": c.speed_kmh,
                    "direction": c.compass_dir,
                }
                for c in top_cells
            ],
            "top_districts": [
                {
                    "name": d.name,
                    "threat": d.threat_level,
                    "eta": _format_eta(d.eta_min),
                    "peak_cover_pct": d.peak_coverage_pct,
                    "rain_mm": d.precipitation_mm,
                    "gust_kmh": d.wind_gusts_kmh,
                }
                for d in districts[:3]
            ],
            "top_basins": [
                {
                    "name": b.name,
                    "risk": b.risk_level,
                    "eta": _format_eta(b.eta_min),
                    "cover_now_pct": b.storm_coverage_pct,
                    "cover_60m_pct": b.projected_60m_pct,
                    "rain_24h_mm": b.rainfall_24h_mm,
                    "downstream": b.downstream_stations[:2],
                    "upstream_surge": b.upstream_surge,
                }
                for b in basins[:3]
            ],
            "active_gauges": [
                {
                    "station": g.station_name,
                    "river": g.river_name,
                    "level_ft": g.water_level_ft,
                    "status": g.status,
                    "trend": g.trend,
                    "rate_ft_hr": g.rate_of_rise_ft_hr,
                }
                for g in active_gauges
            ],
            "upstream_surges": [s.message for s in surges[:2]],
        }

    async def generate_advisory(
        self,
        storm_cells: Sequence[ProjectedStormCell],
        districts: Sequence[DistrictThreat],
        basins: Sequence[RiverBasinRisk],
        gauges: Sequence[RiverGaugeReading] = (),
        surges: Sequence[UpstreamSurgeWarning] = (),
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> AdvisoryResult:
        """
        Generates a 2-sentence weather & flood advisory:
          - Immediately returns `source="template"` when disabled or below OLLAMA_MIN_SEVERITY.
          - Otherwise calls CPU-pinned local Ollama `/api/generate` with strict timeout and
            low-RAM/2-thread options, falling back seamlessly to `source="template_fallback"`.
        """
        t0 = time.perf_counter()
        template_text = self.generate_template_advisory(
            storm_cells=storm_cells,
            districts=districts,
            basins=basins,
            gauges=gauges,
            surges=surges,
        )

        if not self.should_invoke_llm(districts, basins, surges):
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            return AdvisoryResult(text=template_text, source="template", model=None, latency_ms=elapsed)

        compact_json = json.dumps(
            self._build_compact_prompt_payload(storm_cells, districts, basins, gauges, surges),
            separators=(",", ":"),
        )
        prompt = (
            "You are a Sri Lanka operational meteorology and hydrology officer. "
            "Write exactly 2 concise, factual plain-English sentences summarizing the highest-risk "
            "districts/river basins, ETA, storm motion, 24h rainfall, and any rising river gauges. "
            "Do not invent figures outside this JSON:\n"
            f"{compact_json}"
        )

        request_body = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "num_ctx": int(self.settings.ollama_num_ctx),
                "num_thread": int(self.settings.ollama_num_thread),
                "num_predict": int(self.settings.ollama_num_predict),
                "temperature": 0.2,
            },
        }

        endpoint = self.settings.ollama_url.rstrip("/") + "/api/generate"
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=float(self.settings.ollama_timeout_sec))
            close_client = True

        try:
            resp = await client.post(
                endpoint,
                json=request_body,
                timeout=float(self.settings.ollama_timeout_sec),
            )
            resp.raise_for_status()
            data = resp.json()
            llm_text = " ".join(str(data.get("response") or "").split()).strip()
            elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
            if llm_text:
                return AdvisoryResult(
                    text=llm_text,
                    source="ollama",
                    model=self.settings.ollama_model,
                    latency_ms=elapsed,
                )
            logger.warning("Ollama returned empty response for %s; using template fallback.", self.settings.ollama_model)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError, ValueError) as exc:
            logger.warning("Ollama advisory generation failed (%s); falling back to template.", exc)
        finally:
            if close_client:
                await client.aclose()

        elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
        return AdvisoryResult(
            text=template_text,
            source="template_fallback",
            model=self.settings.ollama_model,
            latency_ms=elapsed,
        )
