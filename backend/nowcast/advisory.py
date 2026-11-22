from __future__ import annotations

from typing import Sequence

from config import Settings
from models import (
    DistrictThreat,
    ProjectedStormCell,
    RiverBasinRisk,
    RiverGaugeReading,
    UpstreamSurgeWarning,
)
from nowcast.advisory_generator import WeatherAdvisoryGenerator


async def generate_advisory(
    settings: Settings,
    storm_cells: Sequence[ProjectedStormCell],
    districts: Sequence[DistrictThreat],
    basins: Sequence[RiverBasinRisk],
    surges: Sequence[UpstreamSurgeWarning],
    gauges: Sequence[RiverGaugeReading] = (),
) -> tuple[str, str]:
    """Backward-compatible helper delegating to WeatherAdvisoryGenerator."""
    gen = WeatherAdvisoryGenerator(settings)
    res = await gen.generate_advisory(
        storm_cells=storm_cells,
        districts=districts,
        basins=basins,
        gauges=gauges,
        surges=surges,
    )
    return res.text, res.source
