from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from inference.base import StormDetector
from inference.cpu_detector import CPUIRStormDetector
from inference.hailo_detector import Hailo8StormDetector, is_hailo_available

logger = logging.getLogger(__name__)


def create_storm_detector(settings: Settings) -> StormDetector:
    """
    Selects Hailo8StormDetector or CPUIRStormDetector based on INFERENCE_BACKEND=auto|hailo|cpu,
    falling back gracefully to CPUIRStormDetector if HailoRT or the .hef model is unavailable.
    """
    requested = settings.inference_backend
    hef_path = Path(settings.model_hef_path)

    if requested == "cpu":
        logger.info("Inference backend forced to CPU fallback (INFERENCE_BACKEND=cpu)")
        return CPUIRStormDetector(confidence_threshold=settings.confidence_threshold)

    if is_hailo_available() and hef_path.exists():
        try:
            return Hailo8StormDetector(
                hef_path=hef_path,
                confidence_threshold=settings.confidence_threshold,
            )
        except Exception as exc:
            if requested == "hailo":
                logger.error("Requested Hailo-8 backend failed to initialize (%s); falling back to CPU.", exc)
            else:
                logger.warning("Hailo-8 auto-init failed (%s); falling back to CPU.", exc)
    else:
        reason = "hailo_platform not installed" if not is_hailo_available() else f".hef not found at {hef_path}"
        logger.info("Using CPUIRStormDetector fallback (%s)", reason)

    return CPUIRStormDetector(confidence_threshold=settings.confidence_threshold)
