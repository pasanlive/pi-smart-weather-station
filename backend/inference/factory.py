from __future__ import annotations

import logging
from pathlib import Path

from config import Settings
from inference.base import StormDetector
from inference.cpu_detector import CPUIRStormDetector
from inference.hailo_detector import Hailo8StormDetector, is_hailo_available
from inference.yolo_detector import YOLOStormDetector, is_ultralytics_available

logger = logging.getLogger(__name__)


def _find_trained_pt_candidates(settings: Settings) -> list[Path]:
    candidates: list[Path] = []
    configured = settings.model_pt_resolved_path
    if configured.exists():
        candidates.append(configured)
    models_dir = configured.parent
    if models_dir.exists():
        for pt in sorted(models_dir.glob("*.pt")):
            if pt.resolve() not in {c.resolve() for c in candidates}:
                candidates.append(pt)
    return candidates


def create_storm_detector(settings: Settings) -> StormDetector:
    """
    Selects the active Vision AI engine based on INFERENCE_BACKEND=auto|hailo|yolo|cpu:
      1. Hailo8StormDetector when HailoRT (`hailo_platform`) + compiled `.hef` exist.
      2. YOLOStormDetector when `ultralytics` + a custom-trained storm `.pt` exist
         (validating classes so COCO 80-class base weights are safely skipped).
      3. CPUIRStormDetector fallback otherwise.
    """
    requested = settings.inference_backend
    hef_path = settings.model_hef_resolved_path

    if requested == "cpu":
        logger.info("Inference backend forced to CPU fallback (INFERENCE_BACKEND=cpu)")
        return CPUIRStormDetector(confidence_threshold=settings.confidence_threshold)

    # 1. Production Pi 5 + Hailo-8 AI HAT+ (.hef)
    if requested in ("auto", "hailo"):
        if is_hailo_available() and hef_path.exists():
            try:
                return Hailo8StormDetector(
                    hef_path=hef_path,
                    confidence_threshold=settings.confidence_threshold,
                )
            except Exception as exc:
                logger.warning("Hailo-8 init failed for %s (%s); checking fallbacks.", hef_path, exc)
        elif hef_path.exists() and not is_hailo_available():
            logger.info(
                "Compiled Hailo-8 model found at %s (%.1f MB), ready for Pi 5 deployment "
                "(hailo_platform unavailable on this host).",
                hef_path.name,
                hef_path.stat().st_size / (1024 * 1024),
            )
        else:
            logger.info("Hailo-8 .hef not found at %s.", hef_path)

    # 2. Optional custom-trained PyTorch YOLOv8-seg storm weights (.pt)
    if requested in ("auto", "yolo") and is_ultralytics_available():
        for pt_path in _find_trained_pt_candidates(settings):
            try:
                return YOLOStormDetector(
                    weights_path=pt_path,
                    confidence_threshold=settings.confidence_threshold,
                )
            except Exception as exc:
                logger.info("Skipping %s for storm inference: %s", pt_path.name, exc)

    logger.info("Using CPUIRStormDetector fallback.")
    return CPUIRStormDetector(confidence_threshold=settings.confidence_threshold)
