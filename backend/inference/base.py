from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from models import DetectionResult


def pixel_to_geo(
    x: float,
    y: float,
    width: int,
    height: int,
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Maps 2D image pixel coordinates (x right, y down) to geographic (lon, lat)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon = lon_min + (x / float(width)) * (lon_max - lon_min)
    lat = lat_max - (y / float(height)) * (lat_max - lat_min)
    return round(lon, 5), round(lat, 5)


def pixel_polygon_to_geo(
    pixel_poly: list[list[float]],
    width: int,
    height: int,
    bbox: tuple[float, float, float, float],
) -> list[list[float]]:
    ring = [list(pixel_to_geo(pt[0], pt[1], width, height, bbox)) for pt in pixel_poly]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


class StormDetector(ABC):
    """Abstract pluggable storm segmentation engine (Hailo-8 NPU or OpenCV CPU fallback)."""

    backend_name: str = "abstract"

    @abstractmethod
    def detect(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float] = (75.0, 3.0, 86.0, 12.0),
    ) -> DetectionResult:
        """Runs storm cell segmentation on a 640x640 Himawari-9 Band 13 IR frame."""
        raise NotImplementedError
