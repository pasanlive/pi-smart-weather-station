from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from active_learning.sampler import ActiveLearningSampler
from config import settings
from models import RiverBasinRisk, RiverGaugeReading, StormDetection
from storage.db import Database


def _synthetic_ir_frame(seed: int, blobs: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Builds a realistic 640x640 grayscale-like Himawari-9 Band 13 IR image with convective cores."""
    rng = np.random.default_rng(seed)
    base = rng.integers(24, 58, size=(640, 640), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (9, 9), 0)

    # Subtle Sri Lanka coastline silhouette for visual orientation in Band 13 IR
    sl_outline = np.array(
        [[286, 175], [308, 195], [335, 250], [342, 325], [322, 392], [292, 412], [272, 370], [268, 285], [276, 215]],
        dtype=np.int32,
    )
    cv2.polylines(base, [sl_outline], True, 82, 1, cv2.LINE_AA)

    for cx, cy, radius, peak in blobs:
        layer = np.zeros((640, 640), dtype=np.uint8)
        cv2.circle(layer, (cx, cy), radius, peak, -1)
        layer = cv2.GaussianBlur(layer, (25, 25), 0)
        base = np.maximum(base, layer)

    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)


def seed_mock_samples(db_path: Path | None = None) -> list[str]:
    db = Database(db_path or settings.db_path)
    sampler = ActiveLearningSampler(settings=settings, db=db)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    created_ids: list[str] = []

    # Sample 1: Low-Confidence Uncertainty (`low_conf`, conf=0.38)
    ts1 = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame1 = _synthetic_ir_frame(
        seed=101,
        blobs=[
            (305, 345, 58, 192),
            (312, 350, 26, 232),
            (225, 290, 44, 175),
        ],
    )
    det1_a = StormDetection(
        cell_id="hailo_1",
        label="deep_convective",
        confidence=0.38,
        pixel_polygon=[[282, 326], [336, 322], [344, 370], [296, 378], [276, 352]],
        normalized_polygon=[
            [282 / 640, 326 / 640],
            [336 / 640, 322 / 640],
            [344 / 640, 370 / 640],
            [296 / 640, 378 / 640],
            [276 / 640, 352 / 640],
        ],
        geo_polygon=[[79.85, 7.12], [80.78, 7.18], [80.91, 6.50], [80.09, 6.39], [79.74, 6.75], [79.85, 7.12]],
        centroid_px=[310.0, 350.0],
        centroid_geo=[80.32, 6.77],
        area_px=2450.0,
        peak_intensity=232.0,
    )
    det1_b = StormDetection(
        cell_id="hailo_2",
        label="cyclonic_band",
        confidence=0.47,
        pixel_polygon=[[188, 255], [262, 248], [272, 325], [194, 330]],
        normalized_polygon=[[188 / 640, 255 / 640], [262 / 640, 248 / 640], [272 / 640, 325 / 640], [194 / 640, 330 / 640]],
        geo_polygon=[[78.23, 8.11], [79.50, 8.21], [79.68, 7.13], [78.33, 7.06], [78.23, 8.11]],
        centroid_px=[228.0, 290.0],
        centroid_geo=[78.92, 7.62],
        area_px=5400.0,
        peak_intensity=184.0,
    )
    s1 = sampler.maybe_capture(
        frame_bgr=frame1,
        frame_timestamp=ts1,
        detections=[det1_a, det1_b],
        backend_name="hailo-8",
        basin_risks=[
            RiverBasinRisk(basin_id="kelani", name="Kelani Ganga", risk_level="WATCH", deep_convective_pct=18.0, storm_coverage_pct=32.0, rainfall_24h_mm=54.2),
            RiverBasinRisk(basin_id="kalu", name="Kalu Ganga", risk_level="NORMAL", deep_convective_pct=5.0, storm_coverage_pct=14.0, rainfall_24h_mm=31.0),
        ],
        basin_rainfall_1h_mm={"kelani": 11.4, "kalu": 6.2, "gin": 2.1, "nilwala": 1.0},
        basin_rainfall_24h_mm={"kelani": 54.2, "kalu": 31.0, "gin": 18.0, "nilwala": 12.5},
        gauges=[
            RiverGaugeReading(
                station_id="kitulgala",
                station_name="Kitulgala",
                river_name="Kelani Ganga",
                basin_id="kelani",
                role="upstream",
                lat=6.989,
                lon=80.418,
                water_level_ft=11.4,
                alert_level_ft=13.0,
                reading_timestamp=ts1,
                fetched_at=ts1,
                rate_of_rise_ft_hr=0.18,
                trend="RISING",
                status="NORMAL",
            )
        ],
    )
    if s1:
        created_ids.append(s1.id)

    # Sample 2: Cross-Sensor Missed Detection (`sensor_mismatch` in Kalu Ganga)
    ts2 = (now - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame2 = _synthetic_ir_frame(
        seed=202,
        blobs=[
            (318, 376, 62, 200),
            (322, 378, 30, 244),
        ],
    )
    s2 = sampler.maybe_capture(
        frame_bgr=frame2,
        frame_timestamp=ts2,
        detections=[],  # Vision model missed the convective core over Kalu Ganga
        backend_name="hailo-8",
        basin_risks=[
            RiverBasinRisk(basin_id="kalu", name="Kalu Ganga", risk_level="WARNING", deep_convective_pct=0.0, storm_coverage_pct=0.0, rainfall_24h_mm=118.4),
            RiverBasinRisk(basin_id="kelani", name="Kelani Ganga", risk_level="NORMAL", deep_convective_pct=0.0, storm_coverage_pct=0.0, rainfall_24h_mm=28.0),
        ],
        basin_rainfall_1h_mm={"kalu": 26.8, "kelani": 8.5, "gin": 7.4, "nilwala": 3.2},
        basin_rainfall_24h_mm={"kalu": 118.4, "kelani": 28.0, "gin": 44.0, "nilwala": 19.0},
        gauges=[
            RiverGaugeReading(
                station_id="ratnapura",
                station_name="Ratnapura",
                river_name="Kalu Ganga",
                basin_id="kalu",
                role="upstream",
                lat=6.6828,
                lon=80.3992,
                water_level_ft=19.6,
                alert_level_ft=17.0,
                minor_flood_ft=24.0,
                major_flood_ft=31.0,
                rainfall_24h_mm=118.4,
                reading_timestamp=ts2,
                fetched_at=ts2,
                rate_of_rise_ft_hr=0.52,
                trend="RISING",
                status="ALERT",
            ),
            RiverGaugeReading(
                station_id="ellagawa",
                station_name="Ellagawa",
                river_name="Kalu Ganga",
                basin_id="kalu",
                role="downstream",
                lat=6.731,
                lon=80.219,
                water_level_ft=29.2,
                alert_level_ft=32.8,
                reading_timestamp=ts2,
                fetched_at=ts2,
                rate_of_rise_ft_hr=0.14,
                trend="RISING",
                status="NORMAL",
            ),
        ],
    )
    if s2:
        created_ids.append(s2.id)

    # Sample 3: Boundary / Edge-Case Artifact (`boundary_artifact`)
    ts3 = (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame3 = _synthetic_ir_frame(
        seed=303,
        blobs=[
            (32, 415, 70, 185),
            (18, 420, 32, 226),
            (380, 290, 48, 165),
        ],
    )
    det3_edge = StormDetection(
        cell_id="hailo_edge_1",
        label="monsoon_shield",
        confidence=0.22,
        pixel_polygon=[[0, 355], [76, 368], [92, 452], [44, 488], [0, 476]],
        normalized_polygon=[[0.0, 355 / 640], [76 / 640, 368 / 640], [92 / 640, 452 / 640], [44 / 640, 488 / 640], [0.0, 476 / 640]],
        geo_polygon=[[75.0, 7.01], [76.31, 6.82], [76.58, 5.64], [75.76, 5.14], [75.0, 5.31], [75.0, 7.01]],
        centroid_px=[38.0, 422.0],
        centroid_geo=[75.65, 6.06],
        area_px=8900.0,
        peak_intensity=195.0,
    )
    s3 = sampler.maybe_capture(
        frame_bgr=frame3,
        frame_timestamp=ts3,
        detections=[det3_edge],
        backend_name="hailo-8",
        basin_risks=[
            RiverBasinRisk(basin_id="gin", name="Gin Ganga", risk_level="NORMAL", deep_convective_pct=0.0, storm_coverage_pct=8.0, rainfall_24h_mm=22.0),
        ],
        basin_rainfall_1h_mm={"gin": 4.2, "kalu": 3.0, "kelani": 1.8, "nilwala": 2.4},
        basin_rainfall_24h_mm={"gin": 22.0, "kalu": 26.0, "kelani": 14.0, "nilwala": 16.0},
        gauges=[],
    )
    if s3:
        created_ids.append(s3.id)

    print(f"Seeded {len(created_ids)} active-learning pending samples: {created_ids}")
    return created_ids


if __name__ == "__main__":
    seed_mock_samples()
