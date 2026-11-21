from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from inference.cpu_detector import CPUIRStormDetector
from nowcast.nowcast_engine import NowcastEngine, bearing_to_compass_16

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DISTRICTS_PATH = DATA_DIR / "sri_lanka_districts.geojson"
BASINS_PATH = DATA_DIR / "sri_lanka_flood_basins.geojson"


def _make_textured_blob_frame(cx: int, cy: int, radius: int = 34) -> np.ndarray:
    """Creates a realistic textured cold IR cloud blob so Farneback optical flow locks cleanly."""
    frame = np.full((640, 640), 25, dtype=np.uint8)
    yy, xx = np.ogrid[:640, :640]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # Smooth Gaussian convective core + internal texture ripples
    core = np.exp(-0.5 * (dist / (radius * 0.65)) ** 2) * 225.0
    texture = (np.sin((xx - cx) * 0.35) * np.cos((yy - cy) * 0.35)) * 18.0 * (dist < radius)
    vals = np.clip(25.0 + core + texture, 0, 255).astype(np.uint8)
    return cv2.cvtColor(vals, cv2.COLOR_GRAY2BGR)


def test_bearing_to_compass_16_points():
    assert bearing_to_compass_16(0.0) == "N"
    assert bearing_to_compass_16(90.0) == "E"
    assert bearing_to_compass_16(180.0) == "S"
    assert bearing_to_compass_16(270.0) == "W"
    assert bearing_to_compass_16(45.0) == "NE"
    assert bearing_to_compass_16(292.5) == "WNW"


def test_optical_flow_recovers_eastward_speed_and_bearing():
    """
    Shift a synthetic storm blob +6 pixels east every 10 minutes across 3 frames.
    In BBOX (75..86E, 3..12N) over 640px:
      6 px / 10 min = 0.6 px/min -> ~0.01031 deg/min -> ~0.6188 deg/hr -> ~68.4 km/h East (bearing ~90 deg).
    """
    frames = [
        ("2026-11-21T10:00:00Z", _make_textured_blob_frame(240, 360)),
        ("2026-11-21T10:10:00Z", _make_textured_blob_frame(246, 360)),
        ("2026-11-21T10:20:00Z", _make_textured_blob_frame(252, 360)),
    ]

    detector = CPUIRStormDetector(confidence_threshold=0.35)
    detections = detector.detect(frames[-1][1]).detections
    assert len(detections) >= 1

    engine = NowcastEngine(DISTRICTS_PATH, BASINS_PATH)
    out = engine.run(frames, detections)

    deep_cells = [c for c in out.storm_cells if c.label == "deep_convective"]
    assert len(deep_cells) >= 1
    cell = deep_cells[0]

    assert 40.0 <= cell.speed_kmh <= 95.0
    assert 70.0 <= cell.bearing_deg <= 110.0
    assert cell.compass_dir in ("ENE", "E", "ESE")
    assert set(cell.geometry_projected.keys()) == {"+30m", "+60m", "+120m"}


def test_projected_trajectories_trigger_district_and_basin_eta():
    """
    Place a storm cell offshore west of Colombo / Kelani Ganga (around 79.45E, 6.95N)
    with an eastward flow such that:
      - At 0m: offshore (0% over Kelani / Colombo)
      - At +30m or +60m: arrives over Colombo & Kelani Ganga triggering ETA = 30 or 60.
    """
    engine = NowcastEngine(DISTRICTS_PATH, BASINS_PATH)

    # Storm polygon currently offshore west of Colombo (79.15..79.55E, 6.80..7.10N)
    # Moving east at +0.012 deg/min -> +0.36 deg at +30m (79.51..79.91E), +0.72 deg at +60m (79.87..80.27E, squarely over Colombo & Kelani!)
    lon_min, lat_min, lon_max, lat_max = engine.bbox
    w, h = 640, 640

    def geo_to_px(lon: float, lat: float) -> list[float]:
        return [((lon - lon_min) / (lon_max - lon_min)) * w, ((lat_max - lat) / (lat_max - lat_min)) * h]

    geo_ring = [
        [79.15, 6.80],
        [79.55, 6.80],
        [79.55, 7.12],
        [79.15, 7.12],
        [79.15, 6.80],
    ]
    px_poly = [geo_to_px(lon, lat) for lon, lat in geo_ring[:-1]]

    from models import StormDetection

    det = StormDetection(
        cell_id="test_storm_1",
        label="deep_convective",
        confidence=0.92,
        pixel_polygon=px_poly,
        normalized_polygon=[[p[0] / w, p[1] / h] for p in px_poly],
        geo_polygon=geo_ring,
        centroid_px=geo_to_px(79.35, 6.96),
        centroid_geo=[79.35, 6.96],
        area_px=600.0,
        peak_intensity=245.0,
    )

    # Constant eastward optical flow: dlon_per_min = 0.012 -> dx_px_per_min = 0.012 * 640 / 11.0 = 0.698 px/min
    flow = np.zeros((640, 640, 2), dtype=np.float32)
    flow[:, :, 0] = 0.698
    flow[:, :, 1] = 0.0

    projected, horizon_polys = engine.project_storms([det], flow)
    assert len(projected) == 1
    assert projected[0].compass_dir == "E"
    assert projected[0].speed_kmh <= 120.0

    districts, basin_covs = engine._evaluate_zones(horizon_polys, [])
    colombo = next(d for d in districts if d.name == "Colombo")
    kelani = basin_covs["kelani"]

    assert colombo.affected_area_pct == 0.0
    assert colombo.eta_min in (30, 60)
    assert colombo.threat_level in ("HIGH", "SEVERE")

    assert kelani.storm_coverage_pct == 0.0
    assert kelani.projected_60m_pct > 25.0
    assert kelani.eta_min == 60


def test_speed_cap_and_unary_union_no_double_counting():
    engine = NowcastEngine(DISTRICTS_PATH, BASINS_PATH)
    # Create two identical overlapping polygons covering all of Colombo
    colombo_poly = next(z.geom for z in engine.districts if z.name == "Colombo")
    _, basin_covs = engine._evaluate_zones(
        {0: [("deep_convective", colombo_poly), ("deep_convective", colombo_poly)], 30: [], 60: [], 120: []},
        [],
    )
    districts, _ = engine._evaluate_zones(
        {0: [("deep_convective", colombo_poly), ("deep_convective", colombo_poly)], 30: [], 60: [], 120: []},
        [],
    )
    colombo = next(d for d in districts if d.name == "Colombo")
    assert colombo.affected_area_pct == 100.0
    assert colombo.eta_min == 0
    assert isinstance(basin_covs, dict)

    # Verify extreme flow is capped at 120 km/h
    from models import StormDetection

    ring = [[80.0, 7.0], [80.2, 7.0], [80.2, 7.2], [80.0, 7.2], [80.0, 7.0]]
    det = StormDetection(
        cell_id="fast_cell",
        label="cyclonic_band",
        confidence=0.8,
        pixel_polygon=[[280, 280], [310, 280], [310, 310], [280, 310]],
        normalized_polygon=[[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]],
        geo_polygon=ring,
        centroid_px=[295, 295],
        centroid_geo=[80.1, 7.1],
        area_px=900.0,
    )
    extreme_flow = np.full((640, 640, 2), 15.0, dtype=np.float32)
    cells, _ = engine.project_storms([det], extreme_flow)
    assert cells[0].speed_kmh == 120.0
