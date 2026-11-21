import cv2
import numpy as np

from inference.cpu_detector import CPUIRStormDetector


def test_cpu_detector_segments_three_ir_tiers():
    frame = np.full((640, 640, 3), 30, dtype=np.uint8)
    # 1. Deep convective bright cold cloud top (intensity 245)
    cv2.circle(frame, (320, 320), 28, (245, 245, 245), -1)
    # 2. Separate cyclonic band (intensity 195)
    cv2.ellipse(frame, (180, 220), (45, 22), 25, 0, 360, (195, 195, 195), -1)
    # 3. Separate monsoon shield (intensity 162)
    cv2.rectangle(frame, (430, 420), (540, 510), (162, 162, 162), -1)

    detector = CPUIRStormDetector(confidence_threshold=0.35)
    result = detector.detect(frame, bbox=(75.0, 3.0, 86.0, 12.0))

    assert result.backend == "cpu-fallback"
    assert result.inference_ms >= 0.0
    labels = {d.label for d in result.detections}
    assert "deep_convective" in labels
    assert "cyclonic_band" in labels
    assert "monsoon_shield" in labels

    for det in result.detections:
        assert 0.35 <= det.confidence <= 1.0
        assert len(det.pixel_polygon) >= 3
        assert len(det.geo_polygon) >= 4  # closed GeoJSON ring
        assert det.geo_polygon[0] == det.geo_polygon[-1]
        for nx, ny in det.normalized_polygon:
            assert 0.0 <= nx <= 1.0
            assert 0.0 <= ny <= 1.0
        for lon, lat in det.geo_polygon:
            assert 75.0 <= lon <= 86.0
            assert 3.0 <= lat <= 12.0


def test_cpu_detector_clear_sky_returns_empty():
    clear_frame = np.full((640, 640, 3), 40, dtype=np.uint8)
    detector = CPUIRStormDetector()
    result = detector.detect(clear_frame)
    assert result.detections == []
    assert result.inference_ms >= 0.0
