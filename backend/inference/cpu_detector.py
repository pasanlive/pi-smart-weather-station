from __future__ import annotations

import time
import cv2
import numpy as np

from inference.base import StormDetector, pixel_polygon_to_geo, pixel_to_geo
from models import DetectionResult, StormClass, StormDetection


class CPUIRStormDetector(StormDetector):
    """
    Zero-model CPU fallback detector for Himawari-9 Band 13 Clean Infrared frames.
    In Band 13 IR imagery, colder (higher-altitude) convective cloud tops appear brighter
    (higher grayscale intensity). Multi-level brightness-temperature thresholding +
    morphological cleanup + contour extraction segments cells into the exact same
    StormDetection schema produced by Hailo8StormDetector.
    """

    backend_name = "cpu-fallback"

    # (label, min_intensity_0_255, min_area_px, morph_kernel_size)
    LEVELS: list[tuple[StormClass, int, float, int]] = [
        ("deep_convective", 212, 110.0, 3),
        ("cyclonic_band", 178, 220.0, 5),
        ("monsoon_shield", 148, 420.0, 7),
    ]

    def __init__(self, confidence_threshold: float = 0.35, max_cells: int = 24):
        self.confidence_threshold = confidence_threshold
        self.max_cells = max_cells

    def detect(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float] = (75.0, 3.0, 86.0, 12.0),
        confidence_threshold: float | None = None,
    ) -> DetectionResult:
        t0 = time.perf_counter()
        eff_conf_thresh = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        if frame_bgr.ndim == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr.copy()

        h, w = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        detections: list[StormDetection] = []
        occupied_mask = np.zeros((h, w), dtype=np.uint8)
        cell_counter = 0

        for label, thresh, min_area, ksize in self.LEVELS:
            _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
            # Exclude already-segmented higher-severity cores so tiers remain cleanly separated
            if label != "deep_convective":
                binary = cv2.bitwise_and(binary, cv2.bitwise_not(occupied_mask))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < min_area:
                    continue

                epsilon = max(1.8, 0.012 * cv2.arcLength(cnt, True))
                approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
                if len(approx) < 3:
                    continue

                cell_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(cell_mask, [cnt], -1, 255, thickness=-1)
                region_vals = gray[cell_mask > 0]
                if region_vals.size == 0:
                    continue

                peak_val = float(np.percentile(region_vals, 95))
                mean_val = float(np.mean(region_vals))
                # Confidence scales smoothly from threshold margin + structural compactness
                intensity_score = np.clip((mean_val - (thresh - 15.0)) / (255.0 - (thresh - 15.0)), 0.0, 1.0)
                area_score = np.clip(area / (min_area * 6.0), 0.25, 1.0)
                conf = float(np.clip(0.45 + 0.40 * intensity_score + 0.14 * area_score, 0.0, 0.99))
                if conf < eff_conf_thresh:
                    continue

                m = cv2.moments(cnt)
                cx = float(m["m10"] / m["m00"]) if m["m00"] > 0 else float(np.mean(approx[:, 0]))
                cy = float(m["m01"] / m["m00"]) if m["m00"] > 0 else float(np.mean(approx[:, 1]))

                px_poly = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in approx]
                norm_poly = [[round(p[0] / float(w), 5), round(p[1] / float(h), 5)] for p in px_poly]
                geo_poly = pixel_polygon_to_geo(px_poly, w, h, bbox)
                clon, clat = pixel_to_geo(cx, cy, w, h, bbox)

                cell_counter += 1
                detections.append(
                    StormDetection(
                        cell_id=f"cell_{cell_counter}",
                        label=label,
                        confidence=round(conf, 3),
                        pixel_polygon=px_poly,
                        normalized_polygon=norm_poly,
                        geo_polygon=geo_poly,
                        centroid_px=[round(cx, 2), round(cy, 2)],
                        centroid_geo=[clon, clat],
                        area_px=round(area, 1),
                        peak_intensity=round(peak_val, 1),
                    )
                )
                cv2.drawContours(occupied_mask, [cnt], -1, 255, thickness=-1)
                if len(detections) >= self.max_cells:
                    break
            if len(detections) >= self.max_cells:
                break

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return DetectionResult(backend=self.backend_name, inference_ms=elapsed_ms, detections=detections)
