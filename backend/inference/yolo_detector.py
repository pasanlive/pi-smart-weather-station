from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from inference.base import StormDetector, pixel_polygon_to_geo, pixel_to_geo
from models import DetectionResult, StormClass, StormDetection

logger = logging.getLogger(__name__)

VALID_STORM_CLASSES: set[str] = {
    "deep_convective",
    "storm_shield",
    "cyclonic_band",
    "monsoon_shield",
}

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def is_ultralytics_available() -> bool:
    return ULTRALYTICS_AVAILABLE


class YOLOStormDetector(StormDetector):
    """
    Ultralytics YOLOv8-seg PyTorch/ONNX detector for running custom-trained storm
    weights (e.g. `models/storm_yolov8s_seg.pt`) on dev machines (Mac MPS/CPU)
    when Hailo-8 hardware is not attached.
    Validates model class names so generic COCO 80-class base weights are never
    mistakenly executed on satellite imagery.
    """

    backend_name = "yolov8-pt"

    def __init__(self, weights_path: str | Path, confidence_threshold: float = 0.35):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics is not installed in this Python environment.")
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLOv8-seg weights not found at {self.weights_path}")

        self.confidence_threshold = confidence_threshold
        self._model = YOLO(str(self.weights_path))
        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        lower_names = {int(k): str(v).strip().lower() for k, v in names.items()}

        if not lower_names or not set(lower_names.values()).issubset(VALID_STORM_CLASSES):
            sample = list(lower_names.values())[:5]
            raise ValueError(
                f"{self.weights_path.name} is not a storm segmentation model "
                f"(found {len(lower_names)} classes: {sample}; expected subset of {sorted(VALID_STORM_CLASSES)})."
            )

        self._class_map: dict[int, StormClass] = {
            k: v for k, v in lower_names.items() if v in VALID_STORM_CLASSES  # type: ignore[misc]
        }
        logger.info(
            "Initialized YOLOStormDetector: %s (classes=%s)",
            self.weights_path.name,
            self._class_map,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float] = (75.0, 3.0, 86.0, 12.0),
        confidence_threshold: float | None = None,
    ) -> DetectionResult:
        t0 = time.perf_counter()
        eff_conf = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        h, w = frame_bgr.shape[:2]

        results = self._model.predict(
            source=frame_bgr,
            imgsz=640,
            conf=eff_conf,
            iou=0.45,
            verbose=False,
        )

        detections: list[StormDetection] = []
        cell_idx = 0
        if results:
            res = results[0]
            boxes = res.boxes
            masks = res.masks
            if boxes is not None and len(boxes) > 0:
                xy_polys = masks.xy if masks is not None else None
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    label: StormClass = self._class_map.get(cls_id, "storm_shield")

                    if xy_polys is not None and i < len(xy_polys) and len(xy_polys[i]) >= 3:
                        pts = np.asarray(xy_polys[i], dtype=np.float32)
                        cnt = pts.reshape(-1, 1, 2)
                        area = float(cv2.contourArea(cnt))
                        if area < 100.0:
                            continue
                        eps = max(1.5, 0.012 * cv2.arcLength(cnt, True))
                        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
                        if len(approx) < 3:
                            continue
                        px_poly = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in approx]
                        m = cv2.moments(cnt)
                        cx = float(m["m10"] / m["m00"]) if m["m00"] > 0 else float(np.mean(approx[:, 0]))
                        cy = float(m["m01"] / m["m00"]) if m["m00"] > 0 else float(np.mean(approx[:, 1]))
                    else:
                        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                        area = (x2 - x1) * (y2 - y1)
                        if area < 100.0:
                            continue
                        px_poly = [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)], [round(x1, 2), round(y2, 2)]]
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    norm_poly = [[round(p[0] / float(w), 5), round(p[1] / float(h), 5)] for p in px_poly]
                    geo_poly = pixel_polygon_to_geo(px_poly, w, h, bbox)
                    clon, clat = pixel_to_geo(cx, cy, w, h, bbox)
                    cell_idx += 1
                    detections.append(
                        StormDetection(
                            cell_id=f"yolo_{cell_idx}",
                            label=label,
                            confidence=round(conf, 3),
                            pixel_polygon=px_poly,
                            normalized_polygon=norm_poly,
                            geo_polygon=geo_poly,
                            centroid_px=[round(cx, 2), round(cy, 2)],
                            centroid_geo=[clon, clat],
                            area_px=round(area, 1),
                            peak_intensity=round(conf * 255.0, 1),
                        )
                    )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return DetectionResult(backend=self.backend_name, inference_ms=elapsed_ms, detections=detections)
