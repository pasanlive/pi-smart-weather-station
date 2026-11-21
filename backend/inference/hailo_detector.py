from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference.base import StormDetector, pixel_polygon_to_geo, pixel_to_geo
from models import DetectionResult, StormClass, StormDetection

logger = logging.getLogger(__name__)

CLASS_LABELS: dict[int, StormClass] = {
    0: "deep_convective",
    1: "cyclonic_band",
    2: "monsoon_shield",
}

try:
    from hailo_platform import (
        ConfigureParams,
        FormatType,
        HEF,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )

    HAILO_PLATFORM_AVAILABLE = True
except ImportError:
    HAILO_PLATFORM_AVAILABLE = False


def is_hailo_available() -> bool:
    return HAILO_PLATFORM_AVAILABLE


class Hailo8StormDetector(StormDetector):
    """
    Production Vision AI detector targeting Raspberry Pi 5 + Hailo-8 AI HAT+ (26 TOPS).
    Executes compiled INT8 YOLOv8-seg / segmentation `.hef` models over PCIe using HailoRT.
    """

    backend_name = "hailo-8"

    def __init__(self, hef_path: str | Path, confidence_threshold: float = 0.35):
        if not HAILO_PLATFORM_AVAILABLE:
            raise RuntimeError("hailo_platform (HailoRT) is not installed in this Python environment.")
        self.hef_path = Path(hef_path)
        if not self.hef_path.exists():
            raise FileNotFoundError(f"Compiled Hailo .hef model not found at {self.hef_path}")

        self.confidence_threshold = confidence_threshold
        self._hef = HEF(str(self.hef_path))
        self._vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=self._hef,
            interface=HailoStreamInterface.PCIe,
        )
        network_groups = self._vdevice.configure(self._hef, configure_params)
        self._network_group = network_groups[0]
        self._network_group_params = self._network_group.create_params()
        self._input_vstream_infos = self._hef.get_input_vstream_infos()
        self._output_vstream_infos = self._hef.get_output_vstream_infos()
        self._input_params = InputVStreamParams.make_from_network_group(
            self._network_group,
            quantized=True,
            format_type=FormatType.UINT8,
        )
        self._output_params = OutputVStreamParams.make_from_network_group(
            self._network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )
        input_shape = self._input_vstream_infos[0].shape
        self.input_h = int(input_shape[0])
        self.input_w = int(input_shape[1])
        logger.info(
            "Initialized Hailo8StormDetector on PCIe: %s (input %dx%d)",
            self.hef_path.name,
            self.input_w,
            self.input_h,
        )

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self.input_h or rgb.shape[1] != self.input_w:
            rgb = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        return np.expand_dims(rgb.astype(np.uint8), axis=0)

    def _decode_outputs(
        self,
        raw_outputs: dict[str, Any],
        frame_shape: tuple[int, int],
        bbox: tuple[float, float, float, float],
        confidence_threshold: float | None = None,
    ) -> list[StormDetection]:
        """
        Decodes either:
          - Semantic segmentation mask tensor (1, H, W, C) / (1, C, H, W), OR
          - YOLOv8-seg NMS box/mask tensors into normalized StormDetection polygons.
        """
        eff_conf_thresh = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        orig_h, orig_w = frame_shape
        detections: list[StormDetection] = []
        cell_idx = 0

        for tensor in raw_outputs.values():
            arr = np.asarray(tensor)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            # Semantic mask map (H, W, C) where C >= 3
            if arr.ndim == 3 and arr.shape[2] >= 3 and arr.shape[0] >= 32 and arr.shape[1] >= 32:
                prob_map = arr
                for cls_id, label in CLASS_LABELS.items():
                    if cls_id >= prob_map.shape[2]:
                        continue
                    cls_probs = prob_map[:, :, cls_id]
                    mask = (cls_probs >= eff_conf_thresh).astype(np.uint8) * 255
                    if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
                        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        cls_probs_resized = cv2.resize(cls_probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        cls_probs_resized = cls_probs
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = float(cv2.contourArea(cnt))
                        if area < 120.0:
                            continue
                        eps = max(1.5, 0.012 * cv2.arcLength(cnt, True))
                        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
                        if len(approx) < 3:
                            continue
                        px_poly = [[float(p[0]), float(p[1])] for p in approx]
                        norm_poly = [[round(p[0] / orig_w, 5), round(p[1] / orig_h, 5)] for p in px_poly]
                        geo_poly = pixel_polygon_to_geo(px_poly, orig_w, orig_h, bbox)
                        m = cv2.moments(cnt)
                        cx = float(m["m10"] / m["m00"]) if m["m00"] > 0 else float(np.mean(approx[:, 0]))
                        cy = float(m["m01"] / m["m00"]) if m["m00"] > 0 else float(np.mean(approx[:, 1]))
                        clon, clat = pixel_to_geo(cx, cy, orig_w, orig_h, bbox)
                        cell_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                        cv2.drawContours(cell_mask, [cnt], -1, 255, thickness=-1)
                        conf = float(np.clip(np.mean(cls_probs_resized[cell_mask > 0]), 0.0, 1.0))
                        cell_idx += 1
                        detections.append(
                            StormDetection(
                                cell_id=f"hailo_{cell_idx}",
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
                break
        return detections

    def detect(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float] = (75.0, 3.0, 86.0, 12.0),
        confidence_threshold: float | None = None,
    ) -> DetectionResult:
        t0 = time.perf_counter()
        input_name = self._input_vstream_infos[0].name
        input_tensor = self._preprocess(frame_bgr)
        with InferVStreams(self._network_group, self._input_params, self._output_params) as infer_pipeline:
            with self._network_group.activate(self._network_group_params):
                raw_outputs = infer_pipeline.infer({input_name: input_tensor})

        h, w = frame_bgr.shape[:2]
        detections = self._decode_outputs(raw_outputs, (h, w), bbox, confidence_threshold=confidence_threshold)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return DetectionResult(backend=self.backend_name, inference_ms=elapsed_ms, detections=detections)
