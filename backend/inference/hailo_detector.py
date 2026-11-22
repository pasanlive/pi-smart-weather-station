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

# Matches ml_pipeline/config.yaml 2-class storm_yolov8s_seg.hef (0=deep_convective, 1=storm_shield)
# and gracefully supports 3-class models (0=deep_convective, 1=cyclonic_band, 2=monsoon_shield).
TWO_CLASS_LABELS: dict[int, StormClass] = {
    0: "deep_convective",
    1: "storm_shield",
}

THREE_CLASS_LABELS: dict[int, StormClass] = {
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    """Normalizes a 3D/4D tensor into (H, W, C) float32."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (2, 3, 32, 64) and a.shape[1] == a.shape[2] and a.shape[1] in (20, 40, 80, 160):
        # Convert NCHW (C, H, W) -> NHWC (H, W, C)
        a = np.transpose(a, (1, 2, 0))
    return a


def decode_yolov8_seg_10_outputs(
    raw_outputs: dict[str, Any],
    *,
    input_size: tuple[int, int] = (640, 640),
    frame_shape: tuple[int, int] = (640, 640),
    bbox: tuple[float, float, float, float] = (75.0, 3.0, 86.0, 12.0),
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    max_detections: int = 40,
    min_area_px: float = 100.0,
) -> list[StormDetection]:
    """
    Decodes the standard Hailo-8 YOLOv8s-seg 10-output split compiled by `ml_pipeline/5_compile_hailo8.py`:
      - P3 (80x80, stride 8):  box DFL (64ch), class logits (nc), mask coeffs (32ch)
      - P4 (40x40, stride 16): box DFL (64ch), class logits (nc), mask coeffs (32ch)
      - P5 (20x20, stride 32): box DFL (64ch), class logits (nc), mask coeffs (32ch)
      - Proto (160x160x32):    32 spatial mask prototypes
    Returns normalized `StormDetection` polygons matching the shared backend schema.
    """
    in_h, in_w = input_size
    orig_h, orig_w = frame_shape

    proto: np.ndarray | None = None
    scales: dict[int, dict[str, np.ndarray]] = {}

    for tensor in raw_outputs.values():
        arr = _to_hwc(tensor)
        if arr.ndim != 3:
            continue
        h, w, c = arr.shape
        if h == w == 160 and c == 32:
            proto = arr
            continue
        if h == w and h in (80, 40, 20):
            bucket = scales.setdefault(h, {})
            if c == 64:
                bucket["box"] = arr
            elif c == 32:
                bucket["mask"] = arr
            elif 1 <= c <= 16:
                bucket["cls"] = arr

    if proto is None or not scales:
        return []

    dfl_bins = np.arange(16, dtype=np.float32)
    cand_boxes_xyxy: list[np.ndarray] = []
    cand_scores: list[np.ndarray] = []
    cand_classes: list[np.ndarray] = []
    cand_masks: list[np.ndarray] = []
    num_classes = 2

    for grid_size in sorted(scales.keys(), reverse=True):
        bucket = scales[grid_size]
        if "box" not in bucket or "cls" not in bucket or "mask" not in bucket:
            continue

        stride = float(in_w) / float(grid_size)
        cls_raw = bucket["cls"]
        num_classes = cls_raw.shape[2]
        # Raw Conv logits -> sigmoid probabilities
        cls_probs = _sigmoid(cls_raw) if (cls_raw.min() < 0.0 or cls_raw.max() > 1.0) else cls_raw
        cls_flat = cls_probs.reshape(-1, num_classes)
        best_cls = np.argmax(cls_flat, axis=1)
        best_score = cls_flat[np.arange(cls_flat.shape[0]), best_cls]

        keep_idx = np.where(best_score >= confidence_threshold)[0]
        if keep_idx.size == 0:
            continue

        box_flat = bucket["box"].reshape(-1, 4, 16)[keep_idx]
        dfl_dist = np.sum(_softmax(box_flat, axis=-1) * dfl_bins, axis=-1)  # (K, 4) -> l, t, r, b

        gy = (keep_idx // grid_size).astype(np.float32)
        gx = (keep_idx % grid_size).astype(np.float32)
        x1 = (gx + 0.5 - dfl_dist[:, 0]) * stride
        y1 = (gy + 0.5 - dfl_dist[:, 1]) * stride
        x2 = (gx + 0.5 + dfl_dist[:, 2]) * stride
        y2 = (gy + 0.5 + dfl_dist[:, 3]) * stride

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(in_w))
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(in_h))

        mask_flat = bucket["mask"].reshape(-1, 32)[keep_idx]

        cand_boxes_xyxy.append(boxes)
        cand_scores.append(best_score[keep_idx])
        cand_classes.append(best_cls[keep_idx])
        cand_masks.append(mask_flat)

    if not cand_boxes_xyxy:
        return []

    all_boxes = np.concatenate(cand_boxes_xyxy, axis=0)
    all_scores = np.concatenate(cand_scores, axis=0)
    all_classes = np.concatenate(cand_classes, axis=0)
    all_masks = np.concatenate(cand_masks, axis=0)

    # Run NMS via OpenCV (xywh format)
    xywh = np.column_stack(
        [
            all_boxes[:, 0],
            all_boxes[:, 1],
            np.maximum(1.0, all_boxes[:, 2] - all_boxes[:, 0]),
            np.maximum(1.0, all_boxes[:, 3] - all_boxes[:, 1]),
        ]
    ).tolist()
    nms_indices = cv2.dnn.NMSBoxes(
        bboxes=xywh,
        scores=all_scores.tolist(),
        score_threshold=float(confidence_threshold),
        nms_threshold=float(iou_threshold),
    )
    if len(nms_indices) == 0:
        return []

    kept = np.asarray(nms_indices).reshape(-1)
    # Sort descending by confidence and cap
    kept = kept[np.argsort(-all_scores[kept])][:max_detections]

    label_map = TWO_CLASS_LABELS if num_classes == 2 else THREE_CLASS_LABELS
    proto_h, proto_w = proto.shape[:2]
    proto_flat = proto.reshape(-1, 32)  # (160*160, 32)

    detections: list[StormDetection] = []
    cell_idx = 0
    sx = float(orig_w) / float(in_w)
    sy = float(orig_h) / float(in_h)
    px_scale = float(proto_w) / float(in_w)
    py_scale = float(proto_h) / float(in_h)

    for k in kept:
        cls_id = int(all_classes[k])
        label: StormClass = label_map.get(cls_id, "storm_shield")
        conf = float(np.clip(all_scores[k], 0.0, 1.0))
        bx1, by1, bx2, by2 = all_boxes[k]

        # Reconstruct spatial mask: sigmoid(proto @ coeffs) cropped to bounding box
        mask_logits = (proto_flat @ all_masks[k]).reshape(proto_h, proto_w)
        mask_prob = _sigmoid(mask_logits)

        mx1 = max(0, int(np.floor(bx1 * px_scale)))
        my1 = max(0, int(np.floor(by1 * py_scale)))
        mx2 = min(proto_w, int(np.ceil(bx2 * px_scale)))
        my2 = min(proto_h, int(np.ceil(by2 * py_scale)))
        if mx2 <= mx1 or my2 <= my1:
            continue

        cropped = np.zeros_like(mask_prob, dtype=np.float32)
        cropped[my1:my2, mx1:mx2] = mask_prob[my1:my2, mx1:mx2]

        resized_mask = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (resized_mask >= 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback to bounding-box polygon if mask threshold yielded tiny pixels
            rx1, ry1, rx2, ry2 = bx1 * sx, by1 * sy, bx2 * sx, by2 * sy
            if (rx2 - rx1) * (ry2 - ry1) < min_area_px:
                continue
            px_poly = [[round(rx1, 2), round(ry1, 2)], [round(rx2, 2), round(ry1, 2)], [round(rx2, 2), round(ry2, 2)], [round(rx1, 2), round(ry2, 2)]]
            area = float((rx2 - rx1) * (ry2 - ry1))
            cx, cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
        else:
            cnt = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(cnt))
            if area < min_area_px:
                continue
            eps = max(1.5, 0.012 * cv2.arcLength(cnt, True))
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            px_poly = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in approx]
            m = cv2.moments(cnt)
            cx = float(m["m10"] / m["m00"]) if m["m00"] > 0 else float(np.mean(approx[:, 0]))
            cy = float(m["m01"] / m["m00"]) if m["m00"] > 0 else float(np.mean(approx[:, 1]))

        norm_poly = [[round(p[0] / float(orig_w), 5), round(p[1] / float(orig_h), 5)] for p in px_poly]
        geo_poly = pixel_polygon_to_geo(px_poly, orig_w, orig_h, bbox)
        clon, clat = pixel_to_geo(cx, cy, orig_w, orig_h, bbox)

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

    return detections


class Hailo8StormDetector(StormDetector):
    """
    Production Vision AI detector targeting Raspberry Pi 5 + Hailo-8 AI HAT+ (26 TOPS).
    Executes compiled INT8 YOLOv8s-seg `.hef` models (e.g. `models/storm_yolov8s_seg.hef`)
    over PCIe using HailoRT and decodes the 10-output YOLOv8s-seg split (P3/P4/P5 box/cls/mask + proto).
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
            "Initialized Hailo8StormDetector on PCIe: %s (input=%dx%d, outputs=%d)",
            self.hef_path.name,
            self.input_w,
            self.input_h,
            len(self._output_vstream_infos),
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
        eff_conf = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)

        # Primary path: 10-output Hailo-8 YOLOv8s-seg split (P3/P4/P5 box/cls/mask + proto)
        if len(raw_outputs) >= 7:
            return decode_yolov8_seg_10_outputs(
                raw_outputs,
                input_size=(self.input_h, self.input_w),
                frame_shape=frame_shape,
                bbox=bbox,
                confidence_threshold=eff_conf,
            )

        # Fallback path: direct semantic probability mask tensor (H, W, C)
        orig_h, orig_w = frame_shape
        detections: list[StormDetection] = []
        cell_idx = 0
        for tensor in raw_outputs.values():
            arr = _to_hwc(tensor)
            if arr.ndim == 3 and arr.shape[2] >= 2 and arr.shape[0] >= 32 and arr.shape[1] >= 32:
                label_map = TWO_CLASS_LABELS if arr.shape[2] == 2 else THREE_CLASS_LABELS
                for cls_id, label in label_map.items():
                    if cls_id >= arr.shape[2]:
                        continue
                    cls_probs = arr[:, :, cls_id]
                    mask = (cls_probs >= eff_conf).astype(np.uint8) * 255
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
        detections = self._decode_outputs(
            raw_outputs,
            (h, w),
            bbox,
            confidence_threshold=confidence_threshold,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return DetectionResult(backend=self.backend_name, inference_ms=elapsed_ms, detections=detections)
