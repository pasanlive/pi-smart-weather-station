from __future__ import annotations

import numpy as np

from config import Settings
from inference.hailo_detector import decode_yolov8_seg_10_outputs


def _dfl_one_hot(dist_bin: int) -> np.ndarray:
    """Produces raw DFL logits peaking sharply at `dist_bin` (0..15)."""
    logits = np.full(16, -8.0, dtype=np.float32)
    logits[dist_bin] = 8.0
    return logits


def test_decode_yolov8_seg_10_outputs_reconstructs_storm_polygon():
    """
    Constructs a synthetic 10-tensor Hailo-8 YOLOv8s-seg output dictionary matching the
    exact layers compiled inside `models/storm_yolov8s_seg.hef` (P3 80x80, P4 40x40,
    P5 20x20 box/cls/mask + 160x160x32 proto) and verifies DFL box decoding, NMS,
    prototype mask reconstruction, and 2-class storm label mapping.
    """
    # Empty background tensors with strong negative class logits
    p3_box = np.zeros((1, 80, 80, 64), dtype=np.float32)
    p3_cls = np.full((1, 80, 80, 2), -6.0, dtype=np.float32)
    p3_mask = np.zeros((1, 80, 80, 32), dtype=np.float32)

    p4_box = np.zeros((1, 40, 40, 64), dtype=np.float32)
    p4_cls = np.full((1, 40, 40, 2), -6.0, dtype=np.float32)
    p4_mask = np.zeros((1, 40, 40, 32), dtype=np.float32)

    p5_box = np.zeros((1, 20, 20, 64), dtype=np.float32)
    p5_cls = np.full((1, 20, 20, 2), -6.0, dtype=np.float32)
    p5_mask = np.zeros((1, 20, 20, 32), dtype=np.float32)

    # Prototype 0 is positive inside a central storm region (60..100, 60..100 in 160x160 proto space)
    proto = np.full((1, 160, 160, 32), -4.0, dtype=np.float32)
    proto[0, 62:98, 62:98, 0] = 6.0

    # Place a high-confidence deep_convective (class 0) detection at P4 grid cell (20, 20) -> center (328, 328)
    # DFL distance = 4 cells * stride 16 = 64 px in all 4 directions -> box [264..392, 264..392]
    dfl4 = _dfl_one_hot(4)
    p4_box[0, 20, 20, :] = np.concatenate([dfl4, dfl4, dfl4, dfl4])
    p4_cls[0, 20, 20, 0] = 3.0  # sigmoid(3.0) ~= 0.953 -> deep_convective
    p4_mask[0, 20, 20, 0] = 1.0  # activate prototype 0

    # Place a storm_shield (class 1) detection at P5 grid cell (5, 5) -> center (176, 176)
    proto[0, 32:56, 32:56, 1] = 6.0
    dfl2 = _dfl_one_hot(2)
    p5_box[0, 5, 5, :] = np.concatenate([dfl2, dfl2, dfl2, dfl2])
    p5_cls[0, 5, 5, 1] = 2.0  # sigmoid(2.0) ~= 0.881 -> storm_shield
    p5_mask[0, 5, 5, 1] = 1.0

    raw_outputs = {
        "storm_yolov8s_seg/conv44": p3_box,
        "storm_yolov8s_seg/conv45": p3_cls,
        "storm_yolov8s_seg/conv46": p3_mask,
        "storm_yolov8s_seg/conv60": p4_box,
        "storm_yolov8s_seg/conv61": p4_cls,
        "storm_yolov8s_seg/conv62": p4_mask,
        "storm_yolov8s_seg/conv73": p5_box,
        "storm_yolov8s_seg/conv74": p5_cls,
        "storm_yolov8s_seg/conv75": p5_mask,
        "storm_yolov8s_seg/conv48": proto,
    }

    dets = decode_yolov8_seg_10_outputs(raw_outputs, confidence_threshold=0.35)
    assert len(dets) == 2

    labels = {d.label for d in dets}
    assert labels == {"deep_convective", "storm_shield"}

    deep = next(d for d in dets if d.label == "deep_convective")
    assert deep.confidence > 0.9
    assert 260.0 <= deep.centroid_px[0] <= 390.0
    assert 260.0 <= deep.centroid_px[1] <= 390.0
    assert len(deep.geo_polygon) >= 4
    assert deep.geo_polygon[0] == deep.geo_polygon[-1]


def test_model_hef_resolved_path_finds_repo_models_file():
    s = Settings(model_hef_path="./models/storm_yolov8s_seg.hef")
    resolved = s.model_hef_resolved_path
    assert resolved.is_absolute()
    assert resolved.name == "storm_yolov8s_seg.hef"
    assert resolved.exists()
    assert resolved.stat().st_size > 1_000_000
