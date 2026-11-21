from __future__ import annotations

import math
from typing import Optional

from models import (
    ActiveLearningPolygon,
    ID_TO_STORM_CLASS,
    STORM_CLASS_TO_ID,
    StormDetection,
)


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def detections_to_yolo_seg(
    detections: list[StormDetection],
    width: int = 640,
    height: int = 640,
) -> str:
    """
    Formats StormDetection objects into standard Ultralytics YOLOv8-seg polygon lines:
        <class_id> x1 y1 x2 y2 ... xn yn
    with normalized coordinates clamped to [0.0, 1.0].
    """
    lines: list[str] = []
    w = float(max(1, width))
    h = float(max(1, height))

    for det in detections:
        class_id = STORM_CLASS_TO_ID.get(det.label)
        if class_id is None:
            continue

        if det.normalized_polygon and len(det.normalized_polygon) >= 3:
            pts = [[_clamp01(p[0]), _clamp01(p[1])] for p in det.normalized_polygon if len(p) >= 2]
        elif det.pixel_polygon and len(det.pixel_polygon) >= 3:
            pts = [[_clamp01(p[0] / w), _clamp01(p[1] / h)] for p in det.pixel_polygon if len(p) >= 2]
        else:
            continue

        # Remove duplicate closing vertex if present (YOLOv8-seg expects open vertex ring)
        if len(pts) > 3 and math.isclose(pts[0][0], pts[-1][0], abs_tol=1e-6) and math.isclose(pts[0][1], pts[-1][1], abs_tol=1e-6):
            pts = pts[:-1]

        if len(pts) < 3:
            continue

        coord_tokens = [f"{x:.6f} {y:.6f}" for x, y in pts]
        lines.append(f"{class_id} " + " ".join(coord_tokens))

    return "\n".join(lines) + ("\n" if lines else "")


def validate_yolo_seg(label_txt: str) -> tuple[bool, Optional[str]]:
    """
    Validates YOLOv8-seg label text. Empty text is valid (background / clear-sky frame).
    Each non-empty line must be: <class_id> x1 y1 x2 y2 x3 y3 ... (even count >= 6 floats in [0, 1]).
    """
    if not label_txt or not label_txt.strip():
        return True, None

    for line_no, raw_line in enumerate(label_txt.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            return False, f"Line {line_no}: expected class_id + at least 3 (x, y) vertices"
        try:
            class_id = int(parts[0])
        except ValueError:
            return False, f"Line {line_no}: class_id '{parts[0]}' is not an integer"
        if class_id not in ID_TO_STORM_CLASS:
            return False, f"Line {line_no}: class_id {class_id} out of valid range {sorted(ID_TO_STORM_CLASS.keys())}"

        coords = parts[1:]
        if len(coords) % 2 != 0:
            return False, f"Line {line_no}: odd number of coordinate values ({len(coords)})"

        for idx, token in enumerate(coords):
            try:
                val = float(token)
            except ValueError:
                return False, f"Line {line_no}: coordinate #{idx + 1} ('{token}') is not a float"
            if math.isnan(val) or math.isinf(val) or val < 0.0 or val > 1.0:
                return False, f"Line {line_no}: coordinate #{idx + 1} ({val}) must be within [0.0, 1.0]"

    return True, None


def parse_yolo_seg(label_txt: str) -> list[ActiveLearningPolygon]:
    """Parses YOLOv8-seg polygon lines into structured normalized polygon objects for UI rendering."""
    polygons: list[ActiveLearningPolygon] = []
    if not label_txt or not label_txt.strip():
        return polygons

    for raw_line in label_txt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        try:
            class_id = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue

        pts: list[list[float]] = []
        for i in range(0, len(vals), 2):
            pts.append([round(_clamp01(vals[i]), 6), round(_clamp01(vals[i + 1]), 6)])

        if len(pts) < 3:
            continue

        label = ID_TO_STORM_CLASS.get(class_id, f"class_{class_id}")
        polygons.append(ActiveLearningPolygon(class_id=class_id, label=label, points=pts))

    return polygons


def count_yolo_seg_classes(label_txt: str) -> dict[str, int]:
    """Returns per-class polygon counts keyed by string class ID (matching ml_pipeline/3_prepare_dataset.py)."""
    counts: dict[str, int] = {str(cid): 0 for cid in sorted(ID_TO_STORM_CLASS.keys())}
    for poly in parse_yolo_seg(label_txt):
        key = str(poly.class_id)
        counts[key] = counts.get(key, 0) + 1
    return counts
