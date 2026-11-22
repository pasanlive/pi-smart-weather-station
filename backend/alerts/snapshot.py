from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

DEFAULT_BBOX = (75.0, 3.0, 86.0, 12.0)
SLT_TZ = timezone(timedelta(hours=5, minutes=30))
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_DISTRICTS_PATH = _DATA_DIR / "sri_lanka_districts.geojson"
_DEFAULT_SEA_ZONES_PATH = _DATA_DIR / "sri_lanka_sea_zones.geojson"

_CACHED_DISTRICTS_GEOJSON: Optional[dict[str, Any]] = None
_CACHED_SEA_ZONES_GEOJSON: Optional[dict[str, Any]] = None

# Basin risk colors in BGR
_BASIN_RISK_BGR: dict[str, tuple[int, int, int]] = {
    "NORMAL": (232, 189, 56),    # sky-cyan
    "WATCH": (21, 204, 250),     # yellow
    "WARNING": (22, 146, 251),   # orange
    "CRITICAL": (94, 63, 244),   # crimson
}


def _load_default_districts() -> Optional[dict[str, Any]]:
    global _CACHED_DISTRICTS_GEOJSON
    if _CACHED_DISTRICTS_GEOJSON is None and _DEFAULT_DISTRICTS_PATH.exists():
        try:
            _CACHED_DISTRICTS_GEOJSON = json.loads(_DEFAULT_DISTRICTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CACHED_DISTRICTS_GEOJSON = {"type": "FeatureCollection", "features": []}
    return _CACHED_DISTRICTS_GEOJSON


def _load_default_sea_zones() -> Optional[dict[str, Any]]:
    global _CACHED_SEA_ZONES_GEOJSON
    if _CACHED_SEA_ZONES_GEOJSON is None and _DEFAULT_SEA_ZONES_PATH.exists():
        try:
            _CACHED_SEA_ZONES_GEOJSON = json.loads(_DEFAULT_SEA_ZONES_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CACHED_SEA_ZONES_GEOJSON = {"type": "FeatureCollection", "features": []}
    return _CACHED_SEA_ZONES_GEOJSON


def _geo_to_px(
    lon: float,
    lat: float,
    width: int,
    height: int,
    bbox: tuple[float, float, float, float],
) -> tuple[int, int]:
    lon_min, lat_min, lon_max, lat_max = bbox
    x = int(round(((lon - lon_min) / (lon_max - lon_min)) * width))
    y = int(round(((lat_max - lat) / (lat_max - lat_min)) * height))
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _iter_geojson_rings(geom: dict[str, Any]) -> list[list[list[float]]]:
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    rings: list[list[list[float]]] = []
    if gtype == "Polygon":
        if coords:
            rings.append(coords[0])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append(poly[0])
    return rings


def _draw_dashed_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    color_bgr: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 7,
    gap_len: int = 5,
) -> None:
    """Draws a crisp anti-aliased dashed closed polygon outline."""
    n = len(pts)
    if n < 2:
        return
    period = max(2, dash_len + gap_len)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        seg_len = math.hypot(dx, dy)
        if seg_len < 1.0:
            continue
        ux, uy = dx / seg_len, dy / seg_len
        dist = 0.0
        while dist < seg_len:
            end_d = min(seg_len, dist + dash_len)
            sx = int(round(p1[0] + ux * dist))
            sy = int(round(p1[1] + uy * dist))
            ex = int(round(p1[0] + ux * end_d))
            ey = int(round(p1[1] + uy * end_d))
            cv2.line(img, (sx, sy), (ex, ey), color_bgr, thickness, cv2.LINE_AA)
            dist += period


def _hazard_prob_style_bgr(prob: int, is_sea_zone: bool) -> tuple[tuple[int, int, int], tuple[int, int, int], float]:
    """Returns (stroke_bgr, fill_bgr, fill_alpha) matching the dashboard Combined Max color ramp."""
    if prob > 75:
        return (94, 63, 244), (68, 68, 239), (0.30 if is_sea_zone else 0.38)
    if prob > 50:
        return (60, 146, 251), (22, 115, 249), (0.24 if is_sea_zone else 0.32)
    if prob >= 30:
        return (21, 204, 250), (8, 179, 234), (0.17 if is_sea_zone else 0.24)
    if is_sea_zone:
        return (195, 145, 56), (160, 105, 30), 0.06
    return (140, 168, 145), (90, 115, 95), 0.04


def _draw_prob_badge(img: np.ndarray, x: int, y: int, text: str, bg_bgr: tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
    bx1 = max(4, min(img.shape[1] - tw - 10, x - tw // 2 - 3))
    by1 = max(42, min(img.shape[0] - 58, y - th // 2 - 2))
    bx2 = bx1 + tw + 6
    by2 = by1 + th + 5
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (10, 15, 25), -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), bg_bgr, 1)
    cv2.putText(img, text, (bx1 + 3, by2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (248, 250, 252), 1, cv2.LINE_AA)


def _format_utc_and_slt(timestamp_str: Optional[str]) -> str:
    dt: datetime
    if timestamp_str:
        try:
            raw = timestamp_str.strip()
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    utc_txt = dt.strftime("%Y-%m-%d %H:%M UTC")
    slt_txt = dt.astimezone(SLT_TZ).strftime("%H:%M SLT")
    return f"{utc_txt} | {slt_txt} (+5:30)"


def _extract_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _prepare_ir_base(
    frame_bgr: Optional[np.ndarray],
    ir_loop_frames: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Prepares the 640x640 Himawari-9 Band 13 IR base (blending subtle IR loop motion trail when provided)."""
    def _norm(img: np.ndarray) -> np.ndarray:
        out = img.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
        if out.shape[0] != 640 or out.shape[1] != 640:
            out = cv2.resize(out, (640, 640), interpolation=cv2.INTER_AREA)
        return out

    if frame_bgr is None or frame_bgr.size == 0:
        if ir_loop_frames:
            valid = [_norm(f) for f in ir_loop_frames if f is not None and f.size > 0]
            if valid:
                frame_bgr = valid[-1]
        if frame_bgr is None or frame_bgr.size == 0:
            return np.full((640, 640, 3), 18, dtype=np.uint8)

    curr = _norm(frame_bgr)
    if ir_loop_frames and len(ir_loop_frames) >= 2:
        valid = [_norm(f) for f in ir_loop_frames[-3:] if f is not None and f.size > 0]
        if len(valid) == 3:
            acc = (
                0.80 * valid[2].astype(np.float32)
                + 0.13 * valid[1].astype(np.float32)
                + 0.07 * valid[0].astype(np.float32)
            )
            return np.clip(acc, 0, 255).astype(np.uint8)
        if len(valid) == 2:
            acc = 0.85 * valid[1].astype(np.float32) + 0.15 * valid[0].astype(np.float32)
            return np.clip(acc, 0, 255).astype(np.uint8)
    return curr


def render_alert_snapshot(
    frame_bgr: Optional[np.ndarray],
    detections: Sequence[Any],
    motions: Sequence[Any],
    basins_geojson: Optional[dict[str, Any]],
    timestamp_str: Optional[str],
    *,
    ir_loop_frames: Optional[Sequence[np.ndarray]] = None,
    districts_geojson: Optional[dict[str, Any]] = None,
    districts: Optional[Sequence[Any]] = None,
    sea_zones_geojson: Optional[dict[str, Any]] = None,
    projected_geojson: Optional[dict[str, dict[str, Any]]] = None,
    highlighted_zones_geojson: Optional[dict[str, Any]] = None,
    bbox: tuple[float, float, float, float] = DEFAULT_BBOX,
    alert_title: str = "SRI LANKA WEATHER NOWCAST ALERT",
    engine_label: str = "CPU Fallback",
    jpeg_quality: int = 85,
) -> bytes:
    """
    Renders a full-stack annotated 640x640 JPEG snapshot for Telegram alerts containing:
      1. Himawari-9 Band 13 IR Loop base (with optional subtle multi-frame trail)
      2. 8 Offshore Sea Zones shaded by Combined Max hazard probability (dashed outlines)
      3. 25 Sri Lanka Districts shaded by Combined Max hazard probability
      4. 4 Wet-Zone River Flood Basins (risk-colored dashed outlines)
      5. +30m / +60m / +120m Projected Storm Trajectories (stepped-opacity dashed polygons)
      6. Current Storm Masks (deep_convective red, storm_shield amber)
      7. Optical-Flow Velocity Vectors (cyan centroid arrows scaled by km/h)
      8. Dark Header & Two-Row Legend Footer (UTC + SLT, active AI engine, color key)
    """
    base = _prepare_ir_base(frame_bgr, ir_loop_frames)
    h, w = base.shape[:2]

    # Build lookup of Combined Max probabilities by district_id
    district_prob: dict[str, int] = {}
    for d in districts or []:
        did = str(_extract_attr(d, "district_id", ""))
        hz = _extract_attr(d, "hazards")
        if did and hz is not None:
            district_prob[did] = int(_extract_attr(hz, "max_hazard_prob", 0) or 0)

    # Also absorb probabilities from optional highlighted_zones_geojson
    highlighted_prob: dict[str, int] = {}
    if highlighted_zones_geojson:
        for feat in highlighted_zones_geojson.get("features", []) or []:
            props = feat.get("properties") or {}
            zid = str(props.get("zone_id") or "")
            if zid:
                highlighted_prob[zid] = int(props.get("max_hazard_prob", 0) or 0)

    # 1. Sea Zones + Combined Max Probability Choropleth (dashed outlines)
    sea_fc = sea_zones_geojson if (sea_zones_geojson and sea_zones_geojson.get("features")) else _load_default_sea_zones()
    if sea_fc:
        for feat in sea_fc.get("features", []) or []:
            props = feat.get("properties") or {}
            zid = str(props.get("zone_id") or "")
            hz = props.get("hazards") or {}
            prob = int(hz.get("max_hazard_prob", highlighted_prob.get(zid, 0)) or 0)
            stroke_bgr, fill_bgr, fill_alpha = _hazard_prob_style_bgr(prob, is_sea_zone=True)
            geom = feat.get("geometry") or {}
            for ring in _iter_geojson_rings(geom):
                if len(ring) < 3:
                    continue
                pts = np.array([_geo_to_px(float(p[0]), float(p[1]), w, h, bbox) for p in ring], dtype=np.int32)
                if fill_alpha > 0.0:
                    ov = base.copy()
                    cv2.fillPoly(ov, [pts], fill_bgr)
                    cv2.addWeighted(ov, fill_alpha, base, 1.0 - fill_alpha, 0.0, dst=base)
                _draw_dashed_polyline(base, pts, stroke_bgr, thickness=2 if prob > 50 else 1, dash_len=7, gap_len=5)

            centroid = props.get("centroid")
            if prob >= 30 and isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
                cx, cy = _geo_to_px(float(centroid[0]), float(centroid[1]), w, h, bbox)
                _draw_prob_badge(base, cx, cy, f"{prob}%", stroke_bgr)

    # 2. 25 Sri Lanka Districts + Combined Max Probability Choropleth
    sl_districts = districts_geojson if districts_geojson is not None else _load_default_districts()
    elevated_district_badges: list[tuple[int, int, int, tuple[int, int, int]]] = []
    if sl_districts:
        for feat in sl_districts.get("features", []) or []:
            props = feat.get("properties") or {}
            did = str(props.get("district_id") or "")
            prob = district_prob.get(did, highlighted_prob.get(did, 0))
            stroke_bgr, fill_bgr, fill_alpha = _hazard_prob_style_bgr(prob, is_sea_zone=False)
            geom = feat.get("geometry") or {}
            for ring in _iter_geojson_rings(geom):
                if len(ring) < 3:
                    continue
                pts = np.array([_geo_to_px(float(p[0]), float(p[1]), w, h, bbox) for p in ring], dtype=np.int32)
                if fill_alpha > 0.0:
                    ov = base.copy()
                    cv2.fillPoly(ov, [pts], fill_bgr)
                    cv2.addWeighted(ov, fill_alpha, base, 1.0 - fill_alpha, 0.0, dst=base)
                cv2.polylines(base, [pts], isClosed=True, color=stroke_bgr, thickness=1, lineType=cv2.LINE_AA)

            if prob >= 50 and "centroid_lon" in props and "centroid_lat" in props:
                cx, cy = _geo_to_px(float(props["centroid_lon"]), float(props["centroid_lat"]), w, h, bbox)
                elevated_district_badges.append((prob, cx, cy, stroke_bgr))

    # 3. 4 Wet-Zone River Flood Basins (dashed risk-colored borders)
    if basins_geojson:
        for feat in basins_geojson.get("features", []) or []:
            props = feat.get("properties") or {}
            risk = str(props.get("risk_level") or "NORMAL").upper()
            b_bgr = _BASIN_RISK_BGR.get(risk, _BASIN_RISK_BGR["NORMAL"])
            geom = feat.get("geometry") or {}
            for ring in _iter_geojson_rings(geom):
                if len(ring) < 3:
                    continue
                pts = np.array([_geo_to_px(float(p[0]), float(p[1]), w, h, bbox) for p in ring], dtype=np.int32)
                ov = base.copy()
                cv2.fillPoly(ov, [pts], b_bgr)
                cv2.addWeighted(ov, 0.16, base, 0.84, 0.0, dst=base)
                _draw_dashed_polyline(base, pts, b_bgr, thickness=2, dash_len=6, gap_len=4)

    # 4. +30m / +60m / +120m Projected Storm Trajectories (stepped-opacity dashed polygons)
    if projected_geojson:
        horizon_specs = [
            ("+120m", (11, 158, 245), 0.08, 4, 6, 1),   # amber dotted
            ("+60m", (60, 146, 251), 0.14, 6, 5, 1),    # orange dashed
            ("+30m", (94, 63, 244), 0.20, 8, 4, 2),     # rose-red crisp dashed
        ]
        for hkey, color_bgr, fill_alpha, dlen, glen, thick in horizon_specs:
            fc = projected_geojson.get(hkey)
            if not fc:
                continue
            ov = base.copy()
            drawn_h = False
            for feat in fc.get("features", []) or []:
                geom = feat.get("geometry") or {}
                for ring in _iter_geojson_rings(geom):
                    if len(ring) < 3:
                        continue
                    pts = np.array([_geo_to_px(float(p[0]), float(p[1]), w, h, bbox) for p in ring], dtype=np.int32)
                    cv2.fillPoly(ov, [pts], color_bgr)
                    drawn_h = True
            if drawn_h:
                cv2.addWeighted(ov, fill_alpha, base, 1.0 - fill_alpha, 0.0, dst=base)
                for feat in fc.get("features", []) or []:
                    geom = feat.get("geometry") or {}
                    for ring in _iter_geojson_rings(geom):
                        if len(ring) < 3:
                            continue
                        pts = np.array([_geo_to_px(float(p[0]), float(p[1]), w, h, bbox) for p in ring], dtype=np.int32)
                        _draw_dashed_polyline(base, pts, color_bgr, thickness=thick, dash_len=dlen, gap_len=glen)

    # 5. Current Storm Masks (0m): deep_convective in red, storm/monsoon shields in amber
    storm_overlay = base.copy()
    drawn_cells = 0
    for det in detections or []:
        label = str(_extract_attr(det, "label", "storm_shield")).lower()
        if label == "deep_convective":
            fill_bgr = (68, 52, 239)
            edge_bgr = (115, 92, 255)
        else:
            fill_bgr = (11, 158, 245)
            edge_bgr = (55, 200, 255)

        pts_arr: Optional[np.ndarray] = None
        px_poly = _extract_attr(det, "pixel_polygon")
        if px_poly and len(px_poly) >= 3:
            pts_arr = np.round(np.asarray(px_poly, dtype=np.float32)).astype(np.int32)
        else:
            geo_poly = _extract_attr(det, "geo_polygon")
            if geo_poly and len(geo_poly) >= 3:
                pts_arr = np.array([_geo_to_px(float(p[0]), float(p[1]), w, h, bbox) for p in geo_poly], dtype=np.int32)
            else:
                geom_now = _extract_attr(det, "geometry_now")
                if isinstance(geom_now, dict):
                    rings = _iter_geojson_rings(geom_now)
                    if rings and len(rings[0]) >= 3:
                        pts_arr = np.array(
                            [_geo_to_px(float(p[0]), float(p[1]), w, h, bbox) for p in rings[0]],
                            dtype=np.int32,
                        )

        if pts_arr is not None and len(pts_arr) >= 3:
            cv2.fillPoly(storm_overlay, [pts_arr], fill_bgr)
            cv2.polylines(base, [pts_arr], isClosed=True, color=edge_bgr, thickness=2, lineType=cv2.LINE_AA)
            drawn_cells += 1

    if drawn_cells > 0:
        cv2.addWeighted(storm_overlay, 0.30, base, 0.70, 0.0, dst=base)

    # 6. Optical-Flow Velocity Vectors (cyan centroid arrows)
    for mot in motions or []:
        speed = float(_extract_attr(mot, "speed_kmh", 0.0) or 0.0)
        if speed < 1.5:
            continue

        clon = _extract_attr(mot, "centroid_lon")
        clat = _extract_attr(mot, "centroid_lat")
        if clon is not None and clat is not None:
            cx, cy = _geo_to_px(float(clon), float(clat), w, h, bbox)
        else:
            cpx = _extract_attr(mot, "centroid_px")
            if not cpx or len(cpx) < 2:
                continue
            cx, cy = int(round(float(cpx[0]))), int(round(float(cpx[1])))

        dx_kmh = _extract_attr(mot, "dx_kmh")
        dy_kmh = _extract_attr(mot, "dy_kmh")
        if dx_kmh is not None and dy_kmh is not None and math.hypot(float(dx_kmh), float(dy_kmh)) > 0.1:
            mag = math.hypot(float(dx_kmh), float(dy_kmh))
            ux = float(dx_kmh) / mag
            uy = -float(dy_kmh) / mag
        else:
            bearing = float(_extract_attr(mot, "bearing_deg", 0.0) or 0.0)
            rad = math.radians(bearing)
            ux = math.sin(rad)
            uy = -math.cos(rad)

        arrow_len = int(round(18 + min(1.0, speed / 80.0) * 30))
        ex = max(6, min(w - 6, int(round(cx + ux * arrow_len))))
        ey = max(44, min(h - 56, int(round(cy + uy * arrow_len))))

        cv2.arrowedLine(base, (cx, cy), (ex, ey), (238, 211, 34), 2, cv2.LINE_AA, tipLength=0.32)
        cv2.circle(base, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Stamp top elevated district probability pills (>50%, up to 6 spaced out)
    elevated_district_badges.sort(key=lambda item: -item[0])
    placed_centers: list[tuple[int, int]] = []
    for prob, cx, cy, bgr in elevated_district_badges:
        if len(placed_centers) >= 6:
            break
        if any(math.hypot(cx - px, cy - py) < 28 for px, py in placed_centers):
            continue
        _draw_prob_badge(base, cx, cy, f"{prob}%", bgr)
        placed_centers.append((cx, cy))

    # 7. Dark Header Banner & Compact Two-Row Bottom Legend Bar
    top_h = 38
    bot_h = 48
    bar_overlay = base.copy()
    cv2.rectangle(bar_overlay, (0, 0), (w, top_h), (10, 15, 26), -1)
    cv2.rectangle(bar_overlay, (0, h - bot_h), (w, h), (10, 15, 26), -1)
    cv2.addWeighted(bar_overlay, 0.88, base, 0.12, 0.0, dst=base)

    cv2.line(base, (0, top_h), (w, top_h), (238, 189, 56), 1)
    cv2.line(base, (0, h - bot_h), (w, h - bot_h), (65, 82, 105), 1)

    title_clean = alert_title.strip()[:46]
    cv2.putText(base, title_clean, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA)

    eng_text = f"AI: {engine_label}"
    (eng_w, _), _ = cv2.getTextSize(eng_text, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    eng_color = (110, 231, 183) if "hailo" in engine_label.lower() else (238, 211, 34)
    cv2.putText(base, eng_text, (max(12, w - eng_w - 12), 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, eng_color, 1, cv2.LINE_AA)

    # Bottom row 1: Timestamp + active cell count
    time_line = _format_utc_and_slt(timestamp_str)
    cv2.putText(base, time_line, (12, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 232, 240), 1, cv2.LINE_AA)

    right_meta = f"Cells: {drawn_cells} | Mode: Combined Max %"
    (rm_w, _), _ = cv2.getTextSize(right_meta, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(base, right_meta, (max(12, w - rm_w - 12), h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (186, 230, 253), 1, cv2.LINE_AA)

    # Bottom row 2: Crisp visual swatches for all layers
    ly = h - 10
    x_cur = 12
    legend_items: list[tuple[tuple[int, int, int], str]] = [
        ((68, 52, 239), "Deep"),
        ((11, 158, 245), "Shield"),
        ((94, 63, 244), "+30/60/120m"),
        ((238, 211, 34), "Vec"),
        ((232, 189, 56), "Basins/Sea"),
        ((21, 204, 250), "30-50%"),
        ((22, 115, 249), "50-75%"),
        ((68, 68, 239), ">75%"),
    ]
    for sw_bgr, lbl in legend_items:
        cv2.rectangle(base, (x_cur, ly - 8), (x_cur + 8, ly), sw_bgr, -1)
        x_cur += 11
        cv2.putText(base, lbl, (x_cur, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (203, 213, 225), 1, cv2.LINE_AA)
        (lw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        x_cur += lw + 10

    ok, encoded = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode annotated alert snapshot to JPEG")
    return encoded.tobytes()
