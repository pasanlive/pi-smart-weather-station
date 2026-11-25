from __future__ import annotations

import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_BBOX = (75.0, 3.0, 86.0, 12.0)
SLT_TZ = timezone(timedelta(hours=5, minutes=30))
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_DISTRICTS_PATH = _DATA_DIR / "sri_lanka_districts.geojson"
_DEFAULT_SEA_ZONES_PATH = _DATA_DIR / "sri_lanka_sea_zones.geojson"

_CACHED_DISTRICTS_GEOJSON: Optional[dict[str, Any]] = None
_CACHED_SEA_ZONES_GEOJSON: Optional[dict[str, Any]] = None

# Clean broadcast weather card output dimensions (rendered at 2x for crisp anti-aliasing)
OUT_W, OUT_H = 1080, 840
SCALE = 2

# Tight geographic framing around Sri Lanka + coastal sea sectors
MAP_LON_MIN, MAP_LON_MAX = 78.70, 82.85
MAP_LAT_MIN, MAP_LAT_MAX = 5.05, 10.65

_COMPASS_16 = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW",
]

# 7 well-spaced Sri Lanka regional hubs for clean map icons (no overlapping labels)
_MAP_HUBS: list[tuple[str, str, float, float, int, int]] = [
    # (district_id, display_name, lon, lat, label_dx_px, label_dy_px)
    ("jaffna", "Jaffna", 80.02, 9.68, 0, -34),
    ("trincomalee", "Trincomalee", 81.21, 8.58, 50, -14),
    ("anuradhapura", "Anuradhapura", 80.40, 8.35, -52, -16),
    ("batticaloa", "Batticaloa", 81.68, 7.72, 46, -24),
    ("kandy", "Kandy", 80.64, 7.29, 0, -32),
    ("colombo", "Colombo", 79.86, 6.93, -52, 4),
    ("galle", "Galle", 80.22, 6.06, -42, 28),
]

_THREAT_COLORS: dict[str, tuple[int, int, int]] = {
    "LOW": (16, 185, 129),       # emerald
    "MODERATE": (250, 204, 21),  # yellow
    "HIGH": (249, 115, 22),      # orange
    "SEVERE": (239, 68, 68),     # crimson
}

_BASIN_COLORS: dict[str, tuple[int, int, int]] = {
    "NORMAL": (16, 185, 129),
    "WATCH": (250, 204, 21),
    "WARNING": (249, 115, 22),
    "CRITICAL": (239, 68, 68),
}


def _load_default_districts() -> dict[str, Any]:
    global _CACHED_DISTRICTS_GEOJSON
    if _CACHED_DISTRICTS_GEOJSON is None and _DEFAULT_DISTRICTS_PATH.exists():
        try:
            _CACHED_DISTRICTS_GEOJSON = json.loads(_DEFAULT_DISTRICTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CACHED_DISTRICTS_GEOJSON = {"type": "FeatureCollection", "features": []}
    return _CACHED_DISTRICTS_GEOJSON or {"type": "FeatureCollection", "features": []}


def _load_default_sea_zones() -> dict[str, Any]:
    global _CACHED_SEA_ZONES_GEOJSON
    if _CACHED_SEA_ZONES_GEOJSON is None and _DEFAULT_SEA_ZONES_PATH.exists():
        try:
            _CACHED_SEA_ZONES_GEOJSON = json.loads(_DEFAULT_SEA_ZONES_PATH.read_text(encoding="utf-8"))
        except Exception:
            _CACHED_SEA_ZONES_GEOJSON = {"type": "FeatureCollection", "features": []}
    return _CACHED_SEA_ZONES_GEOJSON or {"type": "FeatureCollection", "features": []}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _deg_to_compass(deg: float) -> str:
    idx = int(((float(deg) % 360.0) + 11.25) // 22.5) % 16
    return _COMPASS_16[idx]


def _extract_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _iter_geojson_rings(geom: dict[str, Any]) -> list[list[list[float]]]:
    gtype = geom.get("type")
    coords = geom.get("coordinates") or []
    rings: list[list[list[float]]] = []
    if gtype == "Polygon" and coords:
        rings.append(coords[0])
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                rings.append(poly[0])
    return rings


def _parse_timestamps(timestamp_str: Optional[str]) -> tuple[str, str, int]:
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

    slt = dt.astimezone(SLT_TZ)
    return slt.strftime("%d %b %Y • %H:%M SLT"), dt.strftime("%H:%M UTC"), slt.hour


def _effective_threat(sat_threat: str, max_prob: int) -> str:
    rank = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "SEVERE": 4}
    base = sat_threat.upper() if sat_threat.upper() in rank else "LOW"
    if max_prob > 75:
        prob_lvl = "SEVERE"
    elif max_prob > 50:
        prob_lvl = "HIGH"
    elif max_prob >= 30:
        prob_lvl = "MODERATE"
    else:
        prob_lvl = "LOW"
    return prob_lvl if rank[prob_lvl] > rank[base] else base


def _prob_to_fill_color(prob: int, threat: str, is_sea: bool = False) -> tuple[int, int, int, int]:
    """Maps probability / threat level to a clean cartographic color (RGBA)."""
    if threat == "SEVERE" or prob > 75:
        return (220, 38, 38, 68 if is_sea else 225)
    if threat == "HIGH" or prob > 50:
        return (234, 88, 12, 58 if is_sea else 218)
    if threat == "MODERATE" or prob >= 30:
        return (202, 138, 4, 52 if is_sea else 212)
    if is_sea:
        return (30, 80, 125, 42)
    return (30, 74, 96, 235)


def _draw_dashed_polygon(
    draw: ImageDraw.ImageDraw,
    pts: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int = 2,
    dash: float = 12.0,
    gap: float = 8.0,
) -> None:
    n = len(pts)
    if n < 2:
        return
    period = dash + gap
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg < 1.0:
            continue
        ux, uy = (x2 - x1) / seg, (y2 - y1) / seg
        d = 0.0
        while d < seg:
            end_d = min(seg, d + dash)
            draw.line(
                [(x1 + ux * d, y1 + uy * d), (x1 + ux * end_d, y1 + uy * end_d)],
                fill=color,
                width=width,
            )
            d += period


def _pick_weather_icon_kind(rain_prob: int, ts_prob: int, wind_prob: int) -> str:
    if ts_prob >= 45:
        return "storm"
    if rain_prob >= 40:
        return "rain"
    if wind_prob >= 45:
        return "wind"
    if max(rain_prob, ts_prob) >= 22:
        return "partly_cloudy"
    return "clear"


def _draw_weather_icon(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    kind: str,
    is_night: bool = False,
) -> None:
    """Draws clean, anti-aliased vector weather icons (sun/moon, partly cloudy, rain, storm, wind)."""
    r = size * 0.5

    def _draw_sun(sx: float, sy: float, sr: float) -> None:
        if is_night and kind == "clear":
            # Calm silver-gold crescent moon for night SLT reports
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(253, 224, 71, 255))
            draw.ellipse(
                [sx - sr * 0.45, sy - sr * 1.1, sx + sr * 1.15, sy + sr * 0.55],
                fill=(13, 28, 52, 255),
            )
            return
        draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(250, 204, 21, 255))
        for deg in range(0, 360, 45):
            rad = math.radians(deg)
            x1 = sx + math.cos(rad) * (sr * 1.28)
            y1 = sy + math.sin(rad) * (sr * 1.28)
            x2 = sx + math.cos(rad) * (sr * 1.72)
            y2 = sy + math.sin(rad) * (sr * 1.72)
            draw.line([(x1, y1), (x2, y2)], fill=(253, 224, 71, 255), width=max(2, int(size * 0.075)))

    def _draw_cloud(kx: float, ky: float, scale: float, fill: tuple[int, int, int, int]) -> None:
        w = size * scale
        h = w * 0.56
        # Soft dark drop-shadow behind cloud for high legibility over map
        sh = (6, 14, 26, 140)
        off = max(2.0, size * 0.04)
        for ox, oy, col in [(off, off, sh), (0.0, 0.0, fill)]:
            draw.ellipse([kx - w * 0.44 + ox, ky - h * 0.12 + oy, kx - w * 0.02 + ox, ky + h * 0.50 + oy], fill=col)
            draw.ellipse([kx - w * 0.24 + ox, ky - h * 0.46 + oy, kx + w * 0.28 + ox, ky + h * 0.34 + oy], fill=col)
            draw.ellipse([kx + w * 0.02 + ox, ky - h * 0.14 + oy, kx + w * 0.46 + ox, ky + h * 0.50 + oy], fill=col)
            draw.rounded_rectangle(
                [kx - w * 0.36 + ox, ky + h * 0.05 + oy, kx + w * 0.38 + ox, ky + h * 0.50 + oy],
                radius=max(2, int(h * 0.22)),
                fill=col,
            )

    if kind == "clear":
        _draw_sun(cx, cy, r * 0.48)
    elif kind == "partly_cloudy":
        _draw_sun(cx + r * 0.26, cy - r * 0.26, r * 0.34)
        _draw_cloud(cx - r * 0.08, cy + r * 0.08, 0.88, (226, 232, 240, 255))
    elif kind == "rain":
        _draw_cloud(cx, cy - r * 0.18, 0.90, (186, 206, 228, 255))
        lw = max(2, int(size * 0.08))
        for dx in (-0.28, 0.0, 0.28):
            rx = cx + size * dx
            ry = cy + r * 0.26
            draw.line([(rx, ry), (rx - size * 0.09, ry + size * 0.24)], fill=(56, 189, 248, 255), width=lw)
    elif kind == "storm":
        _draw_cloud(cx, cy - r * 0.20, 0.94, (148, 163, 184, 255))
        # Crisp golden lightning bolt
        bolt = [
            (cx + size * 0.03, cy + r * 0.06),
            (cx - size * 0.12, cy + r * 0.42),
            (cx + size * 0.01, cy + r * 0.42),
            (cx - size * 0.08, cy + r * 0.80),
            (cx + size * 0.16, cy + r * 0.30),
            (cx + size * 0.02, cy + r * 0.30),
        ]
        draw.polygon(bolt, fill=(250, 204, 21, 255))
        lw = max(2, int(size * 0.07))
        for dx in (-0.30, 0.28):
            rx = cx + size * dx
            ry = cy + r * 0.30
            draw.line([(rx, ry), (rx - size * 0.07, ry + size * 0.18)], fill=(56, 189, 248, 255), width=lw)
    elif kind == "wind":
        lw = max(2, int(size * 0.085))
        for idx, (dy, span) in enumerate([(-0.24, 0.72), (0.04, 0.88), (0.30, 0.56)]):
            y = cy + r * dy * 1.4
            x0 = cx - r * 0.68
            x1 = x0 + size * span
            col = (56, 189, 248, 255) if idx != 1 else (226, 232, 240, 255)
            draw.line([(x0, y), (x1, y)], fill=col, width=lw)
            draw.arc([x1 - size * 0.12, y - size * 0.14, x1 + size * 0.10, y + size * 0.04], 240, 90, fill=col, width=lw)


def _draw_wind_compass(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    radius: float,
    direction_deg: float,
    font_sm: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draws a crisp circular meteorological wind compass with directional arrow."""
    # Outer ring & inner dial
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(11, 25, 45, 255),
        outline=(56, 189, 248, 170),
        width=3,
    )
    inner_r = radius * 0.80
    draw.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        outline=(51, 78, 108, 140),
        width=2,
    )

    # Cardinal labels N / E / S / W
    cardinals = [("N", 0, -1), ("E", 1, 0), ("S", 0, 1), ("W", -1, 0)]
    for lbl, dx, dy in cardinals:
        lx = cx + dx * (radius * 0.66)
        ly = cy + dy * (radius * 0.66)
        col = (248, 113, 113, 255) if lbl == "N" else (148, 163, 184, 255)
        draw.text((lx, ly), lbl, fill=col, font=font_sm, anchor="mm")

    # Meteorological wind direction: arrow points in the direction the wind is blowing TO
    # (Open-Meteo wind_direction_10m is direction wind blows FROM, so flow vector is deg + 180)
    flow_rad = math.radians((direction_deg + 180.0) % 360.0)
    ux = math.sin(flow_rad)
    uy = -math.cos(flow_rad)
    px, py = -uy, ux

    tail = (cx - ux * radius * 0.44, cy - uy * radius * 0.44)
    tip = (cx + ux * radius * 0.50, cy + uy * radius * 0.50)
    draw.line([tail, tip], fill=(56, 189, 248, 255), width=max(4, int(radius * 0.09)))

    head_base = (cx + ux * radius * 0.24, cy + uy * radius * 0.24)
    wing = radius * 0.17
    arrow_poly = [
        tip,
        (head_base[0] + px * wing, head_base[1] + py * wing),
        (head_base[0] - px * wing, head_base[1] - py * wing),
    ]
    draw.polygon(arrow_poly, fill=(56, 189, 248, 255))
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(248, 250, 252, 255))


def _draw_small_wind_arrow(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    length: float,
    direction_deg: float,
    color: tuple[int, int, int, int] = (56, 189, 248, 240),
) -> None:
    flow_rad = math.radians((direction_deg + 180.0) % 360.0)
    ux, uy = math.sin(flow_rad), -math.cos(flow_rad)
    px, py = -uy, ux
    x0, y0 = cx - ux * length * 0.45, cy - uy * length * 0.45
    x1, y1 = cx + ux * length * 0.55, cy + uy * length * 0.55
    draw.line([(x0, y0), (x1, y1)], fill=color, width=3)
    hb_x, hb_y = cx + ux * length * 0.18, cy + uy * length * 0.18
    wing = length * 0.26
    draw.polygon([(x1, y1), (hb_x + px * wing, hb_y + py * wing), (hb_x - px * wing, hb_y - py * wing)], fill=color)


def _create_gradient_bg(w: int, h: int) -> Image.Image:
    """Returns an RGB gradient canvas so ImageDraw.Draw(img, 'RGBA') alpha-composites natively."""
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    r = (7.0 + y * 8.0).astype(np.uint8)
    g = (18.0 + y * 15.0).astype(np.uint8)
    b = (36.0 + y * 24.0).astype(np.uint8)
    rgb = np.stack([np.broadcast_to(r, (h, w)), np.broadcast_to(g, (h, w)), np.broadcast_to(b, (h, w))], axis=-1)
    return Image.fromarray(rgb)


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
    alert_title: str = "SRI LANKA WEATHER REPORT",
    engine_label: str = "CPU Fallback",
    jpeg_quality: int = 88,
) -> bytes:
    """
    Renders a clean, broadcast weather-report card (1080x840 JPEG) tailored for Telegram:
      - Zero satellite clutter, storm masks, or optical-flow vectors
      - Clean Sri Lanka map shaded by rain/storm probability & threat level with regional weather icons
      - Overall Threat Level badge, Wind Compass Dial (direction + speed + gusts), Top Districts table,
        and Wet-Zone River Basin flood status strip.
    """
    _ = (frame_bgr, detections, motions, ir_loop_frames, projected_geojson, bbox)  # kept for signature compat
    S = SCALE
    W, H = OUT_W * S, OUT_H * S

    img = _create_gradient_bg(W, H)
    draw = ImageDraw.Draw(img, "RGBA")

    # Typography
    f_title = _load_font(25 * S, bold=True)
    f_sub = _load_font(13 * S, bold=False)
    f_badge = _load_font(15 * S, bold=True)
    f_sec = _load_font(13 * S, bold=True)
    f_kpi = _load_font(28 * S, bold=True)
    f_kpi_lbl = _load_font(11 * S, bold=True)
    f_body_b = _load_font(14 * S, bold=True)
    f_body = _load_font(13 * S, bold=False)
    f_sm_b = _load_font(11 * S, bold=True)
    f_sm = _load_font(11 * S, bold=False)
    f_xs = _load_font(10 * S, bold=True)

    # Normalize district data
    dist_map: dict[str, dict[str, Any]] = {}
    for d in districts or []:
        did = str(_extract_attr(d, "district_id", "")).lower()
        name = str(_extract_attr(d, "name", did.title()))
        threat = str(_extract_attr(d, "threat_level", "LOW")).upper()
        hz = _extract_attr(d, "hazards")
        rain_p = int(_extract_attr(hz, "rainfall_prob", 0) or 0) if hz is not None else 0
        ts_p = int(_extract_attr(hz, "thunderstorm_prob", 0) or 0) if hz is not None else 0
        wind_p = int(_extract_attr(hz, "high_wind_prob", 0) or 0) if hz is not None else 0
        max_p = int(_extract_attr(hz, "max_hazard_prob", max(rain_p, ts_p, wind_p)) or 0) if hz is not None else 0
        w_spd = float(_extract_attr(hz, "wind_speed_kmh", 0.0) or 0.0) if hz is not None else 0.0
        w_gust = float(_extract_attr(hz, "wind_gusts_kmh", _extract_attr(d, "wind_gusts_kmh", 0.0)) or 0.0)
        w_deg = float(_extract_attr(hz, "wind_direction_deg", 0.0) or 0.0) if hz is not None else 0.0
        w_comp = str(_extract_attr(hz, "wind_compass_dir", _deg_to_compass(w_deg)) or _deg_to_compass(w_deg))
        if did:
            eff_threat = _effective_threat(threat, max_p)
            dist_map[did] = {
                "id": did,
                "name": name,
                "threat": eff_threat,
                "rain": rain_p,
                "storm": ts_p,
                "wind_prob": wind_p,
                "max_prob": max_p,
                "wind_spd": w_spd if w_spd > 0 else round(w_gust * 0.65, 1),
                "wind_gust": w_gust,
                "wind_deg": w_deg,
                "wind_comp": w_comp,
            }

    # Absorb highlighted_zones_geojson probabilities if districts list wasn't passed
    if highlighted_zones_geojson:
        for feat in highlighted_zones_geojson.get("features", []) or []:
            props = feat.get("properties") or {}
            zid = str(props.get("zone_id") or "").lower()
            prob = int(props.get("max_hazard_prob", 0) or 0)
            if zid in dist_map and prob > dist_map[zid]["max_prob"]:
                dist_map[zid]["max_prob"] = prob

    # Normalize sea zones
    sea_fc = sea_zones_geojson if (sea_zones_geojson and sea_zones_geojson.get("features")) else _load_default_sea_zones()
    sea_list: list[dict[str, Any]] = []
    for feat in sea_fc.get("features", []) or []:
        props = feat.get("properties") or {}
        hz = props.get("hazards") or {}
        w_deg = float(hz.get("wind_direction_deg", 0.0) or 0.0)
        sea_list.append(
            {
                "id": str(props.get("zone_id") or ""),
                "name": str(props.get("name") or ""),
                "centroid": props.get("centroid") or [80.0, 6.0],
                "rain": int(hz.get("rainfall_prob", 0) or 0),
                "storm": int(hz.get("thunderstorm_prob", 0) or 0),
                "wind_prob": int(hz.get("high_wind_prob", 0) or 0),
                "max_prob": int(hz.get("max_hazard_prob", 0) or 0),
                "wind_spd": float(hz.get("wind_speed_kmh", 0.0) or 0.0),
                "wind_gust": float(hz.get("wind_gusts_kmh", 0.0) or 0.0),
                "wind_deg": w_deg,
                "wind_comp": str(hz.get("wind_compass_dir") or _deg_to_compass(w_deg)),
                "wave_m": hz.get("wave_height_m"),
                "geometry": feat.get("geometry") or {},
            }
        )

    # Compute island-wide overall threat, peak rain/storm, prevailing wind
    threat_rank = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "SEVERE": 4}
    all_dist_vals = list(dist_map.values())
    peak_rain = max((d["rain"] for d in all_dist_vals), default=0)
    peak_storm = max((d["storm"] for d in all_dist_vals), default=0)
    peak_wind_prob = max((d["wind_prob"] for d in all_dist_vals), default=0)
    overall_threat = "LOW"
    for d in all_dist_vals:
        if threat_rank.get(d["threat"], 1) > threat_rank[overall_threat]:
            overall_threat = d["threat"]
    # Escalate overall badge visually if combined probabilities are elevated
    peak_any = max(peak_rain, peak_storm, peak_wind_prob)
    if peak_any > 75 and threat_rank[overall_threat] < 4:
        overall_threat = "SEVERE"
    elif peak_any > 50 and threat_rank[overall_threat] < 3:
        overall_threat = "HIGH"
    elif peak_any >= 30 and threat_rank[overall_threat] < 2:
        overall_threat = "MODERATE"

    # Prevailing island wind (circular mean of direction, average speed, max gust)
    wind_samples = [d for d in all_dist_vals if d["wind_spd"] > 0 or d["wind_gust"] > 0] or [
        s for s in sea_list if s["wind_spd"] > 0 or s["wind_gust"] > 0
    ]
    if wind_samples:
        sin_sum = sum(math.sin(math.radians(x["wind_deg"])) for x in wind_samples)
        cos_sum = sum(math.cos(math.radians(x["wind_deg"])) for x in wind_samples)
        mean_wind_deg = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0
        avg_wind_spd = sum(x["wind_spd"] for x in wind_samples) / len(wind_samples)
        max_wind_gust = max(x["wind_gust"] for x in wind_samples)
    else:
        mean_wind_deg, avg_wind_spd, max_wind_gust = 45.0, 12.0, 20.0
    mean_wind_comp = _deg_to_compass(mean_wind_deg)

    slt_str, utc_str, slt_hour = _parse_timestamps(timestamp_str)
    is_night = slt_hour < 6 or slt_hour >= 19

    # =========================================================================
    # 1. TOP HEADER BAR
    # =========================================================================
    hdr_h = 80 * S
    draw.rectangle([0, 0, W, hdr_h], fill=(9, 20, 38, 245))
    draw.line([(0, hdr_h), (W, hdr_h)], fill=(38, 72, 114, 255), width=2 * S)

    # Clean title (normalize noisy uppercase prefixes into clean broadcast headline)
    clean_title = alert_title.strip()
    if len(clean_title) > 38:
        clean_title = clean_title[:37] + "…"
    draw.text((28 * S, 16 * S), clean_title, fill=(248, 250, 252, 255), font=f_title)
    draw.text(
        (28 * S, 48 * S),
        f"{slt_str}   |   {utc_str}   •   Sri Lanka Weather & Coastal Outlook",
        fill=(148, 175, 206, 255),
        font=f_sub,
    )

    # Overall Threat Level Pill on Top-Right (dynamically sized and centered)
    th_rgb = _THREAT_COLORS.get(overall_threat, (16, 185, 129))
    badge_txt = f"THREAT: {overall_threat}"
    btb = draw.textbbox((0, 0), badge_txt, font=f_badge)
    btw = btb[2] - btb[0]
    pill_w = max(176 * S, btw + 56 * S)
    pill_h = 42 * S
    px2 = W - 24 * S
    px1 = px2 - pill_w
    py1 = 19 * S
    py2 = py1 + pill_h
    draw.rounded_rectangle(
        [px1, py1, px2, py2],
        radius=21 * S,
        fill=(th_rgb[0], th_rgb[1], th_rgb[2], 46),
        outline=(th_rgb[0], th_rgb[1], th_rgb[2], 235),
        width=2 * S,
    )
    dot_r = 7 * S
    content_w = dot_r * 2 + 10 * S + btw
    start_x = px1 + (pill_w - content_w) / 2
    mid_py = (py1 + py2) / 2
    draw.ellipse([start_x, mid_py - dot_r, start_x + dot_r * 2, mid_py + dot_r], fill=(*th_rgb, 255))
    draw.text(
        (start_x + dot_r * 2 + 10 * S, mid_py),
        badge_txt,
        fill=(248, 250, 252, 255),
        font=f_badge,
        anchor="lm",
    )

    # =========================================================================
    # 2. LEFT PANEL — CLEAN CARTOGRAPHIC SRI LANKA WEATHER MAP
    # =========================================================================
    map_x1, map_y1, map_x2, map_y2 = 24 * S, 98 * S, 548 * S, 774 * S
    draw.rounded_rectangle(
        [map_x1, map_y1, map_x2, map_y2],
        radius=16 * S,
        fill=(10, 26, 48, 245),
        outline=(30, 58, 92, 255),
        width=2 * S,
    )

    # Maintain true geographic aspect ratio inside map card (cos(7.8 deg) ~= 0.99)
    pad_x, pad_y = 24 * S, 36 * S
    inner_w = (map_x2 - map_x1) - 2 * pad_x
    inner_h = (map_y2 - map_y1) - 2 * pad_y
    lon_span = MAP_LON_MAX - MAP_LON_MIN
    lat_span = MAP_LAT_MAX - MAP_LAT_MIN
    scale_px = min(inner_w / lon_span, inner_h / lat_span)
    proj_w = lon_span * scale_px
    proj_h = lat_span * scale_px
    ox = map_x1 + pad_x + (inner_w - proj_w) * 0.5
    oy = map_y1 + pad_y + (inner_h - proj_h) * 0.5

    def geo_to_map(lon: float, lat: float) -> tuple[float, float]:
        return (ox + (lon - MAP_LON_MIN) * scale_px, oy + (MAP_LAT_MAX - lat) * scale_px)

    # Subtle ocean coordinate gridlines
    for glon in (79, 80, 81, 82):
        gx, _ = geo_to_map(glon, 7.5)
        if map_x1 + 12 * S < gx < map_x2 - 12 * S:
            draw.line([(gx, map_y1 + 10 * S), (gx, map_y2 - 10 * S)], fill=(26, 48, 76, 110), width=1 * S)
    for glat in (6, 7, 8, 9, 10):
        _, gy = geo_to_map(80.5, glat)
        if map_y1 + 12 * S < gy < map_y2 - 12 * S:
            draw.line([(map_x1 + 10 * S, gy), (map_x2 - 10 * S, gy)], fill=(26, 48, 76, 110), width=1 * S)

    # Draw subtle 8 Coastal Sea Zones (dashed light borders)
    for sz in sea_list:
        fill_rgba = _prob_to_fill_color(sz["max_prob"], "LOW", is_sea=True)
        edge_rgba = (249, 115, 22, 180) if sz["max_prob"] > 50 else ((250, 204, 21, 140) if sz["max_prob"] >= 30 else (56, 130, 180, 110))
        for ring in _iter_geojson_rings(sz["geometry"]):
            if len(ring) < 3:
                continue
            pts = [geo_to_map(float(p[0]), float(p[1])) for p in ring]
            draw.polygon(pts, fill=fill_rgba)
            _draw_dashed_polygon(draw, pts, edge_rgba, width=max(2, S), dash=10 * S, gap=7 * S)

    # Draw 25 Sri Lanka Districts shaded by Rain/Storm Probability & Threat
    sl_districts = districts_geojson if districts_geojson is not None else _load_default_districts()
    for feat in sl_districts.get("features", []) or []:
        props = feat.get("properties") or {}
        did = str(props.get("district_id") or "").lower()
        dinfo = dist_map.get(did)
        prob = dinfo["max_prob"] if dinfo else 0
        threat = dinfo["threat"] if dinfo else "LOW"
        fill_rgba = _prob_to_fill_color(prob, threat, is_sea=False)
        for ring in _iter_geojson_rings(feat.get("geometry") or {}):
            if len(ring) < 3:
                continue
            pts = [geo_to_map(float(p[0]), float(p[1])) for p in ring]
            draw.polygon(pts, fill=fill_rgba)
            draw.line(pts + [pts[0]], fill=(186, 215, 240, 155), width=max(1, int(1.3 * S)))

    # 4 Clean Offshore Wind Indicators around Sri Lanka (West, South, East-SE, North-East)
    offshore_picks = [
        ("puttalam_colombo", 79.16, 7.62),
        ("galle_hambantota", 80.70, 5.46),
        ("hambantota_pottuvil", 82.16, 6.82),
        ("trincomalee_kks", 81.36, 9.75),
    ]
    sea_by_id = {s["id"]: s for s in sea_list}
    for sid, slon, slat in offshore_picks:
        sz = sea_by_id.get(sid)
        if not sz:
            continue
        sx, sy = geo_to_map(slon, slat)
        _draw_small_wind_arrow(draw, sx, sy - 10 * S, 22 * S, sz["wind_deg"], (125, 211, 252, 245))
        w_txt = f"{sz['wind_comp']} {sz['wind_spd']:.0f} km/h"
        draw.text((sx, sy + 10 * S), w_txt, fill=(186, 226, 255, 240), font=f_xs, anchor="mm")

    # 7 Regional Weather Icons + Clean City/Probability Badges
    for did, label, lon, lat, ldx, ldy in _MAP_HUBS:
        hx, hy = geo_to_map(lon, lat)
        dinfo = dist_map.get(did, {"rain": 0, "storm": 0, "wind_prob": 0, "max_prob": 0})
        kind = _pick_weather_icon_kind(dinfo["rain"], dinfo["storm"], dinfo["wind_prob"])
        _draw_weather_icon(draw, hx, hy, 34 * S, kind, is_night=is_night)

        # Compact readable pill offset neatly from icon
        p_val = max(dinfo["rain"], dinfo["storm"])
        pill_txt = f"{label} {p_val}%"
        tb = draw.textbbox((0, 0), pill_txt, font=f_xs)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        bx = max(map_x1 + 10 * S, min(map_x2 - tw - 24 * S, hx + ldx * S - tw / 2 - 6 * S))
        by = max(map_y1 + 12 * S, min(map_y2 - th - 18 * S, hy + ldy * S - th / 2 - 4 * S))
        b_col = (249, 115, 22, 220) if p_val > 50 else ((250, 204, 21, 200) if p_val >= 30 else (56, 120, 175, 190))
        draw.rounded_rectangle(
            [bx, by, bx + tw + 12 * S, by + th + 8 * S],
            radius=6 * S,
            fill=(8, 19, 36, 225),
            outline=b_col,
            width=max(1, S),
        )
        draw.text((bx + 6 * S, by + 3 * S), pill_txt, fill=(241, 245, 249, 255), font=f_xs)

    # Small clean map caption inside top-left of map frame
    draw.text((map_x1 + 16 * S, map_y1 + 14 * S), "RAIN & STORM PROBABILITY MAP", fill=(148, 175, 206, 230), font=f_xs)

    # =========================================================================
    # 3. RIGHT PANEL — 3 CLEAN BROADCAST INTELLIGENCE CARDS
    # =========================================================================
    rx1, rx2 = 568 * S, W - 24 * S

    # --- Card 1: Island Summary + Wind Compass Dial (y: 98..282) ---
    c1_y1, c1_y2 = 98 * S, 282 * S
    draw.rounded_rectangle(
        [rx1, c1_y1, rx2, c1_y2],
        radius=16 * S,
        fill=(15, 31, 56, 245),
        outline=(34, 64, 102, 255),
        width=2 * S,
    )
    draw.text((rx1 + 18 * S, c1_y1 + 14 * S), "ISLAND OUTLOOK & WIND", fill=(125, 211, 252, 255), font=f_sec)

    # Left half: Overall Weather Icon + Peak Rain & Storm % KPIs
    overall_kind = _pick_weather_icon_kind(peak_rain, peak_storm, peak_wind_prob)
    _draw_weather_icon(draw, rx1 + 54 * S, c1_y1 + 92 * S, 64 * S, overall_kind, is_night=is_night)

    # Rain KPI
    draw.text((rx1 + 104 * S, c1_y1 + 52 * S), "PEAK RAIN", fill=(148, 163, 184, 255), font=f_kpi_lbl)
    draw.text((rx1 + 104 * S, c1_y1 + 66 * S), f"{peak_rain}%", fill=(56, 189, 248, 255), font=f_kpi)

    # Storm KPI
    draw.text((rx1 + 104 * S, c1_y1 + 114 * S), "PEAK STORM", fill=(148, 163, 184, 255), font=f_kpi_lbl)
    st_col = (248, 113, 113, 255) if peak_storm > 50 else ((250, 204, 21, 255) if peak_storm >= 30 else (226, 232, 240, 255))
    draw.text((rx1 + 104 * S, c1_y1 + 128 * S), f"{peak_storm}%", fill=st_col, font=f_kpi)

    # Vertical divider
    mid_x = rx1 + 226 * S
    draw.line([(mid_x, c1_y1 + 42 * S), (mid_x, c1_y2 - 16 * S)], fill=(34, 64, 102, 200), width=2 * S)

    # Right half: Wind Compass Dial + Speed/Direction/Gusts
    comp_cx, comp_cy = mid_x + 64 * S, c1_y1 + 106 * S
    _draw_wind_compass(draw, comp_cx, comp_cy, 48 * S, mean_wind_deg, f_xs)

    wx = mid_x + 126 * S
    draw.text((wx, c1_y1 + 52 * S), "WIND FLOW", fill=(148, 163, 184, 255), font=f_kpi_lbl)
    draw.text((wx, c1_y1 + 68 * S), f"{avg_wind_spd:.0f} km/h", fill=(248, 250, 252, 255), font=_load_font(22 * S, bold=True))
    draw.text((wx, c1_y1 + 98 * S), f"Dir: {mean_wind_comp} ({mean_wind_deg:.0f}°)", fill=(125, 211, 252, 255), font=f_body_b)
    draw.text((wx, c1_y1 + 122 * S), f"Gusts: {max_wind_gust:.0f} km/h", fill=(253, 224, 71, 255), font=f_body)
    max_wave = max((s["wave_m"] for s in sea_list if s["wave_m"] is not None), default=None)
    if max_wave is not None:
        draw.text((wx, c1_y1 + 144 * S), f"Seas: {max_wave:.1f} m", fill=(148, 175, 206, 255), font=f_sm)

    # --- Card 2: Top 6 Districts Table (y: 298..636) ---
    c2_y1, c2_y2 = 298 * S, 636 * S
    draw.rounded_rectangle(
        [rx1, c2_y1, rx2, c2_y2],
        radius=16 * S,
        fill=(15, 31, 56, 245),
        outline=(34, 64, 102, 255),
        width=2 * S,
    )
    draw.text((rx1 + 18 * S, c2_y1 + 14 * S), "TOP DISTRICTS — RAIN / STORM / WIND", fill=(125, 211, 252, 255), font=f_sec)

    # Column sub-headers
    col_y = c2_y1 + 40 * S
    draw.text((rx1 + 48 * S, col_y), "DISTRICT", fill=(100, 128, 160, 255), font=f_xs)
    draw.text((rx1 + 178 * S, col_y), "RAIN", fill=(100, 128, 160, 255), font=f_xs)
    draw.text((rx1 + 232 * S, col_y), "STORM", fill=(100, 128, 160, 255), font=f_xs)
    draw.text((rx1 + 292 * S, col_y), "WIND", fill=(100, 128, 160, 255), font=f_xs)
    draw.text((rx2 - 66 * S, col_y), "THREAT", fill=(100, 128, 160, 255), font=f_xs)
    draw.line([(rx1 + 16 * S, col_y + 18 * S), (rx2 - 16 * S, col_y + 18 * S)], fill=(30, 58, 92, 200), width=1 * S)

    sorted_districts = sorted(
        all_dist_vals,
        key=lambda d: (-threat_rank.get(d["threat"], 1), -d["max_prob"], -d["storm"], -d["rain"], d["name"]),
    )
    top_rows = sorted_districts[:6]
    if not top_rows:
        draw.text(
            ((rx1 + rx2) / 2, (c2_y1 + c2_y2) / 2),
            "All 25 districts currently clear / low risk",
            fill=(148, 163, 184, 255),
            font=f_body,
            anchor="mm",
        )
    else:
        row_h = 44 * S
        for idx, d in enumerate(top_rows):
            ry = col_y + 24 * S + idx * row_h
            if idx % 2 == 1:
                draw.rounded_rectangle(
                    [rx1 + 12 * S, ry, rx2 - 12 * S, ry + row_h - 4 * S],
                    radius=8 * S,
                    fill=(20, 40, 70, 150),
                )
            mid_y = ry + (row_h - 4 * S) / 2
            kind = _pick_weather_icon_kind(d["rain"], d["storm"], d["wind_prob"])
            _draw_weather_icon(draw, rx1 + 30 * S, mid_y, 24 * S, kind, is_night=is_night)

            dname = d["name"][:13]
            draw.text((rx1 + 48 * S, mid_y), dname, fill=(241, 245, 249, 255), font=f_body_b, anchor="lm")

            r_col = (56, 189, 248, 255) if d["rain"] >= 30 else (148, 163, 184, 255)
            s_col = (248, 113, 113, 255) if d["storm"] > 50 else ((250, 204, 21, 255) if d["storm"] >= 30 else (148, 163, 184, 255))
            draw.text((rx1 + 178 * S, mid_y), f"{d['rain']}%", fill=r_col, font=f_body_b, anchor="lm")
            draw.text((rx1 + 236 * S, mid_y), f"{d['storm']}%", fill=s_col, font=f_body_b, anchor="lm")

            w_txt = f"{d['wind_spd']:.0f} km/h {d['wind_comp']}"
            draw.text((rx1 + 292 * S, mid_y), w_txt, fill=(203, 213, 225, 255), font=f_sm, anchor="lm")

            # Threat badge pill
            tc = _THREAT_COLORS.get(d["threat"], (16, 185, 129))
            short_th = "MOD" if d["threat"] == "MODERATE" else ("SEV" if d["threat"] == "SEVERE" else d["threat"])
            bx1, bx2 = rx2 - 74 * S, rx2 - 18 * S
            by1, by2 = mid_y - 11 * S, mid_y + 11 * S
            draw.rounded_rectangle(
                [bx1, by1, bx2, by2],
                radius=6 * S,
                fill=(tc[0], tc[1], tc[2], 45),
                outline=(tc[0], tc[1], tc[2], 210),
                width=max(1, S),
            )
            draw.text(((bx1 + bx2) / 2, mid_y), short_th, fill=(248, 250, 252, 255), font=f_xs, anchor="mm")

    # --- Card 3: Wet-Zone River Flood Status Strip (y: 652..774) ---
    c3_y1, c3_y2 = 652 * S, 774 * S
    draw.rounded_rectangle(
        [rx1, c3_y1, rx2, c3_y2],
        radius=16 * S,
        fill=(15, 31, 56, 245),
        outline=(34, 64, 102, 255),
        width=2 * S,
    )
    draw.text((rx1 + 18 * S, c3_y1 + 14 * S), "WET-ZONE RIVER BASINS", fill=(125, 211, 252, 255), font=f_sec)

    basin_items: list[tuple[str, str, float]] = []
    for feat in (basins_geojson or {}).get("features", []) or []:
        props = feat.get("properties") or {}
        bname = str(props.get("name") or props.get("basin_id") or "").replace(" Ganga", "")
        brisk = str(props.get("risk_level") or "NORMAL").upper()
        brain = float(props.get("rainfall_24h_mm", 0.0) or 0.0)
        basin_items.append((bname, brisk, brain))

    if not basin_items:
        basin_items = [("Kelani", "NORMAL", 0.0), ("Kalu", "NORMAL", 0.0), ("Gin", "NORMAL", 0.0), ("Nilwala", "NORMAL", 0.0)]

    tile_gap = 10 * S
    tile_w = ((rx2 - rx1) - 36 * S - tile_gap) / 2
    tile_h = 34 * S
    for idx, (bname, brisk, brain) in enumerate(basin_items[:4]):
        col_i = idx % 2
        row_i = idx // 2
        tx1 = rx1 + 18 * S + col_i * (tile_w + tile_gap)
        ty1 = c3_y1 + 40 * S + row_i * (tile_h + 8 * S)
        tx2 = tx1 + tile_w
        ty2 = ty1 + tile_h
        bc = _BASIN_COLORS.get(brisk, (16, 185, 129))
        draw.rounded_rectangle(
            [tx1, ty1, tx2, ty2],
            radius=8 * S,
            fill=(10, 22, 42, 230),
            outline=(bc[0], bc[1], bc[2], 170),
            width=max(1, S),
        )
        tmid = (ty1 + ty2) / 2
        draw.ellipse([tx1 + 10 * S, tmid - 4 * S, tx1 + 18 * S, tmid + 4 * S], fill=(*bc, 255))
        draw.text((tx1 + 24 * S, tmid), bname[:9], fill=(241, 245, 249, 255), font=f_sm_b, anchor="lm")
        draw.text((tx2 - 10 * S, tmid), f"{brisk} • {brain:.0f}mm", fill=(*bc, 255), font=f_xs, anchor="rm")

    # =========================================================================
    # 4. BOTTOM LEGEND FOOTER BAR
    # =========================================================================
    ft_y1 = 790 * S
    draw.rectangle([0, ft_y1, W, H], fill=(9, 20, 38, 245))
    draw.line([(0, ft_y1), (W, ft_y1)], fill=(30, 58, 92, 255), width=2 * S)

    lx = 26 * S
    ly = (ft_y1 + H) / 2
    for lbl, col in [
        ("LOW (<30%)", (16, 185, 129)),
        ("MODERATE (30-50%)", (250, 204, 21)),
        ("HIGH (50-75%)", (249, 115, 22)),
        ("SEVERE (>75%)", (239, 68, 68)),
    ]:
        draw.rounded_rectangle([lx, ly - 6 * S, lx + 12 * S, ly + 6 * S], radius=3 * S, fill=(*col, 255))
        lx += 18 * S
        draw.text((lx, ly), lbl, fill=(203, 213, 225, 255), font=f_sm, anchor="lm")
        tb = draw.textbbox((0, 0), lbl, font=f_sm)
        lx += (tb[2] - tb[0]) + 22 * S

    draw.text(
        (W - 24 * S, ly),
        f"SL Edge Weather Report  •  AI: {engine_label}",
        fill=(148, 175, 206, 255),
        font=f_sm_b,
        anchor="rm",
    )

    # Downsample 2x -> 1x with LANCZOS for crisp anti-aliased edges and encode JPEG
    final_rgb = img.convert("RGB").resize((OUT_W, OUT_H), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    final_rgb.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    return buf.getvalue()
