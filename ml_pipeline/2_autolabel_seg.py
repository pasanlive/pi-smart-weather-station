#!/usr/bin/env python3
"""Step 2: OpenCV IR brightness-temperature thresholding, YOLOv8-seg polygon extraction & QA grid."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils import (
    DEFAULT_CONFIG_PATH,
    atomic_write_json,
    atomic_write_text,
    get_pipeline_paths,
    load_config,
    parse_frame_timestamp,
    set_seed,
    setup_logging,
)

LOGGER = setup_logging("step2.autolabel")


def extract_polygons_from_gray(
    gray: np.ndarray,
    classes_cfg: dict[int | str, dict[str, Any]],
    morph_open_kernel: int = 3,
    morph_close_kernel: int = 5,
    approx_eps_ratio: float = 0.005,
    min_polygon_points: int = 3,
) -> tuple[list[tuple[int, np.ndarray]], dict[int, int]]:
    """Segment grayscale IR image into class polygons normalized to [0.0, 1.0].

    Returns:
        polygons: list of `(class_id, normalized_xy_array)` with shape `(M, 2)` float64 in `[0, 1]`.
        class_counts: mapping `class_id -> instance_count`.
    """
    height, width = gray.shape[:2]
    open_se = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_kernel, morph_open_kernel))
        if morph_open_kernel > 1
        else None
    )
    close_se = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel))
        if morph_close_kernel > 1
        else None
    )

    polygons: list[tuple[int, np.ndarray]] = []
    class_counts: dict[int, int] = {}

    for raw_cid in sorted(classes_cfg.keys(), key=lambda x: int(x)):
        cid = int(raw_cid)
        spec = classes_cfg[raw_cid]
        min_i = int(spec["min_intensity"])
        max_i = int(spec["max_intensity"])
        min_area = float(spec["min_area_px"])
        class_counts[cid] = 0

        mask = cv2.inRange(gray, min_i, max_i)
        if open_se is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_se)
        if close_se is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_se)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue
            epsilon = approx_eps_ratio * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            pts = approx.reshape(-1, 2)
            if len(pts) < min_polygon_points:
                continue

            norm_pts = np.empty_like(pts, dtype=np.float64)
            norm_pts[:, 0] = np.clip(pts[:, 0] / float(width), 0.0, 1.0)
            norm_pts[:, 1] = np.clip(pts[:, 1] / float(height), 0.0, 1.0)
            polygons.append((cid, norm_pts))
            class_counts[cid] += 1

    return polygons, class_counts


def format_yolo_seg_lines(polygons: list[tuple[int, np.ndarray]]) -> str:
    """Serialize `(class_id, normalized_pts)` entries into YOLOv8-seg label text."""
    lines: list[str] = []
    for cid, norm_pts in polygons:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm_pts)
        lines.append(f"{cid} {coords}")
    return ("\n".join(lines) + "\n") if lines else ""


def render_qa_montage(
    records: list[dict[str, Any]],
    raw_frames_dir: Path,
    classes_cfg: dict[int | str, dict[str, Any]],
    output_path: Path,
    num_samples: int = 16,
    seed: int = 42,
    tile_size: int = 320,
) -> Path | None:
    """Render a 4x4 (or ceil-sqrt grid) QA visual overlay of auto-labeled polygons."""
    if not records:
        LOGGER.warning("No retained frames available for QA montage.")
        return None

    rng = random.Random(seed)
    deep_frames = [r for r in records if r["class_counts"].get("0", 0) > 0]
    shield_only = [
        r
        for r in records
        if r["class_counts"].get("0", 0) == 0 and r["class_counts"].get("1", 0) > 0
    ]
    clear_frames = [r for r in records if not r["is_positive"]]

    # Stratified QA selection so deep convective, storm shield, and clear-sky are all visible
    selected: list[dict[str, Any]] = []
    target = min(num_samples, len(records))
    quotas = [
        (deep_frames, max(1, int(round(target * 0.55)))),
        (shield_only, max(1, int(round(target * 0.30)))),
        (clear_frames, max(1, target - int(round(target * 0.55)) - int(round(target * 0.30)))),
    ]
    used_names: set[str] = set()
    for pool, quota in quotas:
        if not pool:
            continue
        picks = rng.sample(pool, min(quota, len(pool)))
        for item in picks:
            if item["filename"] not in used_names and len(selected) < target:
                selected.append(item)
                used_names.add(item["filename"])

    if len(selected) < target:
        remaining = [r for r in records if r["filename"] not in used_names]
        extra = rng.sample(remaining, min(target - len(selected), len(remaining)))
        selected.extend(extra)

    cols = int(math.ceil(math.sqrt(len(selected))))
    rows = int(math.ceil(len(selected) / cols))
    header_h = 44
    canvas = np.full((header_h + rows * tile_size, cols * tile_size, 3), 20, dtype=np.uint8)

    # Header legend banner
    cv2.putText(
        canvas,
        f"Himawari-9 Band 13 Auto-Label QA ({len(selected)} samples)",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    legend_x = max(420, canvas.shape[1] - 430)
    for idx, raw_cid in enumerate(sorted(classes_cfg.keys(), key=lambda x: int(x))):
        spec = classes_cfg[raw_cid]
        color = tuple(int(c) for c in spec.get("qa_color_bgr", [0, 255, 0]))
        x0 = legend_x + idx * 205
        cv2.rectangle(canvas, (x0, 13), (x0 + 18, 31), color, -1)
        cv2.putText(
            canvas,
            f"{raw_cid}: {spec['name']}",
            (x0 + 24, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    for slot, rec in enumerate(selected):
        r_idx = slot // cols
        c_idx = slot % cols
        img_path = raw_frames_dir / rec["filename"]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        overlay = img.copy()
        for poly in rec["_polygons"]:
            cid, norm_pts = poly
            spec = classes_cfg.get(cid, classes_cfg.get(str(cid), {}))
            color = tuple(int(c) for c in spec.get("qa_color_bgr", [0, 255, 0]))
            px_pts = np.round(norm_pts * np.array([w, h])).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [px_pts], color)
            cv2.polylines(img, [px_pts], isClosed=True, color=color, thickness=2)

        blended = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
        tile = cv2.resize(blended, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

        # Dark caption strip
        cv2.rectangle(tile, (0, 0), (tile_size, 24), (15, 15, 15), -1)
        dc_n = rec["class_counts"].get("0", 0)
        ss_n = rec["class_counts"].get("1", 0)
        tag = f"DC:{dc_n} SS:{ss_n}" if rec["is_positive"] else "CLEAR-SKY"
        short_name = rec["filename"].replace("himawari_b13_", "").replace(".png", "")
        cv2.putText(
            tile,
            f"{short_name} | {tag}",
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), (70, 70, 70), 1)

        y0 = header_h + r_idx * tile_size
        x0 = c_idx * tile_size
        canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    LOGGER.info("Wrote QA visual montage (%d tiles) -> %s", len(selected), output_path)
    return output_path


def autolabel_frames(
    cfg: dict[str, Any],
    raw_frames_dir: Path,
    autolabel_dir: Path,
    qa_output_path: Path | None = None,
) -> dict[str, Any]:
    """Run Step 2 auto-labeling across all harvested PNGs and generate `qa_preview.png`."""
    al_cfg = cfg["autolabel"]
    seed = int(al_cfg.get("seed", 42))
    set_seed(seed)
    rng = random.Random(seed)

    png_files = sorted(raw_frames_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG frames found in {raw_frames_dir}. Run Step 1 first.")

    labels_dir = autolabel_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale labels from previous runs for reproducibility
    for stale in labels_dir.glob("*.txt"):
        stale.unlink()

    classes_cfg = al_cfg["classes"]
    open_k = int(al_cfg.get("morph_open_kernel", 3))
    close_k = int(al_cfg.get("morph_close_kernel", 5))
    eps_ratio = float(al_cfg.get("approx_eps_ratio", 0.005))
    min_pts = int(al_cfg.get("min_polygon_points", 3))
    neg_keep_ratio = float(al_cfg.get("negative_keep_ratio", 0.10))

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    unreadable = 0

    for png_path in png_files:
        gray = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            LOGGER.warning("Skipping unreadable image: %s", png_path.name)
            unreadable += 1
            continue

        polygons, class_counts = extract_polygons_from_gray(
            gray=gray,
            classes_cfg=classes_cfg,
            morph_open_kernel=open_k,
            morph_close_kernel=close_k,
            approx_eps_ratio=eps_ratio,
            min_polygon_points=min_pts,
        )
        dt = parse_frame_timestamp(png_path.name)
        is_positive = len(polygons) > 0
        record = {
            "filename": png_path.name,
            "stem": png_path.stem,
            "timestamp": dt.isoformat() if dt else None,
            "is_positive": is_positive,
            "num_instances": len(polygons),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "_polygons": polygons,
        }
        if is_positive:
            positives.append(record)
        else:
            negatives.append(record)

    # Retain ~10% of the final retained dataset as negative (clear-sky) frames
    if positives and 0.0 < neg_keep_ratio < 1.0:
        desired_neg = int(round(len(positives) * (neg_keep_ratio / (1.0 - neg_keep_ratio))))
        keep_neg_count = min(len(negatives), max(1 if negatives else 0, desired_neg))
    elif not positives and negatives:
        keep_neg_count = max(1, int(round(len(negatives) * max(neg_keep_ratio, 0.1))))
    else:
        keep_neg_count = 0

    kept_negatives = rng.sample(negatives, keep_neg_count) if keep_neg_count > 0 else []
    kept_neg_names = {r["filename"] for r in kept_negatives}

    retained = sorted(positives + kept_negatives, key=lambda r: r["filename"])
    dropped_negatives = [r for r in negatives if r["filename"] not in kept_neg_names]

    total_instances_by_class: dict[str, int] = {str(int(k)): 0 for k in classes_cfg.keys()}
    for rec in retained:
        txt_path = labels_dir / f"{rec['stem']}.txt"
        atomic_write_text(txt_path, format_yolo_seg_lines(rec["_polygons"]))
        for cid_str, count in rec["class_counts"].items():
            total_instances_by_class[cid_str] = total_instances_by_class.get(cid_str, 0) + count

    qa_path = qa_output_path if qa_output_path else (autolabel_dir / "qa_preview.png")
    render_qa_montage(
        records=retained,
        raw_frames_dir=raw_frames_dir,
        classes_cfg=classes_cfg,
        output_path=qa_path,
        num_samples=int(al_cfg.get("qa_grid_samples", 16)),
        seed=seed,
    )

    manifest = {
        "raw_frames_dir": str(raw_frames_dir),
        "labels_dir": str(labels_dir),
        "qa_preview": str(qa_path),
        "summary": {
            "total_scanned": len(png_files),
            "unreadable": unreadable,
            "positive_frames": len(positives),
            "total_clear_sky_frames": len(negatives),
            "retained_clear_sky_frames": len(kept_negatives),
            "dropped_clear_sky_frames": len(dropped_negatives),
            "total_retained_frames": len(retained),
            "retained_negative_fraction": round(
                len(kept_negatives) / max(1, len(retained)), 4
            ),
            "instances_by_class": total_instances_by_class,
        },
        "retained_frames": [
            {k: v for k, v in rec.items() if not k.startswith("_")} for rec in retained
        ],
    }
    manifest_path = autolabel_dir / "label_manifest.json"
    atomic_write_json(manifest_path, manifest)

    LOGGER.info(
        "Auto-labeling complete: retained=%d (positive=%d, clear_sky=%d [%.1f%%]), "
        "instances=%s. Manifest: %s",
        manifest["summary"]["total_retained_frames"],
        manifest["summary"]["positive_frames"],
        manifest["summary"]["retained_clear_sky_frames"],
        manifest["summary"]["retained_negative_fraction"] * 100.0,
        total_instances_by_class,
        manifest_path,
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Auto-label Himawari-9 IR frames into YOLOv8-seg polygon format."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--input-dir", type=Path, default=None, help="Override raw_frames_dir")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override autolabel_dir")
    parser.add_argument("--qa-output", type=Path, default=None, help="Override qa_preview.png output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_pipeline_paths(cfg)

    raw_dir = args.input_dir.resolve() if args.input_dir else paths["raw_frames_dir"]
    out_dir = args.output_dir.resolve() if args.output_dir else paths["autolabel_dir"]
    qa_path = args.qa_output.resolve() if args.qa_output else None

    manifest = autolabel_frames(cfg, raw_dir, out_dir, qa_path)
    return 0 if manifest["summary"]["total_retained_frames"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
