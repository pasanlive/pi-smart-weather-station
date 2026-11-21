#!/usr/bin/env python3
"""Step 3: Temporally grouped 80/20 stratified splitter, data.yaml writer, and .npy calibration builder."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

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

LOGGER = setup_logging("step3.prepare")


def _frame_stratum(rec: dict[str, Any]) -> str:
    """Assign each frame to a storm severity stratum for balanced splitting and calibration."""
    counts = rec.get("class_counts", {})
    if int(counts.get("0", 0)) > 0:
        return "deep_convective"
    if int(counts.get("1", 0)) > 0:
        return "storm_shield"
    return "clear_sky"


def _record_timestamp(rec: dict[str, Any], fallback_idx: int) -> datetime:
    ts_str = rec.get("timestamp")
    if ts_str:
        return datetime.fromisoformat(ts_str).astimezone(timezone.utc)
    parsed = parse_frame_timestamp(rec["filename"])
    if parsed:
        return parsed
    # Deterministic synthetic fallback far in the past if filename lacks timestamp
    return datetime(2000, 1, 1, tzinfo=timezone.utc)


def temporal_stratified_split(
    records: list[dict[str, Any]],
    train_ratio: float = 0.80,
    block_hours: int = 24,
    min_temporal_gap_hours: float = 4.0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split frames into train/val using contiguous temporal blocks to prevent leakage.

    1. Groups frames into fixed `block_hours` UTC buckets so snapshots from the same
       storm window stay together.
    2. Stratifies blocks by dominant storm class (`deep_convective` > `storm_shield` > `clear_sky`)
       and greedily assigns whole blocks to `val` until `~1 - train_ratio` of frames are allocated.
    3. Enforces `min_temporal_gap_hours` across any `train <-> val` transition by quarantining
       boundary `val` frames that sit too close in time to a `train` frame.

    Returns:
        (train_records, val_records, quarantined_boundary_records)
    """
    if not records:
        return [], [], []
    if len(records) == 1:
        return list(records), [], []

    rng = random.Random(seed)
    block_sec = max(1, int(block_hours)) * 3600

    enriched = []
    for idx, rec in enumerate(records):
        dt = _record_timestamp(rec, idx)
        bucket_id = int(dt.timestamp() // block_sec)
        enriched.append((dt, bucket_id, rec))
    enriched.sort(key=lambda item: item[0])

    blocks_map: dict[int, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for dt, bucket_id, rec in enriched:
        blocks_map[bucket_id].append((dt, rec))

    blocks: list[dict[str, Any]] = []
    for bucket_id in sorted(blocks_map.keys()):
        items = blocks_map[bucket_id]
        strata = [_frame_stratum(r) for _, r in items]
        if "deep_convective" in strata:
            dominant = "deep_convective"
        elif "storm_shield" in strata:
            dominant = "storm_shield"
        else:
            dominant = "clear_sky"
        blocks.append(
            {
                "bucket_id": bucket_id,
                "dominant": dominant,
                "items": items,
                "count": len(items),
            }
        )

    # Edge case: if all frames fell into a single temporal block, do a chronological 80/20 cut
    if len(blocks) == 1:
        items = blocks[0]["items"]
        cut = max(1, min(len(items) - 1, int(round(len(items) * train_ratio))))
        train_items = items[:cut]
        val_items = items[cut:]
    else:
        total_frames = len(records)
        target_val_frames = max(1, int(round(total_frames * (1.0 - train_ratio))))

        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for blk in blocks:
            by_stratum[blk["dominant"]].append(blk)

        val_bucket_ids: set[int] = set()
        val_count = 0

        # Allocate proportionally within each stratum first
        for stratum_name in ("deep_convective", "storm_shield", "clear_sky"):
            stratum_blocks = by_stratum.get(stratum_name, [])
            if not stratum_blocks:
                continue
            stratum_total = sum(b["count"] for b in stratum_blocks)
            stratum_val_target = int(round(stratum_total * (1.0 - train_ratio)))
            if stratum_val_target == 0 and len(stratum_blocks) >= 4:
                stratum_val_target = min(b["count"] for b in stratum_blocks)

            shuffled = list(stratum_blocks)
            rng.shuffle(shuffled)
            stratum_val_acc = 0
            for blk in shuffled:
                if stratum_val_acc >= stratum_val_target:
                    break
                # Keep at least 1 block in train globally
                if len(val_bucket_ids) + 1 >= len(blocks):
                    break
                val_bucket_ids.add(blk["bucket_id"])
                stratum_val_acc += blk["count"]
                val_count += blk["count"]

        # Top up val if overall count is still short
        if val_count < target_val_frames:
            remaining = [b for b in blocks if b["bucket_id"] not in val_bucket_ids]
            rng.shuffle(remaining)
            for blk in remaining:
                if val_count >= target_val_frames or len(val_bucket_ids) + 1 >= len(blocks):
                    break
                val_bucket_ids.add(blk["bucket_id"])
                val_count += blk["count"]

        train_items = []
        val_items = []
        for blk in blocks:
            if blk["bucket_id"] in val_bucket_ids:
                val_items.extend(blk["items"])
            else:
                train_items.extend(blk["items"])

    # Enforce strict minimum temporal gap across train <-> val boundaries
    quarantined: list[dict[str, Any]] = []
    if min_temporal_gap_hours > 0 and train_items and val_items:
        gap_sec = min_temporal_gap_hours * 3600.0
        train_epochs = np.array([dt.timestamp() for dt, _ in train_items], dtype=np.float64)
        filtered_val: list[tuple[datetime, dict[str, Any]]] = []
        for dt, rec in val_items:
            min_dist = float(np.min(np.abs(train_epochs - dt.timestamp())))
            if min_dist < gap_sec:
                quarantined.append(rec)
            else:
                filtered_val.append((dt, rec))

        # Never leave val completely empty if quarantining dropped every val frame
        if filtered_val:
            val_items = filtered_val
        else:
            LOGGER.warning(
                "Boundary gap guard (%.1fh) would empty validation split; relaxing guard for this dataset.",
                min_temporal_gap_hours,
            )
            quarantined.clear()

    train_records = [r for _, r in sorted(train_items, key=lambda x: x[0])]
    val_records = [r for _, r in sorted(val_items, key=lambda x: x[0])]
    return train_records, val_records, quarantined


def build_calibration_set(
    train_records: list[dict[str, Any]],
    raw_frames_dir: Path,
    calibration_dir: Path,
    image_size: int = 640,
    calib_num_frames: int = 256,
    calib_min_frames: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Export uncropped RGB uint8 (N, H, W, 3) calibration .npy tensor + image copies."""
    if not train_records:
        raise ValueError("Cannot build calibration set from an empty training split.")

    rng = random.Random(seed)
    target_n = min(calib_num_frames, len(train_records))
    if target_n < calib_min_frames:
        LOGGER.warning(
            "Only %d training frames available for calibration (recommended: %d-%d).",
            target_n,
            calib_min_frames,
            calib_num_frames,
        )

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in train_records:
        by_stratum[_frame_stratum(rec)].append(rec)

    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    total_train = len(train_records)

    for stratum in ("deep_convective", "storm_shield", "clear_sky"):
        pool = by_stratum.get(stratum, [])
        if not pool:
            continue
        quota = max(1, int(round(target_n * (len(pool) / float(total_train)))))
        picks = rng.sample(pool, min(quota, len(pool)))
        for rec in picks:
            if rec["filename"] not in selected_names and len(selected) < target_n:
                selected.append(rec)
                selected_names.add(rec["filename"])

    if len(selected) < target_n:
        leftover = [r for r in train_records if r["filename"] not in selected_names]
        extra = rng.sample(leftover, min(target_n - len(selected), len(leftover)))
        selected.extend(extra)

    selected.sort(key=lambda r: r["filename"])

    calibration_dir.mkdir(parents=True, exist_ok=True)
    calib_images_dir = calibration_dir / "calib_images"
    if calib_images_dir.exists():
        shutil.rmtree(calib_images_dir)
    calib_images_dir.mkdir(parents=True, exist_ok=True)

    frames_rgb: list[np.ndarray] = []
    exported_names: list[str] = []
    for rec in selected:
        src = raw_frames_dir / rec["filename"]
        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            LOGGER.warning("Skipping unreadable calibration frame: %s", src)
            continue
        if bgr.shape[0] != image_size or bgr.shape[1] != image_size:
            bgr = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
        exported_names.append(rec["filename"])
        shutil.copy2(src, calib_images_dir / rec["filename"])

    calib_array = np.stack(frames_rgb, axis=0).astype(np.uint8)
    npy_path = calibration_dir / "calib_set.npy"
    np.save(npy_path, calib_array)

    manifest = {
        "calib_npy": str(npy_path),
        "calib_images_dir": str(calib_images_dir),
        "shape": list(calib_array.shape),
        "dtype": str(calib_array.dtype),
        "color_order": "RGB",
        "cropped": False,
        "num_frames": int(calib_array.shape[0]),
        "strata_counts": {
            s: sum(1 for r in selected if _frame_stratum(r) == s)
            for s in ("deep_convective", "storm_shield", "clear_sky")
        },
        "filenames": exported_names,
    }
    atomic_write_json(calibration_dir / "calib_manifest.json", manifest)
    LOGGER.info(
        "Exported calibration set: %s | shape=%s dtype=%s strata=%s",
        npy_path,
        tuple(calib_array.shape),
        calib_array.dtype,
        manifest["strata_counts"],
    )
    return manifest


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_dataset(
    cfg: dict[str, Any],
    raw_frames_dir: Path,
    autolabel_dir: Path,
    dataset_dir: Path,
    calibration_dir: Path,
    calib_num_frames_override: int | None = None,
) -> dict[str, Any]:
    """Build Ultralytics dataset directory, data.yaml, and Hailo-8 .npy calibration set."""
    ds_cfg = cfg["dataset"]
    seed = int(ds_cfg.get("seed", 42))
    set_seed(seed)

    manifest_path = autolabel_dir / "label_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Auto-label manifest not found at {manifest_path}. Run Step 2 first."
        )

    al_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    retained_records = al_manifest.get("retained_frames", [])
    labels_src_dir = Path(al_manifest["labels_dir"])
    if not retained_records:
        raise ValueError("No retained frames found in label_manifest.json.")

    train_records, val_records, quarantined = temporal_stratified_split(
        records=retained_records,
        train_ratio=float(ds_cfg.get("train_ratio", 0.80)),
        block_hours=int(ds_cfg.get("temporal_block_hours", 24)),
        min_temporal_gap_hours=float(ds_cfg.get("min_temporal_gap_hours", 4.0)),
        seed=seed,
    )

    for split_name, split_records in (("train", train_records), ("val", val_records)):
        img_dst = dataset_dir / "images" / split_name
        lbl_dst = dataset_dir / "labels" / split_name
        _reset_dir(img_dst)
        _reset_dir(lbl_dst)

        for rec in split_records:
            src_img = raw_frames_dir / rec["filename"]
            src_lbl = labels_src_dir / f"{rec['stem']}.txt"
            if not src_img.exists():
                raise FileNotFoundError(f"Missing image: {src_img}")
            if not src_lbl.exists():
                raise FileNotFoundError(f"Missing label: {src_lbl}")
            shutil.copy2(src_img, img_dst / rec["filename"])
            shutil.copy2(src_lbl, lbl_dst / f"{rec['stem']}.txt")

    names_map = {
        int(cid): str(spec["name"])
        for cid, spec in sorted(cfg["autolabel"]["classes"].items(), key=lambda kv: int(kv[0]))
    }
    data_yaml_payload = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names_map,
    }
    data_yaml_path = dataset_dir / "data.yaml"
    atomic_write_text(data_yaml_path, yaml.safe_dump(data_yaml_payload, sort_keys=False))

    calib_n = (
        calib_num_frames_override
        if calib_num_frames_override is not None
        else int(ds_cfg.get("calib_num_frames", 256))
    )
    calib_manifest = build_calibration_set(
        train_records=train_records,
        raw_frames_dir=raw_frames_dir,
        calibration_dir=calibration_dir,
        image_size=int(cfg["region"]["image_size"]),
        calib_num_frames=calib_n,
        calib_min_frames=int(ds_cfg.get("calib_min_frames", 128)),
        seed=seed,
    )

    def _strata_summary(recs: list[dict[str, Any]]) -> dict[str, int]:
        return {
            s: sum(1 for r in recs if _frame_stratum(r) == s)
            for s in ("deep_convective", "storm_shield", "clear_sky")
        }

    summary = {
        "data_yaml": str(data_yaml_path),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "quarantined_boundary_count": len(quarantined),
        "train_strata": _strata_summary(train_records),
        "val_strata": _strata_summary(val_records),
        "calibration": {
            "calib_npy": calib_manifest["calib_npy"],
            "shape": calib_manifest["shape"],
            "num_frames": calib_manifest["num_frames"],
        },
    }
    atomic_write_json(dataset_dir / "split_summary.json", summary)
    LOGGER.info(
        "Dataset prepared: train=%d %s | val=%d %s | quarantined=%d | data.yaml=%s",
        summary["train_count"],
        summary["train_strata"],
        summary["val_count"],
        summary["val_strata"],
        summary["quarantined_boundary_count"],
        data_yaml_path,
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3: Temporally grouped 80/20 split, data.yaml writer & .npy calibration builder."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--calib-frames", type=int, default=None, help="Calibration frame count (128-256)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_pipeline_paths(cfg)
    summary = prepare_dataset(
        cfg=cfg,
        raw_frames_dir=paths["raw_frames_dir"],
        autolabel_dir=paths["autolabel_dir"],
        dataset_dir=paths["dataset_dir"],
        calibration_dir=paths["calibration_dir"],
        calib_num_frames_override=args.calib_frames,
    )
    return 0 if summary["train_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
