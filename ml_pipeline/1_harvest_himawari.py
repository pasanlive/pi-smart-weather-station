#!/usr/bin/env python3
"""Step 1: Async/batched NASA GIBS WMS harvester for Himawari-9 Band 13 IR frames."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from utils import (
    DEFAULT_CONFIG_PATH,
    atomic_write_bytes,
    atomic_write_json,
    format_frame_filename,
    get_pipeline_paths,
    load_config,
    setup_logging,
)

LOGGER = setup_logging("step1.harvest")


def build_timestamp_schedule(
    start_date: date,
    end_date: date,
    step_hours: int,
    max_frames: int,
) -> list[datetime]:
    """Generate evenly spaced UTC timestamps across [start_date, end_date].

    When total timestamps exceed `max_frames`, uniformly subsamples across the full
    date span so synoptic and seasonal diversity is preserved.
    """
    if step_hours <= 0:
        raise ValueError("step_hours must be > 0")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if end_date < start_date:
        raise ValueError(f"end_date ({end_date}) cannot be earlier than start_date ({start_date})")

    start_dt = datetime(start_date.year, start_date.month, start_date.day, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, 23, 59, tzinfo=timezone.utc)

    all_ts: list[datetime] = []
    cur = start_dt
    delta = timedelta(hours=step_hours)
    while cur <= end_dt:
        all_ts.append(cur)
        cur += delta

    if len(all_ts) <= max_frames:
        return all_ts

    indices = np.linspace(0, len(all_ts) - 1, num=max_frames, dtype=int)
    # Deduplicate while preserving chronological order
    unique_indices = list(dict.fromkeys(indices.tolist()))
    return [all_ts[i] for i in unique_indices]


def build_wms_params(cfg: dict[str, Any], dt: datetime) -> dict[str, str]:
    """Build NASA GIBS WMS GetMap query parameters for a single UTC timestamp."""
    region = cfg["region"]
    harvest = cfg["harvest"]
    min_lon, min_lat, max_lon, max_lat = region["bbox"]
    version = str(harvest.get("wms_version", "1.1.1"))
    size = str(int(region["image_size"]))

    # WMS 1.1.1 uses SRS & lon,lat order; WMS 1.3.0 with EPSG:4326 flips to CRS & lat,lon
    if version.startswith("1.3"):
        crs_key = "CRS"
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    else:
        crs_key = "SRS"
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    return {
        "SERVICE": "WMS",
        "VERSION": version,
        "REQUEST": "GetMap",
        "LAYERS": harvest["layer"],
        "STYLES": "",
        crs_key: region["crs"],
        "BBOX": bbox_str,
        "WIDTH": size,
        "HEIGHT": size,
        "FORMAT": harvest.get("image_format", "image/png"),
        "TIME": dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:00Z"),
    }


def validate_png_bytes(
    data: bytes,
    expected_size: int,
    min_file_bytes: int = 4096,
    min_pixel_std: float = 3.0,
    black_pixel_threshold: int = 5,
    max_black_ratio: float = 0.95,
    max_transparent_ratio: float = 0.95,
) -> tuple[bool, str, dict[str, float]]:
    """Validate that downloaded bytes are a non-blank, non-corrupt IR PNG frame."""
    if len(data) < min_file_bytes:
        return False, f"file_too_small({len(data)}B<{min_file_bytes}B)", {}

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "cv2_decode_failed", {}

    height, width = img.shape[:2]
    if height != expected_size or width != expected_size:
        return False, f"wrong_dimensions({width}x{height}!={expected_size}x{expected_size})", {}

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        transparent_ratio = float(np.mean(alpha == 0))
        if transparent_ratio > max_transparent_ratio:
            return False, f"mostly_transparent({transparent_ratio:.3f})", {}
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    pixel_std = float(np.std(gray))
    pixel_mean = float(np.mean(gray))
    black_ratio = float(np.mean(gray <= black_pixel_threshold))

    metrics = {
        "bytes": float(len(data)),
        "mean": round(pixel_mean, 3),
        "std": round(pixel_std, 3),
        "black_ratio": round(black_ratio, 4),
    }

    if pixel_std < min_pixel_std:
        return False, f"low_variance(std={pixel_std:.2f}<{min_pixel_std})", metrics
    if black_ratio > max_black_ratio:
        return False, f"mostly_black(ratio={black_ratio:.3f}>{max_black_ratio})", metrics

    return True, "ok", metrics


async def _fetch_single_frame(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    cfg: dict[str, Any],
    dt: datetime,
    out_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Download, validate, and persist a single Himawari-9 frame with retries."""
    harvest = cfg["harvest"]
    expected_size = int(cfg["region"]["image_size"])
    filename = format_frame_filename(dt)
    dest = out_dir / filename
    ts_iso = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:00Z")

    val_kwargs = {
        "expected_size": expected_size,
        "min_file_bytes": int(harvest.get("min_file_bytes", 4096)),
        "min_pixel_std": float(harvest.get("min_pixel_std", 3.0)),
        "black_pixel_threshold": int(harvest.get("black_pixel_threshold", 5)),
        "max_black_ratio": float(harvest.get("max_black_ratio", 0.95)),
        "max_transparent_ratio": float(harvest.get("max_transparent_ratio", 0.95)),
    }

    # Resumability: verify existing file on disk before skipping
    if dest.exists() and not force:
        existing_bytes = dest.read_bytes()
        valid, reason, metrics = validate_png_bytes(existing_bytes, **val_kwargs)
        if valid:
            return {
                "filename": filename,
                "timestamp": ts_iso,
                "status": "resumed",
                "reason": "existing_valid",
                "metrics": metrics,
            }
        LOGGER.warning("Existing file %s failed validation (%s); re-downloading.", filename, reason)

    params = build_wms_params(cfg, dt)
    max_retries = int(harvest.get("max_retries", 3))
    backoff_base = float(harvest.get("retry_backoff_sec", 1.5))

    async with semaphore:
        last_reason = "unknown_error"
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.get(harvest["wms_url"], params=params)
                if response.status_code in {429, 500, 502, 503, 504}:
                    last_reason = f"http_{response.status_code}"
                    if attempt < max_retries:
                        await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
                        continue
                    break

                if response.status_code != 200:
                    last_reason = f"http_{response.status_code}"
                    break

                content_type = response.headers.get("content-type", "").lower()
                if "image/png" not in content_type:
                    last_reason = f"non_png_content_type({content_type or 'missing'})"
                    break

                valid, reason, metrics = validate_png_bytes(response.content, **val_kwargs)
                if not valid:
                    last_reason = reason
                    # Remove any stale/corrupt file on disk
                    if dest.exists():
                        dest.unlink()
                    return {
                        "filename": filename,
                        "timestamp": ts_iso,
                        "status": "discarded",
                        "reason": last_reason,
                        "metrics": metrics,
                    }

                atomic_write_bytes(dest, response.content)
                return {
                    "filename": filename,
                    "timestamp": ts_iso,
                    "status": "downloaded",
                    "reason": "ok",
                    "metrics": metrics,
                }
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_reason = f"network_error({type(exc).__name__})"
                if attempt < max_retries:
                    await asyncio.sleep(backoff_base * (2 ** (attempt - 1)))
                    continue
                break

    return {
        "filename": filename,
        "timestamp": ts_iso,
        "status": "failed",
        "reason": last_reason,
        "metrics": {},
    }


def resolve_date_range(
    cfg: dict[str, Any],
    start_date_arg: str | None = None,
    end_date_arg: str | None = None,
) -> tuple[date, date]:
    """Resolve harvest [start_date, end_date] from CLI args, config, or lookback default."""
    harvest = cfg["harvest"]
    end_raw = end_date_arg or harvest.get("end_date")
    start_raw = start_date_arg or harvest.get("start_date")

    if end_raw:
        end_d = date.fromisoformat(str(end_raw))
    else:
        end_d = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    if start_raw:
        start_d = date.fromisoformat(str(start_raw))
    else:
        lookback = int(harvest.get("default_lookback_days", 60))
        start_d = end_d - timedelta(days=max(1, lookback - 1))

    return start_d, end_d


async def harvest_himawari_async(
    cfg: dict[str, Any],
    start_date: date,
    end_date: date,
    step_hours: int,
    max_frames: int,
    concurrency: int,
    out_dir: Path,
    force: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Run async batched NASA GIBS WMS harvesting and write a summary manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamps = build_timestamp_schedule(start_date, end_date, step_hours, max_frames)
    LOGGER.info(
        "Harvesting %d Himawari-9 Band 13 frames (%s -> %s, step=%dh, concurrency=%d)",
        len(timestamps),
        start_date.isoformat(),
        end_date.isoformat(),
        step_hours,
        concurrency,
    )

    semaphore = asyncio.Semaphore(max(1, concurrency))
    timeout = httpx.Timeout(float(cfg["harvest"].get("timeout_sec", 30.0)))
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "follow_redirects": True,
        "headers": {"User-Agent": "pi-smart-weather-station-ml-pipeline/1.0"},
    }
    if transport is not None:
        client_kwargs["transport"] = transport

    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(**client_kwargs) as client:
        tasks = [
            _fetch_single_frame(client, semaphore, cfg, dt, out_dir, force=force)
            for dt in timestamps
        ]
        for idx, coro in enumerate(asyncio.as_completed(tasks), start=1):
            res = await coro
            results.append(res)
            if idx % 25 == 0 or idx == len(tasks):
                LOGGER.info("Progress: %d / %d frames processed", idx, len(tasks))

    results.sort(key=lambda item: item["timestamp"])
    counts = {
        "requested": len(timestamps),
        "downloaded": sum(1 for r in results if r["status"] == "downloaded"),
        "resumed": sum(1 for r in results if r["status"] == "resumed"),
        "discarded": sum(1 for r in results if r["status"] == "discarded"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
    }
    counts["usable_total"] = counts["downloaded"] + counts["resumed"]

    manifest = {
        "layer": cfg["harvest"]["layer"],
        "bbox": cfg["region"]["bbox"],
        "crs": cfg["region"]["crs"],
        "image_size": int(cfg["region"]["image_size"]),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "step_hours": step_hours,
        "max_frames": max_frames,
        "counts": counts,
        "frames": results,
    }
    manifest_path = out_dir / "harvest_manifest.json"
    atomic_write_json(manifest_path, manifest)

    LOGGER.info(
        "Harvest complete: usable=%d (downloaded=%d, resumed=%d), discarded=%d, failed=%d. Manifest: %s",
        counts["usable_total"],
        counts["downloaded"],
        counts["resumed"],
        counts["discarded"],
        counts["failed"],
        manifest_path,
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: Harvest historical Himawari-9 Band 13 IR frames from NASA GIBS WMS."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--start-date", type=str, default=None, help="Start UTC date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End UTC date (YYYY-MM-DD)")
    parser.add_argument("--step-hours", type=int, default=None, help="Sampling interval in hours (default: 2)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to fetch (default: 600)")
    parser.add_argument("--concurrency", type=int, default=None, help="Concurrent HTTP requests")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override raw_frames_dir")
    parser.add_argument("--force", action="store_true", help="Re-download existing valid files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = get_pipeline_paths(cfg)
    harvest = cfg["harvest"]

    start_d, end_d = resolve_date_range(cfg, args.start_date, args.end_date)
    step_hours = args.step_hours if args.step_hours is not None else int(harvest.get("step_hours", 2))
    max_frames = args.max_frames if args.max_frames is not None else int(harvest.get("max_frames", 600))
    concurrency = args.concurrency if args.concurrency is not None else int(harvest.get("concurrency", 8))
    out_dir = args.output_dir.resolve() if args.output_dir else paths["raw_frames_dir"]

    manifest = asyncio.run(
        harvest_himawari_async(
            cfg=cfg,
            start_date=start_d,
            end_date=end_d,
            step_hours=step_hours,
            max_frames=max_frames,
            concurrency=concurrency,
            out_dir=out_dir,
            force=args.force,
        )
    )
    return 0 if manifest["counts"]["usable_total"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
