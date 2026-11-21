from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import httpx
import numpy as np

from config import Settings
from ingestion.wms import build_wms_url, discover_latest_gibs_timestamp, format_gibs_timestamp
from models import FrameMetadata
from storage.db import Database

logger = logging.getLogger(__name__)


class HimawariPoller:
    """Polls NASA GIBS Himawari-9 Band 13 Clean IR frames and maintains a rolling 12-frame ring buffer."""

    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.frames_dir = settings.frames_path
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.bbox = settings.bbox_tuple

    @staticmethod
    def _is_valid_ir_frame(png_bytes: bytes) -> tuple[bool, Optional[np.ndarray], float]:
        """Validates that the PNG decodes to a non-empty 640x640 satellite image (not transparent/blank)."""
        if not png_bytes or len(png_bytes) < 512:
            return False, None, 0.0
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, None, 0.0
        # Reject fully transparent frames (GIBS returns transparent RGBA for missing/housekeeping slots)
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            non_transparent_ratio = float(np.mean(alpha > 10))
            if non_transparent_ratio < 0.25:
                return False, None, 0.0
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        std_val = float(np.std(gray))
        mean_val = float(np.mean(gray))
        if std_val < 2.0 and mean_val < 5.0:
            return False, None, mean_val
        return True, img, mean_val

    def _filename_for_ts(self, iso_ts: str) -> str:
        safe = iso_ts.replace(":", "").replace("-", "").replace("+", "_")
        return f"himawari_b13_{safe}.png"

    async def fetch_slot(self, client: httpx.AsyncClient, slot_dt: datetime) -> Optional[FrameMetadata]:
        iso_ts = format_gibs_timestamp(slot_dt)
        filename = self._filename_for_ts(iso_ts)
        file_path = self.frames_dir / filename

        # Skip download if already buffered on disk
        existing = {f["timestamp"]: f for f in self.db.list_frames(limit=self.settings.satellite_ring_buffer_size)}
        if iso_ts in existing and file_path.exists():
            row = existing[iso_ts]
            return FrameMetadata(
                timestamp=row["timestamp"],
                filename=row["filename"],
                image_url=f"/api/frames/{row['filename']}",
                width=row["width"],
                height=row["height"],
                mean_intensity=row["mean_intensity"],
                fetched_at=row["fetched_at"],
            )

        url = build_wms_url(timestamp=iso_ts, bbox=self.bbox, width=640, height=640)
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            if "image/png" not in resp.headers.get("content-type", ""):
                logger.warning("Unexpected GIBS content-type for %s: %s", iso_ts, resp.headers.get("content-type"))
                return None

            valid, img, mean_intensity = self._is_valid_ir_frame(resp.content)
            if not valid or img is None:
                logger.debug("Skipping blank/housekeeping GIBS frame for %s", iso_ts)
                return None

            file_path.write_bytes(resp.content)
            fetched_at = datetime.now(timezone.utc).isoformat()
            h, w = img.shape[:2]
            self.db.upsert_frame(
                timestamp=iso_ts,
                filename=filename,
                file_path=str(file_path),
                width=w,
                height=h,
                mean_intensity=round(mean_intensity, 2),
                fetched_at=fetched_at,
            )
            self._prune_ring_buffer()
            logger.info("Buffered Himawari-9 IR frame %s (mean=%.1f)", iso_ts, mean_intensity)
            return FrameMetadata(
                timestamp=iso_ts,
                filename=filename,
                image_url=f"/api/frames/{filename}",
                width=w,
                height=h,
                mean_intensity=round(mean_intensity, 2),
                fetched_at=fetched_at,
            )
        except Exception as exc:
            logger.warning("Error fetching Himawari frame %s: %s", iso_ts, exc)
            return None

    def _prune_ring_buffer(self) -> None:
        pruned_paths = self.db.prune_frames(keep=self.settings.satellite_ring_buffer_size)
        for p in pruned_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass

    async def _anchor_latest_time(self, client: httpx.AsyncClient) -> datetime:
        latest = await discover_latest_gibs_timestamp(client=client)
        if latest is not None:
            return latest
        # Fallback: start probing 90 minutes behind UTC now
        now = datetime.now(timezone.utc) - timedelta(minutes=90)
        floored = (now.minute // 10) * 10
        return now.replace(minute=floored, second=0, microsecond=0)

    async def backfill(self, target_frames: Optional[int] = None) -> list[FrameMetadata]:
        """Backfills the latest N (default 6) valid Himawari-9 frames spaced 10-20 minutes apart."""
        count = target_frames or self.settings.satellite_backfill_frames
        collected: list[FrameMetadata] = []
        async with httpx.AsyncClient(timeout=25.0) as client:
            anchor = await self._anchor_latest_time(client)
            # Walk backward in 10-min increments up to 4 hours until `count` valid frames are buffered
            for step in range(24):
                if len(collected) >= count:
                    break
                slot = anchor - timedelta(minutes=step * 10)
                meta = await self.fetch_slot(client, slot)
                if meta is not None:
                    collected.append(meta)

        collected.sort(key=lambda m: m.timestamp)
        logger.info("Himawari backfill complete: %d valid frames buffered", len(collected))
        return collected

    async def poll_latest(self) -> Optional[FrameMetadata]:
        """Fetches the latest available Himawari-9 frame, probing backward if the newest slot is a housekeeping gap."""
        async with httpx.AsyncClient(timeout=25.0) as client:
            anchor = await self._anchor_latest_time(client)
            for step in range(6):
                slot = anchor - timedelta(minutes=step * 10)
                meta = await self.fetch_slot(client, slot)
                if meta is not None:
                    return meta
        return None

    def get_buffered_frames(self) -> list[FrameMetadata]:
        rows = self.db.list_frames(limit=self.settings.satellite_ring_buffer_size)
        out: list[FrameMetadata] = []
        for r in rows:
            if Path(r["file_path"]).exists():
                out.append(
                    FrameMetadata(
                        timestamp=r["timestamp"],
                        filename=r["filename"],
                        image_url=f"/api/frames/{r['filename']}",
                        width=r["width"],
                        height=r["height"],
                        mean_intensity=r["mean_intensity"],
                        fetched_at=r["fetched_at"],
                    )
                )
        return out

    def load_recent_bgr_frames(self, count: int = 3) -> list[tuple[str, np.ndarray]]:
        """Loads up to the last `count` chronological frames as 640x640 BGR numpy arrays."""
        rows = self.db.list_frames(limit=self.settings.satellite_ring_buffer_size)
        valid_rows = [r for r in rows if Path(r["file_path"]).exists()]
        recent = valid_rows[-count:]
        loaded: list[tuple[str, np.ndarray]] = []
        for r in recent:
            img = cv2.imread(r["file_path"], cv2.IMREAD_COLOR)
            if img is not None:
                if img.shape[0] != 640 or img.shape[1] != 640:
                    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
                loaded.append((r["timestamp"], img))
        return loaded
