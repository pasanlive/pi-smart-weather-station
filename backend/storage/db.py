from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


class Database:
    """Thread-safe SQLite store for ring-buffer frames, nowcast snapshots, alerts, and 72h gauge history."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS frames (
                    timestamp TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    width INTEGER NOT NULL DEFAULT 640,
                    height INTEGER NOT NULL DEFAULT 640,
                    mean_intensity REAL NOT NULL DEFAULT 0.0,
                    fetched_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS nowcast_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_timestamp TEXT,
                    backend_used TEXT NOT NULL,
                    inference_ms REAL NOT NULL DEFAULT 0.0,
                    advisory_text TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_key TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    sent_at TEXT NOT NULL,
                    webhook_status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_alert_entity_time
                    ON alert_history(entity_key, sent_at);

                CREATE TABLE IF NOT EXISTS gauge_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT NOT NULL,
                    station_name TEXT NOT NULL,
                    river_name TEXT NOT NULL,
                    basin_id TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    water_level_ft REAL NOT NULL,
                    alert_level_ft REAL,
                    minor_flood_ft REAL,
                    major_flood_ft REAL,
                    rainfall_24h_mm REAL,
                    reading_timestamp TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    UNIQUE(station_id, reading_timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_gauge_station_time
                    ON gauge_readings(station_id, reading_timestamp);

                CREATE TABLE IF NOT EXISTS active_learning_samples (
                    id TEXT PRIMARY KEY,
                    frame_timestamp TEXT NOT NULL UNIQUE,
                    captured_at TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_detail TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    model_filename TEXT NOT NULL,
                    min_confidence REAL,
                    max_confidence REAL,
                    detection_count INTEGER NOT NULL DEFAULT 0,
                    dir_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    meta_json TEXT NOT NULL,
                    reviewed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_al_status_captured
                    ON active_learning_samples(status, captured_at DESC);
                CREATE INDEX IF NOT EXISTS idx_al_captured_at
                    ON active_learning_samples(captured_at);
                """
            )

    # ------------------------------------------------------------------ Frames
    def upsert_frame(
        self,
        timestamp: str,
        filename: str,
        file_path: str,
        width: int,
        height: int,
        mean_intensity: float,
        fetched_at: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO frames (timestamp, filename, file_path, width, height, mean_intensity, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    filename=excluded.filename,
                    file_path=excluded.file_path,
                    width=excluded.width,
                    height=excluded.height,
                    mean_intensity=excluded.mean_intensity,
                    fetched_at=excluded.fetched_at
                """,
                (timestamp, filename, file_path, width, height, mean_intensity, fetched_at),
            )

    def list_frames(self, limit: int = 12) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, filename, file_path, width, height, mean_intensity, fetched_at
                FROM frames
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        # Return chronologically ascending for playback and optical flow
        return [dict(r) for r in reversed(rows)]

    def prune_frames(self, keep: int = 12) -> list[str]:
        """Deletes frames older than the newest `keep` records and returns pruned file paths."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, file_path FROM frames
                ORDER BY timestamp DESC
                LIMIT -1 OFFSET ?
                """,
                (keep,),
            ).fetchall()
            if not rows:
                return []
            timestamps = [r["timestamp"] for r in rows]
            paths = [r["file_path"] for r in rows]
            conn.executemany("DELETE FROM frames WHERE timestamp = ?", [(t,) for t in timestamps])
            return paths

    # ------------------------------------------------------------ Nowcast Runs
    def save_nowcast_run(
        self,
        frame_timestamp: Optional[str],
        backend_used: str,
        inference_ms: float,
        advisory_text: str,
        payload_dict: dict[str, Any],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO nowcast_runs (frame_timestamp, backend_used, inference_ms, advisory_text, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (frame_timestamp, backend_used, inference_ms, advisory_text, json.dumps(payload_dict), now),
            )
            # Keep last 200 runs to stay tiny on SD card
            conn.execute(
                """
                DELETE FROM nowcast_runs
                WHERE id NOT IN (SELECT id FROM nowcast_runs ORDER BY id DESC LIMIT 200)
                """
            )

    def get_latest_nowcast(self) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM nowcast_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return json.loads(row["payload_json"])

    # ----------------------------------------------------------- Alert History
    def last_alert_time(self, entity_key: str) -> Optional[datetime]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT sent_at FROM alert_history
                WHERE entity_key = ?
                ORDER BY sent_at DESC
                LIMIT 1
                """,
                (entity_key,),
            ).fetchone()
        if not row:
            return None
        return datetime.fromisoformat(row["sent_at"])

    def record_alert(
        self,
        entity_key: str,
        entity_type: str,
        level: str,
        webhook_status: str,
        payload: dict[str, Any],
        sent_at: Optional[datetime] = None,
    ) -> None:
        ts = (sent_at or datetime.now(timezone.utc)).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alert_history (entity_key, entity_type, level, sent_at, webhook_status, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (entity_key, entity_type, level, ts, webhook_status, json.dumps(payload)),
            )

    # ---------------------------------------------------------- Gauge Readings
    def upsert_gauge_readings(self, readings: list[dict[str, Any]]) -> None:
        if not readings:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO gauge_readings (
                    station_id, station_name, river_name, basin_id,
                    lat, lon, water_level_ft, alert_level_ft, minor_flood_ft, major_flood_ft,
                    rainfall_24h_mm, reading_timestamp, fetched_at
                )
                VALUES (
                    :station_id, :station_name, :river_name, :basin_id,
                    :lat, :lon, :water_level_ft, :alert_level_ft, :minor_flood_ft, :major_flood_ft,
                    :rainfall_24h_mm, :reading_timestamp, :fetched_at
                )
                ON CONFLICT(station_id, reading_timestamp) DO UPDATE SET
                    water_level_ft=excluded.water_level_ft,
                    alert_level_ft=COALESCE(excluded.alert_level_ft, gauge_readings.alert_level_ft),
                    minor_flood_ft=COALESCE(excluded.minor_flood_ft, gauge_readings.minor_flood_ft),
                    major_flood_ft=COALESCE(excluded.major_flood_ft, gauge_readings.major_flood_ft),
                    rainfall_24h_mm=COALESCE(excluded.rainfall_24h_mm, gauge_readings.rainfall_24h_mm),
                    fetched_at=excluded.fetched_at
                """,
                readings,
            )

    def prune_gauge_readings(self, max_age_hours: int = 72) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM gauge_readings WHERE reading_timestamp < ?", (cutoff,))
            return cur.rowcount

    def get_gauge_history(self, station_id: str, hours: int = 24) -> list[dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT station_id, station_name, river_name, basin_id,
                       lat, lon, water_level_ft, alert_level_ft, minor_flood_ft, major_flood_ft,
                       rainfall_24h_mm, reading_timestamp, fetched_at
                FROM gauge_readings
                WHERE station_id = ? AND reading_timestamp >= ?
                ORDER BY reading_timestamp ASC
                """,
                (station_id, cutoff),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latest_gauges(self, max_age_hours: int = 72) -> list[dict[str, Any]]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT g.*
                FROM gauge_readings g
                INNER JOIN (
                    SELECT station_id, MAX(reading_timestamp) AS max_ts
                    FROM gauge_readings
                    WHERE reading_timestamp >= ?
                    GROUP BY station_id
                ) latest
                ON g.station_id = latest.station_id AND g.reading_timestamp = latest.max_ts
                ORDER BY g.basin_id, g.station_name
                """,
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------ Active Learning Samples
    def has_al_frame_timestamp(self, frame_timestamp: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM active_learning_samples WHERE frame_timestamp = ? LIMIT 1",
                (frame_timestamp,),
            ).fetchone()
        return row is not None

    def insert_al_sample(
        self,
        *,
        sample_id: str,
        frame_timestamp: str,
        captured_at: str,
        trigger_type: str,
        trigger_detail: str,
        status: str,
        model_filename: str,
        min_confidence: Optional[float],
        max_confidence: Optional[float],
        detection_count: int,
        dir_path: str,
        size_bytes: int,
        meta_json: str,
        reviewed_at: Optional[str] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO active_learning_samples (
                    id, frame_timestamp, captured_at, trigger_type, trigger_detail,
                    status, model_filename, min_confidence, max_confidence,
                    detection_count, dir_path, size_bytes, meta_json, reviewed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    frame_timestamp,
                    captured_at,
                    trigger_type,
                    trigger_detail,
                    status,
                    model_filename,
                    min_confidence,
                    max_confidence,
                    detection_count,
                    dir_path,
                    size_bytes,
                    meta_json,
                    reviewed_at,
                ),
            )

    def get_al_sample(self, sample_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM active_learning_samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_al_samples(
        self,
        status: Optional[str] = "pending",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if status and status != "all":
                rows = conn.execute(
                    """
                    SELECT * FROM active_learning_samples
                    WHERE status = ?
                    ORDER BY captured_at DESC
                    LIMIT ?
                    """,
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM active_learning_samples
                    ORDER BY captured_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def list_approved_al_samples(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM active_learning_samples
                WHERE status = 'approved'
                ORDER BY frame_timestamp ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def update_al_decision(
        self,
        *,
        sample_id: str,
        status: str,
        reviewed_at: str,
        detection_count: Optional[int] = None,
        size_bytes: Optional[int] = None,
        meta_json: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM active_learning_samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
            if not existing:
                return None
            new_count = detection_count if detection_count is not None else existing["detection_count"]
            new_size = size_bytes if size_bytes is not None else existing["size_bytes"]
            new_meta = meta_json if meta_json is not None else existing["meta_json"]
            conn.execute(
                """
                UPDATE active_learning_samples
                SET status = ?, reviewed_at = ?, detection_count = ?, size_bytes = ?, meta_json = ?
                WHERE id = ?
                """,
                (status, reviewed_at, new_count, new_size, new_meta, sample_id),
            )
            updated = conn.execute(
                "SELECT * FROM active_learning_samples WHERE id = ?",
                (sample_id,),
            ).fetchone()
        return dict(updated) if updated else None

    def count_al_captures_on_date(self, utc_date_prefix: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM active_learning_samples WHERE captured_at LIKE ?",
                (f"{utc_date_prefix}%",),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_al_counts_and_bytes(self) -> dict[str, int]:
        counts = {"pending": 0, "approved": 0, "rejected": 0, "total_bytes": 0}
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS cnt, COALESCE(SUM(size_bytes), 0) AS bytes_sum
                FROM active_learning_samples
                GROUP BY status
                """
            ).fetchall()
        for r in rows:
            st = str(r["status"])
            if st in counts:
                counts[st] = int(r["cnt"])
            counts["total_bytes"] += int(r["bytes_sum"] or 0)
        return counts

    def list_prunable_al_samples(self) -> list[dict[str, Any]]:
        """
        Returns samples eligible for automatic storage-quota pruning:
        oldest `rejected` first, then oldest `pending`. Never auto-prunes `approved`.
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, status, dir_path, size_bytes, captured_at
                FROM active_learning_samples
                WHERE status IN ('rejected', 'pending')
                ORDER BY
                    CASE status WHEN 'rejected' THEN 0 ELSE 1 END ASC,
                    captured_at ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_al_sample(self, sample_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM active_learning_samples WHERE id = ?", (sample_id,))
            return cur.rowcount > 0
