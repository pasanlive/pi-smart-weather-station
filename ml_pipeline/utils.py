"""Shared configuration, path resolution, logging, and atomic I/O utilities."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "config.yaml"

FRAME_TIMESTAMP_RE = re.compile(r"(\d{8}T\d{4})Z")


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Configure a clean console logger without duplicating handlers."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline YAML config and attach the resolved config directory."""
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg["_config_path"] = str(path)
    cfg["_base_dir"] = str(path.parent)
    return cfg


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a relative path against the config directory (or PIPELINE_ROOT)."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    anchor = Path(base_dir) if base_dir else PIPELINE_ROOT
    return (anchor / candidate).resolve()


def get_pipeline_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Resolve all standard working directories defined under `paths`."""
    base_dir = Path(cfg.get("_base_dir", PIPELINE_ROOT))
    raw_paths = cfg.get("paths", {})
    return {key: resolve_path(val, base_dir) for key, val in raw_paths.items()}


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for deterministic pipeline behavior."""
    random.seed(seed)
    np.random.seed(seed)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write binary data atomically via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text atomically."""
    atomic_write_bytes(path, content.encode(encoding))


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write pretty-printed JSON atomically."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False) + "\n")


def parse_frame_timestamp(filename: str) -> datetime | None:
    """Parse a UTC datetime from filenames like `himawari_b13_20251101T0600Z.png`."""
    match = FRAME_TIMESTAMP_RE.search(filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)


def format_frame_filename(dt: datetime) -> str:
    """Return canonical filename for a UTC Himawari-9 Band 13 frame."""
    dt_utc = dt.astimezone(timezone.utc)
    return f"himawari_b13_{dt_utc.strftime('%Y%m%dT%H%M')}Z.png"
