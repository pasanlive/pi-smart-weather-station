# Sri Lanka Edge-Native Weather Nowcasting & Flood-Prediction System

An autonomous, 24/7 edge-native satellite nowcasting and hydrological flood-prediction platform for **Sri Lanka**, engineered to run on a **Raspberry Pi 5 (16 GB RAM)** with a **Hailo-8 AI HAT+ (26 TOPS)** while seamlessly falling back to OpenCV CPU IR segmentation in development or zero-model environments.

---

## Architecture Highlights

- **Himawari-9 Band 13 Clean IR Ingestion (`/backend/ingestion/himawari.py`):** Polls NASA GIBS WMS every 10 minutes (with 6-frame startup backfill and automatic 12-frame / 2-hour disk ring buffer pruning), resolving the latest published GIBS slot via `GetCapabilities`.
- **Open-Meteo Surface & Catchment Fusion (`/backend/ingestion/open_meteo.py`):** Polls 7 Sri Lanka surface stations (Colombo, Kandy, Galle, Ratnapura, Trincomalee, Jaffna, Batticaloa) + 4 wet-zone upper-catchment centroids every 30 minutes for precipitation, wind gusts, 3-hour pressure drops, and rolling 24-hour catchment rainfall.
- **Sri Lanka Dept. of Irrigation ArcGIS Telemetry (`/backend/ingestion/irrigation_arcgis.py`):** Polls public ArcGIS Online REST Feature Services (`gauges` + `24hr_rainfall`) every 15 minutes, persists 72-hour history in SQLite to calculate 1–3h **rate of rise (`ft/hr`)**, **trend** (`RISING`/`STEADY`/`FALLING`), and **threshold status** (`NORMAL`/`ALERT`/`MINOR_FLOOD`/`MAJOR_FLOOD`), with explicit `stale`/`unavailable` reporting and optional `waterlevels.lk` fallback.
- **Pluggable Vision AI (`/backend/inference`):**
  - `Hailo8StormDetector`: INT8 PCIe inference via `hailo_platform` (`HEF`, `VDevice`, `InferVStreams`).
  - `CPUIRStormDetector`: Multi-level Band 13 brightness-temperature thresholding + contour polygon extraction matching the exact same schema.
- **Optical Flow & Geospatial ETA (`NowcastEngine`, `/backend/nowcast/nowcast_engine.py`):** Computes averaged dense Farneback optical flow across the last 2–3 frames, rasterizes median per-cell velocity (capped at `120 km/h`, 16-point compass bearing), projects polygons to `+30/+60/+120 min` via `shapely.affinity.translate`, and intersects `unary_union` horizons against all **25 Sri Lanka districts** and **4 wet-zone river basins**.
- **Satellite + Hydrological Flood Fusion (`BasinFloodAnalyzer`, `/backend/nowcast/basin_flood.py`):** Combines satellite storm coverage, 24h rainfall, and live upstream/downstream gauges across **Kelani, Kalu, Gin, and Nilwala** river basins, detecting **Upstream Surges** with drainage-lag crest arrival windows and triggering optional webhook alerts (30-min SQLite cooldown).
- **Dark Mission-Control Dashboard (`/frontend`):** React + TypeScript + Tailwind + Leaflet with animated 2-hour IR playback, toggleable District Threats, dashed Flood Basins with popups, River Gauges with 24-hour SVG sparklines, stepped-opacity `+30/+60/+120m` trajectories, speed-scaled velocity arrows, and a sortable 25-district threat matrix.

---

## Local Development Quick Start

### 1. Backend (FastAPI + Python 3.11+)
```bash
cp .env.example .env
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend (React + Vite + Leaflet)
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173`. Vite proxies `/api` and `/ws` directly to FastAPI on `:8000`.

### 3. Run Backend Unit Tests
```bash
cd backend
.venv/bin/pytest -v
```

---

## Raspberry Pi 5 + Hailo-8 AI HAT+ (26 TOPS) Production Setup

### 1. Enable PCIe Gen 3 & Install HailoRT
On Raspberry Pi OS (64-bit Bookworm or newer):
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y hailo-all python3-venv python3-opencv
sudo reboot
```
Verify the Hailo-8 NPU is detected over PCIe:
```bash
hailortcli fw-control identify
```

### 2. Drop In Your Compiled `.hef` Model
Place your Hailo-8 compiled YOLOv8-seg or segmentation `.hef` model at the configured path:
```bash
mkdir -p /opt/pi-smart-weather-station/models
cp storm_yolov8s_seg.hef /opt/pi-smart-weather-station/models/
```
Update `.env`:
```env
INFERENCE_BACKEND=auto
MODEL_HEF_PATH=/opt/pi-smart-weather-station/models/storm_yolov8s_seg.hef
```
> **Note:** Create the Python venv with `--system-site-packages` on Pi 5 so `hailo_platform` from `hailo-all` is visible inside `.venv`:
> ```bash
> python3 -m venv --system-site-packages /opt/pi-smart-weather-station/backend/.venv
> ```

### 3. Configure Sri Lanka Department of Irrigation ArcGIS Endpoints
Set the FeatureServer `/0/query` URLs in `.env` (no code changes needed if ArcGIS URLs rotate):
```env
IRRIGATION_ARCGIS_GAUGES_URL=https://services3.arcgis.com/<org_id>/arcgis/rest/services/gauges_2_view/FeatureServer/0/query
IRRIGATION_ARCGIS_RAINFALL_URL=https://services3.arcgis.com/<org_id>/arcgis/rest/services/24hr_rainfall/FeatureServer/0/query
IRRIGATION_POLL_INTERVAL_MIN=15
IRRIGATION_FALLBACK_ENABLED=true
```

### 4. Build Frontend & Enable `systemd` Service
FastAPI automatically serves `/frontend/dist` at `/` when built, so a single service runs everything on port `8000`:
```bash
cd /opt/pi-smart-weather-station/frontend
npm ci && npm run build

sudo cp /opt/pi-smart-weather-station/deploy/sl-nowcast.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sl-nowcast.service
sudo journalctl -u sl-nowcast.service -f
```

---

## REST & WebSocket API

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/status` | `GET` | Health, active backend (`hailo-8` vs `cpu-fallback`), inference latency, hydrology status, Pi 5 CPU/RAM/temp |
| `/api/frames` | `GET` | Rolling 12-frame Himawari-9 IR ring buffer metadata and image URLs |
| `/api/frames/{filename}` | `GET` | Serves cached `640×640` Band 13 PNG frame |
| `/api/districts` | `GET` | Lightweight GeoJSON of Sri Lanka's 25 districts |
| `/api/gauges` | `GET` | Latest normalized river gauges, 24h history, and source health (`arcgis_live`/`fallback`/`stale`/`unavailable`) |
| `/api/nowcast/latest` | `GET` | Full nowcast: `storm_cells`, `+30/+60/+120m` projections, `districts`, `river_basins`, `gauges`, `upstream_surge_warning`, and `advisory_text` |
| `/api/nowcast/refresh` | `POST` | Triggers immediate satellite, weather, gauge, inference, and flood-fusion cycle |
| `/ws/nowcast` | `WS` | Real-time push stream of fresh nowcast snapshots |
