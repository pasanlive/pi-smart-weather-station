# Compiling `storm_yolov8s_seg.hef` with Hailo AI SW Suite (x86_64 Linux)

The **Hailo Dataflow Compiler (DFC)** (`hailo_sdk_client`) only runs on **x86_64 Linux**. You can harvest, auto-label, prepare the calibration tensor, and train `yolov8s-seg` on macOS (Apple Silicon MPS), Linux GPU, or cloud notebooks, then run **Step 5** inside the official Hailo AI SW Suite Docker container.

## 1. Prerequisites (on your x86_64 Linux host)
1. Download the **Hailo AI Software Suite Docker** archive from the [Hailo Developer Zone](https://hailo.ai/developer-zone/software-downloads/).
2. Load the image into Docker:
   ```bash
   docker load -i hailo_ai_sw_suite_<version>.tar.gz
   docker images | grep hailo
   ```
3. Ensure Steps 1–4 have produced:
   - `ml_pipeline/work/runs/storm_yolov8s_seg/weights/best.pt` (and/or `best.onnx`)
   - `ml_pipeline/work/calibration/calib_set.npy` (128–256 uncropped `640×640×3` RGB `uint8` frames)
   - `ml_pipeline/work/dataset/data.yaml`

## 2. Build the Helper Image (one-time)
Replace `hailo_ai_sw_suite:<version>` with the tag loaded in Step 1:
```bash
cd ml_pipeline
docker build \
  -f docker/Dockerfile.hailo_dfc \
  --build-arg BASE_IMAGE=hailo_ai_sw_suite:<version> \
  -t sl-storm-hailo-dfc:latest \
  .
```

## 3. Compile to INT8 `.hef` for Hailo-8 (26 TOPS)
Run the wrapper script (supports `auto`, `ultralytics`, or `dfc` fallback):
```bash
# Default auto mode (tries Ultralytics native format="hailo", falls back to DFC ClientRunner)
./docker/compile_in_docker.sh

# Or force direct Hailo DFC ClientRunner using calib_set.npy
COMPILE_MODE=dfc ./docker/compile_in_docker.sh
```

### Outputs
- `ml_pipeline/work/compiled/storm_yolov8s_seg.hef`
- `ml_pipeline/work/compiled/compile_metadata.json` (SHA-256 checksum, class mapping, calibration count)
- `models/storm_yolov8s_seg.hef` (auto-copied to match `MODEL_HEF_PATH=./models/storm_yolov8s_seg.hef` in `.env.example`)

## 4. Deploy to Raspberry Pi 5 + AI HAT+ (Hailo-8)
Copy the compiled `.hef` to your Raspberry Pi 5 and verify it with `hailortcli`:
```bash
scp models/storm_yolov8s_seg.hef pi@raspberrypi:/opt/pi-smart-weather-station/models/
ssh pi@raspberrypi "hailortcli parse-hef /opt/pi-smart-weather-station/models/storm_yolov8s_seg.hef"
```
