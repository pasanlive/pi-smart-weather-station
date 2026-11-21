#!/usr/bin/env bash
# =============================================================================
# Run Step 5 (Hailo-8 INT8 .hef Compilation) inside the Hailo AI SW Suite Docker
# Usage:
#   ./docker/compile_in_docker.sh [extra args passed to 5_compile_hailo8.py]
# Environment variables:
#   HAILO_DOCKER_IMAGE  Image name (default: sl-storm-hailo-dfc:latest)
#   COMPILE_MODE        auto | ultralytics | dfc (default: auto)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PIPELINE_DIR}/.." && pwd)"

HAILO_DOCKER_IMAGE="${HAILO_DOCKER_IMAGE:-sl-storm-hailo-dfc:latest}"
COMPILE_MODE="${COMPILE_MODE:-auto}"

mkdir -p "${REPO_ROOT}/models"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker CLI not found on PATH." >&2
  exit 1
fi

if [[ "$(uname -m)" != "x86_64" && "$(uname -m)" != "amd64" ]]; then
  echo "WARNING: Host architecture is $(uname -m). Hailo DFC requires linux/amd64 (x86_64)." >&2
  echo "         Running under emulation may be very slow or fail; prefer an x86_64 Linux host." >&2
fi

echo "Launching Hailo-8 compilation inside ${HAILO_DOCKER_IMAGE} (mode=${COMPILE_MODE})..."
exec docker run --rm --platform linux/amd64 \
  -v "${PIPELINE_DIR}:/workspace/ml_pipeline" \
  -v "${REPO_ROOT}/models:/workspace/models" \
  -w /workspace/ml_pipeline \
  "${HAILO_DOCKER_IMAGE}" \
  --config /workspace/ml_pipeline/config.yaml \
  --mode "${COMPILE_MODE}" \
  "$@"
