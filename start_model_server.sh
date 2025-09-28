#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /Users/nadiajelani/projects/wound-segmentation/api_env_clean/bin/activate
export UNET_WEIGHTS_PATH="/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
exec uvicorn apps.model_server.server:create_app --factory --host 127.0.0.1 --port 9100 --workers 1 --log-level info