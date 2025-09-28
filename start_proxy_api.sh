#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /Users/nadiajelani/projects/wound-segmentation/api_env_clean/bin/activate
export MODEL_SERVER_URL="http://127.0.0.1:9100"
exec python -m uvicorn apps.local_api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info