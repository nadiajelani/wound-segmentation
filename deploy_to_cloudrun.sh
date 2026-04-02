#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# deploy_to_cloudrun.sh
# One-shot deploy: wound-segmentation Flask API → Google Cloud Run (free tier)
#
# USAGE:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Run: bash deploy_to_cloudrun.sh
#
# COST: $0 for low traffic (free tier = 2M requests/month + 360k CPU-sec/month)
# ══════════════════════════════════════════════════════════════════════════════
set -e

# ── CONFIG — edit these ────────────────────────────────────────────────────────
PROJECT_ID=""                        # leave blank to auto-create or use existing
SERVICE_NAME="wound-api"
REGION="us-central1"                 # cheapest / most capacity
IMAGE_NAME="wound-segmentation"
MEMORY="4Gi"                         # needs ~1.5GB for TF model; 4Gi = safe
CPU="2"
TIMEOUT="600s"                       # model download can take ~60s on first boot

# ── Secrets (set these before running) ────────────────────────────────────────
GITHUB_TOKEN=""                      # your GitHub token (for model download)
API_KEYS="wai_changeme123"           # comma-separated API keys
ADMIN_KEY="admin_changeme"
SECRET_KEY="wound-secret-$(date +%s)"

# Model location (matches your existing Railway setup)
SIMCLR_MODEL_URL="https://github.com/nadiajelani/wound-segmentation/releases/download/v1.0.0/simclr_unet_patch_wound.keras"
SIMCLR_MODEL_PATH="/tmp/models/simclr_unet_patch_wound.keras"
SIMCLR_MODEL_TAG="v1.0.0"
SIMCLR_MODEL_REPO="nadiajelani/wound-segmentation"
SIMCLR_MODEL_ASSET="simclr_unet_patch_wound.keras"
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   WoundAI → Google Cloud Run  (free tier deployment)    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Check gcloud is installed ─────────────────────────────────────────
if ! command -v gcloud &> /dev/null; then
    echo "❌  gcloud CLI not found. Install it from:"
    echo "    https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "    macOS:   brew install --cask google-cloud-sdk"
    echo "    Ubuntu:  snap install google-cloud-cli --classic"
    exit 1
fi
echo "✅  gcloud CLI found: $(gcloud version --format='value(Google Cloud SDK)')"

# ── Step 2: Auth ───────────────────────────────────────────────────────────────
echo ""
echo "🔐  Checking authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' | grep -q '@'; then
    echo "    Logging in..."
    gcloud auth login
fi
echo "✅  Authenticated as: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"

# ── Step 3: Project setup ──────────────────────────────────────────────────────
echo ""
if [ -z "$PROJECT_ID" ]; then
    # Try to use existing project
    EXISTING=$(gcloud config get-value project 2>/dev/null)
    if [ -n "$EXISTING" ] && [ "$EXISTING" != "(unset)" ]; then
        PROJECT_ID="$EXISTING"
        echo "📁  Using existing project: $PROJECT_ID"
    else
        # Create a new project
        PROJECT_ID="wound-ai-$(date +%s)"
        echo "📁  Creating new project: $PROJECT_ID"
        gcloud projects create "$PROJECT_ID" --name="WoundAI"
        echo "    ⚠️  You may need to link a billing account at:"
        echo "    https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
        echo "    (Cloud Run has a generous free tier — no charges for low traffic)"
        read -p "    Press Enter after linking billing (or Ctrl+C to abort)..."
    fi
fi

gcloud config set project "$PROJECT_ID"
echo "✅  Project: $PROJECT_ID"

# ── Step 4: Enable required APIs ──────────────────────────────────────────────
echo ""
echo "🔧  Enabling Cloud Run & Container Registry APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    --quiet
echo "✅  APIs enabled"

# ── Step 5: Create Artifact Registry repo ─────────────────────────────────────
echo ""
echo "📦  Setting up Artifact Registry..."
REPO_EXISTS=$(gcloud artifacts repositories list \
    --location="$REGION" \
    --filter="name:wound-images" \
    --format='value(name)' 2>/dev/null || echo "")

if [ -z "$REPO_EXISTS" ]; then
    gcloud artifacts repositories create wound-images \
        --repository-format=docker \
        --location="$REGION" \
        --description="WoundAI Docker images"
    echo "✅  Repository created"
else
    echo "✅  Repository already exists"
fi

# Configure Docker auth
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── Step 6: Build & push Docker image ─────────────────────────────────────────
echo ""
echo "🐳  Building Docker image (this takes 5-10 min first time)..."
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/wound-images/${IMAGE_NAME}:latest"

# Build using Cloud Build (no Docker needed locally!)
gcloud builds submit \
    --tag "$IMAGE_URI" \
    --timeout=20m \
    --machine-type=e2-highcpu-8 \
    .

echo "✅  Image built and pushed: $IMAGE_URI"

# ── Step 7: Deploy to Cloud Run ───────────────────────────────────────────────
echo ""
echo "🚀  Deploying to Cloud Run..."

gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_URI" \
    --platform managed \
    --region "$REGION" \
    --memory "$MEMORY" \
    --cpu "$CPU" \
    --timeout "$TIMEOUT" \
    --min-instances 0 \
    --max-instances 3 \
    --allow-unauthenticated \
    --set-env-vars "\
KERAS_BACKEND=tensorflow,\
PYTHONUNBUFFERED=1,\
TF_NUM_INTRAOP_THREADS=2,\
TF_NUM_INTEROP_THREADS=2,\
SIMCLR_MODEL_URL=${SIMCLR_MODEL_URL},\
SIMCLR_MODEL_PATH=${SIMCLR_MODEL_PATH},\
SIMCLR_MODEL_TAG=${SIMCLR_MODEL_TAG},\
SIMCLR_MODEL_REPO=${SIMCLR_MODEL_REPO},\
SIMCLR_MODEL_ASSET=${SIMCLR_MODEL_ASSET},\
GITHUB_TOKEN=${GITHUB_TOKEN},\
API_KEYS=${API_KEYS},\
AUTH_ENABLED=true,\
RATE_LIMIT_ANALYZE=10,\
RATE_LIMIT_DEFAULT=60,\
ADMIN_KEY=${ADMIN_KEY},\
SECRET_KEY=${SECRET_KEY},\
CORS_ORIGINS=*"

# ── Step 8: Get URL and test ───────────────────────────────────────────────────
echo ""
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$REGION" \
    --format='value(status.url)')

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  ✅  DEPLOYED!                           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  URL: $SERVICE_URL"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "⏳  Testing health endpoint (model loads on first request)..."
echo "    This may take 60-120s on first boot while the model downloads..."
echo ""

# Poll health until model is loaded
MAX_TRIES=30
for i in $(seq 1 $MAX_TRIES); do
    HTTP_CODE=$(curl -s -o /tmp/health_resp.json -w "%{http_code}" \
        "${SERVICE_URL}/health" 2>/dev/null || echo "000")
    
    if [ "$HTTP_CODE" = "200" ]; then
        MODEL_LOADED=$(python3 -c \
            "import json; d=json.load(open('/tmp/health_resp.json')); print(d.get('model_loaded','?'))" \
            2>/dev/null || echo "?")
        echo "    Health: ✅ 200 | model_loaded: $MODEL_LOADED"
        if [ "$MODEL_LOADED" = "True" ] || [ "$MODEL_LOADED" = "true" ]; then
            break
        fi
    else
        echo "    Attempt $i/$MAX_TRIES: HTTP $HTTP_CODE — waiting..."
    fi
    sleep 10
done

echo ""
echo "🎯  Next steps:"
echo ""
echo "   1. Update your React frontend:"
echo "      const API_URL = \"${SERVICE_URL}\";"
echo ""
echo "   2. Test the analyze endpoint:"
echo "      curl -X POST ${SERVICE_URL}/analyze \\"
echo "        -H 'X-API-Key: ${API_KEYS}' \\"
echo "        -F 'image=@your_wound.jpg'"
echo ""
echo "   3. View logs:"
echo "      gcloud run services logs read ${SERVICE_NAME} --region ${REGION}"
echo ""
echo "   4. Monitor usage (stay in free tier):"
echo "      https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics"
echo ""
echo "💡  Free tier: 2M requests/month + 360,000 CPU-seconds/month"
echo "    At ~2s/analysis = ~180,000 free analyses/month"
