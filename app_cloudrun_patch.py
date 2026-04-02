"""
app_cloudrun_patch.py
=====================
Cloud Run stores files in /tmp (ephemeral, but fine — model is re-downloaded
on each new container instance). The only thing to change in app.py is the
default model path.

FIND this line in app.py:
    model_path = os.getenv("SIMCLR_MODEL_PATH", "/app/models/simclr_unet_patch_wound.keras")

REPLACE WITH:
    model_path = os.getenv("SIMCLR_MODEL_PATH", "/tmp/models/simclr_unet_patch_wound.keras")

That's the only required code change. Everything else is handled by the
Dockerfile and deploy script.

WHY /tmp?
Cloud Run containers have a read-only filesystem except for /tmp.
The model is downloaded there on first request (or startup).
/tmp persists for the lifetime of the container instance — typically
hours to days — so the model only downloads once per instance.

OPTIONAL — make startup faster with a startup probe:
Cloud Run supports a startup probe. Add this to your deploy command:
    --startup-probe-failure-threshold=30
    --startup-probe-initial-delay=10

This tells Cloud Run to wait up to 5 min for your app to be ready
instead of killing it if the model download takes a while.
"""

# ── Patched model path line (copy into app.py) ─────────────────────────────────

PATCHED_LINE = """
    model_path = os.getenv("SIMCLR_MODEL_PATH", "/tmp/models/simclr_unet_patch_wound.keras")
"""

# ── Also ensure the /tmp/models directory exists (add near top of load_model()) ──

MKDIR_LINE = """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
"""
