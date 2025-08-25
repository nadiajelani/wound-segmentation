## Phase 2: Local Mac app integration

### Objectives
- Provide a polished local experience while reusing the `woundseg` core.
- Unify desktop GUI and optional local web UI using the same API.

### Options
- Desktop GUI: PySide6/Qt or Swift (SwiftUI) shell invoking a local HTTP API.
- Local API: FastAPI/Flask app calling `woundseg` (runs as a background process/launch agent).

### Step-by-step
1) Consolidate backend
   - Create `apps/local_api/main.py` (FastAPI) exposing `/healthz`, `/readyz`, `/analyze`, `/report/{id}`.
   - Use `woundseg` providers for one-time model load; warm-up on startup.

2) Desktop GUI
   - Replace Tkinter with PySide6 for better UX or keep Tkinter minimal and call HTTP API.
   - Implement: file picker, options (use MedSAM, generate report), progress, results preview.

3) Local packaging
   - Create `Makefile` targets: `make run-local-api`, `make app`.
   - Bundle models in `~/Library/Application Support/WoundSeg/models` or prompt user to download.

4) macOS integration
   - Create a LaunchAgent plist to start local API on login (optional).
   - Harden notarization/signing if distributing outside dev.

5) Performance/UX
   - Toggle features (SHAP/Grad-CAM) off by default; enable via settings.
   - Cache last N results; open report in default PDF viewer.

6) Testing
   - UI smoke tests; API integration tests against local API.

### Acceptance criteria
- App selects image → calls local API → shows mask/severity → opens PDF.
- Local API shares the same `woundseg` package and produces identical results as CLI.


