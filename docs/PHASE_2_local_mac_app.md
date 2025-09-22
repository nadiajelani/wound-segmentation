## Phase 2: Local web UI + optional Mac shell

### Objectives
- Provide a polished local browser experience while reusing the `woundseg` core.
- Make a small SPA the primary UI that talks to a local API with the same contract as cloud.
- Keep an optional desktop shell (PySide6/Tkinter) as a convenience wrapper around the same API.

### UI and API
- Web UI (primary): React + Vite + TypeScript SPA
  - Uses `VITE_API_BASE_URL` to point at the local API during dev.
  - OpenAPI schema from the backend generates a typed API client.
  - CORS enabled on local API for `http://localhost:*`.
- Local API: FastAPI app calling `woundseg` providers (runs as a background process or via Makefile target).
- Optional Desktop: PySide6/Tkinter shell that simply embeds a webview or calls the same local HTTP endpoints.

### Step-by-step
1) Local API
   - Create `apps/local_api/main.py` (FastAPI) exposing `/healthz`, `/readyz`, `/analyze`, `/report/{id}` and OpenAPI.
   - Use `woundseg` providers for one-time model load; warm-up on startup.
   - Enable CORS for `http://localhost:5173` (Vite) and `http://localhost:3000` (common alt).
   - Done when:
     - `GET /healthz` and `GET /readyz` return 200.
     - OpenAPI available at `/docs` and `/openapi.json`.
     - First-start warm-up log confirms models loaded once.
     - CORS allows SPA origins to call `/analyze` without errors.

2) Web UI (SPA)
   - Create `apps/web/` with React + Vite + TypeScript.
   - Configure `VITE_API_BASE_URL` for local dev and build-time env switching.
   - Generate a typed API client from FastAPI OpenAPI schema (e.g., openapi-typescript, openapi-generator).
   - Implement: image picker, options (use MedSAM, generate report), upload/progress, results preview, open report.
   - Done when:
     - `npm run dev` serves the SPA and connects to local API (no CORS issues).
     - Typed client compiles and is used in API calls.
     - User can upload an image, set options, see mask/severity, and open PDF.

3) Optional Desktop Shell
   - Provide a minimal PySide6/Tkinter wrapper that either embeds the SPA in a webview or calls the same HTTP API.
   - Done when:
     - App launches, loads embedded SPA or invokes API endpoints successfully.
     - End-to-end flow mirrors the SPA behavior.

4) Local packaging
   - Makefile targets: `make api`, `make web`, `make dev` (run both), `make app` (optional desktop bundle).
   - Store/download models under `~/Library/Application Support/WoundSeg/models` or project-local `models/`.
   - Done when:
     - `make dev` runs API and SPA together; `make api` and `make web` work independently.
     - Models directory exists and app reads weights from configured path.

5) macOS integration (optional)
   - LaunchAgent plist to start local API on login.
   - Notarization/signing if distributing beyond development.
   - Done when:
     - LaunchAgent loads at login and API is reachable without manual start.
     - Codesigning/notarization steps documented and validated on a test machine.

6) Performance/UX
   - Feature flags (SHAP/Grad-CAM) off by default; toggle in settings.
   - Cache last N results; open PDF in default viewer.
   - Done when:
     - Flags default off; toggling reflects in API requests/behavior.
     - Recent results list works; PDFs open via system viewer.

7) Testing
   - UI smoke tests; API integration tests against local API; contract tests from OpenAPI.
   - Done when:
     - Smoke tests pass in CI; contract tests confirm schema compatibility.
     - Basic E2E test covers upload → analyze → report.

### Progress checklist
- [ ] Local API
  - [ ] `GET /healthz` and `GET /readyz` return 200
  - [ ] OpenAPI available at `/docs` and `/openapi.json`
  - [ ] First-start warm-up log confirms models loaded once
  - [ ] CORS allows SPA origins to call `/analyze`
- [ ] Web UI (SPA)
  - [ ] `npm run dev` connects to local API without CORS issues
  - [ ] Typed client compiles and is used for API calls
  - [ ] Upload image, set options, see mask/severity, open PDF
- [ ] Optional Desktop Shell
  - [ ] App launches and embeds SPA or calls API endpoints successfully
  - [ ] End-to-end flow mirrors SPA behavior
- [ ] Local packaging
  - [ ] `make dev` runs API and SPA; `make api` and `make web` work independently
  - [ ] Models directory present; app reads weights from configured path
- [ ] macOS integration (optional)
  - [ ] LaunchAgent starts API on login and API is reachable
  - [ ] Codesigning/notarization documented and validated on a test machine
- [ ] Performance/UX
  - [ ] Flags default off; toggling changes requests/behavior
  - [ ] Recent results list works; PDFs open via system viewer
- [ ] Testing
  - [ ] Smoke tests pass in CI; contract tests confirm schema compatibility
  - [ ] E2E test covers upload → analyze → report

### Acceptance criteria
- SPA runs locally, selects image → calls local API → shows mask/severity → opens PDF.
- Local API shares the same `woundseg` package and produces identical results as CLI.
- Switching `VITE_API_BASE_URL` points SPA to another base URL without code changes.


