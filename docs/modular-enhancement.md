Title: Modularization plan for robust Python core and clean Web integration

Goals
- Improve maintainability, testability, and clarity of the Python codebase.
- Decouple core ML logic from I/O and frameworks (Flask/Tkinter/CLI) to enable reuse.
- Provide a stable, typed interface for Web/App integrations.
- Standardize configuration, error handling, and logging.
- Enable background processing and scalable deployment.

Current pain points (observed)
- Hardcoded absolute paths for models and datasets.
- Core logic entangled with Flask/Tkinter endpoints and file I/O.
- Two Flask apps with different response shapes; template mismatches.
- Model loading occurs in multiple places; risk of repeated heavy loads.
- MedSAM usage and U-Net logic distributed across files; mixed dependencies.
- Inconsistent error handling and duplicate route blocks.

Proposed package structure (Python package: woundseg/)
```
woundseg/
  __init__.py
  config.py                 # Centralized settings (env + defaults)
  types.py                  # Dataclasses / Pydantic models for IO contracts
  logging.py                # Structured logging setup

  models/
    __init__.py
    provider.py             # Lazy, thread-safe model loader singletons
    unet.py                 # Build/load U-Net
    medsam.py               # Load and run MedSAM
    classifier.py           # ResNet50 classifier (optional)

  pipelines/
    __init__.py
    preprocess.py           # Color conversion, resizing, QA checks
    segment.py              # U-Net segmentation, MedSAM segmentation, hybrid combine
    postprocess.py          # Resize to original size, contouring, area calc
    analyze.py              # Orchestrates end-to-end analysis for one image

  services/
    __init__.py
    reporting.py            # PDF generation (patient/clinician)
    voice.py                # gTTS wrapper (optional)
    storage.py              # File storage abstraction (local/S3)
    validation.py           # IoU validation, quality scoring heuristics

  training/
    __init__.py
    data.py                 # Loaders + augmentation
    unet_train.py           # U-Net training utilities
    classifier_train.py     # ResNet training utilities

  utils/
    __init__.py
    images.py               # Image IO helpers, base64 encode/decode
    concurrency.py          # Background jobs, thread/process pools
    exceptions.py           # Domain exceptions (ConfigurationError, ModelLoadError, etc.)
```

Core abstractions and interfaces
- Config (woundseg.config)
  - Use environment variables (dotenv optional) with sane defaults.
  - Example:
    - UNET_WEIGHTS_PATH, MEDSAM_WEIGHTS_PATH, DEVICE=cpu|mps|cuda, OUTPUT_DIR, ENABLE_VOICE, ENABLE_MEDSAM

- Typed contracts (woundseg.types)
  - Prefer Pydantic models (for Web) or dataclasses (for core) with conversion helpers.
  - Examples:
    - AnalysisOptions: { use_medsam: bool, generate_report: bool, patient: { name, age, diabetes? } }
    - AnalysisResult: { severity, healing_potential, area_mm2, mask: np.ndarray | bytes, report_path?, artifacts: { vis_path?, heatmap_path? } }

- ModelProvider (woundseg.models.provider)
  - Lazy singletons with thread-safe initialization and reference counting.
  - Responsible for device placement (CPU/GPU/MPS) and quantization decisions.
  - Methods: get_unet(), get_medsam(), get_classifier()

- Pipeline functions (pure or side-effect-free where possible)
  - preprocess_image(image: np.ndarray, options) -> PreprocessedImage
  - segment_unet(image) -> np.ndarray
  - segment_medsam(image) -> np.ndarray | None
  - combine_masks(mask1, mask2, method='union') -> np.ndarray
  - postprocess(mask, original_shape) -> np.ndarray
  - assess(mask, image, patient) -> { severity, healing_potential, area_mm2 }
  - analyze_image(image: np.ndarray, options: AnalysisOptions) -> AnalysisResult

- Services
  - ReportingService: generate_patient_report(), generate_clinician_report()
  - StorageService: save_temp(), save_report(), url_for(path)
  - ValidationService: check_quality(), validate_segmentation()
  - VoiceService: synthesize_summary(text) -> path | None

Unified Web integration (Flask or FastAPI)
- Keep web layer very thin; it should:
  - Accept request, parse into AnalysisOptions, load bytes → np.ndarray
  - Call woundseg.pipelines.analyze.analyze_image()
  - Map AnalysisResult to response schema (JSON), generate URLs via StorageService
  - No model loading or heavy logic in routes

- Prefer FastAPI for:
  - Pydantic validation, OpenAPI docs (/docs), async file IO, background tasks
  - Still compatible with the same core package

Example core API (framework-agnostic)
```
from woundseg.types import AnalysisOptions, AnalysisResult
from woundseg.pipelines.analyze import analyze_image

def analyze_from_bytes(image_bytes: bytes, options: AnalysisOptions) -> AnalysisResult:
    # decode → np.ndarray (RGB)
    # call analyze_image → return AnalysisResult
    ...
```

Web route (FastAPI example; Flask similar)
```
@app.post('/analyze', response_model=ApiAnalysisResponse)
def analyze(image: UploadFile, use_medsam: bool=False, generate_report: bool=False, name: str='Unknown', age: int=0):
    img_bytes = image.file.read()
    opts = AnalysisOptions(use_medsam=use_medsam, generate_report=generate_report, patient={'name': name, 'age': age})
    result = analyze_from_bytes(img_bytes, opts)
    return map_result_to_api(result)
```

Configuration standardization
- Replace absolute paths with environment-based config and sensible defaults.
- Provide .env example and fallback to local `models/` and `outputs/` within project.
- One place to choose device (cpu/mps/cuda) and enable MedSAM.

Thread-safety and performance
- Singletons: ensure one-time model load per process; reuse across requests.
- Warm-up on startup with small synthetic image to reduce first-request latency.
- Optionally offload heavy tasks (PDF, SHAP) to background jobs (RQ/Celery) and return 202 + polling.

Error handling and logging
- Use domain exceptions and map them to HTTP errors with clear messages.
- Structured logs (JSON optional) with request IDs; include timings for each pipeline stage.

Testing
- Unit: pipelines (preprocess, segment, combine, assess), services (reporting/validation), ModelProvider mocks.
- Integration: golden-image tests comparing expected masks/metrics.
- Contract: API schemas validated via Pydantic + OpenAPI.

Packaging and deployment
- Add `pyproject.toml` (or setup.cfg) to package `woundseg` for reuse in CLI and Web.
- Produce `requirements.txt` with pinned versions and extras: `[medsam]`, `[voice]`, `[train]`.
- Provide Dockerfile (multi-stage) and gunicorn/uvicorn entrypoints.

Migration plan (incremental)
1) Create package `woundseg/` with `config.py`, `types.py`, `models/provider.py`.
2) Lift U-Net build/load and MedSAM load into `models/` (from `wound_medsam.py`).
3) Move pure functions from `wound_medsam.py` into `pipelines/` and `services/` (reporting/validation).
4) Implement `analyze_image()` orchestration in `pipelines/analyze.py`.
5) Refactor `analyze_wound.py` and `app.py` to call the package API; unify response schemas.
6) Remove absolute paths; switch to env-config; add `.env.example`.
7) Consolidate HTML templates and static paths; fix route/template mismatches.
8) Add tests and a simple CLI (Typer) that calls the same `analyze_image()`.

Unifying the API contract
- Request (multipart/form-data): `image`, `use_medsam` (bool), `generate_report` (bool), `name`, `age`.
- Response (JSON):
  - severity, healing_potential, area_mm2
  - mask_base64 (optional), image_base64 (optional)
  - report_url (if generated), artifacts: { vis_url?, heatmap_url? }

Integration checklist for the Web App
- Use a single backend app (Flask/FastAPI) with consistent routes and templates.
- Serve static and reports from a well-defined `storage` path; generate stable URLs.
- Enforce upload size limits and validate image formats early.
- Add `/healthz` and `/readyz` endpoints; pre-load models on startup.
- Add CORS, rate limiting, and timeouts appropriate to deployment environment.

Optional enhancements
- Switch to ONNX Runtime/TensorRT for inference speed where applicable.
- Add feature flags for SHAP/Grad-CAM to reduce latency in web mode.
- Provide a lightweight mode for mobile/web (U-Net only) and a full mode (U-Net + MedSAM + reports).

Concrete mappings from current files → modules
- `wound_medsam.py` → models.unet, models.medsam, pipelines.segment, pipelines.postprocess, services.reporting, services.validation, pipelines.analyze
- `analyze_wound.py` → web route layer that calls pipelines.analyze (remove direct model usage)
- `app.py` → same as above (or merge into one app with Blueprints/Routers)
- `wound_checker.py` → thin GUI client that calls the same package or the HTTP API
- `train_resnet*.py` → training/*.py (refactor as importable functions)
- `clean_dataset.py` → training.data and/or a CLI command in the package


