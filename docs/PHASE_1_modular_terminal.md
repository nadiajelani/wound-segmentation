## Phase 1: Modularization and SE best practices (terminal-first)

### Objectives
- Refactor into a reusable Python package while preserving terminal workflows.
- Centralize configuration; remove absolute paths.
- Establish typed interfaces, logging, tests, and a simple CLI.

### Target structure (new)
```
woundseg/
  __init__.py
  config.py
  types.py
  logging.py
  models/{provider.py, keras_custom.py, unet.py, medsam.py, classifier.py}
  pipelines/{preprocess.py, segment.py, postprocess.py, analyze.py}
  services/{reporting.py, validation.py, storage.py, voice.py, explain.py}
  training/{data.py, unet_train.py, classifier_train.py, synthetic.py}
  utils/{images.py, concurrency.py, exceptions.py}
cli/
  ws_cli.py  # Typer/Click CLI entrypoints
```

### Step-by-step
1) Config and constants
   - Create `woundseg/config.py` with env-driven settings (UNET_WEIGHTS_PATH, MEDSAM_WEIGHTS_PATH, OUTPUT_DIR, DEVICE, FEATURE_FLAGS like ENABLE_MEDSAM, ENABLE_EXPLAIN, ENABLE_SYNTHETIC).
   - Add `.env.example` and document usage; default to `./models` and `./outputs`.
   - Set mixed-precision and TF threading here (single place) based on DEVICE.

2) Types and contracts
   - Add `woundseg/types.py` with dataclasses for `Patient`, `AnalysisOptions`, `AnalysisResult`, `Artifacts`.

3) Model providers
   - Implement `woundseg/models/provider.py` with lazy singletons for U-Net, MedSAM, Classifier, device selection.
   - Migrate build/load from `wound_medsam.py` into `unet.py` and `medsam.py`.
   - Register Keras custom objects in `models/keras_custom.py` (FocalTverskyLoss, IOUScore) only once.

4) Pipelines
   - Extract from `wound_medsam.py`:
     - `advanced_augmentation`, loaders → `pipelines/preprocess.py`
     - segmentation helpers (TTA, combine_masks, uncertainty) → `pipelines/segment.py`
     - resize-back + feature extraction → `pipelines/postprocess.py`
     - orchestration (`analyze_single_image`, batch evaluation loop) → `pipelines/analyze.py`

5) Services
   - `services/reporting.py`: PDF generation (patient + clinician variants).
   - `services/validation.py`: image QA and IoU validation.
   - `services/explain.py`: Grad-CAM-like and SHAP. Use lazy imports for SHAP.
   - `services/storage.py`: local filesystem abstraction (phase 1 uses local disk).
   - `services/voice.py`: optional gTTS wrapper guarded by feature flag.

6) CLI
   - Add `cli/ws_cli.py` with Typer commands:
     - `analyze --image <path> [--use-medsam] [--report] [--name] [--age]` → saves outputs to OUTPUT_DIR.
     - `train-unet` and `train-classifier` wrappers calling training modules.

7) Refactor existing scripts
   - Update `analyze_wound.py`, `app.py`, `wound_checker.py` to use package APIs (no direct model loads). For phase 1, keep terminal-only focus; web/UI unchanged.
   - Move training code into `training/` modules and import in scripts.

8) Logging and errors
   - Add structured logger `woundseg/logging.py`; replace prints with logger.
   - Create `utils/exceptions.py` and replace broad excepts with specific domain errors.

9) Optional extras (guarded by feature flags)
   - `training/synthetic.py` for Stable Diffusion synthetic wounds (ENABLE_SYNTHETIC).
   - Explainability (ENABLE_EXPLAIN) toggles SHAP/Grad-CAM usage.
   - MedSAM usage via ENABLE_MEDSAM + presence of weights.

10) Testing
   - Add `tests/` with unit tests for preprocess, segment, combine, assess, validation, reporting.
   - Include 1–2 golden images to verify deterministic outputs.

11) Packaging
   - Add `pyproject.toml` with project metadata; install locally via `pip install -e .`.
   - Generate `requirements.txt` and `requirements-train.txt` (extras for training/medsam/voice/explain).

### Acceptance criteria
- `ws analyze --image sample.jpg` produces mask and report in OUTPUT_DIR.
- No absolute paths; env-config works on another Mac.
- U-Net weights loaded via provider singletons; MedSAM optional behind flag.
- Mixed precision and device selection applied centrally via config.
- Unit tests pass locally; lint clean.


