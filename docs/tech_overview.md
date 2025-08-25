## Wound Segmentation – Technical Overview

### Purpose
AI-assisted wound assessment platform to:
- Segment wound regions from images
- Classify wound presence
- Estimate severity and healing potential
- Generate patient-/clinician-facing reports
- Provide local web and desktop interfaces for uploads and analysis

### System Architecture
- **Modeling**
  - Segmentation: custom U-Net; optional ensemble with `segmentation_models` ResNet34 U-Net; MedSAM integration
  - Classification: ResNet50 binary classifier (wound vs non-wound)
  - Training utilities: K-fold CV, mixed precision, TTA, adversarial training, output calibration
- **Inference Services**
  - Flask APIs: `app.py` (U-Net focus + voice summary), `analyze_wound.py` (U-Net + MedSAM hybrid, PDF report)
- **User Interfaces**
  - Web: `wound_whisperer.html` landing/UX; `templates/wound-wisperer.html` template referenced by Flask
  - Desktop: `wound_checker.py` Tkinter app for upload/capture and on-device inference
- **Data Tooling**
  - Dataset scanning & quarantine: `clean_dataset.py`
  - Training pipelines: `train_resnet.py` (TL), `train_resnet_optimized.py` (SimCLR pretrain + fine-tune)

### Key Components

#### Segmentation and Clinical Analysis (`wound_medsam.py`)
- Models
  - `build_unet` custom U-Net (128×128 input, sigmoid output)
  - Optional ensemble with `sm.Unet('resnet34')`
  - MedSAM: `load_medsam_model` (with dynamic quantization on CPU, MPS/GPU support), `medsam_segment`
- Training
  - Loss: `FocalTverskyLoss`; metric: IoU
  - K-fold training, early stopping, LR scheduling, visualization callback of predictions
  - Mixed precision; `tf.data` pipelines; TQDM progress
  - Advanced augmentation (Albumentations), IsolationForest cleanup, active learning sampler
  - Adversarial training utility; test-time augmentation; uncertainty estimation
  - Output calibration via `CalibratedClassifierCV`
- Inference/Validation
  - Image quality checks (brightness/contrast), segmentation validation (IoU-based)
  - Hybrid masks: U-Net + MedSAM via union/intersection/average
  - Explainability: Grad-CAM heatmaps and SHAP values
  - Clinical heuristics: severity tiers, healing potential; PDF report generation
  - Batch evaluation, visualization assets, error logging

#### Web APIs
- `analyze_wound.py`
  - Routes:
    - `POST /upload`: save image → U-Net + MedSAM → combine mask → severity/healing/area → PDF + visualization → returns JSON `{ report_url }`
    - `GET /report/<filename>`: serve PDF from `reports/`
    - `GET /`: render `wound-wisperer.html`
  - Notes: strict input resizing/dtype, grayscale handling; absolute model paths; duplicate route/except blocks to tidy up
- `app.py`
  - Routes:
    - `GET /`: render `process_image.html` (missing in repo)
    - `GET /report/<file>`, `GET /voice/<file>`
    - `POST /upload`: U-Net prediction, optional voice summary (gTTS); returns JSON with base64 images, metrics, and `pdf_url`
  - Notes: references `wound_segmentation` (likely `wound_medsam`); MedSAM fallback not implemented here; absolute model paths

#### Desktop GUI (`wound_checker.py`)
- Tkinter app: upload or capture via webcam, classify (ResNet50) then segment (U-Net + placeholder MedSAM), save TXT + mask PNG, display results.

#### Classification Pipelines
- `train_resnet.py`
  - ResNet50 TL (frozen base → GAP → Dense 512 + Dropout → sigmoid)
  - Data augmentation; checkpointing; early stopping; training plots
- `train_resnet_optimized.py`
  - SimCLR-style pretraining with strong augmentation → supervised head training → fine-tuning
  - Class weighting, checkpoints, LR scheduling, resume from checkpoints
  - Mixed precision, GPU checks, logging, metrics (AUC/PR/confusion matrix), training plots

#### Dataset Sanitation (`clean_dataset.py`)
- Verifies images, flags unsupported extensions, broken symlinks, permission issues; moves problematic files to `quarantine/`; logs class distribution.

### End-to-End Flow (API)
1. Client uploads image to `/upload`
2. Server saves to `uploads/`, preprocesses RGB float [0,1], resizes to 128×128
3. U-Net predicts mask; optionally get MedSAM mask and combine
4. Resize mask to original size
5. Compute severity, healing potential, area; generate PDF; (optional) voice summary
6. Return JSON with URLs (PDF, optional voice) and encodings/metrics (in `app.py` variant)

### Feature List
- Segmentation
  - U-Net segmentation with IoU metric
  - Optional ResNet34 U-Net ensemble
  - Hybrid U-Net + MedSAM (union/intersection/average)
  - TTA and uncertainty estimation
  - Visualizations: mask, contours, Grad-CAM, SHAP, edges
- Classification
  - ResNet50 wound vs non-wound classifier
  - SimCLR pretraining pipeline + fine-tune
- Clinical Analytics
  - Severity levels (Mild/Moderate/Severe)
  - Healing potential estimate; area estimation
  - Quality checks; segmentation validation; clinician/patient reports
- Reporting
  - Patient-friendly PDF (image, mask, outline, explanations)
  - Clinician report with recommendations
  - Optional voice summary (gTTS)
- Interfaces
  - Flask APIs for upload/serve
  - Web landing/UX; analyzer trigger
  - Desktop GUI for offline processing
- Operational
  - Device selection (CPU/GPU/MPS); MedSAM dynamic quantization on CPU
  - CORS, logging, CSV decision logs

### API Shapes (current)
- `POST /upload` (`analyze_wound.py`)
  - Request: multipart form with `image`; optional `name`, `age`, `use_medsam`
  - Response: `{ "report_url": "/report/<file>.pdf" }`
- `POST /upload` (`app.py`)
  - Request: multipart form with `image`; flags `use_medsam`, `voice_summary`
  - Response: `{ pdf_url, voice_summary_url?, severity, healing_potential, area_mm2, image_base64, mask_base64, ... }`

### Notable Gaps and Risks
- Hardcoded absolute paths for models/datasets across files
- Template/route mismatches and missing templates (`process_image.html`)
- Duplicate decorators/exception blocks in `analyze_wound.py`
- `app.py` imports `wound_segmentation` (likely `wound_medsam`)
- MedSAM placeholder in `wound_checker.py` not aligned with actual architecture
- No auth/rate limiting; no Docker/CI; heavy model deps (SAM/SD) impact deployability

### Recommendations
- Centralize configuration (env vars + `config.py`) for paths/features
- Consolidate into a single Flask app with a consistent response schema; align UI expectations
- Fix template routing/names; add missing templates
- Replace GUI MedSAM placeholder with proper loader or route GUI through the API
- Add Dockerfile and `requirements.txt`; introduce simple auth and upload limits
- Lazy, singleton model loading; add health/readiness endpoints
- Add tests (unit for utils; golden-image integration tests)

### References (files)
- Inference: `analyze_wound.py`, `app.py`, `wound_medsam.py`
- UI: `wound_whisperer.html`, `templates/wound-wisperer.html`
- Desktop: `wound_checker.py`
- Training: `train_resnet.py`, `train_resnet_optimized.py`
- Data: `clean_dataset.py`


