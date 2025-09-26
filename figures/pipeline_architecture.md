# Figure 1: Hybrid Wound Segmentation Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Image   │    │   Input Image   │    │   Input Image   │
│   128×128×3     │    │   1024×1024×3   │    │   Original      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   U-Net Model   │    │  MedSAM Model   │    │  Clinical       │
│   Encoder-      │    │   ViT-B +       │    │  Feature        │
│   Decoder       │    │   Mask Decoder  │    │  Extraction     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   U-Net Mask    │    │  MedSAM Mask    │    │  Geometric      │
│   128×128×1     │    │   128×128×1     │    │  Features       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────┬───────────┘                      │
                     │                                  │
                     ▼                                  │
          ┌─────────────────┐                          │
          │  Mask Fusion    │                          │
          │  (Union/Inter/  │                          │
          │   Weighted)     │                          │
          └─────────┬───────┘                          │
                    │                                  │
                    ▼                                  ▼
          ┌─────────────────┐                ┌─────────────────┐
          │  Hybrid Mask    │                │  SHAP Analysis  │
          │   128×128×1     │                │  & Uncertainty  │
          └─────────┬───────┘                └─────────┬───────┘
                    │                                  │
                    ▼                                  ▼
          ┌─────────────────┐                ┌─────────────────┐
          │  Clinical       │◄──────────────►│  Explainable    │
          │  Assessment     │                │  AI Module      │
          │  - Severity     │                └─────────────────┘
          │  - Healing      │
          │  - Area/Shape   │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Automated      │
          │  PDF Report     │
          │  Generation     │
          └─────────────────┘
```

**Caption**: The hybrid wound segmentation pipeline combines U-Net and MedSAM architectures for robust segmentation, integrated with clinical feature extraction and explainable AI components to generate comprehensive wound assessment reports.