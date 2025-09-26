# Figure 2: Sample Wound Segmentation Results

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Original      │    │   U-Net Mask    │    │  MedSAM Mask    │
│   Wound Image   │    │   (IoU: 0.782)  │    │  (IoU: 0.756)   │
│                 │    │                 │    │                 │
│  [Wound Image]  │    │  [Binary Mask]  │    │  [Binary Mask]  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Hybrid Mask   │    │  Clinical       │    │  SHAP           │
│   (Union)       │    │  Assessment     │    │  Explanation    │
│   (IoU: 0.821)  │    │  - Area: 7.28mm²│    │                 │
│                 │    │  - Severity:    │    │  [Heatmap]      │
│  [Binary Mask]  │    │    Mild         │    │                 │
│                 │    │  - Healing:     │    │                 │
│                 │    │    80%          │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Caption**: Sample wound segmentation results showing (a) original wound image, (b) U-Net segmentation mask, (c) MedSAM segmentation mask, (d) hybrid union mask achieving superior IoU of 0.821, (e) clinical assessment with area measurement and severity classification, and (f) SHAP explanation highlighting important regions for segmentation decisions.