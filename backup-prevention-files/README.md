# 🏥 Wound Segmentation AI Platform

An AI-powered wound assessment platform that uses machine learning to segment wound regions from medical images, classify wound presence, estimate severity and healing potential, and generate clinical reports.

## 🚀 Quick Start

### **One-Command Setup (Recommended)**
```bash
# Clone the repository
git clone <repository-url>
cd wound-segmentation

# Complete environment setup
make setup

# Verify everything works
make check

# Test the application
make test
```

### **Alternative Setup**
```bash
# Run automated setup script
./setup_environment.sh

# Run health check
python health_check.py

# Test your script
python test_wound_progress.py
```

## 📋 Prerequisites

- **macOS** (tested on macOS 15.6.1)
- **8GB RAM** (16GB recommended for GPU)
- **10GB free disk space** for models and dependencies
- **Homebrew** package manager

## 🛠️ Available Commands

```bash
# Setup & Installation
make setup          # Run automated environment setup
make install-deps   # Install Python dependencies
make install-models # Download required model files

# Verification & Testing
make check          # Run health check
make test           # Run test script
make test-gui       # Test GUI functionality

# Running Applications
make run-gui        # Run GUI application
make run-api        # Run web API
make run-cli        # Run CLI with sample image

# Maintenance
make clean          # Clean temporary files
make update-deps    # Update dependencies
make freeze-deps    # Generate requirements from current environment

# Help
make help           # Show all available commands
```

## 🏗️ System Architecture

### **Core Models**
- **Segmentation**: Custom U-Net (128×128 input) with optional MedSAM integration
- **Classification**: ResNet50 binary classifier (wound vs non-wound)
- **Hybrid Approach**: Combines U-Net and MedSAM masks via union/intersection/average

### **Key Features**
- K-fold cross-validation training
- Test-time augmentation (TTA)
- Adversarial training
- Output calibration
- Explainability (Grad-CAM, SHAP)
- Clinical heuristics for severity assessment
- PDF report generation
- Optional voice summaries (gTTS)

### **User Interfaces**
- **Web API**: Flask-based REST API
- **Desktop GUI**: Tkinter application
- **CLI**: Command-line interface

## 📁 Project Structure

```
wound-segmentation/
├── models/                     # Model files
│   └── simclr_unet_patch_wound.keras
├── outputs/                    # Generated outputs
├── reports/                    # PDF reports
├── wound_progress_report/      # Progress tracking
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── setup_environment.sh        # Automated setup
├── health_check.py            # System validation
├── Makefile                   # Convenient commands
├── SETUP_GUIDE.md            # Detailed setup guide
└── CONFIGURATION_PREVENTION.md # Prevention strategy
```

## 🔧 Configuration

### **Environment Variables**
Copy the example configuration:
```bash
cp env.example .env
```

Key settings in `.env`:
```bash
# Model Paths
UNET_WEIGHTS_PATH=models/simclr_unet_patch_wound.keras
MEDSAM_WEIGHTS_PATH=models/medsam_model.pth

# Device Configuration
DEVICE=auto  # Options: auto, cpu, gpu, mps

# Feature Flags
ENABLE_MEDSAM=true
ENABLE_EXPLAINABILITY=true
ENABLE_VOICE_SUMMARY=false
```

## 🧪 Testing & Validation

### **Health Check**
```bash
python health_check.py
```

Validates:
- ✅ Python version compatibility
- ✅ All required packages
- ✅ Tkinter functionality
- ✅ GPU availability
- ✅ Model files
- ✅ Directory structure
- ✅ Basic functionality tests

### **Test Your Setup**
```bash
# Test GUI functionality
make test-gui

# Test complete pipeline
make test

# Test with your own image
python test_wound_progress.py
```

## 🚨 Troubleshooting

### **Common Issues & Solutions**

#### 1. `ModuleNotFoundError: No module named '_tkinter'`
```bash
make setup  # Automated fix
```

#### 2. `ValueError: numpy.dtype size changed`
```bash
make setup  # Automated fix
```

#### 3. Model file not found
```bash
# Check if model files exist
ls -la models/

# Download models if needed
make install-models
```

#### 4. GPU not detected
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU if needed
export CUDA_VISIBLE_DEVICES=""
```

### **Getting Help**
1. **Run health check**: `make check`
2. **Check logs**: Look in `logs/` directory
3. **Read documentation**: `SETUP_GUIDE.md`
4. **Review troubleshooting**: `CONFIGURATION_PREVENTION.md`

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions
- **[CONFIGURATION_PREVENTION.md](CONFIGURATION_PREVENTION.md)** - Prevention strategy
- **[docs/](docs/)** - Technical documentation
  - `PHASE_1_modular_terminal.md` - Modularization plan
  - `tech_overview.md` - Technical architecture
  - `modular-enhancement.md` - Enhancement roadmap

## 🔄 Development Workflow

### **Daily Workflow**
```bash
# Start work
make check

# Work on your code
# ... your development ...

# End work (no cleanup needed)
```

### **Adding Dependencies**
1. Add to `requirements.txt` with pinned version
2. Run `make check` to verify
3. Test functionality

### **Weekly Maintenance**
```bash
make check          # Verify environment
make update-deps    # Check for updates
```

## 🎯 Model Sources

- **MedSAM Model**: [Google Drive](https://drive.google.com/drive/search?q=medsam_vit_b.pth)
- **Training Images**: [Kaggle Dataset](https://www.kaggle.com/datasets/leoscode/wound-segmentation-images)

## 🏆 Features

### **Segmentation**
- U-Net segmentation with IoU metric
- Optional ResNet34 U-Net ensemble
- Hybrid U-Net + MedSAM (union/intersection/average)
- TTA and uncertainty estimation
- Visualizations: mask, contours, Grad-CAM, SHAP, edges

### **Classification**
- ResNet50 wound vs non-wound classifier
- SimCLR pretraining pipeline + fine-tune

### **Clinical Analytics**
- Severity levels (Mild/Moderate/Severe)
- Healing potential estimate
- Area estimation
- Quality checks and segmentation validation
- Clinician/patient reports

### **Reporting**
- Patient-friendly PDF (image, mask, outline, explanations)
- Clinician report with recommendations
- Optional voice summary (gTTS)

### **Interfaces**
- Flask APIs for upload/serve
- Web landing/UX analyzer
- Desktop GUI for offline processing

## 🔮 Roadmap

### **Phase 1: Modularization** (Current)
- Refactor into reusable Python package
- Centralize configuration
- Establish typed interfaces and logging
- Create simple CLI

### **Phase 2: Local Mac App**
- Polish local desktop experience
- Unify GUI and web UI
- macOS integration

### **Phase 3: AWS Hosted**
- Containerize with Docker
- Deploy on ECS
- Maintain same UX as local version

### **Phase 4: Enterprise Architecture**
- Scale with managed AWS services
- Advanced security and compliance
- Multi-region deployment

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes**
4. **Run tests**: `make check && make test`
5. **Commit changes**: `git commit -m "Add feature"`
6. **Push to branch**: `git push origin feature-name`
7. **Submit a pull request**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MedSAM**: Medical SAM for medical image segmentation
- **U-Net**: Convolutional networks for biomedical image segmentation
- **TensorFlow**: Machine learning platform
- **OpenCV**: Computer vision library

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `make check` for diagnostics
3. Review the documentation in `docs/`
4. Ensure all prerequisites are met

---

**Happy wound segmentation! 🏥✨**

*For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)*
*For prevention strategies, see [CONFIGURATION_PREVENTION.md](CONFIGURATION_PREVENTION.md)*