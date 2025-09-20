# Wound Segmentation Setup Guide

This guide helps you set up the wound segmentation environment and prevent common configuration issues.

## 🚀 Quick Setup (Recommended)

### Automated Setup
Run the automated setup script:
```bash
./setup_environment.sh
```

This script will:
- Install system dependencies (Homebrew, Tcl/Tk)
- Set up Python 3.11.8 with Tkinter support
- Install all required Python packages
- Verify the installation

### Manual Setup
If you prefer manual setup, follow the steps below.

## 📋 Prerequisites

### System Requirements
- macOS (tested on macOS 15.6.1)
- At least 8GB RAM (16GB recommended for GPU)
- 10GB free disk space for models and dependencies

### Required Software
1. **Homebrew** (package manager)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **pyenv** (Python version manager)
   ```bash
   brew install pyenv
   ```

## 🐍 Python Environment Setup

### 1. Install System Dependencies
```bash
# Install Tcl/Tk for Tkinter support
brew install tcl-tk

# Install Python Tkinter support
brew install python-tk@3.11
```

### 2. Set Environment Variables
Add to your `~/.zshrc` or `~/.bash_profile`:
```bash
export PATH="/opt/homebrew/opt/tcl-tk/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/tcl-tk/lib"
export CPPFLAGS="-I/opt/homebrew/opt/tcl-tk/include"

# pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload your shell:
```bash
source ~/.zshrc
```

### 3. Install Python 3.11.8
```bash
pyenv install 3.11.8
pyenv global 3.11.8
```

### 4. Verify Tkinter Installation
```bash
python -c "import tkinter; print('Tkinter works!')"
```

If this fails, copy the Tkinter binary:
```bash
cp /opt/homebrew/Cellar/python-tk@3.11/3.11.13/libexec/_tkinter.cpython-311-darwin.so \
   ~/.pyenv/versions/3.11.8/lib/python3.11/lib-dynload/
```

## 📦 Python Dependencies

### Install from Requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Manual Installation (if needed)
```bash
# Core ML packages
pip install numpy==1.26.4 pandas==1.5.3 tensorflow==2.20.0
pip install opencv-python==4.12.0.88 scikit-image==0.25.2
pip install matplotlib==3.10.5 pillow==11.3.0

# Reporting
pip install reportlab==4.4.4

# Optional packages
pip install albumentations==2.0.8 segmentation-models==1.0.1
pip install gTTS==2.5.4 Flask==3.1.2
```

## 🔧 Configuration

### 1. Environment Variables
Copy the example configuration:
```bash
cp env.example .env
```

Edit `.env` with your specific paths and preferences.

### 2. Model Files
Ensure your model files are in the correct locations:
```
models/
├── simclr_unet_patch_wound.keras
├── medsam_model.pth (optional)
└── resnet_classifier.h5 (optional)
```

### 3. Directory Structure
Create required directories:
```bash
mkdir -p outputs reports wound_progress_report logs
```

## ✅ Verification

### Run Health Check
```bash
python health_check.py
```

This will verify:
- Python version compatibility
- All required packages
- Tkinter functionality
- GPU availability (if applicable)
- Model files
- Directory structure
- Basic functionality tests

### Test Your Setup
```bash
python test_wound_progress.py
```

## 🚨 Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named '_tkinter'`
**Solution:**
```bash
# Install Tkinter support
brew install python-tk@3.11

# Copy binary module
cp /opt/homebrew/Cellar/python-tk@3.11/3.11.13/libexec/_tkinter.cpython-311-darwin.so \
   ~/.pyenv/versions/3.11.8/lib/python3.11/lib-dynload/
```

#### 2. `ValueError: numpy.dtype size changed`
**Solution:**
```bash
# Clear cache and reinstall
pip cache purge
pip uninstall numpy pandas -y
pip install --no-cache-dir numpy==1.26.4 pandas==1.5.3
```

#### 3. TensorFlow GPU Issues
**Solution:**
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Force CPU if GPU issues
export CUDA_VISIBLE_DEVICES=""
```

#### 4. Model File Not Found
**Solution:**
- Check if model files exist in `models/` directory
- Verify file permissions
- Update paths in `.env` file

### Getting Help

1. **Run Health Check**: `python health_check.py`
2. **Check Logs**: Look in `logs/` directory
3. **Verify Environment**: `python -c "import sys; print(sys.path)"`
4. **Test Imports**: Try importing each package individually

## 🔄 Maintenance

### Regular Updates
```bash
# Update packages
pip list --outdated
pip install --upgrade package_name

# Update requirements
pip freeze > requirements_current.txt
```

### Environment Reset
If you encounter persistent issues:
```bash
# Remove virtual environment
rm -rf venv

# Reinstall Python
pyenv uninstall 3.11.8
pyenv install 3.11.8

# Reinstall packages
pip install -r requirements.txt
```

## 📝 Best Practices

1. **Always use pinned versions** in requirements.txt
2. **Run health checks** before starting work
3. **Keep models in version control** (use Git LFS for large files)
4. **Document any custom configurations**
5. **Test on clean environment** before deployment
6. **Use virtual environments** for different projects

## 🎯 Next Steps

After successful setup:
1. Run the health check to verify everything works
2. Test with sample images
3. Explore the documentation in `docs/` folder
4. Check out the modularization plan in Phase 1 docs

## 📞 Support

If you encounter issues not covered in this guide:
1. Check the troubleshooting section above
2. Run `python health_check.py` for detailed diagnostics
3. Review error logs in the `logs/` directory
4. Ensure all prerequisites are met

---

**Happy wound segmentation! 🏥✨**