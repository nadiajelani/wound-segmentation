#!/bin/bash

# Wound Segmentation Environment Setup Script
# This script sets up a proper Python environment with all required dependencies

set -e  # Exit on any error

echo "🏥 Setting up Wound Segmentation Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS. Please adapt for your OS."
    exit 1
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    print_error "Homebrew is not installed. Please install it first:"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

print_status "Installing system dependencies..."

# Install Tcl/Tk for Tkinter support
if ! brew list tcl-tk &> /dev/null; then
    print_status "Installing tcl-tk..."
    brew install tcl-tk
else
    print_success "tcl-tk already installed"
fi

# Install Python Tkinter support
if ! brew list python-tk@3.11 &> /dev/null; then
    print_status "Installing python-tk@3.11..."
    brew install python-tk@3.11
else
    print_success "python-tk@3.11 already installed"
fi

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    print_warning "pyenv is not installed. Installing..."
    brew install pyenv
    
    # Add pyenv to shell profile
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    
    # Reload shell profile
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    print_success "pyenv installed and configured"
else
    print_success "pyenv already installed"
fi

# Set up environment variables for Python compilation with Tkinter
print_status "Setting up Python compilation environment..."
export PATH="/opt/homebrew/opt/tcl-tk/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/tcl-tk/lib"
export CPPFLAGS="-I/opt/homebrew/opt/tcl-tk/include"

# Install Python 3.11.8 if not already installed
if ! pyenv versions | grep -q "3.11.8"; then
    print_status "Installing Python 3.11.8 with Tkinter support..."
    pyenv install 3.11.8
else
    print_success "Python 3.11.8 already installed"
fi

# Set Python 3.11.8 as global version
pyenv global 3.11.8
print_success "Python 3.11.8 set as global version"

# Verify Tkinter is available
print_status "Verifying Tkinter installation..."
if python -c "import tkinter" 2>/dev/null; then
    print_success "Tkinter is working correctly"
else
    print_warning "Tkinter not found, copying binary module..."
    
    # Copy Tkinter binary module to pyenv Python
    TKINTER_PATH="/opt/homebrew/Cellar/python-tk@3.11/3.11.13/libexec/_tkinter.cpython-311-darwin.so"
    PYENV_PATH="$HOME/.pyenv/versions/3.11.8/lib/python3.11/lib-dynload/"
    
    if [ -f "$TKINTER_PATH" ]; then
        cp "$TKINTER_PATH" "$PYENV_PATH"
        print_success "Tkinter binary module copied"
    else
        print_error "Could not find Tkinter binary module at $TKINTER_PATH"
        exit 1
    fi
    
    # Verify again
    if python -c "import tkinter" 2>/dev/null; then
        print_success "Tkinter is now working correctly"
    else
        print_error "Tkinter still not working after copying binary module"
        exit 1
    fi
fi

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies installed from requirements.txt"
else
    print_warning "requirements.txt not found, installing essential packages..."
    pip install numpy==1.26.4 pandas==1.5.3 tensorflow==2.20.0 opencv-python==4.12.0.88 reportlab==4.4.4 matplotlib==3.10.5 pillow==11.3.0
fi

# Verify critical imports
print_status "Verifying critical package imports..."
python -c "
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import tkinter
import reportlab
print('✅ All critical packages imported successfully!')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'ReportLab: {reportlab.__version__}')
"

print_success "Environment setup completed successfully!"
print_status "You can now run your wound segmentation scripts."