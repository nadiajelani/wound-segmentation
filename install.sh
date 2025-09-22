#!/bin/bash

# WoundSeg Installation Script
# This script installs the woundseg package and its dependencies

set -e  # Exit on any error

echo "🏥 WoundSeg Installation Script"
echo "================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION is not supported. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip3 detected"

# Upgrade pip
echo "📦 Upgrading pip..."
pip3 install --upgrade pip

# Install the package in editable mode
echo "🔧 Installing woundseg package..."
pip3 install -e .

# Test installation
echo "🧪 Testing installation..."
if python3 -c "import woundseg; print('✅ woundseg package imported successfully')" 2>/dev/null; then
    echo "✅ Package installation successful!"
else
    echo "❌ Package installation failed. Please check the error messages above."
    exit 1
fi

# Test CLI
echo "🔧 Testing CLI..."
if ws --help &>/dev/null; then
    echo "✅ CLI command 'ws' is working!"
else
    echo "⚠️  CLI command 'ws' is not working. You can still use the package via Python imports."
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "Usage examples:"
echo "  ws --help                    # Show CLI help"
echo "  ws analyze --image path.jpg  # Analyze a wound image"
echo "  python3 -c 'import woundseg' # Import the package"
echo ""
echo "For more information, see the README.md file."