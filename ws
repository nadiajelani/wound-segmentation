#!/usr/bin/env python3
"""
Wound Segmentation CLI Wrapper.

This script provides easy access to the wound segmentation CLI.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the CLI
from cli.ws_cli import app

if __name__ == "__main__":
    app()