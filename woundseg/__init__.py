"""
Wound Segmentation Package

A modular AI-assisted wound assessment platform for medical image analysis.
"""

__version__ = "1.0.0"
__author__ = "Wound Segmentation Team"

# Import main components
from .config import Config
from .types import Patient, AnalysisOptions, AnalysisResult

__all__ = [
    "Config",
    "Patient", 
    "AnalysisOptions",
    "AnalysisResult",
]