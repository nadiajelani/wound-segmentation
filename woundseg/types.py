"""
Type definitions and data models for wound segmentation package.

Provides typed contracts for all data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import numpy as np

@dataclass
class Patient:
    """Patient information for wound analysis."""
    name: str = "Unknown"
    age: Optional[int] = None
    diabetes: Optional[bool] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate patient data."""
        if self.age is not None and (self.age < 0 or self.age > 150):
            raise ValueError(f"Invalid age: {self.age}")
        
        if not self.name or self.name.strip() == "":
            self.name = "Unknown"

@dataclass
class AnalysisOptions:
    """Options for wound analysis."""
    use_medsam: bool = True
    generate_report: bool = True
    generate_voice_summary: bool = False
    generate_heatmap: bool = True
    confidence_threshold: float = 0.5
    patient: Optional[Patient] = None
    output_dir: Optional[Union[str, Path]] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate analysis options."""
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {self.confidence_threshold}")
        
        if self.patient is None:
            self.patient = Patient()

@dataclass
class Artifacts:
    """Generated artifacts from analysis."""
    mask_path: Optional[Union[str, Path]] = None
    overlay_path: Optional[Union[str, Path]] = None
    heatmap_path: Optional[Union[str, Path]] = None
    report_path: Optional[Union[str, Path]] = None
    voice_summary_path: Optional[Union[str, Path]] = None
    additional_files: Dict[str, Union[str, Path]] = field(default_factory=dict)
    
    def get_all_paths(self) -> List[Union[str, Path]]:
        """Get all artifact paths as a list."""
        paths = []
        for attr_name in ['mask_path', 'overlay_path', 'heatmap_path', 'report_path', 'voice_summary_path']:
            path = getattr(self, attr_name)
            if path is not None:
                paths.append(path)
        
        paths.extend(self.additional_files.values())
        return [p for p in paths if p is not None]

@dataclass
class AnalysisResult:
    """Complete result from wound analysis."""
    # Core analysis results
    severity: str  # "Mild", "Moderate", "Severe"
    healing_potential: str  # "Good", "Fair", "Poor"
    area_mm2: float
    confidence_score: float
    
    # Masks and images
    original_image: np.ndarray
    mask: np.ndarray
    overlay: Optional[np.ndarray] = None
    
    # Patient and options
    patient: Optional[Patient] = None
    options: Optional[AnalysisOptions] = None
    
    # Generated artifacts
    artifacts: Optional[Artifacts] = None
    
    # Additional metadata
    processing_time: Optional[float] = None
    model_versions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate analysis result."""
        if self.severity not in ["Mild", "Moderate", "Severe"]:
            raise ValueError(f"Invalid severity: {self.severity}")
        
        if self.healing_potential not in ["Good", "Fair", "Poor"]:
            raise ValueError(f"Invalid healing potential: {self.healing_potential}")
        
        if self.area_mm2 < 0:
            raise ValueError(f"Area cannot be negative: {self.area_mm2}")
        
        if self.confidence_score < 0.0 or self.confidence_score > 1.0:
            raise ValueError(f"Confidence score must be between 0 and 1, got {self.confidence_score}")

@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    path: Union[str, Path]
    version: Optional[str] = None
    device: Optional[str] = None
    is_loaded: bool = False
    load_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PreprocessingResult:
    """Result from image preprocessing."""
    processed_image: np.ndarray
    original_shape: tuple
    processed_shape: tuple
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SegmentationResult:
    """Result from segmentation."""
    mask: np.ndarray
    confidence: float
    method: str  # "unet", "medsam", "hybrid"
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result from validation."""
    is_valid: bool
    iou_score: Optional[float] = None
    quality_score: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

# Type aliases for common use cases
ImageArray = np.ndarray
MaskArray = np.ndarray
PathType = Union[str, Path]

# Validation functions
def validate_image(image: np.ndarray) -> bool:
    """Validate that an image array is properly formatted."""
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False
    
    return True

def validate_mask(mask: np.ndarray) -> bool:
    """Validate that a mask array is properly formatted."""
    if not isinstance(mask, np.ndarray):
        return False
    
    if len(mask.shape) not in [2, 3]:
        return False
    
    # Check if mask contains only 0s and 1s (or values between 0 and 1)
    if mask.min() < 0 or mask.max() > 1:
        return False
    
    return True