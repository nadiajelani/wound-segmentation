#!/usr/bin/env python3
"""
Domain-specific exceptions for the wound segmentation system.

This module defines custom exception classes for different types of errors
that can occur in the wound analysis pipeline.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

class WoundSegmentationError(Exception):
    """Base exception for all wound segmentation errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class ModelLoadingError(WoundSegmentationError):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, model_path: Optional[Path] = None, reason: Optional[str] = None):
        message = f"Failed to load model '{model_name}'"
        if model_path:
            message += f" from {model_path}"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="MODEL_LOADING_ERROR",
            context={
                'model_name': model_name,
                'model_path': str(model_path) if model_path else None,
                'reason': reason
            }
        )

class ImageValidationError(WoundSegmentationError):
    """Raised when image validation fails."""
    
    def __init__(self, image_path: Optional[Path] = None, validation_errors: Optional[List[str]] = None):
        message = "Image validation failed"
        if image_path:
            message += f" for {image_path}"
        if validation_errors:
            message += f": {', '.join(validation_errors)}"
        
        super().__init__(
            message=message,
            error_code="IMAGE_VALIDATION_ERROR",
            context={
                'image_path': str(image_path) if image_path else None,
                'validation_errors': validation_errors or []
            }
        )

class ReportGenerationError(WoundSegmentationError):
    """Raised when report generation fails."""
    
    def __init__(self, report_type: str, reason: Optional[str] = None):
        message = f"Failed to generate {report_type} report"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="REPORT_GENERATION_ERROR",
            context={
                'report_type': report_type,
                'reason': reason
            }
        )

class ConfigurationError(WoundSegmentationError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: Optional[str] = None, reason: Optional[str] = None):
        message = "Configuration error"
        if config_key:
            message += f" for '{config_key}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context={
                'config_key': config_key,
                'reason': reason
            }
        )

class DataProcessingError(WoundSegmentationError):
    """Raised when data processing fails."""
    
    def __init__(self, operation: str, reason: Optional[str] = None, data_info: Optional[Dict[str, Any]] = None):
        message = f"Data processing failed during {operation}"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR",
            context={
                'operation': operation,
                'reason': reason,
                'data_info': data_info or {}
            }
        )

class ValidationError(WoundSegmentationError):
    """Raised when validation fails."""
    
    def __init__(self, validation_type: str, reason: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = f"Validation failed for {validation_type}"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context={
                'validation_type': validation_type,
                'reason': reason,
                'details': details or {}
            }
        )

class VoiceGenerationError(WoundSegmentationError):
    """Raised when voice generation fails."""
    
    def __init__(self, reason: Optional[str] = None, audio_format: Optional[str] = None):
        message = "Voice generation failed"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="VOICE_GENERATION_ERROR",
            context={
                'reason': reason,
                'audio_format': audio_format
            }
        )

class StorageError(WoundSegmentationError):
    """Raised when storage operations fail."""
    
    def __init__(self, operation: str, path: Optional[Path] = None, reason: Optional[str] = None):
        message = f"Storage operation failed: {operation}"
        if path:
            message += f" for {path}"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            context={
                'operation': operation,
                'path': str(path) if path else None,
                'reason': reason
            }
        )

class AnalysisError(WoundSegmentationError):
    """Raised when wound analysis fails."""
    
    def __init__(self, stage: str, reason: Optional[str] = None, analysis_context: Optional[Dict[str, Any]] = None):
        message = f"Analysis failed at {stage} stage"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="ANALYSIS_ERROR",
            context={
                'stage': stage,
                'reason': reason,
                'analysis_context': analysis_context or {}
            }
        )

class TrainingError(WoundSegmentationError):
    """Raised when model training fails."""
    
    def __init__(self, training_stage: str, reason: Optional[str] = None, training_context: Optional[Dict[str, Any]] = None):
        message = f"Training failed at {training_stage} stage"
        if reason:
            message += f": {reason}"
        
        super().__init__(
            message=message,
            error_code="TRAINING_ERROR",
            context={
                'training_stage': training_stage,
                'reason': reason,
                'training_context': training_context or {}
            }
        )

# Convenience functions for common error patterns
def raise_model_not_found(model_name: str, model_path: Path) -> None:
    """Raise ModelLoadingError for missing model files."""
    if not model_path.exists():
        raise ModelLoadingError(
            model_name=model_name,
            model_path=model_path,
            reason="Model file not found"
        )

def raise_invalid_image(image_path: Path, validation_errors: List[str]) -> None:
    """Raise ImageValidationError for invalid images."""
    raise ImageValidationError(
        image_path=image_path,
        validation_errors=validation_errors
    )

def raise_missing_config(config_key: str) -> None:
    """Raise ConfigurationError for missing configuration."""
    raise ConfigurationError(
        config_key=config_key,
        reason="Required configuration not found"
    )

def raise_analysis_failure(stage: str, original_error: Exception) -> None:
    """Raise AnalysisError wrapping an original error."""
    raise AnalysisError(
        stage=stage,
        reason=str(original_error),
        analysis_context={'original_error_type': type(original_error).__name__}
    ) from original_error