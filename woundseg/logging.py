#!/usr/bin/env python3
"""
Centralized logging configuration for the woundseg package.

This module provides structured logging with medical-grade audit trails,
configurable output levels, and professional error tracking.
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'logger': record.name
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry, default=str)

class MedicalAuditFormatter(logging.Formatter):
    """Specialized formatter for medical audit trails."""
    
    def format(self, record):
        """Format log record for medical compliance."""
        # Ensure no PII (Personally Identifiable Information) in logs
        message = record.getMessage()
        
        # Remove or mask potential PII
        import re
        message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', message)
        message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', message)
        message = re.sub(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', '[CARD]', message)
        
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': message,
            'logger': record.name,
            'audit_id': getattr(record, 'audit_id', None),
            'user_id': getattr(record, 'user_id', None),
            'session_id': getattr(record, 'session_id', None)
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True,
    structured: bool = True,
    medical_audit: bool = False,
    log_dir: Optional[Path] = None
) -> None:
    """
    Centralized logging configuration for the entire woundseg package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default location)
        console_output: Whether to output to console
        structured: Whether to use structured JSON logging
        medical_audit: Whether to use medical audit formatting
        log_dir: Directory for log files (if None, uses default)
    """
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Determine log file location
    if log_file is None:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"wound_analysis_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create formatter
    if medical_audit:
        formatter = MedicalAuditFormatter()
    elif structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific loggers to reduce noise
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Log the configuration
    root_logger.info("Logging system initialized", extra={
        'level': level,
        'log_file': str(log_file) if log_file else None,
        'console_output': console_output,
        'structured': structured,
        'medical_audit': medical_audit
    })

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the configured settings.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def log_analysis_start(logger: logging.Logger, image_path: str, patient_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Log the start of a wound analysis with audit trail.
    
    Args:
        logger: Logger instance
        image_path: Path to the image being analyzed
        patient_info: Optional patient information
        
    Returns:
        Analysis session ID for tracking
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    
    extra_data = {
        'session_id': session_id,
        'image_path': str(image_path),
        'action': 'analysis_start'
    }
    
    if patient_info:
        extra_data.update({
            'patient_id': patient_info.get('id'),
            'user_id': patient_info.get('user_id')
        })
    
    logger.info("Wound analysis started", extra=extra_data)
    return session_id

def log_analysis_complete(logger: logging.Logger, session_id: str, result: Dict[str, Any]) -> None:
    """
    Log the completion of a wound analysis.
    
    Args:
        logger: Logger instance
        session_id: Analysis session ID
        result: Analysis results
    """
    extra_data = {
        'session_id': session_id,
        'action': 'analysis_complete',
        'severity': result.get('severity'),
        'area_mm2': result.get('area_mm2'),
        'confidence_score': result.get('confidence_score'),
        'processing_time': result.get('processing_time')
    }
    
    logger.info("Wound analysis completed", extra=extra_data)

def log_error(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with full context and stack trace.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    extra_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'action': 'error_occurred'
    }
    
    if context:
        extra_data.update(context)
    
    logger.error(f"Error occurred: {error}", extra=extra_data, exc_info=True)

def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs) -> None:
    """
    Log performance metrics for monitoring.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional performance metrics
    """
    extra_data = {
        'operation': operation,
        'duration_seconds': duration,
        'action': 'performance_metric'
    }
    extra_data.update(kwargs)
    
    logger.info(f"Performance: {operation} took {duration:.3f}s", extra=extra_data)

def log_model_usage(logger: logging.Logger, model_name: str, model_version: str, **kwargs) -> None:
    """
    Log model usage for tracking and compliance.
    
    Args:
        logger: Logger instance
        model_name: Name of the model used
        model_version: Version of the model
        **kwargs: Additional model information
    """
    extra_data = {
        'model_name': model_name,
        'model_version': model_version,
        'action': 'model_usage'
    }
    extra_data.update(kwargs)
    
    logger.info(f"Model used: {model_name} v{model_version}", extra=extra_data)

def log_file_operation(logger: logging.Logger, operation: str, file_path: str, **kwargs) -> None:
    """
    Log file operations for audit trails.
    
    Args:
        logger: Logger instance
        operation: Type of file operation (create, read, update, delete)
        file_path: Path to the file
        **kwargs: Additional file information
    """
    extra_data = {
        'file_operation': operation,
        'file_path': str(file_path),
        'action': 'file_operation'
    }
    extra_data.update(kwargs)
    
    logger.info(f"File {operation}: {file_path}", extra=extra_data)

# Convenience functions for common logging patterns
def setup_development_logging() -> None:
    """Setup logging for development environment."""
    setup_logging(
        level="DEBUG",
        console_output=True,
        structured=True,
        medical_audit=False
    )

def setup_production_logging(log_dir: Optional[Path] = None) -> None:
    """Setup logging for production environment."""
    setup_logging(
        level="INFO",
        console_output=False,
        structured=True,
        medical_audit=True,
        log_dir=log_dir
    )

def setup_testing_logging() -> None:
    """Setup logging for testing environment."""
    setup_logging(
        level="WARNING",
        console_output=True,
        structured=False,
        medical_audit=False
    )