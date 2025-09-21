"""
Validation service for image QA and IoU validation.

This module provides validation functionality for input images
and segmentation results.
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from ..types import ValidationResult
from .storage import get_storage_service

logger = logging.getLogger(__name__)


class ValidationService:
    """
    Validation service for image quality assurance and segmentation validation.
    
    Provides comprehensive validation for input images and segmentation results.
    """
    
    def __init__(self, storage_service=None):
        """
        Initialize the validation service.
        
        Args:
            storage_service: Storage service instance (optional)
        """
        self.storage_service = storage_service or get_storage_service()
        
        # Validation thresholds
        self.min_image_size = (64, 64)  # Minimum image dimensions
        self.max_image_size = (4096, 4096)  # Maximum image dimensions
        self.min_brightness = 10  # Minimum brightness (0-255)
        self.max_brightness = 245  # Maximum brightness (0-255)
        self.min_contrast = 20  # Minimum contrast
        self.min_iou_threshold = 0.3  # Minimum IoU for valid segmentation
        
        logger.info("ValidationService initialized")
    
    def validate_image_quality(self, image: np.ndarray, 
                             image_path: Optional[str] = None) -> ValidationResult:
        """
        Validate image quality for wound analysis.
        
        Args:
            image: Input image as numpy array
            image_path: Optional path to the image file
            
        Returns:
            ValidationResult: Validation result with quality metrics
        """
        try:
            # Basic image properties
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            # Initialize validation result
            result = ValidationResult(
                is_valid=True
            )
            result.image_path = image_path
            result.validation_type = "image_quality"
            result.metrics = {}
            
            # Check image dimensions
            if width < self.min_image_size[0] or height < self.min_image_size[1]:
                result.is_valid = False
                result.issues.append(f"Image too small: {width}x{height} (minimum: {self.min_image_size[0]}x{self.min_image_size[1]})")
            
            if width > self.max_image_size[0] or height > self.max_image_size[1]:
                result.is_valid = False
                result.issues.append(f"Image too large: {width}x{height} (maximum: {self.max_image_size[0]}x{self.max_image_size[1]})")
            
            # Check channels
            if channels != 3:
                result.is_valid = False
                result.issues.append(f"Invalid number of channels: {channels} (expected: 3)")
            
            # Calculate quality metrics
            metrics = self._calculate_image_metrics(image)
            result.metrics.update(metrics)
            
            # Validate brightness
            if metrics['brightness'] < self.min_brightness:
                result.is_valid = False
                result.issues.append(f"Image too dark: brightness {metrics['brightness']:.1f} (minimum: {self.min_brightness})")
            
            if metrics['brightness'] > self.max_brightness:
                result.is_valid = False
                result.issues.append(f"Image too bright: brightness {metrics['brightness']:.1f} (maximum: {self.max_brightness})")
            
            # Validate contrast
            if metrics['contrast'] < self.min_contrast:
                result.is_valid = False
                result.issues.append(f"Low contrast: {metrics['contrast']:.1f} (minimum: {self.min_contrast})")
            
            # Validate sharpness
            if metrics['sharpness'] < 50:  # Arbitrary threshold
                result.recommendations.append(f"Image may be blurry: sharpness {metrics['sharpness']:.1f}")
            
            logger.info(f"Image quality validation: {'PASSED' if result.is_valid else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate image quality: {e}")
            result = ValidationResult(is_valid=False)
            result.image_path = image_path
            result.validation_type = "image_quality"
            result.issues = [f"Validation error: {str(e)}"]
            return result
    
    def validate_segmentation_result(self, predicted_mask: np.ndarray,
                                   ground_truth_mask: Optional[np.ndarray] = None,
                                   image_path: Optional[str] = None) -> ValidationResult:
        """
        Validate segmentation result quality.
        
        Args:
            predicted_mask: Predicted segmentation mask
            ground_truth_mask: Optional ground truth mask for IoU calculation
            image_path: Optional path to the original image
            
        Returns:
            ValidationResult: Validation result with segmentation metrics
        """
        try:
            # Initialize validation result
            result = ValidationResult(is_valid=True)
            result.image_path = image_path
            result.validation_type = "segmentation_quality"
            result.metrics = {}
            
            # Basic mask properties
            height, width = predicted_mask.shape
            mask_area = np.sum(predicted_mask > 0)
            total_pixels = height * width
            coverage = mask_area / total_pixels
            
            # Calculate segmentation metrics
            metrics = {
                'mask_area_pixels': int(mask_area),
                'coverage_percentage': coverage * 100,
                'mask_dimensions': (height, width)
            }
            
            # Validate mask coverage
            if coverage < 0.001:  # Less than 0.1% coverage
                result.is_valid = False
                result.issues.append(f"Mask coverage too low: {coverage*100:.3f}%")
            elif coverage > 0.8:  # More than 80% coverage
                result.recommendations.append(f"High mask coverage: {coverage*100:.1f}% - may indicate over-segmentation")
            
            # Calculate IoU if ground truth is provided
            if ground_truth_mask is not None:
                iou = self._calculate_iou(predicted_mask, ground_truth_mask)
                metrics['iou'] = iou
                
                if iou < self.min_iou_threshold:
                    result.is_valid = False
                    result.issues.append(f"IoU too low: {iou:.3f} (minimum: {self.min_iou_threshold})")
                elif iou < 0.5:
                    result.recommendations.append(f"Low IoU: {iou:.3f} - segmentation may need improvement")
            
            # Check for reasonable mask shape
            if mask_area > 0:
                contours, _ = cv2.findContours(predicted_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 10:  # Too many separate regions
                    result.recommendations.append(f"Multiple disconnected regions detected: {len(contours)}")
                
                # Calculate shape metrics
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    metrics['circularity'] = circularity
                    
                    if circularity < 0.1:  # Very irregular shape
                        result.recommendations.append(f"Irregular wound shape detected: circularity {circularity:.3f}")
            
            result.metrics.update(metrics)
            
            logger.info(f"Segmentation validation: {'PASSED' if result.is_valid else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate segmentation result: {e}")
            result = ValidationResult(is_valid=False)
            result.image_path = image_path
            result.validation_type = "segmentation_quality"
            result.issues = [f"Validation error: {str(e)}"]
            return result
    
    def _calculate_image_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Brightness (mean pixel value)
            brightness = np.mean(gray)
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # Color distribution (if RGB)
            color_metrics = {}
            if len(image.shape) == 3:
                for i, color in enumerate(['red', 'green', 'blue']):
                    color_metrics[f'{color}_mean'] = np.mean(image[:, :, i])
                    color_metrics[f'{color}_std'] = np.std(image[:, :, i])
            
            metrics = {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness
            }
            metrics.update(color_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate image metrics: {e}")
            return {'brightness': 0, 'contrast': 0, 'sharpness': 0}
    
    def _calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two masks."""
        try:
            # Ensure masks are binary
            pred_binary = (pred_mask > 0).astype(np.uint8)
            gt_binary = (gt_mask > 0).astype(np.uint8)
            
            # Calculate intersection and union
            intersection = np.logical_and(pred_binary, gt_binary)
            union = np.logical_or(pred_binary, gt_binary)
            
            intersection_area = np.sum(intersection)
            union_area = np.sum(union)
            
            if union_area == 0:
                return 1.0 if intersection_area == 0 else 0.0
            
            iou = intersection_area / union_area
            return float(iou)
            
        except Exception as e:
            logger.error(f"Failed to calculate IoU: {e}")
            return 0.0
    
    def validate_analysis_result(self, analysis_result: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete analysis result.
        
        Args:
            analysis_result: Complete analysis result dictionary
            
        Returns:
            ValidationResult: Validation result for the analysis
        """
        try:
            result = ValidationResult(is_valid=True)
            result.validation_type = "analysis_result"
            result.metrics = {}
            
            # Check required fields
            required_fields = ['wound_area_mm2', 'segmentation_confidence', 'timestamp']
            for field in required_fields:
                if field not in analysis_result:
                    result.is_valid = False
                    result.issues.append(f"Missing required field: {field}")
            
            # Validate wound area
            if 'wound_area_mm2' in analysis_result:
                area = analysis_result['wound_area_mm2']
                if area < 0:
                    result.is_valid = False
                    result.issues.append(f"Invalid wound area: {area} (must be positive)")
                elif area > 10000:  # 100 cm² seems unreasonably large
                    result.recommendations.append(f"Unusually large wound area: {area} mm²")
            
            # Validate confidence
            if 'segmentation_confidence' in analysis_result:
                confidence = analysis_result['segmentation_confidence']
                if confidence < 0 or confidence > 1:
                    result.is_valid = False
                    result.issues.append(f"Invalid confidence: {confidence} (must be 0-1)")
                elif confidence < 0.5:
                    result.recommendations.append(f"Low segmentation confidence: {confidence:.3f}")
            
            # Validate healing trend
            if 'healing_trend' in analysis_result:
                trend = analysis_result['healing_trend']
                valid_trends = ['positive', 'negative', 'stable', 'unknown']
                if trend not in valid_trends:
                    result.is_valid = False
                    result.issues.append(f"Invalid healing trend: {trend} (must be one of {valid_trends})")
            
            result.metrics = analysis_result.copy()
            
            logger.info(f"Analysis result validation: {'PASSED' if result.is_valid else 'FAILED'}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate analysis result: {e}")
            result = ValidationResult(is_valid=False)
            result.validation_type = "analysis_result"
            result.issues = [f"Validation error: {str(e)}"]
            return result
    
    def get_validation_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Get summary of multiple validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            dict: Summary of validation results
        """
        if not validation_results:
            return {'total_validations': 0, 'passed': 0, 'failed': 0}
        
        total = len(validation_results)
        passed = sum(1 for r in validation_results if r.is_valid)
        failed = total - passed
        
        # Collect all issues and recommendations
        all_issues = []
        all_recommendations = []
        
        for result in validation_results:
            all_issues.extend(result.issues)
            all_recommendations.extend(result.recommendations)
        
        # Count issue types
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        return {
            'total_validations': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'total_issues': len(all_issues),
            'total_recommendations': len(all_recommendations),
            'common_issues': issue_counts,
            'all_issues': all_issues,
            'all_recommendations': all_recommendations
        }


# Global validation service instance
_validation_service: Optional[ValidationService] = None


def get_validation_service() -> ValidationService:
    """
    Get the global validation service instance.
    
    Returns:
        ValidationService: The global validation service instance
    """
    global _validation_service
    if _validation_service is None:
        _validation_service = ValidationService()
    return _validation_service


def reset_validation_service() -> None:
    """Reset the global validation service (useful for testing)."""
    global _validation_service
    _validation_service = None
    logger.info("Validation service reset")