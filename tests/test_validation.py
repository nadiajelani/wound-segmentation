"""
Unit tests for validation functions.

Tests image quality assessment, IoU validation metrics, and validation
service functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from woundseg.services.validation import ValidationService
from woundseg.types import AnalysisOptions
from woundseg.utils.exceptions import ValidationError


class TestValidationService:
    """Test cases for ValidationService class."""
    
    def test_init(self):
        """Test ValidationService initialization."""
        service = ValidationService()
        assert service is not None
    
    def test_validate_image_quality_good_image(self, sample_image: np.ndarray):
        """Test image quality validation with good image."""
        service = ValidationService()
        
        result = service.validate_image_quality(sample_image)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'quality_score' in result
        assert 'issues' in result
        assert result['is_valid'] is True
        assert 0.0 <= result['quality_score'] <= 1.0
    
    def test_validate_image_quality_blurry_image(self, test_utils):
        """Test image quality validation with blurry image."""
        service = ValidationService()
        
        # Create a blurry image (low frequency content)
        blurry_image = np.ones((128, 128, 3), dtype=np.uint8) * 128
        
        result = service.validate_image_quality(blurry_image)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'quality_score' in result
        assert result['quality_score'] < 0.5  # Should be low quality
    
    def test_validate_image_quality_dark_image(self, test_utils):
        """Test image quality validation with dark image."""
        service = ValidationService()
        
        # Create a very dark image
        dark_image = np.ones((128, 128, 3), dtype=np.uint8) * 10
        
        result = service.validate_image_quality(dark_image)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'quality_score' in result
        assert 'issues' in result
        assert 'dark' in result['issues']
    
    def test_validate_image_quality_bright_image(self, test_utils):
        """Test image quality validation with overexposed image."""
        service = ValidationService()
        
        # Create an overexposed image
        bright_image = np.ones((128, 128, 3), dtype=np.uint8) * 250
        
        result = service.validate_image_quality(bright_image)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'quality_score' in result
        assert 'issues' in result
        assert 'overexposed' in result['issues']
    
    def test_validate_image_quality_small_image(self, test_utils):
        """Test image quality validation with small image."""
        service = ValidationService()
        
        # Create a very small image
        small_image = test_utils.create_test_image(32, 32, 3)
        
        result = service.validate_image_quality(small_image)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'issues' in result
        assert 'small' in result['issues']
    
    def test_validate_image_quality_invalid_input(self):
        """Test image quality validation with invalid input."""
        service = ValidationService()
        
        # Test with invalid image shape
        invalid_image = np.random.rand(128, 128)  # Missing channel dimension
        
        with pytest.raises(ValidationError):
            service.validate_image_quality(invalid_image)
    
    def test_calculate_iou(self, test_utils):
        """Test IoU calculation between two masks."""
        service = ValidationService()
        
        # Create two overlapping masks
        mask1 = np.zeros((128, 128), dtype=np.uint8)
        mask1[50:100, 50:100] = 255
        
        mask2 = np.zeros((128, 128), dtype=np.uint8)
        mask2[75:125, 75:125] = 255
        
        iou = service.calculate_iou(mask1, mask2)
        
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0
        assert iou > 0.0  # Should have some overlap
    
    def test_calculate_iou_identical_masks(self, test_utils):
        """Test IoU calculation with identical masks."""
        service = ValidationService()
        
        mask = test_utils.create_test_mask(128, 128)
        
        iou = service.calculate_iou(mask, mask)
        
        assert iou == 1.0  # Perfect overlap
    
    def test_calculate_iou_no_overlap(self, test_utils):
        """Test IoU calculation with no overlapping masks."""
        service = ValidationService()
        
        mask1 = np.zeros((128, 128), dtype=np.uint8)
        mask1[0:50, 0:50] = 255
        
        mask2 = np.zeros((128, 128), dtype=np.uint8)
        mask2[75:125, 75:125] = 255
        
        iou = service.calculate_iou(mask1, mask2)
        
        assert iou == 0.0  # No overlap
    
    def test_calculate_iou_different_sizes(self):
        """Test IoU calculation with different sized masks."""
        service = ValidationService()
        
        mask1 = np.zeros((128, 128), dtype=np.uint8)
        mask1[50:100, 50:100] = 255
        
        mask2 = np.zeros((64, 64), dtype=np.uint8)
        mask2[25:50, 25:50] = 255
        
        with pytest.raises(ValidationError):
            service.calculate_iou(mask1, mask2)
    
    def test_validate_segmentation_quality(self, test_utils):
        """Test segmentation quality validation."""
        service = ValidationService()
        
        # Create ground truth and prediction masks
        gt_mask = test_utils.create_test_mask(128, 128)
        pred_mask = test_utils.create_test_mask(128, 128)
        
        result = service.validate_segmentation_quality(gt_mask, pred_mask)
        
        assert isinstance(result, dict)
        assert 'iou' in result
        assert 'dice_score' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1_score' in result
        assert 0.0 <= result['iou'] <= 1.0
        assert 0.0 <= result['dice_score'] <= 1.0
    
    def test_validate_segmentation_quality_perfect_match(self, test_utils):
        """Test segmentation quality validation with perfect match."""
        service = ValidationService()
        
        mask = test_utils.create_test_mask(128, 128)
        
        result = service.validate_segmentation_quality(mask, mask)
        
        assert result['iou'] == 1.0
        assert result['dice_score'] == 1.0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0
    
    def test_validate_segmentation_quality_no_match(self, test_utils):
        """Test segmentation quality validation with no match."""
        service = ValidationService()
        
        mask1 = np.zeros((128, 128), dtype=np.uint8)
        mask1[0:50, 0:50] = 255
        
        mask2 = np.zeros((128, 128), dtype=np.uint8)
        mask2[75:125, 75:125] = 255
        
        result = service.validate_segmentation_quality(mask1, mask2)
        
        assert result['iou'] == 0.0
        assert result['dice_score'] == 0.0
        assert result['precision'] == 0.0
        assert result['recall'] == 0.0
        assert result['f1_score'] == 0.0
    
    def test_validate_input_image(self, sample_image: np.ndarray):
        """Test input image validation."""
        service = ValidationService()
        
        result = service.validate_input_image(sample_image)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'quality_score' in result
        assert 'issues' in result
    
    def test_validate_input_image_invalid_format(self):
        """Test input image validation with invalid format."""
        service = ValidationService()
        
        # Test with invalid image shape
        invalid_image = np.random.rand(128, 128)  # Missing channel dimension
        
        result = service.validate_input_image(invalid_image)
        
        assert result['is_valid'] is False
        assert 'invalid_format' in result['issues']
    
    def test_validate_output_mask(self, test_utils):
        """Test output mask validation."""
        service = ValidationService()
        
        mask = test_utils.create_test_mask(128, 128)
        
        result = service.validate_output_mask(mask)
        
        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'area' in result
        assert 'coverage' in result
    
    def test_validate_output_mask_empty_mask(self):
        """Test output mask validation with empty mask."""
        service = ValidationService()
        
        empty_mask = np.zeros((128, 128), dtype=np.uint8)
        
        result = service.validate_output_mask(empty_mask)
        
        assert result['is_valid'] is False
        assert result['area'] == 0
        assert result['coverage'] == 0.0
    
    def test_validate_output_mask_invalid_values(self):
        """Test output mask validation with invalid values."""
        service = ValidationService()
        
        # Create mask with invalid values (not 0 or 255)
        invalid_mask = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        
        result = service.validate_output_mask(invalid_mask)
        
        assert result['is_valid'] is False
        assert 'invalid_values' in result['issues']


class TestValidationUtils:
    """Test cases for validation utility functions."""
    
    def test_calculate_dice_score(self, test_utils):
        """Test dice score calculation."""
        from woundseg.services.validation import calculate_dice_score
        
        mask1 = test_utils.create_test_mask(128, 128)
        mask2 = test_utils.create_test_mask(128, 128)
        
        dice_score = calculate_dice_score(mask1, mask2)
        
        assert isinstance(dice_score, float)
        assert 0.0 <= dice_score <= 1.0
    
    def test_calculate_dice_score_identical(self, test_utils):
        """Test dice score calculation with identical masks."""
        from woundseg.services.validation import calculate_dice_score
        
        mask = test_utils.create_test_mask(128, 128)
        
        dice_score = calculate_dice_score(mask, mask)
        
        assert dice_score == 1.0
    
    def test_calculate_dice_score_no_overlap(self):
        """Test dice score calculation with no overlap."""
        from woundseg.services.validation import calculate_dice_score
        
        mask1 = np.zeros((128, 128), dtype=np.uint8)
        mask1[0:50, 0:50] = 255
        
        mask2 = np.zeros((128, 128), dtype=np.uint8)
        mask2[75:125, 75:125] = 255
        
        dice_score = calculate_dice_score(mask1, mask2)
        
        assert dice_score == 0.0
    
    def test_calculate_precision_recall(self, test_utils):
        """Test precision and recall calculation."""
        from woundseg.services.validation import calculate_precision_recall
        
        mask1 = test_utils.create_test_mask(128, 128)
        mask2 = test_utils.create_test_mask(128, 128)
        
        precision, recall = calculate_precision_recall(mask1, mask2)
        
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
    
    def test_calculate_precision_recall_identical(self, test_utils):
        """Test precision and recall calculation with identical masks."""
        from woundseg.services.validation import calculate_precision_recall
        
        mask = test_utils.create_test_mask(128, 128)
        
        precision, recall = calculate_precision_recall(mask, mask)
        
        assert precision == 1.0
        assert recall == 1.0
    
    def test_assess_image_quality(self, test_utils):
        """Test image quality assessment."""
        from woundseg.services.validation import assess_image_quality
        
        # Test with good quality image
        good_image = test_utils.create_test_image(128, 128, 3)
        quality_score = assess_image_quality(good_image)
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_assess_image_quality_blurry(self):
        """Test image quality assessment with blurry image."""
        from woundseg.services.validation import assess_image_quality
        
        # Create blurry image (low frequency content)
        blurry_image = np.ones((128, 128, 3), dtype=np.uint8) * 128
        
        quality_score = assess_image_quality(blurry_image)
        
        assert quality_score < 0.5  # Should be low quality
    
    def test_detect_image_issues(self, test_utils):
        """Test image issue detection."""
        from woundseg.services.validation import detect_image_issues
        
        # Test with normal image
        normal_image = test_utils.create_test_image(128, 128, 3)
        issues = detect_image_issues(normal_image)
        
        assert isinstance(issues, list)
    
    def test_detect_image_issues_dark(self):
        """Test image issue detection with dark image."""
        from woundseg.services.validation import detect_image_issues
        
        # Create very dark image
        dark_image = np.ones((128, 128, 3), dtype=np.uint8) * 10
        
        issues = detect_image_issues(dark_image)
        
        assert 'dark' in issues
    
    def test_detect_image_issues_bright(self):
        """Test image issue detection with bright image."""
        from woundseg.services.validation import detect_image_issues
        
        # Create overexposed image
        bright_image = np.ones((128, 128, 3), dtype=np.uint8) * 250
        
        issues = detect_image_issues(bright_image)
        
        assert 'overexposed' in issues
    
    def test_detect_image_issues_small(self, test_utils):
        """Test image issue detection with small image."""
        from woundseg.services.validation import detect_image_issues
        
        # Create small image
        small_image = test_utils.create_test_image(32, 32, 3)
        
        issues = detect_image_issues(small_image)
        
        assert 'small' in issues


@pytest.mark.integration
class TestValidationIntegration:
    """Integration tests for validation pipeline."""
    
    def test_validation_pipeline(self, sample_image: np.ndarray, test_utils):
        """Test complete validation pipeline."""
        service = ValidationService()
        
        # Create a test mask
        mask = test_utils.create_test_mask(128, 128)
        
        # Run validation pipeline
        result = service.validate_pipeline(sample_image, mask)
        
        assert isinstance(result, dict)
        assert 'input_validation' in result
        assert 'output_validation' in result
        assert 'quality_metrics' in result
    
    def test_validation_with_real_data(self, sample_image: np.ndarray, test_utils):
        """Test validation with realistic data."""
        service = ValidationService()
        
        # Create realistic wound mask
        mask = np.zeros((128, 128), dtype=np.uint8)
        # Add irregular wound shape
        y, x = np.ogrid[:128, :128]
        center_x, center_y = 64, 64
        wound_area = (x - center_x)**2 + (y - center_y)**2 < 30**2
        mask[wound_area] = 255
        
        # Validate
        result = service.validate_pipeline(sample_image, mask)
        
        assert result['input_validation']['is_valid'] is True
        assert result['output_validation']['is_valid'] is True
        assert result['quality_metrics']['area'] > 0
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        service = ValidationService()
        
        # Test with invalid inputs
        invalid_image = np.random.rand(128, 128)  # Missing channel dimension
        invalid_mask = np.random.rand(128, 128)  # Wrong dtype
        
        with pytest.raises(ValidationError):
            service.validate_pipeline(invalid_image, invalid_mask)
    
    def test_validation_performance(self, sample_image: np.ndarray, test_utils):
        """Test validation performance."""
        import time
        
        service = ValidationService()
        mask = test_utils.create_test_mask(128, 128)
        
        start_time = time.time()
        result = service.validate_pipeline(sample_image, mask)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should be reasonably fast (less than 1 second)
        assert processing_time < 1.0
        assert isinstance(result, dict)


@pytest.mark.golden
class TestGoldenImageValidation:
    """Golden image tests for validation consistency."""
    
    def test_validation_consistency(self, sample_image: np.ndarray, test_utils):
        """Test that validation results are consistent."""
        service = ValidationService()
        mask = test_utils.create_test_mask(128, 128)
        
        # Run validation multiple times
        results = []
        for _ in range(3):
            result = service.validate_pipeline(sample_image, mask)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert results[0]['quality_metrics']['iou'] == results[i]['quality_metrics']['iou']
            assert results[0]['quality_metrics']['dice_score'] == results[i]['quality_metrics']['dice_score']
    
    def test_validation_deterministic(self, sample_image: np.ndarray, test_utils):
        """Test that validation is deterministic."""
        service = ValidationService()
        mask = test_utils.create_test_mask(128, 128)
        
        # Set random seed
        np.random.seed(42)
        
        # Run validation
        result1 = service.validate_pipeline(sample_image, mask)
        
        # Reset seed and run again
        np.random.seed(42)
        result2 = service.validate_pipeline(sample_image, mask)
        
        # Results should be identical
        assert result1['quality_metrics']['iou'] == result2['quality_metrics']['iou']
        assert result1['quality_metrics']['dice_score'] == result2['quality_metrics']['dice_score']