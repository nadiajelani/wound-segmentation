"""
Unit tests for segmentation functions.

Tests U-Net model loading, prediction, mask generation, and postprocessing
in the segmentation pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from woundseg.pipelines.segment import SegmentationPipeline
from woundseg.models.provider import ModelProvider
from woundseg.types import AnalysisOptions
from woundseg.utils.exceptions import ModelError, ValidationError


class TestSegmentationPipeline:
    """Test cases for SegmentationPipeline class."""
    
    def test_init(self):
        """Test SegmentationPipeline initialization."""
        pipeline = SegmentationPipeline()
        assert pipeline is not None
        assert pipeline.model_provider is not None
    
    def test_init_with_custom_provider(self):
        """Test SegmentationPipeline initialization with custom provider."""
        mock_provider = Mock()
        pipeline = SegmentationPipeline(model_provider=mock_provider)
        assert pipeline.model_provider == mock_provider
    
    @patch('woundseg.models.provider.ModelProvider')
    def test_load_model_success(self, mock_provider_class):
        """Test successful model loading."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.load_unet_model.return_value = True
        
        pipeline = SegmentationPipeline()
        result = pipeline.load_model()
        
        assert result is True
        mock_provider.load_unet_model.assert_called_once()
    
    @patch('woundseg.models.provider.ModelProvider')
    def test_load_model_failure(self, mock_provider_class):
        """Test model loading failure."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.load_unet_model.return_value = False
        
        pipeline = SegmentationPipeline()
        result = pipeline.load_model()
        
        assert result is False
    
    @patch('woundseg.models.provider.ModelProvider')
    def test_load_model_exception(self, mock_provider_class):
        """Test model loading with exception."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.load_unet_model.side_effect = ModelError("Model not found")
        
        pipeline = SegmentationPipeline()
        
        with pytest.raises(ModelError):
            pipeline.load_model()
    
    def test_predict_mask_valid_input(self, sample_image: np.ndarray):
        """Test mask prediction with valid input."""
        pipeline = SegmentationPipeline()
        
        # Mock the model prediction
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            mock_predict.return_value = np.random.rand(128, 128, 1)
            
            mask = pipeline.predict_mask(sample_image)
            
            assert isinstance(mask, np.ndarray)
            assert mask.shape == (128, 128, 1)
            assert mask.dtype == np.float32
            mock_predict.assert_called_once()
    
    def test_predict_mask_invalid_input(self):
        """Test mask prediction with invalid input."""
        pipeline = SegmentationPipeline()
        
        # Test with invalid image shape
        invalid_image = np.random.rand(128, 128)  # Missing channel dimension
        
        with pytest.raises(ValidationError):
            pipeline.predict_mask(invalid_image)
    
    def test_predict_mask_with_tta(self, sample_image: np.ndarray):
        """Test mask prediction with test-time augmentation."""
        pipeline = SegmentationPipeline()
        
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            mock_predict.return_value = np.random.rand(128, 128, 1)
            
            mask = pipeline.predict_mask(sample_image, use_tta=True)
            
            assert isinstance(mask, np.ndarray)
            # TTA should be called multiple times
            assert mock_predict.call_count > 1
    
    def test_postprocess_mask(self, test_utils):
        """Test mask postprocessing."""
        pipeline = SegmentationPipeline()
        
        # Create a test mask
        raw_mask = np.random.rand(128, 128, 1)
        
        processed_mask = pipeline.postprocess_mask(raw_mask)
        
        assert isinstance(processed_mask, np.ndarray)
        assert processed_mask.shape == (128, 128)
        assert processed_mask.dtype == np.uint8
        assert np.all(np.isin(processed_mask, [0, 255]))
    
    def test_postprocess_mask_with_threshold(self, test_utils):
        """Test mask postprocessing with custom threshold."""
        pipeline = SegmentationPipeline()
        
        # Create a test mask
        raw_mask = np.random.rand(128, 128, 1)
        
        processed_mask = pipeline.postprocess_mask(raw_mask, threshold=0.7)
        
        assert isinstance(processed_mask, np.ndarray)
        assert processed_mask.shape == (128, 128)
        assert processed_mask.dtype == np.uint8
    
    def test_combine_masks(self, test_utils):
        """Test mask combination."""
        pipeline = SegmentationPipeline()
        
        # Create multiple masks
        masks = [
            test_utils.create_test_mask(128, 128),
            test_utils.create_test_mask(128, 128),
            test_utils.create_test_mask(128, 128)
        ]
        
        combined_mask = pipeline.combine_masks(masks)
        
        assert isinstance(combined_mask, np.ndarray)
        assert combined_mask.shape == (128, 128)
        assert combined_mask.dtype == np.uint8
    
    def test_estimate_uncertainty(self, test_utils):
        """Test uncertainty estimation."""
        pipeline = SegmentationPipeline()
        
        # Create multiple predictions
        predictions = [
            np.random.rand(128, 128, 1),
            np.random.rand(128, 128, 1),
            np.random.rand(128, 128, 1)
        ]
        
        uncertainty = pipeline.estimate_uncertainty(predictions)
        
        assert isinstance(uncertainty, np.ndarray)
        assert uncertainty.shape == (128, 128)
        assert uncertainty.dtype == np.float32
        assert np.all(uncertainty >= 0.0) and np.all(uncertainty <= 1.0)
    
    def test_segment_pipeline(self, sample_image: np.ndarray):
        """Test complete segmentation pipeline."""
        pipeline = SegmentationPipeline()
        
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            mock_predict.return_value = np.random.rand(128, 128, 1)
            
            result = pipeline.segment_pipeline(sample_image)
            
            assert 'mask' in result
            assert 'confidence' in result
            assert 'uncertainty' in result
            assert isinstance(result['mask'], np.ndarray)
    
    def test_segment_pipeline_with_options(self, sample_image: np.ndarray, test_analysis_options: AnalysisOptions):
        """Test segmentation pipeline with analysis options."""
        pipeline = SegmentationPipeline()
        
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            mock_predict.return_value = np.random.rand(128, 128, 1)
            
            result = pipeline.segment_pipeline(sample_image, options=test_analysis_options)
            
            assert 'mask' in result
            assert 'confidence' in result
            assert isinstance(result['mask'], np.ndarray)
    
    def test_segmentation_consistency(self, sample_image: np.ndarray):
        """Test that segmentation is consistent (deterministic)."""
        pipeline = SegmentationPipeline()
        
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            # Use fixed random seed for consistent results
            np.random.seed(42)
            mock_predict.return_value = np.random.rand(128, 128, 1)
            
            # Run segmentation multiple times
            results = []
            for _ in range(3):
                result = pipeline.segment_pipeline(sample_image)
                results.append(result['mask'])
            
            # All results should be identical
            for i in range(1, len(results)):
                np.testing.assert_array_equal(results[0], results[i])


class TestSegmentationUtils:
    """Test cases for segmentation utility functions."""
    
    def test_apply_morphological_operations(self, test_utils):
        """Test morphological operations on masks."""
        from woundseg.pipelines.segment import apply_morphological_operations
        
        test_mask = test_utils.create_test_mask(128, 128)
        processed_mask = apply_morphological_operations(test_mask)
        
        assert isinstance(processed_mask, np.ndarray)
        assert processed_mask.shape == test_mask.shape
        assert processed_mask.dtype == np.uint8
    
    def test_remove_small_objects(self, test_utils):
        """Test removal of small objects from mask."""
        from woundseg.pipelines.segment import remove_small_objects
        
        # Create mask with small objects
        test_mask = np.zeros((128, 128), dtype=np.uint8)
        test_mask[10:15, 10:15] = 255  # Small object
        test_mask[50:80, 50:80] = 255  # Large object
        
        processed_mask = remove_small_objects(test_mask, min_size=100)
        
        assert isinstance(processed_mask, np.ndarray)
        assert processed_mask.shape == test_mask.shape
        # Small object should be removed, large object should remain
        assert np.sum(processed_mask[10:15, 10:15]) == 0
        assert np.sum(processed_mask[50:80, 50:80]) > 0
    
    def test_fill_holes(self, test_utils):
        """Test hole filling in masks."""
        from woundseg.pipelines.segment import fill_holes
        
        # Create mask with holes
        test_mask = np.zeros((128, 128), dtype=np.uint8)
        test_mask[50:80, 50:80] = 255
        test_mask[60:70, 60:70] = 0  # Create a hole
        
        filled_mask = fill_holes(test_mask)
        
        assert isinstance(filled_mask, np.ndarray)
        assert filled_mask.shape == test_mask.shape
        # Hole should be filled
        assert np.all(filled_mask[60:70, 60:70] == 255)
    
    def test_smooth_mask_boundaries(self, test_utils):
        """Test mask boundary smoothing."""
        from woundseg.pipelines.segment import smooth_mask_boundaries
        
        test_mask = test_utils.create_test_mask(128, 128)
        smoothed_mask = smooth_mask_boundaries(test_mask)
        
        assert isinstance(smoothed_mask, np.ndarray)
        assert smoothed_mask.shape == test_mask.shape
        assert smoothed_mask.dtype == np.uint8


class TestModelProvider:
    """Test cases for ModelProvider integration."""
    
    @patch('woundseg.models.provider.ModelProvider')
    def test_model_provider_initialization(self, mock_provider_class):
        """Test ModelProvider initialization."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        
        provider = ModelProvider()
        
        assert provider is not None
        mock_provider_class.assert_called_once()
    
    @patch('woundseg.models.provider.ModelProvider')
    def test_load_unet_model(self, mock_provider_class):
        """Test U-Net model loading."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.load_unet_model.return_value = True
        
        provider = ModelProvider()
        result = provider.load_unet_model()
        
        assert result is True
        mock_provider.load_unet_model.assert_called_once()
    
    @patch('woundseg.models.provider.ModelProvider')
    def test_predict_unet(self, mock_provider_class):
        """Test U-Net prediction."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        mock_provider.predict_unet.return_value = np.random.rand(128, 128, 1)
        
        provider = ModelProvider()
        test_image = np.random.rand(128, 128, 3)
        
        result = provider.predict_unet(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128, 128, 1)
        mock_provider.predict_unet.assert_called_once_with(test_image)


@pytest.mark.integration
class TestSegmentationIntegration:
    """Integration tests for segmentation pipeline."""
    
    def test_segmentation_with_real_model(self, sample_image: np.ndarray):
        """Test segmentation with real model (if available)."""
        pipeline = SegmentationPipeline()
        
        # Try to load real model
        if pipeline.load_model():
            result = pipeline.segment_pipeline(sample_image)
            
            assert 'mask' in result
            assert 'confidence' in result
            assert isinstance(result['mask'], np.ndarray)
            assert result['mask'].shape == (128, 128)
        else:
            pytest.skip("Model not available for integration test")
    
    def test_segmentation_performance(self, sample_image: np.ndarray):
        """Test segmentation performance."""
        import time
        
        pipeline = SegmentationPipeline()
        
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            mock_predict.return_value = np.random.rand(128, 128, 1)
            
            start_time = time.time()
            result = pipeline.segment_pipeline(sample_image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should be reasonably fast (less than 5 seconds for a single image)
            assert processing_time < 5.0
            assert 'mask' in result
    
    def test_segmentation_error_handling(self):
        """Test segmentation error handling."""
        pipeline = SegmentationPipeline()
        
        # Test with invalid input
        invalid_image = np.random.rand(128, 128)  # Missing channel dimension
        
        with pytest.raises(ValidationError):
            pipeline.segment_pipeline(invalid_image)
    
    def test_batch_segmentation(self, test_utils, temp_dir: Path):
        """Test batch segmentation of multiple images."""
        pipeline = SegmentationPipeline()
        
        # Create multiple test images
        images = []
        for i in range(3):
            test_image = test_utils.create_test_image(128, 128, 3)
            images.append(test_image)
        
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            mock_predict.return_value = np.random.rand(128, 128, 1)
            
            results = pipeline.segment_batch(images)
            
            assert len(results) == 3
            for result in results:
                assert 'mask' in result
                assert isinstance(result['mask'], np.ndarray)


@pytest.mark.slow
class TestSegmentationPerformance:
    """Performance tests for segmentation (marked as slow)."""
    
    def test_large_image_segmentation(self, test_utils):
        """Test segmentation of large images."""
        pipeline = SegmentationPipeline()
        
        # Create a large test image
        large_image = test_utils.create_test_image(512, 512, 3)
        
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            mock_predict.return_value = np.random.rand(512, 512, 1)
            
            result = pipeline.segment_pipeline(large_image)
            
            assert 'mask' in result
            assert result['mask'].shape == (512, 512)
    
    def test_memory_usage(self, sample_image: np.ndarray):
        """Test memory usage during segmentation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        pipeline = SegmentationPipeline()
        
        with patch.object(pipeline.model_provider, 'predict_unet') as mock_predict:
            mock_predict.return_value = np.random.rand(128, 128, 1)
            
            # Run segmentation multiple times
            for _ in range(10):
                result = pipeline.segment_pipeline(sample_image)
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024