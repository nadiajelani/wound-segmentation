"""
Unit tests for preprocessing functions.

Tests image loading, preprocessing, data validation, and transformation
in the preprocessing pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import os

from woundseg.pipelines.preprocess import ImagePreprocessor
from woundseg.types import AnalysisOptions
from woundseg.utils.exceptions import ValidationError


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""
    
    def test_init(self):
        """Test ImagePreprocessor initialization."""
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None
        assert preprocessor.target_size == (128, 128)
    
    def test_init_with_custom_size(self):
        """Test ImagePreprocessor initialization with custom size."""
        preprocessor = ImagePreprocessor(target_size=(256, 256))
        assert preprocessor.target_size == (256, 256)
    
    def test_load_image_valid_path(self, sample_image_path: Path, test_utils):
        """Test loading a valid image file."""
        preprocessor = ImagePreprocessor()
        
        # Create a test image if it doesn't exist
        if not sample_image_path.exists():
            test_image = test_utils.create_test_image(256, 256, 3)
            test_utils.save_test_image(test_image, sample_image_path)
        
        image = preprocessor.load_image(sample_image_path)
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (256, 256, 3)
        assert image.dtype == np.uint8
    
    def test_load_image_invalid_path(self):
        """Test loading an invalid image path."""
        preprocessor = ImagePreprocessor()
        
        with pytest.raises(FileNotFoundError):
            preprocessor.load_image(Path("nonexistent_image.jpg"))
    
    def test_load_image_invalid_format(self, temp_dir: Path):
        """Test loading an invalid image format."""
        preprocessor = ImagePreprocessor()
        
        # Create a text file with .jpg extension
        invalid_image = temp_dir / "invalid.jpg"
        invalid_image.write_text("This is not an image")
        
        with pytest.raises(ValidationError):
            preprocessor.load_image(invalid_image)
    
    def test_preprocess_image(self, sample_image: np.ndarray):
        """Test image preprocessing."""
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        
        processed_image = preprocessor.preprocess_image(sample_image)
        
        assert isinstance(processed_image, np.ndarray)
        assert processed_image.shape == (128, 128, 3)
        assert processed_image.dtype == np.float32
        assert np.all(processed_image >= 0.0) and np.all(processed_image <= 1.0)
    
    def test_preprocess_image_with_normalization(self, sample_image: np.ndarray):
        """Test image preprocessing with normalization."""
        preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=True)
        
        processed_image = preprocessor.preprocess_image(sample_image)
        
        assert isinstance(processed_image, np.ndarray)
        assert processed_image.shape == (128, 128, 3)
        assert processed_image.dtype == np.float32
        
        # Check normalization (mean should be close to 0, std close to 1)
        mean = np.mean(processed_image)
        std = np.std(processed_image)
        assert abs(mean) < 0.1  # Should be close to 0
        assert 0.8 < std < 1.2  # Should be close to 1
    
    def test_preprocess_image_different_sizes(self, test_utils):
        """Test preprocessing images of different sizes."""
        preprocessor = ImagePreprocessor(target_size=(64, 64))
        
        # Test with different input sizes
        sizes = [(128, 128), (256, 256), (64, 128), (128, 64)]
        
        for width, height in sizes:
            test_image = test_utils.create_test_image(width, height, 3)
            processed_image = preprocessor.preprocess_image(test_image)
            
            assert processed_image.shape == (64, 64, 3)
    
    def test_validate_image_valid(self, sample_image: np.ndarray):
        """Test image validation with valid image."""
        preprocessor = ImagePreprocessor()
        
        # Should not raise any exception
        preprocessor.validate_image(sample_image)
    
    def test_validate_image_invalid_shape(self):
        """Test image validation with invalid shape."""
        preprocessor = ImagePreprocessor()
        
        # Test with 2D array (missing channel dimension)
        invalid_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        with pytest.raises(ValidationError):
            preprocessor.validate_image(invalid_image)
    
    def test_validate_image_invalid_dtype(self):
        """Test image validation with invalid data type."""
        preprocessor = ImagePreprocessor()
        
        # Test with float64 instead of uint8
        invalid_image = np.random.rand(256, 256, 3)
        
        with pytest.raises(ValidationError):
            preprocessor.validate_image(invalid_image)
    
    def test_validate_image_invalid_values(self):
        """Test image validation with invalid pixel values."""
        preprocessor = ImagePreprocessor()
        
        # Test with values outside 0-255 range
        invalid_image = np.random.randint(-10, 300, (256, 256, 3), dtype=np.uint8)
        
        with pytest.raises(ValidationError):
            preprocessor.validate_image(invalid_image)
    
    def test_preprocess_pipeline(self, sample_image_path: Path, test_utils):
        """Test complete preprocessing pipeline."""
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        
        # Create test image if it doesn't exist
        if not sample_image_path.exists():
            test_image = test_utils.create_test_image(256, 256, 3)
            test_utils.save_test_image(test_image, sample_image_path)
        
        # Run complete pipeline
        processed_image = preprocessor.preprocess_pipeline(sample_image_path)
        
        assert isinstance(processed_image, np.ndarray)
        assert processed_image.shape == (128, 128, 3)
        assert processed_image.dtype == np.float32
    
    def test_preprocess_pipeline_with_options(self, sample_image_path: Path, test_utils, test_analysis_options: AnalysisOptions):
        """Test preprocessing pipeline with analysis options."""
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        
        # Create test image if it doesn't exist
        if not sample_image_path.exists():
            test_image = test_utils.create_test_image(256, 256, 3)
            test_utils.save_test_image(test_image, sample_image_path)
        
        # Run pipeline with options
        processed_image = preprocessor.preprocess_pipeline(sample_image_path, options=test_analysis_options)
        
        assert isinstance(processed_image, np.ndarray)
        assert processed_image.shape == (128, 128, 3)
    
    def test_batch_preprocessing(self, test_utils, temp_dir: Path):
        """Test batch preprocessing of multiple images."""
        preprocessor = ImagePreprocessor(target_size=(64, 64))
        
        # Create multiple test images
        image_paths = []
        for i in range(3):
            test_image = test_utils.create_test_image(128, 128, 3)
            image_path = temp_dir / f"test_image_{i}.jpg"
            test_utils.save_test_image(test_image, image_path)
            image_paths.append(image_path)
        
        # Process batch
        processed_images = preprocessor.preprocess_batch(image_paths)
        
        assert len(processed_images) == 3
        for processed_image in processed_images:
            assert processed_image.shape == (64, 64, 3)
            assert processed_image.dtype == np.float32
    
    def test_preprocessing_consistency(self, sample_image: np.ndarray):
        """Test that preprocessing is consistent (deterministic)."""
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        
        # Process the same image multiple times
        results = []
        for _ in range(3):
            processed = preprocessor.preprocess_image(sample_image)
            results.append(processed)
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    
    def test_memory_efficiency(self, test_utils, temp_dir: Path):
        """Test that preprocessing doesn't use excessive memory."""
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        
        # Create a large test image
        large_image = test_utils.create_test_image(1024, 1024, 3)
        large_image_path = temp_dir / "large_image.jpg"
        test_utils.save_test_image(large_image, large_image_path)
        
        # Process the large image
        processed_image = preprocessor.preprocess_pipeline(large_image_path)
        
        # Result should be the target size, not the original size
        assert processed_image.shape == (128, 128, 3)
        
        # Clean up
        large_image_path.unlink()


class TestPreprocessingUtils:
    """Test cases for preprocessing utility functions."""
    
    def test_resize_image(self, test_utils):
        """Test image resizing utility."""
        from woundseg.pipelines.preprocess import resize_image
        
        original_image = test_utils.create_test_image(256, 256, 3)
        resized_image = resize_image(original_image, (128, 128))
        
        assert resized_image.shape == (128, 128, 3)
        assert resized_image.dtype == original_image.dtype
    
    def test_normalize_image(self, test_utils):
        """Test image normalization utility."""
        from woundseg.pipelines.preprocess import normalize_image
        
        test_image = test_utils.create_test_image(128, 128, 3)
        normalized_image = normalize_image(test_image)
        
        assert normalized_image.dtype == np.float32
        assert np.all(normalized_image >= 0.0) and np.all(normalized_image <= 1.0)
    
    def test_denormalize_image(self, test_utils):
        """Test image denormalization utility."""
        from woundseg.pipelines.preprocess import denormalize_image
        
        # Create normalized image
        normalized_image = np.random.rand(128, 128, 3).astype(np.float32)
        denormalized_image = denormalize_image(normalized_image)
        
        assert denormalized_image.dtype == np.uint8
        assert np.all(denormalized_image >= 0) and np.all(denormalized_image <= 255)
    
    def test_normalize_denormalize_roundtrip(self, test_utils):
        """Test that normalize -> denormalize preserves the original image."""
        from woundseg.pipelines.preprocess import normalize_image, denormalize_image
        
        original_image = test_utils.create_test_image(128, 128, 3)
        normalized_image = normalize_image(original_image)
        denormalized_image = denormalize_image(normalized_image)
        
        # Should be very close to original (within rounding error)
        np.testing.assert_array_almost_equal(original_image, denormalized_image, decimal=0)


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_preprocessing_with_real_image_formats(self, temp_dir: Path, test_utils):
        """Test preprocessing with different image formats."""
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        
        # Test different image formats
        formats = ['JPEG', 'PNG', 'BMP']
        test_image = test_utils.create_test_image(256, 256, 3)
        
        for fmt in formats:
            image_path = temp_dir / f"test_image.{fmt.lower()}"
            Image.fromarray(test_image).save(image_path, fmt)
            
            processed_image = preprocessor.preprocess_pipeline(image_path)
            assert processed_image.shape == (128, 128, 3)
    
    def test_preprocessing_error_handling(self, temp_dir: Path):
        """Test preprocessing error handling."""
        preprocessor = ImagePreprocessor()
        
        # Test with corrupted image file
        corrupted_path = temp_dir / "corrupted.jpg"
        corrupted_path.write_bytes(b"corrupted image data")
        
        with pytest.raises(ValidationError):
            preprocessor.preprocess_pipeline(corrupted_path)
    
    def test_preprocessing_performance(self, sample_image: np.ndarray):
        """Test preprocessing performance."""
        import time
        
        preprocessor = ImagePreprocessor(target_size=(128, 128))
        
        # Time the preprocessing
        start_time = time.time()
        processed_image = preprocessor.preprocess_image(sample_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should be reasonably fast (less than 1 second for a single image)
        assert processing_time < 1.0
        assert processed_image.shape == (128, 128, 3)