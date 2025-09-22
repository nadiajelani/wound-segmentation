"""
Unit tests for model functionality.

Tests model loading, prediction, device selection, and model provider
functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from woundseg.models.provider import ModelProvider
from woundseg.models.unet import UNetProvider
from woundseg.models.device import DeviceManager
from woundseg.models.keras_custom import register_custom_objects
from woundseg.utils.exceptions import ModelError, DeviceError


class TestModelProvider:
    """Test cases for ModelProvider class."""
    
    def test_init(self):
        """Test ModelProvider initialization."""
        provider = ModelProvider()
        assert provider is not None
        assert provider.device_manager is not None
    
    def test_init_with_custom_device_manager(self):
        """Test ModelProvider initialization with custom device manager."""
        mock_device_manager = Mock()
        provider = ModelProvider(device_manager=mock_device_manager)
        assert provider.device_manager == mock_device_manager
    
    @patch('woundseg.models.provider.UNetProvider')
    def test_load_unet_model_success(self, mock_unet_provider_class):
        """Test successful U-Net model loading."""
        mock_unet_provider = Mock()
        mock_unet_provider_class.return_value = mock_unet_provider
        mock_unet_provider.load_model.return_value = True
        
        provider = ModelProvider()
        result = provider.load_unet_model()
        
        assert result is True
        mock_unet_provider.load_model.assert_called_once()
    
    @patch('woundseg.models.provider.UNetProvider')
    def test_load_unet_model_failure(self, mock_unet_provider_class):
        """Test U-Net model loading failure."""
        mock_unet_provider = Mock()
        mock_unet_provider_class.return_value = mock_unet_provider
        mock_unet_provider.load_model.return_value = False
        
        provider = ModelProvider()
        result = provider.load_unet_model()
        
        assert result is False
    
    @patch('woundseg.models.provider.UNetProvider')
    def test_load_unet_model_exception(self, mock_unet_provider_class):
        """Test U-Net model loading with exception."""
        mock_unet_provider = Mock()
        mock_unet_provider_class.return_value = mock_unet_provider
        mock_unet_provider.load_model.side_effect = ModelError("Model not found")
        
        provider = ModelProvider()
        
        with pytest.raises(ModelError):
            provider.load_unet_model()
    
    @patch('woundseg.models.provider.UNetProvider')
    def test_predict_unet(self, mock_unet_provider_class):
        """Test U-Net prediction."""
        mock_unet_provider = Mock()
        mock_unet_provider_class.return_value = mock_unet_provider
        mock_unet_provider.predict.return_value = np.random.rand(128, 128, 1)
        
        provider = ModelProvider()
        test_image = np.random.rand(128, 128, 3)
        
        result = provider.predict_unet(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128, 128, 1)
        mock_unet_provider.predict.assert_called_once_with(test_image)
    
    @patch('woundseg.models.provider.UNetProvider')
    def test_predict_unet_with_tta(self, mock_unet_provider_class):
        """Test U-Net prediction with test-time augmentation."""
        mock_unet_provider = Mock()
        mock_unet_provider_class.return_value = mock_unet_provider
        mock_unet_provider.predict.return_value = np.random.rand(128, 128, 1)
        
        provider = ModelProvider()
        test_image = np.random.rand(128, 128, 3)
        
        result = provider.predict_unet(test_image, use_tta=True)
        
        assert isinstance(result, np.ndarray)
        # TTA should be called multiple times
        assert mock_unet_provider.predict.call_count > 1
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        provider = ModelProvider()
        
        info = provider.get_model_info()
        
        assert isinstance(info, dict)
        assert 'unet_loaded' in info
        assert 'device' in info
        assert 'model_paths' in info
    
    def test_is_model_loaded(self):
        """Test model loading status check."""
        provider = ModelProvider()
        
        # Initially no model should be loaded
        assert provider.is_model_loaded('unet') is False
        
        # After loading (mocked), should return True
        with patch.object(provider, 'unet_provider') as mock_unet:
            mock_unet.is_loaded.return_value = True
            assert provider.is_model_loaded('unet') is True


class TestUNetProvider:
    """Test cases for UNetProvider class."""
    
    def test_init(self):
        """Test UNetProvider initialization."""
        provider = UNetProvider()
        assert provider is not None
        assert provider.model is None
        assert provider.is_loaded is False
    
    def test_init_with_model_path(self, mock_model_path: Path):
        """Test UNetProvider initialization with model path."""
        provider = UNetProvider(model_path=mock_model_path)
        assert provider.model_path == mock_model_path
    
    @patch('woundseg.models.unet.tf.keras.models.load_model')
    def test_load_model_success(self, mock_load_model, mock_model_path: Path):
        """Test successful model loading."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        provider = UNetProvider(model_path=mock_model_path)
        result = provider.load_model()
        
        assert result is True
        assert provider.model is not None
        assert provider.is_loaded is True
        mock_load_model.assert_called_once_with(mock_model_path)
    
    @patch('woundseg.models.unet.tf.keras.models.load_model')
    def test_load_model_failure(self, mock_load_model, mock_model_path: Path):
        """Test model loading failure."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        provider = UNetProvider(model_path=mock_model_path)
        result = provider.load_model()
        
        assert result is False
        assert provider.model is None
        assert provider.is_loaded is False
    
    def test_load_model_file_not_found(self, temp_dir: Path):
        """Test model loading with non-existent file."""
        non_existent_path = temp_dir / "non_existent_model.keras"
        
        provider = UNetProvider(model_path=non_existent_path)
        result = provider.load_model()
        
        assert result is False
        assert provider.model is None
        assert provider.is_loaded is False
    
    @patch('woundseg.models.unet.tf.keras.models.load_model')
    def test_predict_success(self, mock_load_model, mock_model_path: Path):
        """Test successful prediction."""
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(1, 128, 128, 1)
        mock_load_model.return_value = mock_model
        
        provider = UNetProvider(model_path=mock_model_path)
        provider.load_model()
        
        test_image = np.random.rand(128, 128, 3)
        result = provider.predict(test_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (128, 128, 1)
        mock_model.predict.assert_called_once()
    
    def test_predict_model_not_loaded(self):
        """Test prediction with model not loaded."""
        provider = UNetProvider()
        
        test_image = np.random.rand(128, 128, 3)
        
        with pytest.raises(ModelError):
            provider.predict(test_image)
    
    @patch('woundseg.models.unet.tf.keras.models.load_model')
    def test_predict_with_tta(self, mock_load_model, mock_model_path: Path):
        """Test prediction with test-time augmentation."""
        mock_model = Mock()
        mock_model.predict.return_value = np.random.rand(1, 128, 128, 1)
        mock_load_model.return_value = mock_model
        
        provider = UNetProvider(model_path=mock_model_path)
        provider.load_model()
        
        test_image = np.random.rand(128, 128, 3)
        result = provider.predict(test_image, use_tta=True)
        
        assert isinstance(result, np.ndarray)
        # TTA should call predict multiple times
        assert mock_model.predict.call_count > 1
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        provider = UNetProvider()
        
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        processed_image = provider.preprocess_image(test_image)
        
        assert isinstance(processed_image, np.ndarray)
        assert processed_image.shape == (128, 128, 3)  # Should be resized
        assert processed_image.dtype == np.float32
        assert np.all(processed_image >= 0.0) and np.all(processed_image <= 1.0)
    
    def test_postprocess_prediction(self):
        """Test prediction postprocessing."""
        provider = UNetProvider()
        
        raw_prediction = np.random.rand(128, 128, 1)
        processed_prediction = provider.postprocess_prediction(raw_prediction)
        
        assert isinstance(processed_prediction, np.ndarray)
        assert processed_prediction.shape == (128, 128, 1)
        assert processed_prediction.dtype == np.float32
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        provider = UNetProvider()
        
        info = provider.get_model_info()
        
        assert isinstance(info, dict)
        assert 'is_loaded' in info
        assert 'model_path' in info
        assert 'input_shape' in info
        assert 'output_shape' in info


class TestDeviceManager:
    """Test cases for DeviceManager class."""
    
    def test_init(self):
        """Test DeviceManager initialization."""
        manager = DeviceManager()
        assert manager is not None
    
    def test_init_with_custom_device(self):
        """Test DeviceManager initialization with custom device."""
        manager = DeviceManager(device="cpu")
        assert manager.device == "cpu"
    
    def test_detect_available_devices(self):
        """Test device detection."""
        manager = DeviceManager()
        
        devices = manager.detect_available_devices()
        
        assert isinstance(devices, list)
        assert "cpu" in devices  # CPU should always be available
    
    @patch('woundseg.models.device.torch.cuda.is_available')
    def test_detect_available_devices_with_cuda(self, mock_cuda_available):
        """Test device detection with CUDA available."""
        mock_cuda_available.return_value = True
        
        manager = DeviceManager()
        devices = manager.detect_available_devices()
        
        assert "cuda" in devices
    
    @patch('woundseg.models.device.torch.backends.mps.is_available')
    def test_detect_available_devices_with_mps(self, mock_mps_available):
        """Test device detection with MPS available."""
        mock_mps_available.return_value = True
        
        manager = DeviceManager()
        devices = manager.detect_available_devices()
        
        assert "mps" in devices
    
    def test_select_best_device(self):
        """Test best device selection."""
        manager = DeviceManager()
        
        best_device = manager.select_best_device()
        
        assert isinstance(best_device, str)
        assert best_device in ["cpu", "cuda", "mps"]
    
    def test_configure_tensorflow(self):
        """Test TensorFlow configuration."""
        manager = DeviceManager()
        
        # Should not raise any exceptions
        manager.configure_tensorflow()
    
    def test_configure_tensorflow_with_cuda(self):
        """Test TensorFlow configuration with CUDA."""
        manager = DeviceManager(device="cuda")
        
        with patch('woundseg.models.device.tf.config.experimental.list_physical_devices') as mock_list_devices:
            mock_list_devices.return_value = [Mock()]
            
            # Should not raise any exceptions
            manager.configure_tensorflow()
    
    def test_get_device_info(self):
        """Test device information retrieval."""
        manager = DeviceManager()
        
        info = manager.get_device_info()
        
        assert isinstance(info, dict)
        assert 'device' in info
        assert 'available_devices' in info
        assert 'tensorflow_configured' in info
    
    def test_benchmark_device(self):
        """Test device benchmarking."""
        manager = DeviceManager()
        
        benchmark_result = manager.benchmark_device()
        
        assert isinstance(benchmark_result, dict)
        assert 'device' in benchmark_result
        assert 'benchmark_time' in benchmark_result
        assert 'memory_usage' in benchmark_result


class TestKerasCustom:
    """Test cases for Keras custom objects."""
    
    def test_register_custom_objects(self):
        """Test custom objects registration."""
        # Should not raise any exceptions
        register_custom_objects()
    
    def test_custom_objects_available(self):
        """Test that custom objects are available after registration."""
        register_custom_objects()
        
        # Import and check that custom objects are available
        from woundseg.models.keras_custom import FocalTverskyLoss, IOUScore, DiceScore
        
        assert FocalTverskyLoss is not None
        assert IOUScore is not None
        assert DiceScore is not None
    
    def test_focal_tversky_loss(self):
        """Test FocalTverskyLoss functionality."""
        from woundseg.models.keras_custom import FocalTverskyLoss
        
        loss_fn = FocalTverskyLoss()
        
        # Create test data
        y_true = np.random.rand(1, 128, 128, 1)
        y_pred = np.random.rand(1, 128, 128, 1)
        
        # Should not raise any exceptions
        loss_value = loss_fn(y_true, y_pred)
        
        assert isinstance(loss_value, float)
        assert loss_value >= 0.0
    
    def test_iou_score(self):
        """Test IOUScore functionality."""
        from woundseg.models.keras_custom import IOUScore
        
        iou_metric = IOUScore()
        
        # Create test data
        y_true = np.random.rand(1, 128, 128, 1)
        y_pred = np.random.rand(1, 128, 128, 1)
        
        # Should not raise any exceptions
        iou_value = iou_metric(y_true, y_pred)
        
        assert isinstance(iou_value, float)
        assert 0.0 <= iou_value <= 1.0
    
    def test_dice_score(self):
        """Test DiceScore functionality."""
        from woundseg.models.keras_custom import DiceScore
        
        dice_metric = DiceScore()
        
        # Create test data
        y_true = np.random.rand(1, 128, 128, 1)
        y_pred = np.random.rand(1, 128, 128, 1)
        
        # Should not raise any exceptions
        dice_value = dice_metric(y_true, y_pred)
        
        assert isinstance(dice_value, float)
        assert 0.0 <= dice_value <= 1.0


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model functionality."""
    
    def test_model_provider_integration(self, sample_image: np.ndarray):
        """Test ModelProvider integration."""
        provider = ModelProvider()
        
        # Test model loading (will fail without real model, but should handle gracefully)
        result = provider.load_unet_model()
        
        # Should return False if model not available
        assert isinstance(result, bool)
    
    def test_device_manager_integration(self):
        """Test DeviceManager integration."""
        manager = DeviceManager()
        
        # Test device detection and configuration
        devices = manager.detect_available_devices()
        best_device = manager.select_best_device()
        
        assert isinstance(devices, list)
        assert isinstance(best_device, str)
        assert best_device in devices
    
    def test_model_loading_with_real_file(self, temp_dir: Path):
        """Test model loading with real file structure."""
        # Create a dummy model file
        model_path = temp_dir / "test_model.keras"
        model_path.write_text("dummy model content")
        
        provider = UNetProvider(model_path=model_path)
        
        # Should fail gracefully with dummy content
        result = provider.load_model()
        assert result is False
    
    def test_model_performance(self, sample_image: np.ndarray):
        """Test model performance."""
        import time
        
        provider = ModelProvider()
        
        # Test device benchmarking
        start_time = time.time()
        benchmark_result = provider.device_manager.benchmark_device()
        end_time = time.time()
        
        benchmark_time = end_time - start_time
        
        # Benchmark should be reasonably fast (less than 10 seconds)
        assert benchmark_time < 10.0
        assert isinstance(benchmark_result, dict)


@pytest.mark.slow
class TestModelPerformance:
    """Performance tests for model functionality (marked as slow)."""
    
    def test_model_loading_performance(self, temp_dir: Path):
        """Test model loading performance."""
        import time
        
        # Create a larger dummy model file
        model_path = temp_dir / "large_model.keras"
        model_path.write_bytes(b"dummy model content" * 1000)  # Larger file
        
        provider = UNetProvider(model_path=model_path)
        
        start_time = time.time()
        result = provider.load_model()
        end_time = time.time()
        
        loading_time = end_time - start_time
        
        # Should be reasonably fast even for larger files
        assert loading_time < 5.0
        assert result is False  # Should fail with dummy content
    
    def test_prediction_performance(self, sample_image: np.ndarray):
        """Test prediction performance."""
        import time
        
        provider = ModelProvider()
        
        # Mock the prediction to test performance
        with patch.object(provider, 'unet_provider') as mock_unet:
            mock_unet.predict.return_value = np.random.rand(128, 128, 1)
            
            start_time = time.time()
            
            # Run multiple predictions
            for _ in range(10):
                result = provider.predict_unet(sample_image)
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / 10
            
            # Average prediction time should be reasonable
            assert avg_time < 1.0  # Less than 1 second per prediction
            assert isinstance(result, np.ndarray)