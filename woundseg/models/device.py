"""
Device management for wound segmentation models.

This module provides device selection and configuration logic for optimal
performance across different hardware configurations (CPU, GPU, MPS).
"""

import logging
import os
from typing import Optional, List, Dict, Any
import tensorflow as tf

from ..config import Config

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Device manager for wound segmentation models.
    
    This class handles device selection, configuration, and optimization
    for TensorFlow operations across different hardware platforms.
    """
    
    def __init__(self):
        """Initialize the device manager."""
        self._available_devices: Dict[str, bool] = {}
        self._selected_device: Optional[str] = None
        self._device_info: Dict[str, Any] = {}
        
        self._detect_available_devices()
        self._select_optimal_device()
        
        logger.info(f"DeviceManager initialized. Selected device: {self.selected_device}")
    
    def _detect_available_devices(self) -> None:
        """Detect available devices on the system."""
        self._available_devices = {
            'cpu': True,  # CPU is always available
            'gpu': False,
            'mps': False
        }
        
        # Check for GPU availability
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self._available_devices['gpu'] = True
                self._device_info['gpu'] = {
                    'count': len(gpus),
                    'devices': [gpu.name for gpu in gpus]
                }
                logger.info(f"GPU devices detected: {len(gpus)}")
        except Exception as e:
            logger.warning(f"Error detecting GPU devices: {e}")
        
        # Check for MPS availability (Apple Silicon)
        try:
            # MPS is available on macOS with Apple Silicon
            if hasattr(tf.config, 'list_physical_devices'):
                # This is a simplified check - MPS detection is complex
                import platform
                if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                    self._available_devices['mps'] = True
                    self._device_info['mps'] = {'available': True}
                    logger.info("MPS (Apple Silicon) detected")
        except Exception as e:
            logger.debug(f"MPS detection: {e}")
    
    def _select_optimal_device(self) -> None:
        """
        Select the optimal device based on availability and configuration.
        
        Priority order: GPU > MPS > CPU
        """
        config_device = Config.DEVICE.lower()
        
        if config_device == 'auto':
            # Auto-select based on availability
            if self._available_devices['gpu']:
                self._selected_device = 'gpu'
            elif self._available_devices['mps']:
                self._selected_device = 'mps'
            else:
                self._selected_device = 'cpu'
        else:
            # Use configured device if available
            if self._available_devices.get(config_device, False):
                self._selected_device = config_device
            else:
                logger.warning(f"Configured device '{config_device}' not available, falling back to CPU")
                self._selected_device = 'cpu'
        
        logger.info(f"Selected device: {self._selected_device}")
    
    @property
    def selected_device(self) -> str:
        """Get the currently selected device."""
        return self._selected_device or 'cpu'
    
    @property
    def available_devices(self) -> Dict[str, bool]:
        """Get dictionary of available devices."""
        return self._available_devices.copy()
    
    @property
    def device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        return self._device_info.copy()
    
    def configure_tensorflow(self) -> None:
        """
        Configure TensorFlow for the selected device.
        
        This method applies the device configuration to TensorFlow.
        """
        try:
            if self.selected_device == 'cpu':
                self._configure_cpu()
            elif self.selected_device == 'gpu':
                self._configure_gpu()
            elif self.selected_device == 'mps':
                self._configure_mps()
            
            logger.info(f"TensorFlow configured for {self.selected_device}")
            
        except Exception as e:
            logger.error(f"Failed to configure TensorFlow for {self.selected_device}: {e}")
            logger.info("Falling back to CPU configuration")
            self._configure_cpu()
    
    def _configure_cpu(self) -> None:
        """Configure TensorFlow to use CPU only."""
        try:
            # Hide GPU devices
            tf.config.set_visible_devices([], 'GPU')
            
            # Set CPU threads
            tf.config.threading.set_inter_op_parallelism_threads(Config.TF_THREADS)
            tf.config.threading.set_intra_op_parallelism_threads(Config.TF_THREADS)
            
            logger.info(f"TensorFlow configured for CPU with {Config.TF_THREADS} threads")
            
        except Exception as e:
            logger.error(f"Error configuring CPU: {e}")
    
    def _configure_gpu(self) -> None:
        """Configure TensorFlow to use GPU."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                raise RuntimeError("No GPU devices available")
            
            # Configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision if enabled
            if Config.TF_MIXED_PRECISION:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("Mixed precision enabled for GPU")
            
            # Set threading
            tf.config.threading.set_inter_op_parallelism_threads(Config.TF_THREADS)
            tf.config.threading.set_intra_op_parallelism_threads(Config.TF_THREADS)
            
            logger.info(f"TensorFlow configured for GPU. Devices: {len(gpus)}")
            
        except Exception as e:
            logger.error(f"Error configuring GPU: {e}")
            raise
    
    def _configure_mps(self) -> None:
        """Configure TensorFlow for Apple Silicon MPS."""
        try:
            # MPS configuration is handled automatically by TensorFlow on Apple Silicon
            # We just need to ensure proper threading
            tf.config.threading.set_inter_op_parallelism_threads(Config.TF_THREADS)
            tf.config.threading.set_intra_op_parallelism_threads(Config.TF_THREADS)
            
            logger.info("TensorFlow configured for MPS (Apple Silicon)")
            
        except Exception as e:
            logger.error(f"Error configuring MPS: {e}")
            raise
    
    def get_device_summary(self) -> Dict[str, Any]:
        """
        Get a summary of device configuration.
        
        Returns:
            dict: Device configuration summary
        """
        return {
            'selected_device': self.selected_device,
            'available_devices': self.available_devices,
            'device_info': self.device_info,
            'tensorflow_version': tf.__version__,
            'mixed_precision': Config.TF_MIXED_PRECISION,
            'threads': Config.TF_THREADS
        }
    
    def benchmark_device(self, model: tf.keras.Model, input_shape: tuple = (1, 128, 128, 3), 
                        iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark the current device with a given model.
        
        Args:
            model: The model to benchmark
            input_shape: Input shape for benchmarking
            iterations: Number of iterations to run
            
        Returns:
            dict: Benchmark results (avg_time_ms, throughput)
        """
        try:
            import time
            
            # Create dummy input
            dummy_input = tf.random.normal(input_shape)
            
            # Warm up
            for _ in range(3):
                _ = model.predict(dummy_input, verbose=0)
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                _ = model.predict(dummy_input, verbose=0)
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) * 1000 / iterations
            throughput = iterations / (end_time - start_time)
            
            results = {
                'avg_time_ms': avg_time_ms,
                'throughput_inferences_per_sec': throughput,
                'device': self.selected_device,
                'iterations': iterations
            }
            
            logger.info(f"Device benchmark completed: {avg_time_ms:.2f}ms per inference")
            return results
            
        except Exception as e:
            logger.error(f"Device benchmarking failed: {e}")
            return {'error': str(e)}
    
    def switch_device(self, device: str) -> bool:
        """
        Switch to a different device.
        
        Args:
            device: Target device ('cpu', 'gpu', 'mps')
            
        Returns:
            bool: True if switch was successful
        """
        if device not in self._available_devices:
            logger.error(f"Device '{device}' not available")
            return False
        
        if not self._available_devices[device]:
            logger.error(f"Device '{device}' not detected")
            return False
        
        old_device = self._selected_device
        self._selected_device = device
        
        try:
            self.configure_tensorflow()
            logger.info(f"Successfully switched from {old_device} to {device}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to {device}: {e}")
            self._selected_device = old_device
            return False


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """
    Get the global device manager instance.
    
    Returns:
        DeviceManager: The global device manager instance
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
        _device_manager.configure_tensorflow()
    return _device_manager


def get_best_device() -> str:
    """
    Get the best available device.
    
    Returns:
        str: Best available device ('gpu', 'mps', or 'cpu')
    """
    manager = get_device_manager()
    return manager.selected_device


def configure_tensorflow_device() -> None:
    """Configure TensorFlow for the optimal device."""
    manager = get_device_manager()
    manager.configure_tensorflow()


def get_device_summary() -> Dict[str, Any]:
    """
    Get device configuration summary.
    
    Returns:
        dict: Device configuration summary
    """
    manager = get_device_manager()
    return manager.get_device_summary()