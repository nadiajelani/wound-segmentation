"""
Models package for wound segmentation.

This package provides model management, loading, and device configuration
for the wound segmentation system.
"""

from .provider import ModelProvider, get_model_provider, reset_model_provider
from .unet import UNetProvider, build_unet
from .medsam import MedSAMProvider, get_medsam_provider, reset_medsam_provider
from .keras_custom import (
    FocalTverskyLoss, 
    IOUScore, 
    DiceScore,
    register_custom_objects,
    get_custom_losses,
    get_custom_metrics
)
from .device import (
    DeviceManager,
    get_device_manager,
    get_best_device,
    configure_tensorflow_device,
    get_device_summary
)

__all__ = [
    # Model Provider
    'ModelProvider',
    'get_model_provider',
    'reset_model_provider',
    
    # U-Net
    'UNetProvider',
    'build_unet',
    
    # MedSAM
    'MedSAMProvider',
    'get_medsam_provider',
    'reset_medsam_provider',
    
    # Custom Objects
    'FocalTverskyLoss',
    'IOUScore',
    'DiceScore',
    'register_custom_objects',
    'get_custom_losses',
    'get_custom_metrics',
    
    # Device Management
    'DeviceManager',
    'get_device_manager',
    'get_best_device',
    'configure_tensorflow_device',
    'get_device_summary'
]