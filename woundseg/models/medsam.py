#!/usr/bin/env python3
"""
MedSAM model provider for wound segmentation.

This module provides MedSAM (Medical Segment Anything Model) integration
with proper feature flag support and error handling.
"""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from ..config import Config
from ..logging import get_logger
from ..utils.exceptions import ModelLoadingError, ConfigurationError

logger = get_logger(__name__)

class MedSAMProvider:
    """
    MedSAM model provider with feature flag support.
    
    Provides MedSAM model loading and inference with proper error handling
    and feature flag integration.
    """
    
    def __init__(self):
        """Initialize MedSAM provider."""
        self.enabled = Config.ENABLE_MEDSAM
        self.model_path = Config.MEDSAM_WEIGHTS_PATH
        self.model = None
        self.device = None
        
        # Check PyTorch availability
        self.pytorch_available = self._check_pytorch()
        
        if self.enabled and self.pytorch_available:
            self._initialize_model()
        elif self.enabled and not self.pytorch_available:
            logger.warning("MedSAM enabled but PyTorch not available - install with: pip install torch")
        else:
            logger.info("MedSAM provider initialized (disabled by config)")
    
    def _check_pytorch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            import torchvision
            logger.info("PyTorch available for MedSAM")
            return True
        except ImportError:
            logger.warning("PyTorch not available - MedSAM disabled")
            return False
    
    def _initialize_model(self):
        """Initialize MedSAM model."""
        try:
            if not Path(self.model_path).exists():
                raise ModelLoadingError(
                    model_name="MedSAM",
                    model_path=Path(self.model_path),
                    reason="Model file not found"
                )
            
            logger.info("Loading MedSAM model...")
            
            # Lazy import to avoid dependency issues
            import torch
            from segment_anything import sam_model_registry, SamPredictor
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"MedSAM using device: {self.device}")
            
            # Load model
            sam = sam_model_registry["vit_b"](checkpoint=self.model_path)
            sam.to(device=self.device)
            
            self.model = SamPredictor(sam)
            
            logger.info("MedSAM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MedSAM model: {e}")
            self.model = None
            raise ModelLoadingError(
                model_name="MedSAM",
                model_path=Path(self.model_path),
                reason=str(e)
            )
    
    def is_available(self) -> bool:
        """
        Check if MedSAM is available and ready.
        
        Returns:
            bool: True if MedSAM is available and loaded
        """
        return (self.enabled and 
                self.pytorch_available and 
                self.model is not None)
    
    def segment_image(self, image: np.ndarray, 
                     prompt_points: Optional[np.ndarray] = None,
                     prompt_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment wound in image using MedSAM.
        
        Args:
            image: Input image as numpy array (BGR format)
            prompt_points: Optional prompt points for segmentation
            prompt_labels: Optional labels for prompt points
            
        Returns:
            Segmentation mask as numpy array
        """
        if not self.is_available():
            raise RuntimeError("MedSAM not available - check configuration and dependencies")
        
        try:
            logger.info("Running MedSAM segmentation")
            
            # Convert BGR to RGB for MedSAM
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set image for predictor
            self.model.set_image(image_rgb)
            
            # Generate prompt points if not provided
            if prompt_points is None:
                # Use center of image as default prompt
                h, w = image.shape[:2]
                prompt_points = np.array([[w//2, h//2]])
                prompt_labels = np.array([1])  # Positive prompt
            
            # Run prediction
            masks, scores, logits = self.model.predict(
                point_coords=prompt_points,
                point_labels=prompt_labels,
                multimask_output=True
            )
            
            # Select best mask
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            # Convert to uint8
            mask = (mask * 255).astype(np.uint8)
            
            logger.info(f"MedSAM segmentation completed, score: {scores[best_mask_idx]:.3f}")
            return mask
            
        except Exception as e:
            logger.error(f"MedSAM segmentation failed: {e}")
            raise
    
    def segment_with_automatic_prompts(self, image: np.ndarray) -> np.ndarray:
        """
        Segment wound using automatic prompt generation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Segmentation mask as numpy array
        """
        if not self.is_available():
            raise RuntimeError("MedSAM not available")
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set image
            self.model.set_image(image_rgb)
            
            # Generate automatic prompts using image analysis
            prompt_points, prompt_labels = self._generate_automatic_prompts(image)
            
            # Run segmentation
            return self.segment_image(image, prompt_points, prompt_labels)
            
        except Exception as e:
            logger.error(f"Automatic MedSAM segmentation failed: {e}")
            raise
    
    def _generate_automatic_prompts(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate automatic prompt points for MedSAM.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (prompt_points, prompt_labels)
        """
        # Simple automatic prompt generation
        # In a real implementation, this could use edge detection,
        # color analysis, or other computer vision techniques
        
        h, w = image.shape[:2]
        
        # Generate multiple prompt points
        points = []
        labels = []
        
        # Center point (positive)
        points.append([w//2, h//2])
        labels.append(1)
        
        # Corner points (negative)
        corner_offset = min(w, h) // 4
        corners = [
            [corner_offset, corner_offset],
            [w - corner_offset, corner_offset],
            [corner_offset, h - corner_offset],
            [w - corner_offset, h - corner_offset]
        ]
        
        for corner in corners:
            points.append(corner)
            labels.append(0)  # Negative prompt
        
        return np.array(points), np.array(labels)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the MedSAM model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'enabled': self.enabled,
            'available': self.is_available(),
            'pytorch_available': self.pytorch_available,
            'model_path': str(self.model_path),
            'device': str(self.device) if self.device else None,
            'model_loaded': self.model is not None
        }
    
    def compare_with_unet(self, image: np.ndarray, unet_mask: np.ndarray) -> Dict[str, Any]:
        """
        Compare MedSAM segmentation with U-Net segmentation.
        
        Args:
            image: Input image as numpy array
            unet_mask: U-Net segmentation mask
            
        Returns:
            Dictionary with comparison results
        """
        if not self.is_available():
            raise RuntimeError("MedSAM not available for comparison")
        
        try:
            # Get MedSAM segmentation
            medsam_mask = self.segment_with_automatic_prompts(image)
            
            # Calculate comparison metrics
            intersection = np.logical_and(medsam_mask > 0, unet_mask > 0)
            union = np.logical_or(medsam_mask > 0, unet_mask > 0)
            
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            
            dice = (2 * np.sum(intersection)) / (np.sum(medsam_mask > 0) + np.sum(unet_mask > 0))
            
            comparison = {
                'iou': float(iou),
                'dice': float(dice),
                'medsam_area': int(np.sum(medsam_mask > 0)),
                'unet_area': int(np.sum(unet_mask > 0)),
                'area_difference': int(np.sum(medsam_mask > 0) - np.sum(unet_mask > 0)),
                'medsam_mask_shape': medsam_mask.shape,
                'unet_mask_shape': unet_mask.shape
            }
            
            logger.info(f"MedSAM vs U-Net comparison - IoU: {iou:.3f}, Dice: {dice:.3f}")
            return comparison
            
        except Exception as e:
            logger.error(f"MedSAM comparison failed: {e}")
            raise


# Global MedSAM provider instance
_medsam_provider: Optional[MedSAMProvider] = None

def get_medsam_provider() -> MedSAMProvider:
    """
    Get the global MedSAM provider instance.
    
    Returns:
        MedSAMProvider: The global MedSAM provider instance
    """
    global _medsam_provider
    if _medsam_provider is None:
        _medsam_provider = MedSAMProvider()
    return _medsam_provider

def reset_medsam_provider() -> None:
    """Reset the global MedSAM provider (useful for testing)."""
    global _medsam_provider
    _medsam_provider = None
    logger.info("MedSAM provider reset")