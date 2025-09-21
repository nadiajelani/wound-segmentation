"""
Image preprocessing pipeline for wound segmentation.

This module handles image loading, preprocessing, and augmentation
specifically for the U-Net model workflow.
"""

import logging
import os
from typing import Tuple, Optional, Union, List
import cv2
import numpy as np
import albumentations as A
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing pipeline for wound segmentation.
    
    This class handles image loading, preprocessing, and augmentation
    specifically for the U-Net model workflow.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target size for image resizing (height, width)
        """
        self.target_size = target_size
        logger.info(f"ImagePreprocessor initialized with target size: {target_size}")
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Loaded image as numpy array (RGB format)
            
        Raises:
            FileNotFoundError: If image file is not found
            ValueError: If image cannot be loaded
        """
        image_path = str(image_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            logger.debug(f"Image loaded successfully: {image_path}, shape: {img.shape}")
            return img
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def preprocess_for_unet(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for U-Net model input.
        
        This applies the same preprocessing used in test_wound_progress.py:
        - Resize to target size
        - Normalize to [0, 1] range
        
        Args:
            image: Input image as numpy array
            
        Returns:
            np.ndarray: Preprocessed image ready for U-Net
        """
        try:
            # Resize image to target size
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0, 1] range (same as test_wound_progress.py)
            normalized = resized.astype(np.float32) / 255.0
            
            logger.debug(f"Image preprocessed: {image.shape} -> {normalized.shape}")
            return normalized
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def preprocess_image_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image ready for U-Net
        """
        image = self.load_image(image_path)
        return self.preprocess_for_unet(image)
    
    def get_augmentation_pipeline(self, is_training: bool = True) -> A.Compose:
        """
        Get augmentation pipeline for training.
        
        This is the advanced_augmentation function extracted from wound_medsam.py.
        
        Args:
            is_training: Whether to apply training augmentations
            
        Returns:
            A.Compose: Albumentations pipeline
        """
        if is_training:
            transforms = [
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Rotate(limit=40, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),
            ]
        else:
            # No augmentation for inference
            transforms = [
                A.Resize(self.target_size[0], self.target_size[1]),
            ]
        
        return A.Compose(transforms, additional_targets={'mask': 'mask'})
    
    def apply_augmentation(self, image: np.ndarray, mask: Optional[np.ndarray] = None, 
                          is_training: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply augmentation to image and optionally mask.
        
        Args:
            image: Input image
            mask: Optional mask for segmentation
            is_training: Whether to apply training augmentations
            
        Returns:
            Augmented image, or tuple of (augmented_image, augmented_mask) if mask provided
        """
        try:
            pipeline = self.get_augmentation_pipeline(is_training)
            
            if mask is not None:
                augmented = pipeline(image=image, mask=mask)
                return augmented['image'], augmented['mask']
            else:
                augmented = pipeline(image=image)
                return augmented['image']
                
        except Exception as e:
            logger.error(f"Failed to apply augmentation: {e}")
            raise
    
    def enhance_image_quality(self, image: np.ndarray, min_brightness: float = 100.0) -> np.ndarray:
        """
        Enhance image quality if needed.
        
        This applies the same enhancement logic from wound_medsam.py.
        
        Args:
            image: Input image
            min_brightness: Minimum brightness threshold
            
        Returns:
            np.ndarray: Enhanced image
        """
        try:
            # Convert to grayscale for brightness calculation
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            
            if brightness < min_brightness:
                # Apply enhancement (same as wound_medsam.py)
                enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
                logger.debug(f"Image enhanced: brightness {brightness:.1f} -> {np.mean(cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)):.1f}")
                return enhanced
            else:
                logger.debug(f"Image brightness OK: {brightness:.1f}")
                return image
                
        except Exception as e:
            logger.error(f"Failed to enhance image: {e}")
            raise
    
    def batch_preprocess(self, image_paths: List[Union[str, Path]], 
                        apply_enhancement: bool = True) -> List[np.ndarray]:
        """
        Preprocess multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            apply_enhancement: Whether to apply quality enhancement
            
        Returns:
            List[np.ndarray]: List of preprocessed images
        """
        preprocessed_images = []
        
        for image_path in image_paths:
            try:
                # Load image
                image = self.load_image(image_path)
                
                # Enhance if needed
                if apply_enhancement:
                    image = self.enhance_image_quality(image)
                
                # Preprocess for U-Net
                preprocessed = self.preprocess_for_unet(image)
                preprocessed_images.append(preprocessed)
                
            except Exception as e:
                logger.warning(f"Failed to preprocess {image_path}: {e}")
                continue
        
        logger.info(f"Batch preprocessing completed: {len(preprocessed_images)}/{len(image_paths)} images processed")
        return preprocessed_images
    
    def get_preprocessing_summary(self) -> dict:
        """
        Get summary of preprocessing configuration.
        
        Returns:
            dict: Preprocessing configuration summary
        """
        return {
            'target_size': self.target_size,
            'normalization_range': [0.0, 1.0],
            'color_format': 'RGB',
            'interpolation': 'INTER_LINEAR',
            'enhancement_enabled': True,
            'min_brightness_threshold': 100.0
        }


# Convenience functions for backward compatibility
def preprocess_image_for_unet(image_path: Union[str, Path], 
                             target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Preprocess image for U-Net (convenience function).
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image
    """
    preprocessor = ImagePreprocessor(target_size)
    return preprocessor.preprocess_image_from_path(image_path)


def get_augmentation_pipeline(is_training: bool = True, 
                             target_size: Tuple[int, int] = (128, 128)) -> A.Compose:
    """
    Get augmentation pipeline (convenience function).
    
    Args:
        is_training: Whether to apply training augmentations
        target_size: Target size for resizing
        
    Returns:
        A.Compose: Albumentations pipeline
    """
    preprocessor = ImagePreprocessor(target_size)
    return preprocessor.get_augmentation_pipeline(is_training)