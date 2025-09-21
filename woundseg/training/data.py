#!/usr/bin/env python3
"""
Training data utilities for wound segmentation and classification.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from woundseg.config import Config

@dataclass
class TrainingData:
    """Container for training data."""
    images: np.ndarray
    masks: np.ndarray
    labels: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

class DataLoader:
    """Data loader for wound segmentation and classification training."""
    
    def __init__(self, data_dir: Path, image_size: Tuple[int, int] = (256, 256)):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.class_names = ['non_wound', 'wound']
        
    def load_segmentation_data(self, split: str = 'train') -> TrainingData:
        """
        Load segmentation training data.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            TrainingData with images and masks
        """
        images_dir = self.data_dir / split / 'images'
        masks_dir = self.data_dir / split / 'masks'
        
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(f"Data directories not found: {images_dir} or {masks_dir}")
        
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        images = []
        masks = []
        
        for img_file in image_files:
            # Load image
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)
            img = img.astype(np.float32) / 255.0
            images.append(img)
            
            # Load corresponding mask
            mask_file = masks_dir / img_file.name
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, self.image_size)
                mask = (mask > 128).astype(np.float32)  # Binary mask
                masks.append(mask)
            else:
                # Create empty mask if not found
                masks.append(np.zeros(self.image_size, dtype=np.float32))
        
        return TrainingData(
            images=np.array(images),
            masks=np.array(masks),
            metadata={'split': split, 'num_samples': len(images)}
        )
    
    def load_classification_data(self, split: str = 'train') -> TrainingData:
        """
        Load classification training data.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            TrainingData with images and labels
        """
        data_dir = self.data_dir / split
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_file in class_dir.glob('*.png') + class_dir.glob('*.jpg'):
                # Load image
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_size)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(class_idx)
        
        return TrainingData(
            images=np.array(images),
            masks=None,
            labels=np.array(labels),
            metadata={'split': split, 'num_samples': len(images), 'class_names': self.class_names}
        )
    
    def create_tf_dataset(self, training_data: TrainingData, batch_size: int = 8, 
                         shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from training data.
        
        Args:
            training_data: TrainingData object
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            TensorFlow dataset
        """
        if training_data.masks is not None:
            # Segmentation dataset
            dataset = tf.data.Dataset.from_tensor_slices({
                'images': training_data.images,
                'masks': training_data.masks
            })
        else:
            # Classification dataset
            dataset = tf.data.Dataset.from_tensor_slices({
                'images': training_data.images,
                'labels': training_data.labels
            })
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(training_data.images))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

class DataAugmentation:
    """Data augmentation utilities for training."""
    
    @staticmethod
    def get_segmentation_augmentation():
        """Get augmentation pipeline for segmentation."""
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1)
        ])
    
    @staticmethod
    def get_classification_augmentation():
        """Get augmentation pipeline for classification."""
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2)
        ])

def validate_data_structure(data_dir: Path) -> Dict[str, Any]:
    """
    Validate training data directory structure.
    
    Args:
        data_dir: Path to training data directory
        
    Returns:
        Dictionary with validation results
    """
    data_dir = Path(data_dir)
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check for required directories
    required_dirs = ['train', 'val', 'test']
    for split in required_dirs:
        split_dir = data_dir / split
        if not split_dir.exists():
            results['warnings'].append(f"Missing {split} directory")
            continue
        
        # Check for images and masks (segmentation)
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'
        
        if images_dir.exists() and masks_dir.exists():
            image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
            mask_files = list(masks_dir.glob('*.png')) + list(masks_dir.glob('*.jpg'))
            
            results['stats'][f'{split}_segmentation'] = {
                'images': len(image_files),
                'masks': len(mask_files)
            }
            
            if len(image_files) != len(mask_files):
                results['warnings'].append(f"Mismatch in {split}: {len(image_files)} images vs {len(mask_files)} masks")
        
        # Check for classification data
        wound_dir = split_dir / 'wound'
        non_wound_dir = split_dir / 'non_wound'
        
        if wound_dir.exists() and non_wound_dir.exists():
            wound_files = list(wound_dir.glob('*.png')) + list(wound_dir.glob('*.jpg'))
            non_wound_files = list(non_wound_dir.glob('*.png')) + list(non_wound_dir.glob('*.jpg'))
            
            results['stats'][f'{split}_classification'] = {
                'wound': len(wound_files),
                'non_wound': len(non_wound_files)
            }
    
    if not results['stats']:
        results['valid'] = False
        results['errors'].append("No valid training data found")
    
    return results