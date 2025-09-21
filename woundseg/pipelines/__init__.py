"""
Pipelines package for wound segmentation.

This package provides the complete processing pipelines for wound analysis,
including preprocessing, segmentation, postprocessing, and orchestration.
"""

from .preprocess import (
    ImagePreprocessor,
    preprocess_image_for_unet,
    get_augmentation_pipeline
)
from .segment import SegmentationPipeline
from .postprocess import (
    PostprocessingPipeline,
    postprocess_mask,
    extract_wound_features
)
from .analyze import (
    AnalysisPipeline,
    analyze_single_image,
    analyze_batch_images
)

__all__ = [
    # Preprocessing
    'ImagePreprocessor',
    'preprocess_image_for_unet',
    'get_augmentation_pipeline',
    
    # Segmentation
    'SegmentationPipeline',
    
    # Postprocessing
    'PostprocessingPipeline',
    'postprocess_mask',
    'extract_wound_features',
    
    # Analysis Orchestration
    'AnalysisPipeline',
    'analyze_single_image',
    'analyze_batch_images'
]