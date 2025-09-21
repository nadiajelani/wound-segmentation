"""
Main analysis orchestration pipeline for wound segmentation.

This module orchestrates the complete wound analysis workflow,
combining preprocessing, segmentation, and postprocessing.
"""

import logging
import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np

from ..models import ModelProvider, get_model_provider
from ..types import AnalysisOptions, AnalysisResult, Patient, Artifacts
from .preprocess import ImagePreprocessor
from .segment import SegmentationPipeline
from .postprocess import PostprocessingPipeline

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """
    Main analysis orchestration pipeline for wound segmentation.
    
    This class orchestrates the complete wound analysis workflow,
    combining preprocessing, segmentation, and postprocessing.
    """
    
    def __init__(self, model_provider: Optional[ModelProvider] = None, 
                 scale_mm_per_pixel: float = 0.1):
        """
        Initialize the analysis pipeline.
        
        Args:
            model_provider: Model provider instance (optional)
            scale_mm_per_pixel: Scale factor for converting pixels to millimeters
        """
        self.model_provider = model_provider or get_model_provider()
        self.scale_mm_per_pixel = scale_mm_per_pixel
        
        # Initialize pipeline components
        self.preprocessor = ImagePreprocessor()
        self.segmentation_pipeline = SegmentationPipeline(self.model_provider)
        self.postprocessor = PostprocessingPipeline(scale_mm_per_pixel)
        
        logger.info("AnalysisPipeline initialized")
    
    def analyze_single_image(self, image_path: Union[str, Path], 
                           options: Optional[AnalysisOptions] = None) -> AnalysisResult:
        """
        Analyze a single wound image.
        
        This is the main analysis function that orchestrates the complete workflow.
        
        Args:
            image_path: Path to the input image
            options: Analysis options (optional)
            
        Returns:
            AnalysisResult: Complete analysis result
            
        Raises:
            FileNotFoundError: If image file is not found
            Exception: If analysis fails
        """
        try:
            logger.info(f"Analyzing single image: {image_path}")
            
            # Set default options if not provided
            if options is None:
                options = AnalysisOptions()
            
            # Step 1: Load and preprocess image
            original_image = self.preprocessor.load_image(image_path)
            preprocessed_image = self.preprocessor.preprocess_for_unet(original_image)
            
            # Step 2: Segment the wound
            segmentation_result = self.segmentation_pipeline.predict_mask(
                preprocessed_image, 
                use_tta=options.additional_params.get('use_tta', False)
            )
            
            # Step 3: Postprocess mask
            processed_mask = self.postprocessor.postprocess_mask(
                segmentation_result.mask, 
                original_shape=original_image.shape[:2]
            )
            
            # Step 4: Extract wound features
            features = self.postprocessor.extract_wound_features(processed_mask)
            
            # Step 5: Analyze wound condition
            analysis = self.postprocessor.analyze_wound_condition(features)
            
            # Step 6: Create artifacts if requested
            artifacts = None
            if options.generate_report or options.generate_heatmap:
                artifacts = self._create_artifacts(
                    original_image, processed_mask, features, 
                    str(options.output_dir) if options.output_dir else None
                )
            
            # Step 7: Create analysis result
            result = AnalysisResult(
                severity=analysis["severity"],
                healing_potential=analysis["healing_potential"],
                area_mm2=features["wound_area_mm2"],
                confidence_score=segmentation_result.confidence,
                original_image=original_image,
                mask=processed_mask,
                overlay=self.postprocessor.create_overlay(original_image, processed_mask) if options.generate_heatmap else None,
                patient=options.patient,
                options=options,
                artifacts=artifacts,
                processing_time=None,  # Could be measured if needed
                model_versions={"unet": "simclr_unet_patch_wound.keras"},
                metadata={
                    "features": features,
                    "analysis": analysis,
                    "segmentation_method": segmentation_result.method
                }
            )
            
            logger.info(f"Analysis completed: {result.severity}, area={result.area_mm2:.2f} mm²")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {e}")
            raise
    
    def analyze_batch(self, image_paths: List[Union[str, Path]], 
                     options: Optional[AnalysisOptions] = None) -> List[AnalysisResult]:
        """
        Analyze multiple wound images in batch.
        
        Args:
            image_paths: List of image file paths
            options: Analysis options (optional)
            
        Returns:
            List[AnalysisResult]: List of analysis results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Batch analysis {i+1}/{len(image_paths)}: {image_path}")
                result = self.analyze_single_image(image_path, options)
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to analyze image {i+1} ({image_path}): {e}")
                # Create a failed result to maintain list length
                results.append(self._create_failed_result(image_path, str(e)))
        
        logger.info(f"Batch analysis completed: {len(results)} results")
        return results
    
    def _create_artifacts(self, original_image: np.ndarray, mask: np.ndarray, 
                         features: Dict[str, Any], output_dir: Optional[str]) -> Artifacts:
        """
        Create analysis artifacts (masks, overlays, etc.).
        
        Args:
            original_image: Original input image
            mask: Processed segmentation mask
            features: Extracted wound features
            output_dir: Output directory for artifacts
            
        Returns:
            Artifacts: Generated artifacts
        """
        try:
            artifacts = Artifacts()
            
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save mask
                mask_path = os.path.join(output_dir, "wound_mask.png")
                import cv2
                cv2.imwrite(mask_path, mask)
                artifacts.mask_path = mask_path
                
                # Save overlay
                overlay = self.postprocessor.create_overlay(original_image, mask)
                overlay_path = os.path.join(output_dir, "wound_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                artifacts.overlay_path = overlay_path
                
                # Save contour visualization
                contour_vis = self.postprocessor.create_contour_visualization(original_image, features)
                contour_path = os.path.join(output_dir, "wound_contours.png")
                cv2.imwrite(contour_path, cv2.cvtColor(contour_vis, cv2.COLOR_RGB2BGR))
                artifacts.additional_files["contour_visualization"] = contour_path
            
            return artifacts
            
        except Exception as e:
            logger.warning(f"Failed to create artifacts: {e}")
            return Artifacts()
    
    def _create_failed_result(self, image_path: Union[str, Path], error_message: str) -> AnalysisResult:
        """
        Create a failed analysis result.
        
        Args:
            image_path: Path to the failed image
            error_message: Error message
            
        Returns:
            AnalysisResult: Failed analysis result
        """
        return AnalysisResult(
            severity="Unknown",
            healing_potential="Unknown",
            area_mm2=0.0,
            confidence_score=0.0,
            original_image=np.zeros((128, 128, 3), dtype=np.uint8),
            mask=np.zeros((128, 128), dtype=np.uint8),
            metadata={"error": error_message, "image_path": str(image_path)}
        )
    
    def run_complete_analysis(self, image_path: Union[str, Path], 
                            patient_info: Optional[Patient] = None,
                            output_dir: Optional[Union[str, Path]] = None,
                            use_tta: bool = False) -> AnalysisResult:
        """
        Run complete analysis with default options.
        
        This is a convenience method that creates default options and runs analysis.
        
        Args:
            image_path: Path to the input image
            patient_info: Patient information (optional)
            output_dir: Output directory for results (optional)
            use_tta: Whether to use test-time augmentation
            
        Returns:
            AnalysisResult: Complete analysis result
        """
        options = AnalysisOptions(
            use_medsam=False,  # MedSAM not used in current workflow
            generate_report=True,
            generate_heatmap=True,
            patient=patient_info,
            output_dir=output_dir,
            additional_params={"use_tta": use_tta}
        )
        
        return self.analyze_single_image(image_path, options)
    
    def get_analysis_summary(self, results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        Get summary statistics for a batch of analysis results.
        
        Args:
            results: List of analysis results
            
        Returns:
            dict: Summary statistics
        """
        try:
            if not results:
                return {"error": "No results to summarize"}
            
            # Calculate statistics
            areas = [r.area_mm2 for r in results if r.area_mm2 > 0]
            confidences = [r.confidence_score for r in results if r.confidence_score > 0]
            
            severity_counts = {}
            healing_counts = {}
            
            for result in results:
                severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
                healing_counts[result.healing_potential] = healing_counts.get(result.healing_potential, 0) + 1
            
            summary = {
                "total_images": len(results),
                "successful_analyses": len([r for r in results if r.area_mm2 > 0]),
                "average_area_mm2": np.mean(areas) if areas else 0,
                "median_area_mm2": np.median(areas) if areas else 0,
                "average_confidence": np.mean(confidences) if confidences else 0,
                "severity_distribution": severity_counts,
                "healing_potential_distribution": healing_counts
            }
            
            logger.info(f"Analysis summary: {summary['successful_analyses']}/{summary['total_images']} successful")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create analysis summary: {e}")
            return {"error": str(e)}
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get summary of pipeline configuration.
        
        Returns:
            dict: Pipeline configuration summary
        """
        return {
            "model_provider": "available" if self.model_provider else "not_available",
            "scale_mm_per_pixel": self.scale_mm_per_pixel,
            "preprocessing": self.preprocessor.get_preprocessing_summary(),
            "segmentation": self.segmentation_pipeline.get_segmentation_summary(),
            "postprocessing": self.postprocessor.get_postprocessing_summary()
        }


# Convenience functions for backward compatibility
def analyze_single_image(image_path: Union[str, Path], 
                        patient_info: Optional[Patient] = None,
                        output_dir: Optional[Union[str, Path]] = None) -> AnalysisResult:
    """
    Analyze a single wound image (convenience function).
    
    Args:
        image_path: Path to the input image
        patient_info: Patient information (optional)
        output_dir: Output directory for results (optional)
        
    Returns:
        AnalysisResult: Analysis result
    """
    pipeline = AnalysisPipeline()
    return pipeline.run_complete_analysis(image_path, patient_info, output_dir)


def analyze_batch_images(image_paths: List[Union[str, Path]], 
                        output_dir: Optional[Union[str, Path]] = None) -> List[AnalysisResult]:
    """
    Analyze multiple wound images (convenience function).
    
    Args:
        image_paths: List of image file paths
        output_dir: Output directory for results (optional)
        
    Returns:
        List[AnalysisResult]: List of analysis results
    """
    pipeline = AnalysisPipeline()
    options = AnalysisOptions(
        generate_report=True,
        generate_heatmap=True,
        output_dir=output_dir
    )
    return pipeline.analyze_batch(image_paths, options)