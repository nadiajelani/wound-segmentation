"""
Postprocessing pipeline for wound segmentation.

This module handles mask postprocessing, feature extraction,
and wound analysis based on the workflow from test_wound_progress.py.
"""

import logging
import math
from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np
from pathlib import Path

from ..types import AnalysisResult

logger = logging.getLogger(__name__)


class PostprocessingPipeline:
    """
    Postprocessing pipeline for wound segmentation.
    
    This class handles mask postprocessing, feature extraction,
    and wound analysis based on the workflow from test_wound_progress.py.
    """
    
    def __init__(self, scale_mm_per_pixel: float = 0.1):
        """
        Initialize the postprocessing pipeline.
        
        Args:
            scale_mm_per_pixel: Scale factor for converting pixels to millimeters
        """
        self.scale_mm_per_pixel = scale_mm_per_pixel
        logger.info(f"PostprocessingPipeline initialized with scale: {scale_mm_per_pixel} mm/pixel")
    
    def postprocess_mask(self, mask: np.ndarray, original_shape: Optional[Tuple[int, int]] = None,
                        threshold_percentile: float = 80.0) -> np.ndarray:
        """
        Postprocess the predicted mask.
        
        This applies the same postprocessing logic from test_wound_progress.py:
        - Adaptive thresholding based on percentile
        - Resize to original image size
        
        Args:
            mask: Raw prediction mask
            original_shape: Original image shape (height, width)
            threshold_percentile: Percentile for adaptive thresholding
            
        Returns:
            np.ndarray: Postprocessed binary mask
        """
        try:
            # Apply adaptive thresholding (same logic as test_wound_progress.py)
            thresh_val = np.percentile(mask, threshold_percentile) * 0.8 + 0.2
            mask_bin = (mask > thresh_val).astype(np.uint8) * 255
            
            # Resize to original image size if provided
            if original_shape is not None:
                mask_resized = cv2.resize(mask_bin, (original_shape[1], original_shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                logger.debug(f"Mask postprocessed and resized: {mask.shape} -> {mask_resized.shape}")
                return mask_resized
            else:
                logger.debug(f"Mask postprocessed: threshold={thresh_val:.3f}")
                return mask_bin
                
        except Exception as e:
            logger.error(f"Failed to postprocess mask: {e}")
            raise
    
    def extract_wound_features(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Extract wound features from the mask.
        
        This implements the same feature extraction logic from test_wound_progress.py.
        
        Args:
            mask: Binary mask (0 and 255 values)
            
        Returns:
            dict: Dictionary containing wound features
        """
        try:
            # Apply binary threshold
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No wound detected in mask!")
                return self._get_empty_features()
            
            # Get the largest contour
            cnt = max(contours, key=cv2.contourArea)
            
            # Calculate basic features
            area_px = cv2.contourArea(cnt)
            perimeter_px = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate centroid
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] else 0
            
            # Calculate shape irregularity
            irregularity = (perimeter_px ** 2) / (4 * math.pi * area_px + 1e-6)
            
            # Convert to millimeters
            area_mm = area_px * (self.scale_mm_per_pixel ** 2)
            perimeter_mm = perimeter_px * self.scale_mm_per_pixel
            
            features = {
                "wound_area_px": round(area_px, 2),
                "perimeter_px": round(perimeter_px, 2),
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "centroid": {"x": cx, "y": cy},
                "shape_irregularity": round(irregularity, 3),
                "wound_area_mm2": round(area_mm, 2),
                "perimeter_mm": round(perimeter_mm, 2),
                "contours": contours,
                "largest_contour": cnt
            }
            
            logger.debug(f"Wound features extracted: area={area_mm:.2f} mm², irregularity={irregularity:.3f}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract wound features: {e}")
            return self._get_empty_features()
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Get empty features when no wound is detected."""
        return {
            "wound_area_px": 0,
            "perimeter_px": 0,
            "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
            "centroid": {"x": 0, "y": 0},
            "shape_irregularity": 0,
            "wound_area_mm2": 0,
            "perimeter_mm": 0,
            "contours": [],
            "largest_contour": None
        }
    
    def analyze_wound_condition(self, features: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze wound condition based on extracted features.
        
        This implements the same condition analysis logic from test_wound_progress.py.
        
        Args:
            features: Wound features dictionary
            
        Returns:
            dict: Dictionary containing condition analysis
        """
        try:
            area_px = features["wound_area_px"]
            irregularity = features["shape_irregularity"]
            
            # Determine condition based on area
            if area_px > 10000:
                condition = "⚠️ Large wound — likely chronic or ulcerative"
            elif area_px < 3000:
                condition = "✅ Small wound — early or healing"
            else:
                condition = "➖ Moderate wound — monitor size"
            
            # Add irregularity analysis
            if irregularity > 2.0:
                condition += " | Edge irregularity may indicate inflammation or infection"
            elif irregularity < 1.5:
                condition += " | Regular edges — likely healing"
            
            # Generate instructions
            instructions = (
                "Consult a doctor if area increases or if red, swollen, or exudative. "
                "Continue monitoring weekly."
            )
            
            analysis = {
                "condition": condition,
                "instructions": instructions,
                "severity": self._determine_severity(area_px, irregularity),
                "healing_potential": self._assess_healing_potential(area_px, irregularity)
            }
            
            logger.debug(f"Wound condition analyzed: {analysis['severity']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze wound condition: {e}")
            return {
                "condition": "Analysis failed",
                "instructions": "Manual assessment recommended",
                "severity": "Unknown",
                "healing_potential": "Unknown"
            }
    
    def _determine_severity(self, area_px: float, irregularity: float) -> str:
        """Determine wound severity based on area and irregularity."""
        if area_px > 10000 or irregularity > 2.0:
            return "Severe"
        elif area_px > 3000 or irregularity > 1.5:
            return "Moderate"
        else:
            return "Mild"
    
    def _assess_healing_potential(self, area_px: float, irregularity: float) -> str:
        """Assess healing potential based on area and irregularity."""
        if area_px < 3000 and irregularity < 1.5:
            return "Good"
        elif area_px < 10000 and irregularity < 2.0:
            return "Fair"
        else:
            return "Poor"
    
    def create_overlay(self, original_image: np.ndarray, mask: np.ndarray, 
                      alpha: float = 0.3) -> np.ndarray:
        """
        Create overlay visualization of mask on original image.
        
        Args:
            original_image: Original image
            mask: Binary mask
            alpha: Transparency for overlay
            
        Returns:
            np.ndarray: Overlay image
        """
        try:
            # Create colored mask
            colored_mask = np.zeros_like(original_image)
            colored_mask[mask > 0] = [0, 255, 0]  # Green for wound area
            
            # Create overlay
            overlay = original_image.copy()
            overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
            
            logger.debug(f"Overlay created with alpha={alpha}")
            return overlay
            
        except Exception as e:
            logger.error(f"Failed to create overlay: {e}")
            raise
    
    def create_contour_visualization(self, original_image: np.ndarray, 
                                   features: Dict[str, Any]) -> np.ndarray:
        """
        Create contour visualization on original image.
        
        Args:
            original_image: Original image
            features: Wound features dictionary
            
        Returns:
            np.ndarray: Image with contours drawn
        """
        try:
            vis_image = original_image.copy()
            contours = features.get("contours", [])
            
            if contours:
                # Draw contours
                cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)
                
                # Draw bounding box
                bbox = features["bounding_box"]
                cv2.rectangle(vis_image, (bbox["x"], bbox["y"]), 
                            (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]), 
                            (255, 0, 0), 2)
                
                # Draw centroid
                centroid = features["centroid"]
                cv2.circle(vis_image, (centroid["x"], centroid["y"]), 5, (0, 0, 255), -1)
            
            logger.debug(f"Contour visualization created")
            return vis_image
            
        except Exception as e:
            logger.error(f"Failed to create contour visualization: {e}")
            raise
    
    def batch_postprocess(self, masks: List[np.ndarray], 
                         original_shapes: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """
        Postprocess multiple masks in batch.
        
        Args:
            masks: List of raw prediction masks
            original_shapes: List of original image shapes
            
        Returns:
            List[Dict]: List of postprocessing results
        """
        results = []
        
        for i, (mask, shape) in enumerate(zip(masks, original_shapes)):
            try:
                # Postprocess mask
                processed_mask = self.postprocess_mask(mask, shape)
                
                # Extract features
                features = self.extract_wound_features(processed_mask)
                
                # Analyze condition
                analysis = self.analyze_wound_condition(features)
                
                result = {
                    "processed_mask": processed_mask,
                    "features": features,
                    "analysis": analysis
                }
                
                results.append(result)
                logger.debug(f"Batch postprocessing {i+1}/{len(masks)} completed")
                
            except Exception as e:
                logger.warning(f"Failed to postprocess mask {i+1}: {e}")
                results.append({
                    "processed_mask": np.zeros_like(masks[i]),
                    "features": self._get_empty_features(),
                    "analysis": {"condition": "Failed", "severity": "Unknown", "healing_potential": "Unknown"}
                })
        
        logger.info(f"Batch postprocessing completed: {len(results)} results")
        return results
    
    def get_postprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of postprocessing pipeline configuration.
        
        Returns:
            dict: Postprocessing pipeline configuration summary
        """
        return {
            "scale_mm_per_pixel": self.scale_mm_per_pixel,
            "threshold_percentile": 80.0,
            "severity_thresholds": {
                "mild": {"max_area_px": 3000, "max_irregularity": 1.5},
                "moderate": {"max_area_px": 10000, "max_irregularity": 2.0},
                "severe": {"min_area_px": 10000, "min_irregularity": 2.0}
            },
            "healing_assessment_criteria": {
                "good": {"max_area_px": 3000, "max_irregularity": 1.5},
                "fair": {"max_area_px": 10000, "max_irregularity": 2.0},
                "poor": {"min_area_px": 10000, "min_irregularity": 2.0}
            }
        }


# Convenience functions for backward compatibility
def postprocess_mask(mask: np.ndarray, original_shape: Optional[Tuple[int, int]] = None,
                    scale_mm_per_pixel: float = 0.1) -> np.ndarray:
    """
    Postprocess mask (convenience function).
    
    Args:
        mask: Raw prediction mask
        original_shape: Original image shape
        scale_mm_per_pixel: Scale factor for mm conversion
        
    Returns:
        np.ndarray: Postprocessed mask
    """
    pipeline = PostprocessingPipeline(scale_mm_per_pixel)
    return pipeline.postprocess_mask(mask, original_shape)


def extract_wound_features(mask: np.ndarray, scale_mm_per_pixel: float = 0.1) -> Dict[str, Any]:
    """
    Extract wound features (convenience function).
    
    Args:
        mask: Binary mask
        scale_mm_per_pixel: Scale factor for mm conversion
        
    Returns:
        dict: Wound features
    """
    pipeline = PostprocessingPipeline(scale_mm_per_pixel)
    return pipeline.extract_wound_features(mask)