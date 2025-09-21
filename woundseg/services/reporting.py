"""
Reporting service for PDF generation.

This module provides PDF report generation for both patient
and clinician variants of wound analysis results.
"""

import logging
import io
import base64
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image

from ..config import Config
from ..types import AnalysisResult, Patient
from .storage import get_storage_service
from .voice import get_voice_service

logger = logging.getLogger(__name__)


class ReportingService:
    """
    Reporting service for PDF generation.
    
    Provides comprehensive PDF report generation for both patient
    and clinician audiences with different content and styling.
    """
    
    def __init__(self, storage_service=None, voice_service=None):
        """
        Initialize the reporting service.
        
        Args:
            storage_service: Storage service instance (optional)
            voice_service: Voice service instance (optional)
        """
        self.storage_service = storage_service or get_storage_service()
        self.voice_service = voice_service or get_voice_service()
        
        # Initialize PDF library
        try:
            from fpdf import FPDF
            self.FPDF = FPDF
            self.pdf_available = True
            logger.info("ReportingService initialized with FPDF support")
        except ImportError:
            logger.warning("FPDF not available - PDF generation disabled")
            self.pdf_available = False
    
    def is_available(self) -> bool:
        """
        Check if reporting service is available.
        
        Returns:
            bool: True if PDF generation is available
        """
        return self.pdf_available
    
    def generate_patient_report(self, analysis_result: AnalysisResult, 
                              patient: Optional[Patient] = None,
                              include_voice: bool = True) -> Optional[Path]:
        """
        Generate patient-friendly PDF report.
        
        Args:
            analysis_result: Analysis result data
            patient: Patient information (optional)
            include_voice: Whether to include voice summary (optional)
            
        Returns:
            Path to saved PDF file, or None if generation failed
        """
        if not self.is_available():
            logger.warning("PDF generation not available - skipping report generation")
            return None
        
        try:
            # Create PDF
            pdf = self.FPDF()
            pdf.add_page()
            
            # Set up patient-friendly styling
            self._setup_patient_styling(pdf)
            
            # Add header
            self._add_patient_header(pdf, patient)
            
            # Add main content
            self._add_patient_content(pdf, analysis_result)
            
            # Add images
            self._add_patient_images(pdf, analysis_result)
            
            # Add footer
            self._add_patient_footer(pdf)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patient_report_{timestamp}.pdf"
            
            # Save PDF
            pdf_data = pdf.output(dest='S')
            if isinstance(pdf_data, bytearray):
                pdf_data = bytes(pdf_data)  # Convert bytearray to bytes
            elif isinstance(pdf_data, str):
                pdf_data = pdf_data.encode('latin1')
            elif isinstance(pdf_data, bytes):
                pass  # Already bytes
            else:
                pdf_data = str(pdf_data).encode('latin1')
            pdf_path = self.storage_service.save_pdf(pdf_data, 'reports', filename)
            
            # Generate voice summary if requested
            if include_voice and self.voice_service.is_available():
                voice_filename = f"patient_voice_{timestamp}.mp3"
                self.voice_service.generate_summary(
                    analysis_result.__dict__, 
                    audience="patient", 
                    filename=voice_filename
                )
            
            logger.info(f"Patient report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate patient report: {e}")
            return None
    
    def generate_clinician_report(self, analysis_result: AnalysisResult,
                                patient: Optional[Patient] = None,
                                include_voice: bool = True) -> Optional[Path]:
        """
        Generate clinician-focused PDF report.
        
        Args:
            analysis_result: Analysis result data
            patient: Patient information (optional)
            include_voice: Whether to include voice summary (optional)
            
        Returns:
            Path to saved PDF file, or None if generation failed
        """
        if not self.is_available():
            logger.warning("PDF generation not available - skipping report generation")
            return None
        
        try:
            # Create PDF
            pdf = self.FPDF()
            pdf.add_page()
            
            # Set up clinician styling
            self._setup_clinician_styling(pdf)
            
            # Add header
            self._add_clinician_header(pdf, patient)
            
            # Add technical content
            self._add_clinician_content(pdf, analysis_result)
            
            # Add detailed images
            self._add_clinician_images(pdf, analysis_result)
            
            # Add technical appendix
            self._add_technical_appendix(pdf, analysis_result)
            
            # Add footer
            self._add_clinician_footer(pdf)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clinician_report_{timestamp}.pdf"
            
            # Save PDF
            pdf_data = pdf.output(dest='S')
            if isinstance(pdf_data, bytearray):
                pdf_data = bytes(pdf_data)  # Convert bytearray to bytes
            elif isinstance(pdf_data, str):
                pdf_data = pdf_data.encode('latin1')
            elif isinstance(pdf_data, bytes):
                pass  # Already bytes
            else:
                pdf_data = str(pdf_data).encode('latin1')
            pdf_path = self.storage_service.save_pdf(pdf_data, 'reports', filename)
            
            # Generate voice summary if requested
            if include_voice and self.voice_service.is_available():
                voice_filename = f"clinician_voice_{timestamp}.mp3"
                self.voice_service.generate_summary(
                    analysis_result.__dict__, 
                    audience="clinician", 
                    filename=voice_filename
                )
            
            logger.info(f"Clinician report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate clinician report: {e}")
            return None
    
    def _setup_patient_styling(self, pdf):
        """Set up patient-friendly styling."""
        # Set font
        pdf.set_font('Arial', '', 12)
        
        # Set colors (patient-friendly)
        pdf.set_text_color(0, 0, 0)  # Black text
        pdf.set_fill_color(240, 248, 255)  # Light blue background
    
    def _setup_clinician_styling(self, pdf):
        """Set up clinician-focused styling."""
        # Set font
        pdf.set_font('Arial', '', 10)
        
        # Set colors (professional)
        pdf.set_text_color(0, 0, 0)  # Black text
        pdf.set_fill_color(255, 255, 255)  # White background
    
    def _add_patient_header(self, pdf, patient):
        """Add patient-friendly header."""
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Your Wound Analysis Report', 0, 1, 'C')
        pdf.ln(5)
        
        if patient and patient.name and patient.name != 'Unknown':
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 8, f'Patient: {patient.name}', 0, 1)
            if patient.age:
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 8, f'Age: {patient.age} years old', 0, 1)
            pdf.ln(3)
        else:
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, 'Patient: Not specified', 0, 1)
            pdf.ln(3)
        
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f'Report Date: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 0, 1)
        pdf.ln(10)
    
    def _add_clinician_header(self, pdf, patient):
        """Add clinician-focused header."""
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'WOUND ANALYSIS REPORT - CLINICAL', 0, 1, 'C')
        pdf.ln(5)
        
        if patient and patient.name and patient.name != 'Unknown':
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 6, f'Patient: {patient.name}', 0, 1)
            if patient.age:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 6, f'Age: {patient.age} years', 0, 1)
            pdf.ln(2)
        else:
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, 'Patient: Not specified', 0, 1)
            pdf.ln(2)
        
        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, f'Report Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', 0, 1)
        pdf.ln(8)
    
    def _add_patient_content(self, pdf, analysis_result):
        """Add patient-friendly content."""
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Analysis Summary', 0, 1)
        pdf.ln(5)
        
        # Wound area in patient-friendly terms
        wound_area = analysis_result.area_mm2
        if wound_area > 0:
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f'Your wound area is {wound_area:.1f} square millimeters.', 0, 1)
        
        # Healing trend in encouraging language
        healing_trend = analysis_result.healing_potential
        pdf.set_font('Arial', '', 12)
        if healing_trend == 'Good':
            pdf.cell(0, 8, 'Great news! Your wound is showing positive healing progress.', 0, 1)
        elif healing_trend == 'Fair':
            pdf.cell(0, 8, 'Your wound appears to be stable and healing normally.', 0, 1)
        elif healing_trend == 'Poor':
            pdf.cell(0, 8, 'We\'ve detected some changes in your wound that may need attention.', 0, 1)
        else:
            pdf.cell(0, 8, 'We\'ve completed the analysis of your wound.', 0, 1)
        
        # Confidence in simple terms
        confidence = analysis_result.confidence_score
        pdf.cell(0, 8, f'The analysis confidence is {confidence:.0f} percent.', 0, 1)
        
        pdf.ln(10)
        
        # Care instructions
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Care Instructions', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, '- Continue following your current treatment plan', 0, 1)
        pdf.cell(0, 8, '- Keep the wound clean and dry', 0, 1)
        pdf.cell(0, 8, '- Monitor for any changes or concerns', 0, 1)
        pdf.cell(0, 8, '- Contact your healthcare provider if you have questions', 0, 1)
    
    def _add_clinician_content(self, pdf, analysis_result):
        """Add clinician-focused content."""
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'CLINICAL ANALYSIS', 0, 1)
        pdf.ln(5)
        
        # Technical measurements
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f'Wound Area: {analysis_result.area_mm2:.2f} mm²', 0, 1)
        pdf.cell(0, 6, f'Segmentation Confidence: {analysis_result.confidence_score:.3f}', 0, 1)
        pdf.cell(0, 6, f'Healing Potential: {analysis_result.healing_potential}', 0, 1)
        
        if hasattr(analysis_result, 'severity_assessment'):
            pdf.cell(0, 6, f'Severity Assessment: {analysis_result.severity_assessment}', 0, 1)
        
        pdf.ln(8)
        
        # Clinical interpretation
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'CLINICAL INTERPRETATION', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        healing_potential = analysis_result.healing_potential
        if healing_potential == 'Good':
            pdf.cell(0, 6, 'Recommendation: Continue current treatment protocol.', 0, 1)
        elif healing_potential == 'Poor':
            pdf.cell(0, 6, 'Recommendation: Consider treatment adjustment or closer monitoring.', 0, 1)
        else:
            pdf.cell(0, 6, 'Recommendation: Maintain current treatment and monitor progress.', 0, 1)
        
        # Confidence assessment
        confidence = analysis_result.confidence_score
        if confidence < 0.7:
            pdf.cell(0, 6, 'Note: Low segmentation confidence - consider manual review.', 0, 1)
    
    def _add_patient_images(self, pdf, analysis_result):
        """Add patient-friendly images."""
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Visual Analysis', 0, 1)
        pdf.ln(5)
        
        try:
            # Add original image
            if hasattr(analysis_result, 'original_image') and analysis_result.original_image is not None:
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, 'Your Wound Image:', 0, 1)
                pdf.ln(2)
                
                # Convert numpy array to PIL Image, then to bytes
                import cv2
                from PIL import Image
                import io
                
                # Analysis pipeline already converts BGR to RGB, so use image as-is
                img_rgb = analysis_result.original_image
                
                # Resize image to fit in PDF (max width 150mm)
                pil_img = Image.fromarray(img_rgb)
                max_width = 150
                if pil_img.width > max_width:
                    ratio = max_width / pil_img.width
                    new_height = int(pil_img.height * ratio)
                    pil_img = pil_img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Save to bytes
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_data = img_buffer.getvalue()
                
                # Add image to PDF
                pdf.image(io.BytesIO(img_data), x=10, w=min(pil_img.width/4, 150))
                pdf.ln(pil_img.height/4 + 5)
            
            # Add overlay if available
            if hasattr(analysis_result, 'overlay') and analysis_result.overlay is not None:
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, 'Analysis Results:', 0, 1)
                pdf.ln(2)
                
                # Overlay should already be in RGB format
                overlay_rgb = analysis_result.overlay
                
                pil_overlay = Image.fromarray(overlay_rgb)
                if pil_overlay.width > max_width:
                    ratio = max_width / pil_overlay.width
                    new_height = int(pil_overlay.height * ratio)
                    pil_overlay = pil_overlay.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Save to bytes
                overlay_buffer = io.BytesIO()
                pil_overlay.save(overlay_buffer, format='PNG')
                overlay_data = overlay_buffer.getvalue()
                
                # Add overlay to PDF
                pdf.image(io.BytesIO(overlay_data), x=10, w=min(pil_overlay.width/4, 150))
                pdf.ln(pil_overlay.height/4 + 5)
                
        except Exception as e:
            logger.warning(f"Could not add images to patient report: {e}")
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, 'Images could not be included in this report.', 0, 1)
    
    def _add_clinician_images(self, pdf, analysis_result):
        """Add clinician-focused images."""
        pdf.ln(8)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'VISUAL ANALYSIS', 0, 1)
        pdf.ln(5)
        
        try:
            import cv2
            from PIL import Image
            import io
            
            # Add original image
            if hasattr(analysis_result, 'original_image') and analysis_result.original_image is not None:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, 'Original Image:', 0, 1)
                pdf.ln(2)
                
                # Analysis pipeline already converts BGR to RGB, so use image as-is
                img_rgb = analysis_result.original_image
                
                pil_img = Image.fromarray(img_rgb)
                max_width = 120
                if pil_img.width > max_width:
                    ratio = max_width / pil_img.width
                    new_height = int(pil_img.height * ratio)
                    pil_img = pil_img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Save to bytes and add to PDF
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_data = img_buffer.getvalue()
                pdf.image(io.BytesIO(img_data), x=10, w=min(pil_img.width/4, 120))
                pdf.ln(pil_img.height/4 + 3)
            
            # Add segmentation mask
            if hasattr(analysis_result, 'mask') and analysis_result.mask is not None:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, 'Segmentation Mask:', 0, 1)
                pdf.ln(2)
                
                # Convert mask to 3-channel for display
                if len(analysis_result.mask.shape) == 2:
                    mask_3ch = cv2.cvtColor(analysis_result.mask, cv2.COLOR_GRAY2RGB)
                else:
                    mask_3ch = analysis_result.mask
                
                pil_mask = Image.fromarray(mask_3ch)
                if pil_mask.width > max_width:
                    ratio = max_width / pil_mask.width
                    new_height = int(pil_mask.height * ratio)
                    pil_mask = pil_mask.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                mask_buffer = io.BytesIO()
                pil_mask.save(mask_buffer, format='PNG')
                mask_data = mask_buffer.getvalue()
                pdf.image(io.BytesIO(mask_data), x=10, w=min(pil_mask.width/4, 120))
                pdf.ln(pil_mask.height/4 + 3)
            
            # Add overlay if available
            if hasattr(analysis_result, 'overlay') and analysis_result.overlay is not None:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, 'Analysis Overlay:', 0, 1)
                pdf.ln(2)
                
                # Overlay should already be in RGB format
                overlay_rgb = analysis_result.overlay
                
                pil_overlay = Image.fromarray(overlay_rgb)
                if pil_overlay.width > max_width:
                    ratio = max_width / pil_overlay.width
                    new_height = int(pil_overlay.height * ratio)
                    pil_overlay = pil_overlay.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                overlay_buffer = io.BytesIO()
                pil_overlay.save(overlay_buffer, format='PNG')
                overlay_data = overlay_buffer.getvalue()
                pdf.image(io.BytesIO(overlay_data), x=10, w=min(pil_overlay.width/4, 120))
                pdf.ln(pil_overlay.height/4 + 3)
                
        except Exception as e:
            logger.warning(f"Could not add images to clinician report: {e}")
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, 'Images could not be included in this report.', 0, 1)
    
    def _add_technical_appendix(self, pdf, analysis_result):
        """Add technical appendix for clinicians."""
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'TECHNICAL APPENDIX', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, f'Model Version: {getattr(analysis_result, "model_version", "Unknown")}', 0, 1)
        pdf.cell(0, 5, f'Processing Time: {getattr(analysis_result, "processing_time", "Unknown")}', 0, 1)
        pdf.cell(0, 5, f'Image Quality Score: {getattr(analysis_result, "image_quality", "Unknown")}', 0, 1)
        
        # Add validation results if available
        if hasattr(analysis_result, 'validation_results'):
            pdf.ln(3)
            pdf.cell(0, 5, 'Validation Results:', 0, 1)
            for result in analysis_result.validation_results:
                pdf.cell(0, 5, f'  - {result}', 0, 1)
    
    def _add_patient_footer(self, pdf):
        """Add patient-friendly footer."""
        pdf.ln(20)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 6, 'This report is for informational purposes only.', 0, 1)
        pdf.cell(0, 6, 'Please consult your healthcare provider for medical advice.', 0, 1)
        pdf.cell(0, 6, f'Generated by Wound Analysis System - {datetime.now().strftime("%Y-%m-%d")}', 0, 1)
    
    def _add_clinician_footer(self, pdf):
        """Add clinician-focused footer."""
        pdf.ln(15)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 5, 'This report contains automated analysis results.', 0, 1)
        pdf.cell(0, 5, 'Clinical judgment should always be used in conjunction with these results.', 0, 1)
        pdf.cell(0, 5, f'Report ID: {datetime.now().strftime("%Y%m%d_%H%M%S")}', 0, 1)
    
    def generate_comparison_report(self, analysis_results: List[AnalysisResult],
                                 patient: Optional[Patient] = None) -> Optional[Path]:
        """
        Generate comparison report for multiple analyses.
        
        Args:
            analysis_results: List of analysis results to compare
            patient: Patient information (optional)
            
        Returns:
            Path to saved PDF file, or None if generation failed
        """
        if not self.is_available() or len(analysis_results) < 2:
            logger.warning("PDF generation not available or insufficient data for comparison")
            return None
        
        try:
            # Create PDF
            pdf = self.FPDF()
            pdf.add_page()
            
            # Set up styling
            self._setup_clinician_styling(pdf)
            
            # Add header
            self._add_clinician_header(pdf, patient)
            
            # Add comparison content
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'PROGRESS COMPARISON REPORT', 0, 1, 'C')
            pdf.ln(10)
            
            # Add comparison table
            self._add_comparison_table(pdf, analysis_results)
            
            # Add trend analysis
            self._add_trend_analysis(pdf, analysis_results)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_report_{timestamp}.pdf"
            
            # Save PDF
            pdf_data = pdf.output(dest='S')
            if isinstance(pdf_data, bytearray):
                pdf_data = bytes(pdf_data)  # Convert bytearray to bytes
            elif isinstance(pdf_data, str):
                pdf_data = pdf_data.encode('latin1')
            elif isinstance(pdf_data, bytes):
                pass  # Already bytes
            else:
                pdf_data = str(pdf_data).encode('latin1')
            pdf_path = self.storage_service.save_pdf(pdf_data, 'reports', filename)
            
            logger.info(f"Comparison report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            return None
    
    def _add_comparison_table(self, pdf, analysis_results):
        """Add comparison table to PDF."""
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Analysis Comparison', 0, 1)
        pdf.ln(5)
        
        # Table headers
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(40, 8, 'Date', 1, 0, 'C')
        pdf.cell(30, 8, 'Area (mm²)', 1, 0, 'C')
        pdf.cell(30, 8, 'Confidence', 1, 0, 'C')
        pdf.cell(30, 8, 'Trend', 1, 0, 'C')
        pdf.cell(30, 8, 'Change %', 1, 1, 'C')
        
        # Table data
        pdf.set_font('Arial', '', 9)
        baseline_area = analysis_results[0].wound_area_mm2
        
        for i, result in enumerate(analysis_results):
            date_str = getattr(result, 'timestamp', f'Analysis {i+1}')
            area = result.wound_area_mm2
            confidence = result.segmentation_confidence
            trend = result.healing_trend
            
            if i == 0:
                change = 0
            else:
                change = ((area - baseline_area) / baseline_area) * 100 if baseline_area > 0 else 0
            
            pdf.cell(40, 6, str(date_str), 1, 0, 'C')
            pdf.cell(30, 6, f'{area:.1f}', 1, 0, 'C')
            pdf.cell(30, 6, f'{confidence:.3f}', 1, 0, 'C')
            pdf.cell(30, 6, trend, 1, 0, 'C')
            pdf.cell(30, 6, f'{change:+.1f}%', 1, 1, 'C')
    
    def _add_trend_analysis(self, pdf, analysis_results):
        """Add trend analysis to PDF."""
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Trend Analysis', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        
        # Calculate trends
        areas = [result.wound_area_mm2 for result in analysis_results]
        confidences = [result.segmentation_confidence for result in analysis_results]
        
        if len(areas) > 1:
            area_trend = "Decreasing" if areas[-1] < areas[0] else "Increasing" if areas[-1] > areas[0] else "Stable"
            confidence_trend = "Improving" if confidences[-1] > confidences[0] else "Declining" if confidences[-1] < confidences[0] else "Stable"
            
            pdf.cell(0, 6, f'Area Trend: {area_trend} ({areas[0]:.1f} → {areas[-1]:.1f} mm²)', 0, 1)
            pdf.cell(0, 6, f'Confidence Trend: {confidence_trend} ({confidences[0]:.3f} → {confidences[-1]:.3f})', 0, 1)
            
            # Calculate percentage change
            total_change = ((areas[-1] - areas[0]) / areas[0]) * 100 if areas[0] > 0 else 0
            pdf.cell(0, 6, f'Total Area Change: {total_change:+.1f}%', 0, 1)
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the reporting service.
        
        Returns:
            dict: Service information
        """
        return {
            'available': self.is_available(),
            'fpdf_available': self.pdf_available,
            'voice_service_available': self.voice_service.is_available(),
            'storage_service': str(self.storage_service.base_dir)
        }


# Global reporting service instance
_reporting_service: Optional[ReportingService] = None


def get_reporting_service() -> ReportingService:
    """
    Get the global reporting service instance.
    
    Returns:
        ReportingService: The global reporting service instance
    """
    global _reporting_service
    if _reporting_service is None:
        _reporting_service = ReportingService()
    return _reporting_service


def reset_reporting_service() -> None:
    """Reset the global reporting service (useful for testing)."""
    global _reporting_service
    _reporting_service = None
    logger.info("Reporting service reset")