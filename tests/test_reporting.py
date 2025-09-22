"""
Unit tests for reporting functions.

Tests PDF report generation, patient and clinician report variants,
and reporting service functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from woundseg.services.reporting import ReportingService
from woundseg.types import Patient, AnalysisOptions, AnalysisResult
from woundseg.utils.exceptions import ReportingError


class TestReportingService:
    """Test cases for ReportingService class."""
    
    def test_init(self):
        """Test ReportingService initialization."""
        service = ReportingService()
        assert service is not None
    
    def test_generate_patient_report(self, mock_analysis_result: AnalysisResult, test_output_dir: Path):
        """Test patient report generation."""
        service = ReportingService()
        
        report_path = service.generate_patient_report(
            mock_analysis_result,
            output_dir=test_output_dir
        )
        
        assert isinstance(report_path, Path)
        assert report_path.exists()
        assert report_path.suffix == '.pdf'
    
    def test_generate_clinician_report(self, mock_analysis_result: AnalysisResult, test_output_dir: Path):
        """Test clinician report generation."""
        service = ReportingService()
        
        report_path = service.generate_clinician_report(
            mock_analysis_result,
            output_dir=test_output_dir
        )
        
        assert isinstance(report_path, Path)
        assert report_path.exists()
        assert report_path.suffix == '.pdf'
    
    def test_generate_report_with_custom_template(self, mock_analysis_result: AnalysisResult, test_output_dir: Path):
        """Test report generation with custom template."""
        service = ReportingService()
        
        # Create a custom template
        custom_template = {
            "title": "Custom Wound Analysis Report",
            "sections": ["summary", "analysis", "recommendations"]
        }
        
        report_path = service.generate_report(
            mock_analysis_result,
            template=custom_template,
            output_dir=test_output_dir
        )
        
        assert isinstance(report_path, Path)
        assert report_path.exists()
    
    def test_generate_report_with_image(self, mock_analysis_result: AnalysisResult, sample_image: np.ndarray, test_output_dir: Path):
        """Test report generation with image."""
        service = ReportingService()
        
        # Save sample image
        image_path = test_output_dir / "wound_image.jpg"
        from PIL import Image
        Image.fromarray(sample_image).save(image_path)
        
        # Update analysis result with image path
        mock_analysis_result.artifacts["image"] = str(image_path)
        
        report_path = service.generate_patient_report(
            mock_analysis_result,
            output_dir=test_output_dir
        )
        
        assert isinstance(report_path, Path)
        assert report_path.exists()
    
    def test_generate_report_with_mask(self, mock_analysis_result: AnalysisResult, test_utils, test_output_dir: Path):
        """Test report generation with mask."""
        service = ReportingService()
        
        # Create and save mask
        mask = test_utils.create_test_mask(128, 128)
        mask_path = test_output_dir / "wound_mask.png"
        from PIL import Image
        Image.fromarray(mask).save(mask_path)
        
        # Update analysis result with mask path
        mock_analysis_result.artifacts["mask"] = str(mask_path)
        
        report_path = service.generate_patient_report(
            mock_analysis_result,
            output_dir=test_output_dir
        )
        
        assert isinstance(report_path, Path)
        assert report_path.exists()
    
    def test_generate_report_with_visualization(self, mock_analysis_result: AnalysisResult, test_utils, test_output_dir: Path):
        """Test report generation with visualization."""
        service = ReportingService()
        
        # Create and save visualization
        viz_image = test_utils.create_test_image(256, 256, 3)
        viz_path = test_output_dir / "visualization.png"
        from PIL import Image
        Image.fromarray(viz_image).save(viz_path)
        
        # Update analysis result with visualization path
        mock_analysis_result.artifacts["visualization"] = str(viz_path)
        
        report_path = service.generate_patient_report(
            mock_analysis_result,
            output_dir=test_output_dir
        )
        
        assert isinstance(report_path, Path)
        assert report_path.exists()
    
    def test_generate_report_invalid_output_dir(self, mock_analysis_result: AnalysisResult):
        """Test report generation with invalid output directory."""
        service = ReportingService()
        
        invalid_dir = Path("/invalid/nonexistent/directory")
        
        with pytest.raises(ReportingError):
            service.generate_patient_report(
                mock_analysis_result,
                output_dir=invalid_dir
            )
    
    def test_generate_report_missing_artifacts(self, test_patient: Patient, test_analysis_options: AnalysisOptions):
        """Test report generation with missing artifacts."""
        service = ReportingService()
        
        # Create analysis result with minimal data
        analysis_result = AnalysisResult(
            patient=test_patient,
            options=test_analysis_options,
            mask_path="",
            confidence_score=0.85,
            processing_time=1.5,
            metadata={},
            artifacts={}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = service.generate_patient_report(
                analysis_result,
                output_dir=Path(temp_dir)
            )
            
            assert isinstance(report_path, Path)
            assert report_path.exists()
    
    def test_generate_report_with_metadata(self, mock_analysis_result: AnalysisResult, test_output_dir: Path):
        """Test report generation with rich metadata."""
        service = ReportingService()
        
        # Add rich metadata
        mock_analysis_result.metadata.update({
            "model_version": "v1.2.3",
            "processing_time": 2.5,
            "image_dimensions": [512, 512, 3],
            "segmentation_confidence": 0.92,
            "quality_score": 0.88,
            "analysis_timestamp": "2025-09-22T10:00:00Z"
        })
        
        report_path = service.generate_patient_report(
            mock_analysis_result,
            output_dir=test_output_dir
        )
        
        assert isinstance(report_path, Path)
        assert report_path.exists()
    
    def test_generate_batch_reports(self, test_utils, test_output_dir: Path):
        """Test batch report generation."""
        service = ReportingService()
        
        # Create multiple analysis results
        analysis_results = []
        for i in range(3):
            patient = Patient(
                name=f"Patient {i}",
                age=30 + i,
                gender="M",
                patient_id=f"TEST{i:03d}",
                date_of_birth="1990-01-01"
            )
            
            analysis_result = AnalysisResult(
                patient=patient,
                options=AnalysisOptions(),
                mask_path=f"/test/mask_{i}.png",
                confidence_score=0.8 + i * 0.05,
                processing_time=1.0 + i * 0.5,
                metadata={"batch_id": i},
                artifacts={}
            )
            analysis_results.append(analysis_result)
        
        report_paths = service.generate_batch_reports(
            analysis_results,
            output_dir=test_output_dir
        )
        
        assert len(report_paths) == 3
        for report_path in report_paths:
            assert isinstance(report_path, Path)
            assert report_path.exists()


class TestReportTemplates:
    """Test cases for report templates."""
    
    def test_patient_template(self):
        """Test patient report template."""
        from woundseg.services.reporting import get_patient_template
        
        template = get_patient_template()
        
        assert isinstance(template, dict)
        assert 'title' in template
        assert 'sections' in template
        assert 'patient' in template['sections']
        assert 'analysis' in template['sections']
    
    def test_clinician_template(self):
        """Test clinician report template."""
        from woundseg.services.reporting import get_clinician_template
        
        template = get_clinician_template()
        
        assert isinstance(template, dict)
        assert 'title' in template
        assert 'sections' in template
        assert 'technical_details' in template['sections']
        assert 'recommendations' in template['sections']
    
    def test_custom_template(self):
        """Test custom template creation."""
        from woundseg.services.reporting import create_custom_template
        
        custom_template = create_custom_template(
            title="Custom Report",
            sections=["summary", "details", "conclusion"]
        )
        
        assert isinstance(custom_template, dict)
        assert custom_template['title'] == "Custom Report"
        assert custom_template['sections'] == ["summary", "details", "conclusion"]
    
    def test_template_validation(self):
        """Test template validation."""
        from woundseg.services.reporting import validate_template
        
        # Valid template
        valid_template = {
            "title": "Test Report",
            "sections": ["summary", "analysis"]
        }
        assert validate_template(valid_template) is True
        
        # Invalid template (missing title)
        invalid_template = {
            "sections": ["summary", "analysis"]
        }
        assert validate_template(invalid_template) is False
        
        # Invalid template (missing sections)
        invalid_template2 = {
            "title": "Test Report"
        }
        assert validate_template(invalid_template2) is False


class TestReportContent:
    """Test cases for report content generation."""
    
    def test_generate_summary_content(self, mock_analysis_result: AnalysisResult):
        """Test summary content generation."""
        from woundseg.services.reporting import generate_summary_content
        
        content = generate_summary_content(mock_analysis_result)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert mock_analysis_result.patient.name in content
    
    def test_generate_analysis_content(self, mock_analysis_result: AnalysisResult):
        """Test analysis content generation."""
        from woundseg.services.reporting import generate_analysis_content
        
        content = generate_analysis_content(mock_analysis_result)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert str(mock_analysis_result.confidence_score) in content
    
    def test_generate_recommendations_content(self, mock_analysis_result: AnalysisResult):
        """Test recommendations content generation."""
        from woundseg.services.reporting import generate_recommendations_content
        
        content = generate_recommendations_content(mock_analysis_result)
        
        assert isinstance(content, str)
        assert len(content) > 0
    
    def test_generate_technical_content(self, mock_analysis_result: AnalysisResult):
        """Test technical content generation."""
        from woundseg.services.reporting import generate_technical_content
        
        content = generate_technical_content(mock_analysis_result)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert str(mock_analysis_result.processing_time) in content
    
    def test_format_confidence_score(self):
        """Test confidence score formatting."""
        from woundseg.services.reporting import format_confidence_score
        
        # Test with different confidence scores
        assert format_confidence_score(0.95) == "95%"
        assert format_confidence_score(0.85) == "85%"
        assert format_confidence_score(0.5) == "50%"
        assert format_confidence_score(0.0) == "0%"
    
    def test_format_processing_time(self):
        """Test processing time formatting."""
        from woundseg.services.reporting import format_processing_time
        
        # Test with different processing times
        assert format_processing_time(1.5) == "1.5 seconds"
        assert format_processing_time(0.5) == "0.5 seconds"
        assert format_processing_time(60.0) == "1.0 minutes"
        assert format_processing_time(120.0) == "2.0 minutes"


class TestReportUtils:
    """Test cases for report utility functions."""
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        from woundseg.services.reporting import sanitize_filename
        
        # Test with various characters
        assert sanitize_filename("Patient Name") == "Patient_Name"
        assert sanitize_filename("Test/File:Name") == "Test_File_Name"
        assert sanitize_filename("File with spaces") == "File_with_spaces"
        assert sanitize_filename("File@with#special$chars") == "File_with_special_chars"
    
    def test_generate_report_filename(self, test_patient: Patient):
        """Test report filename generation."""
        from woundseg.services.reporting import generate_report_filename
        
        filename = generate_report_filename(test_patient, "patient")
        
        assert isinstance(filename, str)
        assert filename.endswith(".pdf")
        assert "Patient" in filename
        assert test_patient.name.replace(" ", "_") in filename
    
    def test_validate_report_data(self, mock_analysis_result: AnalysisResult):
        """Test report data validation."""
        from woundseg.services.reporting import validate_report_data
        
        # Valid data
        assert validate_report_data(mock_analysis_result) is True
        
        # Invalid data (missing patient)
        invalid_result = AnalysisResult(
            patient=None,
            options=AnalysisOptions(),
            mask_path="",
            confidence_score=0.85,
            processing_time=1.5,
            metadata={},
            artifacts={}
        )
        assert validate_report_data(invalid_result) is False
    
    def test_extract_report_metadata(self, mock_analysis_result: AnalysisResult):
        """Test report metadata extraction."""
        from woundseg.services.reporting import extract_report_metadata
        
        metadata = extract_report_metadata(mock_analysis_result)
        
        assert isinstance(metadata, dict)
        assert 'patient_name' in metadata
        assert 'confidence_score' in metadata
        assert 'processing_time' in metadata
        assert 'timestamp' in metadata


@pytest.mark.integration
class TestReportingIntegration:
    """Integration tests for reporting pipeline."""
    
    def test_complete_reporting_pipeline(self, sample_image: np.ndarray, test_utils, test_output_dir: Path):
        """Test complete reporting pipeline."""
        service = ReportingService()
        
        # Create analysis result with all artifacts
        patient = Patient(
            name="Integration Test Patient",
            age=45,
            gender="F",
            patient_id="INT001",
            date_of_birth="1980-01-01"
        )
        
        # Save artifacts
        image_path = test_output_dir / "wound_image.jpg"
        mask_path = test_output_dir / "wound_mask.png"
        viz_path = test_output_dir / "visualization.png"
        
        from PIL import Image
        Image.fromarray(sample_image).save(image_path)
        Image.fromarray(test_utils.create_test_mask(128, 128)).save(mask_path)
        Image.fromarray(test_utils.create_test_image(256, 256, 3)).save(viz_path)
        
        analysis_result = AnalysisResult(
            patient=patient,
            options=AnalysisOptions(),
            mask_path=str(mask_path),
            confidence_score=0.92,
            processing_time=2.5,
            metadata={
                "model_version": "v1.0.0",
                "image_dimensions": [512, 512, 3],
                "analysis_timestamp": "2025-09-22T10:00:00Z"
            },
            artifacts={
                "image": str(image_path),
                "mask": str(mask_path),
                "visualization": str(viz_path)
            }
        )
        
        # Generate both patient and clinician reports
        patient_report = service.generate_patient_report(analysis_result, test_output_dir)
        clinician_report = service.generate_clinician_report(analysis_result, test_output_dir)
        
        assert patient_report.exists()
        assert clinician_report.exists()
        assert patient_report != clinician_report  # Different reports
    
    def test_reporting_with_missing_artifacts(self, test_patient: Patient, test_output_dir: Path):
        """Test reporting with missing artifacts."""
        service = ReportingService()
        
        analysis_result = AnalysisResult(
            patient=test_patient,
            options=AnalysisOptions(),
            mask_path="",
            confidence_score=0.85,
            processing_time=1.5,
            metadata={},
            artifacts={}  # No artifacts
        )
        
        # Should still generate report
        report_path = service.generate_patient_report(analysis_result, test_output_dir)
        
        assert report_path.exists()
    
    def test_reporting_performance(self, mock_analysis_result: AnalysisResult, test_output_dir: Path):
        """Test reporting performance."""
        import time
        
        service = ReportingService()
        
        start_time = time.time()
        report_path = service.generate_patient_report(mock_analysis_result, test_output_dir)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should be reasonably fast (less than 5 seconds)
        assert processing_time < 5.0
        assert report_path.exists()
    
    def test_reporting_error_handling(self, test_patient: Patient):
        """Test reporting error handling."""
        service = ReportingService()
        
        # Test with invalid analysis result
        invalid_result = AnalysisResult(
            patient=test_patient,
            options=AnalysisOptions(),
            mask_path="",
            confidence_score=0.85,
            processing_time=1.5,
            metadata={},
            artifacts={}
        )
        
        # Test with invalid output directory
        with pytest.raises(ReportingError):
            service.generate_patient_report(
                invalid_result,
                output_dir=Path("/invalid/path")
            )


@pytest.mark.golden
class TestGoldenReportGeneration:
    """Golden tests for report generation consistency."""
    
    def test_report_consistency(self, mock_analysis_result: AnalysisResult, test_output_dir: Path):
        """Test that reports are generated consistently."""
        service = ReportingService()
        
        # Generate report multiple times
        report_paths = []
        for i in range(3):
            report_path = service.generate_patient_report(
                mock_analysis_result,
                output_dir=test_output_dir / f"test_{i}"
            )
            report_paths.append(report_path)
        
        # All reports should exist
        for report_path in report_paths:
            assert report_path.exists()
        
        # Reports should have similar file sizes (within 10% tolerance)
        file_sizes = [path.stat().st_size for path in report_paths]
        avg_size = sum(file_sizes) / len(file_sizes)
        
        for size in file_sizes:
            assert abs(size - avg_size) / avg_size < 0.1
    
    def test_report_deterministic(self, mock_analysis_result: AnalysisResult, test_output_dir: Path):
        """Test that report generation is deterministic."""
        service = ReportingService()
        
        # Set random seed
        import random
        random.seed(42)
        
        # Generate report
        report_path1 = service.generate_patient_report(mock_analysis_result, test_output_dir / "test1")
        
        # Reset seed and generate again
        random.seed(42)
        report_path2 = service.generate_patient_report(mock_analysis_result, test_output_dir / "test2")
        
        # Reports should be identical
        assert report_path1.stat().st_size == report_path2.stat().st_size