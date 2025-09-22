"""
Unit tests for service layer.

Tests all service modules including reporting, validation, storage,
voice, and explainability services.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json

from woundseg.services import (
    get_storage_service,
    get_voice_service,
    get_reporting_service,
    get_validation_service,
    get_explainability_service
)
from woundseg.services.storage import StorageService
from woundseg.services.voice import VoiceService
from woundseg.services.explain import ExplainabilityService
from woundseg.types import Patient, AnalysisOptions, AnalysisResult
from woundseg.utils.exceptions import ServiceError


class TestServiceFactory:
    """Test cases for service factory functions."""
    
    def test_get_storage_service(self):
        """Test storage service factory."""
        service = get_storage_service()
        assert isinstance(service, StorageService)
    
    def test_get_voice_service(self):
        """Test voice service factory."""
        service = get_voice_service()
        assert service is not None
    
    def test_get_reporting_service(self):
        """Test reporting service factory."""
        service = get_reporting_service()
        assert service is not None
    
    def test_get_validation_service(self):
        """Test validation service factory."""
        service = get_validation_service()
        assert service is not None
    
    def test_get_explainability_service(self):
        """Test explainability service factory."""
        service = get_explainability_service()
        assert service is not None


class TestStorageService:
    """Test cases for StorageService."""
    
    def test_init(self):
        """Test StorageService initialization."""
        service = StorageService()
        assert service is not None
    
    def test_save_image(self, sample_image: np.ndarray, temp_dir: Path):
        """Test image saving."""
        service = StorageService()
        
        image_path = service.save_image(sample_image, temp_dir / "test_image.jpg")
        
        assert isinstance(image_path, Path)
        assert image_path.exists()
        assert image_path.suffix == '.jpg'
    
    def test_save_mask(self, test_utils, temp_dir: Path):
        """Test mask saving."""
        service = StorageService()
        
        mask = test_utils.create_test_mask(128, 128)
        mask_path = service.save_mask(mask, temp_dir / "test_mask.png")
        
        assert isinstance(mask_path, Path)
        assert mask_path.exists()
        assert mask_path.suffix == '.png'
    
    def test_save_metadata(self, temp_dir: Path):
        """Test metadata saving."""
        service = StorageService()
        
        metadata = {
            "patient_id": "TEST001",
            "analysis_date": "2025-09-22",
            "confidence_score": 0.85
        }
        
        metadata_path = service.save_metadata(metadata, temp_dir / "metadata.json")
        
        assert isinstance(metadata_path, Path)
        assert metadata_path.exists()
        assert metadata_path.suffix == '.json'
        
        # Verify content
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata
    
    def test_load_metadata(self, temp_dir: Path):
        """Test metadata loading."""
        service = StorageService()
        
        metadata = {
            "patient_id": "TEST002",
            "analysis_date": "2025-09-22",
            "confidence_score": 0.92
        }
        
        # Save metadata first
        metadata_path = temp_dir / "test_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Load metadata
        loaded_metadata = service.load_metadata(metadata_path)
        
        assert loaded_metadata == metadata
    
    def test_create_session_directory(self, temp_dir: Path):
        """Test session directory creation."""
        service = StorageService()
        
        session_dir = service.create_session_directory(temp_dir)
        
        assert isinstance(session_dir, Path)
        assert session_dir.exists()
        assert session_dir.is_dir()
        assert session_dir.parent == temp_dir
    
    def test_organize_outputs(self, temp_dir: Path, test_utils):
        """Test output organization."""
        service = StorageService()
        
        # Create test files
        image_path = temp_dir / "image.jpg"
        mask_path = temp_dir / "mask.png"
        report_path = temp_dir / "report.pdf"
        
        test_utils.save_test_image(test_utils.create_test_image(128, 128, 3), image_path)
        test_utils.save_test_image(test_utils.create_test_mask(128, 128), mask_path)
        report_path.write_text("dummy report content")
        
        # Organize outputs
        organized_paths = service.organize_outputs(
            temp_dir,
            {
                "image": image_path,
                "mask": mask_path,
                "report": report_path
            }
        )
        
        assert isinstance(organized_paths, dict)
        assert "image" in organized_paths
        assert "mask" in organized_paths
        assert "report" in organized_paths
    
    def test_cleanup_old_files(self, temp_dir: Path):
        """Test cleanup of old files."""
        service = StorageService()
        
        # Create old file
        old_file = temp_dir / "old_file.txt"
        old_file.write_text("old content")
        
        # Modify timestamp to make it old
        import time
        old_timestamp = time.time() - 86400  # 1 day ago
        old_file.touch()
        os.utime(old_file, (old_timestamp, old_timestamp))
        
        # Cleanup files older than 1 hour
        service.cleanup_old_files(temp_dir, max_age_hours=1)
        
        # File should be deleted
        assert not old_file.exists()
    
    def test_get_storage_stats(self, temp_dir: Path):
        """Test storage statistics."""
        service = StorageService()
        
        # Create some test files
        for i in range(3):
            test_file = temp_dir / f"test_file_{i}.txt"
            test_file.write_text(f"test content {i}")
        
        stats = service.get_storage_stats(temp_dir)
        
        assert isinstance(stats, dict)
        assert "total_files" in stats
        assert "total_size" in stats
        assert "file_types" in stats
        assert stats["total_files"] == 3


class TestVoiceService:
    """Test cases for VoiceService."""
    
    def test_init(self):
        """Test VoiceService initialization."""
        service = VoiceService()
        assert service is not None
    
    @patch('woundseg.services.voice.Config.ENABLE_VOICE_SUMMARY', True)
    def test_generate_voice_summary_enabled(self, mock_analysis_result: AnalysisResult, temp_dir: Path):
        """Test voice summary generation when enabled."""
        service = VoiceService()
        
        with patch('woundseg.services.voice.gTTS') as mock_gtts:
            mock_tts = Mock()
            mock_gtts.return_value = mock_tts
            
            audio_path = service.generate_voice_summary(
                mock_analysis_result,
                output_dir=temp_dir
            )
            
            assert isinstance(audio_path, Path)
            assert audio_path.suffix == '.mp3'
            mock_gtts.assert_called_once()
            mock_tts.save.assert_called_once()
    
    @patch('woundseg.services.voice.Config.ENABLE_VOICE_SUMMARY', False)
    def test_generate_voice_summary_disabled(self, mock_analysis_result: AnalysisResult, temp_dir: Path):
        """Test voice summary generation when disabled."""
        service = VoiceService()
        
        audio_path = service.generate_voice_summary(
            mock_analysis_result,
            output_dir=temp_dir
        )
        
        assert audio_path is None
    
    def test_generate_summary_text(self, mock_analysis_result: AnalysisResult):
        """Test summary text generation."""
        service = VoiceService()
        
        summary_text = service.generate_summary_text(mock_analysis_result)
        
        assert isinstance(summary_text, str)
        assert len(summary_text) > 0
        assert mock_analysis_result.patient.name in summary_text
    
    def test_generate_summary_text_with_confidence(self, mock_analysis_result: AnalysisResult):
        """Test summary text generation with confidence score."""
        service = VoiceService()
        
        # Set high confidence
        mock_analysis_result.confidence_score = 0.95
        
        summary_text = service.generate_summary_text(mock_analysis_result)
        
        assert "high confidence" in summary_text.lower() or "95%" in summary_text
    
    def test_generate_summary_text_with_low_confidence(self, mock_analysis_result: AnalysisResult):
        """Test summary text generation with low confidence."""
        service = VoiceService()
        
        # Set low confidence
        mock_analysis_result.confidence_score = 0.45
        
        summary_text = service.generate_summary_text(mock_analysis_result)
        
        assert "low confidence" in summary_text.lower() or "45%" in summary_text
    
    def test_voice_service_error_handling(self, mock_analysis_result: AnalysisResult, temp_dir: Path):
        """Test voice service error handling."""
        service = VoiceService()
        
        with patch('woundseg.services.voice.Config.ENABLE_VOICE_SUMMARY', True):
            with patch('woundseg.services.voice.gTTS') as mock_gtts:
                mock_gtts.side_effect = Exception("TTS service error")
                
                with pytest.raises(ServiceError):
                    service.generate_voice_summary(mock_analysis_result, temp_dir)


class TestExplainabilityService:
    """Test cases for ExplainabilityService."""
    
    def test_init(self):
        """Test ExplainabilityService initialization."""
        service = ExplainabilityService()
        assert service is not None
    
    @patch('woundseg.services.explain.Config.ENABLE_EXPLAINABILITY', True)
    def test_generate_gradcam_enabled(self, sample_image: np.ndarray, test_utils, temp_dir: Path):
        """Test Grad-CAM generation when enabled."""
        service = ExplainabilityService()
        
        with patch.object(service, '_generate_gradcam') as mock_gradcam:
            mock_gradcam.return_value = test_utils.create_test_image(128, 128, 3)
            
            gradcam_path = service.generate_gradcam(
                sample_image,
                output_dir=temp_dir
            )
            
            assert isinstance(gradcam_path, Path)
            assert gradcam_path.suffix == '.png'
            mock_gradcam.assert_called_once()
    
    @patch('woundseg.services.explain.Config.ENABLE_EXPLAINABILITY', False)
    def test_generate_gradcam_disabled(self, sample_image: np.ndarray, temp_dir: Path):
        """Test Grad-CAM generation when disabled."""
        service = ExplainabilityService()
        
        gradcam_path = service.generate_gradcam(
            sample_image,
            output_dir=temp_dir
        )
        
        assert gradcam_path is None
    
    @patch('woundseg.services.explain.Config.ENABLE_EXPLAINABILITY', True)
    def test_generate_shap_enabled(self, sample_image: np.ndarray, test_utils, temp_dir: Path):
        """Test SHAP generation when enabled."""
        service = ExplainabilityService()
        
        with patch.object(service, '_generate_shap') as mock_shap:
            mock_shap.return_value = test_utils.create_test_image(128, 128, 3)
            
            shap_path = service.generate_shap(
                sample_image,
                output_dir=temp_dir
            )
            
            assert isinstance(shap_path, Path)
            assert shap_path.suffix == '.png'
            mock_shap.assert_called_once()
    
    @patch('woundseg.services.explain.Config.ENABLE_EXPLAINABILITY', False)
    def test_generate_shap_disabled(self, sample_image: np.ndarray, temp_dir: Path):
        """Test SHAP generation when disabled."""
        service = ExplainabilityService()
        
        shap_path = service.generate_shap(
            sample_image,
            output_dir=temp_dir
        )
        
        assert shap_path is None
    
    def test_generate_explanations(self, sample_image: np.ndarray, test_utils, temp_dir: Path):
        """Test complete explanations generation."""
        service = ExplainabilityService()
        
        with patch.object(service, 'generate_gradcam') as mock_gradcam:
            with patch.object(service, 'generate_shap') as mock_shap:
                mock_gradcam.return_value = temp_dir / "gradcam.png"
                mock_shap.return_value = temp_dir / "shap.png"
                
                explanations = service.generate_explanations(
                    sample_image,
                    output_dir=temp_dir
                )
                
                assert isinstance(explanations, dict)
                assert "gradcam" in explanations
                assert "shap" in explanations
    
    def test_explainability_service_error_handling(self, sample_image: np.ndarray, temp_dir: Path):
        """Test explainability service error handling."""
        service = ExplainabilityService()
        
        with patch('woundseg.services.explain.Config.ENABLE_EXPLAINABILITY', True):
            with patch.object(service, '_generate_gradcam') as mock_gradcam:
                mock_gradcam.side_effect = Exception("Grad-CAM error")
                
                with pytest.raises(ServiceError):
                    service.generate_gradcam(sample_image, temp_dir)


class TestServiceIntegration:
    """Integration tests for service layer."""
    
    def test_services_work_together(self, sample_image: np.ndarray, test_utils, temp_dir: Path):
        """Test that services work together."""
        # Get all services
        storage_service = get_storage_service()
        voice_service = get_voice_service()
        reporting_service = get_reporting_service()
        validation_service = get_validation_service()
        explainability_service = get_explainability_service()
        
        # Create analysis result
        patient = Patient(
            name="Integration Test Patient",
            age=35,
            gender="F",
            patient_id="INT001",
            date_of_birth="1990-01-01"
        )
        
        # Save artifacts using storage service
        image_path = storage_service.save_image(sample_image, temp_dir / "wound_image.jpg")
        mask = test_utils.create_test_mask(128, 128)
        mask_path = storage_service.save_mask(mask, temp_dir / "wound_mask.png")
        
        analysis_result = AnalysisResult(
            patient=patient,
            options=AnalysisOptions(),
            mask_path=str(mask_path),
            confidence_score=0.88,
            processing_time=2.0,
            metadata={"test": "integration"},
            artifacts={
                "image": str(image_path),
                "mask": str(mask_path)
            }
        )
        
        # Use all services
        with patch('woundseg.services.voice.Config.ENABLE_VOICE_SUMMARY', True):
            with patch('woundseg.services.voice.gTTS') as mock_gtts:
                mock_tts = Mock()
                mock_gtts.return_value = mock_tts
                
                # Generate voice summary
                voice_path = voice_service.generate_voice_summary(analysis_result, temp_dir)
                
                # Generate report
                report_path = reporting_service.generate_patient_report(analysis_result, temp_dir)
                
                # Validate
                validation_result = validation_service.validate_pipeline(sample_image, mask)
                
                # Generate explanations
                with patch('woundseg.services.explain.Config.ENABLE_EXPLAINABILITY', True):
                    with patch.object(explainability_service, '_generate_gradcam') as mock_gradcam:
                        mock_gradcam.return_value = test_utils.create_test_image(128, 128, 3)
                        
                        explanations = explainability_service.generate_explanations(sample_image, temp_dir)
        
        # Verify all services worked
        assert voice_path is not None
        assert report_path.exists()
        assert validation_result['input_validation']['is_valid'] is True
        assert explanations is not None
    
    def test_service_error_propagation(self, sample_image: np.ndarray, temp_dir: Path):
        """Test that service errors are properly propagated."""
        storage_service = get_storage_service()
        
        # Test with invalid output directory
        invalid_dir = Path("/invalid/nonexistent/directory")
        
        with pytest.raises(ServiceError):
            storage_service.save_image(sample_image, invalid_dir / "test.jpg")
    
    def test_service_performance(self, sample_image: np.ndarray, test_utils, temp_dir: Path):
        """Test service performance."""
        import time
        
        storage_service = get_storage_service()
        
        # Test storage service performance
        start_time = time.time()
        
        for i in range(10):
            image_path = storage_service.save_image(sample_image, temp_dir / f"test_{i}.jpg")
            mask = test_utils.create_test_mask(128, 128)
            mask_path = storage_service.save_mask(mask, temp_dir / f"mask_{i}.png")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should be reasonably fast (less than 5 seconds for 10 files)
        assert processing_time < 5.0


@pytest.mark.golden
class TestServiceConsistency:
    """Golden tests for service consistency."""
    
    def test_service_initialization_consistency(self):
        """Test that services are initialized consistently."""
        # Get services multiple times
        services1 = {
            'storage': get_storage_service(),
            'voice': get_voice_service(),
            'reporting': get_reporting_service(),
            'validation': get_validation_service(),
            'explainability': get_explainability_service()
        }
        
        services2 = {
            'storage': get_storage_service(),
            'voice': get_voice_service(),
            'reporting': get_reporting_service(),
            'validation': get_validation_service(),
            'explainability': get_explainability_service()
        }
        
        # Services should be the same instances (singletons)
        for service_name in services1:
            assert services1[service_name] is services2[service_name]
    
    def test_service_output_consistency(self, sample_image: np.ndarray, test_utils, temp_dir: Path):
        """Test that services produce consistent outputs."""
        storage_service = get_storage_service()
        
        # Save the same image multiple times
        paths = []
        for i in range(3):
            path = storage_service.save_image(sample_image, temp_dir / f"consistent_{i}.jpg")
            paths.append(path)
        
        # All files should have the same size
        file_sizes = [path.stat().st_size for path in paths]
        assert all(size == file_sizes[0] for size in file_sizes)