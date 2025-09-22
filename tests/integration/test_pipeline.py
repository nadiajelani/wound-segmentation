"""
Integration tests for the complete analysis pipeline.

Tests the full workflow from image input to report generation,
including all pipeline components working together.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from woundseg.pipelines.analyze import AnalysisPipeline
from woundseg.types import Patient, AnalysisOptions
from woundseg.utils.exceptions import PipelineError


class TestAnalysisPipeline:
    """Test cases for the complete analysis pipeline."""
    
    def test_init(self):
        """Test AnalysisPipeline initialization."""
        pipeline = AnalysisPipeline()
        assert pipeline is not None
        assert pipeline.preprocessor is not None
        assert pipeline.segmenter is not None
        assert pipeline.postprocessor is not None
        assert pipeline.reporting_service is not None
        assert pipeline.validation_service is not None
    
    def test_init_with_custom_components(self):
        """Test AnalysisPipeline initialization with custom components."""
        mock_preprocessor = Mock()
        mock_segmenter = Mock()
        mock_postprocessor = Mock()
        mock_reporting = Mock()
        mock_validation = Mock()
        
        pipeline = AnalysisPipeline(
            preprocessor=mock_preprocessor,
            segmenter=mock_segmenter,
            postprocessor=mock_postprocessor,
            reporting_service=mock_reporting,
            validation_service=mock_validation
        )
        
        assert pipeline.preprocessor == mock_preprocessor
        assert pipeline.segmenter == mock_segmenter
        assert pipeline.postprocessor == mock_postprocessor
        assert pipeline.reporting_service == mock_reporting
        assert pipeline.validation_service == mock_validation
    
    @patch('woundseg.pipelines.analyze.ImagePreprocessor')
    @patch('woundseg.pipelines.analyze.SegmentationPipeline')
    @patch('woundseg.pipelines.analyze.PostprocessingPipeline')
    def test_analyze_image_success(self, mock_postprocessor_class, mock_segmenter_class, mock_preprocessor_class, sample_image: np.ndarray, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test successful image analysis."""
        # Mock all components
        mock_preprocessor = Mock()
        mock_preprocessor.preprocess_pipeline.return_value = np.random.rand(128, 128, 3)
        mock_preprocessor_class.return_value = mock_preprocessor
        
        mock_segmenter = Mock()
        mock_segmenter.segment_pipeline.return_value = {
            'mask': np.random.rand(128, 128, 1),
            'confidence': 0.85,
            'uncertainty': np.random.rand(128, 128)
        }
        mock_segmenter_class.return_value = mock_segmenter
        
        mock_postprocessor = Mock()
        mock_postprocessor.postprocess_pipeline.return_value = {
            'final_mask': np.random.randint(0, 255, (128, 128), dtype=np.uint8),
            'metrics': {'area': 1000, 'perimeter': 200}
        }
        mock_postprocessor_class.return_value = mock_postprocessor
        
        # Mock services
        with patch('woundseg.pipelines.analyze.get_reporting_service') as mock_reporting:
            with patch('woundseg.pipelines.analyze.get_validation_service') as mock_validation:
                mock_reporting.return_value = Mock()
                mock_validation.return_value = Mock()
                
                pipeline = AnalysisPipeline()
                
                result = pipeline.analyze_image(
                    image_path=temp_dir / "test_image.jpg",
                    patient=test_patient,
                    options=test_analysis_options,
                    output_dir=temp_dir
                )
                
                assert isinstance(result, dict)
                assert 'success' in result
                assert 'analysis_result' in result
                assert 'artifacts' in result
                assert result['success'] is True
    
    def test_analyze_image_invalid_input(self, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test image analysis with invalid input."""
        pipeline = AnalysisPipeline()
        
        # Test with non-existent image
        non_existent_path = temp_dir / "nonexistent_image.jpg"
        
        result = pipeline.analyze_image(
            image_path=non_existent_path,
            patient=test_patient,
            options=test_analysis_options,
            output_dir=temp_dir
        )
        
        assert isinstance(result, dict)
        assert result['success'] is False
        assert 'error' in result
    
    def test_analyze_image_validation_failure(self, sample_image: np.ndarray, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test image analysis with validation failure."""
        pipeline = AnalysisPipeline()
        
        # Mock validation service to return failure
        with patch.object(pipeline.validation_service, 'validate_input_image') as mock_validate:
            mock_validate.return_value = {'is_valid': False, 'issues': ['poor_quality']}
            
            # Save sample image
            from PIL import Image
            image_path = temp_dir / "test_image.jpg"
            Image.fromarray(sample_image).save(image_path)
            
            result = pipeline.analyze_image(
                image_path=image_path,
                patient=test_patient,
                options=test_analysis_options,
                output_dir=temp_dir
            )
            
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'validation_error' in result
    
    def test_analyze_image_segmentation_failure(self, sample_image: np.ndarray, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test image analysis with segmentation failure."""
        pipeline = AnalysisPipeline()
        
        # Mock segmentation to fail
        with patch.object(pipeline.segmenter, 'segment_pipeline') as mock_segment:
            mock_segment.side_effect = Exception("Segmentation failed")
            
            # Save sample image
            from PIL import Image
            image_path = temp_dir / "test_image.jpg"
            Image.fromarray(sample_image).save(image_path)
            
            result = pipeline.analyze_image(
                image_path=image_path,
                patient=test_patient,
                options=test_analysis_options,
                output_dir=temp_dir
            )
            
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'segmentation_error' in result
    
    def test_analyze_image_reporting_failure(self, sample_image: np.ndarray, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test image analysis with reporting failure."""
        pipeline = AnalysisPipeline()
        
        # Mock reporting to fail
        with patch.object(pipeline.reporting_service, 'generate_patient_report') as mock_report:
            mock_report.side_effect = Exception("Reporting failed")
            
            # Save sample image
            from PIL import Image
            image_path = temp_dir / "test_image.jpg"
            Image.fromarray(sample_image).save(image_path)
            
            result = pipeline.analyze_image(
                image_path=image_path,
                patient=test_patient,
                options=test_analysis_options,
                output_dir=temp_dir
            )
            
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'reporting_error' in result
    
    def test_analyze_batch(self, test_utils, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test batch image analysis."""
        pipeline = AnalysisPipeline()
        
        # Create multiple test images
        image_paths = []
        for i in range(3):
            test_image = test_utils.create_test_image(128, 128, 3)
            image_path = temp_dir / f"test_image_{i}.jpg"
            test_utils.save_test_image(test_image, image_path)
            image_paths.append(image_path)
        
        # Mock all components for successful processing
        with patch.object(pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess:
            with patch.object(pipeline.segmenter, 'segment_pipeline') as mock_segment:
                with patch.object(pipeline.postprocessor, 'postprocess_pipeline') as mock_postprocess:
                    with patch.object(pipeline.reporting_service, 'generate_patient_report') as mock_report:
                        with patch.object(pipeline.validation_service, 'validate_input_image') as mock_validate:
                            
                            mock_preprocess.return_value = np.random.rand(128, 128, 3)
                            mock_segment.return_value = {
                                'mask': np.random.rand(128, 128, 1),
                                'confidence': 0.85,
                                'uncertainty': np.random.rand(128, 128)
                            }
                            mock_postprocess.return_value = {
                                'final_mask': np.random.randint(0, 255, (128, 128), dtype=np.uint8),
                                'metrics': {'area': 1000, 'perimeter': 200}
                            }
                            mock_report.return_value = temp_dir / "report.pdf"
                            mock_validate.return_value = {'is_valid': True, 'quality_score': 0.9}
                            
                            results = pipeline.analyze_batch(
                                image_paths=image_paths,
                                patient=test_patient,
                                options=test_analysis_options,
                                output_dir=temp_dir
                            )
                            
                            assert isinstance(results, list)
                            assert len(results) == 3
                            for result in results:
                                assert isinstance(result, dict)
                                assert 'success' in result
    
    def test_analyze_batch_with_failures(self, test_utils, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test batch analysis with some failures."""
        pipeline = AnalysisPipeline()
        
        # Create test images
        image_paths = []
        for i in range(3):
            test_image = test_utils.create_test_image(128, 128, 3)
            image_path = temp_dir / f"test_image_{i}.jpg"
            test_utils.save_test_image(test_image, image_path)
            image_paths.append(image_path)
        
        # Mock validation to fail for one image
        with patch.object(pipeline.validation_service, 'validate_input_image') as mock_validate:
            mock_validate.side_effect = [
                {'is_valid': True, 'quality_score': 0.9},  # First image OK
                {'is_valid': False, 'issues': ['poor_quality']},  # Second image fails
                {'is_valid': True, 'quality_score': 0.8}   # Third image OK
            ]
            
            results = pipeline.analyze_batch(
                image_paths=image_paths,
                patient=test_patient,
                options=test_analysis_options,
                output_dir=temp_dir
            )
            
            assert isinstance(results, list)
            assert len(results) == 3
            assert results[0]['success'] is True
            assert results[1]['success'] is False
            assert results[2]['success'] is True
    
    def test_pipeline_performance(self, sample_image: np.ndarray, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test pipeline performance."""
        import time
        
        pipeline = AnalysisPipeline()
        
        # Save sample image
        from PIL import Image
        image_path = temp_dir / "test_image.jpg"
        Image.fromarray(sample_image).save(image_path)
        
        # Mock all components for fast processing
        with patch.object(pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess:
            with patch.object(pipeline.segmenter, 'segment_pipeline') as mock_segment:
                with patch.object(pipeline.postprocessor, 'postprocess_pipeline') as mock_postprocess:
                    with patch.object(pipeline.reporting_service, 'generate_patient_report') as mock_report:
                        with patch.object(pipeline.validation_service, 'validate_input_image') as mock_validate:
                            
                            mock_preprocess.return_value = np.random.rand(128, 128, 3)
                            mock_segment.return_value = {
                                'mask': np.random.rand(128, 128, 1),
                                'confidence': 0.85,
                                'uncertainty': np.random.rand(128, 128)
                            }
                            mock_postprocess.return_value = {
                                'final_mask': np.random.randint(0, 255, (128, 128), dtype=np.uint8),
                                'metrics': {'area': 1000, 'perimeter': 200}
                            }
                            mock_report.return_value = temp_dir / "report.pdf"
                            mock_validate.return_value = {'is_valid': True, 'quality_score': 0.9}
                            
                            start_time = time.time()
                            result = pipeline.analyze_image(
                                image_path=image_path,
                                patient=test_patient,
                                options=test_analysis_options,
                                output_dir=temp_dir
                            )
                            end_time = time.time()
                            
                            processing_time = end_time - start_time
                            
                            # Should be reasonably fast (less than 5 seconds)
                            assert processing_time < 5.0
                            assert result['success'] is True
    
    def test_pipeline_error_handling(self, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test pipeline error handling."""
        pipeline = AnalysisPipeline()
        
        # Test with invalid image path
        invalid_path = Path("/invalid/path/image.jpg")
        
        result = pipeline.analyze_image(
            image_path=invalid_path,
            patient=test_patient,
            options=test_analysis_options,
            output_dir=temp_dir
        )
        
        assert isinstance(result, dict)
        assert result['success'] is False
        assert 'error' in result
    
    def test_pipeline_with_different_options(self, sample_image: np.ndarray, test_patient: Patient, temp_dir: Path):
        """Test pipeline with different analysis options."""
        pipeline = AnalysisPipeline()
        
        # Save sample image
        from PIL import Image
        image_path = temp_dir / "test_image.jpg"
        Image.fromarray(sample_image).save(image_path)
        
        # Test with different options
        options_variants = [
            AnalysisOptions(confidence_threshold=0.3, generate_report=True),
            AnalysisOptions(confidence_threshold=0.7, generate_report=False),
            AnalysisOptions(confidence_threshold=0.9, generate_voice_summary=True)
        ]
        
        for options in options_variants:
            # Mock all components
            with patch.object(pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess:
                with patch.object(pipeline.segmenter, 'segment_pipeline') as mock_segment:
                    with patch.object(pipeline.postprocessor, 'postprocess_pipeline') as mock_postprocess:
                        with patch.object(pipeline.reporting_service, 'generate_patient_report') as mock_report:
                            with patch.object(pipeline.validation_service, 'validate_input_image') as mock_validate:
                                
                                mock_preprocess.return_value = np.random.rand(128, 128, 3)
                                mock_segment.return_value = {
                                    'mask': np.random.rand(128, 128, 1),
                                    'confidence': 0.85,
                                    'uncertainty': np.random.rand(128, 128)
                                }
                                mock_postprocess.return_value = {
                                    'final_mask': np.random.randint(0, 255, (128, 128), dtype=np.uint8),
                                    'metrics': {'area': 1000, 'perimeter': 200}
                                }
                                mock_report.return_value = temp_dir / "report.pdf"
                                mock_validate.return_value = {'is_valid': True, 'quality_score': 0.9}
                                
                                result = pipeline.analyze_image(
                                    image_path=image_path,
                                    patient=test_patient,
                                    options=options,
                                    output_dir=temp_dir
                                )
                                
                                assert isinstance(result, dict)
                                assert result['success'] is True


@pytest.mark.golden
class TestPipelineConsistency:
    """Golden tests for pipeline consistency."""
    
    def test_pipeline_deterministic(self, sample_image: np.ndarray, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test that pipeline produces deterministic results."""
        pipeline = AnalysisPipeline()
        
        # Save sample image
        from PIL import Image
        image_path = temp_dir / "test_image.jpg"
        Image.fromarray(sample_image).save(image_path)
        
        # Mock all components with fixed random seeds
        with patch.object(pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess:
            with patch.object(pipeline.segmenter, 'segment_pipeline') as mock_segment:
                with patch.object(pipeline.postprocessor, 'postprocess_pipeline') as mock_postprocess:
                    with patch.object(pipeline.reporting_service, 'generate_patient_report') as mock_report:
                        with patch.object(pipeline.validation_service, 'validate_input_image') as mock_validate:
                            
                            # Use fixed random seed for consistent results
                            np.random.seed(42)
                            mock_preprocess.return_value = np.random.rand(128, 128, 3)
                            mock_segment.return_value = {
                                'mask': np.random.rand(128, 128, 1),
                                'confidence': 0.85,
                                'uncertainty': np.random.rand(128, 128)
                            }
                            mock_postprocess.return_value = {
                                'final_mask': np.random.randint(0, 255, (128, 128), dtype=np.uint8),
                                'metrics': {'area': 1000, 'perimeter': 200}
                            }
                            mock_report.return_value = temp_dir / "report.pdf"
                            mock_validate.return_value = {'is_valid': True, 'quality_score': 0.9}
                            
                            # Run pipeline multiple times
                            results = []
                            for i in range(3):
                                result = pipeline.analyze_image(
                                    image_path=image_path,
                                    patient=test_patient,
                                    options=test_analysis_options,
                                    output_dir=temp_dir / f"run_{i}"
                                )
                                results.append(result)
                            
                            # All results should be identical
                            for i in range(1, len(results)):
                                assert results[0]['success'] == results[i]['success']
                                if results[0]['success']:
                                    assert results[0]['analysis_result'].confidence_score == results[i]['analysis_result'].confidence_score
    
    def test_pipeline_consistency_across_runs(self, sample_image: np.ndarray, test_patient: Patient, test_analysis_options: AnalysisOptions, temp_dir: Path):
        """Test pipeline consistency across multiple runs."""
        pipeline = AnalysisPipeline()
        
        # Save sample image
        from PIL import Image
        image_path = temp_dir / "test_image.jpg"
        Image.fromarray(sample_image).save(image_path)
        
        # Mock all components
        with patch.object(pipeline.preprocessor, 'preprocess_pipeline') as mock_preprocess:
            with patch.object(pipeline.segmenter, 'segment_pipeline') as mock_segment:
                with patch.object(pipeline.postprocessor, 'postprocess_pipeline') as mock_postprocess:
                    with patch.object(pipeline.reporting_service, 'generate_patient_report') as mock_report:
                        with patch.object(pipeline.validation_service, 'validate_input_image') as mock_validate:
                            
                            mock_preprocess.return_value = np.random.rand(128, 128, 3)
                            mock_segment.return_value = {
                                'mask': np.random.rand(128, 128, 1),
                                'confidence': 0.85,
                                'uncertainty': np.random.rand(128, 128)
                            }
                            mock_postprocess.return_value = {
                                'final_mask': np.random.randint(0, 255, (128, 128), dtype=np.uint8),
                                'metrics': {'area': 1000, 'perimeter': 200}
                            }
                            mock_report.return_value = temp_dir / "report.pdf"
                            mock_validate.return_value = {'is_valid': True, 'quality_score': 0.9}
                            
                            # Run pipeline multiple times
                            results = []
                            for i in range(5):
                                result = pipeline.analyze_image(
                                    image_path=image_path,
                                    patient=test_patient,
                                    options=test_analysis_options,
                                    output_dir=temp_dir / f"run_{i}"
                                )
                                results.append(result)
                            
                            # All runs should succeed
                            for result in results:
                                assert result['success'] is True
                            
                            # Processing times should be similar (within 50% variance)
                            processing_times = [result.get('processing_time', 0) for result in results if 'processing_time' in result]
                            if processing_times:
                                avg_time = sum(processing_times) / len(processing_times)
                                for time_val in processing_times:
                                    assert abs(time_val - avg_time) / avg_time < 0.5