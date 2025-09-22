"""
Test configuration and fixtures for wound segmentation tests.

This module provides shared test fixtures, configuration, and utilities
for all test modules in the wound segmentation system.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import numpy as np
from PIL import Image
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ["ENABLE_SYNTHETIC_DATA"] = "false"
os.environ["ENABLE_VOICE_SUMMARY"] = "false"
os.environ["ENABLE_EXPLAINABILITY"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

from woundseg.config import Config
from woundseg.types import Patient, AnalysisOptions, AnalysisResult


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration settings."""
    return {
        "test_mode": True,
        "models_dir": str(project_root / "tests" / "fixtures" / "models"),
        "output_dir": str(project_root / "tests" / "fixtures" / "outputs"),
        "temp_dir": str(project_root / "tests" / "fixtures" / "temp"),
        "sample_images_dir": str(project_root / "tests" / "fixtures" / "sample_images"),
        "golden_images_dir": str(project_root / "tests" / "fixtures" / "golden_images"),
    }


@pytest.fixture(scope="session")
def temp_dir(test_config: Dict[str, Any]) -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    temp_path = Path(test_config["temp_dir"])
    temp_path.mkdir(parents=True, exist_ok=True)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_path(test_config: Dict[str, Any]) -> Path:
    """Provide path to a sample test image."""
    return Path(test_config["sample_images_dir"]) / "test_wound.jpg"


@pytest.fixture
def sample_image(sample_image_path: Path) -> np.ndarray:
    """Provide a sample wound image as numpy array."""
    # Create a simple test image if it doesn't exist
    if not sample_image_path.exists():
        sample_image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a simple test image (256x256 RGB)
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add some "wound-like" features
        center_x, center_y = 128, 128
        y, x = np.ogrid[:256, :256]
        mask = (x - center_x)**2 + (y - center_y)**2 < 50**2
        test_image[mask] = [150, 50, 50]  # Red wound area
        
        # Save as JPEG
        Image.fromarray(test_image).save(sample_image_path, "JPEG")
    
    # Load and return the image
    image = Image.open(sample_image_path)
    return np.array(image)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Provide a sample wound mask as numpy array."""
    # Create a simple binary mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Add circular wound area
    center_x, center_y = 128, 128
    y, x = np.ogrid[:256, :256]
    wound_area = (x - center_x)**2 + (y - center_y)**2 < 50**2
    mask[wound_area] = 255
    
    return mask


@pytest.fixture
def test_patient() -> Patient:
    """Provide a test patient object."""
    return Patient(
        name="Test Patient",
        age=45,
        gender="M",
        patient_id="TEST001",
        date_of_birth="1978-01-01"
    )


@pytest.fixture
def test_analysis_options() -> AnalysisOptions:
    """Provide test analysis options."""
    return AnalysisOptions(
        confidence_threshold=0.5,
        generate_report=True,
        generate_voice_summary=False,
        include_explainability=False
    )


@pytest.fixture
def mock_model_path(test_config: Dict[str, Any]) -> Path:
    """Provide path to a mock model file."""
    models_dir = Path(test_config["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy model file
    model_path = models_dir / "test_model.keras"
    if not model_path.exists():
        # Create a small dummy file
        model_path.write_text("dummy model content")
    
    return model_path


@pytest.fixture
def test_output_dir(test_config: Dict[str, Any]) -> Path:
    """Provide a test output directory."""
    output_dir = Path(test_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def golden_image_path(test_config: Dict[str, Any]) -> Path:
    """Provide path to a golden image for deterministic testing."""
    golden_dir = Path(test_config["golden_images_dir"])
    golden_dir.mkdir(parents=True, exist_ok=True)
    
    return golden_dir / "golden_wound_mask.png"


@pytest.fixture
def test_metadata() -> Dict[str, Any]:
    """Provide test metadata for analysis results."""
    return {
        "image_path": "/test/path/image.jpg",
        "analysis_timestamp": "2025-09-22T10:00:00Z",
        "model_version": "test_v1.0",
        "processing_time": 1.5,
        "image_dimensions": [256, 256, 3],
        "confidence_score": 0.85
    }


@pytest.fixture
def mock_analysis_result(
    test_patient: Patient,
    test_analysis_options: AnalysisOptions,
    test_metadata: Dict[str, Any]
) -> AnalysisResult:
    """Provide a mock analysis result."""
    return AnalysisResult(
        patient=test_patient,
        options=test_analysis_options,
        mask_path="/test/path/mask.png",
        confidence_score=0.85,
        processing_time=1.5,
        metadata=test_metadata,
        artifacts={
            "visualization": "/test/path/viz.png",
            "report": "/test/path/report.pdf"
        }
    )


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set test environment variables
    os.environ["TEST_MODE"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"
    
    # Disable feature flags for testing
    os.environ["ENABLE_SYNTHETIC_DATA"] = "false"
    os.environ["ENABLE_VOICE_SUMMARY"] = "false"
    os.environ["ENABLE_EXPLAINABILITY"] = "false"
    
    yield
    
    # Cleanup after test
    # Remove test environment variables
    test_vars = ["TEST_MODE", "ENABLE_SYNTHETIC_DATA", "ENABLE_VOICE_SUMMARY", "ENABLE_EXPLAINABILITY"]
    for var in test_vars:
        os.environ.pop(var, None)


class TestUtils:
    """Utility class for test helpers."""
    
    @staticmethod
    def create_test_image(width: int = 256, height: int = 256, channels: int = 3) -> np.ndarray:
        """Create a test image with specified dimensions."""
        return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    
    @staticmethod
    def create_test_mask(width: int = 256, height: int = 256) -> np.ndarray:
        """Create a test binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add some random wound areas
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        wound_area = (x - center_x)**2 + (y - center_y)**2 < (min(width, height) // 4)**2
        mask[wound_area] = 255
        
        return mask
    
    @staticmethod
    def save_test_image(image: np.ndarray, path: Path) -> None:
        """Save a test image to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
    
    @staticmethod
    def load_test_image(path: Path) -> np.ndarray:
        """Load a test image from file."""
        return np.array(Image.open(path))
    
    @staticmethod
    def assert_image_equal(image1: np.ndarray, image2: np.ndarray, tolerance: float = 0.01) -> None:
        """Assert that two images are equal within tolerance."""
        assert image1.shape == image2.shape, f"Image shapes differ: {image1.shape} vs {image2.shape}"
        
        # Calculate mean absolute difference
        diff = np.abs(image1.astype(float) - image2.astype(float))
        mean_diff = np.mean(diff)
        
        assert mean_diff <= tolerance, f"Images differ by {mean_diff:.4f}, tolerance: {tolerance}"


@pytest.fixture
def test_utils() -> TestUtils:
    """Provide test utilities."""
    return TestUtils()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "golden: marks tests as golden image tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.nodeid for keyword in ["model", "generation", "training"]):
            item.add_marker(pytest.mark.slow)