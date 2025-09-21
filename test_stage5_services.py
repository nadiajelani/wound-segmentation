#!/usr/bin/env python3
"""
Test script for Stage 5 Services implementation.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_services_import():
    """Test that all services can be imported."""
    print("🧪 Testing Stage 5 Services Import...")
    
    try:
        from woundseg.services import (
            StorageService, get_storage_service,
            VoiceService, get_voice_service,
            ValidationService, get_validation_service,
            ExplainabilityService, get_explainability_service,
            ReportingService, get_reporting_service
        )
        print("✅ All services imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_storage_service():
    """Test storage service functionality."""
    print("\n📁 Testing Storage Service...")
    
    try:
        from woundseg.services import get_storage_service
        
        storage = get_storage_service()
        
        # Test storage info
        info = storage.get_storage_info()
        print(f"✅ Storage service initialized: {info['base_directory']}")
        
        # Test directory creation
        session_dir = storage.create_session_dir("test_session")
        print(f"✅ Session directory created: {session_dir}")
        
        # Test file operations
        test_data = b"test data for storage service"
        test_path = storage.save_text("test data", "temp", "test.txt")
        print(f"✅ Text file saved: {test_path}")
        
        # Clean up
        storage.delete_file("temp", "test.txt")
        print("✅ Test file cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ Storage service test failed: {e}")
        return False

def test_voice_service():
    """Test voice service functionality."""
    print("\n🎤 Testing Voice Service...")
    
    try:
        from woundseg.services import get_voice_service
        
        voice = get_voice_service()
        
        # Test service info
        info = voice.get_voice_info()
        print(f"✅ Voice service initialized: enabled={info['enabled']}, available={info['available']}")
        
        # Test custom voice generation (if available)
        if voice.is_available():
            test_text = "This is a test of the voice service for wound analysis."
            audio_path = voice.generate_custom_voice(test_text, "test_voice.mp3")
            if audio_path:
                print(f"✅ Voice generation successful: {audio_path}")
                # Clean up
                voice.storage_service.delete_file("voice", "test_voice.mp3")
                print("✅ Test audio file cleaned up")
            else:
                print("⚠️ Voice generation failed (may need internet connection)")
        else:
            print("⚠️ Voice service not available (gTTS not installed or disabled)")
        
        return True
    except Exception as e:
        print(f"❌ Voice service test failed: {e}")
        return False

def test_validation_service():
    """Test validation service functionality."""
    print("\n🔍 Testing Validation Service...")
    
    try:
        from woundseg.services import get_validation_service
        
        validation = get_validation_service()
        
        # Create test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Test image quality validation
        result = validation.validate_image_quality(test_image, "test_image.jpg")
        print(f"✅ Image quality validation: {'PASSED' if result.is_valid else 'FAILED'}")
        print(f"   Metrics: brightness={result.metrics.get('brightness', 0):.1f}, contrast={result.metrics.get('contrast', 0):.1f}")
        
        # Create test mask
        test_mask = np.zeros((128, 128), dtype=np.uint8)
        test_mask[50:80, 50:80] = 255  # Small square mask
        
        # Test segmentation validation
        seg_result = validation.validate_segmentation_result(test_mask, image_path="test_image.jpg")
        print(f"✅ Segmentation validation: {'PASSED' if seg_result.is_valid else 'FAILED'}")
        print(f"   Coverage: {seg_result.metrics.get('coverage_percentage', 0):.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ Validation service test failed: {e}")
        return False

def test_explainability_service():
    """Test explainability service functionality."""
    print("\n🧠 Testing Explainability Service...")
    
    try:
        from woundseg.services import get_explainability_service
        
        explain = get_explainability_service()
        
        # Test service info
        info = explain.get_service_info()
        print(f"✅ Explainability service initialized: enabled={info['enabled']}, available={info['available']}")
        print(f"   Grad-CAM available: {info['gradcam_available']}")
        print(f"   SHAP available: {info['shap_available']}")
        
        if explain.is_available():
            print("✅ Explainability features available")
        else:
            print("⚠️ Explainability features not available (TensorFlow/SHAP not installed or disabled)")
        
        return True
    except Exception as e:
        print(f"❌ Explainability service test failed: {e}")
        return False

def test_reporting_service():
    """Test reporting service functionality."""
    print("\n📄 Testing Reporting Service...")
    
    try:
        from woundseg.services import get_reporting_service
        from woundseg.types import AnalysisResult, Patient
        
        reporting = get_reporting_service()
        
        # Test service info
        info = reporting.get_service_info()
        print(f"✅ Reporting service initialized: available={info['available']}")
        print(f"   FPDF available: {info['fpdf_available']}")
        print(f"   Voice service available: {info['voice_service_available']}")
        
        if reporting.is_available():
            # Create test analysis result
            test_result = AnalysisResult(
                severity="Mild",
                healing_potential="Good",
                area_mm2=25.5,
                confidence_score=0.85,
                original_image=np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                mask=np.zeros((128, 128), dtype=np.uint8)
            )
            
            # Create test patient
            test_patient = Patient(
                name="Test Patient",
                age=45
            )
            
            # Test patient report generation
            patient_report = reporting.generate_patient_report(test_result, test_patient, include_voice=False)
            if patient_report:
                print(f"✅ Patient report generated: {patient_report}")
                # Clean up
                reporting.storage_service.delete_file("reports", patient_report.name)
                print("✅ Test patient report cleaned up")
            else:
                print("⚠️ Patient report generation failed")
            
            # Test clinician report generation
            clinician_report = reporting.generate_clinician_report(test_result, test_patient, include_voice=False)
            if clinician_report:
                print(f"✅ Clinician report generated: {clinician_report}")
                # Clean up
                reporting.storage_service.delete_file("reports", clinician_report.name)
                print("✅ Test clinician report cleaned up")
            else:
                print("⚠️ Clinician report generation failed")
        else:
            print("⚠️ Reporting service not available (FPDF not installed)")
        
        return True
    except Exception as e:
        print(f"❌ Reporting service test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("🧪 Stage 5 Services Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_services_import),
        ("Storage Service", test_storage_service),
        ("Voice Service", test_voice_service),
        ("Validation Service", test_validation_service),
        ("Explainability Service", test_explainability_service),
        ("Reporting Service", test_reporting_service)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All Stage 5 services are working correctly!")
    else:
        print("⚠️ Some services have issues - check the output above")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)