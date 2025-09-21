#!/usr/bin/env python3
"""
Test the new modular services architecture.

This script demonstrates how to use the new modular services
instead of the old test_wound_progress.py approach.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_new_modular_services():
    """Test all the new modular services."""
    
    print("🏗️ Testing New Modular Services")
    print("=" * 50)
    
    try:
        # Import the new modular services
        from woundseg.services import (
            get_storage_service,
            get_voice_service,
            get_reporting_service,
            get_validation_service,
            get_explainability_service
        )
        from woundseg.types import Patient, AnalysisResult
        
        print("✅ Successfully imported all modular services")
        
        # Test 1: Storage Service
        print("\n1️⃣ Testing Storage Service...")
        storage = get_storage_service()
        print(f"   Base directory: {storage.base_dir}")
        print(f"   Available directories: {list(storage.dirs.keys())}")
        
        # Test 2: Voice Service
        print("\n2️⃣ Testing Voice Service...")
        voice = get_voice_service()
        if voice.is_available():
            print("   ✅ Voice service is available")
            
            # Create mock analysis result
            analysis_result = {
                'area_mm2': 25.5,
                'healing_potential': 'Good',
                'confidence_score': 0.88,
                'severity': 'Mild',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Generate patient voice
            patient_voice = voice.generate_summary(
                analysis_result, 
                audience="patient", 
                filename="test_patient_voice.mp3"
            )
            print(f"   Patient voice: {patient_voice}")
            
            # Generate clinician voice
            clinician_voice = voice.generate_summary(
                analysis_result, 
                audience="clinician", 
                filename="test_clinician_voice.mp3"
            )
            print(f"   Clinician voice: {clinician_voice}")
        else:
            print("   ⚠️ Voice service not available (ENABLE_VOICE_SUMMARY=false)")
        
        # Test 3: Reporting Service
        print("\n3️⃣ Testing Reporting Service...")
        reporting = get_reporting_service()
        
        # Create mock patient and analysis
        patient = Patient(
            name="Test Patient",
            age=45
        )
        
        # Create mock numpy arrays for the required fields
        import numpy as np
        
        analysis_result_obj = AnalysisResult(
            area_mm2=25.5,
            healing_potential="Good",
            confidence_score=0.88,
            severity="Mild",
            original_image=np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            mask=np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        )
        
        # Generate patient report
        patient_report = reporting.generate_patient_report(
            patient, analysis_result_obj, "test_patient_report.pdf"
        )
        print(f"   Patient report: {patient_report}")
        
        # Generate clinician report
        clinician_report = reporting.generate_clinician_report(
            patient, analysis_result_obj, "test_clinician_report.pdf"
        )
        print(f"   Clinician report: {clinician_report}")
        
        # Test 4: Validation Service
        print("\n4️⃣ Testing Validation Service...")
        validation = get_validation_service()
        
        # Test analysis result validation
        validation_result = validation.validate_analysis_result(analysis_result_obj)
        print(f"   Validation result: {validation_result.is_valid}")
        print(f"   Issues: {validation_result.issues}")
        print(f"   Recommendations: {validation_result.recommendations}")
        
        # Test 5: Explainability Service
        print("\n5️⃣ Testing Explainability Service...")
        explainability = get_explainability_service()
        
        if explainability.is_available():
            print("   ✅ Explainability service is available")
            print("   Features: Grad-CAM, SHAP analysis")
        else:
            print("   ⚠️ Explainability service not available (ENABLE_EXPLAINABILITY=false)")
        
        # Show file locations
        print("\n📁 Files Generated:")
        print("=" * 30)
        
        # Check what files were created
        for name, path in storage.dirs.items():
            if path.exists():
                files = list(path.glob("*"))
                if files:
                    print(f"   {name}:")
                    for file in files:
                        print(f"      - {file.name}")
        
        print("\n✅ All modular services tested successfully!")
        print("🎯 Files are organized in the outputs/ directory structure")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing modular services: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_comparison():
    """Show comparison between old and new approaches."""
    
    print("\n" + "=" * 60)
    print("🔄 OLD vs NEW Comparison")
    print("=" * 60)
    
    print("\n❌ OLD (test_wound_progress.py):")
    print("   • Files saved to: wound_progress_report/")
    print("   • Hardcoded paths")
    print("   • Mixed functionality")
    print("   • No voice service")
    print("   • No validation service")
    print("   • No explainability service")
    
    print("\n✅ NEW (Modular Services):")
    print("   • Files saved to: outputs/ (organized)")
    print("   • Clean service separation")
    print("   • Voice service with gTTS")
    print("   • Validation service")
    print("   • Explainability service")
    print("   • Feature flags")
    print("   • Professional architecture")

def main():
    """Main function."""
    
    print("🚀 New Modular Services Test")
    print("=" * 60)
    print("This script tests the new modular services architecture")
    print("instead of the old test_wound_progress.py approach.")
    print("=" * 60)
    
    # Test the services
    success = test_new_modular_services()
    
    # Show comparison
    show_comparison()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("The new modular services are working correctly.")
        print("Files are organized in outputs/ directory.")
        print("\n💡 To use in your own code:")
        print("   from woundseg.services import get_*_service")
        print("   service = get_*_service()")
        print("   result = service.method(...)")
    else:
        print("\n❌ FAILED!")
        print("There were errors testing the modular services.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)