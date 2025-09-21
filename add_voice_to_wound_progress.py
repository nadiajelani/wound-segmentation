#!/usr/bin/env python3
"""
Script to add voice functionality to your existing test_wound_progress.py
"""

import os
import sys
from pathlib import Path

def add_voice_imports():
    """Add voice service imports to the top of your file."""
    return '''
# Add this import at the top of your test_wound_progress.py file
from woundseg.services import get_voice_service
'''

def add_voice_initialization():
    """Add voice service initialization code."""
    return '''
# Add this after your model loading section
# -------- Initialize Voice Service --------
ENABLE_VOICE = True  # Set to False to disable voice generation
voice_service = None

if ENABLE_VOICE:
    try:
        voice_service = get_voice_service()
        if voice_service.is_available():
            logger.info("Voice service initialized successfully")
        else:
            logger.warning("Voice service not available - check gTTS installation and ENABLE_VOICE_SUMMARY config")
            voice_service = None
    except Exception as e:
        logger.warning(f"Failed to initialize voice service: {e}")
        voice_service = None
'''

def add_voice_generation_function():
    """Add voice generation function."""
    return '''
def generate_voice_summary(metrics, voice_service):
    """Generate voice summary of wound analysis."""
    if not voice_service or not voice_service.is_available():
        logger.warning("Voice service not available - skipping voice generation")
        return None
    
    try:
        # Create analysis result for voice service
        analysis_result = {
            'area_mm2': metrics.get("Wound Area (mm²)", 0),
            'healing_potential': 'Good' if 'healing well' in metrics.get("Condition", "").lower() else 'Fair' if 'stable' in metrics.get("Condition", "").lower() else 'Poor',
            'confidence_score': 0.85,  # Mock confidence score
            'severity': 'Mild' if metrics.get("Wound Area (mm²)", 0) < 10 else 'Moderate' if metrics.get("Wound Area (mm²)", 0) < 50 else 'Severe',
            'timestamp': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Generate patient voice summary
        patient_audio = voice_service.generate_summary(
            analysis_result, 
            "patient", 
            "patient_voice_summary.mp3"
        )
        
        if patient_audio:
            logger.info(f"Patient voice summary generated: {patient_audio}")
        
        # Generate clinician voice summary
        clinician_audio = voice_service.generate_summary(
            analysis_result, 
            "clinician", 
            "clinician_voice_summary.mp3"
        )
        
        if clinician_audio:
            logger.info(f"Clinician voice summary generated: {clinician_audio}")
        
        return {
            'patient_audio': patient_audio,
            'clinician_audio': clinician_audio,
            'analysis_result': analysis_result
        }
        
    except Exception as e:
        logger.error(f"Failed to generate voice summary: {e}")
        return None
'''

def add_voice_to_main():
    """Add voice generation call to main function."""
    return '''
# Add this in your main function after calculating metrics
# Generate voice summary
voice_info = None
if ENABLE_VOICE and voice_service:
    logger.info("Generating voice summary...")
    voice_info = generate_voice_summary(metrics, voice_service)

# Add this to your final summary
if voice_info:
    logger.info("Voice summaries generated:")
    if voice_info.get('patient_audio'):
        logger.info(f"  - Patient: {voice_info['patient_audio']}")
    if voice_info.get('clinician_audio'):
        logger.info(f"  - Clinician: {voice_info['clinician_audio']}")
'''

def create_test_script():
    """Create a simple test script to verify gTTS integration."""
    test_script = '''#!/usr/bin/env python3
"""
Simple test to verify gTTS integration in your wound progress analysis
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_voice_integration():
    """Test voice service integration."""
    print("🎤 Testing Voice Integration for Wound Progress Analysis...")
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("⚠️ Voice service not available")
            print("💡 To enable voice service:")
            print("   1. Install gTTS: pip install gtts")
            print("   2. Set environment variable: ENABLE_VOICE_SUMMARY=true")
            return False
        
        print("✅ Voice service is available")
        
        # Test with mock wound analysis data
        mock_metrics = {
            "Wound Area (mm²)": 25.5,
            "Condition": "Medium wound - stable",
            "Instructions": "Maintain current treatment. Regular monitoring recommended."
        }
        
        # Create analysis result
        analysis_result = {
            'area_mm2': mock_metrics["Wound Area (mm²)"],
            'healing_potential': 'Fair',
            'confidence_score': 0.85,
            'severity': 'Moderate',
            'timestamp': '2025-09-21'
        }
        
        # Generate voice summaries
        print("🎤 Generating patient voice summary...")
        patient_audio = voice_service.generate_summary(analysis_result, "patient", "test_patient.mp3")
        
        if patient_audio:
            print(f"✅ Patient voice generated: {patient_audio}")
            # Clean up
            voice_service.storage_service.delete_file("voice", "test_patient.mp3")
        
        print("🎤 Generating clinician voice summary...")
        clinician_audio = voice_service.generate_summary(analysis_result, "clinician", "test_clinician.mp3")
        
        if clinician_audio:
            print(f"✅ Clinician voice generated: {clinician_audio}")
            # Clean up
            voice_service.storage_service.delete_file("voice", "test_clinician.mp3")
        
        print("🎉 Voice integration test successful!")
        print("💡 You can now add voice functionality to your test_wound_progress.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_voice_integration()
    sys.exit(0 if success else 1)
'''
    
    with open("test_voice_integration.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_voice_integration.py")

def main():
    """Main function to show how to add voice to your wound progress script."""
    print("=" * 60)
    print("🎤 Adding Voice Functionality to test_wound_progress.py")
    print("=" * 60)
    
    print("\n📝 Here's how to add voice functionality to your existing script:")
    
    print("\n1️⃣ ADD IMPORTS:")
    print(add_voice_imports())
    
    print("\n2️⃣ ADD VOICE SERVICE INITIALIZATION:")
    print(add_voice_initialization())
    
    print("\n3️⃣ ADD VOICE GENERATION FUNCTION:")
    print(add_voice_generation_function())
    
    print("\n4️⃣ ADD VOICE GENERATION TO MAIN FUNCTION:")
    print(add_voice_to_main())
    
    print("\n5️⃣ TEST THE INTEGRATION:")
    create_test_script()
    
    print("\n" + "=" * 60)
    print("🚀 QUICK START:")
    print("=" * 60)
    print("1. Enable voice service: ENABLE_VOICE_SUMMARY=true")
    print("2. Run the test: python test_voice_integration.py")
    print("3. If successful, add the code snippets above to your test_wound_progress.py")
    print("4. Run your enhanced script: python test_wound_progress.py")
    
    print("\n💡 ALTERNATIVE:")
    print("Use the complete enhanced version: test_wound_progress_with_voice.py")

if __name__ == "__main__":
    main()