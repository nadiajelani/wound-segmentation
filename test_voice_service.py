#!/usr/bin/env python3
"""
Simple test of the voice service with new modular architecture.

This script tests the voice service without requiring a specific input image.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_voice_service():
    """Test the voice service."""
    
    print("🎤 Testing Voice Service")
    print("=" * 40)
    
    try:
        from woundseg.services import get_voice_service
        
        # Get voice service
        voice_service = get_voice_service()
        
        # Check if available
        if not voice_service.is_available():
            print("❌ Voice service not available")
            print("💡 Enable with: ENABLE_VOICE_SUMMARY=true")
            return False
        
        print("✅ Voice service is available")
        
        # Create mock analysis result
        analysis_result = {
            'area_mm2': 25.5,
            'healing_potential': 'Good',
            'confidence_score': 0.88,
            'severity': 'Mild',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"📊 Analysis result: {analysis_result['area_mm2']} mm² wound, {analysis_result['healing_potential']} healing")
        
        # Generate patient voice
        print("\n👤 Generating patient voice...")
        patient_audio = voice_service.generate_summary(
            analysis_result, 
            audience="patient", 
            filename="test_patient_voice.mp3"
        )
        
        if patient_audio:
            print(f"✅ Patient voice: {patient_audio}")
        else:
            print("❌ Patient voice generation failed")
            return False
        
        # Generate clinician voice
        print("\n👨‍⚕️ Generating clinician voice...")
        clinician_audio = voice_service.generate_summary(
            analysis_result, 
            audience="clinician", 
            filename="test_clinician_voice.mp3"
        )
        
        if clinician_audio:
            print(f"✅ Clinician voice: {clinician_audio}")
        else:
            print("❌ Clinician voice generation failed")
            return False
        
        # Generate custom voice
        print("\n🎵 Generating custom voice...")
        custom_text = "Hello! This is a custom voice message for wound analysis. The system has detected a wound area of 25 square millimeters with good healing potential."
        
        custom_audio = voice_service.generate_custom_voice(
            text=custom_text,
            filename="test_custom_voice.mp3",
            lang="en",
            slow=False
        )
        
        if custom_audio:
            print(f"✅ Custom voice: {custom_audio}")
        else:
            print("❌ Custom voice generation failed")
            return False
        
        # Show file locations
        print("\n📁 Files saved to:")
        print(f"   Patient voice: {patient_audio}")
        print(f"   Clinician voice: {clinician_audio}")
        print(f"   Custom voice: {custom_audio}")
        
        print("\n🎉 Voice service test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Voice service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_voice_info():
    """Show voice service information."""
    
    print("\n📋 Voice Service Information")
    print("=" * 40)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        info = voice_service.get_voice_info()
        
        for key, value in info.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error getting voice info: {e}")

def cleanup_test_files():
    """Clean up test files."""
    
    print("\n🗑️ Cleaning up test files...")
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        # Clean up voice files
        test_files = ["test_patient_voice.mp3", "test_clinician_voice.mp3", "test_custom_voice.mp3"]
        for filename in test_files:
            voice_service.storage_service.delete_file("voice", filename)
        
        print("✅ Test files cleaned up")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

def main():
    """Main function."""
    
    print("🚀 Voice Service Test")
    print("=" * 50)
    print("This script tests the voice service with the new modular architecture.")
    print("=" * 50)
    
    # Test voice service
    success = test_voice_service()
    
    # Show voice info
    show_voice_info()
    
    if success:
        print("\n✅ SUCCESS!")
        print("The voice service is working correctly.")
        print("Files are saved to outputs/voice_summaries/")
        
        # Ask about cleanup
        cleanup = input("\n🗑️ Clean up test files? (y/n): ").lower().strip()
        if cleanup == 'y':
            cleanup_test_files()
    else:
        print("\n❌ FAILED!")
        print("There were errors testing the voice service.")
        print("\n💡 Troubleshooting:")
        print("1. Make sure gTTS is installed: pip install gtts")
        print("2. Enable voice service: ENABLE_VOICE_SUMMARY=true")
        print("3. Check internet connection (gTTS needs internet)")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)