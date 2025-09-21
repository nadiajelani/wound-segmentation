#!/usr/bin/env python3
"""
Test script for gTTS integration with the voice service.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_gtts_direct():
    """Test gTTS directly."""
    print("🎤 Testing gTTS Direct Integration...")
    
    try:
        from gtts import gTTS
        import io
        
        # Test basic gTTS functionality
        text = "Hello, this is a test of Google Text to Speech for wound analysis."
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Generate audio data
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_data = audio_buffer.getvalue()
        
        print(f"✅ gTTS direct test successful: {len(audio_data)} bytes generated")
        return True
        
    except Exception as e:
        print(f"❌ gTTS direct test failed: {e}")
        return False

def test_voice_service():
    """Test the voice service integration."""
    print("\n🎤 Testing Voice Service Integration...")
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        # Test service info
        info = voice_service.get_voice_info()
        print(f"✅ Voice service info: {info}")
        
        # Test availability
        if voice_service.is_available():
            print("✅ Voice service is available")
            
            # Test custom voice generation
            test_text = "This is a test of the voice service for wound analysis."
            audio_path = voice_service.generate_custom_voice(test_text, "test_voice_service.mp3")
            
            if audio_path:
                print(f"✅ Voice service test successful: {audio_path}")
                
                # Check if file exists and has content
                if audio_path.exists():
                    file_size = audio_path.stat().st_size
                    print(f"✅ Audio file created: {file_size} bytes")
                    
                    # Clean up
                    voice_service.storage_service.delete_file("voice", "test_voice_service.mp3")
                    print("✅ Test file cleaned up")
                    return True
                else:
                    print("❌ Audio file was not created")
                    return False
            else:
                print("❌ Voice generation failed")
                return False
        else:
            print("⚠️ Voice service not available (check gTTS installation and config)")
            return False
            
    except Exception as e:
        print(f"❌ Voice service test failed: {e}")
        return False

def test_analysis_voice_generation():
    """Test voice generation with analysis results."""
    print("\n🏥 Testing Analysis Voice Generation...")
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("⚠️ Voice service not available - skipping analysis test")
            return True
        
        # Create mock analysis result
        analysis_result = {
            'area_mm2': 25.5,
            'healing_potential': 'Good',
            'confidence_score': 0.85,
            'severity': 'Mild',
            'timestamp': '2025-09-21'
        }
        
        # Test patient voice generation
        patient_audio = voice_service.generate_summary(analysis_result, "patient", "test_patient_voice.mp3")
        if patient_audio:
            print(f"✅ Patient voice generated: {patient_audio}")
            voice_service.storage_service.delete_file("voice", "test_patient_voice.mp3")
        else:
            print("❌ Patient voice generation failed")
            return False
        
        # Test clinician voice generation
        clinician_audio = voice_service.generate_summary(analysis_result, "clinician", "test_clinician_voice.mp3")
        if clinician_audio:
            print(f"✅ Clinician voice generated: {clinician_audio}")
            voice_service.storage_service.delete_file("voice", "test_clinician_voice.mp3")
        else:
            print("❌ Clinician voice generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis voice generation test failed: {e}")
        return False

def test_voice_content_differences():
    """Test that patient and clinician voices have different content."""
    print("\n📝 Testing Voice Content Differences...")
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("⚠️ Voice service not available - skipping content test")
            return True
        
        # Create mock analysis result
        analysis_result = {
            'area_mm2': 15.2,
            'healing_potential': 'Fair',
            'confidence_score': 0.72,
            'severity': 'Moderate'
        }
        
        # Generate both types of content
        patient_text = voice_service._generate_patient_text(analysis_result)
        clinician_text = voice_service._generate_clinician_text(analysis_result)
        
        print(f"✅ Patient text length: {len(patient_text)} characters")
        print(f"✅ Clinician text length: {len(clinician_text)} characters")
        
        # Check that they're different
        if patient_text != clinician_text:
            print("✅ Patient and clinician content are different (as expected)")
            
            # Show snippets
            print(f"Patient snippet: {patient_text[:100]}...")
            print(f"Clinician snippet: {clinician_text[:100]}...")
            
            return True
        else:
            print("❌ Patient and clinician content are identical (unexpected)")
            return False
            
    except Exception as e:
        print(f"❌ Voice content test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("🎤 gTTS Voice Service Test Suite")
    print("=" * 60)
    
    tests = [
        ("gTTS Direct Test", test_gtts_direct),
        ("Voice Service Integration", test_voice_service),
        ("Analysis Voice Generation", test_analysis_voice_generation),
        ("Voice Content Differences", test_voice_content_differences)
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
        print("🎉 All gTTS voice service tests passed!")
    else:
        print("⚠️ Some tests failed - check the output above")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)