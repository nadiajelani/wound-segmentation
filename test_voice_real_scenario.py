#!/usr/bin/env python3
"""
Test voice service with a realistic wound analysis scenario.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_realistic_wound_analysis_voice():
    """Test voice generation with realistic wound analysis data."""
    print("🏥 Testing Realistic Wound Analysis Voice Generation...")
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("⚠️ Voice service not available - enable with ENABLE_VOICE_SUMMARY=true")
            return False
        
        # Create realistic analysis results for different scenarios
        scenarios = [
            {
                'name': 'Healing Well',
                'data': {
                    'area_mm2': 12.5,
                    'healing_potential': 'Good',
                    'confidence_score': 0.92,
                    'severity': 'Mild',
                    'timestamp': '2025-09-21'
                }
            },
            {
                'name': 'Needs Attention',
                'data': {
                    'area_mm2': 45.8,
                    'healing_potential': 'Poor',
                    'confidence_score': 0.67,
                    'severity': 'Severe',
                    'timestamp': '2025-09-21'
                }
            },
            {
                'name': 'Stable Progress',
                'data': {
                    'area_mm2': 28.3,
                    'healing_potential': 'Fair',
                    'confidence_score': 0.81,
                    'severity': 'Moderate',
                    'timestamp': '2025-09-21'
                }
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📋 Testing scenario: {scenario['name']}")
            
            # Generate patient voice
            patient_audio = voice_service.generate_summary(
                scenario['data'], 
                "patient", 
                f"patient_{scenario['name'].lower().replace(' ', '_')}.mp3"
            )
            
            if patient_audio:
                print(f"✅ Patient voice: {patient_audio.name}")
                # Show the generated text
                patient_text = voice_service._generate_patient_text(scenario['data'])
                print(f"   Text: {patient_text}")
            else:
                print("❌ Patient voice generation failed")
                return False
            
            # Generate clinician voice
            clinician_audio = voice_service.generate_summary(
                scenario['data'], 
                "clinician", 
                f"clinician_{scenario['name'].lower().replace(' ', '_')}.mp3"
            )
            
            if clinician_audio:
                print(f"✅ Clinician voice: {clinician_audio.name}")
                # Show the generated text
                clinician_text = voice_service._generate_clinician_text(scenario['data'])
                print(f"   Text: {clinician_text}")
            else:
                print("❌ Clinician voice generation failed")
                return False
        
        print(f"\n🎉 All {len(scenarios)} scenarios tested successfully!")
        
        # Ask if user wants to keep the files
        keep_files = input("\n🗑️ Keep the generated audio files? (y/n): ").lower().strip()
        if keep_files != 'y':
            # Clean up all generated files
            for scenario in scenarios:
                voice_service.storage_service.delete_file("voice", f"patient_{scenario['name'].lower().replace(' ', '_')}.mp3")
                voice_service.storage_service.delete_file("voice", f"clinician_{scenario['name'].lower().replace(' ', '_')}.mp3")
            print("✅ All test files cleaned up")
        else:
            print("📁 Audio files kept for review")
        
        return True
        
    except Exception as e:
        print(f"❌ Realistic scenario test failed: {e}")
        return False

def test_voice_service_edge_cases():
    """Test voice service with edge cases."""
    print("\n🔍 Testing Voice Service Edge Cases...")
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("⚠️ Voice service not available - skipping edge case tests")
            return True
        
        # Test with minimal data
        minimal_data = {
            'area_mm2': 0,
            'healing_potential': 'unknown',
            'confidence_score': 0.0
        }
        
        print("📋 Testing minimal data scenario...")
        patient_audio = voice_service.generate_summary(minimal_data, "patient", "minimal_patient.mp3")
        if patient_audio:
            print("✅ Minimal data patient voice generated")
            voice_service.storage_service.delete_file("voice", "minimal_patient.mp3")
        else:
            print("❌ Minimal data patient voice failed")
            return False
        
        # Test with extreme values
        extreme_data = {
            'area_mm2': 999.9,
            'healing_potential': 'Poor',
            'confidence_score': 0.99,
            'severity': 'Severe'
        }
        
        print("📋 Testing extreme values scenario...")
        clinician_audio = voice_service.generate_summary(extreme_data, "clinician", "extreme_clinician.mp3")
        if clinician_audio:
            print("✅ Extreme values clinician voice generated")
            voice_service.storage_service.delete_file("voice", "extreme_clinician.mp3")
        else:
            print("❌ Extreme values clinician voice failed")
            return False
        
        print("✅ All edge cases handled correctly")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("🎤 Realistic Voice Service Test Suite")
    print("=" * 60)
    
    tests = [
        ("Realistic Wound Analysis", test_realistic_wound_analysis_voice),
        ("Edge Cases", test_voice_service_edge_cases)
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
        print("🎉 All realistic voice service tests passed!")
        print("💡 The voice service is ready for production use!")
    else:
        print("⚠️ Some tests failed - check the output above")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)