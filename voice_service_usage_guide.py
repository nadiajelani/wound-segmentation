#!/usr/bin/env python3
"""
Comprehensive guide on how to use the voice service.

This script demonstrates all the different ways to use the voice service
for wound analysis and other applications.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def example_1_basic_usage():
    """Example 1: Basic voice service usage."""
    print("=" * 60)
    print("📖 Example 1: Basic Voice Service Usage")
    print("=" * 60)
    
    try:
        from woundseg.services import get_voice_service
        
        # Get the voice service
        voice_service = get_voice_service()
        
        # Check if voice service is available
        if not voice_service.is_available():
            print("❌ Voice service not available")
            print("💡 Enable it with: ENABLE_VOICE_SUMMARY=true")
            return False
        
        print("✅ Voice service is available")
        
        # Generate a simple voice summary
        analysis_result = {
            'area_mm2': 15.5,
            'healing_potential': 'Good',
            'confidence_score': 0.88,
            'severity': 'Mild',
            'timestamp': '2025-09-21'
        }
        
        # Generate patient voice
        patient_audio = voice_service.generate_summary(
            analysis_result, 
            audience="patient", 
            filename="basic_patient.mp3"
        )
        
        if patient_audio:
            print(f"✅ Patient voice generated: {patient_audio}")
            # Clean up
            voice_service.storage_service.delete_file("voice", "basic_patient.mp3")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic usage failed: {e}")
        return False

def example_2_custom_voice():
    """Example 2: Custom voice generation."""
    print("\n" + "=" * 60)
    print("📖 Example 2: Custom Voice Generation")
    print("=" * 60)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("❌ Voice service not available")
            return False
        
        # Generate custom voice with your own text
        custom_text = "Hello! This is a custom voice message for wound analysis. The system has detected a wound area of 25 square millimeters with good healing potential."
        
        custom_audio = voice_service.generate_custom_voice(
            text=custom_text,
            filename="custom_message.mp3",
            lang='en',
            slow=False
        )
        
        if custom_audio:
            print(f"✅ Custom voice generated: {custom_audio}")
            print(f"📝 Text: {custom_text}")
            # Clean up
            voice_service.storage_service.delete_file("voice", "custom_message.mp3")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom voice failed: {e}")
        return False

def example_3_patient_vs_clinician():
    """Example 3: Patient vs Clinician voice differences."""
    print("\n" + "=" * 60)
    print("📖 Example 3: Patient vs Clinician Voice Differences")
    print("=" * 60)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("❌ Voice service not available")
            return False
        
        # Same analysis result
        analysis_result = {
            'area_mm2': 35.2,
            'healing_potential': 'Fair',
            'confidence_score': 0.75,
            'severity': 'Moderate',
            'timestamp': '2025-09-21'
        }
        
        # Generate both patient and clinician voices
        patient_audio = voice_service.generate_summary(
            analysis_result, 
            audience="patient", 
            filename="comparison_patient.mp3"
        )
        
        clinician_audio = voice_service.generate_summary(
            analysis_result, 
            audience="clinician", 
            filename="comparison_clinician.mp3"
        )
        
        if patient_audio and clinician_audio:
            print("✅ Both voices generated successfully")
            
            # Show the text differences
            patient_text = voice_service._generate_patient_text(analysis_result)
            clinician_text = voice_service._generate_clinician_text(analysis_result)
            
            print(f"\n👤 Patient Text ({len(patient_text)} chars):")
            print(f"   {patient_text}")
            
            print(f"\n👨‍⚕️ Clinician Text ({len(clinician_text)} chars):")
            print(f"   {clinician_text}")
            
            # Clean up
            voice_service.storage_service.delete_file("voice", "comparison_patient.mp3")
            voice_service.storage_service.delete_file("voice", "comparison_clinician.mp3")
        
        return True
        
    except Exception as e:
        print(f"❌ Patient vs clinician comparison failed: {e}")
        return False

def example_4_different_languages():
    """Example 4: Voice generation in different languages."""
    print("\n" + "=" * 60)
    print("📖 Example 4: Different Languages")
    print("=" * 60)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("❌ Voice service not available")
            return False
        
        # Test different languages
        languages = [
            {'code': 'en', 'name': 'English', 'text': 'Your wound is healing well. Continue your treatment.'},
            {'code': 'es', 'name': 'Spanish', 'text': 'Tu herida está sanando bien. Continúa con tu tratamiento.'},
            {'code': 'fr', 'name': 'French', 'text': 'Votre blessure guérit bien. Continuez votre traitement.'},
            {'code': 'de', 'name': 'German', 'text': 'Ihre Wunde heilt gut. Setzen Sie Ihre Behandlung fort.'}
        ]
        
        for lang in languages:
            print(f"🌍 Testing {lang['name']} ({lang['code']})...")
            
            audio = voice_service.generate_custom_voice(
                text=lang['text'],
                filename=f"multilingual_{lang['code']}.mp3",
                lang=lang['code'],
                slow=False
            )
            
            if audio:
                print(f"   ✅ {lang['name']} voice generated")
                # Clean up
                voice_service.storage_service.delete_file("voice", f"multilingual_{lang['code']}.mp3")
            else:
                print(f"   ❌ {lang['name']} voice failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Multilingual test failed: {e}")
        return False

def example_5_integration_with_analysis():
    """Example 5: Integration with wound analysis workflow."""
    print("\n" + "=" * 60)
    print("📖 Example 5: Integration with Wound Analysis Workflow")
    print("=" * 60)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("❌ Voice service not available")
            return False
        
        # Simulate a complete wound analysis workflow
        print("🔬 Simulating wound analysis workflow...")
        
        # Step 1: Analyze wound (mock data)
        wound_analysis = {
            'area_mm2': 22.8,
            'perimeter_mm': 18.5,
            'healing_potential': 'Good',
            'confidence_score': 0.91,
            'severity': 'Mild',
            'shape_irregularity': 'Regular',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"📊 Analysis complete: {wound_analysis['area_mm2']} mm² wound detected")
        
        # Step 2: Generate voice summaries
        print("🎤 Generating voice summaries...")
        
        patient_audio = voice_service.generate_summary(
            wound_analysis, 
            audience="patient", 
            filename="workflow_patient.mp3"
        )
        
        clinician_audio = voice_service.generate_summary(
            wound_analysis, 
            audience="clinician", 
            filename="workflow_clinician.mp3"
        )
        
        # Step 3: Generate additional custom messages
        if wound_analysis['healing_potential'] == 'Good':
            encouragement_text = "Great news! Your wound is showing excellent healing progress. Keep up the good work with your treatment plan."
        else:
            encouragement_text = "Your wound requires attention. Please follow your treatment plan carefully and consult your healthcare provider if needed."
        
        encouragement_audio = voice_service.generate_custom_voice(
            text=encouragement_text,
            filename="workflow_encouragement.mp3"
        )
        
        # Step 4: Report results
        if patient_audio and clinician_audio and encouragement_audio:
            print("✅ All voice summaries generated successfully!")
            print(f"   📁 Patient summary: {patient_audio}")
            print(f"   📁 Clinician summary: {clinician_audio}")
            print(f"   📁 Encouragement message: {encouragement_audio}")
            
            # Clean up
            voice_service.storage_service.delete_file("voice", "workflow_patient.mp3")
            voice_service.storage_service.delete_file("voice", "workflow_clinician.mp3")
            voice_service.storage_service.delete_file("voice", "workflow_encouragement.mp3")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow integration failed: {e}")
        return False

def example_6_error_handling():
    """Example 6: Error handling and edge cases."""
    print("\n" + "=" * 60)
    print("📖 Example 6: Error Handling and Edge Cases")
    print("=" * 60)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        # Test 1: Service not available
        print("🧪 Test 1: Service not available")
        if not voice_service.is_available():
            print("   ✅ Gracefully handled: Service not available")
        else:
            print("   ✅ Service is available")
        
        # Test 2: Invalid audience
        print("\n🧪 Test 2: Invalid audience")
        try:
            invalid_audio = voice_service.generate_summary(
                {'area_mm2': 10}, 
                audience="invalid", 
                filename="test.mp3"
            )
            print("   ❌ Should have failed with invalid audience")
        except ValueError as e:
            print(f"   ✅ Correctly caught error: {e}")
        
        # Test 3: Empty text
        print("\n🧪 Test 3: Empty text")
        if voice_service.is_available():
            empty_audio = voice_service.generate_custom_voice(
                text="",
                filename="empty.mp3"
            )
            if empty_audio:
                print("   ⚠️ Empty text handled (may generate silence)")
                voice_service.storage_service.delete_file("voice", "empty.mp3")
            else:
                print("   ✅ Empty text correctly rejected")
        
        # Test 4: Very long text
        print("\n🧪 Test 4: Very long text")
        if voice_service.is_available():
            long_text = "This is a very long text. " * 100  # 2500+ characters
            long_audio = voice_service.generate_custom_voice(
                text=long_text,
                filename="long_text.mp3"
            )
            if long_audio:
                print("   ✅ Long text handled successfully")
                voice_service.storage_service.delete_file("voice", "long_text.mp3")
            else:
                print("   ❌ Long text failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def example_7_service_info():
    """Example 7: Getting service information."""
    print("\n" + "=" * 60)
    print("📖 Example 7: Service Information")
    print("=" * 60)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        # Get service information
        info = voice_service.get_voice_info()
        
        print("📋 Voice Service Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Get storage information
        storage_info = voice_service.storage_service.get_storage_info()
        print(f"\n📁 Storage Information:")
        print(f"   Base directory: {storage_info['base_directory']}")
        print(f"   Total size: {storage_info['total_size_bytes']} bytes")
        
        return True
        
    except Exception as e:
        print(f"❌ Service info failed: {e}")
        return False

def main():
    """Main function to run all examples."""
    print("🎤 Voice Service Usage Guide")
    print("=" * 60)
    print("This guide demonstrates all the different ways to use the voice service.")
    print("Make sure to enable voice service with: ENABLE_VOICE_SUMMARY=true")
    print("=" * 60)
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Custom Voice", example_2_custom_voice),
        ("Patient vs Clinician", example_3_patient_vs_clinician),
        ("Different Languages", example_4_different_languages),
        ("Analysis Integration", example_5_integration_with_analysis),
        ("Error Handling", example_6_error_handling),
        ("Service Information", example_7_service_info)
    ]
    
    results = []
    for name, example_func in examples:
        try:
            result = example_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Examples Summary:")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} examples passed")
    
    if passed == len(results):
        print("🎉 All examples completed successfully!")
        print("💡 You now know how to use the voice service in different scenarios!")
    else:
        print("⚠️ Some examples failed - check the output above")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)