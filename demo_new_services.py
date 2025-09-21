#!/usr/bin/env python3
"""
Demo of New Modular Services

This script demonstrates the new modular services architecture
without requiring a specific input image.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def demo_voice_service():
    """Demo the voice service."""
    print("🎤 Voice Service Demo")
    print("=" * 30)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
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
        
        # Generate patient voice
        patient_audio = voice_service.generate_summary(
            analysis_result, 
            audience="patient", 
            filename="demo_patient_voice.mp3"
        )
        
        # Generate clinician voice
        clinician_audio = voice_service.generate_summary(
            analysis_result, 
            audience="clinician", 
            filename="demo_clinician_voice.mp3"
        )
        
        if patient_audio and clinician_audio:
            print(f"✅ Patient voice: {patient_audio}")
            print(f"✅ Clinician voice: {clinician_audio}")
            return True
        else:
            print("❌ Voice generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Voice service demo failed: {e}")
        return False

def demo_storage_service():
    """Demo the storage service."""
    print("\n📁 Storage Service Demo")
    print("=" * 30)
    
    try:
        from woundseg.services import get_storage_service
        
        storage_service = get_storage_service()
        
        print(f"✅ Base directory: {storage_service.base_dir}")
        print("✅ Available directories:")
        for name, path in storage_service.dirs.items():
            print(f"   - {name}: {path}")
        
        # Test file creation
        test_text = "This is a test file created by the storage service."
        text_path = storage_service.save_text(
            test_text, "temp", "demo_test.txt"
        )
        
        if text_path:
            print(f"✅ Test file created: {text_path}")
            return True
        else:
            print("❌ File creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Storage service demo failed: {e}")
        return False

def demo_custom_voice():
    """Demo custom voice generation."""
    print("\n🎵 Custom Voice Demo")
    print("=" * 30)
    
    try:
        from woundseg.services import get_voice_service
        
        voice_service = get_voice_service()
        
        if not voice_service.is_available():
            print("❌ Voice service not available")
            return False
        
        # Generate custom voice
        custom_text = "Hello! This is a custom voice message for wound analysis. The system has detected a wound area of 25 square millimeters with good healing potential."
        
        custom_audio = voice_service.generate_custom_voice(
            text=custom_text,
            filename="demo_custom_voice.mp3",
            lang="en",
            slow=False
        )
        
        if custom_audio:
            print(f"✅ Custom voice: {custom_audio}")
            print(f"📝 Text: {custom_text}")
            return True
        else:
            print("❌ Custom voice generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Custom voice demo failed: {e}")
        return False

def show_file_locations():
    """Show where files are saved."""
    print("\n📂 File Locations")
    print("=" * 30)
    
    try:
        from woundseg.services import get_storage_service
        
        storage_service = get_storage_service()
        
        print("Files are saved to:")
        for name, path in storage_service.dirs.items():
            if path.exists():
                files = list(path.glob("*"))
                if files:
                    print(f"   {name}:")
                    for file in files:
                        print(f"      - {file.name}")
                else:
                    print(f"   {name}: (empty)")
            else:
                print(f"   {name}: (does not exist)")
        
    except Exception as e:
        print(f"❌ Error showing file locations: {e}")

def cleanup_demo_files():
    """Clean up demo files."""
    print("\n🗑️ Cleaning up demo files...")
    
    try:
        from woundseg.services import get_storage_service
        
        storage_service = get_storage_service()
        
        # Clean up voice files
        voice_files = ["demo_patient_voice.mp3", "demo_clinician_voice.mp3", "demo_custom_voice.mp3"]
        for filename in voice_files:
            storage_service.delete_file("voice", filename)
        
        # Clean up temp files
        storage_service.delete_file("temp", "demo_test.txt")
        
        print("✅ Demo files cleaned up")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

def main():
    """Main demo function."""
    
    print("🚀 New Modular Services Demo")
    print("=" * 50)
    print("This demo shows the new modular services architecture")
    print("without requiring a specific input image.")
    print("=" * 50)
    
    # Run demos
    demos = [
        ("Storage Service", demo_storage_service),
        ("Voice Service", demo_voice_service),
        ("Custom Voice", demo_custom_voice)
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            result = demo_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} demo crashed: {e}")
            results.append((name, False))
    
    # Show file locations
    show_file_locations()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Demo Summary:")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} demos passed")
    
    if passed == len(results):
        print("🎉 All demos completed successfully!")
        print("💡 The new modular services are working correctly!")
    else:
        print("⚠️ Some demos failed - check the output above")
    
    # Ask about cleanup
    cleanup = input("\n🗑️ Clean up demo files? (y/n): ").lower().strip()
    if cleanup == 'y':
        cleanup_demo_files()
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)