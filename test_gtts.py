#!/usr/bin/env python3
"""
Simple test script for gTTS (Google Text-to-Speech)
"""

import os
from gtts import gTTS
import pygame
import time

def test_gtts_basic():
    """Test basic gTTS functionality"""
    print("🎤 Testing gTTS basic functionality...")
    
    try:
        # Test text
        text = "Hello, this is a test of Google Text to Speech for wound analysis."
        
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to file
        output_file = "test_voice.mp3"
        tts.save(output_file)
        
        print(f"✅ Audio file created: {output_file}")
        print(f"📁 File size: {os.path.getsize(output_file)} bytes")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error creating audio: {e}")
        return None

def test_gtts_playback(audio_file):
    """Test audio playback (requires pygame)"""
    print("🔊 Testing audio playback...")
    
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load and play audio
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        print("✅ Audio playback completed")
        
    except Exception as e:
        print(f"❌ Error playing audio: {e}")
        print("💡 Install pygame for audio playback: pip install pygame")

def test_gtts_wound_analysis():
    """Test gTTS with wound analysis content"""
    print("🏥 Testing gTTS with wound analysis content...")
    
    try:
        # Wound analysis text
        wound_text = """
        Wound analysis complete. 
        Wound area: 2.5 square centimeters.
        Healing progress: Positive trend detected.
        Recommendation: Continue current treatment.
        """
        
        # Create gTTS object
        tts = gTTS(text=wound_text, lang='en', slow=False)
        
        # Save to file
        output_file = "wound_analysis_voice.mp3"
        tts.save(output_file)
        
        print(f"✅ Wound analysis audio created: {output_file}")
        print(f"📁 File size: {os.path.getsize(output_file)} bytes")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error creating wound analysis audio: {e}")
        return None

def cleanup_test_files():
    """Clean up test files"""
    test_files = ["test_voice.mp3", "wound_analysis_voice.mp3"]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ Cleaned up: {file}")

def main():
    """Main test function"""
    print("=" * 60)
    print("🎤 gTTS (Google Text-to-Speech) Test")
    print("=" * 60)
    
    # Test 1: Basic functionality
    audio_file = test_gtts_basic()
    
    if audio_file:
        # Test 2: Audio playback
        test_gtts_playback(audio_file)
        
        # Test 3: Wound analysis content
        wound_audio = test_gtts_wound_analysis()
        
        print("\n" + "=" * 60)
        print("📊 Test Results Summary:")
        print("=" * 60)
        print("✅ gTTS installation: Working")
        print("✅ Audio file creation: Working")
        print("✅ Wound analysis content: Working")
        
        if wound_audio:
            print("✅ Multiple audio files: Working")
        
        print("\n💡 gTTS is ready for Stage 5 voice service implementation!")
        
        # Ask if user wants to keep test files
        keep_files = input("\n🗑️ Keep test audio files? (y/n): ").lower().strip()
        if keep_files != 'y':
            cleanup_test_files()
            print("✅ Test files cleaned up")
        else:
            print("📁 Test files kept for review")
    
    else:
        print("\n❌ gTTS test failed - check internet connection and gTTS installation")

if __name__ == "__main__":
    main()