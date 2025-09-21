#!/usr/bin/env python3
"""
Quick Reference Guide for Voice Service Usage

Copy and paste these code snippets into your projects.
"""

# =============================================================================
# 1. BASIC SETUP
# =============================================================================

def setup_voice_service():
    """Basic setup for voice service."""
    from woundseg.services import get_voice_service
    
    # Get voice service
    voice_service = get_voice_service()
    
    # Check if available
    if voice_service.is_available():
        print("✅ Voice service ready")
        return voice_service
    else:
        print("❌ Voice service not available")
        print("💡 Enable with: ENABLE_VOICE_SUMMARY=true")
        return None

# =============================================================================
# 2. GENERATE WOUND ANALYSIS VOICE
# =============================================================================

def generate_wound_voice(voice_service, wound_area_mm2, healing_potential="Good"):
    """Generate voice summary for wound analysis."""
    analysis_result = {
        'area_mm2': wound_area_mm2,
        'healing_potential': healing_potential,
        'confidence_score': 0.85,
        'severity': 'Mild' if wound_area_mm2 < 10 else 'Moderate' if wound_area_mm2 < 50 else 'Severe',
        'timestamp': '2025-09-21'
    }
    
    # Generate patient voice
    patient_audio = voice_service.generate_summary(
        analysis_result, 
        audience="patient", 
        filename="patient_summary.mp3"
    )
    
    # Generate clinician voice
    clinician_audio = voice_service.generate_summary(
        analysis_result, 
        audience="clinician", 
        filename="clinician_summary.mp3"
    )
    
    return patient_audio, clinician_audio

# =============================================================================
# 3. CUSTOM VOICE MESSAGES
# =============================================================================

def create_custom_voice(voice_service, text, filename="custom.mp3", language="en"):
    """Create custom voice message."""
    audio = voice_service.generate_custom_voice(
        text=text,
        filename=filename,
        lang=language,
        slow=False
    )
    return audio

# =============================================================================
# 4. MULTILINGUAL SUPPORT
# =============================================================================

def create_multilingual_voices(voice_service):
    """Create voices in different languages."""
    messages = {
        'en': "Your wound is healing well. Continue your treatment.",
        'es': "Tu herida está sanando bien. Continúa con tu tratamiento.",
        'fr': "Votre blessure guérit bien. Continuez votre traitement.",
        'de': "Ihre Wunde heilt gut. Setzen Sie Ihre Behandlung fort."
    }
    
    audio_files = {}
    for lang, text in messages.items():
        audio = voice_service.generate_custom_voice(
            text=text,
            filename=f"message_{lang}.mp3",
            lang=lang
        )
        audio_files[lang] = audio
    
    return audio_files

# =============================================================================
# 5. INTEGRATION WITH YOUR EXISTING CODE
# =============================================================================

def add_voice_to_existing_analysis():
    """Example of adding voice to existing wound analysis."""
    
    # Your existing analysis code here...
    wound_area = 25.5  # Your calculated wound area
    condition = "Medium wound - stable"  # Your condition assessment
    
    # Add voice generation
    voice_service = get_voice_service()
    if voice_service and voice_service.is_available():
        # Determine healing potential from your condition
        if "healing well" in condition.lower():
            healing_potential = "Good"
        elif "stable" in condition.lower():
            healing_potential = "Fair"
        else:
            healing_potential = "Poor"
        
        # Generate voices
        patient_audio, clinician_audio = generate_wound_voice(
            voice_service, wound_area, healing_potential
        )
        
        print(f"Patient voice: {patient_audio}")
        print(f"Clinician voice: {clinician_audio}")
    
    return wound_area, condition

# =============================================================================
# 6. ERROR HANDLING
# =============================================================================

def safe_voice_generation(voice_service, analysis_result):
    """Safely generate voice with error handling."""
    try:
        if not voice_service or not voice_service.is_available():
            print("⚠️ Voice service not available - skipping voice generation")
            return None
        
        audio = voice_service.generate_summary(
            analysis_result, 
            audience="patient", 
            filename="safe_voice.mp3"
        )
        
        if audio:
            print(f"✅ Voice generated: {audio}")
            return audio
        else:
            print("❌ Voice generation failed")
            return None
            
    except Exception as e:
        print(f"❌ Voice generation error: {e}")
        return None

# =============================================================================
# 7. COMPLETE EXAMPLE
# =============================================================================

def complete_example():
    """Complete example showing all voice service features."""
    print("🎤 Complete Voice Service Example")
    print("=" * 40)
    
    # Setup
    voice_service = setup_voice_service()
    if not voice_service:
        return
    
    # Example 1: Wound analysis voice
    print("\n1️⃣ Wound Analysis Voice:")
    patient_audio, clinician_audio = generate_wound_voice(voice_service, 18.5, "Good")
    
    # Example 2: Custom message
    print("\n2️⃣ Custom Message:")
    custom_audio = create_custom_voice(
        voice_service, 
        "Thank you for using our wound analysis system!",
        "thank_you.mp3"
    )
    
    # Example 3: Multilingual
    print("\n3️⃣ Multilingual Messages:")
    multilingual_voices = create_multilingual_voices(voice_service)
    
    # Example 4: Service info
    print("\n4️⃣ Service Information:")
    info = voice_service.get_voice_info()
    print(f"   Available: {info['available']}")
    print(f"   gTTS: {info['gtts_available']}")
    
    print("\n✅ Complete example finished!")
    
    # Clean up (optional)
    cleanup = input("\n🗑️ Clean up generated files? (y/n): ").lower().strip()
    if cleanup == 'y':
        voice_service.storage_service.delete_file("voice", "patient_summary.mp3")
        voice_service.storage_service.delete_file("voice", "clinician_summary.mp3")
        voice_service.storage_service.delete_file("voice", "thank_you.mp3")
        for lang in ['en', 'es', 'fr', 'de']:
            voice_service.storage_service.delete_file("voice", f"message_{lang}.mp3")
        print("✅ Files cleaned up")

# =============================================================================
# 8. QUICK COPY-PASTE SNIPPETS
# =============================================================================

QUICK_SNIPPETS = {
    "basic_setup": '''
# Basic voice service setup
from woundseg.services import get_voice_service
voice_service = get_voice_service()
if voice_service.is_available():
    print("Voice service ready")
''',
    
    "wound_voice": '''
# Generate wound analysis voice
analysis_result = {
    'area_mm2': 25.5,
    'healing_potential': 'Good',
    'confidence_score': 0.85,
    'severity': 'Mild',
    'timestamp': '2025-09-21'
}
audio = voice_service.generate_summary(analysis_result, "patient", "voice.mp3")
''',
    
    "custom_voice": '''
# Generate custom voice
audio = voice_service.generate_custom_voice(
    text="Your custom message here",
    filename="custom.mp3",
    lang="en"
)
''',
    
    "multilingual": '''
# Generate voice in different languages
audio = voice_service.generate_custom_voice(
    text="Your message",
    filename="message.mp3",
    lang="es"  # Spanish: es, French: fr, German: de, etc.
)
''',
    
    "error_handling": '''
# Safe voice generation with error handling
try:
    if voice_service.is_available():
        audio = voice_service.generate_summary(analysis_result, "patient", "voice.mp3")
        if audio:
            print(f"Voice generated: {audio}")
except Exception as e:
    print(f"Voice generation failed: {e}")
'''
}

def show_quick_snippets():
    """Show quick copy-paste snippets."""
    print("\n📋 QUICK COPY-PASTE SNIPPETS:")
    print("=" * 50)
    
    for name, snippet in QUICK_SNIPPETS.items():
        print(f"\n🔹 {name.upper().replace('_', ' ')}:")
        print(snippet)

if __name__ == "__main__":
    print("🎤 Voice Service Quick Reference")
    print("=" * 50)
    
    # Run complete example
    complete_example()
    
    # Show quick snippets
    show_quick_snippets()
    
    print("\n💡 USAGE TIPS:")
    print("1. Enable voice service: ENABLE_VOICE_SUMMARY=true")
    print("2. Check availability: voice_service.is_available()")
    print("3. Use patient/clinician audiences for different content")
    print("4. Handle errors gracefully with try/except")
    print("5. Clean up files when done (optional)")