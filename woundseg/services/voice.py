"""
Voice service for text-to-speech functionality.

This module provides a wrapper around gTTS (Google Text-to-Speech)
for generating voice summaries of wound analysis results.
"""

import logging
import io
from typing import Optional, Union, Dict, Any
from pathlib import Path

from ..config import Config
from .storage import get_storage_service

logger = logging.getLogger(__name__)


class VoiceService:
    """
    Voice service for text-to-speech functionality.
    
    Provides a clean interface for generating voice summaries
    of wound analysis results using gTTS.
    """
    
    def __init__(self, storage_service=None):
        """
        Initialize the voice service.
        
        Args:
            storage_service: Storage service instance (optional)
        """
        self.storage_service = storage_service or get_storage_service()
        self.enabled = Config.ENABLE_VOICE_SUMMARY
        
        if self.enabled:
            try:
                from gtts import gTTS
                self.gtts = gTTS
                logger.info("VoiceService initialized with gTTS support")
            except ImportError:
                logger.warning("gTTS not available - voice features disabled")
                self.enabled = False
                self.gtts = None
        else:
            logger.info("VoiceService initialized (disabled by config)")
            self.gtts = None
    
    def is_available(self) -> bool:
        """
        Check if voice service is available.
        
        Returns:
            bool: True if voice service is available and enabled
        """
        return self.enabled and self.gtts is not None
    
    def generate_summary(self, analysis_result: Dict[str, Any], 
                        audience: str = "patient", 
                        filename: Optional[str] = None) -> Optional[Path]:
        """
        Generate voice summary from analysis result.
        
        Args:
            analysis_result: Analysis result dictionary
            audience: Target audience ("patient" or "clinician")
            filename: Optional filename for the audio file
            
        Returns:
            Path to saved audio file, or None if generation failed
        """
        if not self.is_available():
            logger.warning("Voice service not available - skipping voice generation")
            return None
        
        try:
            # Generate text based on audience
            text = self._generate_text(analysis_result, audience)
            
            # Create filename if not provided
            if filename is None:
                timestamp = analysis_result.get('timestamp', 'unknown')
                filename = f"wound_analysis_{audience}_{timestamp}.mp3"
            
            # Generate audio
            audio_path = self._text_to_speech(text, filename)
            
            logger.info(f"Voice summary generated for {audience}: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Failed to generate voice summary: {e}")
            return None
    
    def _generate_text(self, analysis_result: Dict[str, Any], audience: str) -> str:
        """
        Generate text content based on analysis result and audience.
        
        Args:
            analysis_result: Analysis result dictionary
            audience: Target audience ("patient" or "clinician")
            
        Returns:
            str: Generated text content
        """
        if audience == "patient":
            return self._generate_patient_text(analysis_result)
        elif audience == "clinician":
            return self._generate_clinician_text(analysis_result)
        else:
            raise ValueError(f"Invalid audience: {audience}. Must be 'patient' or 'clinician'")
    
    def _generate_patient_text(self, analysis_result: Dict[str, Any]) -> str:
        """Generate patient-friendly text."""
        # Extract patient information
        patient = analysis_result.get('patient', {})
        patient_name = patient.get('name', 'Patient') if isinstance(patient, dict) else getattr(patient, 'name', 'Patient')
        patient_age = patient.get('age', None) if isinstance(patient, dict) else getattr(patient, 'age', None)
        
        # Extract key information
        wound_area = analysis_result.get('area_mm2', 0)
        healing_trend = analysis_result.get('healing_potential', 'unknown')
        confidence = analysis_result.get('confidence_score', 0)
        
        # Create personalized greeting
        if patient_name and patient_name != 'Unknown':
            greeting = f"Hello {patient_name}."
            if patient_age:
                greeting += f" You are {patient_age} years old."
        else:
            greeting = "Hello."
        
        # Convert to patient-friendly language
        if wound_area > 0:
            area_text = f"Your wound area is {wound_area:.1f} square millimeters."
        else:
            area_text = "We've analyzed your wound area."
        
        if healing_trend == 'Good':
            trend_text = "Great news! Your wound is showing positive healing progress."
        elif healing_trend == 'Fair':
            trend_text = "Your wound appears to be stable and healing normally."
        elif healing_trend == 'Poor':
            trend_text = "We've detected some changes in your wound that may need attention."
        else:
            trend_text = "We've completed the analysis of your wound."
        
        confidence_text = f"The analysis confidence is {confidence:.0f} percent."
        
        return f"{greeting} {area_text} {trend_text} {confidence_text} Please continue following your treatment plan and consult your healthcare provider if you have any concerns."
    
    def _generate_clinician_text(self, analysis_result: Dict[str, Any]) -> str:
        """Generate clinician-focused text."""
        # Extract patient information
        patient = analysis_result.get('patient', {})
        patient_name = patient.get('name', 'Unknown') if isinstance(patient, dict) else getattr(patient, 'name', 'Unknown')
        patient_age = patient.get('age', None) if isinstance(patient, dict) else getattr(patient, 'age', None)
        
        # Extract detailed information
        wound_area = analysis_result.get('area_mm2', 0)
        healing_trend = analysis_result.get('healing_potential', 'unknown')
        confidence = analysis_result.get('confidence_score', 0)
        severity = analysis_result.get('severity', 'unknown')
        
        # Create patient identification
        if patient_name and patient_name != 'Unknown':
            patient_info = f"Patient: {patient_name}"
            if patient_age:
                patient_info += f", Age: {patient_age}"
        else:
            patient_info = "Patient: Unknown"
        
        text_parts = [
            f"Clinical wound analysis for {patient_info}.",
            f"Wound area: {wound_area:.2f} square millimeters.",
            f"Segmentation confidence: {confidence:.1f} percent.",
            f"Healing trend assessment: {healing_trend}.",
            f"Severity classification: {severity}."
        ]
        
        # Add recommendations based on analysis
        if healing_trend == 'Good':
            text_parts.append("Recommendation: Continue current treatment protocol.")
        elif healing_trend == 'Poor':
            text_parts.append("Recommendation: Consider treatment adjustment or closer monitoring.")
        else:
            text_parts.append("Recommendation: Maintain current treatment and monitor progress.")
        
        return " ".join(text_parts)
    
    def _text_to_speech(self, text: str, filename: str) -> Path:
        """
        Convert text to speech and save audio file.
        
        Args:
            text: Text to convert to speech
            filename: Filename for the audio file
            
        Returns:
            Path: Path to saved audio file
        """
        try:
            # Create gTTS object
            tts = self.gtts(text=text, lang='en', slow=False)
            
            # Generate audio data
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_data = audio_buffer.getvalue()
            
            # Save to storage
            audio_path = self.storage_service.save_audio(audio_data, filename)
            
            logger.debug(f"Audio generated: {len(audio_data)} bytes")
            return audio_path
            
        except Exception as e:
            logger.error(f"Failed to convert text to speech: {e}")
            raise
    
    def generate_custom_voice(self, text: str, filename: str, 
                            lang: str = 'en', slow: bool = False) -> Optional[Path]:
        """
        Generate custom voice from arbitrary text.
        
        Args:
            text: Text to convert to speech
            filename: Filename for the audio file
            lang: Language code (default: 'en')
            slow: Whether to speak slowly (default: False)
            
        Returns:
            Path to saved audio file, or None if generation failed
        """
        if not self.is_available():
            logger.warning("Voice service not available - skipping custom voice generation")
            return None
        
        try:
            # Create gTTS object with custom parameters
            tts = self.gtts(text=text, lang=lang, slow=slow)
            
            # Generate audio data
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_data = audio_buffer.getvalue()
            
            # Save to storage
            audio_path = self.storage_service.save_audio(audio_data, filename)
            
            logger.info(f"Custom voice generated: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Failed to generate custom voice: {e}")
            return None
    
    def get_voice_info(self) -> Dict[str, Any]:
        """
        Get information about the voice service.
        
        Returns:
            dict: Voice service information
        """
        return {
            'enabled': self.enabled,
            'available': self.is_available(),
            'gtts_available': self.gtts is not None,
            'storage_service': str(self.storage_service.base_dir)
        }


# Global voice service instance
_voice_service: Optional[VoiceService] = None


def get_voice_service() -> VoiceService:
    """
    Get the global voice service instance.
    
    Returns:
        VoiceService: The global voice service instance
    """
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service


def reset_voice_service() -> None:
    """Reset the global voice service (useful for testing)."""
    global _voice_service
    _voice_service = None
    logger.info("Voice service reset")