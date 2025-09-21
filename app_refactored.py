#!/usr/bin/env python3
"""
Refactored app.py using woundseg package APIs.
This is a simplified Flask web interface that leverages the modular woundseg package.
"""

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple

# Import the modular woundseg package
from woundseg.pipelines import AnalysisPipeline
from woundseg.services import get_storage_service, get_voice_service, get_reporting_service
from woundseg.types import AnalysisOptions, Patient
from woundseg.config import Config

# Flask setup
app = Flask(__name__, template_folder='.')
CORS(app)

# Use centralized configuration
UPLOAD_FOLDER = Config.OUTPUT_DIR / "uploads"
REPORT_FOLDER = Config.OUTPUT_DIR / "reports"
VOICE_FOLDER = Config.OUTPUT_DIR / "voice_summaries"

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
VOICE_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize services
storage_service = get_storage_service()
voice_service = get_voice_service()
reporting_service = get_reporting_service()

def analyze_wound_simple(
    image_path: str, 
    patient_name: str = "Unknown", 
    patient_age: int = None,
    use_medsam: bool = False,
    generate_voice: bool = False
) -> Dict[str, Any]:
    """
    Simple wound analysis using the modular woundseg package API.
    
    Args:
        image_path: Path to the wound image
        patient_name: Patient's name
        patient_age: Patient's age
        use_medsam: Whether to use MedSAM model
        generate_voice: Whether to generate voice summary
    
    Returns:
        Dictionary with analysis results and file paths
    """
    try:
        # Create patient object
        patient = Patient(name=patient_name, age=patient_age)
        
        # Create analysis options
        options = AnalysisOptions(
            patient=patient,
            use_medsam=use_medsam,
            generate_report=True,
            generate_voice_summary=generate_voice
        )
        
        # Use the analysis pipeline
        pipeline = AnalysisPipeline()
        analysis_result = pipeline.analyze_single_image(image_path, options)
        
        # Generate PDF report
        report_path = reporting_service.generate_patient_report(analysis_result)
        
        # Generate voice summary if requested
        voice_path = None
        if generate_voice:
            analysis_dict = {
                'area_mm2': analysis_result.area_mm2,
                'healing_potential': analysis_result.healing_potential,
                'confidence_score': analysis_result.confidence_score,
                'severity': analysis_result.severity,
                'patient': patient,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            voice_path = voice_service.generate_summary(
                analysis_dict, 
                audience="patient", 
                filename=f"voice_summary_{analysis_dict['timestamp']}.mp3"
            )
        
        return {
            "success": True,
            "severity": analysis_result.severity,
            "healing_potential": analysis_result.healing_potential,
            "wound_area_mm2": analysis_result.area_mm2,
            "confidence_score": analysis_result.confidence_score,
            "report_path": str(report_path),
            "voice_path": str(voice_path) if voice_path else None,
            "patient_info": {
                "name": patient_name,
                "age": patient_age
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "severity": None,
            "healing_potential": None,
            "wound_area_mm2": None,
            "confidence_score": None,
            "report_path": None,
            "voice_path": None,
            "patient_info": {"name": patient_name, "age": patient_age}
        }

@app.route("/upload", methods=["POST"])
def upload() -> Tuple[Dict[str, Any], int]:
    """Handle image upload and analysis using package API."""
    try:
        # Check if image was uploaded
        if "image" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Get form data
        patient_name = request.form.get("name", "Unknown")
        patient_age_str = request.form.get("age", "")
        use_medsam = request.form.get("use_medsam", "false").lower() == "true"
        voice_summary = request.form.get("voice_summary", "false").lower() == "true"
        
        # Parse age
        patient_age = None
        if patient_age_str and patient_age_str.isdigit():
            patient_age = int(patient_age_str)

        # Save uploaded file
        filename = secure_filename(file.filename)
        if not filename:
            filename = f"{uuid.uuid4()}.png"
        
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        # Analyze using package API
        result = analyze_wound_simple(
            image_path=str(filepath),
            patient_name=patient_name,
            patient_age=patient_age,
            use_medsam=use_medsam,
            generate_voice=voice_summary
        )
        
        if result["success"]:
            # Return success response
            response_data = {
                "success": True,
                "message": "Analysis completed successfully",
                "analysis": {
                    "severity": result["severity"],
                    "healing_potential": result["healing_potential"],
                    "wound_area_mm2": result["wound_area_mm2"],
                    "confidence_score": result["confidence_score"]
                },
                "patient": result["patient_info"],
                "files": {
                    "report": f"/report/{os.path.basename(result['report_path'])}",
                    "voice": f"/voice/{os.path.basename(result['voice_path'])}" if result['voice_path'] else None
                }
            }
            return jsonify(response_data)
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route("/report/<filename>")
def serve_report(filename: str):
    """Serve generated reports."""
    return send_from_directory(str(REPORT_FOLDER), filename)

@app.route("/voice/<filename>")
def serve_voice(filename: str):
    """Serve generated voice summaries."""
    return send_from_directory(str(VOICE_FOLDER), filename)

@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("wound-wisperer.html")

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "package_version": "1.0.0",
        "services": {
            "storage": "available",
            "voice": "available" if Config.ENABLE_VOICE_SUMMARY else "disabled",
            "reporting": "available"
        },
        "models": {
            "unet": "available",
            "medsam": "available" if Config.ENABLE_MEDSAM else "disabled"
        }
    })

if __name__ == "__main__":
    from woundseg.logging import setup_development_logging, get_logger
    
    # Setup logging
    setup_development_logging()
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting Simple Wound Analysis Web Interface")
    logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"📁 Report folder: {REPORT_FOLDER}")
    logger.info(f"🎤 Voice service: {'Enabled' if Config.ENABLE_VOICE_SUMMARY else 'Disabled'}")
    logger.info(f"🤖 MedSAM available: {'Yes' if Config.ENABLE_MEDSAM else 'No'}")
    
    app.run(debug=True, host="0.0.0.0", port=5001)