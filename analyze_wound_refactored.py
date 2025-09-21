#!/usr/bin/env python3
"""
Refactored analyze_wound.py using woundseg package APIs.
This is a clean Flask web interface that leverages the modular woundseg package.
"""

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import uuid
from datetime import datetime
from typing import Dict, Any

# Import the modular woundseg package
from woundseg.pipelines import AnalysisPipeline
from woundseg.services import get_storage_service, get_voice_service, get_reporting_service
from woundseg.types import AnalysisOptions, Patient
from woundseg.config import Config

# Flask setup
app = Flask(__name__)
CORS(app)

# Use centralized configuration
UPLOAD_FOLDER = Config.OUTPUT_DIR / "uploads"
REPORT_FOLDER = Config.OUTPUT_DIR / "reports"

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize services
storage_service = get_storage_service()
voice_service = get_voice_service()
reporting_service = get_reporting_service()

def analyze_wound_with_package_api(
    image_path: str, 
    patient_name: str = "Unknown", 
    patient_age: int = None,
    use_medsam: bool = False
) -> Dict[str, Any]:
    """
    Analyze wound using the modular woundseg package API.
    
    Args:
        image_path: Path to the wound image
        patient_name: Patient's name
        patient_age: Patient's age
        use_medsam: Whether to use MedSAM model
    
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
            generate_voice_summary=True,
            generate_heatmap=True
        )
        
        # Use the analysis pipeline
        pipeline = AnalysisPipeline()
        analysis_result = pipeline.analyze_single_image(image_path, options)
        
        # Generate reports using services
        patient_pdf_path = reporting_service.generate_patient_report(analysis_result)
        clinician_pdf_path = reporting_service.generate_clinician_report(analysis_result)
        
        # Generate voice summaries
        analysis_dict = {
            'area_mm2': analysis_result.area_mm2,
            'healing_potential': analysis_result.healing_potential,
            'confidence_score': analysis_result.confidence_score,
            'severity': analysis_result.severity,
            'patient': patient,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        patient_audio_path = voice_service.generate_summary(
            analysis_dict, 
            audience="patient", 
            filename=f"patient_voice_{analysis_dict['timestamp']}.mp3"
        )
        
        clinician_audio_path = voice_service.generate_summary(
            analysis_dict, 
            audience="clinician", 
            filename=f"clinician_voice_{analysis_dict['timestamp']}.mp3"
        )
        
        # Save visualization using storage service
        session_id = analysis_dict['timestamp']
        if hasattr(analysis_result, 'overlay') and analysis_result.overlay is not None:
            vis_path = storage_service.save_image(
                analysis_result.overlay, 
                'visualizations', 
                f"wound_analysis_{session_id}.png"
            )
        else:
            vis_path = None
        
        return {
            "success": True,
            "analysis_result": {
                "severity": analysis_result.severity,
                "healing_potential": analysis_result.healing_potential,
                "area_mm2": analysis_result.area_mm2,
                "confidence_score": analysis_result.confidence_score
            },
            "files": {
                "patient_report": str(patient_pdf_path),
                "clinician_report": str(clinician_pdf_path),
                "patient_audio": str(patient_audio_path) if patient_audio_path else None,
                "clinician_audio": str(clinician_audio_path) if clinician_audio_path else None,
                "visualization": str(vis_path) if vis_path else None
            },
            "patient_info": {
                "name": patient_name,
                "age": patient_age
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "analysis_result": None,
            "files": None,
            "patient_info": {"name": patient_name, "age": patient_age}
        }

@app.route("/upload", methods=["POST"])
def upload() -> Dict[str, Any]:
    """Handle image upload and analysis using package API."""
    try:
        # Validate request
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Get form data
        patient_name = request.form.get("name", "Unknown")
        patient_age_str = request.form.get("age", "")
        use_medsam = request.form.get("use_medsam", "false").lower() == "true"
        
        # Parse age
        patient_age = None
        if patient_age_str and patient_age_str.isdigit():
            patient_age = int(patient_age_str)
        
        # Save uploaded file
        filename = f"{uuid.uuid4()}.png"
        image_path = UPLOAD_FOLDER / filename
        file.save(str(image_path))
        
        # Analyze using package API
        result = analyze_wound_with_package_api(
            image_path=str(image_path),
            patient_name=patient_name,
            patient_age=patient_age,
            use_medsam=use_medsam
        )
        
        if result["success"]:
            # Return success response with file URLs
            response_data = {
                "success": True,
                "message": "Analysis completed successfully",
                "analysis": result["analysis_result"],
                "patient": result["patient_info"],
                "files": {
                    "patient_report": f"/report/{os.path.basename(result['files']['patient_report'])}",
                    "visualization": f"/report/{os.path.basename(result['files']['visualization'])}" if result['files']['visualization'] else None
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
    """Serve generated reports and visualizations."""
    return send_from_directory(str(REPORT_FOLDER), filename)

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
        }
    })

if __name__ == "__main__":
    print("🚀 Starting Wound Analysis Web Interface")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Report folder: {REPORT_FOLDER}")
    print(f"🎤 Voice service: {'Enabled' if Config.ENABLE_VOICE_SUMMARY else 'Disabled'}")
    print(f"🤖 MedSAM available: {'Yes' if Config.ENABLE_MEDSAM else 'No'}")
    
    app.run(debug=True, host="0.0.0.0", port=5000)