#!/usr/bin/env python3
"""
Check where patient reports, doctor reports, and audio files are saved.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def check_storage_locations():
    """Check where different types of files are saved."""
    print("📁 File Storage Locations")
    print("=" * 60)
    
    try:
        from woundseg.services import get_storage_service, get_reporting_service, get_voice_service
        from woundseg.types import Patient, AnalysisResult
        
        # Get services
        storage_service = get_storage_service()
        reporting_service = get_reporting_service()
        voice_service = get_voice_service()
        
        # Show base directory
        print(f"🏠 Base Directory: {storage_service.base_dir}")
        print(f"   Absolute path: {storage_service.base_dir.absolute()}")
        print()
        
        # Show all storage directories
        print("📂 Storage Directory Structure:")
        for name, path in storage_service.dirs.items():
            print(f"   {name:15} → {path}")
        print()
        
        # Test file creation to show exact paths
        print("🧪 Testing File Creation:")
        
        # 1. Test PDF report creation
        print("\n1️⃣ PDF Reports:")
        try:
            # Create mock data
            patient = Patient(
                name="Test Patient",
                age=45,
                gender="Female",
                patient_id="TEST001"
            )
            
            analysis_result = AnalysisResult(
                area_mm2=25.5,
                perimeter_mm=18.2,
                healing_potential="Good",
                confidence_score=0.88,
                severity="Mild",
                shape_irregularity="Regular",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Generate patient report
            patient_report = reporting_service.generate_patient_report(
                patient, analysis_result, "test_patient_report.pdf"
            )
            print(f"   Patient Report: {patient_report}")
            
            # Generate clinician report
            clinician_report = reporting_service.generate_clinician_report(
                patient, analysis_result, "test_clinician_report.pdf"
            )
            print(f"   Clinician Report: {clinician_report}")
            
        except Exception as e:
            print(f"   ❌ PDF report test failed: {e}")
        
        # 2. Test voice file creation
        print("\n2️⃣ Voice Files:")
        try:
            if voice_service.is_available():
                # Generate patient voice
                patient_voice = voice_service.generate_summary(
                    {
                        'area_mm2': 25.5,
                        'healing_potential': 'Good',
                        'confidence_score': 0.88,
                        'severity': 'Mild',
                        'timestamp': '2025-09-21'
                    },
                    audience="patient",
                    filename="test_patient_voice.mp3"
                )
                print(f"   Patient Voice: {patient_voice}")
                
                # Generate clinician voice
                clinician_voice = voice_service.generate_summary(
                    {
                        'area_mm2': 25.5,
                        'healing_potential': 'Good',
                        'confidence_score': 0.88,
                        'severity': 'Mild',
                        'timestamp': '2025-09-21'
                    },
                    audience="clinician",
                    filename="test_clinician_voice.mp3"
                )
                print(f"   Clinician Voice: {clinician_voice}")
            else:
                print("   ⚠️ Voice service not available (ENABLE_VOICE_SUMMARY=false)")
                
        except Exception as e:
            print(f"   ❌ Voice file test failed: {e}")
        
        # 3. Test other file types
        print("\n3️⃣ Other File Types:")
        try:
            # Test image save
            test_image_data = b"fake_image_data"
            image_path = storage_service.save_image(
                test_image_data, "masks", "test_mask.png"
            )
            print(f"   Mask Image: {image_path}")
            
            # Test text save
            text_path = storage_service.save_text(
                "Test analysis data", "reports", "test_analysis.txt"
            )
            print(f"   Analysis Text: {text_path}")
            
        except Exception as e:
            print(f"   ❌ Other file test failed: {e}")
        
        # 4. Show actual directory contents
        print("\n4️⃣ Current Directory Contents:")
        for name, path in storage_service.dirs.items():
            if path.exists():
                files = list(path.glob("*"))
                if files:
                    print(f"   {name}:")
                    for file in files[:5]:  # Show first 5 files
                        print(f"      - {file.name}")
                    if len(files) > 5:
                        print(f"      ... and {len(files) - 5} more files")
                else:
                    print(f"   {name}: (empty)")
            else:
                print(f"   {name}: (does not exist)")
        
        # 5. Show file sizes
        print("\n5️⃣ Directory Sizes:")
        for name, path in storage_service.dirs.items():
            if path.exists():
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print(f"   {name:15}: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking storage locations: {e}")
        return False

def show_quick_paths():
    """Show quick reference for file paths."""
    print("\n" + "=" * 60)
    print("📋 Quick Reference - File Locations")
    print("=" * 60)
    
    try:
        from woundseg.config import Config
        
        base_dir = Path(Config.OUTPUT_DIR)
        
        print(f"🏠 Base Directory: {base_dir.absolute()}")
        print()
        print("📂 File Types and Locations:")
        print(f"   Patient Reports    → {base_dir / 'reports'}")
        print(f"   Doctor Reports     → {base_dir / 'reports'}")
        print(f"   Voice Files        → {base_dir / 'voice_summaries'}")
        print(f"   Mask Images        → {base_dir / 'masks'}")
        print(f"   Visualizations     → {base_dir / 'visualizations'}")
        print(f"   Uploaded Images    → {base_dir / 'uploads'}")
        print(f"   Temporary Files    → {base_dir / 'temp'}")
        print()
        print("💡 To access files:")
        print(f"   cd {base_dir.absolute()}")
        print(f"   ls -la reports/")
        print(f"   ls -la voice_summaries/")
        
    except Exception as e:
        print(f"❌ Error showing quick paths: {e}")

def main():
    """Main function."""
    print("🔍 Checking File Storage Locations")
    print("=" * 60)
    
    # Check storage locations
    success = check_storage_locations()
    
    # Show quick reference
    show_quick_paths()
    
    if success:
        print("\n✅ File location check completed!")
    else:
        print("\n❌ File location check failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)