#!/usr/bin/env python3
"""
Wound Segmentation CLI.

Professional command-line interface for wound analysis and training.
"""

import sys
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable for voice service
os.environ["ENABLE_VOICE_SUMMARY"] = "true"

from woundseg.config import Config
from woundseg.services import get_storage_service, get_voice_service, get_reporting_service

# Initialize Typer app
app = typer.Typer(
    name="ws",
    help="Wound Segmentation CLI - Professional wound analysis and training",
    add_completion=False,
    rich_markup_mode="rich",
)

# Initialize Rich console
console = Console()

@app.command()
def analyze(
    image: Path = typer.Argument(..., help="Path to wound image file"),
    use_medsam: bool = typer.Option(False, "--use-medsam", help="Use MedSAM model instead of U-Net"),
    report: bool = typer.Option(True, "--report/--no-report", help="Generate PDF reports"),
    voice: bool = typer.Option(True, "--voice/--no-voice", help="Generate voice summaries"),
    name: Optional[str] = typer.Option(None, "--name", help="Patient name"),
    age: Optional[int] = typer.Option(None, "--age", help="Patient age"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Custom output directory"),
    confidence: float = typer.Option(0.5, "--confidence", help="Confidence threshold (0.0-1.0)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Analyze a wound image and generate comprehensive reports.
    
    This command performs complete wound analysis including:
    - Image segmentation using AI models
    - Wound metrics calculation (area, perimeter, condition)
    - PDF report generation (patient and clinician versions)
    - Voice summaries (patient and clinician audio)
    - Visualizations (masks, overlays, heatmaps, trends)
    """
    
    # Validate inputs
    if not image.exists():
        console.print(f"[red]Error: Image file not found: {image}[/red]")
        raise typer.Exit(1)
    
    if confidence < 0.0 or confidence > 1.0:
        console.print(f"[red]Error: Confidence must be between 0.0 and 1.0, got {confidence}[/red]")
        raise typer.Exit(1)
    
    if age is not None and (age < 0 or age > 150):
        console.print(f"[red]Error: Age must be between 0 and 150, got {age}[/red]")
        raise typer.Exit(1)
    
    # Display analysis info
    console.print(Panel.fit(
        f"[bold blue]Wound Analysis Starting[/bold blue]\n"
        f"Image: {image.name}\n"
        f"Model: {'MedSAM' if use_medsam else 'SIMCLR U-Net'}\n"
        f"Reports: {'Yes' if report else 'No'}\n"
        f"Voice: {'Yes' if voice else 'No'}\n"
        f"Patient: {name or 'Unknown'}\n"
        f"Age: {age or 'Not specified'}\n"
        f"Confidence: {confidence}",
        title="Analysis Configuration",
        border_style="blue"
    ))
    
    try:
        # Import analysis components
        from woundseg.pipelines import AnalysisPipeline
        from woundseg.types import Patient, AnalysisOptions
        from woundseg.models import get_model_provider
        import cv2
        import numpy as np
        from datetime import datetime
        
        # Initialize services
        storage_service = get_storage_service()
        voice_service = get_voice_service()
        reporting_service = get_reporting_service()
        
        # Create patient info
        patient = Patient(
            name=name or "Unknown",
            age=age,
            diabetes=None,
            additional_info={"image_source": image.name}
        )
        
        # Create analysis options
        analysis_options = AnalysisOptions(
            use_medsam=use_medsam,
            generate_report=report,
            generate_voice_summary=voice,
            generate_heatmap=True,
            confidence_threshold=confidence,
            patient=patient
        )
        
        # Load and preprocess image
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading image...", total=None)
            
            original_image = cv2.imread(str(image))
            if original_image is None:
                console.print(f"[red]Error: Could not read image file: {image}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="✅ Image loaded successfully")
        
        # Run analysis pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running wound analysis...", total=None)
            
            # Initialize analysis pipeline
            pipeline = AnalysisPipeline()
            
            # Run analysis
            analysis_result = pipeline.analyze_single_image(
                image,
                analysis_options
            )
            
            progress.update(task, description="✅ Analysis completed")
        
        # Generate reports and summaries
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if report:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating PDF reports...", total=None)
                
                try:
                    # Generate patient report (already saves to storage)
                    patient_pdf_path = reporting_service.generate_patient_report(analysis_result)
                    
                    # Generate clinician report (already saves to storage)
                    clinician_pdf_path = reporting_service.generate_clinician_report(analysis_result)
                    
                    progress.update(task, description="✅ PDF reports generated")
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: PDF generation failed: {e}[/yellow]")
        
        if voice and voice_service.is_available():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating voice summaries...", total=None)
                
                try:
                    # Generate voice summaries
                    analysis_dict = {
                        'area_mm2': analysis_result.area_mm2,
                        'healing_potential': analysis_result.healing_potential,
                        'confidence_score': analysis_result.confidence_score,
                        'severity': analysis_result.severity,
                        'patient': patient,  # Include patient information
                        'timestamp': session_id
                    }
                    
                    patient_audio = voice_service.generate_summary(
                        analysis_dict, 
                        audience="patient", 
                        filename=f"patient_voice_{session_id}.mp3"
                    )
                    
                    clinician_audio = voice_service.generate_summary(
                        analysis_dict, 
                        audience="clinician", 
                        filename=f"clinician_voice_{session_id}.mp3"
                    )
                    
                    progress.update(task, description="✅ Voice summaries generated")
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Voice generation failed: {e}[/yellow]")
        
        # Save analysis results
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Saving results...", total=None)
            
            # Create session directory
            session_dir = storage_service.create_session_dir(session_id)
            
            # Save original image
            _, original_encoded = cv2.imencode('.jpg', original_image)
            original_path = storage_service.save_image(
                original_encoded.tobytes(), 'uploads', f"original_{session_id}.jpg"
            )
            
            # Save mask
            _, mask_encoded = cv2.imencode('.png', analysis_result.mask)
            mask_path = storage_service.save_image(
                mask_encoded.tobytes(), 'masks', f"segmentation_mask_{session_id}.png"
            )
            
            # Save overlay if available
            if analysis_result.overlay is not None:
                _, overlay_encoded = cv2.imencode('.png', analysis_result.overlay)
                overlay_path = storage_service.save_image(
                    overlay_encoded.tobytes(), 'visualizations', f"overlay_{session_id}.png"
                )
            
            progress.update(task, description="✅ Results saved")
        
        # Display results summary
        results_table = Table(title="Analysis Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        results_table.add_column("Status", style="yellow")
        
        results_table.add_row("Wound Area", f"{analysis_result.area_mm2:.2f} mm²", "✅")
        results_table.add_row("Healing Potential", analysis_result.healing_potential, "✅")
        results_table.add_row("Severity", analysis_result.severity, "✅")
        results_table.add_row("Confidence", f"{analysis_result.confidence_score:.2f}", "✅")
        
        console.print(results_table)
        
        # Display file locations
        console.print(Panel.fit(
            f"[bold green]Analysis Complete![/bold green]\n\n"
            f"📁 Results saved to: {storage_service.base_dir}\n"
            f"📂 Session directory: {session_dir}\n"
            f"🖼️  Original image: {original_path}\n"
            f"🎭 Segmentation mask: {mask_path}\n"
            f"📄 Reports: outputs/reports/\n"
            f"🎤 Voice summaries: outputs/voice_summaries/",
            title="File Locations",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

@app.command()
def train_unet(
    data_dir: Path = typer.Argument(..., help="Path to training data directory"),
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(8, "--batch-size", help="Training batch size"),
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Model output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Train a U-Net model for wound segmentation.
    
    This command trains a new U-Net model using the provided training data.
    """
    console.print(f"[yellow]Training command not yet implemented[/yellow]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {learning_rate}")

@app.command()
def train_classifier(
    data_dir: Path = typer.Argument(..., help="Path to training data directory"),
    epochs: int = typer.Option(50, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(16, "--batch-size", help="Training batch size"),
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Model output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Train a classifier model for wound classification.
    
    This command trains a new classifier model using the provided training data.
    """
    console.print(f"[yellow]Training command not yet implemented[/yellow]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {learning_rate}")

@app.command()
def info():
    """
    Display system information and configuration.
    """
    console.print(Panel.fit(
        f"[bold blue]Wound Segmentation System[/bold blue]\n\n"
        f"Version: 1.0.0\n"
        f"Project Root: {Config.PROJECT_ROOT}\n"
        f"Models Directory: {Config.MODELS_DIR}\n"
        f"Output Directory: {Config.OUTPUT_DIR}\n"
        f"Device: {Config.DEVICE}\n"
        f"Mixed Precision: {Config.MIXED_PRECISION}\n\n"
        f"Feature Flags:\n"
        f"• MedSAM: {Config.ENABLE_MEDSAM}\n"
        f"• Voice Summary: {Config.ENABLE_VOICE_SUMMARY}\n"
        f"• Explainability: {Config.ENABLE_EXPLAINABILITY}\n"
        f"• Synthetic Data: {Config.ENABLE_SYNTHETIC_DATA}",
        title="System Information",
        border_style="blue"
    ))

@app.command()
def version():
    """
    Display version information.
    """
    console.print("Wound Segmentation CLI v1.0.0")

if __name__ == "__main__":
    app()