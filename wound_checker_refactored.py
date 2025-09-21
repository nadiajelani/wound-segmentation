#!/usr/bin/env python3
"""
Refactored wound_checker.py using woundseg package APIs.
This is a clean Tkinter GUI that leverages the modular woundseg package.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from datetime import datetime
from typing import Optional, Dict, Any

# Import the modular woundseg package
from woundseg.pipelines import AnalysisPipeline
from woundseg.services import get_storage_service, get_voice_service, get_reporting_service
from woundseg.types import AnalysisOptions, Patient
from woundseg.config import Config

class WoundCheckerGUI:
    """Modern Tkinter GUI for wound analysis using the woundseg package."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Wound Analysis Tool - Professional Edition")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize services
        self.storage_service = get_storage_service()
        self.voice_service = get_voice_service()
        self.reporting_service = get_reporting_service()
        
        # Analysis pipeline
        self.pipeline = AnalysisPipeline()
        
        # GUI variables
        self.image_path = tk.StringVar()
        self.patient_name = tk.StringVar(value="Unknown")
        self.patient_age = tk.StringVar()
        self.use_medsam = tk.BooleanVar(value=False)
        self.generate_voice = tk.BooleanVar(value=True)
        self.generate_report = tk.BooleanVar(value=True)
        
        # Results storage
        self.last_analysis_result = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Wound Analysis Tool", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Image selection
        ttk.Label(main_frame, text="Wound Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.image_path, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 5), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_image).grid(
            row=1, column=2, padx=(5, 0), pady=5)
        
        # Patient information
        ttk.Label(main_frame, text="Patient Name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.patient_name, width=30).grid(
            row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Label(main_frame, text="Patient Age:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.patient_age, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding="10")
        options_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=20)
        options_frame.columnconfigure(0, weight=1)
        
        ttk.Checkbutton(options_frame, text="Use MedSAM Model (Advanced)", 
                       variable=self.use_medsam).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Generate Voice Summary", 
                       variable=self.generate_voice).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Generate PDF Report", 
                       variable=self.generate_report).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Analyze button
        self.analyze_button = ttk.Button(main_frame, text="Analyze Wound", 
                                        command=self.analyze_wound, style='Accent.TButton')
        self.analyze_button.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        self.results_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        self.results_frame.columnconfigure(1, weight=1)
        
        # Configure main frame row weight
        main_frame.rowconfigure(7, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def browse_image(self):
        """Browse for wound image file."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Wound Image",
            filetypes=filetypes
        )
        if filename:
            self.image_path.set(filename)
            self.status_var.set(f"Selected: {os.path.basename(filename)}")
    
    def analyze_wound(self):
        """Analyze the wound using the package API."""
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select a wound image first.")
            return
        
        if not os.path.exists(self.image_path.get()):
            messagebox.showerror("Error", "Selected image file does not exist.")
            return
        
        # Disable analyze button and start progress
        self.analyze_button.config(state='disabled')
        self.progress.start()
        self.status_var.set("Analyzing wound...")
        
        # Clear previous results
        self.clear_results()
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self._analyze_wound_thread)
        thread.daemon = True
        thread.start()
    
    def _analyze_wound_thread(self):
        """Run wound analysis in background thread."""
        try:
            # Get patient age
            patient_age = None
            if self.patient_age.get() and self.patient_age.get().isdigit():
                patient_age = int(self.patient_age.get())
            
            # Create patient object
            patient = Patient(
                name=self.patient_name.get() or "Unknown",
                age=patient_age
            )
            
            # Create analysis options
            options = AnalysisOptions(
                patient=patient,
                use_medsam=self.use_medsam.get(),
                generate_report=self.generate_report.get(),
                generate_voice_summary=self.generate_voice.get()
            )
            
            # Run analysis
            analysis_result = self.pipeline.analyze_single_image(
                self.image_path.get(), 
                options
            )
            
            # Store result
            self.last_analysis_result = analysis_result
            
            # Update UI in main thread
            self.root.after(0, self._update_results_ui, analysis_result)
            
        except Exception as e:
            # Update UI with error in main thread
            self.root.after(0, self._show_error, str(e))
    
    def _update_results_ui(self, analysis_result):
        """Update the UI with analysis results."""
        try:
            # Display results
            results = [
                ("Severity:", analysis_result.severity),
                ("Healing Potential:", analysis_result.healing_potential),
                ("Wound Area:", f"{analysis_result.area_mm2:.2f} mm²"),
                ("Confidence Score:", f"{analysis_result.confidence_score:.2f}")
            ]
            
            for i, (label, value) in enumerate(results):
                ttk.Label(self.results_frame, text=label, font=('Arial', 10, 'bold')).grid(
                    row=i, column=0, sticky=tk.W, pady=2)
                ttk.Label(self.results_frame, text=str(value)).grid(
                    row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            
            # Add action buttons
            button_frame = ttk.Frame(self.results_frame)
            button_frame.grid(row=len(results), column=0, columnspan=2, pady=10)
            
            if self.generate_report.get():
                ttk.Button(button_frame, text="Open Report", 
                          command=self.open_report).pack(side=tk.LEFT, padx=5)
            
            if self.generate_voice.get():
                ttk.Button(button_frame, text="Play Voice Summary", 
                          command=self.play_voice).pack(side=tk.LEFT, padx=5)
            
            ttk.Button(button_frame, text="Open Output Folder", 
                      command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
            
            self.status_var.set("Analysis completed successfully!")
            
        except Exception as e:
            self._show_error(f"Error displaying results: {str(e)}")
        finally:
            # Re-enable analyze button and stop progress
            self.analyze_button.config(state='normal')
            self.progress.stop()
    
    def _show_error(self, error_message):
        """Show error message and reset UI."""
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n\n{error_message}")
        self.status_var.set("Analysis failed")
        self.analyze_button.config(state='normal')
        self.progress.stop()
    
    def clear_results(self):
        """Clear previous analysis results."""
        for widget in self.results_frame.winfo_children():
            widget.destroy()
    
    def open_report(self):
        """Open the generated PDF report."""
        if self.last_analysis_result:
            try:
                # Generate report if not already generated
                report_path = self.reporting_service.generate_patient_report(self.last_analysis_result)
                
                # Open with default application
                import subprocess
                import platform
                
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', str(report_path)])
                elif platform.system() == 'Windows':
                    os.startfile(str(report_path))
                else:  # Linux
                    subprocess.run(['xdg-open', str(report_path)])
                    
            except Exception as e:
                messagebox.showerror("Error", f"Could not open report: {str(e)}")
    
    def play_voice(self):
        """Play the generated voice summary."""
        if self.last_analysis_result:
            try:
                # Generate voice summary
                analysis_dict = {
                    'area_mm2': self.last_analysis_result.area_mm2,
                    'healing_potential': self.last_analysis_result.healing_potential,
                    'confidence_score': self.last_analysis_result.confidence_score,
                    'severity': self.last_analysis_result.severity,
                    'patient': self.last_analysis_result.patient,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                
                voice_path = self.voice_service.generate_summary(
                    analysis_dict, 
                    audience="patient", 
                    filename=f"voice_summary_{analysis_dict['timestamp']}.mp3"
                )
                
                # Play audio
                import subprocess
                import platform
                
                if platform.system() == 'Darwin':  # macOS
                    subprocess.run(['afplay', str(voice_path)])
                elif platform.system() == 'Windows':
                    subprocess.run(['start', str(voice_path)], shell=True)
                else:  # Linux
                    subprocess.run(['mpg123', str(voice_path)])
                    
            except Exception as e:
                messagebox.showerror("Error", f"Could not play voice summary: {str(e)}")
    
    def open_output_folder(self):
        """Open the output folder in file explorer."""
        try:
            import subprocess
            import platform
            
            output_dir = Config.OUTPUT_DIR
            
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(output_dir)])
            elif platform.system() == 'Windows':
                subprocess.run(['explorer', str(output_dir)])
            else:  # Linux
                subprocess.run(['xdg-open', str(output_dir)])
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder: {str(e)}")

def main():
    """Main function to run the GUI application."""
    print("🚀 Starting Wound Analysis GUI")
    print(f"📁 Output directory: {Config.OUTPUT_DIR}")
    print(f"🎤 Voice service: {'Enabled' if Config.ENABLE_VOICE_SUMMARY else 'Disabled'}")
    print(f"🤖 MedSAM available: {'Yes' if Config.ENABLE_MEDSAM else 'No'}")
    
    root = tk.Tk()
    app = WoundCheckerGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()