#!/usr/bin/env python3
"""
Simple Desktop GUI for Wound Segmentation using Local API
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import base64
import io
from PIL import Image, ImageTk
import json
import threading
import time

class WoundSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wound Segmentation - SimCLR U-Net")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # API Configuration
        self.api_url = "http://127.0.0.1:8000"
        self.model_url = "http://127.0.0.1:9100"
        
        # Variables
        self.selected_file = None
        self.analysis_result = None
        
        self.setup_ui()
        self.check_api_status()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🏥 Wound Segmentation Analysis", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Status Frame
        status_frame = tk.Frame(self.root, bg='#f0f0f0')
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="🔍 Checking API status...", 
                                   font=('Arial', 10), fg='#e74c3c', bg='#f0f0f0')
        self.status_label.pack(side='left')
        
        # Main Content Frame
        content_frame = tk.Frame(self.root, bg='#f0f0f0')
        content_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left Panel - File Selection
        left_panel = tk.Frame(content_frame, bg='#ecf0f1', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # File Selection
        file_frame = tk.LabelFrame(left_panel, text="📁 Select Image", font=('Arial', 12, 'bold'), 
                                  bg='#ecf0f1', fg='#2c3e50')
        file_frame.pack(fill='x', padx=10, pady=10)
        
        self.file_label = tk.Label(file_frame, text="No file selected", 
                                  font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d')
        self.file_label.pack(pady=5)
        
        select_btn = tk.Button(file_frame, text="📂 Choose Image", command=self.select_file,
                              font=('Arial', 10, 'bold'), bg='#3498db', fg='white',
                              relief='raised', bd=2, padx=20, pady=5)
        select_btn.pack(pady=5)
        
        # Analysis Options
        options_frame = tk.LabelFrame(left_panel, text="⚙️ Analysis Options", 
                                     font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        options_frame.pack(fill='x', padx=10, pady=10)
        
        # Threshold slider
        tk.Label(options_frame, text="Threshold:", font=('Arial', 10), bg='#ecf0f1').pack(anchor='w', padx=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = tk.Scale(options_frame, from_=0.1, to=0.9, resolution=0.1, 
                                  orient='horizontal', variable=self.threshold_var,
                                  bg='#ecf0f1', font=('Arial', 9))
        threshold_scale.pack(fill='x', padx=5, pady=2)
        
        # Analyze Button
        self.analyze_btn = tk.Button(left_panel, text="🔬 Analyze Wound", command=self.analyze_image,
                                    font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                    relief='raised', bd=3, padx=30, pady=10, state='disabled')
        self.analyze_btn.pack(pady=20)
        
        # Progress Bar
        self.progress = ttk.Progressbar(left_panel, mode='indeterminate')
        self.progress.pack(fill='x', padx=10, pady=5)
        
        # Right Panel - Results
        right_panel = tk.Frame(content_frame, bg='#ecf0f1', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Results
        results_frame = tk.LabelFrame(right_panel, text="📊 Analysis Results", 
                                     font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=15, width=40, 
                                   font=('Arial', 10), bg='white', fg='#2c3e50',
                                   relief='sunken', bd=2, wrap='word')
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar for results
        scrollbar = tk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # Initial message
        self.results_text.insert('1.0', "Welcome to Wound Segmentation Analysis!\n\n")
        self.results_text.insert('end', "1. Select an image file\n")
        self.results_text.insert('end', "2. Adjust the threshold if needed\n")
        self.results_text.insert('end', "3. Click 'Analyze Wound' to start\n\n")
        self.results_text.insert('end', "The analysis will use the SimCLR U-Net model\n")
        self.results_text.insert('end', "to detect and segment wound areas.")
    
    def check_api_status(self):
        """Check if the API servers are running"""
        def check_status():
            try:
                # Check model server
                model_response = requests.get(f"{self.model_url}/readyz", timeout=3)
                model_ready = model_response.json().get('ready', False)
                
                # Check proxy API
                api_response = requests.get(f"{self.api_url}/readyz", timeout=3)
                api_ready = api_response.json().get('ready', False)
                
                if model_ready and api_ready:
                    self.status_label.config(text="✅ API Ready - SimCLR U-Net Model Loaded", fg='#27ae60')
                    self.analyze_btn.config(state='normal')
                else:
                    self.status_label.config(text="❌ API Not Ready - Check servers", fg='#e74c3c')
                    self.analyze_btn.config(state='disabled')
                    
            except Exception as e:
                self.status_label.config(text=f"❌ API Error: {str(e)[:30]}...", fg='#e74c3c')
                self.analyze_btn.config(state='disabled')
        
        # Run in separate thread to avoid blocking UI
        threading.Thread(target=check_status, daemon=True).start()
    
    def select_file(self):
        """Select an image file"""
        filetypes = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=filetypes
        )
        
        if filename:
            self.selected_file = filename
            self.file_label.config(text=f"📁 {filename.split('/')[-1]}")
            self.analyze_btn.config(state='normal')
            self.results_text.delete('1.0', 'end')
            self.results_text.insert('1.0', f"Selected: {filename.split('/')[-1]}\n\n")
            self.results_text.insert('end', "Ready to analyze! Click 'Analyze Wound' to start.")
    
    def analyze_image(self):
        """Analyze the selected image"""
        if not self.selected_file:
            messagebox.showerror("Error", "Please select an image file first!")
            return
        
        # Disable button and start progress
        self.analyze_btn.config(state='disabled', text="🔄 Analyzing...")
        self.progress.start()
        
        # Run analysis in separate thread
        threading.Thread(target=self._analyze_thread, daemon=True).start()
    
    def _analyze_thread(self):
        """Run analysis in background thread"""
        try:
            # Convert image to base64
            with open(self.selected_file, 'rb') as f:
                img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode()
            
            # Prepare request
            payload = {
                "image_b64": img_b64,
                "threshold": self.threshold_var.get()
            }
            
            # Send request to API
            response = requests.post(f"{self.api_url}/analyze", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                self.analysis_result = result
                
                # Update UI in main thread
                self.root.after(0, self._display_results, result)
            else:
                error_msg = f"API Error: {response.status_code}\n{response.text}"
                self.root.after(0, self._show_error, error_msg)
                
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, self._show_error, error_msg)
    
    def _display_results(self, result):
        """Display analysis results"""
        self.progress.stop()
        self.analyze_btn.config(state='normal', text="🔬 Analyze Wound")
        
        # Clear and display results
        self.results_text.delete('1.0', 'end')
        
        analysis = result.get('result', {})
        
        # Basic info
        self.results_text.insert('end', "🎯 WOUND ANALYSIS RESULTS\n")
        self.results_text.insert('end', "=" * 30 + "\n\n")
        
        # Wound metrics
        area_px = analysis.get('mask_area_px', 0)
        percentage = analysis.get('wound_percentage', 0.0)
        perimeter = analysis.get('perimeter_px', 0.0)
        
        self.results_text.insert('end', f"📏 Wound Area: {area_px:,} pixels\n")
        self.results_text.insert('end', f"📊 Wound Percentage: {percentage:.4f} ({percentage*100:.2f}%)\n")
        self.results_text.insert('end', f"📐 Perimeter: {perimeter:.1f} pixels\n\n")
        
        # Assessment
        severity = analysis.get('severity', 'Unknown')
        healing = analysis.get('healing_potential', 'Unknown')
        
        self.results_text.insert('end', f"⚠️  Severity: {severity}\n")
        self.results_text.insert('end', f"💊 Healing Potential: {healing}\n\n")
        
        # Technical details
        self.results_text.insert('end', "🔧 Technical Details:\n")
        self.results_text.insert('end', f"• Model: SimCLR U-Net\n")
        self.results_text.insert('end', f"• Threshold: {self.threshold_var.get()}\n")
        self.results_text.insert('end', f"• Report ID: {result.get('id', 'N/A')}\n")
        self.results_text.insert('end', f"• Mode: {result.get('mode', 'N/A')}\n")
        
        # Show success message
        messagebox.showinfo("Analysis Complete", 
                           f"Wound analysis completed!\n\n"
                           f"Area: {area_px:,} pixels ({percentage*100:.2f}%)\n"
                           f"Severity: {severity}\n"
                           f"Healing Potential: {healing}")
    
    def _show_error(self, error_msg):
        """Show error message"""
        self.progress.stop()
        self.analyze_btn.config(state='normal', text="🔬 Analyze Wound")
        
        self.results_text.delete('1.0', 'end')
        self.results_text.insert('end', f"❌ ERROR\n")
        self.results_text.insert('end', "=" * 20 + "\n\n")
        self.results_text.insert('end', error_msg)
        
        messagebox.showerror("Analysis Failed", error_msg)

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = WoundSegmentationGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()