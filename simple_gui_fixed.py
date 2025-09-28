#!/usr/bin/env python3
"""
Simple Working GUI for Wound Segmentation
This GUI works directly with the model without needing API servers.
Based on the working test_working_script.py approach.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageTk
import os
import sys

class WoundSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wound Segmentation GUI")
        self.root.geometry("1200x800")
        
        # Model variables
        self.model = None
        self.current_image = None
        self.current_mask = None
        self.current_results = None
        
        # Settings
        self.IMG_SIZE = (128, 128)
        self.model_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
        self.scale_mm_per_pixel = 0.1  # mm per pixel
        
        # Setup GUI
        self.setup_gui()
        
        # Load model
        self.load_model()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Load image button
        self.load_btn = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Analyze button
        self.analyze_btn = ttk.Button(control_frame, text="Analyze Wound", command=self.analyze_wound, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready - Load an image to begin")
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Image display frame
        self.image_frame = ttk.LabelFrame(results_frame, text="Original Image", padding="5")
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Results display frame
        self.results_frame = ttk.LabelFrame(results_frame, text="Analysis Results", padding="5")
        self.results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Image label
        self.image_label = ttk.Label(self.image_frame, text="No image loaded")
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=10, width=40, wrap=tk.WORD)
        self.results_text.pack(expand=True, fill=tk.BOTH)
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.configure(yscrollcommand=scrollbar.set)
    
    def load_model(self):
        """Load the UNet model"""
        try:
            self.status_label.config(text="Loading model...")
            self.root.update()
            
            if not os.path.exists(self.model_path):
                messagebox.showerror("Error", f"Model file not found: {self.model_path}")
                return False
            
            # Load model directly like in test_working_script.py
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
            self.status_label.config(text="Model loaded successfully")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_label.config(text="Model loading failed")
            return False
    
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if not file_path:
            return
        
        try:
            # Load and display image
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                messagebox.showerror("Error", "Could not load image")
                return
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            display_image = self.resize_for_display(image_rgb, max_size=(400, 400))
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(display_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update image label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Enable analyze button
            self.analyze_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Image loaded - Click 'Analyze Wound' to proceed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def analyze_wound(self):
        """Analyze the loaded image for wound segmentation"""
        if self.current_image is None or self.model is None:
            return
        
        try:
            self.status_label.config(text="Analyzing wound...")
            self.analyze_btn.config(state=tk.DISABLED)
            self.root.update()
            
            # Load and preprocess image like in test_working_script.py
            # Convert current image to PIL format
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize(self.IMG_SIZE)
            img_array = np.array(img_pil) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run segmentation
            prediction = self.model.predict(img_array, verbose=0)
            mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255
            
            # Calculate metrics
            area_pixels = np.sum(mask > 0)
            area_mm2 = area_pixels * (self.scale_mm_per_pixel ** 2)
            
            # Calculate perimeter
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter_px = cv2.arcLength(contours[0], True) if contours else 0.0
            
            # Calculate wound percentage
            total_pixels = mask.size
            wound_percentage = (area_pixels / total_pixels) * 100
            
            # Determine severity and healing potential
            if wound_percentage < 1.0:
                severity = "Mild"
                healing_potential = "Good"
            elif wound_percentage < 5.0:
                severity = "Moderate"
                healing_potential = "Fair"
            else:
                severity = "Severe"
                healing_potential = "Poor"
            
            # Store results
            self.current_mask = mask
            self.current_results = {
                'wound_area_mm2': area_mm2,
                'severity': severity,
                'healing_potential': healing_potential,
                'mask_pixels': int(area_pixels),
                'total_pixels': total_pixels,
                'wound_percentage': wound_percentage,
                'perimeter_px': perimeter_px
            }
            
            # Display results
            self.display_results()
            
            self.status_label.config(text="Analysis complete")
            self.analyze_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_label.config(text="Analysis failed")
            self.analyze_btn.config(state=tk.NORMAL)
    
    def display_results(self):
        """Display analysis results"""
        if self.current_results is None:
            return
        
        results_text = f"""Wound Analysis Results:

Wound Area: {self.current_results['wound_area_mm2']:.2f} mm²
Wound Pixels: {self.current_results['mask_pixels']:,} pixels
Total Pixels: {self.current_results['total_pixels']:,} pixels
Wound Percentage: {self.current_results['wound_percentage']:.3f}%
Perimeter: {self.current_results['perimeter_px']:.1f} pixels

Severity: {self.current_results['severity']}
Healing Potential: {self.current_results['healing_potential']}

Analysis completed successfully!
"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
    
    def resize_for_display(self, image, max_size=(400, 400)):
        """Resize image for display while maintaining aspect ratio"""
        h, w = image.shape[:2]
        max_h, max_w = max_size
        
        # Calculate scaling factor
        scale = min(max_h / h, max_w / w)
        
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            return cv2.resize(image, (new_w, new_h))
        
        return image

def main():
    root = tk.Tk()
    app = WoundSegmentationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()