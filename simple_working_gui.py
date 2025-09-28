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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from PIL import Image, ImageTk
import os
import sys

# Add the project root to the path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import configuration and custom objects
from woundseg.config import Config
from woundseg.models.keras_custom import register_custom_objects
from wound_medsam import build_unet, predict_healing_potential

# Register custom Keras objects
register_custom_objects()

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
            
            model_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
            
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return False
            
            # Load model with custom objects
            self.model = tf.keras.models.load_model(model_path, compile=False)
            
            # Compile the model
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
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
            
            # Preprocess image
            img_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB) / 255.0
            img_resized = tf.image.resize(img_rgb, (128, 128))
            img_batch = tf.expand_dims(img_resized, 0)
            
            # Run inference
            prediction = self.model.predict(img_batch, verbose=0)[0, :, :, 0]
            mask = (prediction > 0.5).astype(np.uint8)
            
            # Resize mask back to original size
            mask_resized = tf.image.resize(mask[..., None], self.current_image.shape[:2], method='nearest')
            mask_resized = mask_resized.numpy().squeeze().astype(np.uint8)
            
            # Calculate metrics
            severity, healing_potential, wound_area_mm2 = predict_healing_potential(
                mask_resized, img_rgb, Config.SCALE_MM_PER_PIXEL
            )
            
            # Store results
            self.current_mask = mask_resized
            self.current_results = {
                'wound_area_mm2': wound_area_mm2,
                'severity': severity,
                'healing_potential': healing_potential,
                'mask_pixels': int(np.sum(mask_resized > 0)),
                'total_pixels': mask_resized.size,
                'wound_percentage': float(np.sum(mask_resized > 0)) / mask_resized.size * 100
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