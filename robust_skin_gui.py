#!/usr/bin/env python3
"""
Robust Skin Color Analysis GUI - Enhanced error handling and debugging
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import sys
from sklearn.cluster import KMeans
import tempfile
import subprocess
import traceback

# Fitzpatrick skin type mapping
TONE_LABEL_TO_FITZPATRICK = {
    'CA': ('Type I',    'Very light'),
    'CB': ('Type II',   'Light'),
    'CC': ('Type II-III', 'Light-medium'),
    'CD': ('Type III',  'Medium'),
    'CE': ('Type IV',   'Medium-dark'),
    'CF': ('Type V',    'Dark'),
    'CG': ('Type V',    'Dark'),
    'CH': ('Type VI',   'Very dark'),
    'CI': ('Type VI',   'Very dark'),
    'CJ': ('Type VI',   'Deep brown'),
    'CK': ('Type VI',   'Black'),
}

def get_fitzpatrick_from_label(label):
    return TONE_LABEL_TO_FITZPATRICK.get(label, ('Unknown', 'Unknown'))

def analyze_skin_color_simple(image_array):
    """Simple skin color analysis using basic color statistics"""
    try:
        print("Starting simple skin color analysis...")
        
        # Convert to RGB if needed
        if len(image_array.shape) == 4:
            image_array = image_array[0]  # Remove batch dimension
        
        print(f"Image shape: {image_array.shape}")
        
        # Calculate average color
        avg_color = np.mean(image_array, axis=(0, 1))
        print(f"Average color (RGB): {avg_color}")
        
        # Calculate brightness
        brightness = np.mean(avg_color)
        print(f"Brightness: {brightness}")
        
        # Simple skin tone classification based on brightness
        if brightness > 0.8:
            skin_tone_level = 1
            skin_tone_desc = "Very Light (Type I)"
        elif brightness > 0.7:
            skin_tone_level = 2
            skin_tone_desc = "Light (Type II)"
        elif brightness > 0.6:
            skin_tone_level = 3
            skin_tone_desc = "Light-Medium (Type III)"
        elif brightness > 0.5:
            skin_tone_level = 4
            skin_tone_desc = "Medium (Type IV)"
        elif brightness > 0.4:
            skin_tone_level = 5
            skin_tone_desc = "Medium-Dark (Type V)"
        elif brightness > 0.3:
            skin_tone_level = 6
            skin_tone_desc = "Dark (Type VI)"
        else:
            skin_tone_level = 7
            skin_tone_desc = "Very Dark (Type VII)"
        
        print(f"Skin tone level: {skin_tone_level}")
        print(f"Skin tone description: {skin_tone_desc}")
        
        return {
            'skin_color_rgb': avg_color,
            'skin_tone_level': skin_tone_level,
            'skin_tone_description': skin_tone_desc,
            'brightness': brightness,
            'method': 'simple_analysis'
        }
        
    except Exception as e:
        print(f"Error in simple analysis: {e}")
        traceback.print_exc()
        return {
            'skin_color_rgb': [0, 0, 0],
            'skin_tone_level': 5,
            'skin_tone_description': 'Unknown',
            'brightness': 0.5,
            'method': 'error'
        }

def analyze_skin_color_kmeans(image_array):
    """Analyze skin color using K-means clustering"""
    try:
        print("Starting K-means skin color analysis...")
        
        # Convert to RGB if needed
        if len(image_array.shape) == 4:
            image_array = image_array[0]  # Remove batch dimension
        
        print(f"Image shape: {image_array.shape}")
        
        # Reshape image to list of pixels
        pixels = image_array.reshape(-1, 3)
        print(f"Pixels shape: {pixels.shape}")
        
        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_
        print(f"Found {len(colors)} dominant colors")
        
        # Find the most dominant skin-like color
        # Skin colors typically have higher red values and moderate green/blue
        skin_scores = []
        for i, color in enumerate(colors):
            r, g, b = color
            # Simple skin detection: higher red, moderate green, lower blue
            skin_score = r * 0.4 + g * 0.3 + b * 0.3
            skin_scores.append(skin_score)
            print(f"Color {i}: RGB({r:.2f}, {g:.2f}, {b:.2f}) - Skin score: {skin_score:.2f}")
        
        # Get the most skin-like color
        dominant_skin_idx = np.argmax(skin_scores)
        dominant_skin_color = colors[dominant_skin_idx]
        
        print(f"Dominant skin color: RGB({dominant_skin_color[0]:.2f}, {dominant_skin_color[1]:.2f}, {dominant_skin_color[2]:.2f})")
        
        # Classify skin tone based on RGB values
        r, g, b = dominant_skin_color
        
        # Calculate skin tone level (0-10 scale)
        # Higher values = darker skin, lower values = lighter skin
        skin_tone_level = classify_skin_tone(r, g, b)
        
        # Get skin tone description
        skin_tone_desc = get_skin_tone_description(skin_tone_level)
        
        print(f"Skin tone level: {skin_tone_level}")
        print(f"Skin tone description: {skin_tone_desc}")
        
        return {
            'skin_color_rgb': dominant_skin_color,
            'skin_tone_level': skin_tone_level,
            'skin_tone_description': skin_tone_desc,
            'all_colors': colors,
            'method': 'kmeans_clustering'
        }
        
    except Exception as e:
        print(f"Error in K-means analysis: {e}")
        traceback.print_exc()
        return {
            'skin_color_rgb': [0, 0, 0],
            'skin_tone_level': 5,
            'skin_tone_description': 'Unknown',
            'all_colors': [],
            'method': 'error'
        }

def classify_skin_tone(r, g, b):
    """Classify skin tone on a 0-10 scale"""
    # Normalize RGB values to 0-1
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    # Calculate overall brightness
    brightness = (r_norm + g_norm + b_norm) / 3
    
    # Calculate skin tone level (0 = very light, 10 = very dark)
    if brightness > 0.8:
        return 1  # Very light
    elif brightness > 0.7:
        return 2  # Light
    elif brightness > 0.6:
        return 3  # Light-medium
    elif brightness > 0.5:
        return 4  # Medium
    elif brightness > 0.4:
        return 5  # Medium-dark
    elif brightness > 0.3:
        return 6  # Dark
    elif brightness > 0.2:
        return 7  # Very dark
    else:
        return 8  # Extremely dark

def get_skin_tone_description(level):
    """Get human-readable skin tone description"""
    descriptions = {
        1: "Very Light (Type I)",
        2: "Light (Type II)", 
        3: "Light-Medium (Type III)",
        4: "Medium (Type IV)",
        5: "Medium-Dark (Type V)",
        6: "Dark (Type VI)",
        7: "Very Dark (Type VII)",
        8: "Extremely Dark (Type VIII)"
    }
    return descriptions.get(level, "Unknown")

class RobustSkinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robust Skin Color Analysis Tool")
        self.root.geometry("900x700")
        
        self.current_image = None
        self.current_result = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image selection
        ttk.Button(main_frame, text="Select Image", command=self.select_image).grid(row=0, column=0, pady=5)
        ttk.Button(main_frame, text="Simple Analysis", command=self.analyze_simple).grid(row=0, column=1, pady=5)
        ttk.Button(main_frame, text="K-means Analysis", command=self.analyze_kmeans).grid(row=0, column=2, pady=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready - Select an image to begin", font=("Arial", 10))
        self.status_label.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Skin Color Analysis Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results labels
        self.skin_tone_label = ttk.Label(results_frame, text="Skin Tone Level: --")
        self.skin_tone_label.grid(row=0, column=0, sticky=tk.W)
        
        self.skin_description_label = ttk.Label(results_frame, text="Skin Type: --")
        self.skin_description_label.grid(row=1, column=0, sticky=tk.W)
        
        self.skin_color_label = ttk.Label(results_frame, text="Dominant Skin Color: --")
        self.skin_color_label.grid(row=2, column=0, sticky=tk.W)
        
        self.method_label = ttk.Label(results_frame, text="Analysis Method: --")
        self.method_label.grid(row=3, column=0, sticky=tk.W)
        
        self.brightness_label = ttk.Label(results_frame, text="Brightness: --")
        self.brightness_label.grid(row=4, column=0, sticky=tk.W)
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        self.original_label = ttk.Label(image_frame)
        self.original_label.grid(row=0, column=0, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(3, weight=1)
        image_frame.columnconfigure(0, weight=1)
    
    def select_image(self):
        """Select an image file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
            )
            if file_path:
                self.current_image = file_path
                self.status_label.config(text=f"Image selected: {os.path.basename(file_path)}")
                self.display_image()
        except Exception as e:
            messagebox.showerror("Error", f"Could not select image: {e}")
            self.status_label.config(text="Error selecting image")
    
    def display_image(self):
        """Display the selected image"""
        if not self.current_image:
            return
        
        try:
            self.status_label.config(text="Loading image...")
            self.root.update()
            
            img = Image.open(self.current_image)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.original_label.config(image=photo)
            self.original_label.image = photo  # Keep a reference
            
            self.status_label.config(text=f"Image loaded: {os.path.basename(self.current_image)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
            self.status_label.config(text="Error loading image")
    
    def analyze_simple(self):
        """Perform simple skin color analysis"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            self.status_label.config(text="Performing simple analysis...")
            self.root.update()
            
            # Load and preprocess image
            img = Image.open(self.current_image).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, 0)  # Add batch dimension
            
            print("Starting simple skin color analysis...")
            result = analyze_skin_color_simple(img_array)
            self.current_result = result
            
            # Update UI
            self.update_results()
            self.status_label.config(text="Simple analysis completed")
            
        except Exception as e:
            error_msg = f"Simple analysis failed: {e}"
            print(error_msg)
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text="Analysis failed")
    
    def analyze_kmeans(self):
        """Perform K-means skin color analysis"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            self.status_label.config(text="Performing K-means analysis...")
            self.root.update()
            
            # Load and preprocess image
            img = Image.open(self.current_image).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, 0)  # Add batch dimension
            
            print("Starting K-means skin color analysis...")
            result = analyze_skin_color_kmeans(img_array)
            self.current_result = result
            
            # Update UI
            self.update_results()
            self.status_label.config(text="K-means analysis completed")
            
        except Exception as e:
            error_msg = f"K-means analysis failed: {e}"
            print(error_msg)
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text="Analysis failed")
    
    def update_results(self):
        """Update the results labels"""
        if not self.current_result:
            return
        
        skin_analysis = self.current_result
        
        # Update labels based on analysis method
        if skin_analysis.get('method') == 'simple_analysis':
            self.skin_tone_label.config(text=f"Skin Tone Level: {skin_analysis['skin_tone_level']}/8")
            self.skin_description_label.config(text=f"Skin Type: {skin_analysis['skin_tone_description']}")
            
            # Format RGB color for display
            rgb_color = skin_analysis['skin_color_rgb']
            color_text = f"RGB({int(rgb_color[0])}, {int(rgb_color[1])}, {int(rgb_color[2])})"
            self.skin_color_label.config(text=f"Average Color: {color_text}")
            self.method_label.config(text="Analysis Method: Simple Color Statistics")
            self.brightness_label.config(text=f"Brightness: {skin_analysis.get('brightness', 0):.3f}")
            
        elif skin_analysis.get('method') == 'kmeans_clustering':
            self.skin_tone_label.config(text=f"Skin Tone Level: {skin_analysis['skin_tone_level']}/8")
            self.skin_description_label.config(text=f"Skin Type: {skin_analysis['skin_tone_description']}")
            
            # Format RGB color for display
            rgb_color = skin_analysis['skin_color_rgb']
            color_text = f"RGB({int(rgb_color[0])}, {int(rgb_color[1])}, {int(rgb_color[2])})"
            self.skin_color_label.config(text=f"Dominant Skin Color: {color_text}")
            self.method_label.config(text="Analysis Method: K-means Clustering")
            self.brightness_label.config(text="Brightness: N/A (K-means)")
            
        else:
            # Error case
            self.skin_tone_label.config(text="Skin Tone Level: Error")
            self.skin_description_label.config(text="Skin Type: Analysis Failed")
            self.skin_color_label.config(text="Dominant Skin Color: Error")
            self.method_label.config(text="Analysis Method: Failed")
            self.brightness_label.config(text="Brightness: Error")

def main():
    root = tk.Tk()
    app = RobustSkinGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()