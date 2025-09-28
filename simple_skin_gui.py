#!/usr/bin/env python3
"""
Simple Skin Color Analysis GUI - Tests skin color analysis without model loading
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

# Fitzpatrick skin type mapping from test_wound_progress_skintype.py
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

def run_stone_classifier(image_array):
    """Run Stone skin tone classifier if available"""
    try:
        # Save image temporarily for Stone classifier
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "temp_skin.jpg")
        
        # Convert numpy array to PIL Image and save
        if len(image_array.shape) == 4:
            image_array = image_array[0]  # Remove batch dimension
        
        # Convert to uint8 and save
        img_uint8 = (image_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img.save(temp_image_path, "JPEG")
        
        # Run Stone classifier
        result_dir = os.path.join(temp_dir, "stone_results")
        os.makedirs(result_dir, exist_ok=True)
        
        subprocess.run([
            "stone", "-i", temp_image_path, "-o", result_dir, "-d"
        ], check=True, capture_output=True)
        
        # Read results
        result_csv = os.path.join(result_dir, "result.csv")
        if os.path.exists(result_csv):
            try:
                import pandas as pd
                df = pd.read_csv(result_csv)
                if len(df) > 0:
                    row = df.iloc[0]
                    tone_label = row['tone label']
                    dominant_color = row['dominant 1']
                    percent = float(row['percent 1']) * 100
                    skin_tone_hex = row['skin tone']
                    acc = row['accuracy(0-100)']
                    fitz_type, description = get_fitzpatrick_from_label(tone_label)
                    
                    # Clean up temp files
                    import shutil
                    shutil.rmtree(temp_dir)
                    
                    return {
                        'skin_color_rgb': [int(skin_tone_hex[i:i+2], 16) for i in (1, 3, 5)],  # Convert hex to RGB
                        'skin_tone_level': tone_label,
                        'skin_tone_description': f"{fitz_type} {description}",
                        'fitzpatrick_type': fitz_type,
                        'fitzpatrick_description': description,
                        'dominant_color': dominant_color,
                        'accuracy': acc,
                        'method': 'stone_classifier'
                    }
            except ImportError:
                print("Pandas not available, Stone classifier results cannot be parsed")
                # Clean up temp files
                import shutil
                shutil.rmtree(temp_dir)
                return None
        
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir)
        return None
        
    except Exception as e:
        print(f"Stone classifier failed: {e}")
        # Clean up temp files if they exist
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
        return None

def analyze_skin_color_kmeans(image_array):
    """Analyze skin color using K-means clustering"""
    try:
        print("Analyzing skin color with K-means...")
        
        # Convert to RGB if needed
        if len(image_array.shape) == 4:
            image_array = image_array[0]  # Remove batch dimension
        
        # Reshape image to list of pixels
        pixels = image_array.reshape(-1, 3)
        
        # Use K-means to find dominant colors
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_
        
        # Find the most dominant skin-like color
        # Skin colors typically have higher red values and moderate green/blue
        skin_scores = []
        for color in colors:
            r, g, b = color
            # Simple skin detection: higher red, moderate green, lower blue
            skin_score = r * 0.4 + g * 0.3 + b * 0.3
            skin_scores.append(skin_score)
        
        # Get the most skin-like color
        dominant_skin_idx = np.argmax(skin_scores)
        dominant_skin_color = colors[dominant_skin_idx]
        
        # Classify skin tone based on RGB values
        r, g, b = dominant_skin_color
        
        # Calculate skin tone level (0-10 scale)
        # Higher values = darker skin, lower values = lighter skin
        skin_tone_level = classify_skin_tone(r, g, b)
        
        # Get skin tone description
        skin_tone_desc = get_skin_tone_description(skin_tone_level)
        
        print(f"Dominant skin color (RGB): {dominant_skin_color}")
        print(f"Skin tone level: {skin_tone_level}")
        print(f"Skin tone description: {skin_tone_desc}")
        
        return {
            'skin_color_rgb': dominant_skin_color,
            'skin_tone_level': skin_tone_level,
            'skin_tone_description': skin_tone_desc,
            'all_colors': colors,
            'method': 'kmeans_fallback'
        }
        
    except Exception as e:
        print(f"Error analyzing skin color: {e}")
        import traceback
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
    # This is a simplified classification
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

class SimpleSkinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Color Analysis Tool")
        self.root.geometry("800x600")
        
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
        ttk.Button(main_frame, text="Analyze Skin Color", command=self.analyze_skin_color).grid(row=0, column=1, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Skin Color Analysis Results", padding="10")
        results_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results labels
        self.skin_tone_label = ttk.Label(results_frame, text="Skin Tone Level: --")
        self.skin_tone_label.grid(row=0, column=0, sticky=tk.W)
        
        self.skin_description_label = ttk.Label(results_frame, text="Skin Type: --")
        self.skin_description_label.grid(row=1, column=0, sticky=tk.W)
        
        self.skin_color_label = ttk.Label(results_frame, text="Dominant Skin Color: --")
        self.skin_color_label.grid(row=2, column=0, sticky=tk.W)
        
        self.method_label = ttk.Label(results_frame, text="Analysis Method: --")
        self.method_label.grid(row=3, column=0, sticky=tk.W)
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Image", padding="10")
        image_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        self.original_label = ttk.Label(image_frame)
        self.original_label.grid(row=0, column=0, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        image_frame.columnconfigure(0, weight=1)
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            self.current_image = file_path
            self.display_image()
    
    def display_image(self):
        """Display the selected image"""
        if not self.current_image:
            return
        
        try:
            img = Image.open(self.current_image)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.original_label.config(image=photo)
            self.original_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
    
    def analyze_skin_color(self):
        """Analyze skin color in the selected image"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            # Load and preprocess image
            img = Image.open(self.current_image).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, 0)  # Add batch dimension
            
            print("Starting skin color analysis...")
            
            # Try Stone classifier first
            stone_result = run_stone_classifier(img_array)
            if stone_result:
                self.current_result = stone_result
                print("✅ Stone classifier analysis completed")
            else:
                # Fallback to K-means
                print("Stone classifier not available, using K-means...")
                kmeans_result = analyze_skin_color_kmeans(img_array)
                self.current_result = kmeans_result
                print("✅ K-means analysis completed")
            
            # Update UI
            self.update_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def update_results(self):
        """Update the results labels"""
        if not self.current_result:
            return
        
        skin_analysis = self.current_result
        
        # Show enhanced skin tone information
        if skin_analysis.get('method') == 'stone_classifier':
            # Stone classifier results
            self.skin_tone_label.config(text=f"Skin Tone: {skin_analysis['skin_tone_level']} ({skin_analysis.get('accuracy', 0):.1f}% accuracy)")
            self.skin_description_label.config(text=f"Fitzpatrick: {skin_analysis.get('fitzpatrick_type', 'Unknown')} - {skin_analysis.get('fitzpatrick_description', 'Unknown')}")
            
            # Format RGB color for display
            rgb_color = skin_analysis['skin_color_rgb']
            color_text = f"RGB({int(rgb_color[0])}, {int(rgb_color[1])}, {int(rgb_color[2])})"
            self.skin_color_label.config(text=f"Skin Color: {color_text} (Method: Stone AI)")
            self.method_label.config(text="Analysis Method: Stone AI Classifier")
        else:
            # K-means fallback results
            self.skin_tone_label.config(text=f"Skin Tone Level: {skin_analysis['skin_tone_level']}/8")
            self.skin_description_label.config(text=f"Skin Type: {skin_analysis['skin_tone_description']}")
            
            # Format RGB color for display
            rgb_color = skin_analysis['skin_color_rgb']
            color_text = f"RGB({int(rgb_color[0])}, {int(rgb_color[1])}, {int(rgb_color[2])})"
            self.skin_color_label.config(text=f"Dominant Skin Color: {color_text} (Method: K-means)")
            self.method_label.config(text="Analysis Method: K-means Clustering")

def main():
    root = tk.Tk()
    app = SimpleSkinGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()