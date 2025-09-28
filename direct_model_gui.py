#!/usr/bin/env python3
"""
Direct Model GUI - Bypasses API and loads model directly like the working GUI
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import sys
from sklearn.cluster import KMeans
# import pandas as pd  # Optional for Stone classifier

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def load_model():
    """Load the real model using TensorFlow's Keras (more compatible)"""
    try:
        import tensorflow as tf
        print(f"Using TensorFlow version: {tf.__version__}")
        print(f"Using Keras version: {tf.keras.__version__}")
        
        model_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
        
        # Custom objects for the model
        custom_objects = {
            'Custom>total_loss': lambda *args, **kwargs: 0.0,
            'total_loss': lambda *args, **kwargs: 0.0,
        }
        
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects=custom_objects
        )
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image for model input"""
    try:
        print(f"Preprocessing image: {image_path}")
        img = Image.open(image_path).convert('RGB')
        print(f"Image loaded, size: {img.size}")
        img = img.resize(target_size, Image.BILINEAR)
        print(f"Image resized to: {img.size}")
        img_array = np.array(img, dtype=np.float32) / 255.0
        print(f"Image array shape: {img_array.shape}")
        return np.expand_dims(img_array, 0)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_wound(model, image_array, threshold=0.5):
    """Predict wound segmentation"""
    try:
        print(f"Starting prediction with input shape: {image_array.shape}")
        prediction = model.predict(image_array, verbose=0)
        print(f"Prediction completed, shape: {prediction.shape}")
        prob_map = prediction[0, :, :, 0]  # Raw probability map
        print(f"Probability map shape: {prob_map.shape}")
        mask = (prob_map >= threshold).astype(np.uint8) * 255
        
        # Generate heatmap from probability map
        print("Generating heatmap...")
        heatmap = generate_heatmap(prob_map)
        print("Heatmap generated successfully")
        
        # Analyze skin color
        print("Analyzing skin color...")
        skin_analysis = analyze_skin_color(image_array)
        print("Skin color analysis completed")
        
        # Calculate metrics
        area_px = int((mask > 0).sum())
        h, w = mask.shape
        percentage = round(area_px / float(h * w), 6)
        
        # Find contours for perimeter
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True)) if contours else 0.0
        
        # Determine severity and healing potential (considering skin tone)
        severity = "Mild" if percentage < 0.01 else "Moderate" if percentage < 0.05 else "Severe"
        healing = "Good" if percentage < 0.01 else "Fair" if percentage < 0.05 else "Poor"
        
        # Adjust healing potential based on skin tone
        skin_tone_level = skin_analysis['skin_tone_level']
        if skin_tone_level >= 6:  # Darker skin tones may have different healing characteristics
            if healing == "Good":
                healing = "Good (Darker skin)"
            elif healing == "Fair":
                healing = "Fair (Darker skin)"
        
        return {
            'mask': mask,
            'heatmap': heatmap,
            'prob_map': prob_map,
            'area_px': area_px,
            'percentage': percentage,
            'perimeter': perimeter,
            'severity': severity,
            'healing': healing,
            'skin_analysis': skin_analysis
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def generate_heatmap(prob_map):
    """Generate a heatmap visualization from probability map"""
    try:
        print(f"Generating heatmap from prob_map shape: {prob_map.shape}")
        # Normalize probabilities to 0-255 range
        prob_normalized = (prob_map * 255).astype(np.uint8)
        print(f"Normalized prob_map shape: {prob_normalized.shape}")
        
        # Apply colormap (using OpenCV's JET colormap for heatmap effect)
        heatmap = cv2.applyColorMap(prob_normalized, cv2.COLORMAP_JET)
        print(f"Heatmap shape after colormap: {heatmap.shape}")
        
        # Convert BGR to RGB for PIL
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        print(f"Final heatmap shape: {heatmap_rgb.shape}")
        
        return heatmap_rgb
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple grayscale version if colormap fails
        prob_normalized = (prob_map * 255).astype(np.uint8)
        return np.stack([prob_normalized] * 3, axis=-1)

def analyze_skin_color(image_array):
    """Analyze skin color using Stone classifier (if available) or fallback to K-means"""
    try:
        print("Analyzing skin color...")
        
        # Convert to RGB if needed
        if len(image_array.shape) == 4:
            image_array = image_array[0]  # Remove batch dimension
        
        # Try Stone classifier first (if available)
        stone_result = run_stone_classifier(image_array)
        if stone_result:
            return stone_result
        
        # Fallback to K-means if Stone is not available
        print("Stone classifier not available, using K-means fallback...")
        
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

def create_skin_color_palette(colors):
    """Create a visual palette of detected skin colors"""
    try:
        if len(colors) == 0:
            return None
            
        # Create a small palette image
        palette_height = 50
        palette_width = len(colors) * 50
        
        palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        for i, color in enumerate(colors):
            start_x = i * 50
            end_x = start_x + 50
            palette[:, start_x:end_x] = color.astype(np.uint8)
        
        return palette
        
    except Exception as e:
        print(f"Error creating skin color palette: {e}")
        return None

class DirectModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Direct Model Wound Analysis")
        self.root.geometry("1200x800")
        
        self.model = None
        self.current_image = None
        self.current_result = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model status
        self.status_label = ttk.Label(main_frame, text="Model: Loading...", font=("Arial", 12, "bold"))
        self.status_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Image selection
        ttk.Button(main_frame, text="Select Image", command=self.select_image).grid(row=1, column=0, pady=5)
        ttk.Button(main_frame, text="Analyze", command=self.analyze_image).grid(row=1, column=1, pady=5)
        
        # Threshold slider
        ttk.Label(main_frame, text="Threshold:").grid(row=2, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = ttk.Scale(main_frame, from_=0.1, to=0.9, variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results labels
        self.area_label = ttk.Label(results_frame, text="Wound Area: --")
        self.area_label.grid(row=0, column=0, sticky=tk.W)
        
        self.percentage_label = ttk.Label(results_frame, text="Percentage: --")
        self.percentage_label.grid(row=1, column=0, sticky=tk.W)
        
        self.perimeter_label = ttk.Label(results_frame, text="Perimeter: --")
        self.perimeter_label.grid(row=2, column=0, sticky=tk.W)
        
        self.severity_label = ttk.Label(results_frame, text="Severity: --")
        self.severity_label.grid(row=3, column=0, sticky=tk.W)
        
        self.healing_label = ttk.Label(results_frame, text="Healing Potential: --")
        self.healing_label.grid(row=4, column=0, sticky=tk.W)
        
        # Skin color analysis labels
        self.skin_tone_label = ttk.Label(results_frame, text="Skin Tone Level: --")
        self.skin_tone_label.grid(row=5, column=0, sticky=tk.W)
        
        self.skin_description_label = ttk.Label(results_frame, text="Skin Type: --")
        self.skin_description_label.grid(row=6, column=0, sticky=tk.W)
        
        self.skin_color_label = ttk.Label(results_frame, text="Dominant Skin Color: --")
        self.skin_color_label.grid(row=7, column=0, sticky=tk.W)
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Images", padding="10")
        image_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        ttk.Label(image_frame, text="Original").grid(row=0, column=0, padx=5)
        self.original_label = ttk.Label(image_frame)
        self.original_label.grid(row=1, column=0, padx=5)
        
        # Mask image
        ttk.Label(image_frame, text="Mask").grid(row=0, column=1, padx=5)
        self.mask_label = ttk.Label(image_frame)
        self.mask_label.grid(row=1, column=1, padx=5)
        
        # Overlay image
        ttk.Label(image_frame, text="Overlay").grid(row=0, column=2, padx=5)
        self.overlay_label = ttk.Label(image_frame)
        self.overlay_label.grid(row=1, column=2, padx=5)
        
        # Heatmap image
        ttk.Label(image_frame, text="Heatmap").grid(row=0, column=3, padx=5)
        self.heatmap_label = ttk.Label(image_frame)
        self.heatmap_label.grid(row=1, column=3, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.columnconfigure(2, weight=1)
        image_frame.columnconfigure(3, weight=1)
    
    def load_model(self):
        """Load the model"""
        self.status_label.config(text="Model: Loading...")
        self.root.update()
        
        self.model = load_model()
        if self.model:
            self.status_label.config(text="Model: ✅ Ready", foreground="green")
        else:
            self.status_label.config(text="Model: ❌ Failed", foreground="red")
    
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.current_image = file_path
            self.display_original_image()
    
    def display_original_image(self):
        """Display the original image"""
        if not self.current_image:
            return
        
        try:
            img = Image.open(self.current_image)
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.original_label.config(image=photo)
            self.original_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {e}")
    
    def analyze_image(self):
        """Analyze the selected image"""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        if not self.current_image:
            messagebox.showerror("Error", "Please select an image first!")
            return
        
        try:
            # Preprocess image
            image_array = preprocess_image(self.current_image)
            if image_array is None:
                messagebox.showerror("Error", "Failed to preprocess image!")
                return
            
            # Get threshold
            threshold = self.threshold_var.get()
            
            # Predict
            result = predict_wound(self.model, image_array, threshold)
            if result is None:
                messagebox.showerror("Error", "Prediction failed!")
                return
            
            self.current_result = result
            self.update_results()
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
    
    def update_results(self):
        """Update the results labels"""
        if not self.current_result:
            return
        
        self.area_label.config(text=f"Wound Area: {self.current_result['area_px']} pixels")
        self.percentage_label.config(text=f"Percentage: {self.current_result['percentage']:.4f}")
        self.perimeter_label.config(text=f"Perimeter: {self.current_result['perimeter']:.2f} pixels")
        self.severity_label.config(text=f"Severity: {self.current_result['severity']}")
        self.healing_label.config(text=f"Healing Potential: {self.current_result['healing']}")
        
        # Update skin color information
        if 'skin_analysis' in self.current_result:
            skin_analysis = self.current_result['skin_analysis']
            
            # Show enhanced skin tone information
            if skin_analysis.get('method') == 'stone_classifier':
                # Stone classifier results
                self.skin_tone_label.config(text=f"Skin Tone: {skin_analysis['skin_tone_level']} ({skin_analysis.get('accuracy', 0):.1f}% accuracy)")
                self.skin_description_label.config(text=f"Fitzpatrick: {skin_analysis.get('fitzpatrick_type', 'Unknown')} - {skin_analysis.get('fitzpatrick_description', 'Unknown')}")
                
                # Format RGB color for display
                rgb_color = skin_analysis['skin_color_rgb']
                color_text = f"RGB({int(rgb_color[0])}, {int(rgb_color[1])}, {int(rgb_color[2])})"
                self.skin_color_label.config(text=f"Skin Color: {color_text} (Method: Stone AI)")
            else:
                # K-means fallback results
                self.skin_tone_label.config(text=f"Skin Tone Level: {skin_analysis['skin_tone_level']}/8")
                self.skin_description_label.config(text=f"Skin Type: {skin_analysis['skin_tone_description']}")
                
                # Format RGB color for display
                rgb_color = skin_analysis['skin_color_rgb']
                color_text = f"RGB({int(rgb_color[0])}, {int(rgb_color[1])}, {int(rgb_color[2])})"
                self.skin_color_label.config(text=f"Dominant Skin Color: {color_text} (Method: K-means)")
        else:
            self.skin_tone_label.config(text="Skin Tone Level: --")
            self.skin_description_label.config(text="Skin Type: --")
            self.skin_color_label.config(text="Dominant Skin Color: --")
    
    def display_results(self):
        """Display the mask and overlay images"""
        if not self.current_result:
            return
        
        try:
            # Display mask
            mask_img = Image.fromarray(self.current_result['mask'])
            mask_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            mask_photo = ImageTk.PhotoImage(mask_img)
            self.mask_label.config(image=mask_photo)
            self.mask_label.image = mask_photo
            
            # Create overlay
            original = Image.open(self.current_image)
            original = original.resize((128, 128), Image.Resampling.LANCZOS)
            mask_resized = Image.fromarray(self.current_result['mask']).resize((128, 128), Image.Resampling.LANCZOS)
            
            # Create colored overlay
            overlay = original.copy()
            mask_array = np.array(mask_resized)
            overlay_array = np.array(overlay)
            
            # Apply red overlay where mask is white
            overlay_array[mask_array > 0] = [255, 0, 0]  # Red color
            
            overlay_img = Image.fromarray(overlay_array)
            overlay_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            overlay_photo = ImageTk.PhotoImage(overlay_img)
            self.overlay_label.config(image=overlay_photo)
            self.overlay_label.image = overlay_photo
            
            # Display heatmap
            if 'heatmap' in self.current_result:
                heatmap_img = Image.fromarray(self.current_result['heatmap'])
                heatmap_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                heatmap_photo = ImageTk.PhotoImage(heatmap_img)
                self.heatmap_label.config(image=heatmap_photo)
                self.heatmap_label.image = heatmap_photo
            
        except Exception as e:
            print(f"Error displaying results: {e}")

def main():
    root = tk.Tk()
    app = DirectModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()