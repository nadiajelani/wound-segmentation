#!/usr/bin/env python3
"""
Ultra Robust Skin Color Analysis GUI - Handles all edge cases and timeouts
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
import time
import threading
import io

class UltraRobustSkinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultra Robust Skin Color Analysis Tool")
        self.root.geometry("1000x800")
        
        self.current_image = None
        self.current_result = None
        self.analysis_thread = None
        self.model = None
        
        self.setup_ui()
        # Load model and wait for completion
        print("🔄 Starting model loading in GUI initialization...")
        model_loaded = self.load_model()
        if model_loaded:
            print("✅ Model loaded successfully in GUI initialization")
            print(f"Model status: {self.model is not None}")
            print(f"Model has predict method: {hasattr(self.model, 'predict') if self.model else False}")
        else:
            print("❌ Model loading failed in GUI initialization")
            print(f"Model status: {self.model is not None}")
            print("Will use basic CV methods for wound analysis")
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Wound Analysis & Skin Color Tool", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(0, 10))
        
        # Image selection
        ttk.Button(main_frame, text="Select Image", command=self.select_image).grid(row=1, column=0, pady=5, padx=2)
        ttk.Button(main_frame, text="Simple Analysis", command=self.analyze_simple).grid(row=1, column=1, pady=5, padx=2)
        ttk.Button(main_frame, text="K-means Analysis", command=self.analyze_kmeans).grid(row=1, column=2, pady=5, padx=2)
        
        # Wound analysis buttons
        ttk.Button(main_frame, text="Generate Mask", command=self.generate_mask).grid(row=2, column=0, pady=5, padx=2)
        ttk.Button(main_frame, text="Generate Segment", command=self.generate_segment).grid(row=2, column=1, pady=5, padx=2)
        ttk.Button(main_frame, text="Generate Heatmap", command=self.generate_heatmap).grid(row=2, column=2, pady=5, padx=2)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready - Select an image to begin", font=("Arial", 10))
        self.status_label.grid(row=3, column=0, columnspan=4, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results labels
        self.skin_tone_label = ttk.Label(results_frame, text="Skin Tone Level: --", font=("Arial", 11))
        self.skin_tone_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.skin_description_label = ttk.Label(results_frame, text="Skin Type: --", font=("Arial", 11))
        self.skin_description_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.skin_color_label = ttk.Label(results_frame, text="Dominant Skin Color: --", font=("Arial", 11))
        self.skin_color_label.grid(row=2, column=0, sticky=tk.W, pady=2)
        
        self.method_label = ttk.Label(results_frame, text="Analysis Method: --", font=("Arial", 11))
        self.method_label.grid(row=3, column=0, sticky=tk.W, pady=2)
        
        self.brightness_label = ttk.Label(results_frame, text="Brightness: --", font=("Arial", 11))
        self.brightness_label.grid(row=4, column=0, sticky=tk.W, pady=2)
        
        self.file_info_label = ttk.Label(results_frame, text="File Info: --", font=("Arial", 10))
        self.file_info_label.grid(row=5, column=0, sticky=tk.W, pady=2)
        
        # Image display frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=6, column=0, columnspan=4, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        self.original_label = ttk.Label(image_frame)
        self.original_label.grid(row=0, column=0, padx=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.rowconfigure(6, weight=1)
        image_frame.columnconfigure(0, weight=1)
    
    def generate_mask(self):
        """Generate wound mask using the trained model"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if self.model is None:
            messagebox.showwarning("Warning", "Model not loaded. Using basic CV method.")
            self._generate_mask_basic()
            return
        
        try:
            self.status_label.config(text="Generating mask using AI model...")
            self.root.update()
            
            # Load image
            img = self._load_image_robust()
            if img is None:
                raise Exception("Could not load image")
            
            # Preprocess image for model
            img_array = np.array(img, dtype=np.float32) / 255.0
            mh, mw = self._model_input_hw()
            img_resized = cv2.resize(img_array, (mw, mh))
            img_input = np.expand_dims(img_resized, 0)
            
            print("Running model prediction...")
            # Get model prediction
            prediction = self.model.predict(img_input, verbose=0)[0, :, :, 0]
            
            # Convert probability to binary mask
            threshold = 0.5
            mask = (prediction > threshold).astype(np.uint8) * 255
            
            # Resize mask back to original size
            mask_resized = cv2.resize(mask, (img.size[0], img.size[1]))
            
            # Save mask
            mask_path = "wound_mask_ai.png"
            cv2.imwrite(mask_path, mask_resized)
            
            # Display mask
            mask_img = Image.fromarray(mask_resized)
            mask_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            mask_photo = ImageTk.PhotoImage(mask_img)
            self.original_label.config(image=mask_photo)
            self.original_label.image = mask_photo
            
            self.status_label.config(text=f"AI Mask generated: {mask_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate AI mask: {e}")
            self.status_label.config(text="Error generating AI mask")
    
    def _generate_mask_basic(self):
        """Fallback basic mask generation"""
        try:
            self.status_label.config(text="Generating basic mask...")
            self.root.update()
            
            # Load image
            img = self._load_image_robust()
            if img is None:
                raise Exception("Could not load image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            
            # Apply threshold to create mask
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save mask
            mask_path = "wound_mask_basic.png"
            cv2.imwrite(mask_path, mask)
            
            # Display mask
            mask_img = Image.fromarray(mask)
            mask_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            mask_photo = ImageTk.PhotoImage(mask_img)
            self.original_label.config(image=mask_photo)
            self.original_label.image = mask_photo
            
            self.status_label.config(text=f"Basic mask generated: {mask_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate basic mask: {e}")
            self.status_label.config(text="Error generating basic mask")
    
    def generate_segment(self):
        """Generate wound segmentation using the trained model"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if self.model is None:
            messagebox.showwarning("Warning", "Model not loaded. Using basic CV method.")
            self._generate_segment_basic()
            return
        
        try:
            self.status_label.config(text="Generating segmentation using AI model...")
            self.root.update()
            
            # Load image
            img = self._load_image_robust()
            if img is None:
                raise Exception("Could not load image")
            
            # Preprocess image for model
            img_array = np.array(img, dtype=np.float32) / 255.0
            mh, mw = self._model_input_hw()
            img_resized = cv2.resize(img_array, (mw, mh))
            img_input = np.expand_dims(img_resized, 0)
            
            print("Running model prediction for segmentation...")
            # Get model prediction
            prediction = self.model.predict(img_input, verbose=0)[0, :, :, 0]
            
            # Convert probability to binary mask
            threshold = 0.5
            mask = (prediction > threshold).astype(np.uint8) * 255
            
            # Resize mask back to original size
            mask_resized = cv2.resize(mask, (img.size[0], img.size[1]))
            
            # Find contours from AI mask
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create segmentation overlay
            segment_img = np.array(img).copy()
            cv2.drawContours(segment_img, contours, -1, (0, 255, 0), 3)
            
            # Save segmentation
            segment_path = "wound_segment_ai.png"
            cv2.imwrite(segment_path, cv2.cvtColor(segment_img, cv2.COLOR_RGB2BGR))
            
            # Display segmentation
            segment_pil = Image.fromarray(segment_img)
            segment_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
            segment_photo = ImageTk.PhotoImage(segment_pil)
            self.original_label.config(image=segment_photo)
            self.original_label.image = segment_photo
            
            self.status_label.config(text=f"AI Segmentation generated: {segment_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate AI segmentation: {e}")
            self.status_label.config(text="Error generating AI segmentation")
    
    def _generate_segment_basic(self):
        """Fallback basic segmentation"""
        try:
            self.status_label.config(text="Generating basic segmentation...")
            self.root.update()
            
            # Load image
            img = self._load_image_robust()
            if img is None:
                raise Exception("Could not load image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create segmentation overlay
            segment_img = np.array(img).copy()
            cv2.drawContours(segment_img, contours, -1, (0, 255, 0), 2)
            
            # Save segmentation
            segment_path = "wound_segment_basic.png"
            cv2.imwrite(segment_path, cv2.cvtColor(segment_img, cv2.COLOR_RGB2BGR))
            
            # Display segmentation
            segment_pil = Image.fromarray(segment_img)
            segment_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
            segment_photo = ImageTk.PhotoImage(segment_pil)
            self.original_label.config(image=segment_photo)
            self.original_label.image = segment_photo
            
            self.status_label.config(text=f"Basic segmentation generated: {segment_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate basic segmentation: {e}")
            self.status_label.config(text="Error generating basic segmentation")
    
    def generate_heatmap(self):
        """Generate wound heatmap using the trained model"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if self.model is None:
            messagebox.showwarning("Warning", "Model not loaded. Using basic CV method.")
            self._generate_heatmap_basic()
            return
        
        try:
            self.status_label.config(text="Generating heatmap using AI model...")
            self.root.update()
            
            # Load image
            img = self._load_image_robust()
            if img is None:
                raise Exception("Could not load image")
            
            # Preprocess image for model
            img_array = np.array(img, dtype=np.float32) / 255.0
            mh, mw = self._model_input_hw()
            img_resized = cv2.resize(img_array, (mw, mh))
            img_input = np.expand_dims(img_resized, 0)
            
            print("Running model prediction for heatmap...")
            # Get model prediction (probability map)
            prediction = self.model.predict(img_input, verbose=0)[0, :, :, 0]
            
            # Resize prediction back to original size
            prediction_resized = cv2.resize(prediction, (img.size[0], img.size[1]))
            
            # Convert probability to 0-255 range for heatmap
            heatmap_intensity = (prediction_resized * 255).astype(np.uint8)
            
            # Create heatmap using color mapping
            heatmap = cv2.applyColorMap(heatmap_intensity, cv2.COLORMAP_JET)
            
            # Save heatmap
            heatmap_path = "wound_heatmap_ai.png"
            cv2.imwrite(heatmap_path, heatmap)
            
            # Display heatmap
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap_pil = Image.fromarray(heatmap_rgb)
            heatmap_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
            heatmap_photo = ImageTk.PhotoImage(heatmap_pil)
            self.original_label.config(image=heatmap_photo)
            self.original_label.image = heatmap_photo
            
            self.status_label.config(text=f"AI Heatmap generated: {heatmap_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate AI heatmap: {e}")
            self.status_label.config(text="Error generating AI heatmap")
    
    def _generate_heatmap_basic(self):
        """Fallback basic heatmap generation"""
        try:
            self.status_label.config(text="Generating basic heatmap...")
            self.root.update()
            
            # Load image
            img = self._load_image_robust()
            if img is None:
                raise Exception("Could not load image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur for smooth heatmap
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # Create heatmap using color mapping
            heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
            
            # Save heatmap
            heatmap_path = "wound_heatmap_basic.png"
            cv2.imwrite(heatmap_path, heatmap)
            
            # Display heatmap
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap_pil = Image.fromarray(heatmap_rgb)
            heatmap_pil.thumbnail((400, 400), Image.Resampling.LANCZOS)
            heatmap_photo = ImageTk.PhotoImage(heatmap_pil)
            self.original_label.config(image=heatmap_photo)
            self.original_label.image = heatmap_photo
            
            self.status_label.config(text=f"Basic heatmap generated: {heatmap_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate basic heatmap: {e}")
            self.status_label.config(text="Error generating basic heatmap")
    
    def _load_image_robust(self):
        """Load image with multiple fallback strategies"""
        if not self.current_image:
            return None
        
        # Strategy 1: PIL
        try:
            return Image.open(self.current_image).convert('RGB')
        except:
            pass
        
        # Strategy 2: OpenCV
        try:
            cv_img = cv2.imread(self.current_image)
            if cv_img is not None:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(cv_img)
        except:
            pass
        
        # Strategy 3: Bytes
        try:
            with open(self.current_image, 'rb') as f:
                img_data = f.read()
            return Image.open(io.BytesIO(img_data)).convert('RGB')
        except:
            pass
        
        return None
    
    def _load_real_model_keras3(self, model_path: str):
        """
        Load a Keras-3-saved .keras model using standalone Keras 3 (not tf.keras).
        Falls back to tf.keras only if Keras 3 import fails.
        """
        import os, traceback
        # Force Keras 3 frontend and TF backend
        os.environ["KERAS_BACKEND"] = "tensorflow"
        os.environ["TF_USE_LEGACY_KERAS"] = "0"

        try:
            import keras  # <-- standalone Keras 3
            try:
                keras.config.disable_traceback_filtering()
            except Exception:
                pass

            # Register custom objects (names must match what was saved)
            from keras.saving import register_keras_serializable
            @register_keras_serializable(package="Custom")
            def total_loss(*args, **kwargs):  # not used at inference; deserialization only
                return 0.0

            print(f"Using external Keras at: {keras.__file__} (version {keras.__version__})")

            # Load with safe_mode=False so serialized names are accepted
            m = keras.models.load_model(
                model_path,
                compile=False,
                safe_mode=False,
                custom_objects={"Custom>total_loss": total_loss, "total_loss": total_loss},
            )
            # Optional compile to prep predict graph; ignore if it fails
            try:
                m.compile(optimizer="adam", loss="binary_crossentropy")
            except Exception:
                pass

            return m

        except Exception as e:
            print("Keras 3 load failed:", e)
            traceback.print_exc()
            # Fallback: try tf.keras (will only work if the file is tf.keras-compatible)
            try:
                import tensorflow as tf
                print(f"Falling back to tf.keras (TF {tf.__version__})")
                m = tf.keras.models.load_model(
                    model_path,
                    compile=False,
                    custom_objects={"Custom>total_loss": (lambda *a, **k: 0.0),
                                    "total_loss": (lambda *a, **k: 0.0)},
                )
                return m
            except Exception as e2:
                print("tf.keras load also failed:", e2)
                traceback.print_exc()
                return None

    def _model_input_hw(self):
        """Return (H, W) from model.input_shape, fallback to (128,128)."""
        try:
            if self.model is not None and hasattr(self.model, "input_shape"):
                _, h, w, c = self.model.input_shape  # (None, H, W, C)
                if h and w:
                    return int(h), int(w)
        except Exception:
            pass
        return 128, 128

    def load_model(self):
        """Load the wound segmentation model (robust Keras-3 first, then tf.keras)."""
        try:
            self.status_label.config(text="Loading wound segmentation model...")
            self.root.update()

            model_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
            print(f"Loading model from: {model_path}")

            # Try standalone Keras 3 first (the file was saved with Keras 3)
            self.model = self._load_real_model_keras3(model_path)

            if self.model is None:
                print("❌ Model loading failed (both Keras 3 and tf.keras).")
                self.status_label.config(text="Model loading failed - Using basic CV methods")
                return False

            # Inspect and warm-up
            in_shape = getattr(self.model, "input_shape", None)
            out_shape = getattr(self.model, "output_shape", None)
            print("✅ Model loaded successfully!")
            print(f"Model input shape: {in_shape}")
            print(f"Model output shape: {out_shape}")
            print(f"Model type: {type(self.model)}")

            # Warm-up with zeros matching input shape (batch=1)
            if isinstance(in_shape, (list, tuple)) and len(in_shape) >= 4:
                h, w = in_shape[1], in_shape[2]
                if h is not None and w is not None:
                    import numpy as np
                    dummy = np.zeros((1, h, w, 3), dtype=np.float32)
                    try:
                        _ = self.model.predict(dummy, verbose=0)
                        print("🔥 Warm-up prediction successful.")
                    except Exception as e:
                        print("Warm-up failed (non-fatal):", e)

            if hasattr(self.model, 'predict'):
                self.status_label.config(text="Model loaded successfully - Ready for analysis")
                return True
            else:
                print("❌ Model object has no predict()")
                self.model = None
                self.status_label.config(text="Model loading failed - Using basic CV methods")
                return False

        except Exception as e:
            import traceback
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            self.model = None
            self.status_label.config(text="Model loading failed - Using basic CV methods")
            return False
    
    def select_image(self):
        """Select an image file with ultra-robust error handling"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
            )
            if file_path:
                self.status_label.config(text="Validating image...")
                self.root.update()
                
                # Check file size first
                file_size = os.path.getsize(file_path)
                self.file_info_label.config(text=f"File Info: {os.path.basename(file_path)} ({file_size:,} bytes)")
                
                if file_size > 50 * 1024 * 1024:  # 50MB limit
                    messagebox.showwarning("Warning", "Image file is very large (>50MB). This may cause timeouts.")
                
                # Test if we can load the image with multiple strategies
                success = False
                img = None
                
                # Strategy 1: Direct PIL load
                try:
                    with Image.open(file_path) as test_img:
                        test_img.verify()
                    success = True
                except Exception as e:
                    print(f"Strategy 1 failed: {e}")
                
                # Strategy 2: OpenCV load
                if not success:
                    try:
                        cv_img = cv2.imread(file_path)
                        if cv_img is not None:
                            # Convert BGR to RGB
                            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(cv_img)
                            success = True
                    except Exception as e:
                        print(f"Strategy 2 failed: {e}")
                
                # Strategy 3: Read as bytes and load
                if not success:
                    try:
                        with open(file_path, 'rb') as f:
                            img_data = f.read()
                        img = Image.open(io.BytesIO(img_data))
                        success = True
                    except Exception as e:
                        print(f"Strategy 3 failed: {e}")
                
                if success:
                    self.current_image = file_path
                    self.status_label.config(text=f"Image validated: {os.path.basename(file_path)}")
                    self.display_image()
                else:
                    messagebox.showerror("Error", "Could not load image with any method. File may be corrupted.")
                    self.status_label.config(text="Invalid image file")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Could not select image: {e}")
            self.status_label.config(text="Error selecting image")
    
    def display_image(self):
        """Display the selected image with ultra-robust loading"""
        if not self.current_image:
            return
        
        try:
            self.status_label.config(text="Loading image preview...")
            self.root.update()
            
            # Try multiple loading strategies
            img = None
            
            # Strategy 1: Direct PIL load
            try:
                img = Image.open(self.current_image).convert('RGB')
            except Exception as e:
                print(f"PIL load failed: {e}")
            
            # Strategy 2: OpenCV load
            if img is None:
                try:
                    cv_img = cv2.imread(self.current_image)
                    if cv_img is not None:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(cv_img)
                except Exception as e:
                    print(f"OpenCV load failed: {e}")
            
            # Strategy 3: Read as bytes
            if img is None:
                try:
                    with open(self.current_image, 'rb') as f:
                        img_data = f.read()
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                except Exception as e:
                    print(f"Bytes load failed: {e}")
            
            if img is None:
                raise Exception("All loading strategies failed")
            
            # Resize for display
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.original_label.config(image=photo)
            self.original_label.image = photo  # Keep a reference
            
            self.status_label.config(text=f"Image loaded: {os.path.basename(self.current_image)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
            self.status_label.config(text="Error loading image")
    
    def analyze_simple(self):
        """Perform simple skin color analysis in a separate thread"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Warning", "Analysis already in progress")
            return
        
        self.analysis_thread = threading.Thread(target=self._analyze_simple_thread)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def analyze_kmeans(self):
        """Perform K-means skin color analysis in a separate thread"""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            messagebox.showwarning("Warning", "Analysis already in progress")
            return
        
        self.analysis_thread = threading.Thread(target=self._analyze_kmeans_thread)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def _analyze_simple_thread(self):
        """Simple analysis in separate thread"""
        try:
            self.root.after(0, lambda: self.status_label.config(text="Performing simple analysis..."))
            self.root.after(0, lambda: self.progress.start())
            
            # Load image with multiple strategies
            img = None
            
            # Strategy 1: PIL
            try:
                img = Image.open(self.current_image).convert('RGB')
            except:
                pass
            
            # Strategy 2: OpenCV
            if img is None:
                try:
                    cv_img = cv2.imread(self.current_image)
                    if cv_img is not None:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(cv_img)
                except:
                    pass
            
            # Strategy 3: Bytes
            if img is None:
                try:
                    with open(self.current_image, 'rb') as f:
                        img_data = f.read()
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                except:
                    pass
            
            if img is None:
                raise Exception("Could not load image with any method")
            
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, 0)  # Add batch dimension
            
            print("Starting simple skin color analysis...")
            result = self.analyze_skin_color_simple(img_array)
            self.current_result = result
            
            # Update UI in main thread
            self.root.after(0, self.update_results)
            self.root.after(0, lambda: self.status_label.config(text="Simple analysis completed"))
            self.root.after(0, lambda: self.progress.stop())
            
        except Exception as e:
            error_msg = f"Simple analysis failed: {e}"
            print(error_msg)
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_label.config(text="Analysis failed"))
            self.root.after(0, lambda: self.progress.stop())
    
    def _analyze_kmeans_thread(self):
        """K-means analysis in separate thread"""
        try:
            self.root.after(0, lambda: self.status_label.config(text="Performing K-means analysis..."))
            self.root.after(0, lambda: self.progress.start())
            
            # Load image with multiple strategies
            img = None
            
            # Strategy 1: PIL
            try:
                img = Image.open(self.current_image).convert('RGB')
            except:
                pass
            
            # Strategy 2: OpenCV
            if img is None:
                try:
                    cv_img = cv2.imread(self.current_image)
                    if cv_img is not None:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(cv_img)
                except:
                    pass
            
            # Strategy 3: Bytes
            if img is None:
                try:
                    with open(self.current_image, 'rb') as f:
                        img_data = f.read()
                    img = Image.open(io.BytesIO(img_data)).convert('RGB')
                except:
                    pass
            
            if img is None:
                raise Exception("Could not load image with any method")
            
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, 0)  # Add batch dimension
            
            print("Starting K-means skin color analysis...")
            result = self.analyze_skin_color_kmeans(img_array)
            self.current_result = result
            
            # Update UI in main thread
            self.root.after(0, self.update_results)
            self.root.after(0, lambda: self.status_label.config(text="K-means analysis completed"))
            self.root.after(0, lambda: self.progress.stop())
            
        except Exception as e:
            error_msg = f"K-means analysis failed: {e}"
            print(error_msg)
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_label.config(text="Analysis failed"))
            self.root.after(0, lambda: self.progress.stop())
    
    def analyze_skin_color_simple(self, image_array):
        """Simple skin color analysis"""
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
            
            print(f"✅ Skin tone level: {skin_tone_level}")
            print(f"✅ Skin tone description: {skin_tone_desc}")
            
            return {
                'skin_color_rgb': avg_color,
                'skin_tone_level': skin_tone_level,
                'skin_tone_description': skin_tone_desc,
                'brightness': brightness,
                'method': 'simple_analysis'
            }
            
        except Exception as e:
            print(f"❌ Error in simple analysis: {e}")
            traceback.print_exc()
            return {
                'skin_color_rgb': [0, 0, 0],
                'skin_tone_level': 5,
                'skin_tone_description': 'Unknown',
                'brightness': 0.5,
                'method': 'error'
            }
    
    def analyze_skin_color_kmeans(self, image_array):
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
            
            print(f"✅ Dominant skin color: RGB({dominant_skin_color[0]:.2f}, {dominant_skin_color[1]:.2f}, {dominant_skin_color[2]:.2f})")
            
            # Classify skin tone based on RGB values
            r, g, b = dominant_skin_color
            
            # Calculate skin tone level (0-10 scale)
            brightness = (r + g + b) / 3 / 255.0
            
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
            
            print(f"✅ Skin tone level: {skin_tone_level}")
            print(f"✅ Skin tone description: {skin_tone_desc}")
            
            return {
                'skin_color_rgb': dominant_skin_color,
                'skin_tone_level': skin_tone_level,
                'skin_tone_description': skin_tone_desc,
                'all_colors': colors,
                'method': 'kmeans_clustering'
            }
            
        except Exception as e:
            print(f"❌ Error in K-means analysis: {e}")
            traceback.print_exc()
            return {
                'skin_color_rgb': [0, 0, 0],
                'skin_tone_level': 5,
                'skin_tone_description': 'Unknown',
                'all_colors': [],
                'method': 'error'
            }
    
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
    app = UltraRobustSkinGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()