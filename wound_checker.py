# wound_checker.py
import os
import sys
import time
import json
import logging
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from skimage.filters.rank import entropy
from skimage.morphology import disk

# Import functions from wound_medsam.py
from wound_medsam import build_unet, analyze_single_image, load_medsam_model, medsam_segment, predict_healing_potential, generate_clinical_report, explainable_ai, visualize_results

# Debug: Confirm module is loaded
print("Successfully imported functions from wound_medsam:", build_unet, analyze_single_image, load_medsam_model, medsam_segment, predict_healing_potential, generate_clinical_report, explainable_ai, visualize_results)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set current date and time
current_time = "06:55 PM BST, May 25, 2025"

class WoundCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Wound Checker - {current_time}")
        self.root.geometry("600x500")
        self.root.configure(bg="#f6f9fc")

        # Variables
        self.image_path = tk.StringVar()
        self.has_diabetes = tk.BooleanVar(value=False)
        self.age = tk.StringVar(value="0")
        self.other_diseases = tk.StringVar(value="")
        self.output_dir = "wound_results"
        self.unet_model_path = "/Users/nadiajelani/projects/wound-segmentation/models/best_unet_wound_model.h5"
        self.medsam_model_path = "/Users/nadiajelani/projects/wound-segmentation/models/medsam.pth"

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        welcome_label = tk.Label(self.root, text=f"Welcome to Wound Checker - {current_time}", font=("Arial", 16, "bold"), bg="#f6f9fc", fg="#3a5199")
        welcome_label.pack(pady=10)

        instructions = tk.Label(self.root, text="Upload a wound photo or take a live photo, tell us about any health conditions, and get a simple report!", 
                               font=("Arial", 10), bg="#f6f9fc", fg="#555", wraplength=500)
        instructions.pack(pady=5)

        image_frame = tk.Frame(self.root, bg="#f6f9fc")
        image_frame.pack(pady=10, fill="x", padx=20)

        tk.Label(image_frame, text="Pick Your Wound Photo:", font=("Arial", 10), bg="#f6f9fc").pack(side="left")
        tk.Entry(image_frame, textvariable=self.image_path, width=40).pack(side="left", padx=5)

        button_frame = tk.Frame(image_frame, bg="#f6f9fc")
        button_frame.pack(side="left", padx=5)

        tk.Button(button_frame, text="Choose File", command=self.browse_image, 
                  bg="#1e3a8a", fg="#ffffff", activebackground="#3b82f6", activeforeground="#ffffff", 
                  font=("Arial", 11, "bold"), relief="raised", bd=3, padx=10, pady=5).pack(side="top", pady=2)

        tk.Button(button_frame, text="Take Photo", command=self.capture_photo,
                  bg="#2563eb", fg="#ffffff", activebackground="#60a5fa", activeforeground="#ffffff", 
                  font=("Arial", 10, "bold"), relief="raised", bd=3, padx=10, pady=5).pack(side="top", pady=2)

        tk.Label(image_frame, text="(e.g., a photo from your phone or live capture)", font=("Arial", 8), fg="#777", bg="#f6f9fc").pack(side="left", padx=5)

        health_frame = tk.Frame(self.root, bg="#f6f9fc")
        health_frame.pack(pady=10, fill="x", padx=20)
        tk.Label(health_frame, text="Health Information:", font=("Arial", 10, "bold"), bg="#f6f9fc").pack(anchor="w")

        tk.Checkbutton(health_frame, text="I have diabetes", variable=self.has_diabetes, bg="#f6f9fc", font=("Arial", 10)).pack(anchor="w")

        age_frame = tk.Frame(health_frame, bg="#f6f9fc")
        age_frame.pack(fill="x", pady=2)
        tk.Label(age_frame, text="Age (Optional):", font=("Arial", 10), bg="#f6f9fc").pack(side="left")
        tk.Entry(age_frame, textvariable=self.age, width=10).pack(side="left", padx=5)
        tk.Label(age_frame, text="(e.g., 45)", font=("Arial", 8), fg="#777", bg="#f6f9fc").pack(side="left")

        other_frame = tk.Frame(health_frame, bg="#f6f9fc")
        other_frame.pack(fill="x", pady=2)
        tk.Label(other_frame, text="Other Conditions (Optional):", font=("Arial", 10), bg="#f6f9fc").pack(side="left")
        tk.Entry(other_frame, textvariable=self.other_diseases, width=40).pack(side="left", padx=5)
        tk.Label(other_frame, text="(e.g., hypertension, asthma)", font=("Arial", 8), fg="#777", bg="#f6f9fc").pack(side="left")

        tk.Button(self.root, text="Check Wound", command=self.process_image, 
                  bg="#15803d", fg="#ffffff", activebackground="#16a34a", activeforeground="#ffffff", 
                  font=("Arial", 14, "bold"), relief="raised", bd=3, padx=15, pady=8).pack(pady=20)

        self.status_label = tk.Label(self.root, text="Status: Ready to start!", font=("Arial", 10), bg="#f6f9fc", fg="#333")
        self.status_label.pack(pady=5)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Pick Your Wound Photo",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image_path.set(file_path)
            self.status_label.config(text=f"Status: Photo selected: {os.path.basename(file_path)}")

    def capture_photo(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam. Ensure your camera is connected and permissions are granted in System Preferences > Security & Privacy > Camera.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        brightness = 1.0
        contrast = 1.0
        flip_mode = 0  # 0: none, 1: horizontal, 2: vertical

        instructions = """
        Controls:
        - Up / Down Arrow: Adjust Brightness
        - Left / Right Arrow: Adjust Contrast
        - 'f': Flip Horizontal
        - 'v': Flip Vertical
        - 'n': No Flip
        - 's': Save Photo
        - 'q': Quit Preview
        """
        messagebox.showinfo("Live Camera Instructions", instructions)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    messagebox.showerror("Camera Error", "Failed to capture image from webcam.")
                    break

                frame_adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness * 50)

                if flip_mode == 1:
                    frame_adjusted = cv2.flip(frame_adjusted, 1)
                elif flip_mode == 2:
                    frame_adjusted = cv2.flip(frame_adjusted, 0)

                overlay_text = f"Brightness: {brightness:.1f}, Contrast: {contrast:.1f}"
                cv2.putText(frame_adjusted, overlay_text, (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame_adjusted, "Press 's' to Save, 'q' to Quit", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

                cv2.imshow("MacBook Camera - Adjust and Capture", frame_adjusted)
                key = cv2.waitKey(1) & 0xFF

                if key == 82:  # Up arrow
                    brightness = min(brightness + 0.1, 3.0)
                elif key == 84:  # Down arrow
                    brightness = max(brightness - 0.1, 0.1)
                elif key == 81:  # Left arrow
                    contrast = max(contrast - 0.1, 0.1)
                elif key == 83:  # Right arrow
                    contrast = min(contrast + 0.1, 3.0)
                elif key == ord('f'):
                    flip_mode = 1
                elif key == ord('v'):
                    flip_mode = 2
                elif key == ord('n'):
                    flip_mode = 0
                elif key == ord('s'):
                    img_path = os.path.join(self.output_dir, "live_capture.jpg")
                    os.makedirs(self.output_dir, exist_ok=True)
                    cv2.imwrite(img_path, frame_adjusted)
                    self.image_path.set(img_path)
                    self.status_label.config(text="Status: Live photo captured")
                    if messagebox.askyesno("Review Photo", "Photo captured! Do you want to use this photo for analysis? Click 'No' to retake."):
                        break
                    else:
                        self.status_label.config(text="Status: Retake photo")
                        continue
                elif key == ord('q'):
                    self.status_label.config(text="Status: Camera preview closed")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def verify_models(self):
        if not os.path.exists(self.unet_model_path):
            messagebox.showwarning("Oops!", f"Couldn’t find the wound analysis model at {self.unet_model_path}.\nPlease select it manually.")
            file_path = filedialog.askopenfilename(
                title="Pick Wound Analysis Model",
                filetypes=[("Model Files", "*.keras *.h5")]
            )
            if file_path and os.path.exists(file_path):
                self.unet_model_path = file_path
                self.status_label.config(text=f"Status: Model selected: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Oh No!", "No model selected. Please provide a valid model file.")
                return False

        if not os.path.exists(self.medsam_model_path):
            logger.info(f"MedSAM model not found at {self.medsam_model_path}. Proceeding without enhanced analysis.")
            self.medsam_model_path = None
        return True

    def has_wound_features(self, img):
        """Enhanced check for wound-like features using color, edge, and texture analysis."""
        # Resize image for faster processing
        img_small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        # Color analysis in HSV space
        img_hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([40, 255, 255])
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])

        mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        mask_black = cv2.inRange(img_hsv, lower_black, upper_black)
        mask = mask_red1 | mask_red2 | mask_yellow | mask_black

        # Ensure a minimum area of wound-like colors
        color_area = np.sum(mask) / 255
        if color_area < 500:  # Minimum 500 pixels of wound-like colors
            return False

        color_ratio = color_area / mask.size
        if color_ratio < 0.1:  # Stricter threshold
            return False

        # Edge detection
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        edge_ratio = np.sum(edges) / (255 * edges.size)
        if edge_ratio < 0.02:  # Stricter threshold
            return False

        # Texture analysis using entropy (irregular patterns in wounds)
        entropy_img = entropy(img_gray, disk(5))
        entropy_mean = np.mean(entropy_img)
        if entropy_mean < 3.0:  # Wounds typically have higher entropy due to irregular texture
            return False

        return True

    def process_image(self):
        image_path = self.image_path.get()
        age = self.age.get()
        other_diseases = self.other_diseases.get().strip()

        if not image_path or not os.path.exists(image_path):
            messagebox.showwarning("Oops!", "Please pick or capture a wound photo to analyze.")
            return

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.has_wound_features(img):
            self.status_label.config(text="Status: No wound detected in the image.")
            messagebox.showwarning("No Wound Detected", "The selected image does not appear to contain a wound. Please upload or capture a different image.")
            return

        if not self.verify_models():
            return

        try:
            age_int = int(age) if age.isdigit() else 0
        except ValueError:
            messagebox.showwarning("Oops!", "Age should be a number. Using default (0).")
            age_int = 0

        patient_info = {
            "has_diabetes": "yes" if self.has_diabetes.get() else "no",
            "age": age_int,
            "other_diseases": other_diseases if other_diseases else "none"
        }

        self.status_label.config(text="Status: Checking your wound...")

        try:
            logger.info(f"Loading model from {self.unet_model_path}")
            model = build_unet(input_shape=(128, 128, 3))
            model.load_weights(self.unet_model_path)

            medsam_path = self.medsam_model_path
            if medsam_path:
                logger.info(f"Loading enhanced model from {medsam_path}")
                load_medsam_model(medsam_path)

            os.makedirs(self.output_dir, exist_ok=True)
            report_dir = os.path.join(self.output_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)

            start_time = time.time()

            img_float = img.astype(np.float32) / 255.0
            img_resized = tf.image.resize(img_float, (128, 128), method='bilinear')
            img_resized = tf.expand_dims(img_resized, 0)

            unet_pred = model.predict(img_resized, verbose=0)[0, ..., 0]
            unet_mask = (unet_pred > 0.5).astype(np.uint8)

            wound_area = np.sum(unet_mask)
            if wound_area < 1000:  # Stricter threshold for segmentation
                self.status_label.config(text="Status: No significant wound detected.")
                messagebox.showwarning("No Wound Detected", "The image does not contain a significant wound area. Please upload or capture a different image.")
                return

            analyze_single_image(image_path, model, medsam_path, patient_info, self.output_dir)

            end_time = time.time()
            runtime = end_time - start_time
            logger.info(f"Processing completed in {runtime:.2f} seconds")

            medsam_pred = medsam_segment(img, medsam_path) if medsam_path else None
            medsam_mask = tf.image.resize(medsam_pred[..., None], (128, 128), method='nearest').numpy().squeeze().astype(np.uint8) if medsam_pred is not None else unet_mask
            hybrid_mask = unet_mask if medsam_pred is None else (unet_mask + medsam_mask > 0).astype(np.uint8)
            hybrid_mask_resized = tf.image.resize(hybrid_mask[..., None], img.shape[:2], method='nearest').numpy().squeeze().astype(np.uint8)

            canny = cv2.Canny(cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), 100, 200) / 255.0
            heatmap = explainable_ai(model, img_resized[0])
            visualize_results(img_resized[0], np.zeros_like(hybrid_mask_resized), hybrid_mask_resized, img_float, heatmap, canny, 0, self.output_dir, medsam_pred)

            self.status_label.config(text=f"Status: Done! Took {runtime:.2f} seconds")
            messagebox.showinfo("Great News!", f"Your wound check is complete!\nSee photos: {os.path.join(self.output_dir, 'val_predictions')}\nReport: {report_dir}\nCheck the log for more details.")

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            self.status_label.config(text="Status: Something went wrong!")
            messagebox.showerror("Oh No!", f"Sorry, we couldn’t check the wound. Error: {str(e)}")

def main():
    root = tk.Tk()
    app = WoundCheckerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()