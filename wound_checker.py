import os
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import cv2
import torch
from torchvision import transforms

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = '/Users/nadiajelani/projects/wound-segmentation/models'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'wound_classifier.h5')  # ResNet50
UNET_PATH = os.path.join(MODEL_DIR, 'best_unet_wound_model.h5')  # U-Net
MEDSAM_PATH = os.path.join(MODEL_DIR, 'best_medsam_wound_model.pth')  # MedSAM
IMG_HEIGHT, IMG_WIDTH = 224, 224
UNET_INPUT_SIZE = (256, 256)  # Common U-Net input size
MEDSAM_INPUT_SIZE = (1024, 1024)  # Typical MedSAM input size
THRESHOLD = 0.5

# Load models
try:
    classifier_model = load_model(CLASSIFIER_PATH)
    logger.info("Loaded ResNet50 classifier from %s", CLASSIFIER_PATH)
    unet_model = load_model(UNET_PATH, compile=False)
    logger.info("Loaded U-Net model from %s", UNET_PATH)
    # Load MedSAM with PyTorch
    medsam_state_dict = torch.load(MEDSAM_PATH, map_location=torch.device('cpu'))
    # Placeholder: Assume a simple MedSAM model structure (replace with actual architecture)
    medsam_model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 1, kernel_size=1)
    )
    medsam_model.load_state_dict(medsam_state_dict)
    medsam_model.eval()
    logger.info("Loaded MedSAM model from %s", MEDSAM_PATH)
except Exception as e:
    logger.error("Failed to load models: %s", str(e))
    raise

def preprocess_image(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Preprocess the image for classification or segmentation."""
    try:
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error("Error preprocessing image %s: %s", image_path, str(e))
        return None

def classify_image(model, img_array):
    """Classify the image as wound or non-wound using ResNet50."""
    prediction = model.predict(img_array, verbose=0)
    is_wound = prediction[0][0] > THRESHOLD
    confidence = prediction[0][0] if is_wound else 1 - prediction[0][0]
    return is_wound, confidence

def torch_to_tf_tensor(torch_tensor):
    """Convert PyTorch tensor to TensorFlow tensor."""
    return tf.convert_to_tensor(torch_tensor.numpy(), dtype=tf.float32)

def segment_wound(unet_model, medsam_model, img_array):
    """Segment the wound using U-Net and MedSAM."""
    # U-Net segmentation
    img_unet = tf.image.resize(img_array[0], UNET_INPUT_SIZE)
    img_unet = np.expand_dims(img_unet, axis=0)
    unet_mask = unet_model.predict(img_unet, verbose=0)
    unet_mask = (unet_mask > 0.5).astype(np.uint8)

    # MedSAM segmentation
    img_medsam = tf.image.resize(img_array[0], MEDSAM_INPUT_SIZE)
    img_medsam = np.transpose(img_medsam, (0, 3, 1, 2))  # Convert to CHW format for PyTorch
    img_medsam_torch = torch.from_numpy(img_medsam).float()
    with torch.no_grad():
        medsam_mask = medsam_model(img_medsam_torch)
    medsam_mask = (medsam_mask > 0.5).float().numpy()
    medsam_mask = np.transpose(medsam_mask, (0, 2, 3, 1))  # Convert back to HWC
    medsam_mask = tf.image.resize(medsam_mask, UNET_INPUT_SIZE).numpy()

    # Combine masks (average for simplicity)
    combined_mask = np.mean([unet_mask[0], medsam_mask[0]], axis=0)
    combined_mask = (combined_mask > 0.5).astype(np.uint8)
    return combined_mask

def save_results(image_path, is_wound, confidence, mask=None):
    """Save classification and segmentation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(os.path.dirname(image_path), 'results')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f'{base_name}_result_{timestamp}.txt'), 'w') as f:
        f.write(f"Date: {datetime.now().strftime('%I:%M %p %Z, %B %d, %Y')}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Prediction: {'Wound' if is_wound else 'Non-wound'}\n")
        f.write(f"Confidence: {confidence:.2f}\n")
    logger.info("Saved classification result to %s", os.path.join(output_dir, f'{base_name}_result_{timestamp}.txt'))

    if is_wound and mask is not None:
        mask_resized = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        plt.imsave(os.path.join(output_dir, f'{base_name}_mask_{timestamp}.png'), mask_resized, cmap='gray')
        logger.info("Saved segmentation mask to %s", os.path.join(output_dir, f'{base_name}_mask_{timestamp}.png'))

def process_image(image_path):
    """Process the uploaded or captured image."""
    img_array = preprocess_image(image_path)
    if img_array is None:
        messagebox.showerror("Error", f"Failed to process image: {image_path}")
        return

    is_wound, confidence = classify_image(classifier_model, img_array)
    result_text = f"Prediction: {'Wound' if is_wound else 'Non-wound'}\nConfidence: {confidence:.2f}"
    messagebox.showinfo("Result", result_text)
    logger.info("%s - Confidence: %.2f", "Wound" if is_wound else "Non-wound", confidence)

    if is_wound:
        mask = segment_wound(unet_model, medsam_model, img_array)
        save_results(image_path, is_wound, confidence, mask)
        plt.imshow(mask, cmap='gray')
        plt.title('Combined Wound Segmentation Mask (U-Net + MedSAM)')
        plt.axis('off')
        plt.show()
    else:
        save_results(image_path, is_wound, confidence)

def upload_image():
    """Handle image upload via GUI."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path:
        process_image(file_path)

def capture_image():
    """Handle image capture (placeholder for webcam)."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(os.path.dirname(CLASSIFIER_PATH), f'captured_image_{timestamp}.jpg')
        cv2.imwrite(image_path, frame)
        cap.release()
        process_image(image_path)
    else:
        messagebox.showerror("Error", "Failed to capture image")
        cap.release()

# GUI Setup
root = tk.Tk()
root.title("Wound Checker")
root.geometry("300x200")

upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=10)

capture_btn = tk.Button(root, text="Capture Image", command=capture_image)
capture_btn.pack(pady=10)

root.mainloop()