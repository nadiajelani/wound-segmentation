import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import math
from skimage import color
from tkinter import Tk, filedialog, messagebox

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Wound Detection and Analysis")
parser.add_argument("--sensitivity", type=str, default="medium", choices=["low", "medium", "high"],
                    help="Sensitivity level for segmentation threshold (low, medium, high)")
parser.add_argument("--video_path", type=str, default=None,
                    help="Path to input video (optional, e.g., .mov)")
parser.add_argument("--test_type", type=str, default=None, choices=["v", "vi"],
                    help="Test with Type V or VI images (optional)")
args = parser.parse_args()

# === Create GUI for file selection ===
root = Tk()
root.withdraw()  # Hide the main window

# === Skin Type Estimation Functions ===
def estimate_skin_type(image_path, method='hybrid'):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = color.rgb2lab(img_rgb)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    avg_hue = np.mean(img_hsv[:, :, 0])
    avg_sat = np.mean(img_hsv[:, :, 1])
    avg_val = np.mean(img_hsv[:, :, 2])
    L = img_lab[:, :, 0]
    B = img_lab[:, :, 2]
    L = np.where(L != 0, L, np.nan)
    B = np.where(B != 0, B, np.nan)
    ita = math.atan2(np.nanmean(L) - 50, np.nanmean(B)) * (180 / np.pi)
    if method == 'ita':
        return ita_to_type(ita)
    elif method == 'hsv':
        return hsv_to_type(avg_hue, avg_sat, avg_val)
    elif method == 'hybrid':
        if avg_sat > 20 and avg_val > 50:
            ita_type = ita_to_type(ita)
            hsv_type = hsv_to_type(avg_hue, avg_sat, avg_val)
            return ita_type if ita_type == hsv_type else f"{ita_type} or {hsv_type}"
        else:
            return "Unknown"

def ita_to_type(ita):
    if ita >= 45: return "Type I"
    elif ita > 28: return "Type II"
    elif ita > 17: return "Type III"
    elif ita > 5: return "Type IV"
    elif ita > -20: return "Type V"
    else: return "Type VI"

def hsv_to_type(hue, sat, val):
    if sat < 20 or val < 50: return "Unknown"
    if hue < 15 or hue > 170: return "Type I-II"
    elif 15 <= hue < 35: return "Type III"
    elif 35 <= hue < 55: return "Type IV"
    elif 55 <= hue < 75: return "Type V"
    elif hue >= 75: return "Type VI"
    return "Unknown"

# === Interactive input for normal skin ===
print("Step 1: Please select a normal skin image for calibration.")
messagebox.showinfo("Why a Skin Image?", 
    "We need a normal skin image to estimate your Fitzpatrick skin type (I–VI). This helps adjust the model for accurate wound and bleeding detection across different skin tones and lighting conditions.")
max_retries = 3
for attempt in range(max_retries):
    skin_file_path = filedialog.askopenfilename(
        title="Select a normal skin image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    if not skin_file_path:
        print("No skin image selected. Exiting.")
        exit()
    try:
        skin_tone = estimate_skin_type(skin_file_path, method='hybrid')
        if skin_tone == "Unknown":
            if attempt < max_retries - 1:
                messagebox.showwarning("Skin Tone Detection Failed",
                                      f"Attempt {attempt + 1}/{max_retries}: Skin tone could not be detected. "
                                      "Please ensure good lighting, use a healthy skin image, and retry.")
            else:
                messagebox.showwarning("Skin Tone Detection Failed",
                                      "Maximum retries reached. Proceeding with reduced accuracy.")
                skin_tone = "Unknown"
                break
        else:
            print(f"✅ Detected skin tone: {skin_tone}")
            break
    except ValueError as e:
        print(f"Error: {e} (Attempt {attempt + 1}/{max_retries}).")
        if attempt == max_retries - 1:
            messagebox.showwarning("Skin Tone Detection Failed",
                                  "Maximum retries reached. Proceeding with reduced accuracy.")
            skin_tone = "Unknown"
            break

print(f"Using skin tone: {skin_tone}")

# === Option to select video or image for wound ===
choice = messagebox.askquestion("Next Step", 
    "Would you like to analyze a video (e.g., .mov) or an image (e.g., .jpg, .png) of the wound?",
    icon='question', default='yes', type='yesno')
if choice == 'yes':
    print("Step 2: Please select a wound video (e.g., .mov) for analysis.")
    file_types = [
        ("Video files", "*.mov *.mp4 *.avi"),
        ("All files", "*.*")
    ]
else:
    print("Step 2: Please select a wound image (e.g., .jpg, .png) for analysis.")
    file_types = [
        ("Image files", "*.jpg *.jpeg *.png"),
        ("All files", "*.*")
    ]

file_path = None
if args.video_path and os.path.exists(args.video_path):
    file_path = args.video_path
    print(f"Using video path: {file_path}")
elif args.test_type:
    test_dir = f"/Users/nadiajelani/projects/wound-segmentation/data/type_{args.test_type}/"
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if image_files:
        file_path = os.path.join(test_dir, image_files[0])
        print(f"Using test image: {file_path}")
    else:
        print(f"No images found in {test_dir}. Exiting.")
        exit()
else:
    file_path = filedialog.askopenfilename(
        title=f"Select a wound {('video' if choice == 'yes' else 'image')}",
        filetypes=file_types,
        initialdir="/"
    )
    if not file_path:
        print("No file selected. Exiting.")
        exit()
    print(f"Selected file: {file_path}")

# === File paths and parameters ===
seg_model_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_wound_segmentation.keras"
cls_model_path = "/Users/nadiajelani/projects/wound-segmentation/models/wound_classifier_optimized.h5"
output_dir = "/Users/nadiajelani/projects/wound-segmentation/wound_progress_report/"
os.makedirs(output_dir, exist_ok=True)

# === Load segmentation model ===
seg_model = tf.keras.models.load_model(seg_model_path, compile=False)

# === Reconstruct classifier model architecture ===
base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
cls_model = tf.keras.models.Model(inputs=base_model.input, outputs=x)

# === Load weights with detailed logging ===
print("Loading weights from:", cls_model_path)
cls_model.load_weights(cls_model_path, by_name=True)
for layer in cls_model.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}, Shape: {layer.output_shape}")

# === Process input (image or video) ===
def process_frame(frame, skin_tone=None):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if skin_tone in ["Type V", "Type VI"] or skin_tone == "Unknown":
        image_rgb = cv2.convertScaleAbs(image_rgb, alpha=1.5, beta=0)
    image_resized = cv2.resize(image_rgb, (224, 224)) / 255.0
    return image_rgb, np.expand_dims(image_resized, axis=0)

if args.video_path and os.path.exists(args.video_path):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}. Check codec compatibility (e.g., install FFmpeg for .mov support) or file path.")
        exit()
    frames_rgb = []
    frames_tensor = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb, input_tensor = process_frame(frame, skin_tone)
        frames_rgb.append(image_rgb)
        frames_tensor.append(input_tensor)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    cap.release()
    input_tensor = np.concatenate(frames_tensor, axis=0)
    mode = "video"
    print(f"Processed {frame_count} frames from video")
else:
    if file_path.lower().endswith(('.mov', '.mp4', '.avi')):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {file_path}. Check codec compatibility (e.g., install FFmpeg for .mov support) or file path.")
            exit()
        frames_rgb = []
        frames_tensor = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb, input_tensor = process_frame(frame, skin_tone)
            frames_rgb.append(image_rgb)
            frames_tensor.append(input_tensor)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")
        cap.release()
        input_tensor = np.concatenate(frames_tensor, axis=0)
        mode = "video"
        print(f"Processed {frame_count} frames from video")
    else:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not load image file {file_path}")
            exit()
        image_rgb, input_tensor = process_frame(image, skin_tone)
        mode = "image"

# === Predict with classifier ===
cls_preds = cls_model.predict(input_tensor)
if mode == "video":
    cls_preds = cls_preds.reshape(-1)
    max_confidence_idx = np.argmax(cls_preds)
    wound_confidence = cls_preds[max_confidence_idx]
    is_wound = wound_confidence > 0.5
    gradcam_strength = np.max(heatmap) if 'heatmap' in locals() else 0
    if gradcam_strength > 0.5 and wound_confidence > 0.3:
        print("🧠 Grad-CAM activation is strong. Overriding low classifier confidence.")
        is_wound = True
    wound_label = "Wound" if is_wound else "Non-Wound"
    input_tensor = input_tensor[max_confidence_idx:max_confidence_idx + 1]
    image_rgb = frames_rgb[max_confidence_idx]
    print(f"Classifier Result: {wound_label} detected with max confidence {wound_confidence:.2f} in frame {max_confidence_idx}")
else:
    wound_confidence = float(cls_preds[0][0])
    is_wound = wound_confidence > 0.5
    gradcam_strength = np.max(heatmap) if 'heatmap' in locals() else 0
    if gradcam_strength > 0.5 and wound_confidence > 0.3:
        print("🧠 Grad-CAM activation is strong. Overriding low classifier confidence.")
        is_wound = True
    wound_label = "Wound" if is_wound else "Non-Wound"
    print(f"Classifier Result: {wound_label} detected with confidence {wound_confidence:.2f}")

# === Conditional processing with debugging ===
if not is_wound:
    print(f"⚠️ Warning: Image or video classified as {wound_label} with confidence {wound_confidence:.2f}.")
    print("This is not recognized as a wound. Please upload a clear image or video of a wound.")
    print("Exiting without further processing.")
    exit()

# === Grad-CAM setup for segmentation model ===
last_conv_layer_name = None
for layer in reversed(seg_model.layers):
    if len(layer.output_shape) == 4:
        last_conv_layer_name = layer.name
        break
print(f"Using last conv layer: {last_conv_layer_name}")

grad_model = tf.keras.models.Model(
    [seg_model.inputs],
    [seg_model.get_layer(last_conv_layer_name).output, seg_model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(input_tensor)
    loss = tf.reduce_mean(predictions)

grads = tape.gradient(loss, conv_outputs)[0]
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

# === Dynamic threshold handling ===
pred_mask = seg_model.predict(input_tensor)[0]
max_pred = np.max(pred_mask)
print(f"Raw prediction max: {max_pred:.4f}, min: {np.min(pred_mask):.4f}")

sensitivity_map = {"low": 0.3, "medium": 0.5, "high": 0.7}
threshold_multiplier = sensitivity_map[args.sensitivity]
dynamic_threshold = max_pred * threshold_multiplier
print(f"Using dynamic threshold: {dynamic_threshold:.4f} based on sensitivity {args.sensitivity}")

pred_bin = (pred_mask > dynamic_threshold).astype(np.uint8) * 255
if np.sum(pred_bin > 0) > 0:
    print(f"Segmentation detected with dynamic threshold {dynamic_threshold:.4f}")
else:
    print("⚠️ No segmentation detected with dynamic threshold. Using lowest threshold (0.1).")
    pred_bin = (pred_mask > 0.1).astype(np.uint8) * 255

# === Enhanced bleeding detection with dark skin support ===
pred_bin_resized = cv2.resize(pred_bin, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
pred_bin_resized_3d = np.expand_dims(pred_bin_resized, axis=2)
masked_image = pred_bin_resized_3d * image_rgb / 255.0
masked_image_uint8 = masked_image.astype(np.uint8)

if skin_tone in ["Type V", "Type VI"]:
    lower_red1 = np.array([0, 40, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 40, 30])
    upper_red2 = np.array([180, 255, 255])
else:
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

masked_hsv = cv2.cvtColor(masked_image_uint8, cv2.COLOR_RGB2HSV)
mask1 = cv2.inRange(masked_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(masked_hsv, lower_red2, upper_red2)
bleeding_mask_hsv = cv2.bitwise_or(mask1, mask2)

img_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
L, A, B = cv2.split(img_lab)
lab_mask = (A > 135) & (L < 100)
lab_mask = lab_mask.astype(np.uint8) * 255
bleeding_mask_lab = cv2.bitwise_and(lab_mask, pred_bin_resized)
bleeding_mask_combined = cv2.bitwise_or(bleeding_mask_hsv, bleeding_mask_lab)
bleed_area = np.sum(bleeding_mask_combined > 0)
wound_area = np.sum(pred_bin_resized > 0)
bleed_ratio = bleed_area / (wound_area + 1e-6)
bleeding_detected = bleed_ratio > 0.25
if bleeding_detected:
    print(f"⚠️ High bleeding ratio detected: {bleed_ratio:.2f}. Suggest immediate medical attention.")

# === Mask + Grad-CAM Fusion ===
if np.sum(pred_bin) < 100 and np.max(heatmap) > 0.5:
    print("🧠 Using Grad-CAM heatmap as fallback due to weak mask.")
    gradcam_mask = (heatmap > 0.3).astype(np.uint8) * 255
    pred_bin = gradcam_mask

# === Convert heatmap to overlay with bleeding emphasis ===
heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image_rgb, 0.6, heatmap_color, 0.4, 0)
bleed_overlay = cv2.bitwise_and(image_rgb, image_rgb, mask=bleeding_mask_combined)
overlay[np.where(bleeding_mask_combined > 0)] = bleed_overlay[np.where(bleeding_mask_combined > 0)]

# === Save output masks ===
cv2.imwrite(os.path.join(output_dir, "simclr_mask.png"), pred_bin)

# === Plot results ===
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb / 255.0)
plt.title(f"Original Image\nClassifier: {wound_label} (Confidence: {wound_confidence:.2f})")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pred_bin, cmap='gray')
plt.title("Predicted Mask (SimCLR-U-Net)" + (" (Failed)" if np.sum(pred_bin > 0) == 0 else ""))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Grad-CAM Activation" + (" (Failed)" if np.max(heatmap) == 0 else "") + (" - Bleeding Detected!" if bleeding_detected else ""))
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gradcam_mask_comparison.png"))
print(f"Saved: {os.path.join(output_dir, 'gradcam_mask_comparison.png')}")