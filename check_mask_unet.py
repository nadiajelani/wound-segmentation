import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import jaccard_score

# CONFIGURATION
IMG_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_images/"
MASK_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_masks/"
MODEL_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/finetuned_segmentation_final.keras"
IMG_SIZE = (224, 224)
THRESHOLD = 0.2

def dice_score(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

model = load_model(MODEL_PATH, custom_objects={"dice_coefficient": dice_score})

results = []
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg'))])

for img_file in image_files:
    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, img_file)

    if not os.path.exists(mask_path):
        continue

    try:
        # Load and preprocess image
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        # Load and preprocess true mask
        mask_true = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_true = cv2.resize(mask_true, IMG_SIZE)
        mask_true_bin = (mask_true > 127).astype(np.uint8)

        # Predict mask
        pred_mask = model.predict(img_input, verbose=0)[0, :, :, 0]
        pred_mask_bin = (pred_mask > THRESHOLD).astype(np.uint8)

        # Skip if both masks are empty (IoU can't be computed)
        if np.sum(mask_true_bin) == 0 and np.sum(pred_mask_bin) == 0:
            dice = 1.0
            iou = 1.0
        else:
            dice = dice_score(mask_true_bin, pred_mask_bin)
            iou = jaccard_score(mask_true_bin.flatten(), pred_mask_bin.flatten(), average='binary', zero_division=0)

        results.append({
            "Image": img_file,
            "Dice Score": round(dice, 4),
            "IoU Score": round(iou, 4)
        })

    except Exception as e:
        print(f"Skipping {img_file} due to error: {e}")
        results.append({
            "Image": img_file,
            "Dice Score": "Error",
            "IoU Score": "Error"
        })

# Save results
df = pd.DataFrame(results)
df.to_csv("unet_simclr_validation_results.csv", index=False)
print("✅ Validation complete. Results saved to 'unet_simclr_validation_results.csv'")
print(df.describe())
