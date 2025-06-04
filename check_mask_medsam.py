import os
import numpy as np
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import sys
import segmentation_models_pytorch as sm

# === Setup ===
sys.path.append('/Users/nadiajelani/projects/wound-segmentation/models')  # adjust if needed

from segment_anything import build_sam_vit_b
from wound_medsam import load_medsam_model, medsam_model

MODEL_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/best_medsam_wound_model.pth"
load_medsam_model(MODEL_PATH)

model = medsam_model

# === CONFIG ===
IMG_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_images/"
MASK_DIR = "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_masks/"
MODEL_PATH = "/Users/nadiajelani/projects/wound-segmentation/models/best_medsam_wound_model.pth"
IMG_SIZE = (1024, 1024)
OUT_SIZE = (224, 224)
THRESHOLD = 0.5

# === Dice score ===
def dice_score(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# === Validation Loop ===
results = []

image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg'))])

for img_file in tqdm(image_files):
    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, img_file)

    if not os.path.exists(mask_path):
        continue

    try:
        # Load image
        image = cv2.imread(img_path)
        image_resized = cv2.resize(image, IMG_SIZE)
        img_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB) / 255.0
        img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = normalize(img_tensor)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = torch.sigmoid(pred).squeeze().numpy()

        pred_mask_bin = (pred_mask > THRESHOLD).astype(np.uint8)
        pred_mask_resized = cv2.resize(pred_mask_bin, OUT_SIZE, interpolation=cv2.INTER_NEAREST)

        # Load true mask
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask_resized = cv2.resize(true_mask, OUT_SIZE)
        true_mask_bin = (true_mask_resized > 127).astype(np.uint8)

        if np.sum(true_mask_bin) == 0 and np.sum(pred_mask_resized) == 0:
            dice = 1.0
            iou = 1.0
        else:
            dice = dice_score(true_mask_bin, pred_mask_resized)
            iou = jaccard_score(true_mask_bin.flatten(), pred_mask_resized.flatten(), average='binary', zero_division=0)

        results.append({
            "Image": img_file,
            "Dice Score": round(dice, 4),
            "IoU Score": round(iou, 4)
        })

    except Exception as e:
        results.append({
            "Image": img_file,
            "Dice Score": "Error",
            "IoU Score": "Error"
        })
        print(f"⚠️ Error with {img_file}: {e}")

# === Save Results ===
df = pd.DataFrame(results)
df.to_csv("medsam_validation_results.csv", index=False)
print("✅ MedSAM validation complete. Saved as 'medsam_validation_results.csv'")
print(df.describe())
