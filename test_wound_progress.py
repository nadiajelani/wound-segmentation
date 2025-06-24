import os
import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime

# -------- Settings --------
mask_path = "/Users/nadiajelani/projects/wound-segmentation/wound_progress_report/simclr_mask.png"
scale_mm_per_pixel = 0.1  # Scale factor (mm per pixel)
scale_bar_length_mm = 10  # Length of scale bar in mm
report_dir = "/Users/nadiajelani/projects/wound-segmentation/wound_progress_report/"
os.makedirs(report_dir, exist_ok=True)
historical_csv = os.path.join(report_dir, "historical_report.csv")

# -------- Load Historical Data --------
historical_data = pd.read_csv(historical_csv) if os.path.exists(historical_csv) else pd.DataFrame(columns=[
    "Date", "Image", "Wound Area (px²)", "Perimeter (px)", "Bounding Box", "Centroid", "Shape Irregularity", 
    "Condition", "Wound Area (mm²)", "Perimeter (mm)", "Instructions", "Disclaimer"])

# -------- Load and Process Mask --------
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    print(f"Error: Mask file {mask_path} not found or unreadable.")
    exit()
_, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    print("No wound detected in the mask.")
    exit()

cnt = max(contours, key=cv2.contourArea)  # Largest contour as the wound
area = cv2.contourArea(cnt)
perimeter = cv2.arcLength(cnt, True)
x, y, w, h = cv2.boundingRect(cnt)
M = cv2.moments(cnt)
cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0  # Centroid x
cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0  # Centroid y
irregularity = (perimeter ** 2) / (4 * math.pi * area + 1e-6)  # Shape irregularity (circularity inverse)

# -------- Convert to Millimeters --------
area_mm = area * (scale_mm_per_pixel ** 2)
perimeter_mm = perimeter * scale_mm_per_pixel

# -------- Classify Condition --------
if area > 10000:
    condition = "⚠️ Large wound — likely chronic or ulcerative"
elif area < 3000:
    condition = "✅ Small wound — early or healing"
else:
    condition = "➖ Moderate wound — monitor size"

if irregularity > 2.0:
    condition += " | Edge irregularity suggests inflammation or infection"
elif irregularity < 1.5:
    condition += " | Regular edges — likely healing"

# -------- Track Progress --------
current_date = datetime.now().strftime("%Y-%m-%d")
progress_note = ""
if not historical_data.empty:
    last_area = historical_data["Wound Area (px²)"].iloc[-1]
    area_change = ((area - last_area) / last_area * 100) if last_area else 0
    if area_change > 10:
        progress_note = f" | Warning: Area increased by {area_change:.1f}% since {historical_data['Date'].iloc[-1]}"
    elif area_change < -10:
        progress_note = f" | Positive: Area decreased by {abs(area_change):.1f}% since {historical_data['Date'].iloc[-1]}"
condition += progress_note

# -------- Generate Heatmap --------
original_path = mask_path.replace("mask_output.png", "original_image.jpg")
original = cv2.imread(original_path)
if original is None:
    print(f"Warning: Original image {original_path} not found, using mask for heatmap.")
    gray = cv2.convertScaleAbs(mask, alpha=1.5)  # Enhance contrast for mask
else:
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5)  # Enhance contrast
heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # Apply colormap
heatmap = cv2.bitwise_and(heatmap, heatmap, mask=thresh)  # Apply mask to heatmap
cv2.imwrite(os.path.join(report_dir, "wound_heatmap.png"), heatmap)
print(f"Saved heatmap to {os.path.join(report_dir, 'wound_heatmap.png')}")

# -------- Overlay Image with Scale and Heatmap --------
overlay = cv2.addWeighted(original, 0.4, heatmap, 0.6, 0.0) if original is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 2)  # Green contour
cv2.circle(overlay, (cx, cy), 3, (255, 0, 0), -1)  # Blue centroid
scale_bar_px = int(scale_bar_length_mm / scale_mm_per_pixel)
bar_start = (10, overlay.shape[0] - 20)
bar_end = (10 + scale_bar_px, overlay.shape[0] - 20)
cv2.line(overlay, bar_start, bar_end, (255, 255, 255), 3)  # White scale bar
cv2.putText(overlay, f"{scale_bar_length_mm} mm", (bar_start[0], bar_start[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
output_image_path = os.path.join(report_dir, "wound_combined_view.png")
cv2.imwrite(output_image_path, overlay)
print(f"Saved combined view to {output_image_path}")

# -------- Feature Analysis and Guided Instructions --------
mask_bool = thresh.astype(bool)  # Define mask_bool for intensity analysis
heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
max_intensity = np.max(heatmap_gray[mask_bool]) if mask_bool.any() else 0
healing_stage = "Healing" if area < 3000 and irregularity < 1.5 else "Monitor" if area < 10000 else "At Risk"
infection_risk = "High" if irregularity > 2.0 and max_intensity > 200 else "Moderate" if irregularity > 2.0 else "Low"
instructions = (
    f"Stage: {healing_stage} | Infection Risk: {infection_risk}\n"
    f"Action: {'Consult a doctor immediately if area increases >10% or red hotspots persist.' if infection_risk == 'High' else "
    f"Monitor weekly and consult if worsening.' if infection_risk == 'Moderate' else 'Continue monitoring weekly.'}"
)

# -------- Save Report --------
report = {
    "Date": current_date,
    "Image": os.path.basename(mask_path),
    "Wound Area (px²)": round(area, 2),
    "Wound Area (mm²)": round(area_mm, 2),
    "Perimeter (px)": round(perimeter, 2),
    "Perimeter (mm)": round(perimeter_mm, 2),
    "Bounding Box": f"x={x}, y={y}, w={w}, h={h}",
    "Centroid": f"({cx}, {cy})",
    "Shape Irregularity": round(irregularity, 3),
    "Condition": condition,
    "Instructions": instructions,
    "Disclaimer": "For monitoring only, consult a healthcare provider for diagnosis."
}

df = pd.DataFrame([report])
historical_data = pd.concat([historical_data, df], ignore_index=True)
historical_data.to_csv(historical_csv, index=False)
csv_output_path = os.path.join(report_dir, "report.csv")
df.to_csv(csv_output_path, index=False)
print(f"Saved wound feature report to {csv_output_path}")
print(instructions)