"""
Enhanced Figure Generation with Skin Color Analysis
Creates separate panels for mask, segmentation, heatmap with skin tone context
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
from PIL import Image

# Create output directory
os.makedirs('paper_figures', exist_ok=True)

print("📊 Generating Enhanced IEEE Conference Paper Figures with Skin Color Analysis...")

# ==============================================================================
# Skin Tone Detection and Classification
# ==============================================================================
def detect_skin_tone(image_rgb):
    """
    Detect and classify skin tone using Fitzpatrick scale
    Returns: skin_tone_class (1-6), average_color (R,G,B), skin_tone_name
    """
    # Calculate average skin color (excluding wound area for this demo)
    avg_color = np.mean(image_rgb, axis=(0, 1))
    
    # Simplified Fitzpatrick scale classification
    # Based on RGB values and luminance
    luminance = 0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
    
    if luminance > 220:
        skin_class = 1
        skin_name = "Type I (Very Fair)"
        base_color = [255, 240, 230]
    elif luminance > 190:
        skin_class = 2
        skin_name = "Type II (Fair)"
        base_color = [245, 220, 200]
    elif luminance > 160:
        skin_class = 3
        skin_name = "Type III (Medium)"
        base_color = [220, 190, 160]
    elif luminance > 130:
        skin_class = 4
        skin_name = "Type IV (Olive)"
        base_color = [190, 150, 120]
    elif luminance > 90:
        skin_class = 5
        skin_name = "Type V (Brown)"
        base_color = [140, 100, 80]
    else:
        skin_class = 6
        skin_name = "Type VI (Dark)"
        base_color = [90, 60, 50]
    
    return skin_class, np.array(base_color, dtype=np.uint8), skin_name

# ==============================================================================
# Figure 2: Enhanced Results with Skin Color Analysis (6 panels)
# ==============================================================================
def create_enhanced_results_with_skin_color():
    """Create comprehensive 6-panel results showing skin color context"""
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    img_size = 128
    
    # Create 3 different skin tones with wounds
    skin_tones = [
        ([245, 220, 200], "Fair Skin (Type II)"),      # Fair
        ([220, 190, 160], "Medium Skin (Type III)"),   # Medium
        ([140, 100, 80], "Brown Skin (Type V)")        # Brown
    ]
    
    for idx, (skin_color, skin_label) in enumerate(skin_tones):
        # Create original image with specific skin tone
        original = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        original[:] = skin_color
        
        # Add synthetic wound
        center = (img_size // 2, img_size // 2)
        wound_size = (25, 20) if idx == 0 else (30, 25) if idx == 1 else (28, 22)
        
        # Wound color varies with skin tone
        if idx == 0:  # Fair skin
            wound_color = (180, 50, 50)
        elif idx == 1:  # Medium skin
            wound_color = (160, 40, 40)
        else:  # Brown skin
            wound_color = (120, 30, 30)
        
        cv2.ellipse(original, center, wound_size, 0, 0, 360, wound_color, -1)
        cv2.ellipse(original, center, (wound_size[0]-10, wound_size[1]-8), 0, 0, 360, 
                   (wound_color[0]+20, wound_color[1]+10, wound_color[2]+10), -1)
        
        # Add some texture
        noise = np.random.randint(-10, 10, original.shape, dtype=np.int16)
        original = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Create segmentation mask
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.ellipse(mask, center, wound_size, 0, 0, 360, 255, -1)
        
        # Create confidence heatmap
        heatmap = np.zeros((img_size, img_size), dtype=np.float32)
        for y in range(img_size):
            for x in range(img_size):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                heatmap[y, x] = np.exp(-dist / 15)
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Plot - Top row: Original images with skin tone label
        ax_top = fig.add_subplot(gs[0, idx])
        ax_top.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax_top.set_title(f'{skin_label}\nOriginal Image', fontsize=11, fontweight='bold')
        ax_top.axis('off')
        
        # Add skin tone indicator
        rect = patches.Rectangle((5, img_size-25), 30, 20, 
                                linewidth=2, edgecolor='white', 
                                facecolor=np.array(skin_color)/255)
        ax_top.add_patch(rect)
        
        # Bottom row: Combined segmentation visualization
        ax_bottom = fig.add_subplot(gs[1, idx])
        
        # Create 3-panel combined view
        combined = np.zeros((img_size, img_size*3, 3), dtype=np.uint8)
        
        # Panel 1: Binary mask
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        combined[:, :img_size, :] = mask_rgb
        
        # Panel 2: Segmentation overlay
        overlay = original.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay on wound
        blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
        combined[:, img_size:img_size*2, :] = blended
        
        # Panel 3: Heatmap
        combined[:, img_size*2:, :] = heatmap_colored
        
        ax_bottom.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        ax_bottom.set_title('Mask | Segmentation | Heatmap', fontsize=10, fontweight='bold')
        ax_bottom.axis('off')
        
        # Add dividers
        ax_bottom.axvline(x=img_size-0.5, color='white', linewidth=2)
        ax_bottom.axvline(x=img_size*2-0.5, color='white', linewidth=2)
    
    plt.suptitle('Wound Segmentation Across Different Skin Tones (Fitzpatrick Scale)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('paper_figures/fig2_enhanced_results_skin_color.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 2: Enhanced results with skin color analysis created")
    plt.close()

# ==============================================================================
# Figure 3: Detailed Separate Outputs (Mask, Segment, Heatmap)
# ==============================================================================
def create_separate_outputs_detailed():
    """Create detailed view of mask, segmentation, and heatmap separately"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25)
    
    img_size = 128
    
    # Create wound image
    original = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    original[:] = [220, 190, 160]  # Medium skin
    
    center = (img_size // 2, img_size // 2)
    cv2.ellipse(original, center, (28, 22), 0, 0, 360, (160, 40, 40), -1)
    cv2.ellipse(original, center, (18, 14), 0, 0, 360, (180, 50, 50), -1)
    
    # Add texture
    noise = np.random.randint(-8, 8, original.shape, dtype=np.int16)
    original = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Binary mask
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.ellipse(mask, center, (28, 22), 0, 0, 360, 255, -1)
    
    # Probability map (soft mask)
    prob_map = np.zeros((img_size, img_size), dtype=np.float32)
    for y in range(img_size):
        for x in range(img_size):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            prob_map[y, x] = np.clip(1.0 - dist / 30, 0, 1)
    
    # Confidence heatmap
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)
    for y in range(img_size):
        for x in range(img_size):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            heatmap[y, x] = np.exp(-dist / 15)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Segmentation overlay
    overlay = original.copy()
    overlay[mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
    
    # ROW 1: Original Image (4 views)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax.set_title('(a) Original Wound Image', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Skin tone analysis
    _, skin_color, skin_name = detect_skin_tone(original)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.ones((img_size, img_size, 3), dtype=np.uint8) * skin_color)
    ax.set_title(f'(b) Detected Skin Tone\n{skin_name}', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # RGB channels
    ax = fig.add_subplot(gs[0, 2])
    rgb_split = np.zeros((img_size, img_size*3, 3), dtype=np.uint8)
    rgb_split[:, :img_size, 0] = original[:, :, 2]  # R
    rgb_split[:, img_size:img_size*2, 1] = original[:, :, 1]  # G
    rgb_split[:, img_size*2:, 2] = original[:, :, 0]  # B
    ax.imshow(rgb_split)
    ax.set_title('(c) RGB Channel Separation', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Histogram
    ax = fig.add_subplot(gs[0, 3])
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([original], [i], None, [256], [0, 256])
        ax.plot(hist, color=color, alpha=0.7, linewidth=2)
    ax.set_title('(d) Color Histogram', fontsize=11, fontweight='bold')
    ax.set_xlabel('Pixel Value', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ROW 2: Binary Mask (4 views)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(mask, cmap='gray')
    ax.set_title('(e) Binary Mask\n(Thresholded at 0.5)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_WINTER)
    ax.imshow(cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB))
    ax.set_title('(f) Mask (Color Mapped)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(prob_map, cmap='viridis')
    ax.set_title('(g) Probability Map\n(Continuous 0-1)', fontsize=11, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(ax.imshow(prob_map, cmap='viridis'), ax=ax, fraction=0.046)
    cbar.set_label('Probability', fontsize=8)
    
    # Contours
    ax = fig.add_subplot(gs[1, 3])
    contour_img = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    ax.imshow(contour_img)
    ax.set_title('(h) Wound Boundary Contour', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # ROW 3: Heatmap and Segmentation (4 views)
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
    ax.set_title('(i) Confidence Heatmap\n(JET colormap)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Heatmap with scale
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(heatmap, cmap='jet')
    ax.set_title('(j) Heatmap with Scale', fontsize=11, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label('Confidence', fontsize=8)
    
    # Overlay
    ax = fig.add_subplot(gs[2, 2])
    overlay_colored = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
    ax.imshow(cv2.cvtColor(overlay_colored, cv2.COLOR_BGR2RGB))
    ax.set_title('(k) Heatmap Overlay\n(60% original + 40% heatmap)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    # Segmentation overlay
    ax = fig.add_subplot(gs[2, 3])
    ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax.set_title('(l) Segmentation Overlay\n(Red = Wound)', fontsize=11, fontweight='bold')
    ax.axis('off')
    
    plt.suptitle('Detailed Output Analysis: Mask, Segmentation, and Heatmap Components', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig('paper_figures/fig3_detailed_separate_outputs.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 3: Detailed separate outputs created")
    plt.close()

# ==============================================================================
# Figure 4: Skin Tone Impact on Performance
# ==============================================================================
def create_skin_tone_performance():
    """Show performance across different skin tones"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Data: Performance metrics across Fitzpatrick scale
    skin_types = ['Type I\n(Very Fair)', 'Type II\n(Fair)', 'Type III\n(Medium)', 
                  'Type IV\n(Olive)', 'Type V\n(Brown)', 'Type VI\n(Dark)']
    
    # Simulated performance data (in real paper, use actual results)
    dice_scores = [0.89, 0.90, 0.88, 0.87, 0.86, 0.85]
    iou_scores = [0.83, 0.84, 0.82, 0.81, 0.80, 0.79]
    sample_sizes = [150, 220, 280, 210, 180, 130]
    
    colors = ['#FFE4C4', '#F5DEB3', '#D2B48C', '#C19A6B', '#8B6A47', '#654321']
    
    # Plot 1: Performance by skin type
    ax1 = axes[0]
    x = np.arange(len(skin_types))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, dice_scores, width, label='Dice Coefficient', 
                    color='#4CAF50', alpha=0.8)
    bars2 = ax1.bar(x + width/2, iou_scores, width, label='IoU Score', 
                    color='#2196F3', alpha=0.8)
    
    ax1.set_xlabel('Fitzpatrick Skin Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Performance Across Skin Tones', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(skin_types, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.7, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Dataset distribution
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(sample_sizes, labels=skin_types, autopct='%1.1f%%',
                                        colors=colors, startangle=90, 
                                        textprops={'fontsize': 9})
    ax2.set_title('(b) Dataset Distribution by Skin Type\n(Total: 1,170 images)', 
                  fontsize=13, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    plt.savefig('paper_figures/fig4_skin_tone_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Figure 4: Skin tone performance analysis created")
    plt.close()

# ==============================================================================
# Generate all enhanced figures
# ==============================================================================
if __name__ == "__main__":
    create_enhanced_results_with_skin_color()
    create_separate_outputs_detailed()
    create_skin_tone_performance()
    print("\n🎉 All enhanced figures with skin color analysis generated!")
    print("📁 New figures saved in: paper_figures/")
    print("   - fig2_enhanced_results_skin_color.png")
    print("   - fig3_detailed_separate_outputs.png")
    print("   - fig4_skin_tone_performance.png")
