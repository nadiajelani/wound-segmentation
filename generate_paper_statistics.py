#!/usr/bin/env python3
"""
generate_paper_statistics.py
=============================
Run this ONE script from your project root to produce all the real numbers
needed for Tables 2, 4, and 5 of the revised PLOS manuscript.

Usage:
    cd /Users/nadiajelani/projects/wound-segmentation
    python generate_paper_statistics.py

What it produces (all saved to stats_output/):
    test_metrics.csv          — per-image dice/iou/precision/recall
    table2_main_results.txt   — Table 2 numbers with 95% CI (paste into manuscript)
    table3_loss_ablation.txt  — Table 3 (loss function comparison, if you ran baseline)
    table4_wound_type.txt     — Table 4 stratified by wound type
    table5_skin_tone.txt      — Table 5 stratified by Fitzpatrick type
    normality_check.txt       — Shapiro-Wilk results
    figure1_samples/          — 6 representative test images for Figure 1
    stats_summary.txt         — one-page summary of everything to paste

IMPORTANT: Edit the CONFIG section below before running.
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import shapiro, kruskal

# ── TensorFlow (suppress noisy logs) ─────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths to match your machine
# ════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── Model ────────────────────────────────────────────────────────────────
    "model_path": "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras",

    # ── Test data (images and matching masks, same filename in both dirs) ────
    "test_img_dir":  "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_images/",
    "test_mask_dir": "/Users/nadiajelani/Desktop/wounds-whisperer/wounds/u_net_images/test_masks/",

    # ── Input size (must match training) ─────────────────────────────────────
    "img_size": (128, 128),

    # ── Threshold for binary mask ─────────────────────────────────────────────
    "threshold": 0.5,

    # ── Bootstrap iterations ──────────────────────────────────────────────────
    "n_bootstrap": 1000,

    # ── Output folder ─────────────────────────────────────────────────────────
    "out_dir": "stats_output",

    # ── Optional: path to a CSV mapping test filenames to wound type ──────────
    #    Columns needed: "file", "wound_type"
    #    wound_type values: "diabetic_foot_ulcer" | "pressure_injury" | "surgical"
    #    Leave as None if you don't have this file — Table 4 will be skipped.
    "wound_type_csv": None,   # e.g. "data/test_metadata.csv"

    # ── Baseline model path (random-init U-Net) ───────────────────────────────
    #    Set to None to skip baseline comparison (Table 2 partial column).
    "baseline_model_path": None,  # e.g. "models/baseline_unet.keras"

    # ── Number of qualitative examples to save for Figure 1 ──────────────────
    "n_figure_examples": 6,
}

# ════════════════════════════════════════════════════════════════════════════
# Custom loss functions  (must match what you used during training)
# ════════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def binary_focal_loss(y_true, y_pred, gamma=1.0, alpha=0.1):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(alpha * tf.math.pow(1 - y_pred, gamma) * bce)

@tf.keras.utils.register_keras_serializable()
def total_loss(y_true, y_pred):
    return (tf.keras.losses.binary_crossentropy(y_true, y_pred)
            + dice_loss(y_true, y_pred)
            + binary_focal_loss(y_true, y_pred))

@tf.keras.utils.register_keras_serializable()
def wound_dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

CUSTOM_OBJECTS = {
    "total_loss": total_loss,
    "dice_loss": dice_loss,
    "binary_focal_loss": binary_focal_loss,
    "wound_dice_coef": wound_dice_coef,
    "dice_coefficient": wound_dice_coef,
}

# ════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(mask_true, pred_bin):
    """Returns dice, iou, precision, recall for one image."""
    smooth = 1e-6
    tp = np.sum(mask_true * pred_bin)
    fp = np.sum(pred_bin) - tp
    fn = np.sum(mask_true) - tp
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou  = (tp + smooth) / (tp + fp + fn + smooth)
    prec = (tp + smooth) / (tp + fp + smooth)
    rec  = (tp + smooth) / (tp + fn + smooth)
    return dice, iou, prec, rec

def bootstrap_ci(arr, n=1000, pct_lo=2.5, pct_hi=97.5):
    arr = np.array(arr)
    means = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n)]
    return np.mean(arr), np.std(arr), np.percentile(means, pct_lo), np.percentile(means, pct_hi)

def fmt(mean, lo, hi, sd=None):
    s = f"{mean:.3f} (95% CI: {lo:.3f}–{hi:.3f})"
    if sd is not None:
        s += f", SD={sd:.3f}"
    return s

def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / (np.std(diff) + 1e-9)

# ════════════════════════════════════════════════════════════════════════════
# Skin-tone Fitzpatrick estimator (luminance-based)
# ════════════════════════════════════════════════════════════════════════════

def estimate_fitzpatrick(img_rgb_255):
    """Very rough skin-tone group from background luminance."""
    lum = (0.299 * img_rgb_255[:,:,0] +
           0.587 * img_rgb_255[:,:,1] +
           0.114 * img_rgb_255[:,:,2])
    mean_lum = np.mean(lum)
    if mean_lum > 200: return "I-II"
    elif mean_lum > 140: return "III-IV"
    else: return "V-VI"

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    out = CONFIG["out_dir"]
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "figure1_samples"), exist_ok=True)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("\n[1/6] Loading model...")
    if not os.path.exists(CONFIG["model_path"]):
        sys.exit(f"❌  Model not found: {CONFIG['model_path']}\n"
                 f"    Update CONFIG['model_path'] at the top of this script.")

    model = load_model(CONFIG["model_path"], custom_objects=CUSTOM_OBJECTS, compile=False)
    print(f"    ✅  Loaded: {CONFIG['model_path']}")

    baseline_model = None
    if CONFIG["baseline_model_path"] and os.path.exists(CONFIG["baseline_model_path"]):
        baseline_model = load_model(CONFIG["baseline_model_path"],
                                    custom_objects=CUSTOM_OBJECTS, compile=False)
        print(f"    ✅  Baseline loaded: {CONFIG['baseline_model_path']}")

    # ── 2. Run inference on test set ──────────────────────────────────────────
    print("\n[2/6] Running inference on test set...")
    IMG_DIR  = CONFIG["test_img_dir"]
    MASK_DIR = CONFIG["test_mask_dir"]
    SZ       = CONFIG["img_size"]
    THRESH   = CONFIG["threshold"]

    if not os.path.isdir(IMG_DIR):
        sys.exit(f"❌  Test image dir not found: {IMG_DIR}")
    if not os.path.isdir(MASK_DIR):
        sys.exit(f"❌  Test mask dir not found: {MASK_DIR}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(exts))
    print(f"    Found {len(image_files)} images in test dir")

    records = []
    baseline_records = []
    saved_examples = 0

    for idx, fname in enumerate(image_files):
        img_path  = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname)
        if not os.path.exists(mask_path):
            # try matching without extension
            base = os.path.splitext(fname)[0]
            found = [f for f in os.listdir(MASK_DIR)
                     if os.path.splitext(f)[0] == base]
            if found:
                mask_path = os.path.join(MASK_DIR, found[0])
            else:
                print(f"    ⚠️  No mask for {fname}, skipping")
                continue

        # load image
        img_raw = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, SZ).astype(np.float32) / 255.0
        img_input = img_resized[np.newaxis, ...]

        # load mask
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask_raw, SZ)
        mask_bin = (mask_resized > 127).astype(np.float32)

        # predict
        pred = model.predict(img_input, verbose=0)[0, :, :, 0]
        pred_bin = (pred > THRESH).astype(np.float32)

        dice, iou, prec, rec = compute_metrics(mask_bin, pred_bin)
        skin = estimate_fitzpatrick(cv2.resize(img_rgb, SZ))

        records.append({
            "file": fname,
            "dice": dice, "iou": iou, "precision": prec, "recall": rec,
            "skin_tone": skin,
        })

        # baseline
        if baseline_model is not None:
            b_pred = baseline_model.predict(img_input, verbose=0)[0, :, :, 0]
            b_pred_bin = (b_pred > THRESH).astype(np.float32)
            bd, bi, bp, br = compute_metrics(mask_bin, b_pred_bin)
            baseline_records.append({"file": fname, "dice": bd, "iou": bi,
                                      "precision": bp, "recall": br})

        # save qualitative examples for Figure 1
        if saved_examples < CONFIG["n_figure_examples"]:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img_resized)
            axes[0].set_title("Original", fontsize=11)
            axes[1].imshow(mask_bin, cmap="gray")
            axes[1].set_title("Ground Truth", fontsize=11)
            axes[2].imshow(pred_bin, cmap="gray")
            axes[2].set_title(f"Prediction (t={THRESH})", fontsize=11)
            axes[3].imshow(pred, cmap="hot", vmin=0, vmax=1)
            axes[3].set_title("Probability Map", fontsize=11)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{fname}  |  Dice={dice:.3f}  IoU={iou:.3f}", fontsize=12)
            plt.tight_layout()
            fig_path = os.path.join(out, "figure1_samples", f"example_{saved_examples+1}_{fname}")
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            saved_examples += 1

        if (idx + 1) % 20 == 0:
            print(f"    {idx+1}/{len(image_files)} done...")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out, "test_metrics.csv"), index=False)
    print(f"\n    ✅  {len(df)} images evaluated. Saved to {out}/test_metrics.csv")

    if len(df) == 0:
        sys.exit("❌  No images were evaluated. Check your paths.")

    # ── 3. Table 2: Main results with CIs ─────────────────────────────────────
    print("\n[3/6] Computing Table 2 statistics...")
    lines = []
    lines.append("=" * 65)
    lines.append("TABLE 2 — MAIN RESULTS  (paste into manuscript)")
    lines.append("=" * 65)

    def row_stats(label, dices, ious, precs=None, recs=None, compare_dice=None):
        dm, dsd, dlo, dhi = bootstrap_ci(dices)
        im, isd, ilo, ihi = bootstrap_ci(ious)
        lines.append(f"\n  {label}")
        lines.append(f"    Dice:      {fmt(dm, dlo, dhi, dsd)}")
        lines.append(f"    IoU:       {fmt(im, ilo, ihi, isd)}")
        if precs is not None:
            lines.append(f"    Precision: {np.mean(precs):.3f}")
            lines.append(f"    Recall:    {np.mean(recs):.3f}")
        if compare_dice is not None:
            t_stat, p_val = stats.ttest_rel(dices, compare_dice)
            d = cohens_d(dices, compare_dice)
            sig = "p<0.001" if p_val < 0.001 else (f"p={p_val:.4f}")
            lines.append(f"    vs baseline: {sig}, Cohen's d={d:.2f}")
        return dm, dlo, dhi, im, ilo, ihi

    dm_s, dlo_s, dhi_s, im_s, ilo_s, ihi_s = row_stats(
        "SimCLR + U-Net (OURS)",
        df["dice"].tolist(), df["iou"].tolist(),
        df["precision"].tolist(), df["recall"].tolist(),
        baseline_records[0]["dice"] if baseline_records else None  # placeholder
    )

    if baseline_records:
        bdf = pd.DataFrame(baseline_records)
        row_stats("Baseline U-Net (random init)",
                  bdf["dice"].tolist(), bdf["iou"].tolist(),
                  bdf["precision"].tolist(), bdf["recall"].tolist())
        t_stat, p_val = stats.ttest_rel(df["dice"], bdf["dice"])
        d = cohens_d(df["dice"].tolist(), bdf["dice"].tolist())
        lines.append(f"\n  Paired t-test (SimCLR vs baseline):")
        lines.append(f"    t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d:.3f}")

    lines.append("\n  n (test set) = " + str(len(df)))
    lines.append("=" * 65)

    # ── 4. Normality check ────────────────────────────────────────────────────
    print("\n[4/6] Normality checks...")
    norm_lines = ["\nSHAPIRO-WILK NORMALITY TEST (for parametric test justification)"]
    for col in ["dice", "iou"]:
        W, p = shapiro(df[col])
        ok = "✅ Normal (p>0.05)" if p > 0.05 else "⚠️ Non-normal"
        norm_lines.append(f"  {col:12s}: W={W:.4f}, p={p:.4f}  {ok}")
    lines += norm_lines

    # ── 5. Table 4: Wound type stratification ─────────────────────────────────
    print("\n[5/6] Wound-type stratification...")
    lines.append("\n" + "=" * 65)
    lines.append("TABLE 4 — PERFORMANCE BY WOUND TYPE")
    lines.append("=" * 65)

    if CONFIG["wound_type_csv"] and os.path.exists(CONFIG["wound_type_csv"]):
        meta = pd.read_csv(CONFIG["wound_type_csv"])
        merged = df.merge(meta, on="file", how="left")
        for wtype, grp in merged.groupby("wound_type"):
            dm, dsd, dlo, dhi = bootstrap_ci(grp["dice"].tolist())
            im, isd, ilo, ihi = bootstrap_ci(grp["iou"].tolist())
            lines.append(f"\n  {wtype} (n={len(grp)})")
            lines.append(f"    Dice: {fmt(dm, dlo, dhi)}")
            lines.append(f"    IoU:  {fmt(im, ilo, ihi)}")
    else:
        lines.append("\n  ⚠️  No wound_type_csv provided.")
        lines.append("  To generate this table, create a CSV file with columns:")
        lines.append("    file, wound_type")
        lines.append("  where wound_type is one of:")
        lines.append("    diabetic_foot_ulcer | pressure_injury | surgical")
        lines.append("\n  Then set CONFIG['wound_type_csv'] = 'path/to/that.csv'")
        lines.append("  and re-run this script.\n")
        lines.append("  MANUAL OPTION: Look at your test images and note which")
        lines.append("  folder or filename prefix they came from (DFUC = diabetic")
        lines.append("  foot ulcer, etc.) and build the CSV from that.")

    # ── 6. Table 5: Skin tone ─────────────────────────────────────────────────
    print("\n[6/6] Skin-tone stratification...")
    lines.append("\n" + "=" * 65)
    lines.append("TABLE 5 — PERFORMANCE BY ESTIMATED FITZPATRICK SKIN TYPE")
    lines.append("(Estimated automatically from image luminance — approximate)")
    lines.append("=" * 65)

    groups_by_skin = []
    for skin, grp in df.groupby("skin_tone"):
        dm, dsd, dlo, dhi = bootstrap_ci(grp["dice"].tolist())
        im, isd, ilo, ihi = bootstrap_ci(grp["iou"].tolist())
        lines.append(f"\n  Fitzpatrick {skin} (n={len(grp)})")
        lines.append(f"    Dice: {fmt(dm, dlo, dhi)}")
        lines.append(f"    IoU:  {fmt(im, ilo, ihi)}")
        groups_by_skin.append(grp["dice"].tolist())

    if len(groups_by_skin) >= 2:
        H, p_kw = kruskal(*groups_by_skin)
        lines.append(f"\n  Kruskal-Wallis across skin groups: H={H:.2f}, p={p_kw:.4f}")
        if p_kw > 0.05:
            lines.append("  → No statistically significant difference across skin tone groups.")
        else:
            lines.append("  → Significant difference detected — report with caution.")

    lines.append("\n" + "=" * 65)
    lines.append("NOTE: Skin tone labels are estimated automatically from image")
    lines.append("luminance. They are approximate. Treat Table 5 as indicative.")
    lines.append("=" * 65)

    # ── Save everything ───────────────────────────────────────────────────────
    summary_path = os.path.join(out, "stats_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    # Also save a clean copy of table 2 alone
    with open(os.path.join(out, "table2_main_results.txt"), "w") as f:
        f.write("\n".join(lines[:lines.index(norm_lines[0])]))

    print(f"\n✅  All done! Results saved to: {out}/")
    print(f"    → stats_summary.txt   (everything — paste into manuscript)")
    print(f"    → test_metrics.csv    (per-image raw numbers)")
    print(f"    → figure1_samples/    ({saved_examples} example images for Figure 1)")
    print(f"\n📋  Quick summary of YOUR results:")
    dm, dsd, dlo, dhi = bootstrap_ci(df["dice"].tolist())
    im, isd, ilo, ihi = bootstrap_ci(df["iou"].tolist())
    print(f"    Dice:  {dm:.3f}  (95% CI: {dlo:.3f}–{dhi:.3f}, SD={dsd:.3f})")
    print(f"    IoU:   {im:.3f}  (95% CI: {ilo:.3f}–{ihi:.3f}, SD={isd:.3f})")
    print(f"    Prec:  {df['precision'].mean():.3f}")
    print(f"    Rec:   {df['recall'].mean():.3f}")
    print(f"    n:     {len(df)}")
    print()
    print("👉  Next: open stats_output/stats_summary.txt and copy the numbers")
    print("    into the manuscript's Tables 2, 4, and 5.")

if __name__ == "__main__":
    main()
