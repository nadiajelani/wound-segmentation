# 🔧 Fix: Only Mask Showing, No Heatmap/Doctor Report

## What I've Added:

### 1. Enhanced Logging in `app.py`
- Detailed logging in `/analyze` endpoint
- Tracks heatmap/overlay generation
- Shows base64 encoding lengths
- Logs all returned data keys

### 2. Debug UI (`debug_analyzer.html`)
- Visual tool to see exactly what API returns
- Shows which images are present/missing
- Displays image data lengths
- Shows doctor report JSON

## 🚀 Deploy & Test Now

### Step 1: Commit and Push to Railway

```bash
# Add the updated files
git add app.py debug_analyzer.html

# Commit
git commit -m "Add enhanced logging and debug UI for heatmap issue"

# Push to trigger Railway deployment
git push origin main
```

### Step 2: Wait for Railway to Deploy

Go to https://railway.app → Your Project → Deployments

Wait for the build to complete (usually 2-3 minutes).

### Step 3: Access Debug UI

Once deployed, go to:

```
https://YOUR-RAILWAY-URL.up.railway.app/debug-ui
```

This special debug page will show you EXACTLY what the API is returning.

### Step 4: Test with Debug UI

1. **Open the debug UI**: `https://YOUR-URL/debug-ui`
2. **Upload any image** (wound or test image)
3. **Click "Analyze & Debug"**
4. **Check the results**

The debug UI will show:

```
✅ mask_image: PRESENT (5000 chars)
✅ heatmap_image: PRESENT (35000 chars)  ← Should be present!
✅ overlay_image: PRESENT (65000 chars)  ← Should be present!
✅ doctor_report: PRESENT                 ← Should be present!
```

Or:

```
✅ mask_image: PRESENT (5000 chars)
❌ heatmap_image: MISSING                 ← Problem!
❌ overlay_image: MISSING                 ← Problem!
```

### Step 5: Check Railway Logs

While testing, watch the Railway logs in real-time:

```bash
railway logs --follow
```

Or in Railway dashboard: Deployments → View Logs

**Look for these NEW log messages:**

```
Building heatmap from pred_map shape: (128, 128), dtype: float32
Heatmap created: shape (128, 128, 3), dtype: uint8
Building overlay from image shape: (128, 128, 3)
Overlay created: shape (128, 128, 3), dtype: uint8
Encoding images to base64...
Image encoding complete:
  - mask_b64 length: 5000
  - heatmap_b64 length: 35000   ← Should be ~35000
  - overlay_b64 length: 65000   ← Should be ~65000
Returning JSON with 7 keys: ['success', 'metrics', 'healing_stage', 'doctor_report', 'mask_image', 'heatmap_image', 'overlay_image']
```

## 📊 Diagnose the Issue

### Scenario A: Debug UI shows ALL images present ✅

**Meaning:** Backend is working correctly, issue is in the frontend HTML.

**Solution:** Check browser console for JavaScript errors

1. Open main wound analyzer: `https://YOUR-URL/`
2. Press F12 (or Cmd+Option+I on Mac)
3. Go to "Console" tab
4. Upload and analyze an image
5. Look for red error messages

Common issues:
- JavaScript error in displaying images
- Wrong element IDs in HTML
- CORS issues (unlikely since debug UI works)

### Scenario B: Debug UI shows heatmap/overlay MISSING ❌

**Meaning:** Backend is not generating the heatmap/overlay

**Solution:** Check Railway logs for errors

Look for:
- Errors in heatmap generation
- OpenCV errors
- Memory issues
- Type errors

Possible causes:
- `make_heatmap()` function failing
- `make_overlay()` function failing  
- `to_base64_png()` encoding failing

### Scenario C: Debug UI shows TINY images (<1000 chars)

**Meaning:** Images are being generated but are blank/empty

**Solution:** pred_map is all zeros (no wound detected)

Check logs for:
```
Prediction min/max: 0.0000/0.0000  ← All zeros!
```

This means model is not predicting anything.

## 🔍 Common Fixes

### Fix 1: If logs show heatmap creation errors

Check if OpenCV is installed:
```bash
railway run python -c "import cv2; print(cv2.__version__)"
```

### Fix 2: If images are being generated but not displayed

Check the frontend JavaScript in `wound_analyzer.html`:

```javascript
// Line 733-735 should be:
document.getElementById('maskImage').src = result.mask_image;
document.getElementById('heatmapImage').src = result.heatmap_image;
document.getElementById('overlayImage').src = result.overlay_image;
```

### Fix 3: If doctor report is missing

Check logs for "doctor_report" generation.

Should see in logs:
```
Returning JSON with 7 keys: ['success', 'metrics', 'healing_stage', 'doctor_report', ...]
```

If "doctor_report" is not in that list, the generation function is failing.

## 📞 Report Back

After testing with the debug UI, tell me:

1. **What does the debug UI show?**
   - Which images are present/missing?
   - What are the image data lengths?

2. **What do the Railway logs say?**
   - Copy the relevant log lines from an analysis
   - Especially the "Image encoding complete" section

3. **Any errors?**
   - JavaScript errors in browser console?
   - Python errors in Railway logs?

Then I can provide the exact fix!

## Quick Reference

```bash
# Deploy
git add app.py debug_analyzer.html
git commit -m "Add debug logging and UI"
git push origin main

# Watch logs
railway logs --follow

# Test
# 1. Open: https://YOUR-URL/debug-ui
# 2. Upload image
# 3. Click "Analyze & Debug"
# 4. Check what's missing
# 5. Check Railway logs
```

## Expected Full Working Output

**Debug UI should show:**
```
✅ Mask | ✅ Heatmap | ✅ Overlay | ✅ Report
```

**Railway logs should show:**
```
Building heatmap from pred_map shape: (128, 128), dtype: float32
Heatmap created: shape (128, 128, 3), dtype: uint8
Building overlay from image shape: (128, 128, 3)
Overlay created: shape (128, 128, 3), dtype: uint8
Image encoding complete:
  - mask_b64 length: 5000+
  - heatmap_b64 length: 35000+
  - overlay_b64 length: 65000+
Returning JSON with 7 keys: ['success', 'metrics', 'healing_stage', 'doctor_report', 'mask_image', 'heatmap_image', 'overlay_image']
```

**Main website should show:**
- 4 images (original, mask, heatmap, overlay)
- Metrics section with area, perimeter, severity
- Healing stage assessment with recommendations
- Clinical report with findings and follow-up

---

**Next step:** Deploy and test with debug UI, then report what you see! 🔍
