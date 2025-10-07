# 🚀 Test Your Railway Website NOW

## Problem: Only seeing mask, missing heatmap and doctor report

Let's diagnose and fix this!

## Step 1: Get Your Railway URL

Go to https://railway.app and find your deployment URL.

It looks like: `https://wound-segmentation-production.up.railway.app`

## Step 2: Run This Test

```bash
python3 test_api_response.py https://YOUR-RAILWAY-URL.up.railway.app
```

Replace `YOUR-RAILWAY-URL` with your actual URL.

### What This Test Does:

✅ Sends a test image to your API  
✅ Checks what the API returns  
✅ Tells you exactly what's missing  

### Expected Results:

If everything works, you'll see:

```
✅ mask_image: PRESENT (5000+ chars)
✅ heatmap_image: PRESENT (35000+ chars)  
✅ overlay_image: PRESENT (65000+ chars)
✅ Healing Stage: [stage name]
✅ Doctor Report ID: WA-20251007-...
✅ Metrics: area, perimeter, severity

✅ ALL REQUIRED DATA PRESENT!
```

### If You See Problems:

#### Problem A: ❌ heatmap_image: MISSING

**Cause:** Backend not generating heatmap  
**Fix:** Model might not be loaded or there's an error in prediction

Check model status:
```bash
curl https://YOUR-URL.up.railway.app/debug
```

Should show: `"model_loaded": true`

#### Problem B: ❌ doctor_report: MISSING

**Cause:** Report generation failed  
**Fix:** Check Railway logs for errors

```bash
railway logs
```

#### Problem C: Images are all 0 bytes or very small

**Cause:** Prediction is failing, returning empty masks  
**Fix:** Model not loaded properly

## Step 3: Test in Browser

After confirming the API works, test in browser:

1. Open: `https://YOUR-URL.up.railway.app`
2. Upload any image (wound or test photo)
3. Click "Analyze Wound"
4. Wait 5-30 seconds

### What You Should See:

**4 Images:**
1. 📷 Original Image
2. 🎯 Segmentation Mask (black & white)
3. 🌡️ Confidence Heatmap (blue→red gradient)
4. 🎨 Heatmap Overlay (colored overlay on original)

**Metrics:**
- Area (pixels & %)
- Perimeter
- Severity badge

**Healing Stage:**
- Stage name
- AI confidence %
- Description
- Recommendations list

**Clinical Report:**
- Report ID
- Generated timestamp
- Clinical findings (area, perimeter, shape, severity)
- Follow-up care recommendations

### Debugging Browser Issues:

Press F12 (or Cmd+Option+I on Mac) to open console.

Look for errors:
- Red text = errors
- Check "Network" tab to see API calls
- Click on the `/analyze` request to see response

## Common Fixes

### Fix 1: If backend is returning data but browser not showing it

**Check wound_analyzer.html is being served:**

```bash
curl https://YOUR-URL.up.railway.app/ | grep "Wound Analyzer"
```

Should show HTML with "Wound Analyzer" title.

### Fix 2: If heatmap images are missing from API response

**Backend issue - need to check app.py is deployed correctly:**

```bash
curl https://YOUR-URL.up.railway.app/debug
```

Look for `model_input_shape` and `model_output_shape` - should not be null.

### Fix 3: CORS errors

Add to Railway environment variables:

```
CORS_ORIGINS=*
```

## Quick Diagnostic Commands

```bash
# 1. Is API alive?
curl https://YOUR-URL.up.railway.app/health

# 2. Is model loaded?
curl https://YOUR-URL.up.railway.app/debug

# 3. Test full API response
python3 test_api_response.py https://YOUR-URL.up.railway.app

# 4. Test all website features
python3 test_railway_website.py https://YOUR-URL.up.railway.app
```

## Report Back

After running `test_api_response.py`, you'll know exactly what's wrong:

### Scenario A: API returns everything correctly

✅ mask_image: PRESENT  
✅ heatmap_image: PRESENT  
✅ overlay_image: PRESENT  

→ **Problem is in the frontend HTML**  
→ Check browser console for JavaScript errors  

### Scenario B: API missing heatmap/overlay

❌ heatmap_image: MISSING  
❌ overlay_image: MISSING  

→ **Problem is in the backend**  
→ Model not loaded or prediction failing  
→ Check Railway logs: `railway logs`

### Scenario C: API returns everything but images are tiny

⚠️ heatmap_image: 100 chars (should be 35000+)  

→ **Images are empty/blank**  
→ Prediction returning all zeros  
→ Model issue - check if it loaded correctly

## Next Steps

1. **Run the test:** `python3 test_api_response.py YOUR-URL`
2. **Share the output** - tell me what you see
3. **I'll provide the exact fix** based on the results

---

**Quick Start:**

```bash
# Replace with your actual Railway URL
export RAILWAY_URL="https://YOUR-APP.up.railway.app"

# Test API
python3 test_api_response.py $RAILWAY_URL

# If that passes, test website
python3 test_railway_website.py $RAILWAY_URL

# Then open in browser
open $RAILWAY_URL
```

Let me know what the test shows! 🔍
