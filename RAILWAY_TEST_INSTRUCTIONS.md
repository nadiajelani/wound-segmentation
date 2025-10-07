# 🧪 Railway Website Testing Instructions

## Quick Test Your Railway Deployment

### Step 1: Find Your Railway URL

1. Go to [railway.app](https://railway.app)
2. Click on your wound-segmentation project
3. Click on "Settings" → "Domains"
4. Copy your deployment URL (e.g., `https://wound-segmentation-production.up.railway.app`)

### Step 2: Test the API Response

Run this command to check what your API is actually returning:

```bash
python3 test_api_response.py https://YOUR-RAILWAY-URL.up.railway.app
```

Replace `YOUR-RAILWAY-URL` with your actual Railway domain.

**Expected output:**
```
✅ mask_image: PRESENT
✅ heatmap_image: PRESENT  
✅ overlay_image: PRESENT
✅ Healing Stage: [stage name]
✅ Doctor Report ID: [report id]
✅ Metrics: [area, severity, etc.]

✅ ALL REQUIRED DATA PRESENT!
```

**If you see:**
```
❌ heatmap_image: MISSING
❌ overlay_image: MISSING
```

Then the backend is not generating the heatmap/overlay properly.

### Step 3: Test the Full Website

Run the comprehensive test:

```bash
python3 test_railway_website.py https://YOUR-RAILWAY-URL.up.railway.app
```

**Expected output:**
```
✅ PASSED: HEALTH
✅ PASSED: HOMEPAGE
✅ PASSED: ANALYZE
✅ PASSED: DOWNLOAD

🎉 ALL TESTS PASSED!
```

### Step 4: Test in Browser

1. Open your Railway URL in a browser: `https://YOUR-RAILWAY-URL.up.railway.app`
2. You should see the "Wound Analyzer" interface
3. Upload a test image (any wound image or even a regular photo)
4. Click "🔍 Analyze Wound"
5. Wait for analysis (may take 5-30 seconds)

**You should see 4 images:**
1. **📷 Original Image** - Your uploaded image
2. **🎯 Segmentation Mask** - Black (wound) and white (normal tissue)
3. **🌡️ Confidence Heatmap** - Color-coded confidence levels (blue to red)
4. **🎨 Heatmap Overlay** - Heatmap blended with original image

**You should also see:**
- 📈 **Wound Metrics** - Area, percentage, perimeter, severity
- 🩹 **Healing Stage Assessment** - Stage, confidence, recommendations
- 📋 **Clinical Report** - Findings, follow-up care

### Common Issues & Fixes

#### Issue 1: Only seeing mask, no heatmap

**Symptoms:**
- Only 1-2 images appear instead of 4
- Heatmap image is blank or missing
- Overlay image is missing

**Possible causes:**
1. Backend not generating heatmap
2. Images not being encoded properly
3. Frontend not displaying images correctly

**Fix:**
Run the test script to check:
```bash
python3 test_api_response.py https://YOUR-URL.up.railway.app
```

If it says heatmap_image is MISSING, then the backend has an issue.

#### Issue 2: No doctor report showing

**Symptoms:**
- Doctor report section is empty or shows "-"
- Clinical findings missing

**Fix:**
Check the API response:
```bash
python3 test_api_response.py https://YOUR-URL.up.railway.app
```

Look for `doctor_report` in the output.

#### Issue 3: API errors or 500 responses

**Symptoms:**
- Error message appears
- Analysis fails
- Console shows errors

**Fix:**
1. Check Railway logs: `railway logs`
2. Check if model is loaded: `curl https://YOUR-URL/debug`
3. Verify environment variables are set correctly

#### Issue 4: CORS errors

**Symptoms:**
- Browser console shows: "CORS policy blocked"
- API calls fail from browser

**Fix:**
In Railway, add environment variable:
```
CORS_ORIGINS=*
```

Or for specific domain:
```
CORS_ORIGINS=https://your-frontend-domain.com
```

### Debugging with Browser Console

1. Open your Railway URL
2. Press F12 (or Cmd+Option+I on Mac)
3. Go to "Console" tab
4. Upload and analyze an image
5. Look for errors (red text)

Common console errors and meanings:

```
❌ "Failed to fetch" → API is down or CORS issue
❌ "404 Not Found" → Wrong API endpoint
❌ "500 Internal Server Error" → Backend error (check Railway logs)
❌ "Cannot read property 'mask_image'" → API returning wrong data structure
```

### Check Backend Logs

To see what's happening on the server:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# View logs
railway logs

# Or view in Railway dashboard:
# https://railway.app → Your Project → Deployments → View Logs
```

Look for:
- Model loading messages
- Analysis request logs
- Error messages or stack traces

### Manual API Test with curl

Test the API directly:

```bash
# Health check
curl https://YOUR-URL.up.railway.app/health

# Debug info
curl https://YOUR-URL.up.railway.app/debug

# Analyze endpoint (with a test image)
curl -X POST https://YOUR-URL.up.railway.app/analyze \
  -F "image=@path/to/test/image.jpg" \
  | jq '.' > response.json

# Check the response
cat response.json
```

Check if `response.json` contains:
- `mask_image`
- `heatmap_image`
- `overlay_image`
- `doctor_report`

### What to Do If Tests Fail

#### If `test_api_response.py` shows MISSING data:

The backend is not generating all required outputs.

**Check:**
1. Is the model loaded? → `curl https://YOUR-URL/debug`
2. Are there errors in logs? → `railway logs`
3. Is `app.py` deployed correctly? → Check Railway build logs

#### If website doesn't load at all:

1. Check if app is running: `curl https://YOUR-URL/health`
2. Check Railway deployment status
3. Verify `wound_analyzer.html` is included in deployment

#### If analysis takes too long:

- First analysis after deployment can take 1-2 minutes (model loading)
- Subsequent analyses should be faster (5-30 seconds)
- Check Railway logs for timeout messages

### Success Criteria

✅ **Healthy deployment should have:**

1. `/health` returns 200 with `"model_loaded": true`
2. Homepage shows Wound Analyzer interface
3. `/analyze` returns all 3 images (mask, heatmap, overlay)
4. `/analyze` returns healing_stage and doctor_report
5. Frontend displays all 4 image panels
6. Metrics, healing stage, and clinical report all populate
7. Download buttons work

### Next Steps After Testing

Once all tests pass:

1. **Test with real wound images**
   - Try different wound types
   - Test various image sizes
   - Check if severity classification makes sense

2. **Share with users**
   - Send them the Railway URL
   - Collect feedback
   - Monitor for errors

3. **Monitor performance**
   - Check Railway metrics
   - Watch for timeouts
   - Monitor memory usage

4. **Optional: Add custom domain**
   - Purchase domain (e.g., from Namecheap)
   - Add CNAME record pointing to Railway
   - Configure in Railway settings

### Support

If you still have issues after running tests:

1. Share the output of `test_api_response.py`
2. Share Railway logs (if any errors)
3. Share browser console errors (if any)

---

**Quick Reference:**

```bash
# Test what API returns
python3 test_api_response.py https://YOUR-URL.up.railway.app

# Full website test
python3 test_railway_website.py https://YOUR-URL.up.railway.app

# Check API health
curl https://YOUR-URL.up.railway.app/health

# View logs
railway logs
```

Good luck! 🚀
