# 🧪 Test Your API Now!

## Step 1: Get Your Railway URL

1. Go to [Railway Dashboard](https://railway.app/)
2. Click on your wound-segmentation service
3. Go to **Settings** → **Networking** (or **Domains**)
4. Copy the public URL (looks like: `https://something.up.railway.app`)

## Step 2: Run the Test

### Option A: Using the Simple Script (Recommended)

```bash
# Replace with your actual Railway URL
./run_test.sh https://your-app.up.railway.app
```

This will:
- ✅ Test all endpoints
- ✅ Automatically use a sample image from uploads/
- ✅ Create a test image if none found
- ✅ Save the segmentation mask
- ✅ Show you detailed results

### Option B: Manual Test with Specific Image

```bash
# Test with a specific wound image
./run_test.sh https://your-app.up.railway.app uploads/wound_20250928_163620.jpg
```

### Option C: Using Python Directly

```bash
# Test all endpoints + create test image
python3 test_api.py https://your-app.up.railway.app

# Test with specific image
python3 test_api.py https://your-app.up.railway.app uploads/wound_20250928_163620.jpg
```

### Option D: Quick cURL Test

```bash
# Just test if API is responding
curl https://your-app.up.railway.app/health | python3 -m json.tool
```

## What You'll See

When you run the test, you should see:

```
🏥 WOUND SEGMENTATION API - COMPREHENSIVE TEST
======================================================================

======================================================================
  Testing Root Endpoint
======================================================================
✅ Status Code: 200
✅ Response:
{
  "message": "Wound Segmentation API",
  "status": "running",
  "model_loaded": true,
  ...
}

======================================================================
  Testing Health Check
======================================================================
✅ Status Code: 200
✅ Response:
{
  "status": "healthy",
  "model_loaded": true,
  "tensorflow_version": "2.16.1",
  ...
}

🎉 Model is loaded and ready!

======================================================================
  Testing Wound Analysis: uploads/wound_20250928_163620.jpg
======================================================================
📤 Uploading image...
✅ Status Code: 200

🎉 Analysis Successful!

📊 Metrics:
  • Area (pixels): 1234
  • Area (%): 7.59%
  • Perimeter: 156.78 pixels
  • Severity: Moderate

💾 Mask saved to: wound_mask_20251005_203045.png
💾 Full result saved to: wound_result_20251005_203045.json

======================================================================
  TEST SUMMARY
======================================================================
✅ ROOT: PASSED
✅ HEALTH: PASSED
✅ READY: PASSED
✅ DEBUG: PASSED
✅ ANALYZE: PASSED

📊 Results: 5 passed, 0 failed, 0 skipped

🎉 ALL TESTS PASSED! Your API is working perfectly! 🎉
```

## Troubleshooting

### If you see "Connection refused" or "Failed to connect"

**Problem:** Can't reach the Railway URL

**Solutions:**
1. Check your Railway URL is correct (no typos)
2. Make sure deployment is running in Railway dashboard
3. Check Railway logs for errors
4. Try accessing the URL in your browser first

### If you see "Model not loaded yet"

**Problem:** Model is still downloading

**Solution:** Wait 30 seconds and run the test again. The model takes ~10 seconds to download on first start.

### If you see "404 Not Found"

**Problem:** Wrong URL or path

**Solutions:**
1. Verify Railway URL in dashboard
2. Check the service is deployed and running
3. Try accessing just the root: `https://your-app.up.railway.app/`

### If you see "500 Internal Server Error"

**Problem:** API is running but crashed during analysis

**Solutions:**
1. Check Railway logs for error details
2. Try with a different/simpler image
3. Check image file is valid (not corrupted)

## Expected Files After Test

After a successful test, you'll have:

1. `wound_mask_TIMESTAMP.png` - The segmentation mask
2. `wound_result_TIMESTAMP.json` - Full analysis results
3. `test_wound_image.jpg` - Test image (if auto-generated)

## Next Steps After Testing

Once your test passes:

1. ✅ **View the mask** - Open the saved mask PNG to see the segmentation
2. ✅ **Check the metrics** - Review the JSON file for detailed results
3. ✅ **Test with more images** - Try different wound types
4. ✅ **Build your frontend** - See NEXT_STEPS.md for frontend options
5. ✅ **Share your API** - Give the URL to others to test

## Quick Reference Commands

```bash
# Test with automatic image selection
./run_test.sh https://your-app.up.railway.app

# Test specific image
./run_test.sh https://your-app.up.railway.app path/to/image.jpg

# Just check health
curl https://your-app.up.railway.app/health

# Get debug info
curl https://your-app.up.railway.app/debug | python3 -m json.tool

# Test analyze endpoint directly
curl -X POST https://your-app.up.railway.app/analyze \
  -F "image=@wound.jpg" -o result.json
```

## Ready to Test?

Run this command now (replace with your Railway URL):

```bash
./run_test.sh https://your-app.up.railway.app
```

Or if you prefer the longer Python command:

```bash
python3 test_api.py https://your-app.up.railway.app uploads/wound_20250928_163620.jpg
```

🎯 **Go ahead and test it now!**
