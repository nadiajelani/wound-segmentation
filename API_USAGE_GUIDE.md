# Wound Segmentation API - Usage Guide

## 🎉 Your API is Live and Working!

Based on the successful deployment logs, your wound segmentation API is now running on Railway with:
- ✅ TensorFlow 2.16.1
- ✅ Keras 3.3.3
- ✅ Model loaded successfully (527MB)
- ✅ Input shape: (128, 128, 3) RGB images
- ✅ Output shape: (128, 128, 1) Segmentation masks

## 🔗 API Endpoints

Replace `https://your-app.up.railway.app` with your actual Railway URL.

### 1. Health Check
```bash
curl https://your-app.up.railway.app/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_status": "loaded",
  "version": "1.0.0",
  "python_version": "3.10.15",
  "tensorflow_version": "2.16.1",
  "timestamp": "2025-10-05T20:20:58.000000"
}
```

### 2. Ready Check
```bash
curl https://your-app.up.railway.app/ready
```

**Response:**
```json
{
  "ready": true,
  "model_loaded": true,
  "message": "Service ready to process requests",
  "timestamp": "2025-10-05T20:20:58.000000"
}
```

### 3. Debug Info
```bash
curl https://your-app.up.railway.app/debug
```

**Response:**
```json
{
  "model_loaded": true,
  "model_exists": true,
  "model_path": "/app/models/simclr_unet_patch_wound.keras",
  "model_path_exists": true,
  "model_input_shape": [null, 128, 128, 3],
  "model_output_shape": [null, 128, 128, 1],
  "timestamp": "2025-10-05T20:20:58.000000"
}
```

### 4. Analyze Wound (Main Endpoint)

#### Using File Upload:
```bash
curl -X POST https://your-app.up.railway.app/analyze \
  -F "image=@/path/to/wound_image.jpg"
```

#### Using Base64 (JSON):
```bash
curl -X POST https://your-app.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }'
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "area_pixels": 1234,
    "area_percentage": 7.59,
    "perimeter": 156.78,
    "severity": "Moderate"
  },
  "mask_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...",
  "timestamp": "2025-10-05T20:25:00.000000"
}
```

## 🧪 Testing Your API

### Quick Test Script

Save this as `test_api.py`:

```python
#!/usr/bin/env python3
import requests
import json
import sys
from pathlib import Path

API_URL = "https://your-app.up.railway.app"  # Replace with your URL

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.json().get('model_loaded', False)

def test_analyze(image_path):
    """Test wound analysis"""
    print("\n" + "="*60)
    print(f"Testing Wound Analysis: {image_path}")
    print("="*60)
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{API_URL}/analyze", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('success')}")
        print(f"Metrics: {json.dumps(result.get('metrics'), indent=2)}")
        
        # Save mask image
        mask_data = result.get('mask_image', '')
        if mask_data.startswith('data:image/png;base64,'):
            import base64
            mask_bytes = base64.b64decode(mask_data.split(',')[1])
            mask_path = Path(image_path).stem + '_mask.png'
            with open(mask_path, 'wb') as f:
                f.write(mask_bytes)
            print(f"Mask saved to: {mask_path}")
    else:
        print(f"Error: {response.text}")

if __name__ == '__main__':
    # Test health
    model_loaded = test_health()
    
    if not model_loaded:
        print("\n⚠️ Model not loaded yet. Wait a moment and try again.")
        sys.exit(1)
    
    # Test analysis
    if len(sys.argv) > 1:
        test_analyze(sys.argv[1])
    else:
        print("\nUsage: python test_api.py <path_to_wound_image>")
        print("Example: python test_api.py wound_sample.jpg")
```

**Run it:**
```bash
# Test health
python test_api.py

# Test with an image
python test_api.py path/to/wound_image.jpg
```

### Using cURL

```bash
# 1. Check health
curl https://your-app.up.railway.app/health

# 2. Analyze a wound image
curl -X POST https://your-app.up.railway.app/analyze \
  -F "image=@wound_sample.jpg" \
  -o result.json

# 3. View results
cat result.json | python -m json.tool
```

### Using Python Requests

```python
import requests
import json

# Your Railway URL
API_URL = "https://your-app.up.railway.app"

# 1. Check if API is ready
response = requests.get(f"{API_URL}/ready")
print(response.json())

# 2. Analyze a wound image
with open('wound_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post(f"{API_URL}/analyze", files=files)

result = response.json()
print(json.dumps(result, indent=2))

# 3. Save the mask
if result['success']:
    import base64
    mask_data = result['mask_image'].split(',')[1]
    mask_bytes = base64.b64decode(mask_data)
    
    with open('wound_mask.png', 'wb') as f:
        f.write(mask_bytes)
    
    print(f"Wound area: {result['metrics']['area_percentage']:.2f}%")
    print(f"Severity: {result['metrics']['severity']}")
```

## 📊 Understanding the Response

### Metrics Explanation

- **area_pixels**: Number of pixels identified as wound
- **area_percentage**: Percentage of image that is wound (0-100%)
- **perimeter**: Perimeter of wound contour in pixels
- **severity**: Classification based on area percentage
  - **Mild**: < 1%
  - **Moderate**: 1-5%
  - **Severe**: > 5%

### Mask Image

The `mask_image` is returned as a base64-encoded PNG:
- White pixels (255) = Wound area
- Black pixels (0) = Non-wound area
- Size: 128x128 (same as input)

## 🔧 Advanced Usage

### Batch Processing

```python
import requests
from pathlib import Path
import json

API_URL = "https://your-app.up.railway.app"

def analyze_batch(image_folder):
    """Analyze all images in a folder"""
    results = []
    
    for image_path in Path(image_folder).glob('*.jpg'):
        print(f"Processing {image_path.name}...")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{API_URL}/analyze", files=files)
        
        if response.status_code == 200:
            result = response.json()
            results.append({
                'filename': image_path.name,
                'metrics': result['metrics']
            })
    
    # Save results
    with open('batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} images")

# Run batch analysis
analyze_batch('wound_images/')
```

### Integration with Web Frontend

```javascript
// JavaScript example
async function analyzeWound(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch('https://your-app.up.railway.app/analyze', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  
  if (result.success) {
    console.log('Wound metrics:', result.metrics);
    
    // Display mask
    const maskImage = document.getElementById('mask');
    maskImage.src = result.mask_image;
  }
}
```

## 🚨 Error Handling

### Common Errors

**400 Bad Request**
```json
{"error": "No image provided"}
```
→ Make sure you're sending an image file or base64 data

**500 Internal Server Error**
```json
{"error": "Analysis failed: ..."}
```
→ Check image format (JPEG, PNG supported) and size (<8MB)

**503 Service Unavailable**
→ Model is still loading, wait a moment and retry

## 🎯 Performance Tips

1. **Image Size**: Resize images to ~512x512 before uploading for faster processing
2. **Batch Processing**: Use concurrent requests for multiple images
3. **Caching**: Cache results for identical images
4. **Error Retry**: Implement exponential backoff for transient errors

## 📈 Monitoring

Check API health periodically:
```bash
# Simple monitoring script
while true; do
  curl -s https://your-app.up.railway.app/health | \
    jq '.model_loaded, .timestamp'
  sleep 60
done
```

## 🔐 Security Notes

- Current CORS: Allow all origins (`*`)
- For production: Set specific origins in Railway `CORS_ORIGINS` variable
- No authentication currently - add if needed
- Rate limiting: Not implemented - consider adding

## 📚 Additional Resources

- **Troubleshooting**: See `HEALTHCHECK_TROUBLESHOOTING.md`
- **Deployment**: See `RAILWAY_KERAS3_SETUP.md`
- **Debug Guide**: See `DEPLOYMENT_DEBUG.md`

## 🎉 Success!

Your wound segmentation API is fully operational. The deployment successfully:
- ✅ Upgraded to Keras 3 and TensorFlow 2.16
- ✅ Downloaded and loaded the 527MB model
- ✅ Passed Railway healthcheck
- ✅ Ready to process wound images

Happy analyzing! 🏥🔬
