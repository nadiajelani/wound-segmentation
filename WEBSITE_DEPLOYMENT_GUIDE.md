# 🌐 Website Deployment Guide

## Wound Analyzer - Frontend with Segmentation & Heatmap Visualization

### Overview

The `wound_analyzer.html` file provides a complete, production-ready web interface for your wound segmentation API with:

✅ **Image Upload** - Drag & drop or file browser  
✅ **Real-time Analysis** - Connects to your Railway API  
✅ **Segmentation Mask** - Binary visualization of wound area  
✅ **Confidence Heatmap** - Color-coded confidence levels  
✅ **Clinical Metrics** - Area, perimeter, severity  
✅ **Download Options** - Save masks, heatmaps, and reports  
✅ **Responsive Design** - Works on desktop and mobile  

---

## 🎨 Features

### 1. **Segmentation Visualization**
- Binary mask showing wound vs normal tissue
- Black = wound area, White = normal tissue
- Clear legend for interpretation

### 2. **Heatmap Overlay**
- Color gradient showing model confidence
- Blue (low) → Cyan → Green → Yellow → Red (high)
- Overlaid on original image for context

### 3. **Interactive UI**
- Drag & drop image upload
- Real-time preview
- Animated loading states
- Professional design

### 4. **Clinical Metrics**
- Area in pixels and percentage
- Perimeter measurement
- Severity classification (Mild/Moderate/Severe)
- Color-coded severity badges

### 5. **Export Capabilities**
- Download segmentation mask (PNG)
- Download heatmap (PNG)
- Download full report (JSON)

---

## 🚀 Quick Start

### Step 1: Update API URL

Open `wound_analyzer.html` and update line 595:

```javascript
const API_URL = 'https://your-app.up.railway.app'; // Replace with your actual Railway URL
```

Replace with your actual Railway deployment URL (e.g., `https://wound-segmentation-production.up.railway.app`).

### Step 2: Test Locally

```bash
# Option 1: Python HTTP Server
python3 -m http.server 8000

# Option 2: Node HTTP Server
npx http-server -p 8000

# Then open: http://localhost:8000/wound_analyzer.html
```

### Step 3: Test the Features

1. **Upload an image** (drag & drop or click)
2. **Click "Analyze Wound"**
3. **View results**:
   - Original image
   - Segmentation mask
   - Confidence heatmap
   - Clinical metrics
4. **Download results**

---

## 📦 Deployment Options

### Option 1: Vercel (Recommended - Free & Easy)

**Fastest deployment:**

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy (from project directory)
vercel

# Follow prompts, then your site is live!
```

**Or use Vercel website:**
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub
3. Import your repository
4. Deploy!

**Result**: `https://wound-analyzer.vercel.app`

### Option 2: Netlify (Free Tier)

**Drag & Drop deployment:**

1. Go to [netlify.com](https://netlify.com)
2. Sign up
3. Drag `wound_analyzer.html` to Netlify
4. Site is live!

**Or use Netlify CLI:**

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Login
netlify login

# Deploy
netlify deploy --prod

# Select 'wound_analyzer.html' as entry point
```

**Result**: `https://wound-analyzer.netlify.app`

### Option 3: GitHub Pages (Free)

**Deploy directly from your repo:**

1. Push `wound_analyzer.html` to your GitHub repo
2. Go to Settings → Pages
3. Select branch: `main` or `main-fixed`
4. Select folder: `/ (root)`
5. Save

**Result**: `https://nadiajelani.github.io/wound-segmentation/wound_analyzer.html`

### Option 4: Railway Static Site

**Deploy alongside your API:**

1. Add to your Railway project
2. Create new service
3. Select "Static Site"
4. Point to `wound_analyzer.html`

**Benefit**: Same domain as API, no CORS issues!

### Option 5: AWS S3 + CloudFront

**For production scale:**

```bash
# Upload to S3
aws s3 cp wound_analyzer.html s3://your-bucket/

# Enable static website hosting
aws s3 website s3://your-bucket/ --index-document wound_analyzer.html

# Optional: Add CloudFront CDN for global delivery
```

---

## 🔧 Configuration

### Update API Endpoint

In `wound_analyzer.html`, find this line:

```javascript
const API_URL = 'https://your-app.up.railway.app';
```

Replace with:
- Railway: `https://your-app-name.up.railway.app`
- Local dev: `http://localhost:8080`
- Custom domain: `https://api.yourdomain.com`

### Enable CORS (if needed)

If deploying to different domain than API, update Railway environment variables:

```bash
# In Railway → Variables
CORS_ORIGINS=https://wound-analyzer.vercel.app,https://wound-analyzer.netlify.app
```

Or allow all (for testing):

```bash
CORS_ORIGINS=*
```

---

## 🎯 How It Works

### 1. Image Upload Flow

```
User selects image
    ↓
Validate file (type, size)
    ↓
Show preview
    ↓
Enable "Analyze" button
```

### 2. Analysis Flow

```
User clicks "Analyze"
    ↓
Show loading spinner
    ↓
Send POST to /analyze endpoint
    ↓
Receive result with mask
    ↓
Generate heatmap from mask
    ↓
Display all visualizations
```

### 3. Heatmap Generation

```javascript
// Pseudo-code
maskImage → Extract pixel intensities
    ↓
For each pixel:
  intensity = pixelValue / 255
  color = mapIntensityToColor(intensity)
    ↓
Overlay colored heatmap on original image
```

**Color Mapping:**
- 0-25%: Blue → Cyan (low confidence)
- 25-50%: Cyan → Green (medium-low)
- 50-75%: Green → Yellow (medium-high)
- 75-100%: Yellow → Red (high confidence)

---

## 📊 UI Components Breakdown

### Upload Section
```html
<div class="upload-section">
  - Drag & drop zone
  - File browser button
  - File validation
  - Preview display
</div>
```

### Results Section
```html
<div class="results-section">
  <div class="image-grid">
    - Original Image
    - Segmentation Mask
    - Confidence Heatmap
  </div>
  <div class="metrics-grid">
    - Area (pixels)
    - Area (percentage)
    - Perimeter
    - Severity badge
  </div>
  <div class="download-buttons">
    - Download mask
    - Download heatmap
    - Download report
  </div>
</div>
```

---

## 🎨 Customization

### Change Colors

**Primary gradient:**
```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

Replace with your brand colors:
```css
background: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
```

### Change Severity Thresholds

In JavaScript section:

```javascript
// Current thresholds
Mild: < 1%
Moderate: 1-5%
Severe: > 5%

// To change, modify in the displayResults function
```

### Add Logo

Add your logo in the header:

```html
<header>
    <img src="your-logo.png" style="height: 50px; margin-bottom: 10px;">
    <h1>🏥 Wound Analyzer</h1>
</header>
```

### Change Heatmap Colors

Modify the `getHeatmapColor()` function:

```javascript
function getHeatmapColor(value) {
    // Customize your gradient here
    // Current: Blue → Cyan → Green → Yellow → Red
    // Change RGB values for different colors
}
```

---

## 🔐 Security Considerations

### 1. API Key (Optional)

If your API requires authentication:

```javascript
const response = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    headers: {
        'X-API-Key': 'YOUR_API_KEY'
    },
    body: formData
});
```

### 2. File Validation

Already implemented:
- ✅ File type check (images only)
- ✅ File size limit (8MB)
- ✅ Error handling

### 3. HTTPS

Always deploy with HTTPS:
- Vercel/Netlify: Automatic HTTPS
- Railway: Automatic HTTPS
- Custom domain: Use Let's Encrypt

---

## 📱 Mobile Optimization

### Responsive Design

Already included:
- ✅ Mobile-friendly grid layout
- ✅ Touch-friendly buttons
- ✅ Responsive images
- ✅ Optimized for small screens

### Testing on Mobile

```bash
# Test with different screen sizes
# Open browser DevTools → Toggle device toolbar
# Or test on actual devices
```

---

## 🐛 Troubleshooting

### Issue: "Could not connect to API"

**Solutions:**
1. Check API URL is correct
2. Verify Railway API is running
3. Check CORS settings
4. Look at browser console for errors

```javascript
// In browser console:
console.log('API URL:', API_URL);
```

### Issue: "Images not loading"

**Solutions:**
1. Check file size < 8MB
2. Verify file is JPEG/PNG
3. Check browser console for errors

### Issue: "Heatmap not showing"

**Solutions:**
1. Ensure mask image loaded successfully
2. Check canvas size is correct
3. Verify mask image format

### Issue: "CORS error"

**Solution:**
```bash
# In Railway → Variables
CORS_ORIGINS=https://your-frontend-domain.com

# Or temporarily for testing:
CORS_ORIGINS=*
```

---

## 📈 Performance Optimization

### 1. Image Optimization

```javascript
// Before upload, resize large images:
function resizeImage(file, maxWidth, maxHeight) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;
                
                if (width > maxWidth || height > maxHeight) {
                    if (width > height) {
                        height *= maxWidth / width;
                        width = maxWidth;
                    } else {
                        width *= maxHeight / height;
                        height = maxHeight;
                    }
                }
                
                canvas.width = width;
                canvas.height = height;
                canvas.getContext('2d').drawImage(img, 0, 0, width, height);
                canvas.toBlob(resolve, 'image/jpeg', 0.9);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    });
}
```

### 2. Loading States

Already implemented:
- ✅ Spinner during analysis
- ✅ Disabled buttons during processing
- ✅ Clear feedback messages

### 3. Caching

Add service worker for offline support:

```javascript
// service-worker.js
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open('wound-analyzer-v1').then((cache) => {
            return cache.addAll([
                '/wound_analyzer.html',
                // Add other assets
            ]);
        })
    );
});
```

---

## 🎯 Testing Checklist

Before deploying:

- [ ] Update API_URL to production endpoint
- [ ] Test image upload (drag & drop)
- [ ] Test image upload (file browser)
- [ ] Test with various image formats
- [ ] Test with large images (near 8MB)
- [ ] Test on mobile devices
- [ ] Test on different browsers
- [ ] Test download functions
- [ ] Verify heatmap colors
- [ ] Check segmentation mask display
- [ ] Test error handling
- [ ] Verify CORS configuration

---

## 🎊 Final Steps

### 1. Deploy Frontend

```bash
# Choose your method (Vercel recommended)
vercel
```

### 2. Update Documentation

Add your frontend URL to `README.md`:

```markdown
## 🌐 Live Demo

- **API**: https://your-api.up.railway.app
- **Web Interface**: https://wound-analyzer.vercel.app
```

### 3. Share with Users

Your users can now:
1. Visit your website
2. Upload wound images
3. Get instant analysis
4. Download results

No API knowledge required! 🎉

---

## 📞 Support

For issues:
1. Check browser console for errors
2. Verify API is running: `curl https://your-api.up.railway.app/health`
3. Check CORS settings in Railway
4. Review `HEALTHCHECK_TROUBLESHOOTING.md`

---

## 🔄 Updates & Maintenance

### Updating API URL

```bash
# Edit wound_analyzer.html
# Change API_URL constant
# Redeploy: vercel --prod
```

### Adding Features

The HTML file is self-contained. Add features by:
1. Editing the HTML/CSS/JavaScript
2. Testing locally
3. Redeploying

### Version Control

```bash
git add wound_analyzer.html WEBSITE_DEPLOYMENT_GUIDE.md
git commit -m "Add website with segmentation and heatmap visualization"
git push
```

---

## ✨ Success!

You now have a complete wound analysis system:

- ✅ AI API on Railway
- ✅ Beautiful web interface
- ✅ Segmentation visualization
- ✅ Confidence heatmap
- ✅ Clinical metrics
- ✅ Download capabilities

**Your users can now analyze wounds with a single click!** 🎉

---

**Next**: Deploy to Vercel/Netlify and share the URL!
