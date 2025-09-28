# 🚀 Phase 3a: Free Public Wound Detection Website

## Quick Start (15 minutes)

### Option 1: Render (Recommended - Easiest)

#### Step 1: Prepare Your Code
```bash
# Ensure you have these files in your repo:
- app_free.py (optimized for free tier)
- requirements_free.txt (minimal dependencies)
- render.yaml (Render configuration)
- index_free.html (frontend)
- models/simclr_unet_patch_wound.keras (your trained model)
```

#### Step 2: Deploy to Render
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Render will auto-detect the configuration
6. Set environment variables:
   - `SECRET_KEY`: Generate a random secret key
   - `PORT`: 10000 (Render requirement)

#### Step 3: Upload Your Model
1. In Render dashboard, go to your service
2. Click "Environment" tab
3. Add your model file to the persistent disk
4. Or use the file upload feature

**Result**: `https://your-app-name.onrender.com`

### Option 2: Railway (Alternative)

#### Step 1: Prepare for Railway
```bash
# Create railway.json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python app_free.py",
    "healthcheckPath": "/health"
  }
}
```

#### Step 2: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects and deploys

**Result**: `https://your-app-name.railway.app`

### Option 3: GitHub Pages + Render API

#### Frontend (GitHub Pages)
1. Push `index_free.html` to your repo
2. Go to repository Settings → Pages
3. Select "Deploy from a branch" → "main"
4. Your site will be at: `https://username.github.io/repo-name`

#### Backend (Render)
1. Deploy `app_free.py` to Render (as above)
2. Update the `API_URL` in `index_free.html` to your Render URL
3. Push the updated frontend

**Result**: 
- Frontend: `https://username.github.io/wound-detection`
- Backend: `https://your-api.onrender.com`

## Configuration

### Environment Variables
```bash
# Required
SECRET_KEY=your-secret-key-here
PORT=10000

# Optional
MAX_CONTENT_LENGTH=8388608  # 8MB
CORS_ORIGINS=https://your-frontend.com
```

### Model Setup
1. **Upload Model**: Add your `simclr_unet_patch_wound.keras` to the models directory
2. **Model Path**: The app will look for `/app/models/simclr_unet_patch_wound.keras`
3. **Fallback**: If model not found, uses basic computer vision fallback

## Free Tier Limitations

### Render Free Tier
- **RAM**: 512MB
- **Hours**: 750/month
- **Sleep**: After 15 minutes of inactivity
- **Storage**: 1GB persistent disk

### Railway Free Tier
- **RAM**: 1GB
- **Hours**: 500/month
- **Sleep**: After inactivity
- **Storage**: 1GB

### Solutions for Limitations
1. **Cold Starts**: Use uptime monitoring to keep service awake
2. **Memory**: Optimize model size or use smaller models
3. **Storage**: Use external model hosting (Hugging Face, Google Drive)

## Testing Your Deployment

### Health Check
```bash
curl https://your-app.onrender.com/health
# Expected: {"status": "healthy", "model_loaded": true}
```

### Test Analysis
```bash
curl -X POST https://your-app.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_data": "base64-encoded-image"}'
```

### Frontend Test
1. Open your GitHub Pages URL
2. Upload a test image
3. Verify analysis works
4. Check that results display correctly

## Monitoring & Maintenance

### Free Monitoring Options
1. **Uptime Robot**: Monitor your API endpoint
2. **Render Dashboard**: Built-in metrics
3. **GitHub Actions**: Automated health checks

### Maintenance Tasks
1. **Weekly**: Check service status
2. **Monthly**: Review usage and performance
3. **As Needed**: Update dependencies

## Troubleshooting

### Common Issues

#### 1. Model Loading Fails
```bash
# Check model path
curl https://your-app.onrender.com/health
# If model_loaded: false, check model file location
```

#### 2. Memory Issues
- Reduce model size
- Use model compression
- Implement model caching

#### 3. Cold Start Delays
- Use uptime monitoring
- Implement health check pings
- Consider paid tier for production

#### 4. CORS Errors
- Check CORS_ORIGINS environment variable
- Ensure frontend URL is allowed

### Debug Commands
```bash
# Check service status
curl -I https://your-app.onrender.com/health

# Test with sample image
python -c "
import base64, requests
with open('test_image.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
response = requests.post('https://your-app.onrender.com/analyze', 
                       json={'image_data': img_b64})
print(response.json())
"
```

## Scaling Up (When Ready)

### Upgrade Paths
1. **Render**: $7/month for more resources
2. **Railway**: $5/month for better performance
3. **Custom Domain**: $10-15/year for professional URL

### Advanced Features
1. **User Accounts**: Add authentication
2. **Database**: Store analysis history
3. **Advanced Analytics**: Detailed reporting
4. **Mobile App**: React Native wrapper

## Success Metrics

### Free Tier Targets
- **Uptime**: 95%+ (with monitoring)
- **Response Time**: <10 seconds (including cold start)
- **Concurrent Users**: 5-10
- **Monthly Requests**: <1000

### Cost Tracking
- **Hosting**: $0/month
- **Domain**: $0 (using subdomains)
- **Monitoring**: $0 (using free tools)
- **Total**: $0/month

## Next Steps

1. **Deploy**: Follow the steps above
2. **Test**: Verify everything works
3. **Monitor**: Set up basic monitoring
4. **Share**: Make your service public
5. **Iterate**: Collect feedback and improve

## Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **GitHub Pages**: [pages.github.com](https://pages.github.com)

Your free wound detection website is now ready for the world! 🌍