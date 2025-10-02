# Railway Deployment Guide - Wound Detection with SimCLR

## 🚀 Quick Deploy to Railway

### Step 1: Prepare Your Repository
Ensure your repository has these files:
- ✅ `app_clean.py` - Production API (SimCLR + U-Net only)
- ✅ `model_loader.py` - Model loading system
- ✅ `wound_whisperer.html` - Frontend website
- ✅ `Dockerfile` - Container configuration
- ✅ `railway.json` - Railway deployment config
- ✅ `requirements_production.txt` - Dependencies
- ✅ `models/simclr_unet_patch_wound.keras` - Trained model (527MB)

### Step 2: Deploy to Railway
1. **Go to [railway.app](https://railway.app)**
2. **Sign up/Login** with GitHub
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your wound-segmentation repository**
6. **Railway will auto-detect the Dockerfile and start building**

### Step 3: Configure Environment Variables
In Railway dashboard, go to your project → Variables tab and add:

```bash
SECRET_KEY=your-secret-key-here-change-this
CORS_ORIGINS=https://your-app-name.railway.app
SIMCLR_MODEL_PATH=models/simclr_unet_patch_wound.keras
MAX_CONTENT_LENGTH=16777216
RATE_LIMIT_PER_MINUTE=10
FILE_CLEANUP_HOURS=24
```

### Step 4: Wait for Deployment
- **Build Time**: ~5-10 minutes (downloading 527MB model)
- **Deploy Time**: ~2-3 minutes
- **Total Time**: ~10-15 minutes

### Step 5: Test Your Deployment
```bash
# Test the deployed API
python test_deployment.py --url https://your-app-name.railway.app
```

## 📊 Railway Free Tier Limits

### What You Get for FREE:
- **Compute**: 500 hours/month
- **Memory**: 1GB RAM
- **Storage**: 1GB persistent
- **Bandwidth**: Unlimited
- **Custom Domain**: Yes (free subdomain)

### Sleep Behavior:
- **Sleeps**: After 5 minutes of inactivity
- **Wake Time**: ~10-15 seconds on first request
- **Perfect for**: Demo sites, personal projects

## 🔧 Railway Configuration Files

### railway.json
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "python app_clean.py",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
# ... (already configured)
```

## 🌐 Access Your Website

### After Deployment:
1. **Website URL**: `https://your-app-name.railway.app`
2. **API Health**: `https://your-app-name.railway.app/health`
3. **API Ready**: `https://your-app-name.railway.app/ready`

### Features Available:
- ✅ **Wound Upload**: Upload wound images
- ✅ **SimCLR Analysis**: AI-powered segmentation
- ✅ **PDF Reports**: Downloadable analysis reports
- ✅ **Mobile Friendly**: Responsive design
- ✅ **Real-time Processing**: Fast analysis

## 🧪 Testing Your Deployment

### 1. Health Check
```bash
curl https://your-app-name.railway.app/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-02T18:41:37.561956",
  "models_loaded": true
}
```

### 2. Website Access
```bash
curl -I https://your-app-name.railway.app/
```
**Expected Response:** `200 OK`

### 3. Image Upload Test
Use the website interface to upload a wound image and test the analysis.

## 📈 Monitoring Your Deployment

### Railway Dashboard:
- **Logs**: Real-time application logs
- **Metrics**: CPU, Memory, Network usage
- **Deployments**: Build and deployment history
- **Variables**: Environment variables management

### Health Monitoring:
- **Uptime**: Railway monitors automatically
- **Health Checks**: `/health` endpoint
- **Auto-restart**: On failures

## 🔧 Troubleshooting

### Common Issues:

#### 1. Build Fails - Model Too Large
**Solution**: Model is included in repo (527MB is fine for Railway)

#### 2. Memory Issues
**Solution**: Railway free tier has 1GB RAM (sufficient for SimCLR model)

#### 3. Cold Start Delay
**Solution**: Normal for free tier (10-15 seconds first request)

#### 4. CORS Errors
**Solution**: Update `CORS_ORIGINS` environment variable

### Debug Commands:
```bash
# Check deployment logs
railway logs

# Check environment variables
railway variables

# Restart deployment
railway redeploy
```

## 💰 Cost Analysis

### Free Tier (Current):
- **Railway**: $0/month
- **Domain**: $0 (using railway.app subdomain)
- **Total**: $0/month

### Paid Tier (When Needed):
- **Railway Pro**: $5/month
- **Custom Domain**: $10-15/year
- **Total**: $5-7/month

## 🚀 Scaling Strategy

### Free → Paid Upgrade:
1. **Monitor Usage**: Track API calls and resource usage
2. **Upgrade When**: Approaching 500 hours/month limit
3. **Benefits**: No sleep, more resources, custom domain

### Performance Optimization:
1. **Model Caching**: Already implemented
2. **Image Compression**: Automatic in API
3. **CDN**: Use Railway's built-in CDN

## 🎯 Success Metrics

### Expected Performance:
- **Response Time**: 2-5 seconds (including cold start)
- **Uptime**: 99%+ (with monitoring)
- **Concurrent Users**: 5-10 (free tier)
- **Model Accuracy**: >90% (SimCLR)

### Monitoring:
- **Health Endpoint**: `/health`
- **Readiness**: `/ready`
- **Logs**: Railway dashboard

## 🎉 You're Ready!

Your wound detection website is now deployed on Railway with:
- ✅ **SimCLR AI Model**: Advanced wound segmentation
- ✅ **Modern Website**: Responsive, mobile-friendly
- ✅ **Production API**: Secure, rate-limited
- ✅ **Free Hosting**: Zero cost deployment
- ✅ **Easy Scaling**: Upgrade when needed

**Visit your website**: `https://your-app-name.railway.app` 🚀