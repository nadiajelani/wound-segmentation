# Phase 3a: Free Deployment Guide - Wound Detection with SimCLR

## Overview
This guide covers deploying the wound detection website with SimCLR U-Net model integration using free hosting platforms.

## Prerequisites
- GitHub account
- Railway account (free) or Render account (free)
- Your wound detection code in a GitHub repository

## Quick Start (Railway - Recommended)

### 1. Prepare Your Repository
Ensure your repository contains:
- `app_production.py` - Production API with SimCLR integration
- `model_loader.py` - Model loading utilities
- `wound_whisperer.html` - Frontend website
- `Dockerfile` - Container configuration
- `railway.json` - Railway deployment config
- `requirements_production.txt` - Python dependencies
- `models/simclr_unet_patch_wound.keras` - Trained SimCLR model

### 2. Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect the Dockerfile and deploy

### 3. Configure Environment Variables
In Railway dashboard, add these environment variables:
```
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=https://your-app-name.railway.app
SIMCLR_MODEL_PATH=models/simclr_unet_patch_wound.keras
MAX_CONTENT_LENGTH=16777216
RATE_LIMIT_PER_MINUTE=10
FILE_CLEANUP_HOURS=24
```

### 4. Test Deployment
```bash
python test_deployment.py --url https://your-app-name.railway.app
```

## Alternative: Render Deployment

### 1. Create render.yaml
The `render.yaml` file is already configured in the repository.

### 2. Deploy to Render
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Render will auto-detect the `render.yaml` configuration
4. Deploy the service

### 3. Configure Environment Variables
In Render dashboard, add the same environment variables as above.

## Model Integration Details

### SimCLR U-Net Model
The system uses a pre-trained SimCLR U-Net model for wound segmentation:
- **Input**: 128x128x3 RGB images
- **Output**: Binary segmentation mask
- **Features**: Self-supervised learning, better generalization

### Model Loading
The `model_loader.py` handles:
- Loading pre-trained SimCLR model
- Fallback to building from scratch if model not found
- Custom loss functions (Dice + Binary Crossentropy)
- Model information and health checks

### API Endpoints
- `GET /` - Serve the website
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /upload` - Upload and analyze wound images
- `GET /report/<filename>` - Download generated reports

## Frontend Integration

### Website Features
- Modern, responsive design
- Real-time image upload
- SimCLR model integration
- PDF report generation
- Base64 image display

### API Communication
The frontend communicates with the API using:
- FormData for image uploads
- JSON responses for analysis results
- Base64 encoding for image display

## Free Tier Limitations

### Railway Free Tier
- **Compute**: 500 hours/month
- **Memory**: 1GB RAM
- **Storage**: 1GB
- **Sleep**: After 5 minutes of inactivity

### Render Free Tier
- **Compute**: 750 hours/month
- **Memory**: 512MB RAM
- **Sleep**: After 15 minutes of inactivity

### Solutions for Limitations
1. **Cold Starts**: Use health check monitoring
2. **Memory Limits**: Optimize model loading
3. **Storage**: Use external storage for models
4. **Sleep**: Implement wake-up strategies

## Monitoring and Maintenance

### Health Monitoring
```bash
# Check API health
curl https://your-app.railway.app/health

# Check readiness
curl https://your-app.railway.app/ready
```

### Logs
- Railway: View logs in dashboard
- Render: View logs in dashboard
- Local: Check `/app/logs/app.log`

### File Cleanup
The system automatically cleans up old files:
- Uploaded images: Removed after processing
- Generated reports: Cleaned up after 24 hours
- Logs: Rotated automatically

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```bash
# Check if model file exists
ls -la models/simclr_unet_patch_wound.keras

# Check model size (should be ~500MB)
du -h models/simclr_unet_patch_wound.keras
```

#### 2. Memory Issues
- Reduce batch size in model loading
- Use model quantization
- Implement model caching

#### 3. CORS Errors
- Check CORS_ORIGINS environment variable
- Ensure frontend URL is included

#### 4. Rate Limiting
- Adjust RATE_LIMIT_PER_MINUTE
- Implement user-based rate limiting

### Debug Commands
```bash
# Test local deployment
python app_production.py

# Test with specific model path
SIMCLR_MODEL_PATH=models/simclr_unet_patch_wound.keras python app_production.py

# Run deployment tests
python test_deployment.py --url http://localhost:8080
```

## Performance Optimization

### Model Optimization
1. **Quantization**: Reduce model precision
2. **Pruning**: Remove unnecessary weights
3. **Caching**: Cache model predictions
4. **Batch Processing**: Process multiple images

### API Optimization
1. **Async Processing**: Use background tasks
2. **Caching**: Cache frequent requests
3. **Compression**: Compress responses
4. **CDN**: Use CDN for static files

## Security Considerations

### API Security
- Rate limiting implemented
- File type validation
- Secure filename handling
- CORS configuration

### Data Privacy
- Images processed in memory
- No persistent storage of user data
- Automatic cleanup of temporary files

## Scaling Strategy

### Free Tier → Paid Tier
1. **Monitor Usage**: Track API calls and resource usage
2. **Upgrade When Needed**: Move to paid plans when limits reached
3. **Load Balancing**: Use multiple free instances
4. **CDN**: Implement CDN for better performance

### Enterprise Features
- User authentication
- Database integration
- Advanced analytics
- Multi-tenant support

## Cost Analysis

### Free Tier Costs
- **Railway**: $0/month (500 hours)
- **Render**: $0/month (750 hours)
- **Domain**: $0 (using subdomains)
- **Total**: $0/month

### Paid Tier Costs (When Needed)
- **Railway Pro**: $5/month
- **Render Starter**: $7/month
- **Custom Domain**: $10-15/year
- **Total**: $5-7/month

## Success Metrics

### Performance Targets
- **Response Time**: < 5 seconds
- **Uptime**: > 99%
- **Concurrent Users**: 5-10 (free tier)
- **Model Accuracy**: > 90% (SimCLR)

### Monitoring
- Health check endpoints
- Response time monitoring
- Error rate tracking
- User usage analytics

## Next Steps

### Phase 3b: Paid Upgrades
- Custom domain
- User authentication
- Database integration
- Advanced features

### Phase 4: Enterprise
- Multi-tenant architecture
- Advanced security
- Compliance features
- Professional support

## Support and Resources

### Documentation
- [Railway Docs](https://docs.railway.app)
- [Render Docs](https://render.com/docs)
- [Flask Docs](https://flask.palletsprojects.com)

### Community
- GitHub Issues
- Stack Overflow
- Discord/Slack channels

### Professional Support
- Custom development
- Enterprise deployment
- Training and consulting

---

**Ready to deploy?** Follow the Quick Start guide above and you'll have a fully functional wound detection website running for free!