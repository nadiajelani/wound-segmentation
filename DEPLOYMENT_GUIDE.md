# 🌐 Public Wound Detection Website Deployment Guide

## Quick Start (Railway - Recommended)

### 1. Prepare Your Code
```bash
# Ensure all files are ready
ls -la Dockerfile app_production.py wound_whisperer.html
```

### 2. Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your wound-segmentation repository
5. Railway will automatically detect the Dockerfile and deploy

### 3. Configure Environment Variables
In Railway dashboard, go to Variables tab and add:
```
SECRET_KEY=your-super-secret-key-here
CORS_ORIGINS=https://your-app-name.railway.app
UNET_MODEL_PATH=/app/models/checkpoints/unet_weights.h5
MEDSAM_MODEL_PATH=/app/models/checkpoints/medsam_model.pth
```

### 4. Upload Your Models
- Upload your trained models to the `/models/checkpoints/` directory
- Or use Railway's volume storage for persistent model files

### 5. Test Your Deployment
- Visit your Railway URL (e.g., `https://your-app-name.railway.app`)
- Upload a test wound image
- Verify analysis results

## Alternative: DigitalOcean App Platform

### 1. Create App Spec
Create `app.yaml`:
```yaml
name: wound-detection
services:
- name: api
  source_dir: /
  github:
    repo: your-username/wound-segmentation
    branch: main
  run_command: gunicorn --bind 0.0.0.0:$PORT --workers 2 app_production:app
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  routes:
  - path: /
  envs:
  - key: SECRET_KEY
    value: your-secret-key
  - key: CORS_ORIGINS
    value: https://your-app-name.ondigitalocean.app
```

### 2. Deploy
```bash
doctl apps create --spec app.yaml
```

## Alternative: Render

### 1. Connect GitHub Repository
1. Go to [render.com](https://render.com)
2. Connect your GitHub account
3. Select your wound-segmentation repository

### 2. Configure Web Service
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app_production:app`
- **Environment**: Python 3.9

### 3. Add Environment Variables
```
SECRET_KEY=your-secret-key
CORS_ORIGINS=https://your-app-name.onrender.com
```

## Security Checklist

### ✅ Before Going Live
- [ ] Change default SECRET_KEY
- [ ] Set up proper CORS origins
- [ ] Configure rate limiting
- [ ] Add file size limits
- [ ] Set up monitoring/logging
- [ ] Test with real wound images
- [ ] Verify PDF generation works
- [ ] Check file cleanup is working

### ✅ Production Monitoring
- [ ] Set up health checks
- [ ] Monitor disk space usage
- [ ] Track API response times
- [ ] Set up error alerting
- [ ] Monitor model performance

## Cost Estimates

| Platform | Monthly Cost | Features |
|----------|-------------|----------|
| Railway | $5-20 | Easy setup, auto-scaling |
| DigitalOcean | $10-30 | More control, custom domains |
| Render | $7-25 | Good for static + API |
| VPS (Self-hosted) | $5-50 | Full control, more setup |

## Custom Domain Setup

### 1. Buy a Domain
- Use Namecheap, GoDaddy, or Google Domains
- Choose something like `woundai.com` or `wounddetector.app`

### 2. Configure DNS
- Point your domain to your hosting platform
- Add SSL certificate (usually automatic)

### 3. Update CORS Settings
Update your environment variables:
```
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

## Scaling Considerations

### For 100+ Users/Day
- Use container orchestration (Kubernetes)
- Add Redis for caching
- Implement queue system for heavy processing
- Consider GPU instances for faster inference

### For 1000+ Users/Day
- Load balancer with multiple instances
- CDN for static assets
- Database for user sessions
- Advanced monitoring and alerting

## Troubleshooting

### Common Issues
1. **Models not loading**: Check file paths and permissions
2. **CORS errors**: Verify CORS_ORIGINS setting
3. **Out of memory**: Increase instance size or optimize models
4. **Slow responses**: Add caching or use GPU instances

### Debug Commands
```bash
# Check if models are loaded
curl https://your-app.com/health

# Test file upload
curl -X POST -F "image=@test_wound.jpg" https://your-app.com/upload
```

## Next Steps After Deployment

1. **Add Analytics**: Track usage and performance
2. **User Feedback**: Add rating system for predictions
3. **Mobile App**: Create React Native or Flutter app
4. **API Documentation**: Add Swagger/OpenAPI docs
5. **Advanced Features**: Batch processing, user accounts

## Support

- Check logs: `railway logs` or platform-specific logging
- Monitor performance: Use platform dashboards
- Scale up: Increase instance size if needed
- Optimize: Profile model inference times