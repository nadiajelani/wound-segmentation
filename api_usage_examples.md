# Wound Segmentation API Usage Examples

## Quick Start Commands

### 1. Start the Servers
```bash
# Terminal 1 - Model Server
./start_model_server.sh

# Terminal 2 - Proxy API  
./start_proxy_api.sh
```

### 2. Check API Status
```bash
# Check if both servers are ready
curl -s http://127.0.0.1:9100/readyz  # Model server
curl -s http://127.0.0.1:8000/readyz  # Proxy API
```

### 3. Analyze an Image

#### Using curl with base64 image:
```bash
# Convert image to base64 first
IMG_B64=$(base64 -i your_image.jpg)

# Send analysis request
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"$IMG_B64\", \"threshold\": 0.5}"
```

#### Using file upload:
```bash
curl -X POST http://127.0.0.1:8000/analyze/upload \
  -F "file=@your_image.jpg" \
  -F "threshold=0.5"
```

### 4. Get Analysis Report
```bash
# Get report by ID (from analysis response)
curl http://127.0.0.1:8000/report/1759048241992
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/readyz` | Check API readiness |
| POST | `/analyze` | Analyze base64 image |
| POST | `/analyze/upload` | Upload and analyze file |
| GET | `/report/{id}` | Get analysis report |

## Response Format

```json
{
  "id": "1759048241992",
  "backend": "simclr_unet", 
  "result": {
    "mask_area_px": 2052,
    "wound_percentage": 0.1252,
    "perimeter_px": 180.5,
    "severity": "Severe",
    "healing_potential": "Poor",
    "mask_uri": "data:image/png;base64,..."
  },
  "mode": "real"
}
```

## Troubleshooting

### Check Model Server Diagnostics:
```bash
curl -s http://127.0.0.1:9100/diagz | python -m json.tool
curl -s http://127.0.0.1:9100/last_errorz
```

### Stop Servers:
```bash
pkill -f uvicorn
```

### Restart Everything:
```bash
pkill -f uvicorn
./start_model_server.sh &
./start_proxy_api.sh &
```