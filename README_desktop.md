# Wound Whisperer Desktop Client

A desktop GUI application that calls the local FastAPI backend for wound segmentation analysis.

## Features

- **Image Selection**: Browse and select wound images
- **Real-time Analysis**: Calls the local API for wound segmentation
- **Results Display**: Shows severity, healing potential, area, percentage, and perimeter
- **Mask Preview**: Displays the segmentation mask as a preview image
- **Status Monitoring**: Shows API connection status and backend information

## Requirements

- Python 3.8+
- tkinter (usually included with Python)
- PIL/Pillow
- requests

## Installation

1. Install dependencies:
```bash
pip install pillow requests
```

2. Make sure your local API is running (see below)

## Usage

### 1. Start the Local API

```bash
# Option 1: Using the run script
python apps/local_api/run.py

# Option 2: Using uvicorn directly
uvicorn apps.local_api.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`

### 2. Start the Desktop Client

```bash
python desktop_client.py
```

### 3. Use the Application

1. **Select Image**: Click "Browse..." to select a wound image
2. **Analyze**: Click "Analyze" to process the image
3. **View Results**: See the analysis results including:
   - Severity assessment
   - Healing potential
   - Wound area in pixels
   - Wound percentage
   - Perimeter measurement
   - Mask preview image

## API Endpoints Used

The desktop client calls these endpoints:

- `GET /readyz` - Check if API is ready and get backend info
- `POST /analyze` - Analyze an image (sends base64 encoded image)
- `GET /report/{id}` - Get stored analysis results

## Technical Details

- **Backend**: SimCLR-UNet model via TensorFlow/Keras
- **Image Processing**: 128x128 input size, 0.5 threshold for binary mask
- **Threading**: Analysis runs in background thread to keep UI responsive
- **Error Handling**: Graceful error handling with user-friendly messages

## Troubleshooting

### API Not Ready
- Make sure the local API is running on port 8000
- Check that the SimCLR-UNet model file exists at the configured path
- Verify the API health at `http://127.0.0.1:8000/healthz`

### Analysis Fails
- Check that the selected image is a valid image file
- Ensure the image is not too large (recommended < 10MB)
- Verify the API is responding to requests

### Preview Not Showing
- The mask preview requires a valid base64 data URI from the API
- Check that the API is returning a `mask_uri` field in the response

## Development

The desktop client is designed to work with the existing FastAPI backend without modifications. It uses the same API contract as the web interface, ensuring consistency across all client applications.