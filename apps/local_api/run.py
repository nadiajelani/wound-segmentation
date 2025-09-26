#!/usr/bin/env python3
"""
Simple script to run the Wound Whisperer Local API
"""
import uvicorn
from apps.local_api.main import app

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )