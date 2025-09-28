#!/usr/bin/env python3
"""
Convert Keras 3.x model to TensorFlow 2.12 compatible format
"""
import os
import sys
import tempfile
import shutil

def convert_model():
    """Convert the Keras 3.x model to TF 2.12 compatible format"""
    
    # Input and output paths
    input_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound.keras"
    output_path = "/Users/nadiajelani/projects/wound-segmentation/models/simclr_unet_patch_wound_converted.h5"
    
    print(f"🔄 Converting model from {input_path} to {output_path}")
    
    try:
        # Try to load with standalone Keras 3.x
        import keras
        print(f"📦 Using Keras version: {keras.__version__}")
        
        # Load the model
        print("🔍 Loading model...")
        model = keras.models.load_model(input_path, compile=False)
        print("✅ Model loaded successfully")
        
        # Print model info
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        
        # Save in HDF5 format (compatible with TF 2.12)
        print("💾 Saving model in HDF5 format...")
        model.save(output_path, save_format="h5")
        print(f"✅ Model saved to {output_path}")
        
        # Verify the converted model
        print("🔍 Verifying converted model...")
        converted_model = keras.models.load_model(output_path, compile=False)
        print("✅ Converted model loads successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_model()
    sys.exit(0 if success else 1)