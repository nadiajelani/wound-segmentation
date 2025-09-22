"""
Synthetic wound data generation using Stable Diffusion.

This module provides functionality to generate synthetic wound images for training
data augmentation. It's controlled by the ENABLE_SYNTHETIC_DATA feature flag.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from ..config import Config
from ..logging import get_logger

logger = get_logger(__name__)

class SyntheticWoundGenerator:
    """
    Generates synthetic wound images using Stable Diffusion.
    
    This class provides methods to generate diverse wound images for training
    data augmentation when the ENABLE_SYNTHETIC_DATA feature flag is enabled.
    """
    
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-2-1-base"):
        """
        Initialize the synthetic wound generator.
        
        Args:
            model_id: Hugging Face model ID for Stable Diffusion
        """
        if not Config.ENABLE_SYNTHETIC_DATA:
            logger.warning("Synthetic data generation is disabled. Set ENABLE_SYNTHETIC_DATA=true to enable.")
            self.pipeline = None
            return
            
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing synthetic wound generator with model: {model_id}")
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the Stable Diffusion pipeline."""
        try:
            # Load the pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Use DPMSolver for faster inference
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            logger.info(f"Synthetic wound generator loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load synthetic wound generator: {e}")
            self.pipeline = None
    
    def generate_wound_prompts(self, num_prompts: int = 10) -> List[str]:
        """
        Generate diverse wound description prompts.
        
        Args:
            num_prompts: Number of prompts to generate
            
        Returns:
            List of wound description prompts
        """
        base_prompts = [
            "medical photograph of a deep surgical wound on skin, high quality, clinical lighting",
            "close-up of a healing wound with red inflamed tissue, medical photography",
            "surgical incision wound with stitches, professional medical image",
            "chronic wound with granulation tissue, clinical photography",
            "burn wound on skin, medical documentation, high resolution",
            "diabetic foot ulcer, medical photograph, clinical setting",
            "pressure sore wound, medical imaging, professional quality",
            "traumatic wound with exposed tissue, clinical photography",
            "infected wound with pus and inflammation, medical documentation",
            "healing wound with scab formation, medical photography"
        ]
        
        # Add variations
        variations = [
            "with good lighting",
            "in clinical setting",
            "high resolution medical image",
            "professional medical photography",
            "clinical documentation quality",
            "with proper medical lighting",
            "detailed medical photograph",
            "clinical wound assessment image"
        ]
        
        import random
        prompts = []
        for _ in range(num_prompts):
            base = random.choice(base_prompts)
            variation = random.choice(variations)
            prompt = f"{base}, {variation}"
            prompts.append(prompt)
        
        return prompts
    
    def generate_synthetic_wounds(
        self, 
        num_images: int = 5,
        output_dir: Optional[Path] = None,
        image_size: Tuple[int, int] = (256, 256)  # Smaller size for faster generation
    ) -> List[Path]:
        """
        Generate synthetic wound images.
        
        Args:
            num_images: Number of images to generate
            output_dir: Directory to save generated images
            image_size: Size of generated images (width, height)
            
        Returns:
            List of paths to generated images
        """
        if not Config.ENABLE_SYNTHETIC_DATA:
            logger.warning("Synthetic data generation is disabled")
            return []
        
        if self.pipeline is None:
            logger.error("Synthetic wound generator not initialized")
            return []
        
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR / "synthetic_wounds"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_images} synthetic wound images...")
        
        # Generate prompts
        prompts = self.generate_wound_prompts(num_images)
        
        generated_paths = []
        
        for i, prompt in enumerate(prompts):
            try:
                logger.info(f"Generating image {i+1}/{num_images}: {prompt[:50]}...")
                
                # Generate image
                with torch.autocast(self.device):
                    result = self.pipeline(
                        prompt=prompt,
                        height=image_size[1],
                        width=image_size[0],
                        num_inference_steps=10,  # Much faster generation
                        guidance_scale=7.5,
                        generator=torch.Generator(device=self.device).manual_seed(42 + i)
                    )
                
                # Save image
                image = result.images[0]
                output_path = output_dir / f"synthetic_wound_{i+1:03d}.png"
                image.save(output_path)
                generated_paths.append(output_path)
                
                logger.info(f"Generated: {output_path}")
                
            except Exception as e:
                logger.error(f"Failed to generate image {i+1}: {e}")
                continue
        
        logger.info(f"Generated {len(generated_paths)} synthetic wound images")
        return generated_paths
    
    def generate_training_batch(
        self,
        batch_size: int = 10,
        output_dir: Optional[Path] = None
    ) -> Dict[str, List[Path]]:
        """
        Generate a batch of synthetic images for training.
        
        Args:
            batch_size: Number of images to generate
            output_dir: Directory to save generated images
            
        Returns:
            Dictionary with 'images' and 'metadata' keys
        """
        if not Config.ENABLE_SYNTHETIC_DATA:
            logger.warning("Synthetic data generation is disabled")
            return {"images": [], "metadata": []}
        
        if output_dir is None:
            output_dir = Config.OUTPUT_DIR / "synthetic_training_batch"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images
        image_paths = self.generate_synthetic_wounds(
            num_images=batch_size,
            output_dir=output_dir / "images"
        )
        
        # Create metadata
        metadata = []
        for i, path in enumerate(image_paths):
            metadata.append({
                "image_path": str(path),
                "synthetic": True,
                "generation_method": "stable_diffusion",
                "model_id": self.model_id,
                "batch_id": f"batch_{i//batch_size}",
                "image_id": f"synthetic_{i:03d}"
            })
        
        # Save metadata
        import json
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated training batch with {len(image_paths)} images")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return {
            "images": image_paths,
            "metadata": metadata,
            "metadata_path": metadata_path
        }
    
    def is_available(self) -> bool:
        """
        Check if synthetic data generation is available.
        
        Returns:
            True if synthetic generation is enabled and pipeline is loaded
        """
        return (
            Config.ENABLE_SYNTHETIC_DATA and 
            self.pipeline is not None
        )


def create_synthetic_training_data(
    num_images: int = 20,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to create synthetic training data.
    
    Args:
        num_images: Number of synthetic images to generate
        output_dir: Directory to save generated data
        
    Returns:
        Dictionary with generation results
    """
    if not Config.ENABLE_SYNTHETIC_DATA:
        logger.warning("Synthetic data generation is disabled. Set ENABLE_SYNTHETIC_DATA=true to enable.")
        return {"success": False, "reason": "Feature disabled"}
    
    generator = SyntheticWoundGenerator()
    
    if not generator.is_available():
        logger.error("Synthetic wound generator not available")
        return {"success": False, "reason": "Generator not available"}
    
    try:
        result = generator.generate_training_batch(
            batch_size=num_images,
            output_dir=output_dir
        )
        
        return {
            "success": True,
            "images_generated": len(result["images"]),
            "output_dir": str(output_dir) if output_dir else str(Config.OUTPUT_DIR / "synthetic_training_batch"),
            "metadata_path": str(result["metadata_path"]),
            "images": [str(p) for p in result["images"]]
        }
        
    except Exception as e:
        logger.error(f"Failed to create synthetic training data: {e}")
        return {"success": False, "reason": str(e)}


# CLI integration function
def generate_synthetic_data_cli(num_images: int = 10) -> None:
    """
    CLI function to generate synthetic wound data.
    
    Args:
        num_images: Number of images to generate
    """
    if not Config.ENABLE_SYNTHETIC_DATA:
        print("❌ Synthetic data generation is disabled.")
        print("   Set ENABLE_SYNTHETIC_DATA=true in your environment to enable.")
        return
    
    print(f"🎨 Generating {num_images} synthetic wound images...")
    
    result = create_synthetic_training_data(num_images=num_images)
    
    if result["success"]:
        print(f"✅ Successfully generated {result['images_generated']} synthetic images")
        print(f"📁 Output directory: {result['output_dir']}")
        print(f"📋 Metadata saved to: {result['metadata_path']}")
    else:
        print(f"❌ Failed to generate synthetic data: {result['reason']}")


if __name__ == "__main__":
    # Test the synthetic data generation
    generate_synthetic_data_cli(num_images=5)