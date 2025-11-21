#!/usr/bin/env python3
"""
Stable Diffusion 1.5 with OpenVINO for Intel GPU
Optimized for Intel Panther Lake iGPU

This script implements text-to-image generation using Stable Diffusion 1.5
with OpenVINO optimization for Intel integrated graphics.
"""

import os
import argparse
import time
import warnings
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import openvino as ov
    from optimum.intel.openvino import OVStableDiffusionPipeline
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
    from transformers import CLIPTokenizer
except ImportError as e:
    print(f"Required packages not found: {e}")
    print("Please install: pip install openvino optimum[openvino] diffusers transformers torch pillow")
    exit(1)


class OpenVINOStableDiffusion:
    """
    Stable Diffusion 1.5 with OpenVINO optimization for Intel GPU
    """
    
    def __init__(
        self, 
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "GPU",
        cache_dir: str = "./models",
        compile_model: bool = True
    ):
        """
        Initialize the OpenVINO Stable Diffusion pipeline
        
        Args:
            model_id: HuggingFace model identifier
            device: OpenVINO device (GPU, CPU, AUTO)
            cache_dir: Directory to cache models
            compile_model: Whether to compile model for optimization
        """
        self.model_id = model_id
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.compile_model = compile_model
        
        print(f"Initializing Stable Diffusion with OpenVINO on {device}")
        self.pipeline = None
        self.core = ov.Core()
        
        # Check available devices
        self._check_devices()
        self._load_pipeline()
    
    def _check_devices(self):
        """Check available OpenVINO devices"""
        # Print OpenVINO version
        try:
            openvino_version = ov.get_version()
            print(f"✓ OpenVINO Version: {openvino_version}")
        except:
            try:
                import openvino
                print(f"✓ OpenVINO Version: {openvino.__version__}")
            except:
                print("⚠️  Could not determine OpenVINO version")
        
        available_devices = self.core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")
        
        if "GPU" in available_devices:
            try:
                gpu_name = self.core.get_property("GPU", "FULL_DEVICE_NAME")
                print(f"✓ Intel GPU detected: {gpu_name}")
            except:
                print("✓ Intel GPU detected and available")
        else:
            print("⚠ GPU not available, falling back to CPU")
            self.device = "CPU"
    
    def _load_pipeline(self):
        """Load and optimize the Stable Diffusion pipeline"""
        try:
            print("Loading Stable Diffusion 1.5 model...")
            
            # Check if OpenVINO model already exists
            ov_model_path = self.cache_dir / "stable_diffusion_ov"
            
            if ov_model_path.exists() and (ov_model_path / "openvino_model.xml").exists():
                print(f"Loading cached OpenVINO model from {ov_model_path}")
                self.pipeline = OVStableDiffusionPipeline.from_pretrained(
                    ov_model_path,
                    device=self.device,
                    compile=self.compile_model
                )
            else:
                print("Converting PyTorch model to OpenVINO format...")
                
                # Load original PyTorch model
                torch_pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    cache_dir=self.cache_dir
                )
                
                # Convert to OpenVINO format
                self.pipeline = OVStableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    export=True,
                    device=self.device,
                    compile=self.compile_model,
                    cache_dir=self.cache_dir
                )
                
                # Save the OpenVINO model
                self.pipeline.save_pretrained(ov_model_path)
                print(f"OpenVINO model saved to {ov_model_path}")
                
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            print("Attempting fallback loading...")
            
            # Fallback: Load without optimization
            self.pipeline = OVStableDiffusionPipeline.from_pretrained(
                self.model_id,
                device="CPU",
                compile=False,
                cache_dir=self.cache_dir
            )
        
        print("Pipeline loaded successfully!")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        num_images: int = 1
    ) -> List[Image.Image]:
        """
        Generate images from text prompt
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Text description of what to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
            num_images: Number of images to generate
            
        Returns:
            List of PIL Images
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        
        # Set default negative prompt if none provided
        if negative_prompt is None:
            negative_prompt = "blurry, bad quality, distorted, deformed"
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        print(f"Generating {num_images} image(s) with prompt: '{prompt}'")
        print(f"Settings: steps={num_inference_steps}, guidance={guidance_scale}, size={width}x{height}")
        
        start_time = time.time()
        
        try:
            # Generate image(s)
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=num_images
            )
            
            images = result.images
            generation_time = time.time() - start_time
            
            print(f"✓ Generation completed in {generation_time:.2f} seconds")
            print(f"✓ Generated {len(images)} image(s)")
            
            return images
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return []
    
    def save_images(self, images: List[Image.Image], output_dir: str = "./outputs", prefix: str = "generated"):
        """Save generated images to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        for i, image in enumerate(images):
            filename = f"{prefix}_{int(time.time())}_{i:03d}.png"
            filepath = output_path / filename
            image.save(filepath, "PNG")
            saved_files.append(filepath)
            print(f"✓ Saved: {filepath}")
        
        return saved_files


def main():
    """Main function to run the Stable Diffusion generation"""
    parser = argparse.ArgumentParser(description="Stable Diffusion 1.5 with OpenVINO")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative", type=str, default=None, help="Negative prompt")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--device", type=str, default="GPU", help="OpenVINO device (GPU, CPU, AUTO)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--model_cache", type=str, default="./models", help="Model cache directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize the pipeline
        sd_generator = OpenVINOStableDiffusion(
            device=args.device,
            cache_dir=args.model_cache
        )
        
        # Generate images
        images = sd_generator.generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
            num_images=args.num_images
        )
        
        if images:
            # Save images
            saved_files = sd_generator.save_images(images, args.output_dir)
            print(f"\n✓ Successfully generated and saved {len(saved_files)} image(s)")
            print(f"Output directory: {Path(args.output_dir).absolute()}")
        else:
            print("❌ Failed to generate images")
            
    except KeyboardInterrupt:
        print("\n⚠ Generation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python stable_diffusion_openvino.py --prompt 'A beautiful sunset over mountains' --steps 25 --guidance 8.0")
        print("\nRunning with default example...")
        
        # Run with example prompt
        sd = OpenVINOStableDiffusion()
        images = sd.generate_image(
            prompt="A serene lake surrounded by mountains at sunset, digital art",
            num_inference_steps=20,
            guidance_scale=7.5
        )
        if images:
            sd.save_images(images)
    else:
        main()
