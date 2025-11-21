#!/usr/bin/env python3
"""
Enhanced Stable Diffusion with OpenVINO GenAI API
Based on OpenVINO Notebooks implementation with additional features and performance metrics

This script implements both text-to-image and image-to-image generation 
using OpenVINO GenAI API for optimal performance on Intel hardware.
"""
        except:
            print("⚠️  Could not determine OpenVINO GenAI version")
        
        # OpenVINO devices
        available_devices = self.core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")
        
        gpu_detected = False
        if "GPU" in available_devices:
            try:
                gpu_name = self.core.get_property("GPU", "FULL_DEVICE_NAME")
                gpu_memory = self.core.get_property("GPU", "GPU_DEVICE_TOTAL_MEM_SIZE")
                print(f"✓ GPU: {gpu_name}")
                print(f"✓ GPU Total Memory: {gpu_memory / (1024**3):.1f} GB")
                
                try:
                    device_id = self.core.get_property("GPU", "GPU_DEVICE_ID")
                    print(f"✓ GPU Device ID: 0x{device_id:04x}")
                except:
                    pass
                
                gpu_detected = True
                self.performance_stats["system_info"]["gpu_name"] = gpu_name
                self.performance_stats["system_info"]["gpu_memory_gb"] = gpu_memory / (1024**3)
                
            except Exception as e:
                print(f"⚠️  GPU detected but couldn't get details: {e}")
        
        # NPU information
        if "NPU" in available_devices:
            print("✓ NPU (Neural Processing Unit) detected")
            try:
                npu_name = self.core.get_property("NPU", "FULL_DEVICE_NAME")
                print(f"✓ NPU: {npu_name}")
                self.performance_stats["system_info"]["npu_name"] = npu_name
            except:
                print("✓ NPU available but details not accessible")
        
        # Device fallback logic
        if self.device == "NPU" and "NPU" not in available_devices:
            print("❌ NPU requested but not available, falling back to GPU")
            self.device = "GPU"
        elif self.device == "GPU" and not gpu_detected:
            print("❌ GPU requested but not available, falling back to CPU")
            self.device = "CPU"
        
        # System memory
        memory = psutil.virtual_memory()
        print(f"✓ System RAM: {memory.total / (1024**3):.1f} GB")
        print(f"✓ Available RAM: {memory.available / (1024**3):.1f} GB")
        
        # CPU info
        print(f"✓ CPU Cores: {psutil.cpu_count()} physical, {psutil.cpu_count(logical=False)} logical")
        
        # Store system info
        self.performance_stats["system_info"].update({
            "ram_total_gb": memory.total / (1024**3),
            "ram_available_gb": memory.available / (1024**3),
            "cpu_cores": psutil.cpu_count(),
            "device": self.device,
            "available_devices": available_devices
        })implementation with additional features

This script implements both text-to-image and image-to-image generation 
using OpenVINO GenAI API for optimal performance on Intel hardware.
"""

import os
import argparse
import time
import statistics
import warnings
from pathlib import Path
from typing import Optional, Union, List, Dict
import gc
import json
import psutil

import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import openvino as ov
    import openvino_genai as ov_genai
    from tqdm import tqdm
    import sys
except ImportError as e:
    print(f"Required packages not found: {e}")
    print("Please install: pip install openvino openvino_genai openvino_tokenizers tqdm pillow psutil")
    exit(1)


class OpenVINOGenAIStableDiffusion:
    """
    Enhanced Stable Diffusion with OpenVINO GenAI API
    Supports both text-to-image and image-to-image generation
    """
    
    def __init__(
        self, 
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "GPU",
        cache_dir: str = "./models",
        model_precision: str = "fp16"
    ):
        """
        Initialize the OpenVINO GenAI Stable Diffusion pipeline
        
        Args:
            model_id: HuggingFace model identifier or local model path
            device: OpenVINO device (GPU, CPU, AUTO)
            cache_dir: Directory to cache models
            model_precision: Model precision (fp16, fp32, int8, int4)
        """
        self.model_id = model_id
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.model_precision = model_precision
        
        # Performance tracking
        self.performance_stats = {
            "init_time": 0,
            "generations": [],
            "system_info": {},
            "version_info": {}
        }
        
        print(f"🚀 Initializing Enhanced Stable Diffusion with OpenVINO GenAI on {device}")
        print(f"Model: {model_id}")
        print(f"Device: {device}")
        print(f"Model Precision: {model_precision}")
        
        self.text2img_pipe = None
        self.img2img_pipe = None
        self.core = ov.Core()
        
        # Check available devices and get system info
        self._check_system_and_devices()
        
        # Load pipelines and measure init time
        init_start = time.time()
        self._prepare_model()
        self.performance_stats["init_time"] = time.time() - init_start
    
    def _check_system_and_devices(self):
        """Check available OpenVINO devices and gather system information"""
        available_devices = self.core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")
        
        if "GPU" in available_devices:
            print("✓ Intel GPU detected and available")
            try:
                gpu_name = self.core.get_property("GPU", "FULL_DEVICE_NAME")
                print(f"GPU: {gpu_name}")
            except:
                print("GPU detected but couldn't get detailed info")
        else:
            print("⚠ GPU not available, falling back to CPU")
            self.device = "CPU"
        
        # System information
        try:
            self.performance_stats["system_info"] = {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq().current,
                "memory": dict(psutil.virtual_memory()._asdict()),
                "swap": dict(psutil.swap_memory()._asdict()),
                "disk": dict(psutil.disk_usage('/')._asdict()),
            }
            
            print("System Information:")
            for key, value in self.performance_stats["system_info"].items():
                print(f"  {key}: {value}")
        
        except Exception as e:
            print(f"Error gathering system info: {e}")
    
    def _prepare_model(self):
        """Prepare OpenVINO model using optimum-cli if needed"""
        # Model directory based on model ID
        model_name = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
        model_dir = self.cache_dir / f"diffusion_pipeline_{model_name}"
        
        if not model_dir.exists():
            print(f"Converting model {self.model_id} to OpenVINO format...")
            self._convert_model_with_optimum_cli(self.model_id, model_dir)
        else:
            print(f"Using cached model from {model_dir}")
        
        self.model_dir = model_dir
        self._load_pipelines()
    
    def _convert_model_with_optimum_cli(self, model_id: str, output_dir: Path):
        """Convert model using optimum-cli"""
        try:
            import subprocess
            
            cmd = [
                "optimum-cli", "export", "openvino",
                "--model", model_id,
                "--task", "text-to-image",
                "--weight-format", self.model_precision,
                str(output_dir)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Model conversion completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Model conversion failed: {e}")
            print(f"Error output: {e.stderr}")
            raise
        except ImportError:
            print("❌ optimum-cli not found. Installing optimum[openvino]...")
            subprocess.run([sys.executable, "-m", "pip", "install", "optimum[openvino]"], check=True)
            # Retry conversion
            self._convert_model_with_optimum_cli(model_id, output_dir)
    
    def _load_pipelines(self):
        """Load Text2Image and Image2Image pipelines"""
        try:
            print("Loading Text2Image pipeline...")
            self.text2img_pipe = ov_genai.Text2ImagePipeline(str(self.model_dir), self.device)
            print("✅ Text2Image pipeline loaded")
            
            print("Loading Image2Image pipeline...")
            self.img2img_pipe = ov_genai.Image2ImagePipeline(str(self.model_dir), self.device)
            print("✅ Image2Image pipeline loaded")
            
        except Exception as e:
            print(f"❌ Error loading pipelines: {e}")
            raise
    
    def generate_text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        num_images: int = 1,
        show_progress: bool = True
    ) -> List[Image.Image]:
        """
        Generate images from text prompt using Text2Image pipeline
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Text description of what to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            width: Image width
            height: Image height
            seed: Random seed for reproducibility
            num_images: Number of images to generate
            show_progress: Show progress bar
            
        Returns:
            List of PIL Images
        """
        if self.text2img_pipe is None:
            raise RuntimeError("Text2Image pipeline not initialized")
        
        print(f"Generating {num_images} image(s) with prompt: '{prompt}'")
        print(f"Settings: steps={num_inference_steps}, guidance={guidance_scale}, size={width}x{height}")
        
        start_time = time.time()
        
        try:
            # Setup generator with seed
            generator = ov_genai.TorchGenerator(seed if seed is not None else int(time.time()))
            
            # Progress bar setup
            if show_progress:
                pbar = tqdm(total=num_inference_steps)
                
                def callback(step, num_steps, latent):
                    if num_steps != pbar.total:
                        pbar.reset(num_steps)
                    pbar.update(1)
                    sys.stdout.flush()
                    return False
            else:
                callback = None
            
            # Generate image(s) - Note: OpenVINO GenAI API differences
            result = self.text2img_pipe.generate(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=num_images,
                generator=generator,
                callback=callback
            )
            
            if show_progress:
                pbar.close()
            
            # Convert to PIL Images
            images = [Image.fromarray(img) for img in result.data]
            
            generation_time = time.time() - start_time
            print(f"✓ Generation completed in {generation_time:.2f} seconds")
            print(f"✓ Generated {len(images)} image(s)")
            
            # Performance tracking
            self.performance_stats["generations"].append({
                "prompt": prompt,
                "num_images": num_images,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "time": generation_time
            })
            
            return images
            
        except Exception as e:
            if show_progress and 'pbar' in locals():
                pbar.close()
            print(f"Error during generation: {e}")
            return []
    
    def generate_image_to_image(
        self,
        prompt: str,
        image: Union[str, Image.Image],
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        strength: float = 0.75,
        seed: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Image.Image]:
        """
        Generate images from text prompt and input image using Image2Image pipeline
        
        Args:
            prompt: Text description of the desired image
            image: Input image (PIL Image or file path)
            negative_prompt: Text description of what to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            strength: How much to transform the input image (0.0 to 1.0)
            seed: Random seed for reproducibility
            show_progress: Show progress bar
            
        Returns:
            List of PIL Images
        """
        if self.img2img_pipe is None:
            raise RuntimeError("Image2Image pipeline not initialized")
        
        # Load input image
        if isinstance(image, str):
            input_image = Image.open(image)
        else:
            input_image = image
        
        # Convert PIL Image to OpenVINO Tensor
        def image_to_tensor(img: Image.Image) -> ov.Tensor:
            pic = img.convert("RGB")
            image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
            return ov.Tensor(image_data)
        
        image_tensor = image_to_tensor(input_image)
        
        print(f"Generating image from prompt: '{prompt}'")
        print(f"Settings: steps={num_inference_steps}, guidance={guidance_scale}, strength={strength}")
        print(f"Input image size: {input_image.size}")
        
        start_time = time.time()
        
        try:
            # Setup generator with seed
            generator = ov_genai.TorchGenerator(seed if seed is not None else int(time.time()))
            
            # Progress bar setup
            if show_progress:
                effective_steps = int(num_inference_steps * strength) + 1
                pbar = tqdm(total=effective_steps)
                
                def callback(step, num_steps, latent):
                    if num_steps != pbar.total:
                        pbar.reset(num_steps)
                    pbar.update(1)
                    sys.stdout.flush()
                    return False
            else:
                callback = None
            
            # Generate image
            result = self.img2img_pipe.generate(
                prompt,
                image_tensor,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                callback=callback
            )
            
            if show_progress:
                pbar.close()
            
            # Convert to PIL Images
            images = [Image.fromarray(img) for img in result.data]
            
            generation_time = time.time() - start_time
            print(f"✓ Generation completed in {generation_time:.2f} seconds")
            print(f"✓ Generated {len(images)} image(s)")
            
            # Performance tracking
            self.performance_stats["generations"].append({
                "prompt": prompt,
                "num_images": len(images),
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "seed": seed,
                "time": generation_time
            })
            
            return images
            
        except Exception as e:
            if show_progress and 'pbar' in locals():
                pbar.close()
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
    
    def cleanup(self):
        """Clean up resources"""
        if self.text2img_pipe:
            del self.text2img_pipe
        if self.img2img_pipe:
            del self.img2img_pipe
        gc.collect()
        print("✅ Resources cleaned up")


def main():
    """Main function to run the Enhanced Stable Diffusion generation"""
    parser = argparse.ArgumentParser(description="Enhanced Stable Diffusion with OpenVINO GenAI")
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
    parser.add_argument("--model_id", type=str, default="prompthero/openjourney", help="HuggingFace model ID")
    parser.add_argument("--model_precision", type=str, default="fp16", choices=["fp16", "fp32", "int8", "int4"], help="Model precision")
    
    # Image-to-Image specific arguments
    parser.add_argument("--input_image", type=str, default=None, help="Input image for image-to-image generation")
    parser.add_argument("--strength", type=float, default=0.75, help="Strength for image-to-image (0.0-1.0)")
    parser.add_argument("--mode", type=str, default="text2img", choices=["text2img", "img2img"], help="Generation mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize the pipeline
        sd_generator = OpenVINOGenAIStableDiffusion(
            model_id=args.model_id,
            device=args.device,
            cache_dir=args.model_cache,
            model_precision=args.model_precision
        )
        
        # Generate images based on mode
        if args.mode == "text2img":
            images = sd_generator.generate_text_to_image(
                prompt=args.prompt,
                negative_prompt=args.negative,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.width,
                height=args.height,
                seed=args.seed,
                num_images=args.num_images
            )
        else:  # img2img
            if not args.input_image:
                print("❌ Error: --input_image required for image-to-image generation")
                return
            
            images = sd_generator.generate_image_to_image(
                prompt=args.prompt,
                image=args.input_image,
                negative_prompt=args.negative,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                strength=args.strength,
                seed=args.seed
            )
        
        if images:
            # Save images
            saved_files = sd_generator.save_images(images, args.output_dir)
            print(f"\n✓ Successfully generated and saved {len(saved_files)} image(s)")
            print(f"Output directory: {Path(args.output_dir).absolute()}")
        else:
            print("❌ Failed to generate images")
        
        # Cleanup
        sd_generator.cleanup()
            
    except KeyboardInterrupt:
        print("\n⚠ Generation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Example usage if run directly
    if len(sys.argv) == 1:
        print("Enhanced Stable Diffusion with OpenVINO GenAI")
        print("\nExample usage:")
        print("Text-to-Image:")
        print("python enhanced_stable_diffusion.py --prompt 'A beautiful sunset over mountains' --steps 25 --guidance 8.0")
        print("\nImage-to-Image:")
        print("python enhanced_stable_diffusion.py --mode img2img --prompt 'watercolor painting' --input_image input.jpg --strength 0.75")
        print("\nRunning with default example...")
        
        # Run with example prompt
        sd = OpenVINOGenAIStableDiffusion()
        images = sd.generate_text_to_image(
            prompt="A serene lake surrounded by mountains at sunset, digital art",
            num_inference_steps=20,
            guidance_scale=7.5
        )
        if images:
            sd.save_images(images)
        sd.cleanup()
    else:
        main()
