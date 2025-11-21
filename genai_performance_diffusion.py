#!/usr/bin/env python3
"""
Performance-Enhanced Stable Diffusion with OpenVINO GenAI API
Based on OpenVINO Notebooks implementation with comprehensive performance metrics

Target: 512x512, Euler sampler, 25 steps in 3-4 seconds on Intel GPU
Uses openvino_genai.Text2ImagePipeline following the official notebook example
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


class PerformanceOpenVINOGenAI:
    """
    Performance-optimized Stable Diffusion with OpenVINO GenAI and comprehensive metrics
    Following the OpenVINO notebook example for openvino_genai.Text2ImagePipeline
    """
    
    def __init__(
        self, 
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "GPU",
        cache_dir: str = "./models"
    ):
        """
        Initialize the OpenVINO GenAI Stable Diffusion pipeline
        
        Args:
            model_id: HuggingFace model identifier or local model path
            device: OpenVINO device (GPU, CPU, AUTO, NPU)
            cache_dir: Directory to cache models
        """
        self.model_id = model_id
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_stats = {
            "init_time": 0,
            "generations": [],
            "system_info": {},
            "version_info": {}
        }
        
        print(f"🚀 Performance-Enhanced Stable Diffusion with OpenVINO GenAI")
        print(f"Model: {model_id}")
        print(f"Device: {device}")
        
        self.text2img_pipe = None
        self.core = ov.Core()
        
        # Check system and load pipeline with timing
        init_start = time.time()
        self._check_system_and_devices()
        self._load_text2img_pipeline()
        self.performance_stats["init_time"] = time.time() - init_start
        
        print(f"✅ Pipeline initialized in {self.performance_stats['init_time']:.2f} seconds")
    
    def _check_system_and_devices(self):
        """Check system capabilities, OpenVINO devices, and gather performance info"""
        print(f"\n{'='*60}")
        print("SYSTEM INFORMATION")
        print(f"{'='*60}")
        
        # Get version information
        try:
            print(f"✓ OpenVINO Version: {ov.__version__}")
            self.performance_stats["version_info"]["openvino_version"] = ov.__version__
        except:
            print("⚠️  Could not determine OpenVINO version")
            
        try:
            print(f"✓ OpenVINO GenAI Version: {ov_genai.__version__}")
            self.performance_stats["version_info"]["openvino_genai_version"] = ov_genai.__version__
        except:
            print("⚠️  Could not determine OpenVINO GenAI version")
        
        # OpenVINO devices
        available_devices = self.core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")
        
        # GPU information
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
        npu_detected = False
        if "NPU" in available_devices:
            print("✓ NPU (Neural Processing Unit) detected")
            try:
                npu_name = self.core.get_property("NPU", "FULL_DEVICE_NAME")
                print(f"✓ NPU: {npu_name}")
                self.performance_stats["system_info"]["npu_name"] = npu_name
                npu_detected = True
            except:
                print("✓ NPU available but details not accessible")
                npu_detected = True
        
        # Device fallback logic
        original_device = self.device
        if self.device == "NPU" and not npu_detected:
            print("❌ NPU requested but not available, falling back to GPU")
            self.device = "GPU"
        elif self.device == "GPU" and not gpu_detected:
            print("❌ GPU requested but not available, falling back to CPU")
            self.device = "CPU"
        
        if original_device != self.device:
            print(f"📋 Device changed from {original_device} to {self.device}")
        
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
        })
    
    def _load_text2img_pipeline(self):
        """Load Text2Image pipeline using OpenVINO GenAI"""
        print(f"\n{'='*60}")
        print("PIPELINE INITIALIZATION")
        print(f"{'='*60}")
        
        try:
            # Check for cached model directory
            model_name = self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
            model_dir = self.cache_dir / f"{model_name}_ov"
            
            if model_dir.exists() and (model_dir / "openvino_model.xml").exists():
                print(f"✓ Loading cached OpenVINO model from: {model_dir}")
                model_path = str(model_dir)
            else:
                print(f"🔄 Using HuggingFace model: {self.model_id}")
                model_path = self.model_id
            
            # Load Text2Image pipeline using OpenVINO GenAI
            print(f"🔄 Loading Text2ImagePipeline on {self.device}...")
            self.text2img_pipe = ov_genai.Text2ImagePipeline(model_path, self.device)
            
            print(f"✅ Text2ImagePipeline loaded successfully")
            print(f"✓ Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            
            # Try fallback to CPU if GPU/NPU failed
            if self.device != "CPU":
                print(f"🔄 Trying fallback to CPU...")
                try:
                    self.device = "CPU"
                    self.text2img_pipe = ov_genai.Text2ImagePipeline(model_path, "CPU")
                    print(f"✅ Pipeline loaded on CPU fallback")
                    self.performance_stats["system_info"]["device"] = "CPU"
                except Exception as e2:
                    print(f"❌ CPU fallback also failed: {e2}")
                    raise e2
            else:
                raise e
    
    def generate_with_metrics(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        save_image: bool = True,
        verbose: bool = False
    ) -> Dict:
        """Generate image with detailed performance metrics using OpenVINO GenAI"""
        
        if self.text2img_pipe is None:
            raise RuntimeError("Text2Image pipeline not initialized")
        
        if verbose:
            print(f"🎨 Generating: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
            print(f"Settings: {width}x{height}, {num_inference_steps} steps, guidance={guidance_scale}")
        
        # Memory before generation
        process = psutil.Process()
        process_memory_before = process.memory_info().rss / (1024**2)  # MB
        system_memory_before = psutil.virtual_memory()
        system_used_before_gb = (system_memory_before.total - system_memory_before.available) / (1024**3)
        
        if verbose:
            print(f"🔍 Memory Before: Process {process_memory_before:.0f} MB, System {system_used_before_gb:.1f} GB used")
        
        start_time = time.time()
        
        try:
            # Set up generator for reproducibility
            if seed is not None:
                generator = ov_genai.TorchGenerator(seed)
            else:
                generator = ov_genai.TorchGenerator(int(time.time()))
            
            # Generate image using OpenVINO GenAI
            result = self.text2img_pipe.generate(
                prompt,
                negative_prompt=negative_prompt or "",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images_per_prompt=1,
                generator=generator
            )
            
            total_time = time.time() - start_time
            
            # Convert result to PIL Images
            if hasattr(result, 'data'):
                images = [Image.fromarray(img) for img in result.data]
            else:
                # Fallback for different result formats
                images = [result] if isinstance(result, Image.Image) else []
            
            # Memory after generation
            process_memory_after = process.memory_info().rss / (1024**2)  # MB
            system_memory_after = psutil.virtual_memory()
            system_used_after_gb = (system_memory_after.total - system_memory_after.available) / (1024**3)
            
            # Calculate memory deltas
            process_delta = process_memory_after - process_memory_before
            system_delta_mb = (system_used_after_gb - system_used_before_gb) * 1024
            
            if verbose:
                print(f"🔍 Memory After:  Process {process_memory_after:.0f} MB, System {system_used_after_gb:.1f} GB used")
                print(f"📈 Memory Delta:  Process +{process_delta:.0f} MB, System +{system_delta_mb:.0f} MB (GPU)")
                print(f"⏱️  Generation completed in {total_time:.2f}s")
            
            # Save image if requested
            image_path = None
            if save_image and images:
                output_dir = Path("./outputs/performance_test")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = int(time.time())
                image_path = output_dir / f"genai_test_{timestamp}_{total_time:.2f}s.png"
                images[0].save(image_path)
                
                if verbose:
                    print(f"✅ Image saved: {image_path}")
            
            return {
                "success": True,
                "total_time": total_time,
                "steps_per_second": num_inference_steps / total_time,
                "image_path": str(image_path) if image_path else None,
                "process_memory_delta_mb": process_delta,
                "system_memory_delta_mb": system_delta_mb,
                "settings": {
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance": guidance_scale,
                    "seed": seed
                },
                "images": images
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    def benchmark_generation(
        self, 
        prompt: str = "a beautiful landscape with mountains and lake, digital art",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        num_runs: int = 3,
        warmup_runs: int = 1
    ) -> Dict:
        """Comprehensive benchmark with multiple runs and detailed metrics"""
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        print(f"Target: {width}x{height}, {num_inference_steps} steps")
        print(f"Prompt: {prompt}")
        print(f"Runs: {warmup_runs} warmup + {num_runs} benchmark")
        
        all_times = []
        memory_stats = []
        
        # Warmup runs
        if warmup_runs > 0:
            print(f"\n🔥 Running {warmup_runs} warmup run(s)...")
            for i in range(warmup_runs):
                try:
                    result = self.generate_with_metrics(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        save_image=False,
                        verbose=False
                    )
                    if result["success"]:
                        print(f"   Warmup {i+1}/{warmup_runs} completed in {result['total_time']:.2f}s")
                except Exception as e:
                    print(f"   Warmup {i+1} failed: {e}")
        
        # Benchmark runs
        print(f"\n⏱️  Running {num_runs} benchmark run(s)...")
        
        for i in range(num_runs):
            try:
                print(f"\n--- Run {i+1}/{num_runs} ---")
                
                result = self.generate_with_metrics(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    save_image=(i == 0),  # Save only first image
                    verbose=True
                )
                
                if result["success"]:
                    all_times.append(result["total_time"])
                    memory_stats.append({
                        "process_delta_mb": result.get("process_memory_delta_mb", 0),
                        "system_delta_mb": result.get("system_memory_delta_mb", 0)
                    })
                    print(f"✓ Run {i+1}: {result['total_time']:.2f}s")
                
                # Force garbage collection between runs
                gc.collect()
                
            except Exception as e:
                print(f"❌ Run {i+1} failed: {e}")
        
        # Calculate statistics
        if not all_times:
            print("❌ No successful runs!")
            return {}
        
        benchmark_results = {
            "target_achieved": min(all_times) <= 4.0,  # 3-4 second target
            "times": all_times,
            "min_time": min(all_times),
            "max_time": max(all_times),
            "mean_time": statistics.mean(all_times),
            "median_time": statistics.median(all_times),
            "std_dev": statistics.stdev(all_times) if len(all_times) > 1 else 0,
            "fps": 1.0 / statistics.mean(all_times),
            "settings": {
                "width": width,
                "height": height,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale
            },
            "memory_stats": memory_stats
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"🎯 Target (3-4s):     {'✅ ACHIEVED' if benchmark_results['target_achieved'] else '❌ NOT ACHIEVED'}")
        print(f"⚡ Best Time:        {benchmark_results['min_time']:.2f}s")
        print(f"🐌 Worst Time:       {benchmark_results['max_time']:.2f}s")
        print(f"📊 Average Time:     {benchmark_results['mean_time']:.2f}s")
        print(f"📈 Median Time:      {benchmark_results['median_time']:.2f}s")
        print(f"📉 Std Deviation:    {benchmark_results['std_dev']:.3f}s")
        print(f"🎬 FPS:              {benchmark_results['fps']:.2f}")
        
        # Memory statistics
        if memory_stats:
            process_deltas = [m["process_delta_mb"] for m in memory_stats]
            system_deltas = [m["system_delta_mb"] for m in memory_stats]
            
            if process_deltas:
                avg_process_delta = statistics.mean(process_deltas)
                print(f"💾 Process Memory Delta: {avg_process_delta:.1f} MB avg")
            
            if system_deltas:
                avg_system_delta = statistics.mean(system_deltas)
                print(f"🎮 GPU Memory Delta (System): {avg_system_delta:.0f} MB avg")
        
        # Performance rating
        best_time = benchmark_results['min_time']
        if best_time <= 2.0:
            rating = "🚀 EXCELLENT"
        elif best_time <= 4.0:
            rating = "✅ GOOD"
        elif best_time <= 6.0:
            rating = "⚠️  ACCEPTABLE"
        else:
            rating = "❌ NEEDS OPTIMIZATION"
        
        print(f"🏆 Performance:      {rating}")
        
        # Optimization suggestions
        if not benchmark_results['target_achieved']:
            print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
            if self.device == "CPU":
                print("   • Switch to GPU device for better performance")
            print("   • Reduce number of inference steps (try 15-20)")
            print("   • Use smaller image size (try 448x448)")
            print("   • Try different schedulers or optimizations")
        
        self.performance_stats["generations"].append(benchmark_results)
        return benchmark_results
    
    def save_performance_report(self, output_file: str = "genai_performance_report.json"):
        """Save detailed performance report"""
        report = {
            "timestamp": time.time(),
            "system_info": self.performance_stats["system_info"],
            "version_info": self.performance_stats["version_info"],
            "init_time": self.performance_stats["init_time"],
            "generations": self.performance_stats["generations"],
            "model_id": self.model_id,
            "device": self.device,
            "api": "openvino_genai"
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Performance report saved: {output_file}")


def main():
    """Main function to run performance benchmarks"""
    parser = argparse.ArgumentParser(description="Performance-Enhanced Stable Diffusion with OpenVINO GenAI")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and lake, digital art", help="Text prompt")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="GPU", help="Device (GPU/CPU/AUTO/NPU)")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID")
    parser.add_argument("--model-cache", type=str, default="./models", help="Model cache directory")
    
    # Benchmark options
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--save-report", action="store_true", help="Save performance report")
    
    args = parser.parse_args()
    
    try:
        # Initialize the pipeline
        genai_sd = PerformanceOpenVINOGenAI(
            model_id=args.model_id,
            device=args.device,
            cache_dir=args.model_cache
        )
        
        if args.benchmark:
            # Run benchmark
            benchmark_results = genai_sd.benchmark_generation(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                num_runs=args.benchmark_runs,
                warmup_runs=args.warmup_runs
            )
            
            if args.save_report:
                genai_sd.save_performance_report("genai_benchmark_report.json")
        
        else:
            # Single generation with metrics
            result = genai_sd.generate_with_metrics(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
                verbose=True
            )
            
            if result["success"]:
                print(f"\n🎉 Successfully generated image in {result['total_time']:.2f}s")
                if result["image_path"]:
                    print(f"📁 Saved to: {result['image_path']}")
            else:
                print(f"\n❌ Generation failed: {result['error']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
