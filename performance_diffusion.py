#!/usr/bin/env python3
"""
Performance-Optimized Stable Diffusion with OpenVINO
Focus on speed and performance metrics evaluation

Target: 512x512, Euler sampler, 25 steps in 3-4 seconds on Intel GPU
"""

import os
import argparse
import time
import statistics
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json

import numpy as np
import torch
from PIL import Image
import psutil

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import openvino as ov
    from optimum.intel.openvino import OVStableDiffusionPipeline
    from diffusers import EulerDiscreteScheduler
    import gc
except ImportError as e:
    print(f"Required packages not found: {e}")
    print("Please install: pip install openvino optimum[openvino] diffusers transformers torch pillow psutil")
    exit(1)


class PerformanceStableDiffusion:
    """
    Performance-optimized Stable Diffusion with comprehensive metrics
    """
    
    def __init__(
        self, 
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "GPU",
        enable_optimizations: bool = True
    ):
        self.model_id = model_id
        self.device = device
        self.pipeline = None
        self.enable_optimizations = enable_optimizations
        
        # Performance tracking
        self.performance_stats = {
            "init_time": 0,
            "generations": [],
            "system_info": {}
        }
        
        print(f"🚀 Initializing Performance-Optimized Stable Diffusion")
        print(f"Model: {model_id}")
        print(f"Device: {device}")
        print(f"Optimizations: {'Enabled' if enable_optimizations else 'Disabled'}")
        
        self._check_system()
        self._load_pipeline()
    
    def _check_system(self):
        """Check system capabilities and OpenVINO devices"""
        print(f"\n{'='*60}")
        print("SYSTEM INFORMATION")
        print(f"{'='*60}")
        
        # OpenVINO version information
        try:
            openvino_version = ov.get_version()
            print(f"✓ OpenVINO Version: {openvino_version}")
        except:
            try:
                import openvino
                print(f"✓ OpenVINO Version: {openvino.__version__}")
            except:
                print("⚠️  Could not determine OpenVINO version")
        
        # Additional version info
        version_info = self._get_version_info()
        print(f"✓ Diffusers Version: {version_info['diffusers_version']}")
        print(f"✓ PyTorch Version: {version_info['torch_version']}")
        print(f"✓ Optimum Version: {version_info['optimum_version']}")
        
        # OpenVINO devices
        core = ov.Core()
        devices = core.available_devices
        print(f"Available OpenVINO devices: {devices}")
        
        gpu_detected = False
        if "GPU" in devices:
            try:
                # Get detailed GPU info
                gpu_info = self._get_detailed_gpu_info()
                
                if "error" not in gpu_info:
                    print(f"✓ GPU: {gpu_info['gpu_name']}")
                    print(f"✓ GPU Total Memory: {gpu_info['gpu_memory_gb']:.1f} GB")
                    
                    if "gpu_driver_version" in gpu_info:
                        print(f"✓ GPU Driver Version: {gpu_info['gpu_driver_version']}")
                    if "gpu_device_id" in gpu_info:
                        print(f"✓ GPU Device ID: {gpu_info['gpu_device_id']}")
                    if "opencl_version" in gpu_info:
                        print(f"✓ OpenCL Version: {gpu_info['opencl_version']}")
                    
                    gpu_detected = True
                    
                    # Store in performance stats
                    self.performance_stats["system_info"]["gpu_name"] = gpu_info['gpu_name']
                    self.performance_stats["system_info"]["gpu_memory_gb"] = gpu_info['gpu_memory_gb']
                    if "gpu_driver_version" in gpu_info:
                        self.performance_stats["system_info"]["gpu_driver_version"] = gpu_info['gpu_driver_version']
                
                else:
                    print(f"⚠️  GPU detected but couldn't get detailed info: {gpu_info['error']}")
                    
            except Exception as e:
                print(f"⚠️  GPU detected but couldn't get details: {e}")
        
        # NPU information
        if "NPU" in devices:
            print(f"✓ NPU detected and available")
            if self.device == "NPU":
                print(f"⚠️  NPU support for Stable Diffusion is experimental and may not work")
        
        # Device fallback logic
        if self.device == "NPU" and "NPU" not in devices:
            print("❌ NPU requested but not detected, falling back to GPU")
            self.device = "GPU"
        elif self.device == "GPU" and not gpu_detected:
            print("❌ No GPU detected, falling back to CPU")
            self.device = "CPU"
        
        # System memory
        memory = psutil.virtual_memory()
        print(f"✓ System RAM: {memory.total / (1024**3):.1f} GB")
        print(f"✓ Available RAM: {memory.available / (1024**3):.1f} GB")
        
        # CPU info
        print(f"✓ CPU Cores: {psutil.cpu_count()} physical, {psutil.cpu_count(logical=False)} logical")
        
        # Store version info in performance stats
        self.performance_stats["version_info"] = version_info
        
        self.performance_stats["system_info"].update({
            "ram_total_gb": memory.total / (1024**3),
            "ram_available_gb": memory.available / (1024**3),
            "cpu_cores": psutil.cpu_count(),
            "device": self.device
        })
    
    def _load_pipeline(self):
        """Load and optimize the Stable Diffusion pipeline"""
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print("PIPELINE INITIALIZATION")
        print(f"{'='*60}")
        
        try:
            # Check if we have a pre-converted model
            converted_model_path = Path("./models/stable_diffusion_ov")
            
            if converted_model_path.exists():
                print(f"✓ Loading pre-converted OpenVINO model from: {converted_model_path}")
                try:
                    self.pipeline = OVStableDiffusionPipeline.from_pretrained(
                        converted_model_path,
                        device=self.device,
                        compile=True  # Enable compilation for better performance
                    )
                except Exception as e:
                    print(f"❌ Failed to load model on {self.device}: {e}")
                    if self.device == "NPU":
                        print("🔄 NPU failed, falling back to GPU...")
                        self.device = "GPU"
                        self.pipeline = OVStableDiffusionPipeline.from_pretrained(
                            converted_model_path,
                            device=self.device,
                            compile=True
                        )
                    else:
                        raise
            else:
                print(f"🔄 Converting and loading model: {self.model_id}")
                try:
                    self.pipeline = OVStableDiffusionPipeline.from_pretrained(
                        self.model_id,
                        export=True,
                        device=self.device,
                        compile=True
                    )
                except Exception as e:
                    print(f"❌ Failed to convert model for {self.device}: {e}")
                    if self.device == "NPU":
                        print("🔄 NPU conversion failed, falling back to GPU...")
                        self.device = "GPU"
                        self.pipeline = OVStableDiffusionPipeline.from_pretrained(
                            self.model_id,
                            export=True,
                            device=self.device,
                            compile=True
                        )
                    else:
                        raise
                
                # Save converted model for future use
                self.pipeline.save_pretrained(converted_model_path)
                print(f"✓ OpenVINO model saved to: {converted_model_path}")
            
            # Optimize scheduler for performance
            if self.enable_optimizations:
                print("🔄 Applying performance optimizations...")
                
                # Use Euler scheduler for speed
                self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    self.pipeline.scheduler.config
                )
                
                # Enable memory efficient attention if available
                try:
                    self.pipeline.enable_attention_slicing()
                    print("✓ Attention slicing enabled")
                except:
                    pass
                
                # Disable safety checker for performance (optional)
                # self.pipeline.safety_checker = None
                # self.pipeline.requires_safety_checker = False
            
            init_time = time.time() - start_time
            self.performance_stats["init_time"] = init_time
            
            print(f"✅ Pipeline initialized in {init_time:.2f} seconds")
            print(f"✓ Scheduler: {type(self.pipeline.scheduler).__name__}")
            print(f"✓ Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load pipeline: {e}")
            raise
    
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
        """
        Comprehensive benchmark with multiple runs and detailed metrics
        """
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
                    _ = self.generate_with_metrics(
                        prompt=prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        save_image=False,
                        verbose=False
                    )
                    print(f"   Warmup {i+1}/{warmup_runs} completed")
                except Exception as e:
                    print(f"   Warmup {i+1} failed: {e}")
        
        # Benchmark runs
        print(f"\n⏱️  Running {num_runs} benchmark run(s)...")
        
        for i in range(num_runs):
            try:
                print(f"\n--- Run {i+1}/{num_runs} ---")
                
                # Memory before
                process = psutil.Process()
                process_memory_before = process.memory_info().rss / (1024**2)  # MB
                system_memory_before = psutil.virtual_memory()
                system_used_before_gb = (system_memory_before.total - system_memory_before.available) / (1024**3)
                
                print(f"🔍 Memory Before: Process {process_memory_before:.0f} MB, System {system_used_before_gb:.1f} GB used")
                
                # Run generation
                result = self.generate_with_metrics(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    save_image=(i == 0),  # Save only first image
                    verbose=True
                )
                
                # Memory after
                process_memory_after = process.memory_info().rss / (1024**2)  # MB
                system_memory_after = psutil.virtual_memory()
                system_used_after_gb = (system_memory_after.total - system_memory_after.available) / (1024**3)
                
                # Calculate memory deltas
                process_delta = process_memory_after - process_memory_before
                system_delta_mb = (system_used_after_gb - system_used_before_gb) * 1024  # Convert to MB
                
                print(f"🔍 Memory After:  Process {process_memory_after:.0f} MB, System {system_used_after_gb:.1f} GB used")
                print(f"📈 Memory Delta:  Process +{process_delta:.0f} MB, System +{system_delta_mb:.0f} MB (GPU)")
                memory_after = process.memory_info().rss / (1024**2)  # MB
                
                # GPU memory after
                gpu_memory_after = get_gpu_memory_usage()
                
                all_times.append(result["total_time"])
                
                memory_stats.append({
                    "process_before_mb": process_memory_before,
                    "process_after_mb": process_memory_after,
                    "process_delta_mb": process_delta,
                    "system_before_gb": system_used_before_gb,
                    "system_after_gb": system_used_after_gb,
                    "system_delta_mb": system_delta_mb  # This represents GPU memory usage
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
                "guidance_scale": guidance_scale,
                "scheduler": type(self.pipeline.scheduler).__name__
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
        
        if memory_stats:
            # Calculate memory statistics
            process_deltas = [m["process_delta_mb"] for m in memory_stats if m.get("process_delta_mb") is not None]
            system_deltas = [m["system_delta_mb"] for m in memory_stats if m.get("system_delta_mb") is not None]
            
            if process_deltas:
                avg_process_delta = statistics.mean(process_deltas)
                print(f"💾 Process Memory Delta: {avg_process_delta:.1f} MB avg")
            
            if system_deltas:
                avg_system_delta = statistics.mean(system_deltas)
                max_system_delta = max(system_deltas)
                min_system_delta = min(system_deltas)
                print(f"🎮 GPU Memory Delta (System): {avg_system_delta:.0f} MB avg, {min_system_delta:.0f}-{max_system_delta:.0f} MB range")
                
                # Show system memory usage pattern
                system_before_values = [m["system_before_gb"] for m in memory_stats if m.get("system_before_gb") is not None]
                system_after_values = [m["system_after_gb"] for m in memory_stats if m.get("system_after_gb") is not None]
                
                if system_before_values and system_after_values:
                    avg_before = statistics.mean(system_before_values)
                    avg_after = statistics.mean(system_after_values)
                    print(f"🎮 System Memory Usage: {avg_before:.1f} → {avg_after:.1f} GB")
            else:
                # Fallback to process memory if system memory tracking failed
                if process_deltas:
                    avg_process_delta = statistics.mean(process_deltas)
                    print(f"💾 Memory Delta (Process only): {avg_process_delta:.1f} MB avg")
                    print(f"⚠️  System memory tracking failed - showing process memory only")
        
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
            print("   • Enable more aggressive optimizations")
            print("   • Use DPM++ scheduler for fewer steps")
        
        self.performance_stats["generations"].append(benchmark_results)
        return benchmark_results
    
    def generate_with_metrics(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = None,
        save_image: bool = True,
        verbose: bool = False
    ) -> Dict:
        """Generate image with detailed performance metrics"""
        
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None
        
        # Start timing
        start_time = time.time()
        
        if verbose:
            print(f"🎨 Generating: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        
        try:
            # Generate image
            images = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images
            
            total_time = time.time() - start_time
            
            # Save image if requested
            image_path = None
            if save_image and images:
                output_dir = Path("./outputs/performance_test")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = int(time.time())
                image_path = output_dir / f"perf_test_{timestamp}_{total_time:.2f}s.png"
                images[0].save(image_path)
                
                if verbose:
                    print(f"✅ Image saved: {image_path}")
            
            if verbose:
                print(f"⏱️  Generation completed in {total_time:.2f}s")
            
            return {
                "success": True,
                "total_time": total_time,
                "steps_per_second": num_inference_steps / total_time,
                "image_path": str(image_path) if image_path else None,
                "settings": {
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance": guidance_scale
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time
            }
    
    def save_performance_report(self, output_file: str = "performance_report.json"):
        """Save detailed performance report"""
        report = {
            "timestamp": time.time(),
            "system_info": self.performance_stats["system_info"],
            "version_info": self.performance_stats.get("version_info", {}),
            "init_time": self.performance_stats["init_time"],
            "generations": self.performance_stats["generations"],
            "model_id": self.model_id,
            "device": self.device,
            "optimizations_enabled": self.enable_optimizations
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Performance report saved: {output_file}")
    def _get_version_info(self):
        """Get version information for all components"""
        version_info = {}
        
        # OpenVINO version
        try:
            import openvino as ov
            version_info["openvino_version"] = ov.__version__
        except:
            version_info["openvino_version"] = "Unknown"
        
        # Diffusers version
        try:
            import diffusers
            version_info["diffusers_version"] = diffusers.__version__
        except:
            version_info["diffusers_version"] = "Unknown"
            
        # Transformers version
        try:
            import transformers
            version_info["transformers_version"] = transformers.__version__
        except:
            version_info["transformers_version"] = "Unknown"
            
        # Optimum version
        try:
            import optimum
            version_info["optimum_version"] = optimum.__version__
        except:
            version_info["optimum_version"] = "Unknown"
            
        # PyTorch version
        try:
            import torch
            version_info["torch_version"] = torch.__version__
        except:
            version_info["torch_version"] = "Unknown"
            
        return version_info

    def _get_detailed_gpu_info(self):
        """Get detailed GPU information including driver version"""
        gpu_info = {}
        
        try:
            import openvino as ov
            core = ov.Core()
            
            if "GPU" in core.available_devices:
                try:
                    # Basic GPU info
                    gpu_info["gpu_name"] = core.get_property("GPU", "FULL_DEVICE_NAME")
                    gpu_info["gpu_memory_bytes"] = core.get_property("GPU", "GPU_DEVICE_TOTAL_MEM_SIZE")
                    gpu_info["gpu_memory_gb"] = gpu_info["gpu_memory_bytes"] / (1024**3)
                    
                    # Try to get more detailed info
                    try:
                        gpu_info["gpu_device_id"] = core.get_property("GPU", "GPU_DEVICE_ID")
                    except:
                        pass
                        
                    try:
                        gpu_info["gpu_driver_version"] = core.get_property("GPU", "GPU_DRIVER_VERSION")
                    except:
                        pass
                    
                    # OpenCL version
                    try:
                        gpu_info["opencl_version"] = core.get_property("GPU", "OPENCL_VERSION")
                    except:
                        pass
                        
                except Exception as e:
                    gpu_info["error"] = str(e)
        except Exception as e:
            gpu_info["error"] = str(e)
            
        return gpu_info
def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    try:
        core = ov.Core()
        if "GPU" in core.available_devices:
            # Try to get GPU memory info through OpenVINO
            try:
                # Get total GPU memory
                total_memory = core.get_property("GPU", "GPU_DEVICE_TOTAL_MEM_SIZE")
                
                # Try to get used memory (if available)
                try:
                    # This might not be available in all OpenVINO versions
                    used_memory = core.get_property("GPU", "GPU_DEVICE_USED_MEM_SIZE")
                    return {
                        "total_mb": total_memory / (1024**2),
                        "used_mb": used_memory / (1024**2),
                        "available_mb": (total_memory - used_memory) / (1024**2)
                    }
                except:
                    # Fallback to basic info
                    return {
                        "total_mb": total_memory / (1024**2),
                        "used_mb": None,
                        "available_mb": None
                    }
            except Exception as e:
                print(f"Could not get GPU memory details: {e}")
                return None
        return None
    except:
        return None

def measure_gpu_memory_with_process():
    """Alternative GPU memory measurement using system tools"""
    try:
        import subprocess
        # Use Windows Performance Toolkit to get GPU memory usage
        result = subprocess.run([
            "powershell", "-Command",
            "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | Where-Object {$_.InstanceName -match 'python'} | Measure-Object CookedValue -Sum | Select-Object -ExpandProperty Sum"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            try:
                gpu_memory_bytes = float(result.stdout.strip())
                return {"used_mb": gpu_memory_bytes / (1024**2), "method": "perfcounter"}
            except:
                pass
        
        # Fallback: Use Task Manager style approach
        result2 = subprocess.run([
            "powershell", "-Command", 
            "Get-Process -Name 'python' | Select-Object -ExpandProperty GPU"
        ], capture_output=True, text=True, timeout=5)
        
        # Another fallback: Use wmic to get total GPU memory
        result3 = subprocess.run([
            "wmic", "path", "win32_VideoController", "get", 
            "AdapterRAM,Name", "/format:csv"
        ], capture_output=True, text=True, timeout=5)
        
        if result3.returncode == 0:
            lines = result3.stdout.strip().split('\n')
            for line in lines:
                if "Intel" in line and ("Arc" in line or "Graphics" in line):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            adapter_ram = int(parts[1])
                            return {"total_mb": adapter_ram / (1024**2), "method": "wmic"}
                        except:
                            pass
        
        return None
    except Exception as e:
        print(f"Debug: GPU memory measurement failed: {e}")
        return None

def get_detailed_memory_usage():
    """Get comprehensive memory usage information"""
    try:
        process = psutil.Process()
        
        # Process memory details
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Try to get GPU memory through OpenVINO
        gpu_info = get_gpu_memory_usage()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        result = {
            "process_rss_mb": memory_info.rss / (1024**2),
            "process_vms_mb": memory_info.vms / (1024**2), 
            "process_percent": memory_percent,
            "system_total_mb": system_memory.total / (1024**2),
            "system_available_mb": system_memory.available / (1024**2),
            "system_used_percent": system_memory.percent
        }
        
        if gpu_info:
            result.update(gpu_info)
            
        return result
        
    except Exception as e:
        print(f"Error getting memory usage: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Performance-Optimized Stable Diffusion with OpenVINO")
    parser.add_argument("--prompt", type=str, default="a beautiful landscape with mountains and lake, digital art", help="Text prompt")
    parser.add_argument("--negative", type=str, default=None, help="Negative prompt")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="GPU", help="Device (GPU/CPU)")
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID")
    
    # Benchmark options
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--no-optimizations", action="store_true", help="Disable performance optimizations")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save-report", action="store_true", help="Save performance report")
    
    args = parser.parse_args()
    
    try:
        # Initialize performance-optimized pipeline
        sd = PerformanceStableDiffusion(
            model_id=args.model_id,
            device=args.device,
            enable_optimizations=not args.no_optimizations
        )
        
        if args.benchmark:
            # Run benchmark
            benchmark_results = sd.benchmark_generation(
                prompt=args.prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                num_runs=args.benchmark_runs,
                warmup_runs=args.warmup_runs
            )
            
            if args.save_report:
                sd.save_performance_report("benchmark_report.json")
        
        else:
            # Single generation with metrics
            result = sd.generate_with_metrics(
                prompt=args.prompt,
                negative_prompt=args.negative,
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
