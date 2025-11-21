#!/usr/bin/env python3
"""
Simple GPU Memory Test for Stable Diffusion
"""

import os
import time
import psutil
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import openvino as ov
    from optimum.intel.openvino import OVStableDiffusionPipeline
    from diffusers import EulerDiscreteScheduler
except ImportError as e:
    print(f"Required packages not found: {e}")
    exit(1)

def get_memory_usage():
    """Get current memory usage"""
    # Process memory
    process = psutil.Process()
    process_memory = process.memory_info().rss / (1024**2)  # MB
    
    # System memory
    system_memory = psutil.virtual_memory()
    
    # Try to get GPU memory usage via Windows Task Manager equivalent
    try:
        import subprocess
        # This might show GPU memory usage
        result = subprocess.run([
            "tasklist", "/fi", f"PID eq {os.getpid()}"
        ], capture_output=True, text=True)
        
        print(f"Process info: {result.stdout}")
    except:
        pass
    
    return {
        "process_mb": process_memory,
        "system_used_gb": (system_memory.total - system_memory.available) / (1024**3),
        "system_available_gb": system_memory.available / (1024**3)
    }

def main():
    print("🔍 GPU Memory Usage Test for Stable Diffusion")
    print("=" * 60)
    
    # Initial memory
    print("📊 Initial Memory State:")
    initial_memory = get_memory_usage()
    print(f"  Process: {initial_memory['process_mb']:.1f} MB")
    print(f"  System Used: {initial_memory['system_used_gb']:.1f} GB")
    
    print(f"\n🚀 Loading Stable Diffusion Pipeline...")
    start_time = time.time()
    
    # Load pipeline
    pipeline = OVStableDiffusionPipeline.from_pretrained(
        "./models/stable_diffusion_ov",
        device="GPU",
        compile=True
    )
    
    # Set Euler scheduler
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    
    load_time = time.time() - start_time
    
    # Memory after loading
    print(f"✅ Pipeline loaded in {load_time:.1f}s")
    print("📊 Memory After Pipeline Load:")
    after_load_memory = get_memory_usage()
    print(f"  Process: {after_load_memory['process_mb']:.1f} MB (+{after_load_memory['process_mb'] - initial_memory['process_mb']:.1f} MB)")
    print(f"  System Used: {after_load_memory['system_used_gb']:.1f} GB (+{after_load_memory['system_used_gb'] - initial_memory['system_used_gb']:.1f} GB)")
    
    # Now let's see what happens during inference
    print(f"\n🎨 Running Inference (512x512, 25 steps)...")
    inference_start = time.time()
    
    # Memory just before inference
    pre_inference_memory = get_memory_usage()
    
    # Generate image
    images = pipeline(
        prompt="a simple test image of a cat",
        width=512,
        height=512,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images
    
    inference_time = time.time() - inference_start
    
    # Memory after inference
    post_inference_memory = get_memory_usage()
    
    print(f"✅ Inference completed in {inference_time:.2f}s")
    print("📊 Memory During Inference:")
    print(f"  Before: {pre_inference_memory['process_mb']:.1f} MB, System: {pre_inference_memory['system_used_gb']:.1f} GB")
    print(f"  After:  {post_inference_memory['process_mb']:.1f} MB, System: {post_inference_memory['system_used_gb']:.1f} GB")
    print(f"  Delta:  Process +{post_inference_memory['process_mb'] - pre_inference_memory['process_mb']:.1f} MB, System +{(post_inference_memory['system_used_gb'] - pre_inference_memory['system_used_gb']) * 1024:.0f} MB")
    
    # Save image
    if images:
        output_path = "./outputs/memory_test.png"
        images[0].save(output_path)
        print(f"📁 Image saved: {output_path}")
    
    print(f"\n📈 SUMMARY:")
    print(f"  Pipeline Load: +{(after_load_memory['system_used_gb'] - initial_memory['system_used_gb']) * 1024:.0f} MB system memory")
    print(f"  Inference: +{(post_inference_memory['system_used_gb'] - pre_inference_memory['system_used_gb']) * 1024:.0f} MB system memory") 
    print(f"  Total: +{(post_inference_memory['system_used_gb'] - initial_memory['system_used_gb']) * 1024:.0f} MB system memory")
    print(f"  Performance: {inference_time:.2f}s for 512x512, 25 steps")

if __name__ == "__main__":
    main()
