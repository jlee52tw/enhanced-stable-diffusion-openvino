#!/usr/bin/env python3
"""
Example usage of Stable Diffusion with OpenVINO
This script demonstrates various ways to use the Stable Diffusion pipeline.
"""

from stable_diffusion_openvino import OpenVINOStableDiffusion
import time

def example_basic_generation():
    """Basic image generation example"""
    print("🎨 Example 1: Basic Image Generation")
    
    # Initialize the pipeline
    sd = OpenVINOStableDiffusion(device="GPU")  # Will fallback to CPU if GPU not available
    
    # Generate a single image
    images = sd.generate_image(
        prompt="A beautiful landscape with mountains and a lake at sunset, digital art",
        num_inference_steps=20,
        guidance_scale=7.5
    )
    
    if images:
        sd.save_images(images, "./outputs", "landscape")
        print("✅ Basic generation completed!")

def example_multiple_images():
    """Generate multiple images with different settings"""
    print("\n🎨 Example 2: Multiple Images with Variations")
    
    sd = OpenVINOStableDiffusion(device="GPU")
    
    # Generate multiple images
    images = sd.generate_image(
        prompt="A cute robot in a futuristic city",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=25,
        guidance_scale=8.0,
        num_images=3,
        seed=42  # For reproducible results
    )
    
    if images:
        sd.save_images(images, "./outputs", "robot_city")
        print("✅ Multiple images generated!")

def example_high_quality():
    """High quality generation with more steps"""
    print("\n🎨 Example 3: High Quality Generation")
    
    sd = OpenVINOStableDiffusion(device="GPU")
    
    # High quality settings
    images = sd.generate_image(
        prompt="A detailed portrait of a wise old wizard with a long beard, fantasy art, highly detailed",
        negative_prompt="blurry, low quality, cartoon, anime, distorted face",
        num_inference_steps=30,
        guidance_scale=8.5,
        width=512,
        height=768  # Portrait orientation
    )
    
    if images:
        sd.save_images(images, "./outputs", "wizard_portrait")
        print("✅ High quality generation completed!")

def example_batch_generation():
    """Generate multiple different prompts"""
    print("\n🎨 Example 4: Batch Generation with Different Prompts")
    
    sd = OpenVINOStableDiffusion(device="GPU")
    
    prompts = [
        "A serene forest path with sunlight filtering through trees",
        "A cyberpunk city street at night with neon lights",
        "A cozy cabin in the mountains during winter",
        "A tropical beach with palm trees and crystal blue water"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i+1}/{len(prompts)}: {prompt}")
        
        images = sd.generate_image(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=i * 100  # Different seed for each
        )
        
        if images:
            sd.save_images(images, "./outputs", f"batch_{i+1}")
            print(f"✅ Image {i+1} completed!")
        
        time.sleep(1)  # Small delay between generations

def run_performance_test():
    """Test generation performance"""
    print("\n⚡ Performance Test")
    
    sd = OpenVINOStableDiffusion(device="GPU")
    
    test_prompt = "A simple test image of a red apple on a table"
    
    # Test different step counts
    step_counts = [10, 15, 20, 25]
    
    for steps in step_counts:
        start_time = time.time()
        
        images = sd.generate_image(
            prompt=test_prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            width=512,
            height=512
        )
        
        generation_time = time.time() - start_time
        
        if images:
            print(f"Steps: {steps:2d} | Time: {generation_time:.2f}s | Speed: {generation_time/steps:.2f}s/step")
        else:
            print(f"Steps: {steps:2d} | Generation failed")

def main():
    """Run all examples"""
    print("🚀 Stable Diffusion OpenVINO Examples")
    print("This script will run several examples to demonstrate the capabilities.")
    print("Make sure you have run setup.py first!")
    
    try:
        # Run examples
        example_basic_generation()
        example_multiple_images()
        example_high_quality()
        example_batch_generation()
        run_performance_test()
        
        print("\n🎉 All examples completed successfully!")
        print("Check the ./outputs directory for generated images.")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure you have run setup.py and installed all requirements.")

if __name__ == "__main__":
    main()
