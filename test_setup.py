
# Quick test of Stable Diffusion setup
from stable_diffusion_openvino import OpenVINOStableDiffusion

print("Testing Stable Diffusion setup...")

try:
    # Initialize with CPU first for compatibility
    sd = OpenVINOStableDiffusion(device="CPU")
    print("✅ Pipeline initialized successfully!")
    
    # Test with a simple prompt
    images = sd.generate_image(
        prompt="a simple red apple",
        num_inference_steps=10,  # Reduced steps for quick test
        width=256,  # Smaller size for quick test
        height=256
    )
    
    if images:
        sd.save_images(images, "./test_outputs", "test")
        print("✅ Test generation completed successfully!")
    else:
        print("❌ Test generation failed")
        
except Exception as e:
    print(f"❌ Test failed: {e}")
