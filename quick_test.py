#!/usr/bin/env python3
"""
Quick test script to verify OpenVINO GenAI Stable Diffusion setup
"""

import os
import sys
from pathlib import Path

# Configure proxy
INTEL_PROXY = "http://proxy-dmz.intel.com:912"
os.environ['HTTP_PROXY'] = INTEL_PROXY
os.environ['HTTPS_PROXY'] = INTEL_PROXY
os.environ['http_proxy'] = INTEL_PROXY
os.environ['https_proxy'] = INTEL_PROXY

try:
    import openvino as ov
    import openvino_genai as ov_genai
    print(f"✓ OpenVINO: {ov.__version__}")
    print(f"✓ OpenVINO GenAI: {ov_genai.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_device_availability():
    """Test OpenVINO device availability"""
    print("\n🔧 OpenVINO Device Check:")
    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")
    
    for device in devices:
        try:
            if device == "NPU":
                print(f"  {device}: Intel Neural Processing Unit")
            elif device.startswith("GPU"):
                gpu_info = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"  {device}: {gpu_info}")
            else:
                print(f"  {device}: Available")
        except Exception as e:
            print(f"  {device}: Available (details not accessible)")

def test_model_loading():
    """Test model loading"""
    print("\n🤖 Model Loading Test:")
    
    model_path = Path("models/stable-diffusion-v1-5_ov")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"✓ Model directory exists: {model_path}")
    
    # Check for required files
    required_files = ["model_index.json"]
    for file in required_files:
        if (model_path / file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ❌ {file}")
            return False
    
    try:
        # Try to load pipeline on CPU first (most compatible)
        print("🔄 Loading Text2ImagePipeline on CPU...")
        pipeline = ov_genai.Text2ImagePipeline(str(model_path), "CPU")
        print("✅ Text2ImagePipeline loaded successfully on CPU!")
        
        # Try GPU if available
        core = ov.Core()
        if "GPU" in core.available_devices:
            print("🔄 Loading Text2ImagePipeline on GPU...")
            gpu_pipeline = ov_genai.Text2ImagePipeline(str(model_path), "GPU")
            print("✅ Text2ImagePipeline loaded successfully on GPU!")
            
            # Quick generation test
            print("🎨 Quick generation test...")
            result = gpu_pipeline.generate("A simple test image", num_inference_steps=5, width=256, height=256)
            print(f"✅ Generation test successful! Generated {len(result.data)} image(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    print("🔍 Quick OpenVINO GenAI Setup Test")
    print("=" * 50)
    
    # Test device availability
    test_device_availability()
    
    # Test model loading
    success = test_model_loading()
    
    print(f"\n{'🎉 All tests passed!' if success else '⚠️ Some tests failed'}")
    
    if success:
        print("\n✅ Setup is working! You can now run:")
        print("  python enhanced_performance_sd.py --benchmark")
        print("  or")
        print("  run_enhanced_performance.bat")
    else:
        print("\n❌ Setup issues detected. Check model conversion status.")

if __name__ == "__main__":
    main()
