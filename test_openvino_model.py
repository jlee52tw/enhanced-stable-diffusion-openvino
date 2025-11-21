#!/usr/bin/env python3
"""
Simple test script for OpenVINO Stable Diffusion
"""

import os
import time
from pathlib import Path

import openvino as ov
import numpy as np
from PIL import Image

def test_openvino_diffusion():
    """Test our converted OpenVINO model"""
    
    print("🚀 Testing OpenVINO Stable Diffusion")
    print(f"Working directory: {Path.cwd()}")
    
    # Check OpenVINO devices
    core = ov.Core()
    devices = core.available_devices
    print(f"Available devices: {devices}")
    
    if "GPU" in devices:
        try:
            gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            print(f"GPU: {gpu_name}")
        except:
            print("GPU detected but couldn't get detailed info")
    
    # Check if our model exists
    model_path = Path("./models/stable-diffusion-v1-5-fp16")
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        return False
        
    print(f"✓ Model found at: {model_path}")
    
    # List model components
    components = list(model_path.iterdir())
    print("Model components:")
    for comp in components:
        print(f"  - {comp.name}")
    
    # Try to load the UNet component as a simple test
    unet_path = model_path / "unet"
    if unet_path.exists():
        try:
            print(f"Loading UNet from: {unet_path}")
            
            # Look for .xml file
            xml_files = list(unet_path.glob("*.xml"))
            if xml_files:
                xml_file = xml_files[0]
                print(f"Found model file: {xml_file}")
                
                # Try to load the model
                model = core.read_model(str(xml_file))
                print(f"✓ Model loaded successfully")
                print(f"Model inputs: {[inp.get_names() for inp in model.inputs]}")
                print(f"Model outputs: {[out.get_names() for out in model.outputs]}")
                
                # Try to compile for GPU
                if "GPU" in devices:
                    try:
                        compiled_model = core.compile_model(model, "GPU")
                        print("✓ Model compiled successfully for GPU")
                        return True
                    except Exception as e:
                        print(f"❌ GPU compilation failed: {e}")
                        
                # Fallback to CPU
                try:
                    compiled_model = core.compile_model(model, "CPU")
                    print("✓ Model compiled successfully for CPU")
                    return True
                except Exception as e:
                    print(f"❌ CPU compilation failed: {e}")
                    
            else:
                print("❌ No .xml model files found in UNet directory")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print("❌ UNet directory not found")
    
    return False

if __name__ == "__main__":
    success = test_openvino_diffusion()
    if success:
        print("\n🎉 OpenVINO model test completed successfully!")
    else:
        print("\n❌ OpenVINO model test failed!")
