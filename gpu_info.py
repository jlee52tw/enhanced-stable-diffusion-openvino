#!/usr/bin/env python3
"""
GPU and Version Information Script
"""

import openvino as ov

def main():
    print("🔍 GPU AND VERSION INFORMATION")
    print("=" * 50)
    
    # OpenVINO version
    print(f"📦 OpenVINO Version: {ov.__version__}")
    
    # Other package versions
    try:
        import diffusers
        print(f"📦 Diffusers Version: {diffusers.__version__}")
    except:
        print("📦 Diffusers Version: Not installed")
    
    try:
        import torch
        print(f"📦 PyTorch Version: {torch.__version__}")
    except:
        print("📦 PyTorch Version: Not installed")
        
    try:
        import transformers
        print(f"📦 Transformers Version: {transformers.__version__}")
    except:
        print("📦 Transformers Version: Not installed")
    
    # OpenVINO devices
    core = ov.Core()
    devices = core.available_devices
    print(f"\n🖥️  Available OpenVINO devices: {devices}")
    
    # GPU details
    if "GPU" in devices:
        print(f"\n🎮 GPU INFORMATION:")
        
        try:
            gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            print(f"   Name: {gpu_name}")
        except:
            print("   Name: Not available")
            
        try:
            gpu_memory = core.get_property("GPU", "GPU_DEVICE_TOTAL_MEM_SIZE")
            print(f"   Total Memory: {gpu_memory / (1024**3):.1f} GB")
        except:
            print("   Total Memory: Not available")
            
        try:
            device_id = core.get_property("GPU", "GPU_DEVICE_ID")
            print(f"   Device ID: 0x{device_id:04x}")
        except:
            print("   Device ID: Not available")
            
        try:
            driver_version = core.get_property("GPU", "GPU_DRIVER_VERSION")
            print(f"   Driver Version: {driver_version}")
        except Exception as e:
            print(f"   Driver Version: Not available ({e})")
            
        try:
            opencl_version = core.get_property("GPU", "OPENCL_VERSION")
            print(f"   OpenCL Version: {opencl_version}")
        except:
            print("   OpenCL Version: Not available")
    else:
        print("\n❌ No GPU detected")

if __name__ == "__main__":
    main()
