#!/usr/bin/env python3
"""
Diagnostic Script for Stable Diffusion OpenVINO Setup
Checks system state, model availability, and network connectivity
"""

import os
import sys
from pathlib import Path
import subprocess
import urllib3
import json

# Configure proxy
INTEL_PROXY = "http://proxy-dmz.intel.com:912"
os.environ['HTTP_PROXY'] = INTEL_PROXY
os.environ['HTTPS_PROXY'] = INTEL_PROXY
os.environ['http_proxy'] = INTEL_PROXY
os.environ['https_proxy'] = INTEL_PROXY

def check_python_environment():
    """Check Python environment and virtual environment"""
    print("🐍 PYTHON ENVIRONMENT")
    print("=" * 50)
    
    print(f"✓ Python version: {sys.version}")
    print(f"✓ Python executable: {sys.executable}")
    
    # Check if we're in a virtual environment
    venv_path = Path("venv")
    if venv_path.exists():
        print("✓ Virtual environment exists: venv/")
        if "venv" in sys.executable.lower():
            print("✓ Currently using virtual environment")
        else:
            print("⚠️  Virtual environment exists but not activated")
            print("   Run: venv\\Scripts\\activate.bat")
    else:
        print("❌ Virtual environment not found")
        print("   Run: setup_with_proxy.bat")

def check_packages():
    """Check required packages"""
    print("\n📦 REQUIRED PACKAGES")
    print("=" * 50)
    
    required_packages = [
        "openvino",
        "openvino_genai", 
        "openvino_tokenizers",
        "optimum",
        "torch",
        "diffusers",
        "transformers",
        "pillow",
        "numpy",
        "psutil",
        "tqdm"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            # Get version if possible
            try:
                module = __import__(package.replace("-", "_"))
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {package}: {version}")
            except:
                print(f"✓ {package}: installed")
        except ImportError:
            print(f"❌ {package}: not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: setup_with_proxy.bat to install")
    
    return len(missing_packages) == 0

def check_model_files():
    """Check OpenVINO model files"""
    print("\n🤖 MODEL FILES")
    print("=" * 50)
    
    # Check different possible model locations
    possible_paths = [
        "models/stable-diffusion-v1-5_ov",
        "models/stable_diffusion_ov",
        "models/stable-diffusion-1-5",
        "./stable-diffusion-v1-5_ov"
    ]
    
    model_found = False
    for path in possible_paths:
        model_path = Path(path)
        if model_path.exists():
            print(f"✓ Model directory found: {model_path}")
            
            # Check required files
            required_files = [
                "openvino_model.xml",
                "openvino_model.bin", 
                "model_index.json"
            ]
            
            for file in required_files:
                file_path = model_path / file
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024*1024)
                    print(f"  ✓ {file}: {size_mb:.1f} MB")
                else:
                    print(f"  ❌ {file}: missing")
            
            model_found = True
            break
    
    if not model_found:
        print("❌ No OpenVINO model found")
        print("Available directories:")
        models_dir = Path("models")
        if models_dir.exists():
            for item in models_dir.iterdir():
                if item.is_dir():
                    print(f"   {item.name}/")
        else:
            print("   models/ directory doesn't exist")
            
        print("\nRun: setup_with_proxy.bat to download and convert model")
    
    return model_found

def check_openvino_devices():
    """Check OpenVINO device availability"""
    print("\n🔧 OPENVINO DEVICES")
    print("=" * 50)
    
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        
        print(f"✓ OpenVINO version: {ov.__version__}")
        print(f"✓ Available devices: {devices}")
        
        # Test each device
        for device in devices:
            try:
                if device == "NPU":
                    print(f"  {device}: Intel Neural Processing Unit")
                elif device.startswith("GPU"):
                    gpu_info = core.get_property(device, "FULL_DEVICE_NAME") 
                    print(f"  {device}: {gpu_info}")
                elif device == "CPU":
                    print(f"  {device}: Available")
                else:
                    print(f"  {device}: Available")
            except Exception as e:
                print(f"  {device}: Available but details not accessible")
        
        return True
    except ImportError:
        print("❌ OpenVINO not installed")
        return False
    except Exception as e:
        print(f"❌ OpenVINO error: {e}")
        return False

def check_network_connectivity():
    """Check network connectivity and proxy"""
    print("\n🌐 NETWORK CONNECTIVITY")  
    print("=" * 50)
    
    print(f"Proxy configured: {INTEL_PROXY}")
    
    try:
        import requests
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        proxies = {
            'http': INTEL_PROXY,
            'https': INTEL_PROXY
        }
        
        # Test basic connectivity
        response = requests.get(
            'https://httpbin.org/ip', 
            proxies=proxies, 
            timeout=10,
            verify=False
        )
        external_ip = response.json().get('origin', 'Unknown')
        print(f"✓ Internet connectivity: OK (IP: {external_ip})")
        
        # Test HuggingFace
        try:
            response = requests.get(
                'https://huggingface.co', 
                proxies=proxies, 
                timeout=10,
                verify=False
            )
            if response.status_code == 200:
                print("✓ HuggingFace connectivity: OK")
            else:
                print(f"⚠️  HuggingFace returned status: {response.status_code}")
        except Exception as e:
            print(f"❌ HuggingFace connectivity failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Network connectivity failed: {e}")
        print("Check proxy settings and internet connection")
        return False

def check_huggingface_cache():
    """Check HuggingFace cache"""
    print("\n📁 HUGGINGFACE CACHE")
    print("=" * 50)
    
    cache_dir = Path("./huggingface_cache")
    if cache_dir.exists():
        print(f"✓ Cache directory exists: {cache_dir}")
        
        # Count files
        total_files = sum(1 for _ in cache_dir.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        
        print(f"  Files: {total_files}")
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
    else:
        print("⚠️  Cache directory doesn't exist")
        print("Will be created on first download")

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 QUICK FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        # Check if model exists
        model_path = None
        for path in ["models/stable-diffusion-v1-5_ov", "models/stable_diffusion_ov"]:
            if Path(path).exists():
                model_path = path
                break
        
        if not model_path:
            print("❌ No model found for testing")
            return False
        
        print(f"Using model: {model_path}")
        
        # Try to import and load
        import openvino_genai as ov_genai
        print("✓ OpenVINO GenAI imported")
        
        # Try to create pipeline (this will test everything)
        pipeline = ov_genai.Text2ImagePipeline(model_path, "CPU")
        print("✓ Pipeline created on CPU")
        
        print("🎉 Basic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("🔍 STABLE DIFFUSION OPENⅤINO DIAGNOSTICS")
    print("=" * 60)
    
    results = {
        "python_env": True,
        "packages": False,
        "model": False,
        "openvino": False,
        "network": False,
        "functionality": False
    }
    
    # Run checks
    check_python_environment()
    results["packages"] = check_packages()
    results["model"] = check_model_files()
    results["openvino"] = check_openvino_devices()
    results["network"] = check_network_connectivity()
    check_huggingface_cache()
    
    if results["packages"] and results["model"] and results["openvino"]:
        results["functionality"] = run_quick_test()
    
    # Summary
    print("\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    all_good = True
    for check, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")
        if not status:
            all_good = False
    
    print(f"\n{'🎉 All systems ready!' if all_good else '⚠️  Issues detected'}")
    
    if not all_good:
        print("\nNext steps:")
        if not results["packages"]:
            print("1. Run: setup_with_proxy.bat")
        elif not results["model"]:
            print("1. Run: python download_model_manual.py")
        elif not results["functionality"]:
            print("1. Try: python enhanced_performance_sd.py --device CPU")
    else:
        print("You can run: run_enhanced_performance.bat")

if __name__ == "__main__":
    main()
