#!/usr/bin/env python3
"""
Manual Model Download Script for Stable Diffusion 1.5
This script provides alternative methods to download and convert the model
when automatic download fails due to network/proxy issues.
"""

import os
import sys
import requests
import urllib3
from pathlib import Path
import subprocess
import time

# Configure proxy for requests
INTEL_PROXY = "http://proxy-dmz.intel.com:912"
os.environ['HTTP_PROXY'] = INTEL_PROXY
os.environ['HTTPS_PROXY'] = INTEL_PROXY
os.environ['http_proxy'] = INTEL_PROXY
os.environ['https_proxy'] = INTEL_PROXY

# Disable SSL warnings for proxy
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def setup_huggingface_cache():
    """Setup HuggingFace cache directories"""
    cache_dir = Path("./huggingface_cache")
    cache_dir.mkdir(exist_ok=True)
    
    os.environ['HF_HUB_CACHE'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)
    
    print(f"✓ HuggingFace cache directory: {cache_dir}")
    return cache_dir

def test_internet_connection():
    """Test internet connection with proxy"""
    print("🔍 Testing internet connection with proxy...")
    
    proxies = {
        'http': INTEL_PROXY,
        'https': INTEL_PROXY
    }
    
    try:
        response = requests.get(
            'https://httpbin.org/ip', 
            proxies=proxies, 
            timeout=10,
            verify=False
        )
        print(f"✓ Internet connection working. External IP: {response.json().get('origin', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Internet connection test failed: {e}")
        return False

def test_huggingface_connection():
    """Test connection to HuggingFace"""
    print("🔍 Testing HuggingFace connection...")
    
    try:
        from huggingface_hub import HfApi
        
        # Test with proxy
        api = HfApi()
        model_info = api.model_info("runwayml/stable-diffusion-v1-5")
        print(f"✓ HuggingFace connection working. Model size: ~{model_info.safetensors['total'] / (1024**3):.1f}GB")
        return True
    except Exception as e:
        print(f"❌ HuggingFace connection failed: {e}")
        return False

def download_model_with_hf_hub():
    """Download model using huggingface_hub directly"""
    print("📥 Downloading model using huggingface_hub...")
    
    try:
        from huggingface_hub import snapshot_download
        
        model_path = snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            cache_dir="./huggingface_cache",
            resume_download=True
        )
        
        print(f"✓ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Direct download failed: {e}")
        return None

def convert_to_openvino(model_path):
    """Convert downloaded model to OpenVINO format"""
    print("🔄 Converting to OpenVINO format...")
    
    output_dir = Path("models/stable-diffusion-v1-5_ov")
    output_dir.parent.mkdir(exist_ok=True)
    
    try:
        # Use optimum-cli for conversion
        cmd = [
            "venv/Scripts/optimum-cli.exe",
            "export", "openvino",
            "--model", str(model_path),
            "--task", "text-to-image", 
            "--weight-format", "fp16",
            str(output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model conversion successful!")
            return str(output_dir)
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return None

def alternative_download_methods():
    """Show alternative download methods"""
    print("\n" + "="*60)
    print("ALTERNATIVE DOWNLOAD METHODS")
    print("="*60)
    
    print("\n1. Manual Git Clone Method:")
    print("   git config --global http.proxy http://proxy-dmz.intel.com:912")
    print("   git clone https://huggingface.co/runwayml/stable-diffusion-v1-5")
    print("   Then convert with: optimum-cli export openvino ...")
    
    print("\n2. Direct Files Download:")
    print("   Download individual files from:")
    print("   https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main")
    
    print("\n3. Use Pre-converted Model:")
    print("   Look for pre-converted OpenVINO models on:")
    print("   https://huggingface.co/OpenVINO/stable-diffusion-v1-5-ov")
    
    print("\n4. Alternative Proxy Configuration:")
    print("   Try different proxy ports: 911, 912, 913")
    print("   Or check with IT for current proxy settings")

def main():
    print("🚀 Manual Model Download and Conversion Script")
    print("="*60)
    
    # Setup cache
    setup_huggingface_cache()
    
    # Check if model already exists
    model_dir = Path("models/stable-diffusion-v1-5_ov")
    if (model_dir / "openvino_model.xml").exists():
        print("✅ OpenVINO model already exists!")
        print(f"Location: {model_dir}")
        return
    
    # Test connections
    internet_ok = test_internet_connection()
    if not internet_ok:
        print("\n❌ Internet connection failed. Check proxy settings.")
        alternative_download_methods()
        return
    
    hf_ok = test_huggingface_connection()
    if not hf_ok:
        print("\n❌ HuggingFace connection failed.")
        alternative_download_methods()
        return
    
    # Try download
    print("\n📥 Starting model download...")
    model_path = download_model_with_hf_hub()
    
    if model_path:
        # Convert to OpenVINO
        ov_path = convert_to_openvino(model_path)
        
        if ov_path:
            print(f"\n✅ Success! OpenVINO model ready at: {ov_path}")
            print("\nYou can now run:")
            print("  python enhanced_performance_sd.py --benchmark")
            print("  or")
            print("  run_enhanced_performance.bat")
        else:
            print("\n❌ Model conversion failed")
            alternative_download_methods()
    else:
        print("\n❌ Model download failed")
        alternative_download_methods()

if __name__ == "__main__":
    main()
