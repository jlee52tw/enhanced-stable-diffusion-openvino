#!/usr/bin/env python3
"""
Setup script for Stable Diffusion with OpenVINO
This script helps you set up the environment and test the installation.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔄 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error:")
        print(e.stderr)
        return False

def create_virtual_environment():
    """Create a Python virtual environment for this project"""
    venv_path = Path("./venv")
    
    if venv_path.exists():
        print(f"✅ Virtual environment already exists at: {venv_path.absolute()}")
        return str(venv_path)
    
    print("🔄 Creating Python virtual environment...")
    
    try:
        # Create virtual environment
        result = subprocess.run([sys.executable, "-m", "venv", "venv"], 
                              check=True, capture_output=True, text=True)
        print("✅ Virtual environment created successfully")
        
        return str(venv_path)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        print(e.stderr)
        return None
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return None

def get_venv_python():
    """Get the Python executable path from virtual environment"""
    venv_path = Path("./venv")
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        python_exe = venv_path / "bin" / "python"
    
    if python_exe.exists():
        return str(python_exe)
    else:
        print(f"⚠️  Virtual environment Python not found at: {python_exe}")
        return sys.executable

def setup_proxy_environment():
    """Setup Intel proxy environment variables for package installation"""
    proxy_settings = {
        "http_proxy": "http://proxy-dmz.intel.com:912",
        "https_proxy": "http://proxy-dmz.intel.com:912",
        "HTTP_PROXY": "http://proxy-dmz.intel.com:912",
        "HTTPS_PROXY": "http://proxy-dmz.intel.com:912",
        "no_proxy": ".intel.com,intel.com,localhost,127.0.0.1",
        "NO_PROXY": ".intel.com,intel.com,localhost,127.0.0.1"
    }
    
    print("🔄 Setting up Intel proxy environment for package installation...")
    
    for key, value in proxy_settings.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")
    
    # Also configure pip proxy settings
    pip_config_dir = Path.home() / ".pip"
    pip_config_dir.mkdir(exist_ok=True)
    
    pip_config_file = pip_config_dir / "pip.conf"
    pip_config_content = f"""[global]
proxy = http://proxy-dmz.intel.com:912
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
               download.pytorch.org
               huggingface.co
"""
    
    try:
        with open(pip_config_file, "w", encoding='utf-8') as f:
            f.write(pip_config_content)
        print(f"✅ Created pip config at: {pip_config_file}")
    except Exception as e:
        print(f"⚠️  Could not create pip config: {e}")
    
    print("✅ Proxy environment configured for Intel network")
    return True

def setup_openvino_environment():
    """Setup OpenVINO environment variables"""
    openvino_setupvars = r"C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat"
    
    if not Path(openvino_setupvars).exists():
        print(f"⚠️  OpenVINO setupvars.bat not found at: {openvino_setupvars}")
        print("Please ensure OpenVINO is properly installed or update the path in setup.py")
        return False
    
    print(f"🔄 Setting up OpenVINO environment from: {openvino_setupvars}")
    
    # Create a batch script that calls setupvars and then python
    batch_script = """
@echo off
call "{}"
echo OpenVINO environment variables set:
echo INTEL_OPENVINO_DIR=%INTEL_OPENVINO_DIR%
echo PATH=%PATH%
""".format(openvino_setupvars)
    
    try:
        with open("temp_setup_openvino.bat", "w", encoding='utf-8') as f:
            f.write(batch_script)
        
        result = subprocess.run("temp_setup_openvino.bat", shell=True, check=True, capture_output=True, text=True)
        print("✅ OpenVINO environment setup completed")
        print(result.stdout)
        
        # Clean up temporary file
        os.remove("temp_setup_openvino.bat")
        
        # Set environment variables in current process
        os.environ["INTEL_OPENVINO_DIR"] = r"C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64"
        openvino_bin = os.path.join(os.environ["INTEL_OPENVINO_DIR"], "runtime", "bin", "intel64", "Release")
        if openvino_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = openvino_bin + ";" + os.environ.get("PATH", "")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up OpenVINO environment: {e}")
        if os.path.exists("temp_setup_openvino.bat"):
            os.remove("temp_setup_openvino.bat")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python 3.8+ required, found Python {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install required packages in virtual environment"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Get virtual environment Python executable
    venv_python = get_venv_python()
    
    print(f"Using Python: {venv_python}")
    
    # Upgrade pip first in virtual environment
    run_command(f'"{venv_python}" -m pip install --upgrade pip', "Upgrading pip in virtual environment")
    
    # Install requirements in virtual environment
    return run_command(
        f'"{venv_python}" -m pip install -r requirements.txt', 
        "Installing requirements in virtual environment"
    )

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🔄 Testing package imports...")
    
    required_packages = [
        ("openvino", "OpenVINO"),
        ("optimum.intel.openvino", "Optimum Intel"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy")
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0

def check_openvino_devices():
    """Check available OpenVINO devices"""
    print("\n🔄 Checking OpenVINO devices...")
    
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        
        print(f"Available devices: {devices}")
        
        if "GPU" in devices:
            print("✅ Intel GPU detected - ready for acceleration!")
            
            # Get GPU info
            try:
                gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
                print(f"GPU: {gpu_name}")
            except:
                print("GPU detected but couldn't get detailed info")
                
        else:
            print("⚠️  No GPU detected, will use CPU")
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking OpenVINO devices: {e}")
        return False

def create_test_script():
    """Create a simple test script"""
    test_script = """
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
"""
    
    with open("test_setup.py", "w", encoding='utf-8') as f:
        f.write(test_script)
    
    print("✅ Test script created: test_setup.py")

def main():
    """Main setup function"""
    print("🚀 Stable Diffusion with OpenVINO Setup")
    print("This script will set up your environment for running Stable Diffusion 1.5 with OpenVINO optimization")
    print(f"Project directory: {Path.cwd().absolute()}")
    
    # Setup proxy environment for Intel network
    setup_proxy_environment()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    if not venv_path:
        print("❌ Failed to create virtual environment")
        sys.exit(1)
    
    # Setup OpenVINO environment
    if not setup_openvino_environment():
        print("⚠️  OpenVINO environment setup failed, but continuing...")
        print("You may need to manually run the setupvars.bat before using the scripts")
    
    # Install requirements in virtual environment (with proxy settings)
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your internet connection and proxy settings.")
        sys.exit(1)
    
    # Test imports (this will test in current environment)
    print("\n🔄 Testing package imports in virtual environment...")
    venv_python = get_venv_python()
    test_script = '''
import sys
try:
    packages = [
        ("openvino", "OpenVINO"),
        ("optimum.intel.openvino", "Optimum Intel"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy")
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"[SUCCESS] {name} imported successfully")
        except ImportError as e:
            print(f"[ERROR] {name} import failed: {e}")
            failed.append(name)
    
    if failed:
        print(f"Failed imports: {failed}")
        sys.exit(1)
    else:
        print("All imports successful!")
        
except Exception as e:
    print(f"Error during import test: {e}")
    sys.exit(1)
'''
    
    with open("temp_test_imports.py", "w", encoding='utf-8') as f:
        f.write(test_script)
    
    import_test_success = run_command(f'"{venv_python}" temp_test_imports.py', "Testing imports in virtual environment")
    os.remove("temp_test_imports.py")
    
    if not import_test_success:
        print("❌ Some packages failed to import in virtual environment.")
        print("Check the error messages above.")
    
    # Check OpenVINO devices using virtual environment
    device_test_script = '''
try:
    import openvino as ov
    core = ov.Core()
    devices = core.available_devices
    
    print(f"Available OpenVINO devices: {devices}")
    
    if "GPU" in devices:
        print("[SUCCESS] Intel GPU detected - ready for acceleration!")
        try:
            gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            print(f"GPU: {gpu_name}")
        except:
            print("GPU detected but couldn't get detailed info")
    else:
        print("[WARNING] No GPU detected, will use CPU")
        
except Exception as e:
    print(f"[ERROR] Error checking OpenVINO devices: {e}")
'''
    
    with open("temp_device_check.py", "w", encoding='utf-8') as f:
        f.write(device_test_script)
    
    run_command(f'"{venv_python}" temp_device_check.py', "Checking OpenVINO devices")
    os.remove("temp_device_check.py")
    
    # Create directories
    Path("./models").mkdir(exist_ok=True)
    Path("./outputs").mkdir(exist_ok=True)
    print("✅ Created models and outputs directories")
    
    # Create test script
    create_test_script()
    
    print("\n🎉 Setup completed successfully!")
    print(f"\n📁 Virtual environment created at: {Path(venv_path).absolute()}")
    print("✅ All packages installed in isolated virtual environment")
    print("✅ Intel proxy settings configured")
    print("\nIMPORTANT: The scripts will automatically use the virtual environment")
    print("when run through the provided batch files.")
    print("\nNext steps:")
    print("1. Run the main script:")
    print("   run_diffusion.bat --prompt 'your prompt here'")
    print("2. Or test the setup:")
    print("   test_with_openvino.bat")
    print("3. For direct Python usage:")
    print(f'   "{get_venv_python()}" stable_diffusion_openvino.py --help')

if __name__ == "__main__":
    main()
