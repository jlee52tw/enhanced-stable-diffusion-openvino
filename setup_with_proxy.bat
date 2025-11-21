@echo off
REM Enhanced Setup Script with Intel Proxy Configuration
REM This script handles proxy configuration for Intel corporate network

echo Setting up Intel proxy configuration...
set http_proxy=http://proxy-dmz.intel.com:912
set https_proxy=http://proxy-dmz.intel.com:912
set HTTP_PROXY=http://proxy-dmz.intel.com:912
set HTTPS_PROXY=http://proxy-dmz.intel.com:912

REM Also set for HuggingFace
set HF_HUB_CACHE=.\huggingface_cache
set TRANSFORMERS_CACHE=.\huggingface_cache
set HUGGINGFACE_HUB_CACHE=.\huggingface_cache

REM Set Git proxy for any Git operations
git config --global http.proxy http://proxy-dmz.intel.com:912
git config --global https.proxy http://proxy-dmz.intel.com:912

echo ✓ Proxy configuration set for Intel corporate network
echo.

REM Setup OpenVINO environment
if exist "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat" (
    echo Setting up OpenVINO environment...
    call "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat"
    echo ✓ OpenVINO environment configured
) else (
    echo ⚠️ OpenVINO setupvars.bat not found, proceeding without it
)

echo.
echo ==================================================
echo  Enhanced Stable Diffusion Setup with Proxy
echo ==================================================
echo.

REM Check if virtual environment exists
if not exist venv\ (
    echo Creating Python virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo ❌ Failed to create virtual environment!
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 🔄 Installing/updating required packages with proxy...

REM Upgrade pip with proxy
python -m pip --proxy http://proxy-dmz.intel.com:912 install --upgrade pip

REM Install packages with proxy and increased timeout
echo Installing core packages...
python -m pip --proxy http://proxy-dmz.intel.com:912 install --timeout 300 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing OpenVINO packages...
python -m pip --proxy http://proxy-dmz.intel.com:912 install --timeout 300 openvino openvino-genai openvino-tokenizers

echo Installing other dependencies...
python -m pip --proxy http://proxy-dmz.intel.com:912 install --timeout 300 optimum[intel,openvino] diffusers transformers accelerate numpy pillow tqdm psutil

echo.
echo ✅ Package installation completed!

REM Check if model already exists
set MODEL_DIR=models\stable-diffusion-v1-5_ov
if exist "%MODEL_DIR%\openvino_model.xml" (
    echo ✅ OpenVINO model already exists at: %MODEL_DIR%
    echo You can run: run_enhanced_performance.bat
    echo.
    pause
    exit /b 0
)

echo.
echo 🔄 Downloading and converting Stable Diffusion model...
echo This may take 10-30 minutes depending on network speed...

REM Create models directory
mkdir models 2>nul

REM Use optimum-cli with proxy environment variables
echo Converting model with proxy settings...
venv\Scripts\optimum-cli.exe export openvino --model runwayml/stable-diffusion-v1-5 --task text-to-image --weight-format fp16 %MODEL_DIR% --cache-dir .\huggingface_cache

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Model conversion failed!
    echo.
    echo Troubleshooting steps:
    echo 1. Check your internet connection
    echo 2. Verify proxy settings: %HTTP_PROXY%
    echo 3. Try running: huggingface-cli login (if needed)
    echo 4. Manual download option available in download_model_manual.py
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Setup completed successfully!
echo.
echo Next steps:
echo 1. Run performance test: run_enhanced_performance.bat
echo 2. Or test basic functionality: python enhanced_performance_sd.py --benchmark
echo.
pause
