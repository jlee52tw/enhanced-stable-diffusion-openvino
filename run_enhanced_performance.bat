@echo off
REM Enhanced Performance Test with Proxy Configuration
REM Run OpenVINO GenAI Stable Diffusion benchmarks

echo Setting up Intel proxy configuration...
set http_proxy=http://proxy-dmz.intel.com:912
set https_proxy=http://proxy-dmz.intel.com:912
set HTTP_PROXY=http://proxy-dmz.intel.com:912
set HTTPS_PROXY=http://proxy-dmz.intel.com:912

REM Setup cache directories
set HF_HUB_CACHE=.\huggingface_cache
set TRANSFORMERS_CACHE=.\huggingface_cache
set HUGGINGFACE_HUB_CACHE=.\huggingface_cache

REM Setup OpenVINO environment
if exist "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat" (
    call "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat"
)

echo.
echo ==================================================
echo  Enhanced Stable Diffusion Performance Test
echo ==================================================
echo.

REM Check if virtual environment exists
if not exist venv\Scripts\activate.bat (
    echo ❌ Virtual environment not found! Please run setup_with_proxy.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if OpenVINO model exists
set MODEL_DIR=models\stable-diffusion-v1-5_ov
if not exist "%MODEL_DIR%\openvino_model.xml" (
    echo ❌ OpenVINO model not found at: %MODEL_DIR%
    echo.
    echo Please run one of:
    echo 1. setup_with_proxy.bat  - Complete setup with model download
    echo 2. python download_model_manual.py - Manual download if automatic fails
    echo.
    pause
    exit /b 1
)

echo ✓ OpenVINO model found at: %MODEL_DIR%
echo.

echo 🚀 Running Enhanced Performance Benchmark...
echo Target: 512x512, 25 steps, 3-4 seconds per image
echo Devices to test: NPU, GPU, CPU (with fallback)
echo.

REM Run the enhanced performance script
python enhanced_performance_sd.py --benchmark --device AUTO --num-warmup 2 --num-runs 5 --width 512 --height 512 --steps 25

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Performance test failed!
    echo.
    echo Troubleshooting:
    echo 1. Check model exists: dir %MODEL_DIR%
    echo 2. Verify device support: python -c "import openvino as ov; print(ov.Core().available_devices)"
    echo 3. Try manual test: python enhanced_performance_sd.py --device CPU
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Performance test completed successfully!
echo Check enhanced_benchmark_report.json for detailed results
echo.

REM Show quick summary if report exists
if exist enhanced_benchmark_report.json (
    echo Quick Summary:
    python -c "import json; data=json.load(open('enhanced_benchmark_report.json')); print(f'Average time: {data.get(\"summary\", {}).get(\"avg_total_time\", \"N/A\"):.2f}s'); print(f'Device: {data.get(\"device_info\", {}).get(\"device\", \"Unknown\")}')"
)

echo.
pause
