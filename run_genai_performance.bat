@echo off
REM Setup and run OpenVINO GenAI Performance Test
REM This script converts the model to OpenVINO format first, then runs performance tests

echo Setting up Intel proxy and OpenVINO environment...
set http_proxy=http://proxy-dmz.intel.com:912
set https_proxy=http://proxy-dmz.intel.com:912
set HTTP_PROXY=http://proxy-dmz.intel.com:912
set HTTPS_PROXY=http://proxy-dmz.intel.com:912

REM Setup OpenVINO environment
call "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat"

echo.
echo ==================================================
echo  OpenVINO GenAI Performance Testing Setup
echo ==================================================
echo.

REM Check if model already exists in OpenVINO format
set MODEL_DIR=models\stable-diffusion-v1-5_ov
if exist "%MODEL_DIR%\openvino_model.xml" (
    echo ✓ OpenVINO model already exists at: %MODEL_DIR%
    goto run_test
)

echo 🔄 Converting Stable Diffusion model to OpenVINO format...
echo This may take several minutes on first run...

REM Create models directory
mkdir models 2>nul

REM Convert model using optimum-cli
echo Running: optimum-cli export openvino --model runwayml/stable-diffusion-v1-5 --task text-to-image --weight-format fp16 %MODEL_DIR%
"venv\Scripts\optimum-cli.exe" export openvino --model runwayml/stable-diffusion-v1-5 --task text-to-image --weight-format fp16 %MODEL_DIR%

if %ERRORLEVEL% neq 0 (
    echo ❌ Model conversion failed!
    pause
    exit /b 1
)

echo ✅ Model conversion completed successfully!

:run_test
echo.
echo 🚀 Running OpenVINO GenAI Performance Test...
echo Target: 512x512, 25 steps, Euler sampler in 3-4 seconds
echo.

REM Run performance test with converted model
"venv\Scripts\python.exe" genai_performance_diffusion.py ^
  --model-id "%MODEL_DIR%" ^
  --benchmark ^
  --benchmark-runs 3 ^
  --warmup-runs 1 ^
  --steps 25 ^
  --width 512 ^
  --height 512 ^
  --device GPU ^
  --save-report ^
  --prompt "a beautiful mountain landscape with a serene lake, digital art, highly detailed"

echo.
echo ✅ Performance test completed!
echo Check genai_benchmark_report.json for detailed results.
echo.
pause
