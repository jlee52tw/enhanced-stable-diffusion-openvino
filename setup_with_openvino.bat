@echo off
echo Setting up OpenVINO environment and running Stable Diffusion setup...

REM Set up Intel proxy environment variables
echo Setting up Intel proxy settings...
set http_proxy=http://proxy-dmz.intel.com:912
set https_proxy=http://proxy-dmz.intel.com:912
set HTTP_PROXY=http://proxy-dmz.intel.com:912
set HTTPS_PROXY=http://proxy-dmz.intel.com:912
set no_proxy=.intel.com,intel.com,localhost,127.0.0.1
set NO_PROXY=.intel.com,intel.com,localhost,127.0.0.1
echo Proxy settings configured for Intel network.

REM Set up OpenVINO environment
call "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat"

if errorlevel 1 (
    echo Error: Failed to setup OpenVINO environment
    pause
    exit /b 1
)

echo OpenVINO environment setup completed.
echo.

REM Run the Python setup script
python setup.py

if errorlevel 1 (
    echo Error: Python setup failed
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo To run Stable Diffusion generation:
echo   run_diffusion.bat --prompt "your prompt here"
echo.
pause
