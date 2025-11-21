@echo off
REM Launch Gradio Web Interface for Enhanced Stable Diffusion

echo Setting up environment for web interface...

REM Set up Intel proxy environment variables
set http_proxy=http://proxy-dmz.intel.com:912
set https_proxy=http://proxy-dmz.intel.com:912
set HTTP_PROXY=http://proxy-dmz.intel.com:912
set HTTPS_PROXY=http://proxy-dmz.intel.com:912
set no_proxy=.intel.com,intel.com,localhost,127.0.0.1
set NO_PROXY=.intel.com,intel.com,localhost,127.0.0.1

REM Set up OpenVINO environment
call "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat"

if errorlevel 1 (
    echo Error: Failed to setup OpenVINO environment
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found!
    echo Please run setup_with_openvino.bat first
    pause
    exit /b 1
)

echo Launching Enhanced Stable Diffusion Web Interface...
echo.
echo The web interface will be available at: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

"%~dp0venv\Scripts\python.exe" gradio_web_interface.py %*

if errorlevel 1 (
    echo Web interface failed to start
    pause
    exit /b 1
)

pause
