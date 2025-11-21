@echo off
REM Activate virtual environment with OpenVINO setup

echo Setting up environment and activating virtual environment...

REM Set up Intel proxy environment variables
set http_proxy=http://proxy-dmz.intel.com:912
set https_proxy=http://proxy-dmz.intel.com:912
set HTTP_PROXY=http://proxy-dmz.intel.com:912
set HTTPS_PROXY=http://proxy-dmz.intel.com:912
set no_proxy=.intel.com,intel.com,localhost,127.0.0.1
set NO_PROXY=.intel.com,intel.com,localhost,127.0.0.1
echo Intel proxy settings configured.

REM Set up OpenVINO environment
call "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat"

if errorlevel 1 (
    echo Error: Failed to setup OpenVINO environment
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup_with_openvino.bat first to create the virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo ✅ Virtual environment activated with OpenVINO support and Intel proxy settings!
echo.
echo You can now run Python scripts directly:
echo   python stable_diffusion_openvino.py --prompt "your prompt"
echo   python examples.py
echo   python test_setup.py
echo.
echo To deactivate, type: deactivate
echo.
