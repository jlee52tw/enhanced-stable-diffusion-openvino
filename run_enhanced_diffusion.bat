@echo off
REM Enhanced Stable Diffusion with OpenVINO GenAI Runner

echo Setting up environment...

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
    echo Please check if the OpenVINO path is correct
    pause
    exit /b 1
)

echo OpenVINO environment ready.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found!
    echo Please run setup_with_openvino.bat first to create the virtual environment
    pause
    exit /b 1
)

echo Using virtual environment Python: %~dp0venv\Scripts\python.exe
echo.

REM Check if arguments were provided
if "%~1"=="" (
    echo Usage: run_enhanced_diffusion.bat --prompt "your prompt here" [additional options]
    echo.
    echo Examples:
    echo   Text-to-Image:
    echo   run_enhanced_diffusion.bat --prompt "A beautiful sunset over mountains" --steps 25
    echo.
    echo   Image-to-Image:
    echo   run_enhanced_diffusion.bat --mode img2img --prompt "watercolor painting" --input_image input.jpg
    echo.
    echo Running with default example prompt...
    "%~dp0venv\Scripts\python.exe" enhanced_stable_diffusion.py
) else (
    echo Running Enhanced Stable Diffusion with provided arguments...
    "%~dp0venv\Scripts\python.exe" enhanced_stable_diffusion.py %*
)

if errorlevel 1 (
    echo Error occurred during generation
    pause
    exit /b 1
)

echo Generation completed!
pause
