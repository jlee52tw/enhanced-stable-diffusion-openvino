@echo off
REM Test Stable Diffusion setup with OpenVINO environment

echo Setting up environment for testing...

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
    echo Please run setup_with_openvino.bat first to create the virtual environment
    pause
    exit /b 1
)

echo Using virtual environment Python: %~dp0venv\Scripts\python.exe
echo Running test script...

"%~dp0venv\Scripts\python.exe" test_setup.py

if errorlevel 1 (
    echo Test failed
    pause
    exit /b 1
)

echo Test completed!
pause
