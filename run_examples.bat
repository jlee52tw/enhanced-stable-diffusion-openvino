@echo off
REM Run examples with OpenVINO environment and virtual environment

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

echo Running examples with virtual environment...
"%~dp0venv\Scripts\python.exe" examples.py

if errorlevel 1 (
    echo Examples failed
    pause
    exit /b 1
)

echo Examples completed!
pause
