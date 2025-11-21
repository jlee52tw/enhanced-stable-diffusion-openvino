@echo off
REM Performance Testing Script for Stable Diffusion with OpenVINO
REM Runs comprehensive benchmarks to test 512x512, 25 steps target

echo Setting up Intel proxy and OpenVINO environment...
set http_proxy=http://proxy-dmz.intel.com:912
set https_proxy=http://proxy-dmz.intel.com:912
set HTTP_PROXY=http://proxy-dmz.intel.com:912
set HTTPS_PROXY=http://proxy-dmz.intel.com:912

REM Setup OpenVINO environment
call "C:\working\gpt-oss\openvino_genai_windows_2026.0.0.0.dev20251114_x86_64\setupvars.bat"

echo.
echo ==================================================
echo  Performance Testing - Stable Diffusion OpenVINO
echo ==================================================
echo.

REM Run comprehensive benchmark
echo Running comprehensive performance benchmark...
echo Target: 512x512, Euler sampler, 25 steps in 3-4 seconds
echo.

"venv\Scripts\python.exe" performance_diffusion.py ^
  --benchmark ^
  --benchmark-runs 5 ^
  --warmup-runs 2 ^
  --steps 25 ^
  --width 512 ^
  --height 512 ^
  --device GPU ^
  --save-report ^
  --prompt "a beautiful mountain landscape with a serene lake, digital art, highly detailed"

echo.
echo Performance test completed!
echo Check benchmark_report.json for detailed results.
echo.
pause
