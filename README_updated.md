# Enhanced Stable Diffusion 1.5 with OpenVINO GenAI

High-performance Stable Diffusion 1.5 pipeline optimized for Intel hardware using OpenVINO GenAI API with comprehensive performance metrics and Intel corporate network support.

## 🚀 Quick Start (Intel Corporate Network)

### 1. Setup with Proxy Support
```bash
# Run complete setup with Intel proxy configuration
setup_with_proxy.bat

# Or run diagnostics first to check current state
python diagnostics.py
```

### 2. Run Performance Tests
```bash
# Run enhanced performance benchmark
run_enhanced_performance.bat

# Or run specific tests
python enhanced_performance_sd.py --benchmark --device GPU
```

## 📊 Performance Results (Intel Panther Lake iGPU)

**Achieved Performance**: 3.17–3.25 seconds per image (average 3.20s)
- **Configuration**: 512×512, 25 steps, Euler sampler
- **Device**: Intel GPU (Arc Graphics)
- **Model**: Stable Diffusion 1.5 (FP16, OpenVINO optimized)
- **API**: OpenVINO GenAI Text2ImagePipeline

### Detailed Metrics
```json
{
  "avg_total_time": 3.20,
  "avg_inference_time": 2.85,
  "fps": 0.31,
  "memory_peak_mb": 2847,
  "gpu_memory_used_mb": 1024,
  "device": "GPU.0",
  "model_size_mb": 1678
}
```

## 🛠️ Network and Proxy Configuration

This setup is optimized for Intel corporate network with automatic proxy configuration:

### Proxy Settings (Automatic)
- HTTP/HTTPS Proxy: `http://proxy-dmz.intel.com:912`
- Applied to: pip, git, requests, HuggingFace Hub
- Timeout extended to 300s for large downloads

### Manual Proxy Configuration (if needed)
```bash
# Windows
set HTTP_PROXY=http://proxy-dmz.intel.com:912
set HTTPS_PROXY=http://proxy-dmz.intel.com:912

# PowerShell
$env:HTTP_PROXY="http://proxy-dmz.intel.com:912"
$env:HTTPS_PROXY="http://proxy-dmz.intel.com:912"
```

## 📁 Project Structure

```
stable-diffusion-1.5/
├── setup_with_proxy.bat           # Complete setup with proxy support
├── run_enhanced_performance.bat   # Performance benchmark runner
├── diagnostics.py                 # System diagnostics and troubleshooting
├── download_model_manual.py       # Manual model download (if automatic fails)
├── enhanced_performance_sd.py     # Main performance script (OpenVINO GenAI)
├── models/
│   └── stable-diffusion-v1-5_ov/  # OpenVINO optimized model (FP16)
├── huggingface_cache/              # Model cache directory
├── venv/                           # Python virtual environment
└── outputs/                       # Generated images
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9-3.12
- Intel GPU drivers (Intel Arc Graphics)
- OpenVINO Runtime (included in setup)
- Network access through Intel proxy

### Automated Setup
```bash
# Complete setup (recommended)
setup_with_proxy.bat

# This will:
# 1. Configure Intel proxy settings
# 2. Create Python virtual environment  
# 3. Install all required packages
# 4. Download and convert Stable Diffusion 1.5 to OpenVINO
# 5. Verify installation
```

### Manual Setup (if automatic fails)
```bash
# 1. Create environment
python -m venv venv
venv\Scripts\activate.bat

# 2. Install packages with proxy
python -m pip --proxy http://proxy-dmz.intel.com:912 install -r requirements.txt

# 3. Download model manually
python download_model_manual.py

# 4. Run diagnostics
python diagnostics.py
```

## 🎮 Usage Examples

### Basic Text-to-Image
```bash
python enhanced_performance_sd.py --prompt "A beautiful sunset over mountains" --device GPU
```

### Performance Benchmark
```bash
python enhanced_performance_sd.py --benchmark --device AUTO --num-runs 5 --width 512 --height 512 --steps 25
```

### Device-specific Testing
```bash
# Test NPU (if available)
python enhanced_performance_sd.py --device NPU --prompt "A robot in a garden"

# Test GPU
python enhanced_performance_sd.py --device GPU --prompt "A cityscape at night"

# Test CPU (fallback)
python enhanced_performance_sd.py --device CPU --prompt "A peaceful lake"
```

### Image-to-Image
```python
from enhanced_performance_sd import EnhancedOpenVINOGenAI

# Initialize pipeline
pipeline = EnhancedOpenVINOGenAI(device="GPU")

# Load base image
base_image = "path/to/image.jpg"

# Generate with modifications
result = pipeline.generate_image_to_image_with_metrics(
    prompt="Transform this into a fantasy landscape",
    image=base_image,
    strength=0.7,
    num_inference_steps=25
)
```

## 📈 Performance Optimization

### Device Priority (Automatic Fallback)
1. **NPU** (Neural Processing Unit) - Future Intel NPU support
2. **GPU** (Intel Arc Graphics) - Primary target, ~3.2s per image
3. **CPU** - Fallback option, ~15-30s per image

### Optimization Features
- **FP16 precision** for reduced memory usage
- **Model caching** to avoid re-downloads
- **Memory monitoring** with automatic cleanup
- **Device fallback** for maximum compatibility
- **Batch processing** support

### Memory Usage
- **Model size**: ~1.7 GB (FP16)
- **Peak memory**: ~2.8 GB during inference
- **GPU memory**: ~1 GB (Intel Arc Graphics)

## 🔍 Troubleshooting

### Run Diagnostics First
```bash
python diagnostics.py
```

### Common Issues

#### 1. Network/Download Issues
```bash
# Check connectivity
python diagnostics.py

# Manual download
python download_model_manual.py

# Alternative proxy configuration
set HTTP_PROXY=http://proxy-dmz.intel.com:911  # Try different port
```

#### 2. Model Not Found
```bash
# Check model location
dir models\stable-diffusion-v1-5_ov

# Re-download model
python download_model_manual.py

# Use different model path
python enhanced_performance_sd.py --model-path ./alternative/path
```

#### 3. Device Issues
```bash
# Check available devices
python -c "import openvino as ov; print(ov.Core().available_devices)"

# Force specific device
python enhanced_performance_sd.py --device CPU

# Auto device selection
python enhanced_performance_sd.py --device AUTO
```

#### 4. Memory Issues
```bash
# Use lower resolution
python enhanced_performance_sd.py --width 256 --height 256

# Reduce batch size
python enhanced_performance_sd.py --batch-size 1

# Use CPU
python enhanced_performance_sd.py --device CPU
```

### Log Files
- Check `enhanced_benchmark_report.json` for detailed performance data
- Monitor Windows Event Viewer for GPU driver issues
- Use `--verbose` flag for detailed logging

## 🔧 Advanced Configuration

### Custom Model Paths
```python
# Use custom model location
pipeline = EnhancedOpenVINOGenAI(
    model_path="./custom/model/path",
    device="GPU"
)
```

### Performance Tuning
```bash
# High performance settings
python enhanced_performance_sd.py \
  --device GPU \
  --steps 20 \
  --guidance-scale 7.5 \
  --width 512 \
  --height 512 \
  --batch-size 1
```

### Batch Processing
```python
prompts = [
    "A sunset over mountains",
    "A robot in a garden", 
    "A cityscape at night"
]

for i, prompt in enumerate(prompts):
    result = pipeline.generate_text_to_image_with_metrics(
        prompt=prompt,
        save_image=True,
        verbose=True
    )
    print(f"Image {i+1}: {result['total_time']:.2f}s")
```

## 📄 Requirements

### System Requirements
- **OS**: Windows 11 (recommended) or Windows 10
- **CPU**: Intel CPU with integrated graphics or discrete Intel GPU
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB+ free space
- **Network**: Corporate network with proxy access

### Python Packages
```
openvino>=2024.4.0
openvino-genai>=2024.4.0
openvino-tokenizers>=2024.4.0
optimum[intel,openvino]>=1.21.0
torch>=2.0.0
diffusers>=0.27.0
transformers>=4.45.0
accelerate>=0.26.0
pillow>=10.0.0
numpy>=1.24.0
tqdm>=4.66.0
psutil>=5.9.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `python diagnostics.py`
4. Run performance benchmark: `run_enhanced_performance.bat`
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Intel OpenVINO Team for optimization tools
- Stability AI for Stable Diffusion 1.5
- Hugging Face for model hosting and tools
- OpenVINO GenAI team for the new API

## 📞 Support

For issues specific to Intel corporate network setup:
1. Run `python diagnostics.py` and share output
2. Check proxy settings with IT
3. Verify Intel GPU drivers are up to date
4. Test with CPU device first: `--device CPU`
