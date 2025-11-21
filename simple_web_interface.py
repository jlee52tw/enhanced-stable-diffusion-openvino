#!/usr/bin/env python3
"""
Simple Gradio Web Interface for Stable Diffusion with OpenVINO
Uses our pre-converted model for fast startup
"""

import gradio as gr
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Import our existing OpenVINO Stable Diffusion class
import sys
sys.path.append('.')
from stable_diffusion_openvino import OpenVINOStableDiffusion

def create_web_interface():
    """Create and launch the Gradio web interface"""
    
    print("🚀 Starting Simple Stable Diffusion Web Interface...")
    print("🎯 Using pre-converted OpenVINO model for fast startup")
    
    # Initialize the model once
    global sd_model
    try:
        sd_model = OpenVINOStableDiffusion(
            model_id="runwayml/stable-diffusion-v1-5",
            device="GPU"
        )
        print("✅ Model initialized successfully!")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        sd_model = None
    
    def generate_image(prompt, negative_prompt, steps, guidance_scale, width, height, seed):
        """Generate image using the initialized model"""
        if sd_model is None:
            return None, "❌ Model not initialized"
        
        try:
            print(f"🎨 Generating: {prompt}")
            start_time = time.time()
            
            # Generate image
            images = sd_model.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance_scale),
                width=int(width),
                height=int(height),
                seed=int(seed) if seed >= 0 else None,
                num_images=1
            )
            
            if images:
                # Save image
                output_dir = Path("./outputs/web_interface")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = int(time.time())
                filename = f"web_generated_{timestamp}.png"
                filepath = output_dir / filename
                
                images[0].save(filepath)
                
                generation_time = time.time() - start_time
                info = f"✅ Generated in {generation_time:.1f}s | Saved: {filepath.name}"
                
                print(info)
                return images[0], info
            else:
                return None, "❌ Generation failed"
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    # Create Gradio interface
    with gr.Blocks(title="Stable Diffusion with OpenVINO") as interface:
        gr.Markdown("# 🎨 Stable Diffusion 1.5 with OpenVINO")
        gr.Markdown("Optimized for Intel GPU acceleration")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3,
                    value="a beautiful sunset over a mountain lake, digital art"
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt (optional)",
                    placeholder="Enter negative prompt here...",
                    lines=2
                )
                
                with gr.Row():
                    steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                    guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                
                seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                output_info = gr.Textbox(label="Generation Info", lines=2)
        
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, steps, guidance_scale, width, height, seed],
            outputs=[output_image, output_info]
        )
        
        # Example prompts
        gr.Markdown("### Example Prompts:")
        examples = [
            "a beautiful sunset over a mountain lake, digital art",
            "a cozy cabin in winter snow, warm lights, photorealistic",
            "a futuristic city with flying cars, cyberpunk style",
            "a majestic dragon in a medieval castle, fantasy art",
            "a peaceful zen garden with cherry blossoms, traditional art"
        ]
        
        for example in examples:
            gr.Button(example, size="sm").click(
                lambda x=example: x,
                outputs=[prompt]
            )
    
    return interface

if __name__ == "__main__":
    # Clear proxy settings for local Gradio server
    if 'http_proxy' in os.environ:
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        del os.environ['https_proxy']
    if 'HTTP_PROXY' in os.environ:
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        del os.environ['HTTPS_PROXY']
    
    interface = create_web_interface()
    
    print("🌐 Launching web interface...")
    print("🔗 Access the interface in your browser once it starts")
    
    interface.launch(
        server_name="127.0.0.1",  # Localhost only to avoid proxy issues
        server_port=7860,         # Default Gradio port
        share=False,              # Don't create public link
        show_error=True,
        inbrowser=True,           # Open browser automatically
        quiet=True                # Reduce verbose output
    )
