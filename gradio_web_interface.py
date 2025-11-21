#!/usr/bin/env python3
"""
Gradio Web Interface for Enhanced Stable Diffusion
Based on OpenVINO Notebooks gradio_helper.py
"""

import gradio as gr
import numpy as np
from PIL import Image
import time
import openvino_genai as ov_genai
from enhanced_stable_diffusion import OpenVINOGenAIStableDiffusion


def make_text2img_demo(pipeline):
    """Create Gradio interface for Text-to-Image generation"""
    
    def generate_text2img(prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, num_images):
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        try:
            # Use the enhanced pipeline
            images = pipeline.generate_text_to_image(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed if seed >= 0 else None,
                num_images=num_images,
                show_progress=False  # Disable progress bar for web interface
            )
            
            if images:
                # Return first image and success message
                return images[0], f"Successfully generated {len(images)} image(s)!"
            else:
                return None, "Generation failed"
                
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="Stable Diffusion Text-to-Image") as demo:
        gr.Markdown("# 🎨 Stable Diffusion Text-to-Image with OpenVINO")
        gr.Markdown("Generate images from text descriptions using OpenVINO optimized Stable Diffusion")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="A beautiful landscape with mountains and a lake at sunset, digital art",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted",
                    lines=2
                )
                
                with gr.Row():
                    num_steps = gr.Slider(1, 50, value=20, step=1, label="Inference Steps")
                    guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                
                with gr.Row():
                    width = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
                
                generate_btn = gr.Button("🎨 Generate Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        # Example prompts
        gr.Examples(
            examples=[
                ["A serene lake surrounded by mountains at sunset, digital art", "", 20, 7.5, 512, 512, 42, 1],
                ["A cyberpunk cityscape at night with neon lights, highly detailed", "blurry, low quality", 25, 8.0, 512, 512, 123, 1],
                ["A cute robot in a futuristic city, 3D render", "", 20, 7.5, 512, 512, 456, 1],
                ["Portrait of a wise old wizard with a long beard, fantasy art", "cartoon, anime", 30, 8.5, 512, 768, 789, 1],
            ],
            inputs=[prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, num_images]
        )
        
        generate_btn.click(
            generate_text2img,
            inputs=[prompt, negative_prompt, num_steps, guidance_scale, width, height, seed, num_images],
            outputs=[output_image, status_text]
        )
    
    return demo


def make_img2img_demo(pipeline):
    """Create Gradio interface for Image-to-Image generation"""
    
    def generate_img2img(input_image, prompt, negative_prompt, num_steps, guidance_scale, strength, seed):
        if input_image is None:
            return None, "Please upload an input image"
        if not prompt.strip():
            return None, "Please enter a prompt"
        
        try:
            # Use the enhanced pipeline
            images = pipeline.generate_image_to_image(
                prompt=prompt,
                image=input_image,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed if seed >= 0 else None,
                show_progress=False  # Disable progress bar for web interface
            )
            
            if images:
                return images[0], "Image transformation completed successfully!"
            else:
                return None, "Generation failed"
                
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="Stable Diffusion Image-to-Image") as demo:
        gr.Markdown("# 🖼️ Stable Diffusion Image-to-Image with OpenVINO")
        gr.Markdown("Transform existing images with text descriptions using OpenVINO optimized Stable Diffusion")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="amazing watercolor painting",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted",
                    lines=2
                )
                
                with gr.Row():
                    num_steps = gr.Slider(1, 50, value=20, step=1, label="Inference Steps")
                    guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                
                with gr.Row():
                    strength = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Strength")
                    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                generate_btn = gr.Button("🎨 Transform Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Transformed Image", type="pil")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        generate_btn.click(
            generate_img2img,
            inputs=[input_image, prompt, negative_prompt, num_steps, guidance_scale, strength, seed],
            outputs=[output_image, status_text]
        )
    
    return demo


def make_combined_demo(pipeline):
    """Create combined Gradio interface with both text-to-image and image-to-image"""
    
    text2img_demo = make_text2img_demo(pipeline)
    img2img_demo = make_img2img_demo(pipeline)
    
    # Create tabbed interface
    with gr.Blocks(title="Enhanced Stable Diffusion with OpenVINO") as demo:
        gr.Markdown("# 🚀 Enhanced Stable Diffusion with OpenVINO")
        gr.Markdown("AI-powered image generation optimized for Intel hardware")
        
        with gr.Tabs():
            with gr.TabItem("Text-to-Image"):
                # Embed the text2img demo content
                text2img_interface = make_text2img_demo(pipeline)
                
            with gr.TabItem("Image-to-Image"):
                # Embed the img2img demo content
                img2img_interface = make_img2img_demo(pipeline)
        
        gr.Markdown("""
        ### 💡 Tips:
        - **Text-to-Image**: Create images from scratch using detailed text descriptions
        - **Image-to-Image**: Transform existing images based on text prompts
        - **Steps**: More steps = higher quality but slower generation (20-30 recommended)
        - **Guidance Scale**: Higher values follow the prompt more closely (7-8 recommended)
        - **Strength** (Image-to-Image): How much to change the input image (0.5-0.8 recommended)
        """)
    
    return demo


def main():
    """Launch the Gradio web interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Stable Diffusion Web Interface")
    parser.add_argument("--model_id", type=str, default="prompthero/openjourney", help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="GPU", help="OpenVINO device")
    parser.add_argument("--model_cache", type=str, default="./models", help="Model cache directory")
    parser.add_argument("--model_precision", type=str, default="fp16", help="Model precision")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    
    args = parser.parse_args()
    
    print("🚀 Starting Enhanced Stable Diffusion Web Interface...")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    
    # Initialize pipeline
    pipeline = OpenVINOGenAIStableDiffusion(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.model_cache,
        model_precision=args.model_precision
    )
    
    # Create and launch demo
    demo = make_combined_demo(pipeline)
    
    print(f"🌐 Launching web interface on port {args.port}")
    if args.share:
        print("🔗 Creating public shareable link...")
    
    try:
        demo.queue().launch(
            server_port=args.port,
            share=args.share,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Web interface stopped by user")
    except Exception as e:
        print(f"❌ Error launching web interface: {e}")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
