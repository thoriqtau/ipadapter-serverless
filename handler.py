import os
import sys
import base64
import time
import json
from io import BytesIO
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alternative_handler")

# Print startup info
logger.info("=" * 40)
logger.info("ALTERNATIVE HANDLER STARTING")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info("=" * 40)

# Import RunPod with error handling
try:
    import runpod
    logger.info("Successfully imported runpod")
except Exception as e:
    logger.error(f"Error importing runpod: {e}")
    import traceback
    traceback.print_exc()
    raise

# Global variable to hold the model
model = None

def load_model():
    """Load an alternative, publicly available model (Stable Diffusion XL)"""
    global model
    
    # Already loaded check
    if model is not None:
        logger.info("Model already loaded, reusing existing model")
        return model
    
    try:
        # Import dependencies inside function to catch and report errors
        logger.info("Importing torch, diffusers, transformers, diffusers.utils ")
        import torch
        from diffusers import AutoPipelineForText2Image, DDIMScheduler
        from transformers import CLIPVisionModelWithProjection
        from diffusers.utils import load_image
        
        # Log environment info
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Model loading with clear progress indication
        logger.info("Starting model download and loading")
        logger.info("This may take several minutes on first run")
        
        start_time = time.time()
        logger.info("Loading model: 'h94/IP-Adapter'")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )
        
        logger.info("Loading model: 'stabilityai/stable-diffusion-xl-base-1.0'")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            image_encoder=image_encoder,
        )
        
        logger.info("Setting up DDIM scheduler")
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        
        logger.info("Setting up IP-Adapter")
        pipeline.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
        )
        
        logger.info("Setting up IP-Adapter scale")
        pipeline.set_ip_adapter_scale([0.7, 0.3])

        # enable_model_cpu_offload to reduce memory usage
        pipeline.enable_model_cpu_offload()
        
        # Move to device
        logger.info(f"Moving model to {device}")
        pipeline = pipeline.to(device)
        
        # Enable memory optimizations for CUDA
        if device == "cuda":
            logger.info("Enabling memory optimizations")
            pipeline.enable_attention_slicing()
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        model = pipeline
        return model

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        model = None
        raise RuntimeError(f"Failed to load model: {str(e)}")

def to_base64_string(image):
    """Convert a PIL image to a base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def handler(event):
    """Handle the serverless request to generate an image from text."""
    try:
        # Log the event, but omit potentially large data
        logger.info(f"Received event type: {type(event)}")
        if isinstance(event, dict):
            logger.info(f"Event keys: {list(event.keys())}")
            
        # Get input parameters from the request
        input_data = event.get("input", {})
        logger.info(f"Input data: {json.dumps(input_data)}")
        
        # Load model if not already loaded
        logger.info("Checking model status")
        global model
        
        if model is None:
            logger.info("Model not loaded, loading now...")
            model = load_model()
            if model is None:
                error_msg = "Failed to load model after multiple attempts"
                logger.error(error_msg)
                return {"error": error_msg}
        
        # Extract parameters with defaults        
        prompt = input_data.get("prompt", "wonderwoman")
        ip_adapter_image = input_data.get('ip_adapter_image', None)
        negative_prompt = input_data.get("negative_prompt", "monochrome, lowres, bad anatomy, worst quality, low quality")
        
        if ip_adapter_image:
            # Load images as BytesIO objects instead of PIL Images
            loaded_images = []
            for image_data in ip_adapter_image:
                image_bytes = base64.b64decode(image_data)
                loaded_images.append(io.BytesIO(image_bytes))

        # Generate image
        logger.info(f"Generating image with prompt: '{prompt}'")
        start_time = time.time()
        
        # Simple generation with fewer parameters to avoid errors
        result = model(
            prompt=prompt,
            ip_adapter_image=loaded_images,
            negative_prompt=negative_prompt
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Generated image in {generation_time:.2f} seconds")
        
        # Convert image to base64
        logger.info("Converting image to base64")
        image_data = to_base64_string(result.images[0])
        
        # Return the result
        return {
            "image": image_data,
            "metrics": {
                "generation_time": generation_time
            }
        }
    
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return {"error": error_msg}

# Startup message before starting the serverless function
logger.info("Starting runpod serverless with alternative handler")

# Start the serverless function
try:
    runpod.serverless.start({"handler": handler})
except Exception as e:
    logger.error(f"Failed to start serverless function: {e}")
    import traceback
    traceback.print_exc()