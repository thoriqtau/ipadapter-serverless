import os
import sys
import base64
import time
import json
from io import BytesIO
import io
import logging
import shutil
import gc

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("handler")

logger.info("=" * 40)
logger.info("RUNPOD HANDLER STARTING")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working dir: {os.getcwd()}")
logger.info("=" * 40)

# Import runpod
try:
    import runpod
    logger.info("Successfully imported runpod")
except Exception as e:
    logger.error(f"Error importing runpod: {e}")
    raise

# Global model object
model = None

def load_model():
    """Load Stable Diffusion XL + IP-Adapter using temporary cache."""
    global model
    if model is not None:
        logger.info("Model already loaded, reusing...")
        return model

    try:
        import torch
        from diffusers import AutoPipelineForText2Image, DDIMScheduler
        from transformers import CLIPVisionModelWithProjection

        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        logger.info("Loading CLIP image encoder from h94/IP-Adapter...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            cache_dir="/tmp/ip_adapter_encoder"
        )

        logger.info("Loading Stable Diffusion XL base...")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            image_encoder=image_encoder,
            cache_dir="/tmp/sdxl_model"
        )

        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

        logger.info("Loading IP-Adapter weights...")
        pipeline.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name=[
                "ip-adapter-plus_sdxl_vit-h.safetensors",
                "ip-adapter-plus-face_sdxl_vit-h.safetensors"
            ],
            cache_dir="/tmp/ip_adapter_weights"
        )

        pipeline.set_ip_adapter_scale([0.7, 0.3])
        pipeline.enable_model_cpu_offload()
        pipeline = pipeline.to(device)

        if device == "cuda":
            pipeline.enable_attention_slicing()

        logger.info("Model loaded successfully.")
        model = pipeline
        return model

    except Exception as e:
        logger.exception("Failed to load model")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def to_base64_string(image):
    """Convert PIL image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def log_disk_space():
    total, used, free = shutil.disk_usage("/")
    logger.info(f"Disk: Total {total//2**30} GB | Used {used//2**30} GB | Free {free//2**30} GB")

def clean_tmp_and_memory():
    """Clean /tmp and free GPU/CPU memory."""
    try:
        logger.info("Cleaning memory and /tmp...")
        if os.path.exists("/tmp"):
            for f in os.listdir("/tmp"):
                f_path = os.path.join("/tmp", f)
                if os.path.isfile(f_path) or os.path.islink(f_path):
                    os.unlink(f_path)
                elif os.path.isdir(f_path):
                    shutil.rmtree(f_path, ignore_errors=True)

        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup complete.")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

def handler(event):
    """Main RunPod handler."""
    try:
        logger.info(f"Received event with keys: {list(event.keys()) if isinstance(event, dict) else type(event)}")

        input_data = event.get("input", {})
        prompt = input_data.get("prompt", "wonderwoman")
        ip_adapter_image = input_data.get("ip_adapter_image", None)
        negative_prompt = input_data.get("negative_prompt", "monochrome, lowres, bad anatomy, worst quality, low quality")

        log_disk_space()

        # Load model
        global model
        if model is None:
            model = load_model()

        # Load image from base64 (IP Adapter input)
        loaded_images = []
        if ip_adapter_image:
            for image_data in ip_adapter_image:
                image_bytes = base64.b64decode(image_data)
                loaded_images.append(io.BytesIO(image_bytes))

        logger.info(f"Generating image for prompt: '{prompt}'")
        start_time = time.time()

        result = model(
            prompt=prompt,
            ip_adapter_image=loaded_images,
            negative_prompt=negative_prompt
        )

        gen_time = time.time() - start_time
        logger.info(f"Image generated in {gen_time:.2f} seconds")

        # Encode result to base64
        image_base64 = to_base64_string(result.images[0])

        # Cleanup to prevent storage overflow
        clean_tmp_and_memory()

        return {
            "image": image_base64,
            "metrics": {
                "generation_time": gen_time
            }
        }

    except Exception as e:
        logger.exception("Error during handler execution")
        return {"error": f"Error generating image: {str(e)}"}

# Run the serverless entry
logger.info("Starting RunPod serverless handler")
try:
    runpod.serverless.start({"handler": handler})
except Exception as e:
    logger.exception("Failed to start serverless handler")
