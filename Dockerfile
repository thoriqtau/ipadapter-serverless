FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install dependencies with explicit version pinning for reliability 
RUN pip3 install --upgrade pip && \ 
    echo "Installing dependencies..." && \ 
    pip3 install --no-cache-dir torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
                diffusers==0.34.0 \ 
                transformers==4.53.2 \ 
                accelerate==1.9.0 \ 
                runpod \ 
                pillow \ 
                safetensors \ 
                huggingface_hub==0.33.4 && \ 
    echo "Dependencies installed successfully" 

# Create working directory 
WORKDIR /app 

# Copy our handler 
COPY handler.py /app/handler.py 

# Set environment variables for better reliability 
ENV RUNPOD_DEBUG_MODE=true 
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface 
ENV HF_HUB_DOWNLOAD_TIMEOUT=600 
ENV SAFETENSORS_FAST_GPU=1 
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 

# Specifically disable hf_transfer 
ENV HF_HUB_ENABLE_HF_TRANSFER=0 

# Generate a welcome message so it shows in the logs 
RUN echo "Container setup complete - alternative_handler without hf_transfer will start on container launch" 

# RunPod will use the handler defined in handler.py 
CMD ["python", "-u", "/app/handler.py"] 
