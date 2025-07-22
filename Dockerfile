FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Install dependencies with version pinning
RUN pip3 install --upgrade pip && \
    echo "Installing dependencies..." && \
    pip3 install diffusers==0.21.4 \
                transformers==4.31.0 \
                accelerate==0.21.0 \
                runpod \
                pillow \
                safetensors \
                huggingface_hub==0.19.4 && \
    echo "Dependencies installed successfully" && \
    # Optional: Clean up pip cache to save space
    rm -rf ~/.cache/pip

# Create working directory
WORKDIR /app

# Copy handler
COPY handler.py /app/handler.py

# Environment variables
ENV RUNPOD_DEBUG_MODE=true
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
ENV SAFETENSORS_FAST_GPU=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV HF_HUB_ENABLE_HF_TRANSFER=0  # Disable fast-transfer for safety

# Message for build logs
RUN echo "Container setup complete - handler ready to run."

# Default command
CMD ["python", "-u", "/app/handler.py"]
