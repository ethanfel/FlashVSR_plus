FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.8 support first (large layer, cached separately)
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
COPY requirements.txt .
# Install everything except torch (already installed above)
RUN pip install --no-cache-dir \
    $(grep -v '^torch==' requirements.txt | grep -v '^torchvision==' | grep -v '^torchaudio==')

# Copy application code
COPY . .

# Create directories for runtime
RUN mkdir -p /app/models /app/outputs /app/_temp

# Volumes for persistent data
VOLUME ["/app/models", "/app/outputs"]

# Expose ports: Gradio WebUI (7860) and FastAPI (8000)
EXPOSE 7860 8000

# Default: launch the Gradio WebUI with LAN access enabled
CMD ["python", "webui.py", "--listen", "--port", "7860"]
