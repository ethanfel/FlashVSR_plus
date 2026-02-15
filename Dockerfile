# ============================================================================
# FlashVSR+ Docker Image — Blackwell (RTX 6000 Pro / RTX 50xx) compatible
#
# Uses CUDA 12.8 devel base + PyTorch nightly (cu128) for sm_120 support.
# SageAttention is built from source since no stable pip wheel supports sm_120.
#
# Build:
#   docker build -t flashvsr-plus .
#
# Run (Gradio UI):
#   docker run --gpus all -p 7860:7860 -v flashvsr-models:/app/models flashvsr-plus
#
# Run (API server):
#   docker run --gpus all -p 8000:8000 -v flashvsr-models:/app/models \
#     flashvsr-plus python api.py
# ============================================================================

# Stage 1: build SageAttention from source (needs CUDA compiler)
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;12.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install PyTorch nightly with CUDA 12.8 (required for sm_120 / Blackwell)
RUN pip install --no-cache-dir --break-system-packages \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Build SageAttention from source with sm_120 support
RUN pip install --no-cache-dir --break-system-packages ninja setuptools wheel && \
    git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && \
    cd /tmp/sageattention && \
    python setup.py bdist_wheel && \
    cp dist/*.whl /tmp/sageattention.whl

# Stage 2: runtime image (smaller, no compiler toolchain)
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Install PyTorch nightly with CUDA 12.8 (sm_120 / Blackwell support)
RUN pip install --no-cache-dir --break-system-packages \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install SageAttention wheel built in stage 1
COPY --from=builder /tmp/sageattention.whl /tmp/sageattention.whl
RUN pip install --no-cache-dir --break-system-packages /tmp/sageattention.whl && \
    rm /tmp/sageattention.whl

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY . .

# Create directories for runtime
RUN mkdir -p /app/models /app/outputs /app/_temp

# Volumes for persistent data (models are large — mount externally)
VOLUME ["/app/models", "/app/outputs"]

# Expose ports: Gradio WebUI (7860) and FastAPI (8000)
EXPOSE 7860 8000

# Default: launch the Gradio WebUI with LAN access
CMD ["python", "webui.py", "--listen", "--port", "7860"]
