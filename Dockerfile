# ============================================================================
# FlashVSR+ Docker Image — Blackwell (RTX 6000 Pro / RTX 50xx) compatible
#
# Uses PyTorch nightly + CUDA 12.8 for sm_120 support.
# Uses uv (https://github.com/astral-sh/uv) instead of pip for faster,
# parallel dependency resolution and downloads.
#
# SageAttention: The local Triton kernels in src/models/sparse_sage/ are
# JIT-compiled at runtime on the user's actual GPU — no build-time CUDA
# compilation needed. Falls back to PyTorch SDPA if Triton fails.
#
# No stable PyTorch release supports Blackwell as of Feb 2026.
# See: https://github.com/pytorch/pytorch/issues/164342
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

ARG PYTHON_VERSION=3.12

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ARG PYTHON_VERSION

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment and make it the default Python
ENV VIRTUAL_ENV=/opt/venv
RUN uv venv --python python${PYTHON_VERSION} $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Install latest PyTorch nightly with CUDA 12.8 (sm_120 / Blackwell support)
RUN uv pip install --no-cache --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install remaining Python dependencies
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

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
