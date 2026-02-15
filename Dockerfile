# ============================================================================
# FlashVSR+ Docker Image — Blackwell (RTX 6000 Pro / RTX 50xx) compatible
#
# Pinned to PyTorch 2.11.0 nightly (2026-02-15) + CUDA 12.8 for sm_120.
# SageAttention is built from source since no stable pip wheel supports sm_120.
#
# Uses uv (https://github.com/astral-sh/uv) instead of pip for faster,
# parallel dependency resolution and downloads.
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
#
# To update the nightly pin, check available versions:
#   uv pip install --dry-run --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
# Then update TORCH_NIGHTLY_VERSION below.
# ============================================================================

# --- Pinned versions ---
ARG TORCH_NIGHTLY_VERSION=2.11.0.dev20260215
ARG TORCHVISION_NIGHTLY_VERSION=0.26.0.dev20260215
ARG TORCHAUDIO_NIGHTLY_VERSION=2.11.0.dev20260215
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.6.6

# ============================================================================
# Stage 1: build SageAttention from source (needs CUDA compiler)
# ============================================================================
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

ARG TORCH_NIGHTLY_VERSION
ARG TORCHVISION_NIGHTLY_VERSION
ARG TORCHAUDIO_NIGHTLY_VERSION
ARG PYTHON_VERSION
ARG UV_VERSION

COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;12.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install pinned PyTorch nightly with CUDA 12.8 (required for sm_120 / Blackwell)
RUN uv pip install --system --no-cache --pre \
    torch==${TORCH_NIGHTLY_VERSION}+cu128 \
    torchvision==${TORCHVISION_NIGHTLY_VERSION}+cu128 \
    torchaudio==${TORCHAUDIO_NIGHTLY_VERSION}+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Build SageAttention from source with sm_120 support
# --no-build-isolation: use the already-installed PyTorch (needed for CUDA extensions)
RUN uv pip install --system --no-cache ninja setuptools wheel packaging && \
    git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && \
    cd /tmp/sageattention && \
    uv pip wheel --no-build-isolation --no-deps --wheel-dir /tmp . && \
    cp /tmp/sageattention-*.whl /tmp/sageattention.whl

# ============================================================================
# Stage 2: runtime image (smaller, no compiler toolchain)
# ============================================================================
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ARG TORCH_NIGHTLY_VERSION
ARG TORCHVISION_NIGHTLY_VERSION
ARG TORCHAUDIO_NIGHTLY_VERSION
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

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

WORKDIR /app

# Install pinned PyTorch nightly with CUDA 12.8 (sm_120 / Blackwell support)
RUN uv pip install --system --no-cache --pre \
    torch==${TORCH_NIGHTLY_VERSION}+cu128 \
    torchvision==${TORCHVISION_NIGHTLY_VERSION}+cu128 \
    torchaudio==${TORCHAUDIO_NIGHTLY_VERSION}+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install SageAttention wheel built in stage 1
COPY --from=builder /tmp/sageattention.whl /tmp/sageattention.whl
RUN uv pip install --system --no-cache /tmp/sageattention.whl && \
    rm /tmp/sageattention.whl

# Install remaining Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

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
