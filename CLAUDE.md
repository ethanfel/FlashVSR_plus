# CLAUDE.md - FlashVSR+ Codebase Guide

## Project Overview

FlashVSR+ is a community fork of [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) — an efficient, real-time diffusion-based video super-resolution (VSR) system. This fork (via [lihaoyun6/FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus)) is optimized for consumer-grade hardware with enhanced UI, memory management, and post-processing tools.

This is an **inference-only** codebase. There are no training scripts, dataset loaders, or training loops. Models are loaded from HuggingFace Hub (`JunhaoZhuang/FlashVSR`).

**License:** Apache 2.0

## Repository Structure

```
FlashVSR_plus/
├── run.py                   # CLI entry point (argparse-based)
├── webui.py                 # Gradio web UI (~4,400 lines, main UI hub)
├── api.py                   # FastAPI REST API (polling-based)
├── api_test.py              # API test script
├── global_config.py         # Pydantic-based global config aggregator
├── storage_client.py        # S3/MinIO storage client
├── requirements.txt         # Python dependencies (no pyproject.toml/setup.py)
├── Dockerfile               # Docker image definition
├── .dockerignore             # Docker build exclusions
├── .env.example             # Environment variables template
│
├── src/                     # Core ML source code
│   ├── __init__.py          # Re-exports: ModelManager, pipelines, schedulers
│   ├── configs/
│   │   └── model_config.py  # Model loader configs (state_dict hash → model type)
│   ├── models/
│   │   ├── model_manager.py # Model loading & lifecycle (ModelManager class)
│   │   ├── wan_video_dit.py # Diffusion Transformer architecture (~1,000 lines)
│   │   ├── wan_video_vae.py # VAE encoder/decoder (~1,000 lines)
│   │   ├── TCDecoder.py     # Tiny Conditional Decoder
│   │   ├── utils.py         # Device utils, VRAM helpers, state_dict tools
│   │   ├── ffmpeg_utils.py  # FFmpeg GPU-accelerated encoding/decoding
│   │   └── sparse_sage/     # Sparse SageAttention (INT8 quantization)
│   ├── pipelines/
│   │   ├── base.py          # BasePipeline abstract class
│   │   ├── flashvsr_full.py # Full quality pipeline
│   │   ├── flashvsr_tiny.py # Lightweight/fast pipeline
│   │   └── flashvsr_tiny_long.py  # Long video streaming pipeline
│   ├── schedulers/
│   │   └── flow_match.py    # FlowMatch noise scheduler
│   └── vram_management/
│       └── layers.py        # AutoWrappedModule for CPU offloading
│
├── configs/                 # API/service configuration (Pydantic models)
│   ├── __init__.py
│   ├── redis.py, bandwidth.py, dashboard.py, score.py, sql.py
│   ├── storage.py, video_scheduler.py, video_upscaler.py
│   └── organic_gateway.py, video_compressor.py
│
├── toolbox/                 # Post-processing tools
│   ├── toolbox.py           # Main toolbox processor (~2,000 lines)
│   ├── rife_core.py         # RIFE frame interpolation handler
│   ├── system_monitor.py    # GPU/CPU monitoring
│   └── RIFE/                # RIFE model architecture
│
├── models/                  # Pre-trained assets (FlashVSR/ downloaded at runtime)
│   └── posi_prompt.pth      # Prompt embedding (~4 MB)
│
└── inputs/                  # Example videos and screenshots
```

## Entry Points

### CLI (`run.py`)
```bash
python run.py -i input.mp4 -s 4 -m tiny output_folder/
```
Key flags: `-m {tiny,tiny-long,full}`, `--tiled-vae`, `--tiled-dit`, `--color-fix`, `-t {fp16,bf16}`, `-a {sage,block}`

### Web UI (`webui.py`)
```bash
python webui.py
```
Multi-tab Gradio interface (Video, Image, Toolbox). Config persisted in `webui_config` file. Default port: **7860**.

### REST API (`api.py`)
```bash
python api.py
```
Polling-based FastAPI server on port **8000**. Key endpoints:
- `POST /api/v1/videos/upload` — Upload video for processing
- `GET /api/v1/videos/{task_id}/status` — Poll task status
- `POST /api/v1/videos/{task_id}/upscaling` — Start upscaling with parameters

## Processing Pipeline

```
Input Video/Image
  → Model Loading (HuggingFace → ModelManager)
  → Pipeline Selection (full / tiny / tiny-long)
  → VAE Encoding (pixel → latent space)
  → DIT Inference (diffusion transformer denoising)
     - FlowMatch scheduler for noise schedule
     - Sparse SageAttention for efficiency
     - Optional tiling for memory management
  → TCDecoder (tiny conditional decoder)
  → VAE Decoding (latent → pixel space)
  → Post-processing (wavelet color correction, optional RIFE interpolation)
  → Output (MP4/PNG, audio preserved)
```

## Development Setup

### Prerequisites
- Python 3.10+ (3.12+ required for Blackwell GPUs)
- FFmpeg on PATH
- CUDA-capable GPU (consumer-grade supported via memory optimizations)

### PyTorch Installation (GPU-dependent)

**Ampere / Ada Lovelace (RTX 30xx, 40xx):**
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu124
pip install sageattention
pip install -r requirements.txt
```

**Blackwell (RTX 50xx, RTX 6000 Pro) — sm_120:**
No stable PyTorch release supports sm_120 yet. You must use nightly + CUDA 12.8:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# SageAttention must be built from source for Blackwell:
pip install ninja setuptools wheel
git clone https://github.com/thu-ml/SageAttention.git && cd SageAttention
TORCH_CUDA_ARCH_LIST="12.0" python setup.py install
cd .. && pip install -r requirements.txt
```

Models are auto-downloaded from HuggingFace on first run to `models/FlashVSR/`.

### Docker (Blackwell-ready)
The Dockerfile uses a multi-stage build with CUDA 12.8 + PyTorch nightly + SageAttention compiled from source. All Python package installs use [uv](https://github.com/astral-sh/uv) instead of pip for faster parallel downloads and safer CUDA extension builds (`--no-build-isolation`):
```bash
docker build -t flashvsr-plus .
docker run --gpus all -p 7860:7860 -v flashvsr-models:/app/models flashvsr-plus
```

### Environment Variables
Copy `.env.example` to `.env` for S3/MinIO storage configuration (only needed for API mode).

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | nightly cu128 (Blackwell) or 2.7.0 (older GPUs) | Core ML framework |
| gradio | >=5.49.0,<6.0.0 | Web UI |
| fastapi | 0.128.0 | REST API |
| einops | ~0.8.0 | Tensor reshaping |
| safetensors | ~0.6.0 | Model weight format |
| sageattention | built from source (Blackwell) or pip (older) | Sparse attention backend |
| huggingface_hub | >=1.0.0,<2.0.0 | Model downloads |
| imageio / ffmpeg-python | — | Video I/O |

## Architecture & Key Patterns

### Pipeline Pattern
- `BasePipeline` (abstract) defines common utilities: preprocessing, VAE output conversion, latent merging
- Concrete pipelines (`FlashVSRFullPipeline`, `FlashVSRTinyPipeline`, `FlashVSRTinyLongPipeline`) implement `forward()` and `__call__()`
- Factory method: `Pipeline.from_model_manager(model_manager, ...)`

### Model Loading
- `ModelManager` orchestrates loading from HuggingFace snapshots
- Each model class provides `state_dict_converter()` for flexible format support (civitai, diffusers, huggingface)
- State dict keys are hashed to auto-detect model type via `model_config.py`

### VRAM Management
- `AutoWrappedModule` / `AutoWrappedLinear` in `src/vram_management/layers.py` — layer-wise CPU offloading
- Tiled VAE decoding and tiled DIT inference reduce peak memory
- `--unload-dit` flag frees DIT before VAE decoding

### Sparse Attention
- `src/models/sparse_sage/` implements INT8 quantized sparse attention via Triton kernels
- Replaces the original Block-Sparse-Attention from the parent project
- `core.py` includes automatic fallback to PyTorch SDPA if Triton kernels fail (e.g. on Blackwell when ptxas doesn't recognize sm_120)
- `sparse_int8_attn.py` auto-selects Triton `num_stages` based on GPU compute capability

### Configuration
- `global_config.py` aggregates all Pydantic config models from `configs/`
- Environment variables loaded via `pydantic-settings` with `__` nested delimiter

## Code Conventions

- **Classes:** PascalCase — `FlashVSRTinyPipeline`, `WanModel`, `ModelManager`
- **Functions/methods:** snake_case — `preprocess_image`, `merge_latents`
- **Constants:** UPPER_CASE — `CACHE_T`, `MAX_CONCURRENT_TASKS`
- **Private:** Leading underscore — `_wavelet_blur`, `_calc_mean_std`
- **Tensor layout:** `(B, C, F, H, W)` — batch, channels, frames, height, width
- **Reshaping:** `einops.rearrange` for `(B,C,F,H,W) ↔ (BF,C,H,W)` conversions
- **Imports:** `src/__init__.py` re-exports core classes; consumers import from `src` directly
- Some files contain Chinese comments (color correction, attention modules)

## Testing

No automated test suite exists. Testing is manual:
- `api_test.py` — end-to-end API workflow testing
- CLI and web UI for manual verification
- Example videos in `inputs/` serve as test assets

## Important Notes for AI Assistants

1. **Inference only** — Do not add training infrastructure unless explicitly requested
2. **Memory sensitive** — Changes to pipeline code must consider VRAM. Use tiling, offloading, and mixed precision patterns already established
3. **Three pipelines** — `tiny` (fast), `tiny-long` (streaming for long videos), `full` (highest quality). Keep them in sync where appropriate
4. **webui.py is the hub** — The API (`api.py`) imports processing functions directly from `webui.py`, making it a core dependency for both UI and API modes
5. **Large files** — `webui.py` is ~4,400 lines. Read specific sections rather than the whole file when possible
6. **PyTorch version is GPU-dependent** — Blackwell (sm_120) requires PyTorch nightly + cu128. Older GPUs can use stable 2.7.0. Do not pin a single torch version in requirements.txt
7. **SageAttention on Blackwell** — Must be built from source with `TORCH_CUDA_ARCH_LIST="12.0"`. The pip package does not include sm_120 kernels. If the Triton kernel fails at runtime, `sparse_sage/core.py` automatically falls back to PyTorch SDPA
8. **FFmpeg required** — Video I/O depends on FFmpeg on PATH. GPU-accelerated encoding attempted first with CPU fallback
9. **No linting/type checking** — No mypy, ruff, or flake8 configuration. Follow existing code style
10. **Model downloads** — Models auto-download from HuggingFace. `models/FlashVSR/` is not committed to git
11. **GPU compatibility** — Tested with NVIDIA Ampere, Ada Lovelace, and Blackwell architectures. Sparse SageAttention Triton kernels auto-tune per architecture with SDPA fallback
