import os
import uuid
import requests
import uvicorn
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
import time
import base64

# Import core logic and helper functions from your webui.py
from webui import (
    run_flashvsr_single, 
    process_video_with_chunks, 
    resize_input_video,      # Added for resizing
    get_video_dimensions,    # Added to calculate half-width
    TEMP_DIR, 
    DEFAULT_OUTPUT_DIR,
    log
)

app = FastAPI(title="FlashVSR+ API")

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

class UpscaleRequest(BaseModel):
    input_path: str = Field(..., description="Local file path or URL to video/image")
    
    half_res_preprocess: bool = Field(False, description="Downscale input to 50% before processing")
    
    mode: str = Field("tiny", pattern="^(tiny|full|tiny-long)$")
    model_version: str = Field("v1.1", pattern="^(v1.0|v1.1)$")
    scale: int = Field(2, ge=2, le=4)
    
    # Performance/Memory Settings
    enable_chunks: bool = False
    chunk_duration: float = 5.0
    tiled_vae: bool = True
    tiled_dit: bool = True
    tile_size: int = 256
    tile_overlap: int = 24
    
    # Advanced Model Settings
    color_fix: bool = True
    unload_dit: bool = False
    dtype_str: str = Field("bf16", pattern="^(fp16|bf16)$")
    seed: int = 0
    device: str = "auto"
    fps_override: int = 30
    quality: int = 5
    attention_mode: str = "sage"
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    
    # Output Settings
    create_comparison: bool = False

def download_video(url: str) -> str:
    """Downloads a file from a URL and returns the local path."""
    try:
        local_filename = f"download_{uuid.uuid4().hex[:8]}_{url.split('/')[-1]}".split('?')[0]
        if not (local_filename.lower().endswith(('.mp4', '.mov', '.avi', '.png', '.jpg', '.jpeg'))):
            local_filename += ".mp4"
            
        path = os.path.join(TEMP_DIR, local_filename)
        
        log(f"Downloading video from URL: {url}", message_type="info")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video from URL: {str(e)}")

def cleanup_files(paths: list):
    """Removes temporary files after the request is finished."""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                log(f"Cleaned up: {path}")
            except:
                pass

@app.post("/upscaling")
async def upscale_video(req: UpscaleRequest, background_tasks: BackgroundTasks):
    target_input = req.input_path
    downloaded = False
    files_to_clean = []

    # 1. Handle URL Input
    if target_input.startswith(("http://", "https://")):
        target_input = download_video(target_input)
        downloaded = True
        files_to_clean.append(target_input)
    
    if not os.path.exists(target_input):
        raise HTTPException(status_code=404, detail="Input file not found.")

    # 2. Setup Progress Mock (FastAPI doesn't use Gradio's Progress)
    class SimpleProgress:
        def __call__(self, val, desc=""):
            log(f"[API Progress {val*100:.0f}%]: {desc}")
        def tqdm(self, iterable, *args, **kwargs):
            return iterable

    a = time.time()
    try:
        # --- NEW LOGIC: PRE-PROCESS RESIZE ---
        if req.half_res_preprocess:
            width, _ = get_video_dimensions(target_input)
            if width > 0:
                half_width = int(width // 2)
                log(f"Pre-processing: Resizing input to half-width ({half_width}px)", message_type="info")
                resized_path = resize_input_video(target_input, half_width, progress=SimpleProgress())
                if resized_path != target_input:
                    target_input = resized_path
                    files_to_clean.append(target_input) # Ensure temp resized file is cleaned

        if req.enable_chunks:
            result = process_video_with_chunks(
                input_path=target_input, chunk_duration=req.chunk_duration, mode=req.mode,
                model_version=req.model_version, scale=req.scale, color_fix=req.color_fix,
                tiled_vae=req.tiled_vae, tiled_dit=req.tiled_dit, tile_size=req.tile_size,
                tile_overlap=req.tile_overlap, unload_dit=req.unload_dit, dtype_str=req.dtype_str,
                seed=req.seed, device=req.device, fps_override=req.fps_override,
                quality=req.quality, attention_mode=req.attention_mode, sparse_ratio=req.sparse_ratio,
                kv_ratio=req.kv_ratio, local_range=req.local_range, autosave=False, progress=SimpleProgress()
            )
        else:
            result = run_flashvsr_single(
                input_path=target_input, mode=req.mode, model_version=req.model_version,
                scale=req.scale, color_fix=req.color_fix, tiled_vae=req.tiled_vae,
                tiled_dit=req.tiled_dit, tile_size=req.tile_size, tile_overlap=req.tile_overlap,
                unload_dit=req.unload_dit, dtype_str=req.dtype_str, seed=req.seed,
                device=req.device, fps_override=req.fps_override, quality=req.quality,
                attention_mode=req.attention_mode, sparse_ratio=req.sparse_ratio,
                kv_ratio=req.kv_ratio, local_range=req.local_range, autosave=False,
                create_comparison=req.create_comparison, progress=SimpleProgress()
            )

        output_path = result[1]
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Processing failed to produce an output.")

        background_tasks.add_task(cleanup_files, files_to_clean)

        b = time.time()
        log(f"Finished in {b-a:.2f} seconds", message_type="info")

        with open(output_path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read()).decode('utf-8')
        
        # Since we encoded it to a string, we can clean up the output file immediately
        files_to_clean.append(output_path)
        background_tasks.add_task(cleanup_files, files_to_clean)
        
        return {
            "status": "success",
            "filename": os.path.basename(output_path),
            "base64": encoded_string
        }
        # return FileResponse(path=output_path, media_type="video/mp4", filename=os.path.basename(output_path))

    except Exception as e:
        log(f"API Error: {str(e)}", message_type="error")
        cleanup_files(files_to_clean)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)