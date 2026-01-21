import os
import uuid
import requests
import uvicorn
import asyncio
import functools
import time
import base64
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, FastAPI, Query
from pydantic import BaseModel, Field
from webui import (
    run_flashvsr_single, 
    process_video_with_chunks, 
    resize_input_video,      
    get_video_dimensions,    
    TEMP_DIR, 
    DEFAULT_OUTPUT_DIR,
    log
)
from storage_client import storage_client


MAX_CONCURRENT_TASKS = 1  # Adjust based on your GPU memory/capacity
CANCELLED_TASKS = set()

app = FastAPI(title="FlashVSR+ Polling API")

# Include API routes
api_router = APIRouter(tags=["videos"])

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

class VideoUploadRequest(BaseModel):
    video_url: str

class VideoUploadResponse(BaseModel):
    task_id: str
    message: str

class UpscalingSelectionRequest(BaseModel):
    # This inherits all your original UpscaleRequest parameters
    half_res_preprocess: bool = False
    mode: str = "tiny"
    model_version: str = "v1.1"
    scale: int = 1
    enable_chunks: bool = True
    chunk_duration: float = 15.0
    tiled_vae: bool = False
    tiled_dit: bool = False
    tile_size: int = 256
    tile_overlap: int = 24
    
    # Advanced Model Settings
    color_fix: bool = True
    unload_dit: bool = False
    dtype_str: str = "bf16"
    seed: int = 1
    device: str = "auto"
    fps_override: int = 30
    quality: int = 5
    attention_mode: str = "sage"
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    
    # Output Settings
    create_comparison: bool = False

class VideoStatusResponse(BaseModel):
    task_id: str
    status: str  # created, downloading, downloaded, processing, completed, failed
    message: str
    output_url: str = None

TASK_QUEUE = asyncio.Queue()
TASKS = {} 
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


async def queue_worker():
    """Continuously pulls from queue and spawns limited concurrent workers."""
    while True:
        task_id, req = await TASK_QUEUE.get()
        # Create a background task for the actual processing so the worker can pull the next item
        asyncio.create_task(run_task_with_semaphore(task_id, req))
        TASK_QUEUE.task_done()

async def run_task_with_semaphore(task_id: str, req: UpscalingSelectionRequest):
    async with SEMAPHORE:
        if task_id in CANCELLED_TASKS:
            TASKS[task_id].update({"status": "cancelled", "message": "Task cancelled before starting"})
            CANCELLED_TASKS.remove(task_id)
            return
            
        await run_processing_task(task_id, req)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(queue_worker())

def download_and_store(task_id: str, url: str):
    try:
        TASKS[task_id]["status"] = "downloading"
        local_filename = f"dl_{task_id}_{url.split('/')[-1]}".split('?')[0]
        if not (local_filename.lower().endswith(('.mp4', '.mov', '.avi', '.png', '.jpg', '.jpeg'))):
            local_filename += ".mp4"
            
        path = os.path.join(TEMP_DIR, local_filename)
        
        log(f"Downloading video from URL: {url}", message_type="info")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        TASKS[task_id].update({"status": "downloaded", "input_path": path})
    except Exception as e:
        TASKS[task_id].update({"status": "failed", "message": f"Download failed: {str(e)}"})

async def run_processing_task(task_id: str, req: UpscalingSelectionRequest):
    task = TASKS.get(task_id)
    target_input = task["input_path"]
    files_to_clean = [target_input]
    
    class SimpleProgress:
        def __call__(self, val, desc=""): 
            TASKS[task_id]["message"] = f"{desc} ({val*100:.0f}%)"
            print(f"[Task {task_id[:8]}] {desc} ({val*100:.0f}%)", flush=True)
            
        def tqdm(self, iterable, *args, **kwargs): 
            # Optional: If you want to see standard tqdm bars in terminal
            from tqdm import tqdm
            return tqdm(iterable, *args, **kwargs)

    a = time.time()
    try:
        TASKS[task_id]["status"] = "processing"
        
        # Original Resizing Logic
        if req.half_res_preprocess:
            width, _ = get_video_dimensions(target_input)
            if width > 0:
                resized_path = await asyncio.to_thread(
                    resize_input_video, target_input, int(width // 2), progress=SimpleProgress()
                )
                if resized_path != target_input:
                    target_input = resized_path
                    files_to_clean.append(target_input)

        # Original Processing Branching
        if req.enable_chunks:
            result = await asyncio.to_thread(
                process_video_with_chunks,
                input_path=target_input,
                chunk_duration=req.chunk_duration,
                mode=req.mode,
                model_version=req.model_version,
                scale=req.scale,
                color_fix=req.color_fix,
                tiled_vae=req.tiled_vae,
                tiled_dit=req.tiled_dit,
                tile_size=req.tile_size,
                tile_overlap=req.tile_overlap,
                unload_dit=req.unload_dit,
                dtype_str=req.dtype_str,
                seed=req.seed,
                device=req.device,
                fps_override=req.fps_override,
                quality=req.quality,
                attention_mode=req.attention_mode,
                sparse_ratio=req.sparse_ratio,
                kv_ratio=req.kv_ratio,
                local_range=req.local_range,
                autosave=False,
                progress=SimpleProgress()
            )
        else:
            result = await asyncio.to_thread(
                run_flashvsr_single,
                input_path=target_input,
                mode=req.mode,
                model_version=req.model_version,
                scale=req.scale,
                color_fix=req.color_fix,
                tiled_vae=req.tiled_vae,
                tiled_dit=req.tiled_dit,
                tile_size=req.tile_size,
                tile_overlap=req.tile_overlap,
                unload_dit=req.unload_dit,
                dtype_str=req.dtype_str,
                seed=req.seed,
                device=req.device,
                fps_override=req.fps_override,
                quality=req.quality,
                attention_mode=req.attention_mode,
                sparse_ratio=req.sparse_ratio,
                kv_ratio=req.kv_ratio,
                local_range=req.local_range,
                autosave=False,
                create_comparison=req.create_comparison,
                progress=SimpleProgress()
            )

        output_path = result[1]
        b = time.time()
        print(f"[Task {task_id[:8]}] Processing finished in {b-a:.2f} seconds", flush=True)
        
        # Storage and Finalization
        storage_key = f"{task_id}_out.mp4"
        await storage_client.upload_file(storage_key, output_path)
        presigned_url = await storage_client.get_presigned_url(storage_key)
        
        TASKS[task_id].update({
            "status": "completed",
            "output_url": presigned_url,
            "message": "Processing finished successfully"
        })
        
        # Cleanup
        files_to_clean.append(output_path)
        for p in files_to_clean:
            if os.path.exists(p): os.remove(p)

    except Exception as e:
        TASKS[task_id].update({"status": "failed", "message": str(e)})

# --- Routes ---

@api_router.post("/videos/upload", response_model=VideoUploadResponse)
async def upload_video(request: VideoUploadRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"status": "created", "message": "Task initialized"}
    background_tasks.add_task(download_and_store, task_id, request.video_url)
    return {"task_id": task_id, "message": "Download initiated"}

@api_router.post("/videos/{task_id}/upscaling")
async def start_upscaling(task_id: str, request: UpscalingSelectionRequest):
    if task_id not in TASKS: raise HTTPException(status_code=404, detail="Task not found")
    if TASKS[task_id]["status"] != "downloaded":
        raise HTTPException(status_code=400, detail=f"Video not ready. Status: {TASKS[task_id]['status']}")
    
    TASKS[task_id].update({"status": "queued", "message": "Waiting in queue"})
    await TASK_QUEUE.put((task_id, request))
    return {"status": "accepted", "message": "Added to queue"}

@api_router.post("/videos/{task_id}/cancel")
async def cancel_task(task_id: str):
    if task_id not in TASKS: raise HTTPException(status_code=404, detail="Task not found")
    
    current_status = TASKS[task_id].get("status")
    if current_status in ["completed", "failed", "cancelled"]:
        return {"message": f"Task already in terminal state: {current_status}"}
    
    CANCELLED_TASKS.add(task_id)
    TASKS[task_id].update({"status": "cancelled", "message": "User requested cancellation"})
    return {"status": "success", "message": "Cancellation request received"}

@api_router.get("/videos/{task_id}/status", response_model=VideoStatusResponse)
async def get_status(task_id: str):
    if task_id not in TASKS: raise HTTPException(status_code=404, detail="Task not found")
    task = TASKS[task_id]
    return {
        "task_id": task_id,
        "status": task.get("status"),
        "message": task.get("message", ""),
        "output_url": task.get("output_url", "")
    }

@api_router.get("/queue/status")
async def get_queue_status():
    return {
        "waiting_in_queue": TASK_QUEUE.qsize(),
        "concurrency_limit": MAX_CONCURRENT_TASKS,
        "active_processing": MAX_CONCURRENT_TASKS - SEMAPHORE._value
    }

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)