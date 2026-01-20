import shutil
import subprocess
import torch
import os

def get_gpu_encoder():
    """
    Detects the best available GPU encoder for FFmpeg.
    Returns encoder name and optional presets.
    """
    # Force GPU detection via torch
    if torch.cuda.is_available():
        return "h264_nvenc"
    elif torch.mps.is_available() or os.uname().sysname == 'Darwin':
        return "h264_videotoolbox"
    
    # Check if ffmpeg has specific encoders
    try:
        encoders = subprocess.check_output(['ffmpeg', '-encoders'], stderr=subprocess.STDOUT).decode()
        if 'h264_nvenc' in encoders:
            return 'h264_nvenc'
        if 'h264_videotoolbox' in encoders:
            return 'h264_videotoolbox'
        if 'h264_qsv' in encoders:
            return 'h264_qsv'
        if 'h264_amf' in encoders:
            return 'h264_amf'
    except:
        pass

    return "libx264" # Fallback

def get_gpu_decoder_args():
    """
    Returns the hardware acceleration flags for decoding.
    """
    if torch.cuda.is_available():
        return ["-hwaccel", "cuda"]
    elif torch.mps.is_available() or os.uname().sysname == 'Darwin':
        return ["-hwaccel", "videotoolbox"]
    
    return []

def get_imageio_settings(fps=30, quality=None, bitrate=None):
    """
    Returns (encoder, ffmpeg_params) for imageio.get_writer.
    """
    encoder = get_gpu_encoder()
    params = []
    
    # Basic hardware acceleration
    if torch.cuda.is_available():
        params.extend(["-hwaccel", "cuda"])
    elif torch.mps.is_available() or os.uname().sysname == 'Darwin':
        params.extend(["-hwaccel", "videotoolbox"])
    
    # Encoder specific tweaks
    if encoder == "h264_nvenc":
        if quality is not None:
            cq = 35 - int(quality * 1.7)
            # Use -rc:v to be more specific, and simplify options
            params.extend(["-cq", str(cq), "-preset", "p4"])
        elif bitrate:
            params.extend(["-b:v", bitrate])
    elif encoder == "h264_videotoolbox":
        if quality is not None:
            q_val = quality / 10.0
            params.extend(["-q:v", str(int(q_val * 100))])
        elif bitrate:
            params.extend(["-b:v", bitrate])
    else:
        # Default libx264 fallbacks
        if quality is not None:
            crf = 35 - int(quality * 2)
            params.extend(["-crf", str(crf)])

    return encoder, params
