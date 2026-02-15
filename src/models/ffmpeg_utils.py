import shutil
import subprocess
import torch
import os

def get_gpu_encoder():
    """
    Detects the best available GPU encoder for FFmpeg.
    Returns encoder name and optional presets.
    """
    # # Force GPU detection via torch
    # if torch.cuda.is_available():
    #     return "h264_nvenc"
    # elif torch.mps.is_available() or os.uname().sysname == 'Darwin':
    #     return "h264_videotoolbox"
    
    # Check if ffmpeg has specific encoders
    try:
        encoders = subprocess.check_output(['ffmpeg', '-encoders'], stderr=subprocess.STDOUT).decode()
        if 'hevc_nvenc' in encoders:
            return 'hevc_nvenc'
        if 'h264_nvenc' in encoders:
            return 'h264_nvenc'
        if 'h264_videotoolbox' in encoders:
            return 'h264_videotoolbox'
        if 'h264_qsv' in encoders:
            return 'h264_qsv'
        if 'h264_amf' in encoders:
            return 'h264_amf'
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
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
    params = ["-v", "error"] # Reduce noise, helps with debugging broken pipes
    
    # Encoder specific tweaks
    if encoder == "h264_nvenc":
        # NVENC options vary by version, using safest ones
        if quality is not None:
            # Map 0-10 to 35-18
            qp_val = 35 - int(quality * 1.7)
            params.extend(["-qp", str(qp_val), "-preset", "p4"])
        elif bitrate:
            params.extend(["-b:v", bitrate])
    elif encoder == "h264_videotoolbox":
        if quality is not None:
            # VideoToolbox quality is 0.0 to 1.0, some versions use -q:v 0-100
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
