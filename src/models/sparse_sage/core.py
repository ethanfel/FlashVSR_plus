"""
https://github.com/jt-zhang/Sparse_SageAttention_API

Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .quant_per_block import per_block_int8
from .sparse_int8_attn import forward as sparse_sageattn_fwd
import torch

# Track whether the custom Triton kernel works on this GPU.
# On Blackwell (sm_120) with older ptxas, Triton may fail with
# "sm_120 is not defined for option 'gpu-name'". In that case,
# fall back to PyTorch scaled_dot_product_attention.
_triton_kernel_tested = False
_triton_kernel_works = True


def _expand_block_mask(mask_id, N_q, N_kv, BLOCK_M=128, BLOCK_N=64):
    """Expand a block-level sparse mask to a full boolean attention mask.

    Args:
        mask_id: (B, H, num_q_blocks, num_kv_blocks) int8 tensor.
                 Non-zero means "attend", zero means "skip".
        N_q: actual query sequence length.
        N_kv: actual key/value sequence length.

    Returns:
        Boolean mask (B, H, N_q, N_kv) suitable for SDPA (True = attend).
    """
    # repeat_interleave expands each block entry to cover its full range
    expanded = mask_id.bool()
    expanded = expanded.repeat_interleave(BLOCK_M, dim=2)[:, :, :N_q, :]
    expanded = expanded.repeat_interleave(BLOCK_N, dim=3)[:, :, :, :N_kv]
    return expanded


def _chunked_sparse_sdpa(q, k, v, mask_id, BLOCK_M=128, BLOCK_N=64):
    """Chunked block-sparse SDPA that processes Q in blocks to avoid O(N^2) memory.

    Instead of expanding the full (N_q x N_kv) mask, this processes BLOCK_M rows
    of Q at a time, expanding only the current row of the block mask. Peak memory
    is O(BLOCK_M x N_kv) instead of O(N_q x N_kv).

    Args:
        q: (B, H, N_q, D) query tensor.
        k: (B, H, N_kv, D) key tensor.
        v: (B, H, N_kv, D) value tensor.
        mask_id: (B, H, num_q_blocks, num_kv_blocks) int8 block mask.
        BLOCK_M: query block size (must match mask generation).
        BLOCK_N: key/value block size (must match mask generation).

    Returns:
        (B, H, N_q, D) attention output.
    """
    B, H, N_q, D = q.shape
    N_kv = k.shape[2]
    num_q_blocks = mask_id.shape[2]

    output = torch.zeros(B, H, N_q, D, dtype=q.dtype, device=q.device)

    for qi in range(num_q_blocks):
        q_start = qi * BLOCK_M
        q_end = min(q_start + BLOCK_M, N_q)
        q_chunk = q[:, :, q_start:q_end, :]

        # Expand only this Q block's row of the mask: (B, H, 1, num_kv_blocks)
        # -> (B, H, chunk_len, N_kv)
        row_mask = mask_id[:, :, qi:qi+1, :].bool()
        row_mask = row_mask.repeat_interleave(BLOCK_N, dim=3)[:, :, :, :N_kv]
        row_mask = row_mask.expand(-1, -1, q_end - q_start, -1)

        output[:, :, q_start:q_end, :] = torch.nn.functional.scaled_dot_product_attention(
            q_chunk, k, v, attn_mask=row_mask
        )

    return output


# Threshold: if the full expanded mask would exceed this many elements,
# use chunked attention instead. 2^30 ≈ 1 billion elements ≈ 1 GiB for bool.
_CHUNKED_THRESHOLD = 1 << 30


def _sdpa_fallback(q, k, v, mask_id=None, is_causal=False, tensor_layout="HND"):
    """Fallback using PyTorch's native scaled_dot_product_attention.

    When mask_id is provided, the block-level sparse mask is respected.
    For small sequences the mask is expanded to a full boolean mask.
    For large sequences, chunked attention is used to avoid O(N^2) memory.
    """
    output_dtype = q.dtype
    if tensor_layout == "HND":
        # q/k/v: (B, H, N, D) — already in the right layout for SDPA
        pass
    elif tensor_layout == "NHD":
        # (B, N, H, D) -> (B, H, N, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    q_fp = q.to(torch.float16) if q.dtype not in (torch.float16, torch.bfloat16) else q
    k_fp = k.to(q_fp.dtype)
    v_fp = v.to(q_fp.dtype)

    if mask_id is None:
        o = torch.nn.functional.scaled_dot_product_attention(
            q_fp, k_fp, v_fp, is_causal=is_causal
        )
    else:
        N_q = q_fp.shape[2]
        N_kv = k_fp.shape[2]
        mask_numel = q_fp.shape[0] * q_fp.shape[1] * N_q * N_kv

        if mask_numel <= _CHUNKED_THRESHOLD:
            # Small enough to expand the full mask
            attn_mask = _expand_block_mask(mask_id, N_q, N_kv)
            o = torch.nn.functional.scaled_dot_product_attention(
                q_fp, k_fp, v_fp, attn_mask=attn_mask, is_causal=is_causal
            )
        else:
            # Large sequence: use chunked attention to avoid OOM
            o = _chunked_sparse_sdpa(q_fp, k_fp, v_fp, mask_id)

    if tensor_layout == "NHD":
        o = o.transpose(1, 2)

    return o.to(output_dtype)


def sparse_sageattn(q, k, v, mask_id = None, is_causal=False, tensor_layout="HND"):
    global _triton_kernel_tested, _triton_kernel_works

    # Fast path: if we already know the Triton kernel doesn't work, use SDPA
    if _triton_kernel_tested and not _triton_kernel_works:
        return _sdpa_fallback(q, k, v, mask_id=mask_id, is_causal=is_causal, tensor_layout=tensor_layout)

    if mask_id is None:
        if tensor_layout == "HND":
            num_q_blocks = (q.shape[2] + 128 - 1) // 128
            num_kv_blocks = (k.shape[2] + 64 - 1) // 64
        else:  # NHD
            num_q_blocks = (q.shape[1] + 128 - 1) // 128
            num_kv_blocks = (k.shape[1] + 64 - 1) // 64
        mask_id = torch.ones((q.shape[0], q.shape[1] if tensor_layout == "HND" else q.shape[2],
                              num_q_blocks, num_kv_blocks), dtype=torch.int8, device=q.device)

    output_dtype = q.dtype
    # The Triton kernel internally computes in fp16; convert v for the kernel
    # then the output is cast back to the original dtype
    if v.dtype != torch.float16:
        v = v.to(torch.float16)

    try:
        seq_dim = 1 if tensor_layout == "NHD" else 2
        km = k.mean(dim=seq_dim, keepdim=True)

        q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, tensor_layout=tensor_layout)

        o = sparse_sageattn_fwd(
            q_int8, k_int8, mask_id, v, q_scale, k_scale,
            is_causal=is_causal, tensor_layout=tensor_layout, output_dtype=output_dtype
        )
        if not _triton_kernel_tested:
            _triton_kernel_tested = True
            _triton_kernel_works = True
        return o
    except RuntimeError as e:
        err_msg = str(e).lower()
        if "sm_120" in err_msg or "ptxas" in err_msg or "no kernel image" in err_msg or "triton" in err_msg:
            if not _triton_kernel_tested:
                _triton_kernel_tested = True
                _triton_kernel_works = False
                print(
                    "[FlashVSR] WARNING: Custom Triton sparse attention kernel failed "
                    f"on this GPU ({e}). Falling back to PyTorch SDPA. "
                    "This may happen on Blackwell (sm_120) GPUs if Triton/ptxas "
                    "does not yet support this architecture. Performance may differ."
                )
            return _sdpa_fallback(q, k, v, mask_id=mask_id, is_causal=is_causal, tensor_layout=tensor_layout)
        raise
