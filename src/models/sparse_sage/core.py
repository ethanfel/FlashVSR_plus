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


def _sdpa_fallback(q, k, v, is_causal=False, tensor_layout="HND"):
    """Fallback using PyTorch's native scaled_dot_product_attention."""
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

    o = torch.nn.functional.scaled_dot_product_attention(
        q_fp, k_fp, v_fp, is_causal=is_causal
    )

    if tensor_layout == "NHD":
        o = o.transpose(1, 2)

    return o.to(output_dtype)


def sparse_sageattn(q, k, v, mask_id = None, is_causal=False, tensor_layout="HND"):
    global _triton_kernel_tested, _triton_kernel_works

    # Fast path: if we already know the Triton kernel doesn't work, use SDPA
    if _triton_kernel_tested and not _triton_kernel_works:
        return _sdpa_fallback(q, k, v, is_causal=is_causal, tensor_layout=tensor_layout)

    if mask_id is None:
        mask_id = torch.ones((q.shape[0], q.shape[1], (q.shape[2] + 128 - 1)//128, (q.shape[3] + 64 - 1)//64), dtype=torch.int8, device=q.device) # TODO

    output_dtype = q.dtype
    # The Triton kernel internally computes in fp16; convert v for the kernel
    # then the output is cast back to the original dtype
    if v.dtype != torch.float16:
        v = v.to(torch.float16)

    seq_dim = 1 if tensor_layout == "NHD" else 2
    km = k.mean(dim=seq_dim, keepdim=True)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, tensor_layout=tensor_layout)

    try:
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
            return _sdpa_fallback(q, k, v, is_causal=is_causal, tensor_layout=tensor_layout)
        raise
