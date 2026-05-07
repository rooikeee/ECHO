import torch
from typing import Any, Optional, Tuple

__version__: str

# Envs
persistent_envs: dict[str, str]

# Configs
def set_num_sms(num_sms: int) -> None: ...
def get_num_sms() -> int: ...
def set_tc_util(tc_util: float) -> None: ...
def get_tc_util() -> float: ...

# Kernels
fp8_gemm_nt: Any
fp8_gemm_nn: Any
fp8_gemm_tn: Any
fp8_gemm_tt: Any
fp8_gemm_nt_skip_head_mid: Any
m_grouped_fp8_gemm_nt_contiguous: Any
m_grouped_fp8_gemm_nn_contiguous: Any
m_grouped_fp8_gemm_nt_masked: Any
k_grouped_fp8_gemm_nt_contiguous: Any
k_grouped_fp8_gemm_tn_contiguous: Any

bf16_gemm_nt: Any
bf16_gemm_nn: Any
bf16_gemm_tn: Any
bf16_gemm_tt: Any
m_grouped_bf16_gemm_nt_contiguous: Any
m_grouped_bf16_gemm_nt_masked: Any

cublaslt_gemm_nt: Any
cublaslt_gemm_nn: Any
cublaslt_gemm_tn: Any
cublaslt_gemm_tt: Any

einsum: Any

fp8_mqa_logits: Any
fp8_mqa_logits_fuse_topk: Any
fp8_mqa_logits_fuse_prefetch: Any
get_paged_mqa_logits_metadata: Any
fp8_paged_mqa_logits: Any

def fp8_paged_mqa_logits_fused_v2(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    page_table_1: torch.Tensor,
    device_pool_buf: torch.Tensor,
    host_pool_buf: torch.Tensor,
    device_pool_loc_alloc_buf: torch.Tensor,
    device_pool_priority: torch.Tensor,
    device_pool_loc_small_priority: torch.Tensor,
    device_token_to_host: torch.Tensor,
    host_token_to_device: torch.Tensor,
    recall_counter: torch.Tensor,
    query_recall_counter: torch.Tensor,
    decode_topk_logits: torch.Tensor,
    clean_logits: bool
) -> torch.Tensor:
    """
    Shape parameters:
        B: batch size

    Args:
        q: (B, next_n=1, num_heads, head_dim). Dtype: torch.float8_e4m3fn.
        fused_kv_cache: (num_kv_blocks, block_kv=64, num_heads_kv=1, head_dim_with_sf). Dtype: torch.uint8 (storage).
            Stores FP8 KV cache and FP32 scales.
        weights: (B, num_heads). Dtype: torch.float.
        context_lens: (B,). Dtype: torch.int. 
        block_table: (B, max_block_len). Dtype: torch.int. Holds indices.
        schedule_meta: (schedule_meta_size, meta_info_size=2). Dtype: torch.int.
        max_context_len: int.

        page_table_1: (B, max_kv_len). Dtype: torch.int. At [q_idx, kv_idx] gives the pool index (in host_token_to_device)
            for the `kv_idx`-th token for the `q_idx`-th query in the batch.
        device_pool_buf: (device_pool_size, mla_num_heads_kv=1, mla_head_dim). Dtype: torch.bfloat16.
            Modified in-place.
        host_pool_buf: (host_pool_size, mla_num_heads_kv=1, mla_head_dim). Dtype: torch.bfloat16.
            Modified in-place.
        device_pool_loc_alloc_buf: (device_pool_size - 1,). Dtype: torch.int32.
            Modified in-place.
        device_pool_priority: (device_pool_size,). Dtype: torch.int32.
            Modified in-place.
        device_pool_loc_small_priority: (device_pool_size,). Dtype: torch.int32.
        device_token_to_host: (device_pool_size,). Dtype: torch.int64. Holds indices.
            Modified in-place.
        host_token_to_device: (host_pool_size + 1,). Dtype: torch.int. Holds indices.
            Modified in-place.
        recall_counter: (1,). Dtype: uint32
            Modified in-place.
        query_recall_counter: (B,). Dtype: uint32.
            Modified in-place.
        decode_topk_logits: (B,). Dtype: torch.float32.
        clean_logits: bool.
    """
    ...

transform_sf_into_required_layout: Any

# Aliases
fp8_m_grouped_gemm_nt_masked: Any
bf16_m_grouped_gemm_nt_masked: Any

# Modules
from . import testing as testing
from . import utils as utils
from .utils import *
