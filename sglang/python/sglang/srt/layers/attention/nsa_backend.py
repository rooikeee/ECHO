from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, TypeAlias

import torch
import triton
import triton.language as tl

from sglang.srt.configs.model_config import get_nsa_index_topk, is_deepseek_nsa
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    BaseIndexerMetadata,
    NSA_FUSE_LOGITS_RECALL,
)
from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.nsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.nsa.utils import (
    NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
    NSA_FUSE_TOPK,
    compute_nsa_seqlens,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_hip

# from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

_is_hip = is_hip()
I32_MAX = torch.iinfo(torch.int32).max

if _is_hip:
    try:
        from aiter import (  # noqa: F401
            flash_attn_varlen_func,
            mha_batch_prefill_func,
            paged_attention_ragged,
        )
        from aiter.mla import mla_decode_fwd, mla_prefill_fwd  # noqa: F401
    except ImportError:
        print(
            "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
        )
else:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

import os
NSA_USE_CUDA_GRAPH=True
# NSA_USE_CUDA_GRAPH=False

@dataclass(frozen=True)
class NSAFlashMLAMetadata:
    """Metadata only needed by FlashMLA"""

    flashmla_metadata: torch.Tensor
    num_splits: torch.Tensor

    def slice(self, sli):
        return NSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=self.num_splits[sli],
        )

    def copy_(self, other: "NSAFlashMLAMetadata"):
        self.flashmla_metadata.copy_(other.flashmla_metadata)
        self.num_splits.copy_(other.num_splits)


@dataclass(frozen=True)
class NSAMetadata:
    page_size: int

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor
    # Maximum sequence length for query
    max_seq_len_q: int
    # Maximum sequence length for key
    max_seq_len_k: int
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor
    # Page table, the index of KV Cache Tables/Blocks
    # this table is always with page_size = 1
    page_table_1: torch.Tensor

    # NOTE(dark): This will property be used in:
    # 1. dense decode/prefill, we use paged flash attention, need real_page_table
    # 2. sparse decode/prefill, indexer need real_page_table to compute the score
    real_page_table: torch.Tensor

    # NSA metadata (nsa prefill are expanded)
    nsa_cache_seqlens_int32: torch.Tensor  # this seqlens is clipped to `topk`
    nsa_cu_seqlens_q: torch.Tensor  # must be arange(0, len(nsa_cu_seqlens_k))
    nsa_cu_seqlens_k: torch.Tensor  # cumsum of `nsa_cache_seqlens_int32`
    nsa_extend_seq_lens_list: List[int]
    nsa_seqlens_expanded: torch.Tensor  # expanded, unclipped `seqlens`
    nsa_max_seqlen_q: Literal[1] = 1  # always 1 for decode, variable for extend
    nsa_extend_seq_to_req: Optional[torch.Tensor] = None    # not need for decode

    # Device scalar storing the number of real Q tokens in the current forward.
    # Used by CUDA-graph KV cache updates to avoid creating a host scalar tensor
    # in the hot path.
    nsa_non_padded: Optional[torch.Tensor] = None

    flashmla_metadata: Optional[NSAFlashMLAMetadata] = None


@dataclass(frozen=True)
class NSAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: NSAMetadata

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.nsa_seqlens_expanded
    
    def get_extend_seq_to_req(self) -> torch.Tensor:
        return self.attn_metadata.nsa_extend_seq_to_req

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        return_topk_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from sgl_kernel import fast_topk_transform_fused, fast_topk_v2

        if not NSA_FUSE_TOPK:
            topk_results = fast_topk_v2(logits, self.get_seqlens_expanded(), topk)
            if return_topk_logits:
                from sglang.srt.layers.attention.nsa.nsa_indexer import (
                    get_topk_logits,
                )

                return topk_results, get_topk_logits(logits, topk_results)
            return topk_results

        # NOTE(dark): if fused, we return a transformed page table directly
        if return_topk_logits:
            topk_results, topk_logits = fast_topk_transform_fused(
                score=logits,
                lengths=self.get_seqlens_expanded(),
                page_table_size_1=self.attn_metadata.page_table_1,
                cu_seqlens_q=self.attn_metadata.cu_seqlens_q,
                topk=topk,
                return_topk_logits=True,
            )
        else:
            topk_results = fast_topk_transform_fused(
                score=logits,
                lengths=self.get_seqlens_expanded(),
                page_table_size_1=self.attn_metadata.page_table_1,
                cu_seqlens_q=self.attn_metadata.cu_seqlens_q,
                topk=topk,
            )
            topk_logits = None

        if self.attn_metadata.nsa_non_padded is not None:
            row_indices = torch.arange(
                topk_results.size(0), device=topk_results.device, dtype=torch.int32
            )
            mask = row_indices >= self.attn_metadata.nsa_non_padded
            # why 0 instead -1: all -1 cause some issues in DG
            topk_results.masked_fill_(mask.unsqueeze(-1), 0)
            if topk_logits is not None:
                topk_logits.masked_fill_(mask, 0)
        if return_topk_logits:
            assert topk_logits is not None
            return topk_results, topk_logits
        return topk_results


def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    assert seqlens.dtype == torch.int32 and seqlens.is_cuda
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


_NSA_IMPL_T: TypeAlias = Literal[
    "flashmla_prefill", "flashmla_decode", "fa3", "tilelang"
]

NSA_PREFILL_IMPL: _NSA_IMPL_T
NSA_DECODE_IMPL: _NSA_IMPL_T


class NativeSparseAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata: NSAMetadata
        self.device = model_runner.device
        assert isinstance(model_runner.page_size, int)
        self.real_page_size = model_runner.page_size
        self.num_splits = (
            1 if model_runner.server_args.enable_deterministic_inference else 0
        )
        self.use_nsa = is_deepseek_nsa(model_runner.model_config.hf_config)
        assert self.use_nsa, "NSA backend only supports DeepSeek NSA"
        self.nsa_kv_cache_store_fp8 = (
            model_runner.token_to_kv_pool.nsa_kv_cache_store_fp8
        )
        self.nsa_index_topk = get_nsa_index_topk(model_runner.model_config.hf_config)
        self.max_context_len = model_runner.model_config.context_len
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_cache_dim = model_runner.token_to_kv_pool.kv_cache_dim

        assert model_runner.req_to_token_pool is not None
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        global NSA_PREFILL_IMPL, NSA_DECODE_IMPL
        NSA_PREFILL_IMPL = model_runner.server_args.nsa_prefill
        NSA_DECODE_IMPL = model_runner.server_args.nsa_decode

        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)

        if _is_hip:
            max_bs = model_runner.req_to_token_pool.size

            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

        self.GLOBAL_TMP_COUNTER = 0

    def get_device_int32_arange(self, l: int) -> torch.Tensor:
        if l > len(self._arange_buf):
            next_pow_of_2 = 1 << (l - 1).bit_length()
            self._arange_buf = torch.arange(
                next_pow_of_2, device=self.device, dtype=torch.int32
            )
        return self._arange_buf[:l]

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        assert (
            forward_batch.spec_info is None
        ), "Spec decoding is not supported for NSA backend now"
        cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        assert forward_batch.seq_lens_cpu is not None
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            cu_seqlens_q = self.get_device_int32_arange(batch_size + 1)
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None
            if any(forward_batch.extend_prefix_lens_cpu):
                max_seqlen_q = max(extend_seq_lens_cpu)
                cu_seqlens_q = compute_cu_seqlens(
                    forward_batch.extend_seq_lens.to(torch.int32)
                )
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )
        else:
            assert False, f"Unsupported {forward_batch.forward_mode = }"

        # 1D, expanded seqlens (1D means cheap to compute, so always compute it)
        nsa_cache_seqlens_int32 = compute_nsa_seqlens(
            original_seq_lens=seqlens_expanded,
            nsa_index_topk=self.nsa_index_topk,
        )
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))

        nsa_extend_seq_to_req = torch.zeros(
            cu_seqlens_q[-1], dtype=torch.int32, device=device
        )
        for b in range(cu_seqlens_q.size(0) - 1):
            nsa_extend_seq_to_req[cu_seqlens_q[b]: cu_seqlens_q[b + 1]] = b

        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table,
            flashmla_metadata=(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens_int32,
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
                if NSA_DECODE_IMPL == "flashmla_decode"
                else None
            ),
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            nsa_extend_seq_lens_list=extend_seq_lens_cpu,
            real_page_table=self._transform_table_1_to_real(page_table),
            nsa_extend_seq_to_req=nsa_extend_seq_to_req,
            nsa_non_padded=cu_seqlens_q[-1:],
        )

        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.decode_cuda_graph_metadata: Dict = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            # fake page_table for sparse_prefill
            "page_table": torch.zeros(
                max_bs,
                self.max_context_len,
                dtype=torch.int32,
                device=self.device,
            ),
            "flashmla_metadata": (
                self._compute_flashmla_metadata(
                    cache_seqlens=torch.ones(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
                if NSA_DECODE_IMPL == "flashmla_decode"
                else None
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        assert forward_mode.is_decode_or_idle(), "Only support decode for now"
        assert (
            spec_info is None
        ), "Speculative decoding is not supported for NSA backend now"

        # Normal Decode
        # Get sequence information
        cache_seqlens_int32 = seq_lens.to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)

        # Use max context length for seq_len_k
        page_table_1 = self.decode_cuda_graph_metadata["page_table"][:bs, :]
        max_seq_len_k = page_table_1.shape[1]

        # Precompute page table
        # Precompute cumulative sequence lengths

        # NOTE(dark): this is always arange, since we are decoding
        cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][: bs + 1]
        nsa_cache_seqlens_int32 = compute_nsa_seqlens(
            cache_seqlens_int32, nsa_index_topk=self.nsa_index_topk
        )
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))
        real_page_table = self._transform_table_1_to_real(page_table_1)

        if NSA_DECODE_IMPL == "flashmla_decode":
            flashmla_metadata = self.decode_cuda_graph_metadata[
                "flashmla_metadata"
            ].slice(slice(0, bs + 1))
            flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens_int32,
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
            )
        else:
            flashmla_metadata = None


        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=1,
            max_seq_len_k=max_seq_len_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            flashmla_metadata=flashmla_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=cache_seqlens_int32,
            real_page_table=real_page_table,
            nsa_extend_seq_lens_list=[1] * bs,
            # NOTE: what about MTP
            nsa_non_padded=torch.tensor([bs], dtype=torch.int32, device=self.device)
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
        raw_num_token: Optional[int] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        assert seq_lens_cpu is not None
        assert forward_mode.is_decode_or_idle(), "Only support decode for now"
        assert (
            spec_info is None
        ), "Speculative decoding is not supported for NSA backend now"
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Normal Decode
        metadata: NSAMetadata = self.decode_cuda_graph_metadata[bs]
        max_len = int(seq_lens_cpu.max().item())

        cache_seqlens = seq_lens.to(torch.int32)
        metadata.cache_seqlens_int32.copy_(cache_seqlens)
        metadata.cu_seqlens_k[1:].copy_(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
        )
        page_indices = self.req_to_token[req_pool_indices, :max_len]
        metadata.page_table_1[:, :max_len].copy_(page_indices)
        assert (
            metadata.nsa_cache_seqlens_int32 is not None
            and metadata.nsa_cu_seqlens_k is not None
            and self.nsa_index_topk is not None
        )
        nsa_cache_seqlens = compute_nsa_seqlens(cache_seqlens, self.nsa_index_topk)
        metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
        metadata.nsa_cu_seqlens_k[1:].copy_(
            torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32)
        )
        # NOTE(dark): (nsa-) cu_seqlens_q is always arange, no need to copy

        # NOTE: what about MTP
        metadata.nsa_non_padded.fill_(raw_num_token)

        assert self.real_page_size == metadata.page_size
        if self.real_page_size > 1:
            real_table = self._transform_table_1_to_real(page_indices)
            new_len = real_table.shape[1]
            metadata.real_page_table[:, :new_len].copy_(real_table)
        else:
            assert metadata.real_page_table is metadata.page_table_1

        if NSA_DECODE_IMPL == "flashmla_decode":
            metadata.flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens,
                    seq_len_q=1,  # TODO handle MTP which is not 1
                )
            )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # if layer.layer_id == 0 and topk_indices is not None:
        #     print(f"{topk_indices.shape=} {topk_indices=}")
        # if layer.layer_id == 0:
        #     print(f"{q_rope.shape=} {k_rope.shape=} {forward_batch.seq_lens=}")

        if (
            os.getenv("SGLANG_NSA_DEBUG_DUMP_TOPK_INDICES") == "1"
            and topk_indices is not None
            and topk_indices.shape[0] > 10
            and topk_indices.shape[0] < 512
        ):
            if layer.layer_id == 0:
                self.GLOBAL_TMP_COUNTER += 1
            torch.save((topk_indices, forward_batch.seq_lens), f"/workspace/loopserve_nsa_topk_indices/layer{layer.layer_id}/{self.GLOBAL_TMP_COUNTER}.pt")

        assert (
            not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        ), "NSA backend doesn't support speculative decoding"
        metadata = self.forward_metadata
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                    metadata=metadata,
                    host_transfer_req_pool_indices=forward_batch.req_pool_indices_cpu,
                )

        causal = not layer.is_cross_attention
        assert causal, "NSA is causal only"

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}

        # Do absorbed multi-latent attention
        assert q_rope is not None
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        if NSA_FUSE_TOPK:
            page_table_1 = topk_indices
        else:
            assert metadata.nsa_extend_seq_lens_list is not None
            page_table_1 = transform_index_page_table_prefill(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                page_size=1,
            )

        if forward_batch.token_to_kv_pool.device == "cpu":
            kv_cache = forward_batch.token_to_kv_pool.device_pool.get_key_buffer(layer.layer_id)
            device_page_table_1 = forward_batch.token_to_kv_pool.host_token_to_device[
                layer.layer_id, page_table_1
            ]
            # print(layer.layer_id, device_page_table_1.shape, device_page_table_1[-20:])

            if os.getenv("EXTEND_OFFLOAD_DEBUG"):
                _device_page_table_1 = torch.where(device_page_table_1 == I32_MAX, -1, device_page_table_1)
                _used_device = torch.zeros(228000, dtype=bool, device=self.device)
                _used_device[_device_page_table_1.flatten()] = 1
                _used_device = torch.sum(_used_device[:-1])

                _page_table_1 = torch.where(device_page_table_1 == I32_MAX, page_table_1, -1)
                _used_host = torch.zeros(128000, dtype=bool, device=self.device)
                _used_host[_page_table_1.flatten()] = 1
                _used_host = torch.sum(_used_host[:-1])

                _used = _used_host + _used_device
                if _used > kv_cache.shape[0]:
                    raise RuntimeError(
                        f"{_used.item()=} {_used_host.item()=} {_used_device.item()=} but {kv_cache.shape[0]=}"
                    )
            if NSA_USE_CUDA_GRAPH:
                # pass
                recall_host_indices = torch.where(
                    device_page_table_1 == I32_MAX, page_table_1, -1
                )   # [q_len, topk]
                forward_batch.token_to_kv_pool.recall_miss_tokens_extend_cuda_graph(
                    recall_host_indices, device_page_table_1, layer.layer_id
                )
                device_page_table_1 = forward_batch.token_to_kv_pool.host_token_to_device[
                    layer.layer_id, page_table_1
                ]
            else:
                raise NotImplementedError

            if os.getenv("EXTEND_OFFLOAD_DEBUG"):
                if device_page_table_1.max().item() >= kv_cache.shape[0]:
                    raise RuntimeError(
                        f"{layer.layer_id=} {device_page_table_1.max()=}, some tokens are not in device cache"
                    )
            q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_prefill(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=device_page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )

        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        # when store in fp8 and compute in fp8, no need to convert dtype
        if not (
            NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 and self.nsa_kv_cache_store_fp8
        ):
            kv_cache = kv_cache.to(q.dtype)

        if NSA_PREFILL_IMPL == "tilelang":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif NSA_PREFILL_IMPL == "flashmla_prefill":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_prefill(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif NSA_PREFILL_IMPL == "flashmla_decode":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_decode(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif NSA_PREFILL_IMPL == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        else:
            raise ValueError(f"Unsupported {NSA_PREFILL_IMPL = }")

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-nt attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                    metadata=self.forward_metadata,
                    host_transfer_req_pool_indices=forward_batch.req_pool_indices_cpu,
                )

        metadata = self.forward_metadata
        causal = not layer.is_cross_attention
        assert causal, "NSA is causal only"

        # Do absorbed multi-latent attention
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        if NSA_FUSE_TOPK:
            page_table_1 = topk_indices
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        if forward_batch.token_to_kv_pool.device == "cpu":
            kv_cache = forward_batch.token_to_kv_pool.device_pool.get_key_buffer(layer.layer_id)

            device_page_table_1 = forward_batch.token_to_kv_pool.host_token_to_device[
                layer.layer_id, page_table_1
            ]
            if NSA_USE_CUDA_GRAPH:
                if NSA_FUSE_LOGITS_RECALL:
                    need_device = device_page_table_1 >= kv_cache.shape[0]
                    recall_host_indices = torch.where(
                        need_device, page_table_1, -1
                    )   # [q_len, topk]
                    recall_size = torch.sum(need_device).to(torch.int32)
                else:
                    recall_host_indices = torch.where(
                        device_page_table_1 == I32_MAX, page_table_1, -1
                    )   # [q_len, topk]
                    recall_size = torch.sum(recall_host_indices >= 0).to(torch.int32)
                # if torch.any(device_page_table_1 == I32_MAX):
                # print(f"{layer.layer_id=} {recall_size=}", flush=True)

                if os.getenv("STAT_RECALL") is not None:
                    need_recall = device_page_table_1 == I32_MAX
                    query_recall_size = torch.sum(need_recall, dim=-1)
                    topk = recall_host_indices.shape[-1]
                    with open(f'recall-rate/{os.getenv("STAT_RECALL")}-gpu{query_recall_size.device.index}.txt', 'a') as f:
                        print(f"{layer.layer_id=} {metadata.nsa_non_padded=} {query_recall_size=} {query_recall_size/topk=}", file=f)

                if NSA_FUSE_LOGITS_RECALL:
                    q_len = page_table_1.size(0)
                    forward_batch.token_to_kv_pool.recall_miss_tokens_decode_cuda_graph_with_prefetch(
                        recall_host_indices,
                        recall_size,
                        device_page_table_1,
                        layer.layer_id,
                        forward_batch.token_to_kv_pool.decode_prefetch_host_loc[:q_len],
                        forward_batch.token_to_kv_pool.decode_prefetch_kv_buf[:q_len],
                        forward_batch.token_to_kv_pool.query_recall_counter[:q_len],
                    )
                else:
                    forward_batch.token_to_kv_pool.recall_miss_tokens_decode_cuda_graph(
                        recall_host_indices,
                        recall_size,
                        device_page_table_1,
                        layer.layer_id,
                    )
            else:
                recall_host_indices = page_table_1[device_page_table_1 == I32_MAX]
                recall_size = recall_host_indices.shape[0]
                if recall_size > 0:
                    forward_batch.token_to_kv_pool.recall_miss_tokens_decode(
                        recall_host_indices, recall_size, device_page_table_1, layer.layer_id,
                    )

            if not (NSA_USE_CUDA_GRAPH and NSA_FUSE_LOGITS_RECALL):
                device_page_table_1 = forward_batch.token_to_kv_pool.host_token_to_device[
                    layer.layer_id, page_table_1
                ]

            if NSA_USE_CUDA_GRAPH:
                def check_device_pool(
                    device_page_table_1: torch.Tensor,
                    pool_size: int,
                ):
                    # NOTE: only work when set TRITON_DEBUG, consider using a flag?
                    qlen, topk = device_page_table_1.shape
                    BLOCK = 128
                    grid = (qlen, triton.cdiv(topk, BLOCK))
                    check_device_pool_kernel[grid](
                        device_page_table_1,
                        pool_size,
                        device_page_table_1.stride(0),
                        BLOCK,
                    )
                
                check_device_pool(device_page_table_1, kv_cache.shape[0])
            else:
                if device_page_table_1.max().item() >= kv_cache.shape[0]:
                    raise RuntimeError(
                        f"{layer.layer_id=} {device_page_table_1.max()=}, some tokens are not in device cache"
                    )
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=device_page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )

        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        if NSA_DECODE_IMPL == "flashmla_prefill":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_prefill(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif NSA_DECODE_IMPL == "flashmla_decode":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_flashmla_decode(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif NSA_DECODE_IMPL == "tilelang":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif NSA_DECODE_IMPL == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif NSA_DECODE_IMPL == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        else:
            assert False, f"Unsupported {NSA_DECODE_IMPL = }"

    def _forward_fa3(
        self,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        q_nope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        sm_scale: float,
        logit_cap: float,
        page_size: int,
    ) -> torch.Tensor:
        k_rope_cache = kv_cache[:, :, v_head_dim:]
        c_kv_cache = kv_cache[:, :, :v_head_dim]
        qk_rope_dim = k_rope_cache.shape[-1]
        k_rope_cache = k_rope_cache.view(-1, page_size, 1, qk_rope_dim)
        c_kv_cache = c_kv_cache.view(-1, page_size, 1, v_head_dim)
        o = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=sm_scale,
            causal=True,
            softcap=logit_cap,
            return_softmax_lse=False,
            num_splits=self.num_splits,
        )
        return o  # type: ignore

    def _forward_flashmla_prefill(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from flash_mla import flash_mla_sparse_fwd

        o, _, _ = flash_mla_sparse_fwd(
            q=q_all,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )
        return o

    def _forward_flashmla_decode(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        sm_scale: float,
        layer,
        metadata: NSAMetadata,
        page_table_1,
    ) -> torch.Tensor:
        from flash_mla import flash_mla_with_kvcache

        cache_seqlens = metadata.nsa_cache_seqlens_int32

        # TODO the 2nd dim is seq_len_q, need to be >1 when MTP
        q_all = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        kv_cache = kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)
        assert self.real_page_size == 64, "only page size 64 is supported"

        if NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 and not self.nsa_kv_cache_store_fp8:
            # inefficiently quantize the whole cache
            kv_cache = quantize_k_cache(kv_cache)

        indices = page_table_1.unsqueeze(1)
        assert (
            indices.shape[-1] == self.nsa_index_topk
        )  # requirement of FlashMLA decode kernel

        o, _ = flash_mla_with_kvcache(
            q=q_all,
            k_cache=kv_cache,
            cache_seqlens=cache_seqlens,
            head_dim_v=v_head_dim,
            tile_scheduler_metadata=metadata.flashmla_metadata.flashmla_metadata,
            num_splits=metadata.flashmla_metadata.num_splits,
            softmax_scale=sm_scale,
            indices=indices,
            # doc says it is not used, but if pass in None then error
            block_table=torch.empty(
                (q_all.shape[0], 0), dtype=torch.int32, device=q_all.device
            ),
            is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
        )
        return o

    def _forward_tilelang(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.tilelang_kernel import tilelang_sparse_fwd

        return tilelang_sparse_fwd(
            q=q_all,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )

    def _forward_aiter(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
        metadata: NSAMetadata,
        bs: int,
    ) -> torch.Tensor:
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        kv_indptr = self.kv_indptr

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)
        kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

        kv_indices = page_table_1[page_table_1 != -1]

        mla_decode_fwd(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            metadata.cu_seqlens_q,
            kv_indptr,
            kv_indices,
            metadata.cu_seqlens_q,
            metadata.max_seq_len_q,
            layer.scaling,
            layer.logit_cap,
        )
        # kv_cache = kv_cache.view(-1, 1, layer.head_dim)
        return o

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> NSAIndexerMetadata:
        return NSAIndexerMetadata(attn_metadata=self.forward_metadata)

    def _compute_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        from flash_mla import get_mla_metadata

        flashmla_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            num_q_tokens_per_head_k=seq_len_q * self.num_q_heads // 1,
            num_heads_k=1,
            num_heads_q=self.num_q_heads,
            is_fp8_kvcache=NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
            topk=self.nsa_index_topk,
        )

        return NSAFlashMLAMetadata(
            flashmla_metadata=flashmla_metadata,
            num_splits=num_splits,
        )


@triton.jit
def check_device_pool_kernel(
    device_page_table_ptr,
    device_pool_size: tl.constexpr,
    device_page_table_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    qid = tl.program_id(0)
    blk = tl.program_id(1)

    offs = qid * device_page_table_stride + blk * BLOCK + tl.arange(0, BLOCK)
    device_index = tl.load(device_page_table_ptr + offs)
    m = tl.max(device_index)
    tl.device_assert(m < device_pool_size)
