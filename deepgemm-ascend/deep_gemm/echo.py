"""Experimental ECHO MQA + bounded staging on Ascend 950/CANN 9.20.

State tensors must be updated in-place between replays. The resident mapping is
read-only during a launch. Downstream exact recall still resolves TopK misses.
"""
from dataclasses import dataclass

import torch


@dataclass
class EchoPrefetchBuffers:
    logits: torch.Tensor
    counts: torch.Tensor
    tokens: torch.Tensor
    hosts: torch.Tensor
    staging: torch.Tensor
    metadata: torch.Tensor
    source_dva: int
    source_keepalive: object
    source_kind: str

    @classmethod
    def allocate(cls, rows, max_context, capacity, source, *, source_dva=None,
                 source_kind="HBM-diagnostic"):
        from . import get_num_sms
        if min(rows, max_context, capacity) <= 0 or capacity > 2048:
            raise ValueError("positive shape and capacity <= 2048 required")
        if source.ndim != 2 or source.dtype != torch.uint8 or not source.is_contiguous():
            raise ValueError("source must be contiguous uint8 [host_tokens, token_bytes]")
        if source.shape[0] < 2 or not 0 < source.shape[1] <= 65536:
            raise ValueError("source must include padding ID zero and token_bytes <= 65536")
        if source_kind not in ("HBM-diagnostic", "MemFabric-DDR"):
            raise ValueError("source_kind must explicitly identify HBM or MemFabric DDR")
        if source_kind == "MemFabric-DDR":
            if source.device.type != "cpu" or source_dva is None or source_dva <= 0:
                raise ValueError("DDR requires a MemFabric mapped CPU pool and its NPU DVA")
            device = torch.device("npu", torch.npu.current_device())
        else:
            if source.device.type != "npu":
                raise ValueError("HBM diagnostic source must be an NPU tensor")
            device = source.device
            if source_dva is not None and source_dva != source.data_ptr():
                raise ValueError("HBM source_dva must match its tensor address")
            source_dva = source.data_ptr()
        stride = (max_context + 255) // 256 * 256
        return cls(
            torch.empty_strided((rows, max_context), (stride, 1), dtype=torch.bfloat16, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
            torch.empty((rows, capacity), dtype=torch.int32, device=device),
            torch.empty((rows, capacity), dtype=torch.int32, device=device),
            torch.empty((rows, capacity, source.shape[1]), dtype=torch.uint8, device=device),
            torch.empty((get_num_sms() + 1, 2), dtype=torch.int32, device=device),
            int(source_dva), source, source_kind,
        )

    def run(self, q, q_scales, kv, weights, context_lens, pages, indices,
            thresholds, stable, active, budgets, host_ids, host_to_device, written):
        """Preallocated launch wrapper intended for graph capture; warm up JIT first.

        Resets are device ops captured and replayed before metadata and scoring.
        Staging bytes outside counts are invalid and deliberately left untouched.
        All referenced tensors/source pool must live until stream completion.
        """
        from . import _C
        if host_to_device.numel() != self.source_keepalive.shape[0]:
            raise ValueError("resident mapping and source pool have different token counts")
        self.counts.zero_()
        self.tokens.fill_(-1)
        self.hosts.fill_(-1)
        self.logits.fill_(float("-inf"))
        _C.echo_paged_mqa_prefetch_out(
            q, q_scales, kv, weights, context_lens, pages, indices, self.metadata,
            thresholds, stable, active, budgets, host_ids, host_to_device, written,
            self.logits, self.counts, self.tokens, self.hosts, self.staging, self.source_dva,
        )
        return self.logits
