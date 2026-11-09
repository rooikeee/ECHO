"""Experimental ECHO MQA + bounded staging on Ascend 950/CANN 9.20.

State tensors must be updated in-place between replays. Missing map entries
acquire D + flattened-staging-slot tags. Consumers wait for launch completion.
"""
from dataclasses import dataclass, field

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
    resident_capacity: int
    _mapping_keepalive: object = field(default=None, init=False, repr=False)
    _buffer_identity: tuple = field(default=(), init=False, repr=False)
    _mapping_identity: tuple = field(default=(), init=False, repr=False)

    def __post_init__(self):
        rows, capacity = self.hosts.shape
        if (not isinstance(self.resident_capacity, int) or self.resident_capacity <= 0
                or self.resident_capacity + rows * capacity > 2**31 - 1):
            raise ValueError("resident_capacity + rows * capacity must fit int32")
        self._buffer_identity = self._identity()

    def _identity(self):
        return (self.resident_capacity, tuple(
            (t.data_ptr(), t.device, tuple(t.shape), tuple(t.stride()))
            for t in (self.logits, self.metadata, self.counts, self.tokens, self.hosts, self.staging)))

    def _check_identity(self):
        if self._identity() != self._buffer_identity:
            raise ValueError("Keep resident_capacity and staging/output buffers fixed for this instance")
        if self._mapping_keepalive is not None:
            if self._map_identity(self._mapping_keepalive) != self._mapping_identity:
                raise ValueError("Keep the bound host_to_device buffer address and layout fixed")

    @staticmethod
    def _map_identity(mapping):
        return (mapping.data_ptr(), mapping.device, tuple(mapping.shape), tuple(mapping.stride()))

    @classmethod
    def allocate(cls, rows, max_context, capacity, source, *, resident_capacity, source_dva=None,
                 source_kind="HBM-diagnostic"):
        from . import get_num_sms
        if min(rows, max_context, capacity) <= 0 or capacity > 2048:
            raise ValueError("positive shape and capacity <= 2048 required")
        if (not isinstance(resident_capacity, int) or resident_capacity <= 0
                or resident_capacity + rows * capacity > 2**31 - 1):
            raise ValueError("resident_capacity + rows * capacity must fit int32")
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
            torch.zeros(rows, dtype=torch.int32, device=device),
            torch.full((rows, capacity), -1, dtype=torch.int32, device=device),
            torch.full((rows, capacity), -1, dtype=torch.int32, device=device),
            torch.empty((rows, capacity, source.shape[1]), dtype=torch.uint8, device=device),
            torch.empty((get_num_sms() + 1, 2), dtype=torch.int32, device=device),
            int(source_dva), source, source_kind, resident_capacity,
        )

    def run(self, q, q_scales, kv, weights, context_lens, pages, indices,
            thresholds, stable, active, budgets, host_ids, host_to_device, written):
        """Preallocated launch wrapper intended for graph capture; warm up JIT first.

        Native cleanup first retires prior staging tags, then resets outputs;
        it is captured and replayed before metadata and scoring. Never clear hosts
        before cleanup. The map and output addresses must stay fixed. Downstream
        map users must understand tagged locations, or run only after explicit
        clear_prefetch_mappings(). Keep this instance/source alive until completion.
        """
        from . import _C
        if host_to_device.numel() != self.source_keepalive.shape[0]:
            raise ValueError("resident mapping and source pool have different token counts")
        self._check_identity()
        if self._mapping_keepalive is None:
            self._mapping_keepalive = host_to_device
            self._mapping_identity = self._map_identity(host_to_device)
        elif self._map_identity(host_to_device) != self._mapping_identity:
            raise ValueError("Reuse the same host_to_device buffer; clear and allocate a new instance to replace it")
        self.logits.fill_(float("-inf"))
        _C.echo_paged_mqa_prefetch_out(
            q, q_scales, kv, weights, context_lens, pages, indices, self.metadata,
            thresholds, stable, active, budgets, host_ids, host_to_device, written,
            self.logits, self.counts, self.tokens, self.hosts, self.staging, self.source_dva, self.resident_capacity,
        )
        return self.logits

    def clear_prefetch_mappings(self):
        """Retire this instance's tags after its final consumer, on the same stream.

        Already promoted resident entries survive. Call before abandoning or
        replacing this instance; wait for completion before freeing its buffers.
        Clearing is idempotent and does not unbind the map or change graph addresses.
        """
        self._check_identity()
        if self._mapping_keepalive is not None:
            from . import _C
            _C.echo_clear_prefetch_mappings(
                self._mapping_keepalive, self.counts, self.tokens, self.hosts, self.resident_capacity,
            )
