"""Synthetic KV connector for ECHO-style decode-only performance experiments.

No real prompt KV is transferred. Generated text is NOT an accuracy result.
Uses upstream DecodeBench scheduling, with zero-initialized Ascend paged caches.
"""

import torch
from vllm.distributed import get_tp_group
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class EchoDecodeBenchConnector(DecodeBenchConnector):
    """Skip prompt computation while retaining real decode/cache allocation."""

    def register_kv_caches(self, kv_caches):
        # Ascend MLA caches are tuples of block-indexed tensors. Offload adds
        # CPU views on rank 0 and resident buffers; the other ranks use None
        # for the shared host views. Initialize once, matching the ECHO fake
        # transfer's allocated cache pools, rather than zeroing every request.
        for cache in kv_caches.values():
            tensors = (cache,) if isinstance(cache, torch.Tensor) else cache
            for tensor in tensors:
                if tensor is not None:
                    tensor.zero_()
        torch.npu.synchronize()
        get_tp_group().barrier()
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)
        logger.warning(
            "ECHO SYNTHETIC KV BENCHMARK: caches initialized to zero; prompt "
            "computation and KV transfer skipped. Output is not meaningful text."
        )

    def start_load_kv(self, forward_context, **kwargs):
        # Upstream scheduler accounts for external prompt tokens and allocates
        # real blocks. Pools are already initialized, so fake transfer is a
        # no-op, including capture/replay. No graph buffer is replaced here.
        return


class EchoSharedDecodeBenchConnector(EchoDecodeBenchConnector):
    """Queue on actual request-owned resident slot availability, not query count."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.resident_capacity = vllm_config.kv_transfer_config.kv_connector_extra_config["echo_shared_hot_cache"][
            "request_capacity"
        ]
        self.resident_owners = set()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.request_id not in self.resident_owners and len(self.resident_owners) >= self.resident_capacity:
            return None, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if num_external_tokens:
            self.resident_owners.add(request.request_id)

    def request_finished(self, request, block_ids):
        self.resident_owners.discard(request.request_id)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(self, request, block_ids):
        self.resident_owners.discard(request.request_id)
        return super().request_finished_all_groups(request, block_ids)
