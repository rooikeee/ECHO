"""Opt-in NSA pool with complete mapped-host KV and bounded layer caches."""

import logging

import torch

from sglang.srt.hardware_backend.npu.echo_recall import EchoRecallCache
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


class EchoNPUMLATokenToKVPool(NPUMLATokenToKVPool):
    def __init__(self, *, size, page_size, dtype, kv_lora_rank, qk_rope_head_dim,
                 index_head_dim, layer_num, device, enable_memory_saver,
                 cache_size, start_layer=None, end_layer=None,
                 attention_tp_rank=0, attention_tp_size=1,
                 attention_dp_rank=0, attention_dp_size=1):
        if not (0 <= attention_tp_rank < attention_tp_size
                and 0 <= attention_dp_rank < attention_dp_size):
            raise ValueError("Invalid ECHO attention TP/DP rank topology")
        if enable_memory_saver:
            raise ValueError("ECHO mapped buffers do not support memory saver")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("ECHO NSA serving currently requires BF16 or FP16 KV")
        if index_head_dim is None or cache_size > size:
            raise ValueError("ECHO needs NSA index K and cache_size <= logical token capacity")
        KVCache.__init__(
            self, size=size, page_size=page_size, dtype=dtype, layer_num=layer_num,
            device=device, enable_memory_saver=False,
            start_layer=start_layer, end_layer=end_layer,
        )
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.cache_size = cache_size
        # IDs are local to this attention DP namespace; never share mappings
        # or registered host storage between DP ranks.
        self.attention_tp_rank = attention_tp_rank
        self.attention_tp_size = attention_tp_size
        self.attention_dp_rank = attention_dp_rank
        self.attention_dp_size = attention_dp_size
        self.echo_errors = torch.zeros(layer_num, dtype=torch.int32, device=device)
        self.index_k_buffer = torch.zeros(
            (layer_num, size // page_size + 1, page_size, 1, index_head_dim),
            dtype=dtype, device=device,
        )
        self.echo_caches = []
        try:
            for local_layer in range(layer_num):
                cache = EchoRecallCache(
                    size + page_size, cache_size, kv_lora_rank,
                    qk_rope_head_dim, dtype, device,
                )
                cache.sticky_errors = self.echo_errors[local_layer:local_layer + 1]
                self.echo_caches.append(cache)
        except Exception:
            for cache in self.echo_caches:
                cache.close()
            raise
        self._finalize_allocation_log(size)
        logger.info(
            "ECHO exact recall enabled: logical tokens=%d, cache tokens/layer=%d, "
            "mapped host KV=%.3f GiB; no fused prefetch; "
            "write_policy=device_first, host_backup=same_stream, "
            "planner=epoch_hit_fastpath, small_write=free_stack, eviction=bounded_lru",
            size, cache_size,
            sum(cache.host.tensor.nbytes for cache in self.echo_caches) / (1 << 30),
        )
        logger.info(
            "ECHO parallel layout: attention_tp_rank=%d/%d, attention_dp_rank=%d/%d; "
            "logical_capacity_per_rank=%d, hot_cache_per_layer_per_rank=%d",
            attention_tp_rank, attention_tp_size, attention_dp_rank, attention_dp_size,
            size, cache_size,
        )

    def get_echo_cache(self, layer_id):
        return self.echo_caches[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id):
        cache = self.get_echo_cache(layer_id)
        return (
            cache.key.view(-1, 128, 1, self.kv_lora_rank),
            cache.rope.view(-1, 128, 1, self.qk_rope_head_dim),
        )

    def get_key_buffer(self, layer_id):
        return self.get_kv_buffer(layer_id)[0]

    def get_value_buffer(self, layer_id):
        return self.get_kv_buffer(layer_id)[1]

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        if cache_v is None:
            cache_k, cache_v = cache_k.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        self.get_echo_cache(layer.layer_id).write(
            loc.contiguous(),
            cache_k.to(self.dtype).view(-1, 1, self.kv_lora_rank).contiguous(),
            cache_v.to(self.dtype).view(-1, 1, self.qk_rope_head_dim).contiguous(),
        )

    def release_echo_tokens(self, loc):
        for cache in self.echo_caches:
            cache.release(loc.contiguous())

    def reset_echo_state(self, preserve_written=False):
        for cache in self.echo_caches:
            cache.clear(preserve_written=preserve_written)

    def check_echo_errors(self):
        values = self.echo_errors.cpu().tolist()
        failures = [(self.start_layer + i, value) for i, value in enumerate(values) if value]
        if failures:
            raise RuntimeError(f"ECHO recall failed in layers {failures}; 1=invalid ID, 2=unwritten KV, 4=capacity")

    def get_kv_size_bytes(self):
        size = self.index_k_buffer.nbytes + self.echo_errors.nbytes
        for cache in self.echo_caches:
            size += sum(value.nbytes for value in vars(cache).values() if isinstance(value, torch.Tensor))
        return size

    def get_contiguous_buf_infos(self):
        raise NotImplementedError("ECHO does not support KV disaggregation")

    def close(self):
        for cache in self.echo_caches:
            cache.close()
