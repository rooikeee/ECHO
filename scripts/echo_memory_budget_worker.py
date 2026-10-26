"""Experiment worker: cap weights plus resident cache at 90% of device RAM."""

import json

import torch
from vllm.logger import init_logger
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.worker.worker import NPUWorker

logger = init_logger(__name__)


def resident_bytes(config, layers, hot_tokens, block_size, heads_dim):
    rows = min(config.max_num_batched_tokens, config.max_num_seqs * 4)
    rows = max(rows, 2 * (config.max_num_seqs + 2))
    return layers * rows * (hot_tokens + 2 * block_size) * heads_dim * 2


class EchoMemoryBudgetWorker(NPUWorker):
    @torch.inference_mode()
    def determine_available_memory(self):
        weights = int(self.model_runner.model_memory_usage)
        budget = int(self.init_snapshot.total_memory * 0.9)
        specs = self.get_kv_cache_spec()
        cfg = get_ascend_config().sparse_kv_offload_config
        hot = 0
        if cfg.enabled:
            layers = sum(bool(getattr(s, "store_on_host", False)) for s in specs.values())
            hf = self.model_config.hf_text_config
            hot = resident_bytes(
                self.scheduler_config,
                layers,
                cfg.topk_buffer_size,
                self.cache_config.block_size,
                hf.kv_lora_rank + hf.qk_rope_head_dim,
            )
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.shared_hot_layout import (
                shared_hot_layout,
            )

            layout = shared_hot_layout(self.vllm_config, cfg.topk_buffer_size, self.cache_config.block_size)
            if layout is not None:
                hot = layers * layout.allocation_rows * layout.stride * (hf.kv_lora_rank + hf.qk_rope_head_dim) * 2
        remaining = budget - weights - hot
        if remaining <= 0:
            raise ValueError(f"Weights plus hot cache exceed 90% budget: {weights=}, {hot=}, {budget=}")
        self.model_runner.profile_run()
        # The residual budget is for all device KV, including indexer KV.
        # Graph, activations, communication and driver allocations use the 10%.
        if cfg.enabled:
            plan = plan_sparse_kv_offload_memory(
                specs, self.vllm_config, remaining, int(cfg.dram_size_per_dp_GB * (1 << 30)), cfg.keep_device_kv_cache
            )
            # Do not shrink the allocated KV pool to the sequence safety cap.
            blocks = min(plan.npu_limit_blocks, plan.dram_limit_blocks)
            page_bytes = sum(s.page_size_bytes for s in specs.values())
            planned = blocks * page_bytes
        else:
            planned = self._apply_kvpp_memory_budget(remaining)
        print(
            "ECHO_MEMORY_BUDGET "
            + json.dumps(
                {
                    "rank": self.rank,
                    "total_bytes": self.init_snapshot.total_memory,
                    "weight_bytes": weights,
                    "resident_hot_bytes": hot,
                    "device_kv_budget_bytes": remaining,
                    "weight_cache_limit_bytes": budget,
                    "runtime_reserve_bytes": self.init_snapshot.total_memory - budget,
                    "planner_return_bytes": planned,
                }
            ),
            flush=True,
        )
        return planned
