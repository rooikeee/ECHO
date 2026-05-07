from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime
import functools
import gc
import io
import os
import random
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import tqdm

import deep_gemm
import recall_ops
from deep_gemm.testing import bench_kineto
from deep_gemm.utils import per_custom_dims_cast_to_fp8

print("DeepGEMM version:", deep_gemm.__version__)

INT32_MAX = 2**31 - 1
TOPK = 2048
REQ_PF_MAX = 256
DEBUG = os.environ.get("DEBUG", "0") == "1"
BENCH_MODE = os.environ.get("DEEPGEMM_BENCH_MODE", "kineto").lower()


# -------------------------------
# Utility helpers
# -------------------------------


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1

    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    x_fp8 = torch.empty((num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8)
    x_fp8[:, : block_size * head_dim] = x_scaled.view(num_blocks, block_size * head_dim).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(dtype=torch.uint8)

    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def iterable_generator(seq):
    for item in seq:
        yield item


def shuffle_seq_generator(iterable: Iterable):
    items = list(iterable)
    random.shuffle(items)
    for item in items:
        yield item


def _print_csv_row_when_ready(row: dict, printed_headers: set[tuple[str, ...]]):
    fieldnames = tuple(row.keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="")
    if fieldnames not in printed_headers:
        writer.writeheader()
        print(buf.getvalue(), flush=True)
        buf.seek(0)
        buf.truncate(0)
        printed_headers.add(fieldnames)
    writer.writerow(row)
    print(buf.getvalue(), flush=True)


# -------------------------------
# Config and fused-kernel context
# -------------------------------


@dataclass
class TestConfig:
    max_model_len: int
    num_kv_blocks: int
    batch_size: int
    device_pool_size: int
    host_pool_size: int
    seq_len: int

    # Derived by __post_init__: device_used_size = device_pool_size - 1 - device_free_size.
    device_free_size: int = 0

    # Number of GT top-k logits already present in device pool (across all batches).
    num_gt_topk_hit: int = 0

    # Number of logits above decode_topk_logits threshold per batch.
    candidate_count: int = 0

    # Constants
    next_n: int = 1
    num_heads: int = 64
    head_dim: int = 128
    kv_block_size: int = 64
    mla_head_dim: int = 576

    def __post_init__(self):
        self.device_used_size = self.device_pool_size - 1 - self.device_free_size

        assert self.next_n == 1, "Only next_n=1 is supported"
        assert self.max_model_len >= self.seq_len, (
            f"max_model_len {self.max_model_len} must be >= seq_len {self.seq_len}"
        )
        assert self.device_pool_size >= self.batch_size * TOPK + 1, (
            f"device_pool_size too small, got {self.device_pool_size}, "
            f"expected at least {self.batch_size * TOPK + 1 = }"
        )
        assert self.host_pool_size >= self.num_kv_blocks * self.kv_block_size, (
            f"host_pool_size too small, got {self.host_pool_size}, "
            f"expected at least {self.num_kv_blocks * self.kv_block_size = }"
        )
        min_num_kv_blocks = self.batch_size * ceil_div(self.seq_len, self.kv_block_size)
        assert self.num_kv_blocks >= min_num_kv_blocks, (
            f"num_kv_blocks too small, got {self.num_kv_blocks}, "
            f"expected at least {min_num_kv_blocks}"
        )

        assert self.device_used_size <= min(self.device_pool_size - 1, self.host_pool_size), (
            "device_used_size larger than pool sizes"
        )
        assert self.device_used_size <= self.batch_size * self.seq_len, (
            "device_used_size should be <= total number of working kv indices"
        )

        assert self.num_gt_topk_hit <= self.batch_size * TOPK, (
            "Too many GT top-k hits compared to batch_size * TOPK. "
            f"Got {self.num_gt_topk_hit = }, expected at most {self.batch_size * TOPK}"
        )
        assert self.device_used_size - self.num_gt_topk_hit <= self.batch_size * (self.seq_len - TOPK), (
            "Not enough non-topk indices to fill remaining device_used_size. "
            f"Got device_used_size {self.device_used_size}, num_gt_topk_hit {self.num_gt_topk_hit}, "
            f"expected at most {self.batch_size * (self.seq_len - TOPK)}"
        )
        assert self.device_used_size >= self.num_gt_topk_hit, (
            f"device_used_size {self.device_used_size} should be >= num_gt_topk_hit {self.num_gt_topk_hit}"
        )


@dataclass
class FuseV2Context:
    q: torch.Tensor
    q_fp8: torch.Tensor
    kv_cache: torch.Tensor
    fused_kv_cache_fp8: torch.Tensor
    weights: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    schedule_meta: torch.Tensor
    max_context_len: int
    page_table_1: torch.Tensor

    device_pool_buf: torch.Tensor
    _device_pool_buf: torch.Tensor
    host_pool_buf: torch.Tensor
    _host_pool_buf: torch.Tensor
    device_pool_loc_alloc_buf: torch.Tensor
    _device_pool_loc_alloc_buf: torch.Tensor
    device_pool_priority: torch.Tensor
    _device_pool_priority: torch.Tensor
    device_pool_loc_small_priority: torch.Tensor
    _device_pool_loc_small_priority: torch.Tensor

    device_token_to_host: torch.Tensor
    _device_token_to_host: torch.Tensor
    host_token_to_device: torch.Tensor
    _host_token_to_device: torch.Tensor

    prefetch_host_loc: torch.Tensor
    _prefetch_host_loc: torch.Tensor
    prefetch_kv_buf: torch.Tensor
    _prefetch_kv_buf: torch.Tensor

    recall_counter: torch.Tensor
    _recall_counter: torch.Tensor
    query_recall_counter: torch.Tensor
    _query_recall_counter: torch.Tensor

    decode_topk_logits: torch.Tensor
    _decode_topk_logits: torch.Tensor
    clean_logits: bool
    _clean_logits: bool

    config: TestConfig | None = None
    gt_logits: torch.Tensor | None = None

    def as_args(self) -> tuple:
        return (
            self.q_fp8,
            self.fused_kv_cache_fp8,
            self.weights,
            self.context_lens,
            self.block_table,
            self.schedule_meta,
            self.max_context_len,
            self.clean_logits,
        )

    def as_fused_v2_args(self) -> tuple:
        return (
            self.q_fp8,
            self.fused_kv_cache_fp8,
            self.weights,
            self.context_lens,
            self.block_table,
            self.schedule_meta,
            self.max_context_len,
            self.page_table_1,
            self.device_pool_buf,
            self.host_pool_buf,
            self.prefetch_host_loc,
            self.prefetch_kv_buf,
            self.host_token_to_device,
            self.query_recall_counter,
            self.decode_topk_logits,
            self.clean_logits,
        )

    @staticmethod
    def dummy() -> "FuseV2Context":
        z = torch.empty(0)
        return FuseV2Context(
            q=z,
            q_fp8=z,
            kv_cache=z,
            fused_kv_cache_fp8=z,
            weights=z,
            context_lens=z,
            block_table=z,
            schedule_meta=z,
            max_context_len=0,
            page_table_1=z,
            device_pool_buf=z,
            _device_pool_buf=z,
            host_pool_buf=z,
            _host_pool_buf=z,
            device_pool_loc_alloc_buf=z,
            _device_pool_loc_alloc_buf=z,
            device_pool_priority=z,
            _device_pool_priority=z,
            device_pool_loc_small_priority=z,
            _device_pool_loc_small_priority=z,
            device_token_to_host=z,
            _device_token_to_host=z,
            host_token_to_device=z,
            _host_token_to_device=z,
            prefetch_host_loc=z,
            _prefetch_host_loc=z,
            prefetch_kv_buf=z,
            _prefetch_kv_buf=z,
            recall_counter=z,
            _recall_counter=z,
            query_recall_counter=z,
            _query_recall_counter=z,
            decode_topk_logits=z,
            _decode_topk_logits=z,
            clean_logits=True,
            _clean_logits=True,
        )

    def record_state(self):
        self._device_pool_buf.copy_(self.device_pool_buf)
        self._host_pool_buf.copy_(self.host_pool_buf)
        self._device_pool_loc_alloc_buf.copy_(self.device_pool_loc_alloc_buf)
        self._device_pool_priority.copy_(self.device_pool_priority)
        self._device_pool_loc_small_priority.copy_(self.device_pool_loc_small_priority)
        self._device_token_to_host.copy_(self.device_token_to_host)
        self._host_token_to_device.copy_(self.host_token_to_device)
        self._prefetch_host_loc.copy_(self.prefetch_host_loc)
        self._prefetch_kv_buf.copy_(self.prefetch_kv_buf)
        self._recall_counter.copy_(self.recall_counter)
        self._query_recall_counter.copy_(self.query_recall_counter)
        self._decode_topk_logits.copy_(self.decode_topk_logits)
        self._clean_logits = self.clean_logits

    def restore_state(self):
        self.device_pool_buf.copy_(self._device_pool_buf)
        self.host_pool_buf.copy_(self._host_pool_buf)
        self.device_pool_loc_alloc_buf.copy_(self._device_pool_loc_alloc_buf)
        self.device_pool_priority.copy_(self._device_pool_priority)
        self.device_pool_loc_small_priority.copy_(self._device_pool_loc_small_priority)
        self.device_token_to_host.copy_(self._device_token_to_host)
        self.host_token_to_device.copy_(self._host_token_to_device)
        self.prefetch_host_loc.copy_(self._prefetch_host_loc)
        self.prefetch_kv_buf.copy_(self._prefetch_kv_buf)
        self.recall_counter.copy_(self._recall_counter)
        self.query_recall_counter.copy_(self._query_recall_counter)
        self.decode_topk_logits.copy_(self._decode_topk_logits)
        self.clean_logits = self._clean_logits

    @contextmanager
    def temp_state(self, record_first: bool = False, exception_raise: bool = False):
        if record_first:
            self.record_state()
        try:
            yield self
        except Exception as e:
            if exception_raise:
                raise e
            import traceback

            print("Exception suppressed in temp_state:", e)
            traceback.print_exc()
        finally:
            self.restore_state()

    def get_global_kv_indices_from_local_indices(self, local_indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(self.page_table_1, dim=1, index=local_indices)

    @staticmethod
    def make_set_from_global_kv_indices(global_kv_indices: torch.Tensor) -> tuple[list[set[int]], set[int]]:
        per_batch = [set(global_kv_indices[i].cpu().numpy().tolist()) for i in range(global_kv_indices.shape[0])]
        total = functools.reduce(lambda a, b: a.union(b), per_batch, set())
        return per_batch, total

    def get_host_indices_present_in_device_pool(self) -> set[int]:
        valid_mask = self.device_token_to_host != INT32_MAX
        device_pool_tokens = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        host_indices = self.device_token_to_host[device_pool_tokens]
        return set(host_indices.cpu().numpy().tolist())

    def get_in_kernel_prefetched(self) -> int:
        capped = torch.clamp(self.query_recall_counter.to(torch.int64), max=REQ_PF_MAX)
        return int(capped.sum().item())


# -------------------------------
# Context construction
# -------------------------------


def make_basic_params(config: TestConfig, ctx: FuseV2Context):
    q = torch.randn(
        (config.batch_size, config.next_n, config.num_heads, config.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8 = q.to(torch.float8_e4m3fn)

    kv_cache = torch.randn(
        (config.num_kv_blocks, config.kv_block_size, 1, config.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    fused_kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

    weights = torch.randn((config.batch_size * config.next_n, config.num_heads), device="cuda", dtype=torch.float32)
    context_lens = torch.full((config.batch_size,), config.seq_len, device="cuda", dtype=torch.int32)

    max_context_lens = context_lens.max().item()
    max_block_len = ceil_div(max_context_lens, config.kv_block_size)

    block_table = torch.zeros((config.batch_size, max_block_len), device="cuda", dtype=torch.int32)
    page_table_1 = torch.zeros((config.batch_size, config.max_model_len), device="cuda", dtype=torch.int32)

    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens,
        config.kv_block_size,
        deep_gemm.get_num_sms(),
    )

    block_index_sample_weight = torch.ones((config.num_kv_blocks,), device="cuda", dtype=torch.float32)
    sampled_block_indices = torch.multinomial(
        input=block_index_sample_weight,
        num_samples=config.batch_size * max_block_len,
        replacement=False,
    ).view(config.batch_size, max_block_len)
    block_table.copy_(sampled_block_indices)

    local_indices = torch.arange(max_context_lens, device="cuda")
    offsets = (local_indices % config.kv_block_size).unsqueeze(0)

    block_table_repeated = torch.repeat_interleave(block_table, repeats=config.kv_block_size, dim=1)
    page_table_1_block_index = block_table_repeated[:, :max_context_lens]
    page_table_1[:, :max_context_lens] = page_table_1_block_index * config.kv_block_size + offsets

    ctx.q = q
    ctx.q_fp8 = q_fp8
    ctx.kv_cache = kv_cache
    ctx.fused_kv_cache_fp8 = fused_kv_cache_fp8
    ctx.weights = weights
    ctx.context_lens = context_lens
    ctx.block_table = block_table
    ctx.schedule_meta = schedule_meta
    ctx.max_context_len = config.max_model_len
    ctx.page_table_1 = page_table_1


def make_buffers(config: TestConfig, ctx: FuseV2Context):
    device_pool_buf = torch.randn((config.device_pool_size, 1, config.mla_head_dim), device="cuda", dtype=torch.bfloat16)
    host_pool_buf = torch.empty(
        (config.host_pool_size, 1, config.mla_head_dim),
        pin_memory=True,
        device="cpu",
        dtype=torch.bfloat16,
    )
    host_pool_buf.fill_(1.0)

    device_pool_loc_alloc_buf = torch.zeros((config.device_pool_size - 1,), device="cuda", dtype=torch.int32)
    prefetch_host_loc = torch.full((config.batch_size, REQ_PF_MAX), -1, device="cuda", dtype=torch.int32)
    prefetch_kv_buf = torch.empty((config.batch_size, REQ_PF_MAX, config.mla_head_dim), device="cuda", dtype=torch.bfloat16)
    prefetch_kv_buf.zero_()

    ctx.device_pool_buf = device_pool_buf
    ctx._device_pool_buf = device_pool_buf.clone()
    ctx.host_pool_buf = host_pool_buf
    ctx._host_pool_buf = host_pool_buf.clone()
    ctx.device_pool_loc_alloc_buf = device_pool_loc_alloc_buf
    ctx._device_pool_loc_alloc_buf = device_pool_loc_alloc_buf.clone()
    ctx.prefetch_host_loc = prefetch_host_loc
    ctx._prefetch_host_loc = prefetch_host_loc.clone()
    ctx.prefetch_kv_buf = prefetch_kv_buf
    ctx._prefetch_kv_buf = prefetch_kv_buf.clone()


def make_counters(config: TestConfig, ctx: FuseV2Context):
    recall_counter = torch.zeros((1,), device="cuda", dtype=torch.uint32)
    query_recall_counter = torch.zeros((config.batch_size,), device="cuda", dtype=torch.uint32)

    ctx.recall_counter = recall_counter
    ctx._recall_counter = recall_counter.clone()
    ctx.query_recall_counter = query_recall_counter
    ctx._query_recall_counter = query_recall_counter.clone()


def make_pools(config: TestConfig, ctx: FuseV2Context):
    host_token_to_device = torch.full((config.host_pool_size + 1,), INT32_MAX, device="cuda", dtype=torch.int32)
    device_token_to_host = torch.full((config.device_pool_size,), INT32_MAX, device="cuda", dtype=torch.int64)
    device_pool_priority = torch.full((config.device_pool_size,), -1, device="cuda", dtype=torch.int32)
    device_pool_priority[0] = INT32_MAX

    ctx.host_token_to_device = host_token_to_device
    ctx.device_token_to_host = device_token_to_host
    ctx.device_pool_priority = device_pool_priority


def fill_pools(config: TestConfig, ctx: FuseV2Context):
    seq_len = config.seq_len
    batch_size = config.batch_size
    device_used_size = config.device_used_size
    num_gt_topk_hit = config.num_gt_topk_hit

    gt_topk_local_indices = torch.topk(ctx.gt_logits, k=TOPK, dim=1, sorted=False).indices
    gt_topk_global_indices_all_batches = torch.gather(ctx.page_table_1, dim=1, index=gt_topk_local_indices)

    global_kv_indices = ctx.get_global_kv_indices_from_local_indices(
        torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    )

    if num_gt_topk_hit == 0:
        sampled_topk_hit_global_indices = torch.empty((0,), device="cuda", dtype=torch.int64)
    else:
        global_hit_flat = gt_topk_global_indices_all_batches.flatten()
        hit_sample_weights = torch.ones_like(global_hit_flat, dtype=torch.float32)
        assert torch.numel(global_hit_flat) >= num_gt_topk_hit

        sampled_hit_indices = torch.multinomial(
            input=hit_sample_weights,
            num_samples=num_gt_topk_hit,
            replacement=False,
        )
        sampled_topk_hit_global_indices = torch.take(global_hit_flat, sampled_hit_indices)

        if DEBUG:
            assert len(set(sampled_topk_hit_global_indices.tolist())) == num_gt_topk_hit

    non_topk_count = device_used_size - num_gt_topk_hit
    if non_topk_count == 0:
        sampled_non_topk_global_indices = torch.empty((0,), device="cuda", dtype=torch.int64)
    else:
        non_hit_sample_weights = torch.ones_like(global_kv_indices, dtype=torch.float32)
        non_hit_sample_weights.scatter_(dim=1, index=gt_topk_local_indices, value=0.0)
        non_hit_sample_weights = non_hit_sample_weights.flatten()

        global_kv_flattened = global_kv_indices.flatten()
        assert torch.numel(global_kv_flattened) - torch.numel(gt_topk_local_indices) >= non_topk_count

        sampled_non_hit_indices = torch.multinomial(
            input=non_hit_sample_weights,
            num_samples=non_topk_count,
            replacement=False,
        )
        sampled_non_topk_global_indices = torch.take(global_kv_flattened, sampled_non_hit_indices)

        if DEBUG:
            assert len(set(sampled_non_topk_global_indices.tolist())) == non_topk_count

    sampled_host_pool_indices = torch.cat([sampled_topk_hit_global_indices, sampled_non_topk_global_indices], dim=0)

    if DEBUG:
        assert len(set(sampled_host_pool_indices.tolist())) == device_used_size

    device_pool_sample_weights = torch.ones((config.device_pool_size,), dtype=torch.float32, device="cuda")
    device_pool_sample_weights[0] = 0.0
    sampled_device_pool_indices = torch.multinomial(
        input=device_pool_sample_weights,
        num_samples=device_used_size,
        replacement=False,
    )

    ctx.host_token_to_device.scatter_(
        dim=0,
        index=sampled_host_pool_indices.to(torch.int64),
        src=sampled_device_pool_indices.to(torch.int32),
    )
    ctx.device_token_to_host.scatter_(
        dim=0,
        index=sampled_device_pool_indices.to(torch.int64),
        src=sampled_host_pool_indices.to(torch.int64),
    )

    random_priorities = torch.randint(1, 9999, (device_used_size,), device="cuda", dtype=torch.int32)
    ctx.device_pool_priority.scatter_(dim=0, index=sampled_device_pool_indices, src=random_priorities)

    _, device_pool_loc_small_priority = torch.sort(ctx.device_pool_priority[1:])
    device_pool_loc_small_priority = (device_pool_loc_small_priority + 1).to(torch.int32)

    ctx._host_token_to_device = ctx.host_token_to_device.clone()
    ctx._device_token_to_host = ctx.device_token_to_host.clone()
    ctx._device_pool_priority = ctx.device_pool_priority.clone()
    ctx.device_pool_loc_small_priority = device_pool_loc_small_priority
    ctx._device_pool_loc_small_priority = device_pool_loc_small_priority.clone()


def set_gt_logits(config: TestConfig, ctx: FuseV2Context) -> torch.Tensor:
    gt_logits = deep_gemm.fp8_paged_mqa_logits(
        ctx.q_fp8,
        ctx.fused_kv_cache_fp8,
        ctx.weights,
        ctx.context_lens,
        ctx.block_table,
        ctx.schedule_meta,
        config.max_model_len,
        clean_logits=True,
    )
    ctx.gt_logits = gt_logits
    return gt_logits


def set_decode_topk_logits(config: TestConfig, ctx: FuseV2Context):
    top_logits_sorted, _ = torch.topk(ctx.gt_logits, k=config.candidate_count + 1, dim=1, sorted=True)
    ctx.decode_topk_logits = top_logits_sorted[:, -1].clone()
    ctx._decode_topk_logits = ctx.decode_topk_logits.clone()


def make_fuse_v2_context(config: TestConfig) -> FuseV2Context:
    ctx = FuseV2Context.dummy()
    ctx.config = config

    make_basic_params(config, ctx)
    make_buffers(config, ctx)
    make_counters(config, ctx)
    set_gt_logits(config, ctx)
    make_pools(config, ctx)
    fill_pools(config, ctx)
    set_decode_topk_logits(config, ctx)

    ctx.clean_logits = True
    ctx._clean_logits = ctx.clean_logits
    return ctx


# -------------------------------
# Kernel dispatch and recall
# -------------------------------


def call_kernel(kernel_name: str, ctx: FuseV2Context) -> torch.Tensor:
    match kernel_name:
        case "v2" | "fused_v2":
            return deep_gemm.fp8_paged_mqa_logits_fused_v2(*ctx.as_fused_v2_args())
        case "vanilla":
            return deep_gemm.fp8_paged_mqa_logits(*ctx.as_args())
        case _:
            raise ValueError(f"Unknown kernel name: {kernel_name}")


def call_fuse_v2(ctx: FuseV2Context) -> torch.Tensor:
    return call_kernel("fused_v2", ctx)


@dataclass
class RecallContext:
    recall_host_indices: torch.Tensor
    recall_device_indices: torch.Tensor
    host_token_to_device: torch.Tensor
    device_token_to_host: torch.Tensor
    layer_device_kv: torch.Tensor
    layer_host_kv: torch.Tensor
    recall_counter: torch.Tensor

    recall_size: int
    snapshot: dict[str, torch.Tensor]

    def record_state(self):
        for field in self.__dataclass_fields__:
            if field == "snapshot":
                continue
            attr = getattr(self, field)
            self.snapshot[field] = attr.clone() if isinstance(attr, torch.Tensor) else attr

    def restore_state(self):
        for field, value in self.snapshot.items():
            attr = getattr(self, field)
            if isinstance(attr, torch.Tensor):
                attr.copy_(value)
            else:
                setattr(self, field, value)

    @contextmanager
    def temp_state(self, record_first: bool = False, exception_raise: bool = False):
        if record_first:
            self.record_state()
        try:
            yield self
        except Exception as e:
            if exception_raise:
                raise e
            import traceback

            print("Exception suppressed in temp_state:", e)
            traceback.print_exc()
        finally:
            self.restore_state()


def make_recall_context(
    gt_topk_global_indices: torch.Tensor,
    host_token_to_device: torch.Tensor,
    device_token_to_host: torch.Tensor,
    layer_device_kv: torch.Tensor,
    layer_host_kv: torch.Tensor,
    recall_counter: torch.Tensor,
) -> RecallContext:
    device_gt_topk_indices = host_token_to_device[gt_topk_global_indices]

    recall_host_indices = torch.where(device_gt_topk_indices == INT32_MAX, gt_topk_global_indices, -1)
    assert (recall_host_indices != INT32_MAX).all(), "recall_host_indices contains INT32_MAX"

    recall_size = int((recall_host_indices != -1).sum().item())

    device_pool_free_mask = device_token_to_host == INT32_MAX
    device_pool_free_mask[0] = 0
    device_pool_free_indices = torch.nonzero(device_pool_free_mask, as_tuple=False).squeeze(1)

    return RecallContext(
        recall_host_indices=recall_host_indices,
        recall_device_indices=device_pool_free_indices[:recall_size],
        host_token_to_device=host_token_to_device,
        device_token_to_host=device_token_to_host,
        layer_device_kv=layer_device_kv,
        layer_host_kv=layer_host_kv,
        recall_counter=recall_counter,
        recall_size=recall_size,
        snapshot={},
    )


def run_recall_update_kernel_inplace(recall_ctx: RecallContext):
    recall_ops.recall_update(
        recall_host_indices=recall_ctx.recall_host_indices,
        recall_device_indices=recall_ctx.recall_device_indices,
        host_token_to_device=recall_ctx.host_token_to_device,
        device_token_to_host=recall_ctx.device_token_to_host,
        layer_device_kv=recall_ctx.layer_device_kv,
        layer_host_kv=recall_ctx.layer_host_kv,
        recall_counter=recall_ctx.recall_counter,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )


def _kernel_fn_and_args(kernel_name: str, context: FuseV2Context):
    match kernel_name:
        case "vanilla":
            return deep_gemm.fp8_paged_mqa_logits, context.as_args
        case "v2":
            return deep_gemm.fp8_paged_mqa_logits_fused_v2, context.as_fused_v2_args
        case _:
            raise ValueError(f"Unknown kernel name: {kernel_name}")


def run_mixed(configs: list[TestConfig]) -> list[dict]:
    ret: list[dict] = []
    printed_csv_headers: set[tuple[str, ...]] = set()

    pbar = tqdm.tqdm(configs)
    for config in pbar:
        try:
            context = make_fuse_v2_context(config)
        except Exception as e:
            print(f"Failed to make context for config: {config}")
            print(e)
            raise

        for kernel_name in ["vanilla", "v2"]:
            kernel, get_args = _kernel_fn_and_args(kernel_name, context)
            total_candidate_count = config.batch_size * config.candidate_count

            with context.temp_state():
                _ = call_kernel(kernel_name, context)

                result = {
                    "kernel": kernel_name,
                    "recall_counter": None if kernel_name == "vanilla" else context.get_in_kernel_prefetched(),
                    "batch_size": config.batch_size,
                    "seq_len": config.seq_len,
                    "device_free_size": config.device_free_size,
                    "all_req_topk_hit": config.num_gt_topk_hit,
                    "hit_rate": round(
                        config.num_gt_topk_hit / total_candidate_count if total_candidate_count > 0 else 0.0,
                        3,
                    ),
                }

            runs = 15
            warmup = 3
            fn = lambda: kernel(*get_args())

            for _ in range(warmup):
                with context.temp_state():
                    fn()

            if BENCH_MODE == "kineto":
                def kineto_fn():
                    with context.temp_state(exception_raise=True):
                        fn()

                kernel_name_to_search = (
                    "fp8_paged_mqa_logits_fused_v2" if kernel_name == "v2" else "fp8_paged_mqa_logits"
                )
                kernel_time_s = bench_kineto(
                    kineto_fn,
                    kernel_name_to_search,
                    num_tests=runs,
                    suppress_kineto_output=True,
                )
                result["avg_time_logits_ms"] = round(kernel_time_s * 1e3, 3)
            elif BENCH_MODE == "event":
                total_time_logits_ms = 0.0
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                for _ in range(runs):
                    with context.temp_state(exception_raise=True):
                        torch.cuda.synchronize()
                        start_event.record()
                        fn()
                        end_event.record()
                        torch.cuda.synchronize()
                    total_time_logits_ms += start_event.elapsed_time(end_event)

                result["avg_time_logits_ms"] = round(total_time_logits_ms / runs, 3)
            else:
                raise ValueError(f"Unknown BENCH_MODE: {BENCH_MODE!r}")

            ret.append(result)
            _print_csv_row_when_ready(result, printed_csv_headers)

        del context
        gc.collect()

    return ret


def _build_recall_context_from_logits(
    ctx: FuseV2Context,
    logits: torch.Tensor,
    topk: int = TOPK,
) -> RecallContext:
    """Build a RecallContext from logits produced by either the vanilla or v2 kernel.

    The logits may contain -inf outside the valid context; topk naturally skips them
    because topk with k <= valid_len picks the largest values.

    Free device-pool indices are derived from the current `device_token_to_host`,
    so this correctly reflects pool state *after* the v2 kernel has consumed some
    free slots for its in-kernel prefetch.
    """
    topk_local = torch.topk(logits, k=topk, dim=1, sorted=False).indices
    topk_global = torch.gather(ctx.page_table_1, dim=1, index=topk_local)

    return make_recall_context(
        gt_topk_global_indices=topk_global,
        host_token_to_device=ctx.host_token_to_device,
        device_token_to_host=ctx.device_token_to_host,
        layer_device_kv=ctx.device_pool_buf,
        layer_host_kv=ctx.host_pool_buf,
        recall_counter=ctx.recall_counter,
    )


def _measure_kernel_ms(
    fn: Callable[[], None],
    kernel_name_to_search: str,
    context: FuseV2Context,
    runs: int,
    warmup: int,
) -> float:
    """Run `fn()` in a temp_state rollback and return avg kernel-only time in ms."""
    for _ in range(warmup):
        with context.temp_state():
            fn()

    if BENCH_MODE == "kineto":
        def kineto_fn():
            with context.temp_state(exception_raise=True):
                fn()

        kernel_time_s = bench_kineto(
            kineto_fn,
            kernel_name_to_search,
            num_tests=runs,
            suppress_kineto_output=True,
        )
        return kernel_time_s * 1e3

    if BENCH_MODE == "event":
        total_ms = 0.0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            with context.temp_state(exception_raise=True):
                torch.cuda.synchronize()
                start_event.record()
                fn()
                end_event.record()
                torch.cuda.synchronize()
            total_ms += start_event.elapsed_time(end_event)
        return total_ms / runs

    raise ValueError(f"Unknown BENCH_MODE: {BENCH_MODE!r}")


def _measure_recall_update_ms(
    recall_ctx: RecallContext,
    runs: int,
    warmup: int,
) -> float:
    """Measure only the `recall_update` Triton kernel time.

    The recall kernel atomically increments `recall_counter` once per valid
    host index and uses the returned position to index into
    `recall_device_indices`.  If we invoke it repeatedly without resetting
    the counter, subsequent iterations read beyond the valid slice and also
    pollute `host_token_to_device` / `device_token_to_host`, corrupting the
    pool state.  To keep every measured iteration doing the same work on
    the same state, we reset the counter to zero immediately before each
    launch.  (The H->D payload copy itself is idempotent for fixed inputs.)
    """
    counter = recall_ctx.recall_counter

    def _launch_once():
        counter.zero_()
        run_recall_update_kernel_inplace(recall_ctx)

    # Warmup (JIT compile, pinned H->D cache priming).
    for _ in range(warmup):
        _launch_once()

    if BENCH_MODE == "kineto":
        kernel_time_s = bench_kineto(
            _launch_once,
            "recall_update",
            num_tests=runs,
            suppress_kineto_output=True,
        )
        return kernel_time_s * 1e3

    if BENCH_MODE == "event":
        total_ms = 0.0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            counter.zero_()
            torch.cuda.synchronize()
            start_event.record()
            run_recall_update_kernel_inplace(recall_ctx)
            end_event.record()
            torch.cuda.synchronize()
            total_ms += start_event.elapsed_time(end_event)
        return total_ms / runs

    raise ValueError(f"Unknown BENCH_MODE: {BENCH_MODE!r}")


def run_decode_prefetch_mixed(configs: list[TestConfig]) -> list[dict]:
    """Benchmark: baseline (vanilla logits + full recall-miss) vs prefetch (v2 + residual recall).

    For each config we measure two flows:

    - "baseline": `fp8_paged_mqa_logits` produces logits; then `recall_update`
      transfers every top-k token that is missing from the device pool.
    - "prefetch": `fp8_paged_mqa_logits_fused_v2` produces logits *and*
      prefetches some top-k tokens during the kernel. Then `recall_update`
      transfers the residual misses (top-k tokens still missing after v2).

    Both phases are timed independently; total = logits + recall.
    """
    runs = 15
    warmup = 3
    ret: list[dict] = []
    printed_csv_headers: set[tuple[str, ...]] = set()

    pbar = tqdm.tqdm(configs)
    for config in pbar:
        try:
            context = make_fuse_v2_context(config)
        except Exception as e:
            print(f"Failed to make context for config: {config}")
            print(e)
            raise

        # --- Baseline: vanilla logits + recall-all-misses --------------------
        # Produce reference logits once; the vanilla kernel does not mutate
        # any pool state, so logits timing and the downstream recall reuse
        # the unchanged initial pool.
        with context.temp_state():
            vanilla_logits = deep_gemm.fp8_paged_mqa_logits(*context.as_args())

        vanilla_time_ms = _measure_kernel_ms(
            fn=lambda: deep_gemm.fp8_paged_mqa_logits(*context.as_args()),
            kernel_name_to_search="fp8_paged_mqa_logits",
            context=context,
            runs=runs,
            warmup=warmup,
        )

        with context.temp_state(record_first=True):
            vanilla_recall_ctx = _build_recall_context_from_logits(context, vanilla_logits)
            vanilla_recall_size = vanilla_recall_ctx.recall_size
            assert vanilla_recall_size <= vanilla_recall_ctx.recall_device_indices.shape[0], (
                "Not enough free device-pool slots for baseline recall; "
                "reduce num_gt_topk_hit or increase device_pool_size."
            )
            vanilla_recall_ms = _measure_recall_update_ms(vanilla_recall_ctx, runs=runs, warmup=warmup)
            # One extra invocation to capture how many tokens the kernel
            # actually copied (recall_counter after one fresh run); the
            # timing loop resets the counter per iteration, so we need a
            # standalone run to snapshot it.
            vanilla_recall_ctx.recall_counter.fill_(0)
            run_recall_update_kernel_inplace(vanilla_recall_ctx)
            vanilla_recall_counter = int(vanilla_recall_ctx.recall_counter.item())

        ret.append({
            "kernel": "baseline",
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "device_free_size": config.device_free_size,
            "all_req_topk_hit": config.num_gt_topk_hit,
            "hit_rate": round(
                config.num_gt_topk_hit / (config.batch_size * TOPK) if config.batch_size * TOPK > 0 else 0.0,
                3,
            ),
            "logits_ms": round(vanilla_time_ms, 3),
            "recall_size": vanilla_recall_size,
            "recall_counter": vanilla_recall_counter,
            "recall_ms": round(vanilla_recall_ms, 3),
            "total_ms": round(vanilla_time_ms + vanilla_recall_ms, 3),
            "in_kernel_prefetched": 0,
            "effective_prefetch": 0,
            "wasted_prefetch": 0,
        })
        _print_csv_row_when_ready(ret[-1], printed_csv_headers)

        # --- Prefetch: v2 logits + residual recall ----------------------------
        # v2 mutates pool state during the kernel. To measure the residual
        # recall fairly, we run v2 once (from the original snapshot), capture
        # the post-v2 pool state, and build the residual-recall context from
        # there. All timings use the per-iteration temp_state() rollback.
        v2_time_ms = _measure_kernel_ms(
            fn=lambda: deep_gemm.fp8_paged_mqa_logits_fused_v2(*context.as_fused_v2_args()),
            kernel_name_to_search="fp8_paged_mqa_logits_fused_v2",
            context=context,
            runs=runs,
            warmup=warmup,
        )

        # Run v2 once to realize the post-prefetch pool state, then measure
        # the residual recall against that state. Wrap everything in a
        # temp_state so the original pool is restored afterwards.
        with context.temp_state(record_first=True):
            v2_logits = deep_gemm.fp8_paged_mqa_logits_fused_v2(*context.as_fused_v2_args())
            in_kernel_prefetched = context.get_in_kernel_prefetched()

            v2_recall_ctx = _build_recall_context_from_logits(context, v2_logits)
            v2_recall_size = v2_recall_ctx.recall_size
            assert v2_recall_size <= v2_recall_ctx.recall_device_indices.shape[0], (
                "Not enough free device-pool slots for residual recall after v2."
            )
            v2_recall_ms = _measure_recall_update_ms(v2_recall_ctx, runs=runs, warmup=warmup)
            # Capture the actual recall_update counter on a fresh run so we
            # can distinguish "tokens actually transferred" from overall
            # launch overhead in the CSV.
            v2_recall_ctx.recall_counter.fill_(0)
            run_recall_update_kernel_inplace(v2_recall_ctx)
            v2_recall_counter = int(v2_recall_ctx.recall_counter.item())

        ret.append({
            "kernel": "prefetch",
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "device_free_size": config.device_free_size,
            "all_req_topk_hit": config.num_gt_topk_hit,
            "hit_rate": round(
                config.num_gt_topk_hit / (config.batch_size * TOPK) if config.batch_size * TOPK > 0 else 0.0,
                3,
            ),
            "logits_ms": round(v2_time_ms, 3),
            "recall_size": v2_recall_size,
            "recall_counter": v2_recall_counter,
            "recall_ms": round(v2_recall_ms, 3),
            "total_ms": round(v2_time_ms + v2_recall_ms, 3),
            "in_kernel_prefetched": in_kernel_prefetched,
            "effective_prefetch": vanilla_recall_size - v2_recall_size,
            "wasted_prefetch": in_kernel_prefetched - (vanilla_recall_size - v2_recall_size),
        })
        _print_csv_row_when_ready(ret[-1], printed_csv_headers)

        del context
        gc.collect()

    return ret


def plot_decode_prefetch_results(df: pd.DataFrame, output_csv: str):
    _plot_avg_time_logits_generic(
        df,
        output_csv,
        base_kernel="baseline",
        compare_kernel="prefetch",
        compare_cols=["batch_size", "seq_len", "device_free_size", "all_req_topk_hit", "hit_rate"],
        y_col="total_ms",
        ylabel="total time (logits + recall) ms",
    )


def _plot_avg_time_logits_generic(
    df: pd.DataFrame,
    output_csv: str,
    base_kernel: str,
    compare_kernel: str,
    compare_cols: list[str],
    y_col: str,
    ylabel: str,
):
    import matplotlib.pyplot as plt

    base_df = df[df["kernel"] == base_kernel].copy()
    compare_df = df[df["kernel"] == compare_kernel].copy()

    if len(base_df) != len(compare_df):
        raise ValueError(f"{base_kernel} and {compare_kernel} row counts do not match")

    if not base_df[compare_cols].reset_index(drop=True).equals(compare_df[compare_cols].reset_index(drop=True)):
        raise ValueError(f"{base_kernel} and {compare_kernel} rows are not aligned")

    x = base_df.index.to_numpy() + 2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, base_df[y_col].to_numpy(), marker="o", linewidth=1.5, label=base_kernel)
    ax.plot(x, compare_df[y_col].to_numpy(), marker="o", linewidth=1.5, label=compare_kernel)
    ax.set_xlabel(f"{base_kernel} Line Number In CSV")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = f"{os.path.splitext(output_csv)[0]}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {os.path.abspath(plot_path)}")


def main_decode_prefetch(output_csv: str):
    configs = build_default_configs()
    result_dicts = run_decode_prefetch_mixed(configs)

    df = pd.DataFrame(result_dicts)
    print(df)
    df.to_csv(output_csv, index=False)
    print(f"Saved output CSV to: {os.path.abspath(output_csv)}")
    plot_decode_prefetch_results(df, output_csv)


@dataclass
class PrefillBenchmarkConfig:
    batch_size: int
    history_len: int
    prefill_len: int
    device_pool_size: int

    num_heads: int = 64
    head_dim: int = 128
    mla_head_dim: int = 576

    def __post_init__(self):
        # The kernel currently loads per-request offsets from a fixed 4-entry array.
        assert 1 <= self.batch_size <= 4, "Prefill benchmark currently supports batch_size in [1, 4]"
        assert self.history_len > 0
        assert self.prefill_len > 0

        self.seq_len = self.batch_size * self.prefill_len
        self.max_model_len = self.history_len + self.prefill_len
        self.seq_len_kv = self.batch_size * self.max_model_len
        self.host_pool_size = self.seq_len_kv + 64
        # Match the kernel's effective prefetch capacity: one reserved slot plus
        # one live slot per in-flight prefill query are not available to prefetch.
        self.device_free_size = self.device_pool_size - 1 - self.seq_len

        assert self.device_pool_size >= self.seq_len + 2, (
            "device_pool_size must leave room for the reserved slot and the live prefill queries"
        )


@dataclass
class PrefillBenchmarkInputs:
    q: torch.Tensor
    q_fp8: torch.Tensor
    kv: torch.Tensor
    kv_fp8_data: torch.Tensor
    kv_fp8_scales: torch.Tensor
    weights: torch.Tensor
    cu_seq_len_k_start: torch.Tensor
    cu_seq_len_k_end: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_to_req: torch.Tensor
    extend_logits_offsets: torch.Tensor

    def as_vanilla_args(self) -> tuple:
        return (
            self.q_fp8,
            (self.kv_fp8_data, self.kv_fp8_scales),
            self.weights,
            self.cu_seq_len_k_start,
            self.cu_seq_len_k_end,
            True,
        )

    def as_prefetch_args(self, context: FuseV2Context) -> tuple:
        return (
            self.q_fp8,
            (self.kv_fp8_data, self.kv_fp8_scales),
            self.weights,
            self.cu_seq_len_k_start,
            self.cu_seq_len_k_end,
            context.page_table_1,
            self.extend_seq_lens,
            self.extend_seq_to_req,
            context.device_pool_buf,
            context.host_pool_buf,
            context.device_pool_loc_alloc_buf,
            context.device_pool_priority,
            context.device_pool_loc_small_priority,
            context.device_token_to_host,
            context.host_token_to_device,
            context.recall_counter,
            self.extend_logits_offsets,
            context.clean_logits,
        )


def make_prefill_page_table(config: PrefillBenchmarkConfig) -> torch.Tensor:
    local_indices = torch.arange(config.max_model_len, device="cuda", dtype=torch.int32)
    req_offsets = (
        torch.arange(config.batch_size, device="cuda", dtype=torch.int32) * config.max_model_len
    ).unsqueeze(1)
    return req_offsets + local_indices.unsqueeze(0)


def make_prefill_benchmark_context(config: PrefillBenchmarkConfig) -> FuseV2Context:
    context = FuseV2Context.dummy()
    context.page_table_1 = make_prefill_page_table(config)

    context.device_pool_buf = torch.empty(
        (config.device_pool_size, 1, config.mla_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    context.device_pool_buf.normal_()
    context._device_pool_buf = context.device_pool_buf.clone()

    context.host_pool_buf = torch.empty(
        (config.host_pool_size, 1, config.mla_head_dim),
        pin_memory=True,
        device="cpu",
        dtype=torch.bfloat16,
    )
    context.host_pool_buf.normal_()
    context._host_pool_buf = context.host_pool_buf.clone()

    context.device_pool_loc_alloc_buf = torch.zeros(
        (config.device_pool_size - 1,),
        device="cuda",
        dtype=torch.int32,
    )
    context._device_pool_loc_alloc_buf = context.device_pool_loc_alloc_buf.clone()

    context.device_pool_priority = torch.zeros((config.device_pool_size,), device="cuda", dtype=torch.int32)
    context.device_pool_priority[0] = INT32_MAX
    context._device_pool_priority = context.device_pool_priority.clone()

    context.device_pool_loc_small_priority = torch.arange(
        1,
        config.device_pool_size,
        device="cuda",
        dtype=torch.int32,
    )
    context._device_pool_loc_small_priority = context.device_pool_loc_small_priority.clone()

    context.device_token_to_host = torch.full(
        (config.device_pool_size,),
        INT32_MAX,
        device="cuda",
        dtype=torch.int64,
    )
    context._device_token_to_host = context.device_token_to_host.clone()

    context.host_token_to_device = torch.full(
        (config.host_pool_size + 1,),
        INT32_MAX,
        device="cuda",
        dtype=torch.int32,
    )
    context._host_token_to_device = context.host_token_to_device.clone()

    context.prefetch_host_loc = torch.full(
        (config.batch_size, REQ_PF_MAX),
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    context._prefetch_host_loc = context.prefetch_host_loc.clone()
    context.prefetch_kv_buf = torch.zeros(
        (config.batch_size, REQ_PF_MAX, config.mla_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    context._prefetch_kv_buf = context.prefetch_kv_buf.clone()

    context.recall_counter = torch.zeros((1,), device="cuda", dtype=torch.uint32)
    context._recall_counter = context.recall_counter.clone()
    context.query_recall_counter = torch.zeros((config.batch_size,), device="cuda", dtype=torch.uint32)
    context._query_recall_counter = context.query_recall_counter.clone()

    context.decode_topk_logits = torch.empty((0,), device="cuda", dtype=torch.float32)
    context._decode_topk_logits = context.decode_topk_logits.clone()

    context.clean_logits = True
    context._clean_logits = context.clean_logits
    return context


def make_prefill_benchmark_inputs(config: PrefillBenchmarkConfig) -> PrefillBenchmarkInputs:
    q = torch.randn(
        (config.seq_len, config.num_heads, config.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_fp8 = q.to(torch.float8_e4m3fn)

    kv = torch.randn(
        (config.seq_len_kv, config.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_fp8_data, kv_fp8_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    weights = torch.randn((config.seq_len, config.num_heads), device="cuda", dtype=torch.float32)

    req_offsets = (
        torch.arange(config.batch_size, device="cuda", dtype=torch.int32) * config.max_model_len
    ).repeat_interleave(config.prefill_len)
    local_prefill_pos = torch.arange(config.prefill_len, device="cuda", dtype=torch.int32).repeat(config.batch_size)

    cu_seq_len_k_start = req_offsets.clone()
    cu_seq_len_k_end = req_offsets + config.history_len + local_prefill_pos + 1

    extend_seq_lens = torch.full((config.batch_size,), config.prefill_len, device="cuda", dtype=torch.int32)
    extend_seq_to_req = torch.arange(config.batch_size, device="cuda", dtype=torch.int32).repeat_interleave(
        config.prefill_len
    )
    extend_logits_offsets = torch.zeros((4,), device="cuda", dtype=torch.float32)

    return PrefillBenchmarkInputs(
        q=q,
        q_fp8=q_fp8,
        kv=kv,
        kv_fp8_data=kv_fp8_data,
        kv_fp8_scales=kv_fp8_scales,
        weights=weights,
        cu_seq_len_k_start=cu_seq_len_k_start,
        cu_seq_len_k_end=cu_seq_len_k_end,
        extend_seq_lens=extend_seq_lens,
        extend_seq_to_req=extend_seq_to_req,
        extend_logits_offsets=extend_logits_offsets,
    )


def run_prefill_mixed(configs: list[PrefillBenchmarkConfig]) -> list[dict]:
    ret: list[dict] = []
    printed_csv_headers: set[tuple[str, ...]] = set()

    pbar = tqdm.tqdm(configs)
    for config in pbar:
        context = make_prefill_benchmark_context(config)
        inputs = make_prefill_benchmark_inputs(config)

        kernel_entries = [
            (
                "vanilla",
                lambda: deep_gemm.fp8_mqa_logits(*inputs.as_vanilla_args()),
                "fp8_mqa_logits",
            ),
            (
                "prefetch",
                lambda: deep_gemm.fp8_mqa_logits_fuse_prefetch(*inputs.as_prefetch_args(context)),
                "fp8_mqa_logits_fuse_prefetch",
            ),
        ]

        for kernel_name, fn, kernel_name_to_search in kernel_entries:
            with context.temp_state():
                _ = fn()
                result = {
                    "kernel": kernel_name,
                    "recall_counter": None if kernel_name == "vanilla" else int(context.recall_counter.item()),
                    "batch_size": config.batch_size,
                    "history_len": config.history_len,
                    "prefill_len": config.prefill_len,
                    "total_q": config.seq_len,
                    "total_kv": config.seq_len_kv,
                    "device_pool_size": config.device_pool_size,
                    "device_free_size": config.device_free_size,
                }

            runs = 15
            warmup = 3

            for _ in range(warmup):
                with context.temp_state():
                    fn()

            if BENCH_MODE == "kineto":
                def kineto_fn():
                    with context.temp_state(exception_raise=True):
                        fn()

                kernel_time_s = bench_kineto(
                    kineto_fn,
                    kernel_name_to_search,
                    num_tests=runs,
                    suppress_kineto_output=True,
                )
                result["avg_time_logits_ms"] = round(kernel_time_s * 1e3, 3)
            elif BENCH_MODE == "event":
                total_time_logits_ms = 0.0
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                for _ in range(runs):
                    with context.temp_state(exception_raise=True):
                        torch.cuda.synchronize()
                        start_event.record()
                        fn()
                        end_event.record()
                        torch.cuda.synchronize()
                    total_time_logits_ms += start_event.elapsed_time(end_event)

                result["avg_time_logits_ms"] = round(total_time_logits_ms / runs, 3)
            else:
                raise ValueError(f"Unknown BENCH_MODE: {BENCH_MODE!r}")
            ret.append(result)
            _print_csv_row_when_ready(result, printed_csv_headers)

        del inputs
        del context
        gc.collect()

    return ret


def build_default_configs() -> list[TestConfig]:
    device_pool_size = 300 * 1024

    # bsz_seqlen_pairs = [(bsz, seqlen) for bsz in [1, 2, 4, 8, 16] for seqlen in [32 * 1024, 64 * 1024, 128 * 1024]]
    # bsz_seqlen_pairs = [(bsz, seqlen) for bsz in [16] for seqlen in [64 * 1024, 128 * 1024, 256 * 1024]]
    bsz_seqlen_pairs = [(bsz, seqlen) for bsz in [16] for seqlen in [128 * 1024]]
    # bsz_seqlen_pairs = [(bsz, seqlen) for bsz in [1] for seqlen in [32 * 1024]]
    candidate_count_options = [TOPK]
    hit_rate_options = np.array([0.8, 0.85, 0.9, 0.95, 0.98])
    # hit_rate_options = np.array([0.5, 0.9, 0.97])

    configs: list[TestConfig] = []

    for bsz, seqlen in bsz_seqlen_pairs:
        allocated_device_size = int(bsz * TOPK * 1.5)
        device_free_size = device_pool_size - allocated_device_size

        for candidate_count in candidate_count_options:
            total_candidate_count = bsz * candidate_count
            num_gt_topk_hit_options = (total_candidate_count * hit_rate_options).astype(int).tolist()
            for num_gt_topk_hit in num_gt_topk_hit_options:
                max_model_len = max(128 * 1024, seqlen)
                host_pool_size = max_model_len * bsz
                host_pool_size += 64
                num_kv_blocks = host_pool_size // 64

                try:
                    config = TestConfig(
                        max_model_len=max_model_len,
                        num_kv_blocks=num_kv_blocks,
                        batch_size=bsz,
                        device_pool_size=device_pool_size,
                        host_pool_size=host_pool_size,
                        seq_len=seqlen,
                        device_free_size=device_free_size,
                        num_gt_topk_hit=num_gt_topk_hit,
                        candidate_count=candidate_count,
                    )
                except AssertionError as e:
                    print(
                        f"Skipping invalid config: bsz={bsz}, seqlen={seqlen}, device_free_size={device_free_size}, "
                        f"num_gt_topk_hit={num_gt_topk_hit}"
                    )
                    print("-", e)
                    continue

                configs.append(config)

    return configs


def build_default_prefill_configs() -> list[PrefillBenchmarkConfig]:
    configs: list[PrefillBenchmarkConfig] = []

    for batch_size in [1]:
        for history_len in [32 * 1024, 64 * 1024, 128 * 1024]:
            for prefill_len in [256, 512]:
                # Size the device pool generously so the prefill_prefetch
                # benchmark can fit the worst-case per-request union of
                # per-row top-K sets (bounded by history_len + prefill_len).
                worst_case_union_per_req = min(prefill_len * TOPK, history_len + prefill_len)
                device_pool_size = max(
                    batch_size * TOPK + 1024,
                    batch_size * prefill_len + 2,
                    batch_size * worst_case_union_per_req + 1024,
                )
                configs.append(
                    PrefillBenchmarkConfig(
                        batch_size=batch_size,
                        history_len=history_len,
                        prefill_len=prefill_len,
                        device_pool_size=device_pool_size,
                    )
                )

    return configs


def _plot_avg_time_logits(
    df: pd.DataFrame,
    output_csv: str,
    base_kernel: str,
    compare_kernel: str,
    compare_cols: list[str],
):
    import matplotlib.pyplot as plt

    base_df = df[df["kernel"] == base_kernel].copy()
    compare_df = df[df["kernel"] == compare_kernel].copy()

    if len(base_df) != len(compare_df):
        raise ValueError(f"{base_kernel} and {compare_kernel} row counts do not match")

    if not base_df[compare_cols].reset_index(drop=True).equals(compare_df[compare_cols].reset_index(drop=True)):
        raise ValueError(f"{base_kernel} and {compare_kernel} rows are not aligned")

    x = base_df.index.to_numpy() + 2

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, base_df["avg_time_logits_ms"].to_numpy(), marker="o", linewidth=1.5, label=base_kernel)
    ax.plot(x, compare_df["avg_time_logits_ms"].to_numpy(), marker="o", linewidth=1.5, label=compare_kernel)
    ax.set_xlabel(f"{base_kernel} Line Number In CSV")
    ax.set_ylabel("avg_time_logits_ms")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = f"{os.path.splitext(output_csv)[0]}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {os.path.abspath(plot_path)}")


def plot_avg_time_logits(df: pd.DataFrame, output_csv: str):
    _plot_avg_time_logits(
        df,
        output_csv,
        base_kernel="vanilla",
        compare_kernel="v2",
        compare_cols=["batch_size", "seq_len", "device_free_size", "all_req_topk_hit", "hit_rate"],
    )


def plot_prefill_avg_time_logits(df: pd.DataFrame, output_csv: str):
    _plot_avg_time_logits(
        df,
        output_csv,
        base_kernel="vanilla",
        compare_kernel="prefetch",
        compare_cols=["batch_size", "history_len", "prefill_len", "total_q", "total_kv", "device_pool_size"],
    )


def main(output_csv: str):
    configs = build_default_configs()
    result_dicts = run_mixed(configs)

    df = pd.DataFrame(result_dicts)
    print(df)
    df.to_csv(output_csv, index=False)
    print(f"Saved output CSV to: {os.path.abspath(output_csv)}")
    plot_avg_time_logits(df, output_csv)


def main_prefill(output_csv: str):
    configs = build_default_prefill_configs()
    result_dicts = run_prefill_mixed(configs)

    df = pd.DataFrame(result_dicts)
    print(df)
    df.to_csv(output_csv, index=False)
    print(f"Saved output CSV to: {os.path.abspath(output_csv)}")
    plot_prefill_avg_time_logits(df, output_csv)


def _build_prefill_recall_context_from_logits(
    ctx: FuseV2Context,
    inputs: PrefillBenchmarkInputs,
    config: PrefillBenchmarkConfig,
    logits: torch.Tensor,
    topk: int = TOPK,
) -> RecallContext:
    """Build a RecallContext for the prefill flow from logits of shape [seq_len, seq_len_kv].

    For each request r, consider the block of logit rows
    [r*prefill_len : (r+1)*prefill_len] and the request's valid kv-range
    [r*max_model_len : r*max_model_len + history_len + prefill_len]. Per-request
    "tokens that must be recalled" is the *union* over every prefill row's
    top-K kv positions (each prefill query independently needs its own top-K
    present in the device pool). We compute the per-row top-K, union across
    rows, then map kv-local indices to global host-pool indices via
    `ctx.page_table_1[r, kv_local_pos]`.

    Union sizes vary per request, so we pad each request's index list with -1
    up to the batch-wide maximum union size. `make_recall_context` /
    `recall_update` skip -1 entries (via `host_token_to_device[-1]` returning
    the INT32_MAX sentinel at the padded last slot).
    """
    batch_size = config.batch_size
    prefill_len = config.prefill_len
    history_len = config.history_len
    max_model_len = config.max_model_len

    per_request_union_local: list[torch.Tensor] = []
    for r in range(batch_size):
        # Each prefill query q in the request attends causally to kv positions
        # [0, history_len + q], so its valid range grows by one per row.
        sub = logits[
            r * prefill_len : (r + 1) * prefill_len,
            r * max_model_len : r * max_model_len + history_len + prefill_len,
        ]
        row_valid_lens = history_len + 1 + torch.arange(prefill_len, device=sub.device)

        row_topk_sets: list[torch.Tensor] = []
        for q in range(prefill_len):
            valid_len = int(row_valid_lens[q].item())
            row = sub[q, :valid_len]
            effective_topk = min(topk, valid_len)
            row_topk = torch.topk(row, k=effective_topk, sorted=False).indices
            row_topk_sets.append(row_topk)

        union_local = torch.unique(torch.cat(row_topk_sets, dim=0))
        per_request_union_local.append(union_local)

    max_union_size = max(int(u.numel()) for u in per_request_union_local)

    per_request_global: list[torch.Tensor] = []
    for r, union_local in enumerate(per_request_union_local):
        global_indices = ctx.page_table_1[r, union_local]
        if global_indices.numel() < max_union_size:
            pad = torch.full(
                (max_union_size - global_indices.numel(),),
                -1,
                device=global_indices.device,
                dtype=global_indices.dtype,
            )
            global_indices = torch.cat([global_indices, pad], dim=0)
        per_request_global.append(global_indices)

    gt_topk_global_indices = torch.stack(per_request_global, dim=0)

    return make_recall_context(
        gt_topk_global_indices=gt_topk_global_indices,
        host_token_to_device=ctx.host_token_to_device,
        device_token_to_host=ctx.device_token_to_host,
        layer_device_kv=ctx.device_pool_buf,
        layer_host_kv=ctx.host_pool_buf,
        recall_counter=ctx.recall_counter,
    )


def run_prefill_prefetch_mixed(configs: list[PrefillBenchmarkConfig]) -> list[dict]:
    """Benchmark: baseline (vanilla prefill logits + full recall) vs prefetch (fused prefill + residual recall).

    Mirrors `run_decode_prefetch_mixed` but for the prefill kernels:
    - "baseline": `fp8_mqa_logits` + recall_update transferring all topk misses.
    - "prefetch": `fp8_mqa_logits_fuse_prefetch` (in-kernel prefetch mutates
      pool state) + residual recall_update on the remaining misses.
    """
    runs = 15
    warmup = 3
    ret: list[dict] = []
    printed_csv_headers: set[tuple[str, ...]] = set()

    pbar = tqdm.tqdm(configs)
    for config in pbar:
        context = make_prefill_benchmark_context(config)
        inputs = make_prefill_benchmark_inputs(config)

        # --- Baseline: vanilla logits + recall-all-misses --------------------
        # `fp8_mqa_logits` does not mutate the device pool, so the logits run
        # and the downstream recall share the same untouched initial state.
        with context.temp_state():
            vanilla_logits = deep_gemm.fp8_mqa_logits(*inputs.as_vanilla_args())

        vanilla_time_ms = _measure_kernel_ms(
            fn=lambda: deep_gemm.fp8_mqa_logits(*inputs.as_vanilla_args()),
            kernel_name_to_search="fp8_mqa_logits",
            context=context,
            runs=runs,
            warmup=warmup,
        )

        vanilla_recall_size = 0
        vanilla_recall_counter = 0
        vanilla_recall_ms = float("nan")
        try:
            with context.temp_state(record_first=True, exception_raise=True):
                vanilla_recall_ctx = _build_prefill_recall_context_from_logits(
                    context, inputs, config, vanilla_logits, topk=TOPK
                )
                vanilla_recall_size = vanilla_recall_ctx.recall_size
                assert vanilla_recall_size <= vanilla_recall_ctx.recall_device_indices.shape[0], (
                    f"Not enough free device-pool slots for baseline prefill recall "
                    f"(need {vanilla_recall_size}, have {vanilla_recall_ctx.recall_device_indices.shape[0]}); "
                    "increase device_pool_size."
                )
                vanilla_recall_ms = _measure_recall_update_ms(vanilla_recall_ctx, runs=runs, warmup=warmup)
                vanilla_recall_ctx.recall_counter.fill_(0)
                run_recall_update_kernel_inplace(vanilla_recall_ctx)
                vanilla_recall_counter = int(vanilla_recall_ctx.recall_counter.item())
        except AssertionError as e:
            print(f"[prefill_prefetch] baseline recall skipped: {e}")

        ret.append({
            "kernel": "baseline",
            "batch_size": config.batch_size,
            "history_len": config.history_len,
            "prefill_len": config.prefill_len,
            "total_q": config.seq_len,
            "total_kv": config.seq_len_kv,
            "device_pool_size": config.device_pool_size,
            "device_free_size": config.device_free_size,
            "logits_ms": round(vanilla_time_ms, 3),
            "recall_size": vanilla_recall_size,
            "recall_counter": vanilla_recall_counter,
            "recall_ms": round(vanilla_recall_ms, 3),
            "total_ms": round(vanilla_time_ms + vanilla_recall_ms, 3),
            "in_kernel_prefetched": 0,
            "effective_prefetch": 0,
            "wasted_prefetch": 0,
        })
        _print_csv_row_when_ready(ret[-1], printed_csv_headers)

        # --- Prefetch: fused prefill logits + residual recall -----------------
        v2_time_ms = _measure_kernel_ms(
            fn=lambda: deep_gemm.fp8_mqa_logits_fuse_prefetch(*inputs.as_prefetch_args(context)),
            kernel_name_to_search="fp8_mqa_logits_fuse_prefetch",
            context=context,
            runs=runs,
            warmup=warmup,
        )

        # Run the fused kernel once to realize post-prefetch pool state, then
        # measure residual recall against it. The fused kernel increments
        # `context.recall_counter` internally, so we snapshot it *before* any
        # subsequent recall_update launches reset/increment it.
        with context.temp_state(record_first=True):
            prefetch_logits = deep_gemm.fp8_mqa_logits_fuse_prefetch(*inputs.as_prefetch_args(context))
            in_kernel_prefetched = int(context.recall_counter.item())

            try:
                v2_recall_ctx = _build_prefill_recall_context_from_logits(
                    context, inputs, config, prefetch_logits, topk=TOPK
                )
                v2_recall_size = v2_recall_ctx.recall_size
                assert v2_recall_size <= v2_recall_ctx.recall_device_indices.shape[0], (
                    "Not enough free device-pool slots for residual prefill recall after fused prefetch."
                )
                v2_recall_ms = _measure_recall_update_ms(v2_recall_ctx, runs=runs, warmup=warmup)
                v2_recall_ctx.recall_counter.fill_(0)
                run_recall_update_kernel_inplace(v2_recall_ctx)
                v2_recall_counter = int(v2_recall_ctx.recall_counter.item())
            except AssertionError as e:
                print(f"[prefill_prefetch] residual recall skipped: {e}")
                v2_recall_ms = float("nan")
                v2_recall_counter = 0
                v2_recall_size = 0

        ret.append({
            "kernel": "prefetch",
            "batch_size": config.batch_size,
            "history_len": config.history_len,
            "prefill_len": config.prefill_len,
            "total_q": config.seq_len,
            "total_kv": config.seq_len_kv,
            "device_pool_size": config.device_pool_size,
            "device_free_size": config.device_free_size,
            "logits_ms": round(v2_time_ms, 3),
            "recall_size": v2_recall_size,
            "recall_counter": v2_recall_counter,
            "recall_ms": round(v2_recall_ms, 3) if v2_recall_ms == v2_recall_ms else v2_recall_ms,
            "total_ms": round(v2_time_ms + v2_recall_ms, 3) if v2_recall_ms == v2_recall_ms else float("nan"),
            "in_kernel_prefetched": in_kernel_prefetched,
            "effective_prefetch": vanilla_recall_size - v2_recall_size,
            "wasted_prefetch": in_kernel_prefetched - (vanilla_recall_size - v2_recall_size),
        })
        _print_csv_row_when_ready(ret[-1], printed_csv_headers)

        del inputs
        del context
        gc.collect()

    return ret


def plot_prefill_prefetch_results(df: pd.DataFrame, output_csv: str):
    _plot_avg_time_logits_generic(
        df,
        output_csv,
        base_kernel="baseline",
        compare_kernel="prefetch",
        compare_cols=["batch_size", "history_len", "prefill_len", "total_q", "total_kv", "device_pool_size"],
        y_col="total_ms",
        ylabel="total time (logits + recall) ms",
    )


def main_prefill_prefetch(output_csv: str):
    configs = build_default_prefill_configs()
    result_dicts = run_prefill_prefetch_mixed(configs)

    df = pd.DataFrame(result_dicts)
    print(df)
    df.to_csv(output_csv, index=False)
    print(f"Saved output CSV to: {os.path.abspath(output_csv)}")
    plot_prefill_prefetch_results(df, output_csv)


def _timestamped_csv_name(filename: str) -> str:
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    stem, ext = os.path.splitext(basename)
    if not ext:
        ext = ".csv"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return os.path.join(dirname, f"{timestamp}_{stem}{ext}")


def _usage():
    print("Usage: python test_fuse.py [decode|decode_prefetch|prefill|prefill_prefetch] [csv_path]")


def _dispatch(argv: list[str]):
    if len(argv) == 1:
        _usage()
        return

    cmd = argv[1]

    if cmd == "decode":
        output_csv = argv[2] if len(argv) >= 3 else _timestamped_csv_name("results_decode/fuse_v2_recall_results2.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        main(output_csv)
        return

    if cmd == "decode_prefetch":
        output_csv = (
            argv[2]
            if len(argv) >= 3
            else _timestamped_csv_name("results_decode_prefetch/fuse_v2_prefetch_results.csv")
        )
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        main_decode_prefetch(output_csv)
        return

    if cmd == "prefill":
        output_csv = argv[2] if len(argv) >= 3 else _timestamped_csv_name("results_prefill/mqa_prefill_prefetch_benchmark.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        main_prefill(output_csv)
        return

    if cmd == "prefill_prefetch":
        output_csv = (
            argv[2]
            if len(argv) >= 3
            else _timestamped_csv_name("results_prefill_prefetch/mqa_prefill_prefetch_results.csv")
        )
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        main_prefill_prefetch(output_csv)
        return

    _usage()


if __name__ == "__main__":
    import sys

    _dispatch(sys.argv)
