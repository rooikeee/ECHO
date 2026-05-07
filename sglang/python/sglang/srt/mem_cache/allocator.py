from __future__ import annotations

"""
Copyright 2025 SGLang Team
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

"""
Page-aligned memory pool.
"""

import abc
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.memory_pool import SWAKVPool
from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, *args, **kwargs):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)


class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """Allocator for SWA hybrid KV cache."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        dtype: torch.dtype,
        device: str,
        kvcache: SWAKVPool,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        assert isinstance(kvcache, SWAKVPool)
        self._size_full = size
        self._size_swa = size_swa
        self.full_attn_allocator = TokenToKVPoolAllocator(
            size,
            dtype,
            device,
            kvcache.full_kv_pool,
            need_sort,
        )
        self.swa_attn_allocator = TokenToKVPoolAllocator(
            size_swa,
            dtype,
            device,
            kvcache.swa_kv_pool,
            need_sort,
        )
        self.full_to_swa_index_mapping = torch.empty(
            size + size_swa + 1,
            dtype=torch.int64,
            device=device,
        )
        self.clear()

        self._kvcache.full_to_swa_index_mapping = self.full_to_swa_index_mapping

    def available_size(self):
        raise NotImplementedError()

    def full_available_size(self):
        return self.full_attn_allocator.available_size()

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    @property
    def size_full(self):
        return self._size_full

    @property
    def size_swa(self):
        return self._size_swa

    def debug_print(self) -> str:
        msg = ""
        msg += f"#swa-available-size: {self.swa_attn_allocator.available_size()}, "
        msg += (
            f"#full-attn-available-size: {self.full_attn_allocator.available_size()}, "
        )
        return msg

    def get_kvcache(self):
        return self._kvcache

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor):
        assert self.full_to_swa_index_mapping is not None
        return self.full_to_swa_index_mapping[kv_indices].to(torch.int32)

    def alloc(self, need_size: int):
        if need_size > self.full_attn_allocator.available_size():
            return None
        if need_size > self.swa_attn_allocator.available_size():
            return None

        alloc_full_indices = self.full_attn_allocator.alloc(need_size)
        alloc_swa_indices = self.swa_attn_allocator.alloc(need_size)
        self.full_to_swa_index_mapping[alloc_full_indices] = alloc_swa_indices
        return alloc_full_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if self.is_not_in_free_group:
            self.full_attn_allocator.free(free_index)
            self.free_swa(free_index)
        else:
            self.free_group.append(free_index)
        assert (
            self.full_attn_allocator.available_size() <= self.full_attn_allocator.size
        )
        assert self.swa_attn_allocator.available_size() <= self.swa_attn_allocator.size

    def free_swa(self, free_index: torch.Tensor):
        swa_indices = self.full_to_swa_index_mapping[free_index]
        swa_indices = swa_indices[swa_indices > 0]
        self.swa_attn_allocator.free(swa_indices)
        self.full_to_swa_index_mapping[free_index] = 0

    def backup_state(self):
        return [
            self.full_attn_allocator.backup_state(),
            self.swa_attn_allocator.backup_state(),
        ]

    def restore_state(self, state):
        assert len(state) == 2
        self.full_attn_allocator.restore_state(state[0])
        self.swa_attn_allocator.restore_state(state[1])

    def clear(self):
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping.fill_(0)
        self.is_not_in_free_group = True
        self.free_group = []


@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
    max_num_extend_tokens: tl.constexpr,
    MANY_PAGE_SEG: tl.constexpr = 8192,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )

    if MANY_PAGE_SEG >= max_num_extend_tokens:
        offset_many_page = tl.arange(0, max_num_extend_tokens)
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset_many_page // page_size,
            mask=offset_many_page < num_part2,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset_many_page,
            page_start * page_size + offset_many_page % page_size,
            mask=offset_many_page < num_part2,
        )
    else:
        offset_many_page = tl.arange(0, MANY_PAGE_SEG)
        num_iter = (max_num_extend_tokens - 1) // MANY_PAGE_SEG + 1
        for i in range(num_iter):
            start = i * MANY_PAGE_SEG
            offset = start + offset_many_page
            page_start = tl.load(
                free_page_ptr + new_page_start_loc + offset // page_size,
                mask=offset < num_part2,
            )
            tl.store(
                out_indices + output_start_loc + num_part1 + offset,
                page_start * page_size + offset % page_size,
                mask=offset < num_part2,
            )

    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.seen_max_num_extend_tokens_next_power_of_2 = 1
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        self.seen_max_num_extend_tokens_next_power_of_2 = max(
            self.seen_max_num_extend_tokens_next_power_of_2,
            next_power_of_2(extend_num_tokens),
        )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )
        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
            self.seen_max_num_extend_tokens_next_power_of_2,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))

            from sglang.srt.mem_cache.memory_pool_host import NSATokenToKVPoolHost
            if isinstance(self._kvcache, NSATokenToKVPoolHost):
                self._kvcache.free_req_device_pool(free_index)
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def get_cpu_copy(self, indices):
        return self._kvcache.get_cpu_copy(indices)

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)

_CUDA_GRAPH_ALLOCATOR_BLOCK = 256
_CUDA_GRAPH_ALLOCATOR_MAX_CTAS = 64


class CudaGraphTokenToKVPoolAllocator:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.device = device
        self.size_in_tensor = torch.tensor(
            [size], dtype=torch.int32, device=self.device
        )
        self.free_pages = torch.empty(
            self.size + 1, dtype=torch.int32, device=self.device
        )  # 1: freed, 0: allocated
        self.free_stack = torch.empty(
            self.size, dtype=torch.int32, device=self.device
        )
        self.available_size_tensor = torch.empty(
            (1,), dtype=torch.int32, device=self.device
        )
        self.last_alloc_size_tensor = torch.empty(
            (1,), dtype=torch.int32, device=self.device
        )
        self.alloc_counter = torch.empty((1,), dtype=torch.int32, device=self.device)
        self._grid = (
            max(
                1,
                min(
                    triton.cdiv(self.size, _CUDA_GRAPH_ALLOCATOR_BLOCK),
                    _CUDA_GRAPH_ALLOCATOR_MAX_CTAS,
                ),
            ),
        )
        self._last_alloc_buffer_data_ptr = None
        self.clear()

    def clear(self):
        _cuda_graph_allocator_clear_kernel[self._grid](
            self.free_pages,
            self.free_stack,
            self.available_size_tensor,
            self.last_alloc_size_tensor,
            self.size,
            BLOCK=_CUDA_GRAPH_ALLOCATOR_BLOCK,
        )

    def available_size(self):
        return self.available_size_tensor[0]

    def alloc(self, need_size: torch.Tensor, device_pool_loc_alloc: torch.Tensor):
        data_ptr = device_pool_loc_alloc.data_ptr()
        if data_ptr != self._last_alloc_buffer_data_ptr:
            if not torch.cuda.is_current_stream_capturing():
                self.last_alloc_size_tensor.fill_(self.size)
            self._last_alloc_buffer_data_ptr = data_ptr

        _cuda_graph_allocator_alloc_kernel[self._grid](
            need_size,
            device_pool_loc_alloc,
            self.free_stack,
            self.free_pages,
            self.available_size_tensor,
            self.last_alloc_size_tensor,
            BLOCK=_CUDA_GRAPH_ALLOCATOR_BLOCK,
        )

    def post_alloc(
        self,
        allocated_loc: torch.Tensor,  # [size]
    ):
        _cuda_graph_allocator_post_alloc_kernel[self._grid](
            allocated_loc,
            self.free_pages,
            allocated_loc.numel(),
            BLOCK=_CUDA_GRAPH_ALLOCATOR_BLOCK,
        )
        self.alloc_counter.fill_(0)
        _cuda_graph_allocator_rebuild_stack_kernel[self._grid](
            self.free_pages,
            self.free_stack,
            self.alloc_counter,
            self.size,
            BLOCK=_CUDA_GRAPH_ALLOCATOR_BLOCK,
        )
        _cuda_graph_allocator_set_available_kernel[(1,)](
            self.available_size_tensor,
            self.alloc_counter,
            self.last_alloc_size_tensor,
            self.size,
        )

    def free(
        self,
        free_index_buf: torch.Tensor,  # [size]
        free_size: torch.Tensor,  # [1]
    ):
        _cuda_graph_allocator_free_kernel[self._grid](
            free_index_buf,
            free_size,
            self.free_pages,
            self.free_stack,
            self.available_size_tensor,
            BLOCK=_CUDA_GRAPH_ALLOCATOR_BLOCK,
        )


@triton.jit
def _cuda_graph_allocator_clear_kernel(
    free_pages_ptr,
    free_stack_ptr,
    available_size_ptr,
    last_alloc_size_ptr,
    pool_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    stride = tl.num_programs(0) * BLOCK
    while base < pool_size:
        offs = base + tl.arange(0, BLOCK)
        mask = offs < pool_size
        loc = offs + 1
        tl.store(free_pages_ptr + loc, 1, mask=mask)
        # Store in reverse so the stack pop path returns low indices first.
        tl.store(free_stack_ptr + offs, pool_size - offs, mask=mask)
        base += stride

    if pid == 0:
        tl.store(free_pages_ptr, 0)
        tl.store(available_size_ptr, pool_size)
        tl.store(last_alloc_size_ptr, pool_size)


@triton.jit
def _cuda_graph_allocator_alloc_kernel(
    need_size_ptr,
    device_pool_loc_alloc_ptr,
    free_stack_ptr,
    free_pages_ptr,
    available_size_ptr,
    last_alloc_size_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    need_size = tl.load(need_size_ptr).to(tl.int32)
    available_size = tl.load(available_size_ptr).to(tl.int32)
    last_alloc_size = tl.load(last_alloc_size_ptr).to(tl.int32)
    alloc_size = tl.minimum(need_size, available_size)

    base = pid * BLOCK
    stride = tl.num_programs(0) * BLOCK
    while base < alloc_size:
        offs = base + tl.arange(0, BLOCK)
        mask = offs < alloc_size
        stack_pos = available_size - 1 - offs
        loc = tl.load(free_stack_ptr + stack_pos, mask=mask, other=0)
        tl.store(device_pool_loc_alloc_ptr + offs, loc, mask=mask)
        tl.store(free_pages_ptr + loc, 0, mask=mask)
        base += stride

    base = pid * BLOCK + alloc_size
    while base < last_alloc_size:
        offs = base + tl.arange(0, BLOCK)
        mask = offs < last_alloc_size
        tl.store(device_pool_loc_alloc_ptr + offs, 0, mask=mask)
        base += stride

    if pid == 0:
        tl.store(available_size_ptr, available_size - alloc_size)
        tl.store(last_alloc_size_ptr, alloc_size)


@triton.jit
def _cuda_graph_allocator_free_kernel(
    free_index_buf_ptr,
    free_size_ptr,
    free_pages_ptr,
    free_stack_ptr,
    available_size_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    free_size = tl.load(free_size_ptr).to(tl.int32)

    base = pid * BLOCK
    stride = tl.num_programs(0) * BLOCK
    while base < free_size:
        offs = base + tl.arange(0, BLOCK)
        mask = offs < free_size
        free_index = tl.load(free_index_buf_ptr + offs, mask=mask, other=0).to(
            tl.int32
        )
        valid = mask & (free_index > 0)
        old_state = tl.atomic_xchg(free_pages_ptr + free_index, 1, mask=valid)
        newly_freed = valid & (old_state == 0)
        newly_freed_i32 = newly_freed.to(tl.int32)
        num_newly_freed = tl.sum(newly_freed_i32, axis=0)

        if num_newly_freed > 0:
            pos = tl.atomic_add(available_size_ptr, num_newly_freed)
            rank = tl.cumsum(newly_freed_i32, axis=0) - newly_freed_i32
            tl.store(
                free_stack_ptr + pos + rank,
                free_index,
                mask=newly_freed,
            )

        base += stride


@triton.jit
def _cuda_graph_allocator_post_alloc_kernel(
    allocated_loc_ptr,
    free_pages_ptr,
    allocated_loc_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    stride = tl.num_programs(0) * BLOCK
    while base < allocated_loc_size:
        offs = base + tl.arange(0, BLOCK)
        mask = offs < allocated_loc_size
        alloc_index = tl.load(allocated_loc_ptr + offs, mask=mask, other=0).to(
            tl.int32
        )
        tl.store(free_pages_ptr + alloc_index, 0, mask=mask & (alloc_index > 0))
        base += stride


@triton.jit
def _cuda_graph_allocator_rebuild_stack_kernel(
    free_pages_ptr,
    free_stack_ptr,
    counter_ptr,
    pool_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    stride = tl.num_programs(0) * BLOCK
    while base < pool_size:
        offs = base + tl.arange(0, BLOCK)
        mask = offs < pool_size
        loc = offs + 1
        is_free = mask & (tl.load(free_pages_ptr + loc, mask=mask, other=0) != 0)
        is_free_i32 = is_free.to(tl.int32)
        num_free = tl.sum(is_free_i32, axis=0)

        if num_free > 0:
            pos = tl.atomic_add(counter_ptr, num_free)
            rank = tl.cumsum(is_free_i32, axis=0) - is_free_i32
            tl.store(free_stack_ptr + pos + rank, loc, mask=is_free)

        base += stride


@triton.jit
def _cuda_graph_allocator_set_available_kernel(
    available_size_ptr,
    counter_ptr,
    last_alloc_size_ptr,
    pool_size: tl.constexpr,
):
    tl.store(available_size_ptr, tl.load(counter_ptr))
    tl.store(last_alloc_size_ptr, pool_size)


@triton.jit
def free_kernel(
    free_index_buf_ptr,
    free_size_ptr,
    free_pages_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    free_size = tl.load(free_size_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < free_size
    free_index = tl.load(free_index_buf_ptr + offs, mask=mask, other=0)
    tl.store(free_pages_ptr + free_index, 1, mask=mask)

@triton.jit
def update_alloc_kernel(
    need_size_ptr,
    alloc_counter_ptr,
    device_pool_loc_alloc_ptr,
    free_pages_ptr,
    pool_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    need_size = tl.load(need_size_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < pool_size

    is_free = tl.load(free_pages_ptr + offs, mask=mask, other=0) != 0
    is_free_i32 = is_free.to(tl.int32)
    num_free = tl.sum(is_free_i32, axis=0)
    if num_free > 0:
        base = tl.atomic_add(alloc_counter_ptr, num_free)
        local_rank = tl.cumsum(is_free_i32, axis=0) - is_free_i32
        alloc_mask = mask & is_free & ((base + local_rank) < need_size)
        tl.store(free_pages_ptr + offs, 0, mask=alloc_mask)
        tl.store(device_pool_loc_alloc_ptr + base + local_rank, offs + 1, mask=alloc_mask)

@triton.jit
def post_alloc_kernel(
    allocated_loc_ptr,
    free_pages_ptr,
):
    idx = tl.program_id(0)
    
    alloc_index = tl.load(allocated_loc_ptr + idx)
    if alloc_index > 0:
        tl.store(free_pages_ptr + alloc_index, 0)
