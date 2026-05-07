import os
import abc
import logging
import threading
from functools import wraps
from typing import Optional, List, Sequence, Set, Tuple

import psutil
import torch

from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
    _view_kv_buffer_as_logical_dtype,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator, CudaGraphTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import set_mla_kv_buffer_triton
from sglang.srt.utils import get_int_env_var, is_npu, is_xpu

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.attention.nsa_backend import NSAMetadata
from sglang.srt.layers.attention.nsa_backend import NSA_USE_CUDA_GRAPH
from sglang.srt.mem_cache.recall_ops import (
    protect, get_free_loc, free_update, set_update,
    recall_check_available_size,
    mark_host_need_recall, recall_update, recall_update_extend,
    recall_update_with_prefetch, clear_decode_prefetch_mappings,
    set_mla_kv_buffer_cuda_graph,
)

_is_npu = is_npu()
_is_xpu = is_xpu()
if not (_is_npu or _is_xpu):
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
    )

logger = logging.getLogger(__name__)
I32_MAX = torch.iinfo(torch.int32).max


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align the host memory pool size to the page size
        self.size = self.size - (self.size % self.page_size)
        self.page_num = self.size // self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    @synchronized
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


class MHATokenToKVPoolHost(HostKVCache):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
        )
        self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
        self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        return torch.empty(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_per_layer(
                    src_k=self.k_buffer[layer_id],
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer[layer_id],
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    item_size=self.token_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_pf_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    dst_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_all_layer(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k_layers=self.k_data_ptrs,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v_layers=self.v_data_ptrs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_lf_pf(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,
                    dst_layers=self.k_data_refs + self.v_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer + device_pool.v_buffer,
                    dst_ptrs=[self.k_buffer, self.v_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + self.page_size, :, :] = (
                data_page.reshape(
                    2,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            )
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + self.page_size, :, :, :] = (
                data_page.reshape(
                    2, self.page_size, self.layer_num, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
            element_size = (
                self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.head_num
                * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class MLATokenToKVPoolHost(HostKVCache):
    device_pool: MLATokenToKVPool

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        use_nsa: bool = False,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
        )
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.use_nsa = use_nsa
        self.nsa_kv_cache_store_fp8 = device_pool.nsa_kv_cache_store_fp8
        self.kv_cache_dim = device_pool.kv_cache_dim
        self.mem_usage = 0

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num

        return (
            (self.kv_lora_rank + self.qk_rope_head_dim)
            * 1
            * self.dtype.itemsize
            * self.layer_num
        )

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (
                self.layer_num,
                self.size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        elif self.layout == "page_first":
            dims = (
                self.size,
                self.layer_num,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = (
            self.kv_lora_rank + self.qk_rope_head_dim
        ) * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        return torch.empty(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.kv_buffer[layer_id],
                    dst=device_pool.kv_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    item_size=self.token_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.kv_buffer,
                    dst=device_pool.kv_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[layer_id]],
                    dst_layers=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=device_pool.data_ptrs,
                    dst_layers=self.data_ptrs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=device_pool.data_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.kv_buffer,
                    dst_layers=self.data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.kv_buffer,
                    dst_ptrs=[self.kv_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.layer_num,
                self.page_size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num,
                self.page_size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        elif self.layout == "page_first":
            self.kv_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size,
                self.layer_num,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1,
                self.layer_num,
                self.page_size,
                1,
                self.kv_lora_rank + self.qk_rope_head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * (self.kv_lora_rank + self.qk_rope_head_dim)
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * (self.kv_lora_rank + self.qk_rope_head_dim)
                        * self.dtype.itemsize
                    )
                    ptr_list.append(k_ptr)
            element_size = (
                self.dtype.itemsize
                * self.page_size
                * (self.kv_lora_rank + self.qk_rope_head_dim)
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * (self.kv_lora_rank + self.qk_rope_head_dim)
                    * self.dtype.itemsize
                )
                ptr_list.append(k_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * (self.kv_lora_rank + self.qk_rope_head_dim)
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


from contextlib import nullcontext
from sglang.srt.layers.attention.nsa import index_buf_accessor

# NOTE: layer_id is abused (maybe from 0 or start_layer)
class NSATokenToKVPoolHost(MLATokenToKVPoolHost):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,  # num_tokens
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        device_pool: MLATokenToKVPool,
        max_num_reqs: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        kv_lora_rank = device_pool.kv_lora_rank
        qk_rope_head_dim = device_pool.qk_rope_head_dim
        layer_num = device_pool.layer_num
        size_per_token = (
            (kv_lora_rank + qk_rope_head_dim)
            * 1
            * device_pool.dtype.itemsize
            * device_pool.layer_num
        )
        super().__init__(
            device_pool=device_pool,
            host_size=size * size_per_token / 1e9,
            host_to_device_ratio=-1,
            page_size=page_size,
            layout="layer_first",
            pin_memory=True,
            device="cpu",
            use_nsa=True,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        # num head == 1 and head dim == 128 for index_k in NSA
        assert index_head_dim == 128

        assert self.page_size == 64
        with (
            nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (self.size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]
            self.host_token_to_device = torch.full(
                (self.layer_num, self.size + 1), I32_MAX,
                device=device_pool.device, dtype=torch.int32
            )   # I32_MAX stands for not in device cache
            self.host_token_to_device[:, 0] = 0     # 0 for mapping padding
            self.host_token_to_device[:, -1] = -1   # -1 for mapping non-selection
            self.device_token_to_host = torch.full(
                (self.layer_num, device_pool.size + 1), I32_MAX,
                device=device_pool.device
            )   # I32_MAX stands for free, +1 to align with allocator

        if NSA_USE_CUDA_GRAPH:
            self.device_pool_allocator = [
                CudaGraphTokenToKVPoolAllocator(
                    device_pool.size,
                    device_pool.dtype,
                    device_pool.device,
                    device_pool,
                    need_sort=False
                ) for _ in range(self.layer_num)
            ]
        else:
            self.device_pool_allocator = [
                TokenToKVPoolAllocator(
                    device_pool.size,
                    device_pool.dtype,
                    device_pool.device,
                    device_pool,
                    need_sort=False
                ) for _ in range(self.layer_num)
            ]
        # self.device_pool_loc_queue = [[]] * self.layer_num
        self.device_pool_priority = torch.full(
            (self.layer_num, device_pool.size + 1), -1,
            device=device_pool.device, dtype=torch.int32
        )   # less ones are evicted first, -1 stands for free
        # empty first slot due to allocator (never allocated)
        self.device_pool_priority[:, 0] = I32_MAX
        self.device_pool_loc_small_priority = torch.zeros(
            (device_pool.size + 1), device=device_pool.device, dtype=torch.int32
        )

        self.fifo_counter = torch.zeros(
            (self.layer_num,), device=device_pool.device, dtype=torch.int32
        )

        host_transfer_stream_count = max(
            1,
            get_int_env_var(
                "SGLANG_NSA_HOST_TRANSFER_STREAMS", min(4, max_num_reqs)
            ),
        )
        if max_num_reqs > 0:
            host_transfer_stream_count = min(host_transfer_stream_count, max_num_reqs)
        self.host_transfer_streams = [
            [torch.cuda.Stream() for _ in range(host_transfer_stream_count)]
            for _ in range(self.layer_num)
        ]
        self.host_transfer_next_lane = [0 for _ in range(self.layer_num)]
        self.host_transfer_req_events = [dict() for _ in range(self.layer_num)]

        self.host_need_recall = torch.zeros(
            self.size + 1, dtype=torch.bool, device=device_pool.device
        )

        self.mem_usage = 0
        self._finalize_allocation_log(self.size)
        if NSA_USE_CUDA_GRAPH:
            # TODO: maybe use a single buf
            self.free_index_device_buf = torch.zeros(
                device_pool.size, dtype=torch.int32, device=device_pool.device
            )
            self.device_pool_loc_alloc_buf = torch.zeros(
                device_pool.size, dtype=torch.int32, device=device_pool.device
            )
            self.recall_counter = torch.tensor(
                [0], device=device_pool.device, dtype=torch.uint32
            )
            self.query_recall_counter = torch.zeros(
                max_num_reqs, device=device_pool.device, dtype=torch.uint32
            )
            self.decode_prefetch_counter = torch.zeros(
                (1,), device=device_pool.device, dtype=torch.int32
            )
            self.decode_prefetch_max = 64
            self.decode_prefetch_host_loc = torch.full(
                (max_num_reqs, self.decode_prefetch_max),
                -1,
                device=device_pool.device,
                dtype=torch.int32,
            )
            self.decode_prefetch_kv_buf = torch.empty(
                (
                    max_num_reqs,
                    self.decode_prefetch_max,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ),
                device=device_pool.device,
                dtype=torch.bfloat16,
            )

    def reset_after_capture(self):
        self.device_pool_priority[:, 1:].fill_(-1)
        self.host_token_to_device[:, 0] = 0
        self.host_token_to_device[:, 1: -1].fill_(I32_MAX)
        self.device_token_to_host.fill_(I32_MAX)
        if hasattr(self, "decode_prefetch_host_loc"):
            self.decode_prefetch_host_loc.fill_(-1)
        for layer_id in range(self.layer_num):
            self.device_pool_allocator[layer_id].clear()

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    # TODO rename later (currently use diff name to avoid confusion)
    def set_index_k_and_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        import numpy as np
        def get_tensor_size_bytes(t: torch.Tensor):
            return np.prod(t.shape) * t.dtype.itemsize

        kv_size_bytes = get_tensor_size_bytes(self.kv_buffer)
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        GB = 1024 * 1024 * 1024
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"NSA Host KV Cache is allocated. #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"NSA Host KV Cache is allocated. #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    def free_device_pool(
        self, 
        free_size: int, 
        layer_id: Optional[int] = None,
        protected_device_index: Optional[torch.Tensor] = None
    ):
        if protected_device_index is not None:
            # print(f"{free_size=} {layer_id=}, {protected_device_index=}, ")
            self.device_pool_priority[layer_id, protected_device_index] = self.fifo_counter[layer_id]
            self.fifo_counter[layer_id] += 1
        # -1 priority has been freed
        device_pool_priority_temp = self.device_pool_priority[layer_id].clone()
        device_pool_priority_temp[device_pool_priority_temp < 0] = I32_MAX
        _, free_index_device = torch.topk(device_pool_priority_temp, free_size, largest=False)
        free_index_host = self.device_token_to_host[layer_id, free_index_device]
        # print(f"{free_index_device=} {free_index_host=}")
        assert torch.all(free_index_host < I32_MAX), f"{free_index_device[free_index_host == I32_MAX]=}"
        # update 3 tensors
        self.device_pool_priority[layer_id, free_index_device] = -1
        self.device_token_to_host[layer_id, free_index_device] = I32_MAX
        self.host_token_to_device[layer_id, free_index_host] = I32_MAX
        self.device_pool_allocator[layer_id].free(free_index_device)

    def free_device_pool_cuda_graph(
        self, 
        free_size: torch.Tensor, 
        layer_id: Optional[int] = None,
        protected_device_index: Optional[torch.Tensor] = None
    ):
        if protected_device_index is not None:
            protect(self.device_pool_priority[layer_id], protected_device_index, self.fifo_counter[layer_id])
            self.fifo_counter[layer_id] += 1
        # -1 priority has been freed
        device_pool_priority_temp = self.device_pool_priority[layer_id].clone()
        device_pool_priority_temp[device_pool_priority_temp < 0] = I32_MAX

        get_free_loc(device_pool_priority_temp, self.device_pool_loc_small_priority, free_size)

        free_update(
            free_size, 
            self.free_index_device_buf,
            self.device_pool_loc_small_priority, 
            self.device_token_to_host[layer_id],
            self.device_pool_priority[layer_id],
            self.host_token_to_device[layer_id],
        )

        self.device_pool_allocator[layer_id].free(self.free_index_device_buf, free_size)

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        metadata: Optional[NSAMetadata] = None,
        host_transfer_req_pool_indices: Optional[Sequence[int]] = None,
    ):
        layer_id = layer.layer_id

        if self.use_nsa and self.nsa_kv_cache_store_fp8:
            raise NotImplementedError("NSA FP8 store not implemented in host pool.")
        else:
            device_kv_buffer = _view_kv_buffer_as_logical_dtype(
                self.device_pool.kv_buffer[layer_id - self.start_layer],
                self.device_pool.dtype,
            )
            host_kv_buffer = _view_kv_buffer_as_logical_dtype(
                self.kv_buffer[layer_id - self.start_layer], self.device_pool.dtype
            )

            if NSA_USE_CUDA_GRAPH:
                if metadata is None:
                    raise RuntimeError(
                        "NSA CUDA-graph KV update requires metadata with a device "
                        "need-size tensor."
                    )

                if metadata.nsa_non_padded is not None:
                    need_size = metadata.nsa_non_padded
                else:
                    need_size = metadata.cu_seqlens_q[-1:]

                assert need_size.is_cuda and need_size.dtype == torch.int32
                assert need_size.numel() == 1
                available_size = self.device_pool_allocator[layer_id].available_size().to(torch.int32)
                free_size = need_size - available_size
                free_size = torch.clamp_min(free_size, 0)
                # print(f"{layer_id=} {need_size=} {free_size=}")
                self.free_device_pool_cuda_graph(free_size, layer_id)
                device_pool_loc = self.device_pool_loc_alloc_buf
                self.device_pool_allocator[layer_id].alloc(need_size, device_pool_loc)

                self.update_priority_when_use(layer_id, device_pool_loc)

                set_update(
                    loc,
                    device_pool_loc,
                    self.device_token_to_host[layer_id - self.start_layer],
                    self.host_token_to_device[layer_id - self.start_layer],
                )
                set_mla_kv_buffer_cuda_graph(
                    need_size,
                    device_kv_buffer,
                    device_pool_loc,
                    cache_k_nope,
                    cache_k_rope,
                )
            else:
                need_size = loc.shape[0]
                if self.device_pool_allocator[layer_id].available_size() < need_size:
                    free_size = need_size - self.device_pool_allocator[layer_id].available_size()
                    # self.free_device_pool(free_size, layer_id)
                    self.free_device_pool(free_size, layer_id)

                device_pool_loc = self.device_pool_allocator[layer_id].alloc(need_size)
                if device_pool_loc is None:
                    raise RuntimeError("Device pool out of memory.")
                # self.device_pool_loc_queue[layer_id].extend(zip(device_pool_loc.tolist(), loc.tolist()))
                # print(f"{layer_id=} alloc {loc=} {device_pool_loc=} {device_pool_loc.shape=}")

                self.update_priority_when_use(layer, device_pool_loc)

                # print(f"set {layer_id=} {loc=} {device_pool_loc=}")
                self.host_token_to_device[layer_id - self.start_layer, loc] = device_pool_loc
                self.device_token_to_host[layer_id - self.start_layer, device_pool_loc] = loc
                set_mla_kv_buffer_triton(
                    device_kv_buffer,
                    device_pool_loc,
                    cache_k_nope,
                    cache_k_rope,
                )

            if torch.cuda.is_current_stream_capturing():
                set_mla_kv_buffer_triton(
                    host_kv_buffer,
                    loc,
                    cache_k_nope,
                    cache_k_rope,
                )
            else:
                current_stream = torch.cuda.current_stream()
                host_transfer_stream, req_ids = self.get_host_transfer_stream(
                    layer_id, host_transfer_req_pool_indices, current_stream
                )
                host_transfer_stream.wait_stream(current_stream)
                with torch.cuda.stream(host_transfer_stream):
                    set_mla_kv_buffer_triton(
                        host_kv_buffer,
                        loc,
                        cache_k_nope,
                        cache_k_rope,
                    )
                    if req_ids is not None:
                        transfer_done = torch.cuda.Event()
                        transfer_done.record(host_transfer_stream)
                        req_events = self.host_transfer_req_events[layer_id]
                        for req_id in req_ids:
                            req_events[req_id] = transfer_done
                if cache_k_nope.is_cuda:
                    cache_k_nope.record_stream(host_transfer_stream)
                if cache_k_rope.is_cuda:
                    cache_k_rope.record_stream(host_transfer_stream)
                if loc.is_cuda:
                    loc.record_stream(host_transfer_stream)

    def get_host_transfer_stream(
        self,
        layer_id: int,
        req_pool_indices: Optional[Sequence[int]],
        current_stream: torch.cuda.Stream,
    ) -> Tuple[torch.cuda.Stream, Optional[Set[int]]]:
        streams = self.host_transfer_streams[layer_id]
        next_lane = self.host_transfer_next_lane[layer_id]
        self.host_transfer_next_lane[layer_id] = (next_lane + 1) % len(streams)

        if req_pool_indices is None:
            for stream in streams:
                current_stream.wait_stream(stream)
            return streams[next_lane], None
        if isinstance(req_pool_indices, torch.Tensor):
            if req_pool_indices.is_cuda:
                for stream in streams:
                    current_stream.wait_stream(stream)
                return streams[next_lane], None
            req_pool_indices = req_pool_indices.tolist()

        req_events = self.host_transfer_req_events[layer_id]
        req_ids = {int(req_id) for req_id in req_pool_indices if req_id is not None}
        if not req_ids:
            return streams[next_lane], None
        for req_id in req_ids:
            event = req_events.get(req_id)
            if event is not None:
                current_stream.wait_event(event)

        return streams[next_lane], req_ids

    def free_req_device_pool(self, free_index_host: torch.Tensor):
        for layer_id in range(self.layer_num):
            free_index_device = self.host_token_to_device[layer_id, free_index_host]
            # some tokens might have been freed during decoding
            free_index_device = free_index_device[free_index_device < I32_MAX]
            if NSA_USE_CUDA_GRAPH:
                free_size = torch.tensor(
                    free_index_device.shape[0],
                    device=self.device_pool.device,
                    dtype=torch.int32,
                )
            self.device_pool_priority[layer_id, free_index_device] = -1
            self.device_token_to_host[layer_id, free_index_device] = I32_MAX
            self.host_token_to_device[layer_id, free_index_host] = I32_MAX
            if NSA_USE_CUDA_GRAPH:
                self.device_pool_allocator[layer_id].free(free_index_device, free_size)
            else:
                self.device_pool_allocator[layer_id].free(free_index_device)
            continue

            # print(f"{layer_id=} {self.device_pool_priority[layer_id, 0]=}")
            print(f"{layer_id=} {(self.device_pool_priority[layer_id] == -1).sum()=}")
            print(f"{layer_id=} {(self.device_token_to_host[layer_id] == I32_MAX).sum()=}")
            # print(f"{layer_id=} {self.host_token_to_device[layer_id][-1]=}")
            assert(self.host_token_to_device[layer_id][-1] == -1), f"{self.host_token_to_device[layer_id][-1]=}"
            assert(self.host_token_to_device[layer_id][0] == 0), f"{self.host_token_to_device[layer_id][0]=}"
            print(f"{layer_id=} {(self.host_token_to_device[layer_id] < I32_MAX).sum()=}")
            print(f"{layer_id=} {self.device_pool_allocator[layer_id].available_size()=}")
            # print(f"{layer_id=} {self.device_pool_allocator[layer_id].free_pages=}")

    def recall_miss_tokens_decode(
        self, 
        recall_host_indices: torch.Tensor, 
        recall_size: int,
        device_page_table_1: torch.Tensor,
        layer_id: int,
    ):
        device_pool_allocator = self.device_pool_allocator[layer_id - self.start_layer]
        assert device_pool_allocator.available_size() == 0

        # self.free_device_pool(recall_size, layer_id, required_host_indices.flatten().unique().tolist())
        device_page_table_1 = device_page_table_1.flatten()
        device_page_table_1 = device_page_table_1[device_page_table_1 < I32_MAX]
        self.free_device_pool(recall_size, layer_id, device_page_table_1)
        recall_device_indices = device_pool_allocator.alloc(recall_size)
        if recall_device_indices is None:
            raise RuntimeError("Device pool out of memory.")

        self.update_priority_when_use(layer_id, recall_device_indices)

        self.host_token_to_device[layer_id - self.start_layer, recall_host_indices] = recall_device_indices
        self.device_token_to_host[layer_id - self.start_layer, recall_device_indices] = recall_host_indices.to(torch.long)
        device_kv = self.device_pool.get_key_buffer(layer_id)
        device_kv[recall_device_indices] = self.kv_buffer[
            layer_id - self.start_layer, recall_host_indices.to("cpu")
        ].to(self.device_pool.device)

    def recall_miss_tokens_extend_cuda_graph(
        self, 
        recall_host_indices: torch.Tensor, 
        device_page_table_1: torch.Tensor,
        layer_id: int,
    ):
        self.host_need_recall.fill_(0)
        mark_host_need_recall(self.host_need_recall, recall_host_indices)
        recall_size = self.host_need_recall[:-1].sum().to(torch.int32)

        device_pool_allocator = self.device_pool_allocator[layer_id - self.start_layer]
        # print(f"{layer_id=} {recall_size=} {device_pool_allocator.available_size()=}", flush=True)
        if os.getenv("EXTEND_OFFLOAD_DEBUG"):
            recall_check_available_size(
                recall_size, device_pool_allocator.available_size(), device_pool_allocator.size_in_tensor
            )

        self.free_device_pool_cuda_graph(recall_size, layer_id, device_page_table_1)

        recall_device_indices = self.device_pool_loc_alloc_buf
        device_pool_allocator.alloc(recall_size, recall_device_indices)

        self.update_priority_when_use(layer_id, recall_device_indices)

        recall_update_extend(
            self.host_need_recall[:-1],
            recall_device_indices,
            self.host_token_to_device[layer_id - self.start_layer],
            self.device_token_to_host[layer_id - self.start_layer],
            self.device_pool.get_key_buffer(layer_id),
            self.kv_buffer[layer_id - self.start_layer],
            self.recall_counter,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        )

    def recall_miss_tokens_decode_cuda_graph(
        self, 
        recall_host_indices: torch.Tensor, 
        recall_size: torch.Tensor,
        device_page_table_1: torch.Tensor,
        layer_id: int,
    ):
        recall_size = recall_size.to(torch.int32)
        device_pool_allocator = self.device_pool_allocator[layer_id - self.start_layer]
        # print(f"{layer_id=} {recall_size=} {device_pool_allocator.available_size()=}", flush=True)
        # Not True for D in PD disagg
        # recall_check_available_size(
        #     recall_size, device_pool_allocator.available_size(), device_pool_allocator.size_in_tensor
        # )

        available_size = device_pool_allocator.available_size().to(torch.int32)
        free_size = recall_size - available_size
        free_size = torch.clamp_min(free_size, 0)
        self.free_device_pool_cuda_graph(free_size, layer_id, device_page_table_1)

        recall_device_indices = self.device_pool_loc_alloc_buf
        device_pool_allocator.alloc(recall_size, recall_device_indices)
        # print(f"{recall_size=} {recall_device_indices=}")
        # if recall_device_indices is None:
        #     raise RuntimeError("Device pool out of memory.")

        self.update_priority_when_use(layer_id, recall_device_indices)

        recall_update(
            recall_host_indices,
            recall_device_indices,
            self.host_token_to_device[layer_id - self.start_layer],
            self.device_token_to_host[layer_id - self.start_layer],
            self.device_pool.get_key_buffer(layer_id),
            self.kv_buffer[layer_id - self.start_layer],
            self.recall_counter,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        )

    def recall_miss_tokens_decode_cuda_graph_with_prefetch(
        self,
        recall_host_indices: torch.Tensor,
        recall_size: torch.Tensor,
        device_page_table_1: torch.Tensor,
        layer_id: int,
        prefetch_host_loc: torch.Tensor,
        prefetch_kv_buf: torch.Tensor,
        query_recall_counter: torch.Tensor,
    ):
        recall_size = recall_size.to(torch.int32)
        device_pool_allocator = self.device_pool_allocator[layer_id - self.start_layer]
        device_pool_buf = self.device_pool.get_key_buffer(layer_id)
        prefetch_device_start = device_pool_buf.size(0)

        available_size = device_pool_allocator.available_size().to(torch.int32)
        free_size = torch.clamp_min(recall_size - available_size, 0)
        protected_device_page_table_1 = torch.where(
            device_page_table_1 < prefetch_device_start,
            device_page_table_1,
            I32_MAX,
        )
        self.free_device_pool_cuda_graph(
            free_size, layer_id, protected_device_page_table_1
        )

        recall_device_indices = self.device_pool_loc_alloc_buf
        device_pool_allocator.alloc(recall_size, recall_device_indices)

        self.update_priority_when_use(layer_id, recall_device_indices)

        recall_update_with_prefetch(
            recall_host_indices,
            device_page_table_1,
            recall_device_indices,
            self.host_token_to_device[layer_id - self.start_layer],
            self.device_token_to_host[layer_id - self.start_layer],
            device_pool_buf,
            self.kv_buffer[layer_id - self.start_layer],
            prefetch_host_loc,
            prefetch_kv_buf,
            self.recall_counter,
            prefetch_device_start,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        )
        clear_decode_prefetch_mappings(
            prefetch_host_loc,
            self.host_token_to_device[layer_id - self.start_layer],
            query_recall_counter,
            prefetch_device_start,
        )
    
    def update_priority_when_use(
        self, 
        layer_id: int,
        indices: torch.Tensor,  # come from device_pool_loc_alloc_buf
    ):
        # FIFO update
        self.device_pool_priority[layer_id, indices] = self.fifo_counter[layer_id]
        self.fifo_counter[layer_id] += 1

        # empty first slot due to allocator (never allocated)
        # but, the 0 in recall_device_indices (indicating not allocated) will change it
        # so reset it
        self.device_pool_priority[layer_id, 0].fill_(I32_MAX)

    def maybe_get_custom_mem_pool(self):
        return None

    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens
    
    def early_evict(self, layer_id: int, topk_low_result: torch.Tensor):
        topk_low_result = topk_low_result.flatten()
        device_topk_low = self.host_token_to_device[layer_id][topk_low_result]
        device_topk_low = torch.where(device_topk_low == I32_MAX, 0, device_topk_low)
        self.device_pool_priority[layer_id][device_topk_low] = 0
        self.device_pool_priority[layer_id][0] = I32_MAX
