import torch
import triton
import triton.language as tl

I32_MAX = torch.iinfo(torch.int32).max


@triton.jit
def _recall_check_available_size_kernel(
    recall_size_ptr,
    available_size_ptr,
    pool_size_ptr,
):
    recall_size = tl.load(recall_size_ptr)
    available_size = tl.load(available_size_ptr)
    pool_size = tl.load(pool_size_ptr)
    tl.device_assert(recall_size == 0 or available_size == 0)
    tl.device_assert(recall_size <= pool_size - available_size)


def recall_check_available_size(
    recall_size: torch.Tensor,
    available_size: torch.Tensor,
    pool_size: torch.Tensor,
):
    _recall_check_available_size_kernel[(1,)](recall_size, available_size, pool_size)


@triton.jit
def _protect_kernel(
    priority_ptr,
    protected_index_ptr,
    fifo_counter_ptr,
    index_stride: tl.constexpr,
    BLOCK: tl.constexpr,
    I32_MAX: tl.constexpr,
):
    bsz = tl.program_id(0)
    blk = tl.program_id(1)

    counter = tl.load(fifo_counter_ptr)
    offs = bsz * index_stride + blk * BLOCK + tl.arange(0, BLOCK)
    index = tl.load(protected_index_ptr + offs)

    # tl.device_assert(index != 0)
    # index == I32_MAX: not in device_pool
    # index < 0: not selected
    tl.store(priority_ptr + index, counter, mask=((index < I32_MAX) & (index > 0)))


def protect(
    priority: torch.Tensor,  # [device_pool.size + 1]
    protected_index: torch.Tensor,  # [bsz, topk]
    fifo_counter: torch.Tensor,  # [1]
):
    bsz, topk = protected_index.shape
    BLOCK = 128
    grid = (bsz, triton.cdiv(topk, BLOCK))
    _protect_kernel[grid](
        priority,
        protected_index,
        fifo_counter,
        protected_index.stride(0),
        BLOCK,
        I32_MAX,
    )


@triton.jit
def _free_update_kernel(
    free_size_ptr,
    free_index_device_ptr,
    sort_loc_ptr,
    device_to_host_ptr,
    priority_ptr,
    host_to_device_ptr,
    I32_MAX: tl.constexpr,
):
    idx = tl.program_id(0)
    free_size = tl.load(free_size_ptr)
    if idx >= free_size:
        return

    free_index_device = tl.load(sort_loc_ptr + idx)
    tl.device_assert(free_index_device > 0)
    free_index_host = tl.load(device_to_host_ptr + free_index_device)
    tl.device_assert(free_index_device < I32_MAX)

    # tl.device_print("free_size", free_size)
    # tl.device_print("free_index_device", free_index_device, free_index_host)

    tl.store(free_index_device_ptr + idx, free_index_device)
    tl.device_assert(tl.load(priority_ptr + free_index_device) < I32_MAX)  # no double free
    tl.store(priority_ptr + free_index_device, -1)
    tl.store(device_to_host_ptr + free_index_device, I32_MAX)
    tl.store(host_to_device_ptr + free_index_host, I32_MAX)


def free_update(
    free_size: torch.Tensor,  # [1]
    free_index_device: torch.Tensor,  # [device_pool.size]
    sort_loc: torch.Tensor,  # [device_pool.size + 1]
    device_to_host: torch.Tensor,  # [device_pool.size + 1]
    priority: torch.Tensor,  # [device_pool.size + 1]
    host_to_device: torch.Tensor,  # [host_pool.size + 1]
):
    device_pool_size = sort_loc.shape[0]
    grid = (device_pool_size,)
    _free_update_kernel[grid](
        free_size,
        free_index_device,
        sort_loc,
        device_to_host,
        priority,
        host_to_device,
        I32_MAX,
    )


@triton.jit
def _set_update_kernel(
    loc_ptr,
    device_pool_loc_ptr,
    device_token_to_host_ptr,
    host_token_to_device_ptr,
):
    idx = tl.program_id(0)
    loc = tl.load(loc_ptr + idx)
    device_pool_loc = tl.load(device_pool_loc_ptr + idx)
    if loc == 0:  # capture (device_pool_loc > 0) or padding (device_pool_loc == 0)
        tl.store(host_token_to_device_ptr + loc, device_pool_loc)
        if device_pool_loc > 0:
            tl.store(device_token_to_host_ptr + device_pool_loc, loc)
    else:
        tl.device_assert(device_pool_loc > 0)
        tl.store(host_token_to_device_ptr + loc, device_pool_loc)
        tl.store(device_token_to_host_ptr + device_pool_loc, loc)


def set_update(
    loc: torch.Tensor,  # [need_size]
    device_pool_loc: torch.Tensor,  # [device_pool.size]
    device_token_to_host: torch.Tensor,  # [device_pool.size + 1]
    host_token_to_device: torch.Tensor,  # [host_pool.size + 1]
):
    grid = (loc.shape[0],)
    _set_update_kernel[grid](
        loc,
        device_pool_loc,
        device_token_to_host,
        host_token_to_device,
    )


@triton.jit
def _set_mla_kv_buffer_cuda_graph_kernel(
    need_size_ptr,
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)
    need_size = tl.load(need_size_ptr)
    if pid_loc >= need_size:
        return

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    mask = offs < total_dim

    loc = tl.load(loc_ptr + pid_loc)
    tl.device_assert(loc > 0)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
        )
    else:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
        )

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_cuda_graph(
    need_size: torch.Tensor,
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = 128
    n_loc = loc.numel()
    grid = (n_loc, triton.cdiv(total_dim, BLOCK))

    _set_mla_kv_buffer_cuda_graph_kernel[grid](
        need_size,
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )


@triton.jit
def _recall_update_kernel(
    recall_counter_ptr,
    recall_host_indices_ptr,
    recall_device_indices_ptr,
    host_token_to_device_ptr,
    device_token_to_host_ptr,
    layer_device_kv_ptr,
    layer_host_kv_ptr,
    recall_host_indices_stride: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.static_assert(kv_lora_rank == 512 and qk_rope_head_dim == 64)

    qid = tl.program_id(0)
    blk = tl.program_id(1)

    nope_range = tl.arange(0, kv_lora_rank)
    rope_range = tl.arange(0, qk_rope_head_dim)
    total_dim = kv_lora_rank + qk_rope_head_dim

    off = qid * recall_host_indices_stride + blk * BLOCK
    # host_indices = tl.load(recall_host_indices_ptr + offs)
    for i in range(BLOCK):
        host_idx = tl.load(recall_host_indices_ptr + off + i).to(tl.int64)
        if host_idx >= 0:
            pos = tl.atomic_add(recall_counter_ptr, 1)
            device_idx = tl.load(recall_device_indices_ptr + pos)
            tl.device_assert(device_idx > 0)
            device_idx_i64 = device_idx.to(tl.int64)
            tl.store(host_token_to_device_ptr + host_idx, device_idx)
            tl.store(device_token_to_host_ptr + device_idx, host_idx)

            host_base = host_idx * total_dim
            device_base = device_idx_i64 * total_dim

            t_nope = tl.load(layer_host_kv_ptr + host_base + nope_range)
            tl.store(layer_device_kv_ptr + device_base + nope_range, t_nope)
            t_rope = tl.load(layer_host_kv_ptr + host_base + kv_lora_rank + rope_range)
            tl.store(layer_device_kv_ptr + device_base + kv_lora_rank + rope_range, t_rope)


def recall_update(
    recall_host_indices: torch.Tensor,  # [q_len, topk]
    recall_device_indices: torch.Tensor,  # [device_pool.size]
    host_token_to_device: torch.Tensor,  # [size]
    device_token_to_host: torch.Tensor,  # [device_pool.size]
    layer_device_kv: torch.Tensor,  # [device_pool.size, 1, 576]
    layer_host_kv: torch.Tensor,  # [size, 1, 576]
    recall_counter: torch.Tensor,  # [1]
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
    """
    Args:
        recall_host_indices: a 2d tensor of size (q_len, topk); each element is a host index to recall; -1 means no recall
        recall_device_indices: a 1d tensor of size (recall_size,) of free slot indices on the device pool to recall into
        host_token_to_device: (host_pool_size + 1,), h2d
        device_token_to_host: (device_pool.size + 1,) d2h
        layer_device_kv: device kv buffer, (device_pool.size, 1, 576)
        layer_host_kv: host kv buffer, (host_pool_size, 1, 576)
        recall_counter: (1,) uint32 counter; should finally be very close to recall_size
        kv_lora_rank: constant int 512
        qk_rope_head_dim: constant int 64
    """
    qlen, topk = recall_host_indices.shape
    BLOCK = 32
    grid = (qlen, triton.cdiv(topk, BLOCK))
    recall_counter.fill_(0)
    _recall_update_kernel[grid](
        recall_counter,
        recall_host_indices,
        recall_device_indices,
        host_token_to_device,
        device_token_to_host,
        layer_device_kv,
        layer_host_kv,
        recall_host_indices.stride(0),
        kv_lora_rank,
        qk_rope_head_dim,
        BLOCK,
    )


@triton.jit
def _recall_update_extend_kernel(
    host_need_recall_ptr,
    recall_counter_ptr,
    recall_device_indices_ptr,
    host_token_to_device_ptr,
    device_token_to_host_ptr,
    layer_device_kv_ptr,
    layer_host_kv_ptr,
    host_size: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.static_assert(kv_lora_rank == 512 and qk_rope_head_dim == 64)

    nope_range = tl.arange(0, kv_lora_rank)
    rope_range = tl.arange(0, qk_rope_head_dim)
    total_dim = kv_lora_rank + qk_rope_head_dim

    tid = tl.program_id(0)
    for i in range(BLOCK):
        host_idx = (tid * BLOCK + i).to(tl.int64)
        if host_idx < host_size:
            need_recall = tl.load(host_need_recall_ptr + host_idx)
            if need_recall:
                poc = tl.atomic_add(recall_counter_ptr, 1)
                device_idx = tl.load(recall_device_indices_ptr + poc)
                tl.device_assert(device_idx > 0)
                device_idx_i64 = device_idx.to(tl.int64)
                tl.store(host_token_to_device_ptr + host_idx, device_idx)
                tl.store(device_token_to_host_ptr + device_idx, host_idx)

                host_base = host_idx * total_dim
                device_base = device_idx_i64 * total_dim

                t_nope = tl.load(layer_host_kv_ptr + host_base + nope_range)
                tl.store(layer_device_kv_ptr + device_base + nope_range, t_nope)
                t_rope = tl.load(layer_host_kv_ptr + host_base + kv_lora_rank + rope_range)
                tl.store(layer_device_kv_ptr + device_base + kv_lora_rank + rope_range, t_rope)


def recall_update_extend(
    host_need_recall: torch.Tensor,  # [host_pool.size]
    recall_device_indices: torch.Tensor,  # [device_pool.size]
    host_token_to_device: torch.Tensor,  # [size]
    device_token_to_host: torch.Tensor,  # [device_pool.size]
    layer_device_kv: torch.Tensor,  # [device_pool.size, 1, 576]
    layer_host_kv: torch.Tensor,  # [size, 1, 576]
    recall_counter: torch.Tensor,  # [1]
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
    """
    Args:
        host_need_recall: a 1d mask of (host_pool_size,); host_need_recall[i] == 1 -> need to recall host index i
        recall_device_indices: a tensor of size (recall_size,) of free slot indices on the device pool to recall into
        host_token_to_device: (host_pool_size + 1,), h2d
        device_token_to_host: (device_pool.size + 1,) d2h
        layer_device_kv: device kv buffer, (device_pool.size, 1, 576)
        layer_host_kv: host kv buffer, (host_pool_size, 1, 576)
        recall_counter: (1,) uint32 counter; should finally be very close to recall_size
        kv_lora_rank: constant int 512
        qk_rope_head_dim: constant int 64
    """
    # recall_update_extend(
    #     host_need_recall[:-1], # a 1d mask of (host_pool_size,); host_need_recall[i] == 1 -> need to recall host index i
    #     recall_device_indices, # a tensor of size (recall_size,) of free slot indices on the device pool to recall into
    #     self.host_token_to_device[layer_id - self.start_layer], # (host_pool_size + 1,), h2d
    #     self.device_token_to_host[layer_id - self.start_layer], # (device_pool.size + 1,) d2h
    #     self.device_pool.get_key_buffer(layer_id), # device kv buffer, (device_pool.size, 1, 576)
    #     self.kv_buffer[layer_id - self.start_layer], # host kv buffer, (host_pool_size, 1, 576)
    #     self.recall_counter, # (1,) uint32 counter; should finally be very close to recall_size
    #     self.kv_lora_rank, # constant int 512
    #     self.qk_rope_head_dim, # constant int 64
    # )
    (host_size,) = host_need_recall.shape
    BLOCK = 256
    grid = (triton.cdiv(host_size, BLOCK),)
    recall_counter.fill_(0)
    _recall_update_extend_kernel[grid](
        host_need_recall,
        recall_counter,
        recall_device_indices,
        host_token_to_device,
        device_token_to_host,
        layer_device_kv,
        layer_host_kv,
        host_size,
        kv_lora_rank,
        qk_rope_head_dim,
        BLOCK,
    )


@triton.jit
def _recall_update_kernel_buggy(
    recall_counter_ptr,
    recall_host_indices_ptr,
    recall_device_indices_ptr,
    host_token_to_device_ptr,
    device_token_to_host_ptr,
    recall_host_indices_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    qid = tl.program_id(0)
    blk = tl.program_id(1)

    offs = qid * recall_host_indices_stride + blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    host_indices = tl.load(recall_host_indices_ptr + offs)

    valid_mask = host_indices >= 0

    num_valid = tl.sum(valid_mask, axis=0)
    start_pos = tl.atomic_add(recall_counter_ptr, num_valid)

    local_offsets = tl.cumsum(valid_mask, axis=0)
    device_idx_positions = start_pos + local_offsets

    device_indices = tl.load(recall_device_indices_ptr + device_idx_positions, mask=valid_mask)
    tl.device_print("valid_mask", valid_mask)
    tl.device_print("local_offl", local_offsets)
    tl.device_print("device_indi", device_indices)

    tl.device_assert((device_indices > 0) | (~valid_mask), "Loaded device index must be positive")

    tl.store(host_token_to_device_ptr + host_indices, device_indices, mask=valid_mask)
    tl.store(device_token_to_host_ptr + device_indices, host_indices, mask=valid_mask)
