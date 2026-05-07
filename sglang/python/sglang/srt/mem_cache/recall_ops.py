import torch
import triton
import triton.language as tl

try:
    from sgl_kernel import fast_argtopk as _fast_argtopk
except ImportError as e:  # pragma: no cover - surfaced when get_free_loc is used
    _fast_argtopk = None
    _fast_argtopk_import_error = e
else:
    _fast_argtopk_import_error = None

try:
    from sgl_kernel import fast_argmin_bounded as _fast_argmin_bounded
except ImportError:
    _fast_argmin_bounded = None

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
    _recall_check_available_size_kernel[(1,)](
        recall_size,
        available_size,
        pool_size
    )


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
    priority: torch.Tensor,                 # [device_pool.size + 1]
    protected_index: torch.Tensor,          # [bsz, topk]
    fifo_counter: torch.Tensor              # [1]
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
    device_pool_loc_small_priority_ptr,
    device_to_host_ptr,
    priority_ptr,
    host_to_device_ptr,
    BLOCK: tl.constexpr,
    I32_MAX: tl.constexpr,
):
    pid = tl.program_id(0)
    free_size = tl.load(free_size_ptr)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < free_size

    free_index_device = tl.load(device_pool_loc_small_priority_ptr + offs, mask=mask, other=0)
    tl.device_assert((free_index_device > 0) | (~mask))
    free_index_host = tl.load(device_to_host_ptr + free_index_device)
    tl.device_assert((free_index_device < I32_MAX) | (~mask))
    tl.device_assert((free_index_host < I32_MAX) | (~mask))

    tl.store(free_index_device_ptr + offs, free_index_device, mask=mask)
    tl.device_assert((tl.load(priority_ptr + free_index_device) < I32_MAX) | (~mask))  # no double free
    tl.store(priority_ptr + free_index_device, -1, mask=mask)
    tl.store(device_to_host_ptr + free_index_device, I32_MAX, mask=mask)
    tl.store(host_to_device_ptr + free_index_host, I32_MAX, mask=mask)
        
def free_update(
    free_size: torch.Tensor,                    # [1], in
    free_index_device: torch.Tensor,            # [device_pool.size], out
    device_pool_loc_small_priority: torch.Tensor,  # [device_pool.size + 1], in
    device_to_host: torch.Tensor,               # [device_pool.size + 1], in & out
    priority: torch.Tensor,                     # [device_pool.size + 1], out
    host_to_device: torch.Tensor,               # [host_pool.size + 1], out
):
    device_pool_size = device_pool_loc_small_priority.shape[0]
    BLOCK = 256
    grid = (triton.cdiv(device_pool_size, BLOCK),)
    _free_update_kernel[grid](
        free_size,
        free_index_device,
        device_pool_loc_small_priority,
        device_to_host,
        priority,
        host_to_device,
        BLOCK,
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
    if loc == 0:    # capture (device_pool_loc > 0) or padding (device_pool_loc == 0)
        tl.store(host_token_to_device_ptr + loc, device_pool_loc)
        if device_pool_loc > 0:
            tl.store(device_token_to_host_ptr + device_pool_loc, loc)
    else:
        tl.device_assert(device_pool_loc > 0)
        tl.store(host_token_to_device_ptr + loc, device_pool_loc)
        tl.store(device_token_to_host_ptr + device_pool_loc, loc)

def set_update(
    loc: torch.Tensor,                          # [need_size]
    device_pool_loc: torch.Tensor,              # [device_pool.size]
    device_token_to_host: torch.Tensor,         # [device_pool.size + 1]
    host_token_to_device: torch.Tensor,         # [host_pool.size + 1]
):
    grid = (loc.shape[0],)
    _set_update_kernel[grid](
        loc,
        device_pool_loc,
        device_token_to_host,
        host_token_to_device,
    )


@triton.jit
def _mark_host_need_recall_kernel(
    host_need_recall_ptr,
    recall_host_indices_ptr,
    qlen: tl.constexpr,
    topk: tl.constexpr,
    stride_q: tl.constexpr,
    stride_k: tl.constexpr,
    host_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < qlen * topk
    q_idx = offs // topk
    k_idx = offs - q_idx * topk
    host_idx = tl.load(
        recall_host_indices_ptr + q_idx * stride_q + k_idx * stride_k,
        mask=mask,
        other=-1,
    )
    valid = mask & (host_idx >= 0) & (host_idx < host_size)
    tl.store(host_need_recall_ptr + host_idx, 1, mask=valid)


def mark_host_need_recall(
    host_need_recall: torch.Tensor,             # [host_pool.size + 1]
    recall_host_indices: torch.Tensor,          # [q_len, topk]
):
    # Avoid Tensor.index_put_ here: this path sees duplicate indices and -1
    # sentinels, and PyTorch's advanced-index assignment can synchronize.
    qlen, topk = recall_host_indices.shape
    if qlen * topk == 0:
        return
    BLOCK = 256
    grid = (triton.cdiv(qlen * topk, BLOCK),)
    _mark_host_need_recall_kernel[grid](
        host_need_recall,
        recall_host_indices,
        qlen,
        topk,
        recall_host_indices.stride(0),
        recall_host_indices.stride(1),
        host_need_recall.shape[0] - 1,
        BLOCK,
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
    nope_dim_stride: tl.constexpr,
    rope_dim_stride: tl.constexpr,
    num_dim_tiles: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    need_size = tl.load(need_size_ptr)
    total_dim = nope_dim + rope_dim
    total_work = need_size * num_dim_tiles
    pid = tl.program_id(0)
    grid = tl.num_programs(0)

    work_id = pid
    while work_id < total_work:
        pid_loc = work_id // num_dim_tiles
        pid_blk = work_id % num_dim_tiles
        base = pid_blk * BLOCK
        offs = base + tl.arange(0, BLOCK)
        mask = offs < total_dim

        loc = tl.load(loc_ptr + pid_loc)
        tl.device_assert(loc > 0)
        dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

        if base + BLOCK <= nope_dim:
            src = tl.load(
                cache_k_nope_ptr + pid_loc * nope_stride + offs * nope_dim_stride,
                mask=mask,
            )
        else:
            offs_rope = offs - nope_dim
            src = tl.load(
                cache_k_rope_ptr + pid_loc * rope_stride + offs_rope * rope_dim_stride,
                mask=mask,
            )

        tl.store(dst_ptr, src.to(kv_buffer_ptr.dtype.element_ty), mask=mask)
        work_id += grid


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
    num_dim_tiles = triton.cdiv(total_dim, BLOCK)
    # In CUDA-graph replay `loc` may be the full allocator buffer while only the
    # first `need_size` rows are live. Cap launch count and let the kernel
    # grid-stride over active work to avoid launching O(loc.numel()) CTAs.
    grid = (min(loc.numel() * num_dim_tiles, 4096),)

    _set_mla_kv_buffer_cuda_graph_kernel[grid](
        need_size,
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        cache_k_nope.stride(-1),
        cache_k_rope.stride(-1),
        num_dim_tiles,
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
    rope_range = kv_lora_rank + tl.arange(0, qk_rope_head_dim)
    total_dim = kv_lora_rank + qk_rope_head_dim
    
    off = qid * recall_host_indices_stride + blk * BLOCK
    offs = off + tl.arange(0, BLOCK)
    col_mask = blk * BLOCK + tl.arange(0, BLOCK) < recall_host_indices_stride
    host_indices = tl.load(recall_host_indices_ptr + offs, mask=col_mask, other=-1)
    valid_mask = col_mask & (host_indices >= 0)
    valid_i32 = valid_mask.to(tl.int32)
    num_valid = tl.sum(valid_i32, axis=0)
    if num_valid == 0:
        return

    start_pos = tl.atomic_add(recall_counter_ptr, num_valid)
    local_pos = start_pos
    for i in range(BLOCK):
        host_idx = tl.load(
            recall_host_indices_ptr + off + i,
            mask=blk * BLOCK + i < recall_host_indices_stride,
            other=-1,
        )
        if host_idx >= 0:
            device_idx = tl.load(recall_device_indices_ptr + local_pos)
            tl.device_assert(device_idx > 0)
            tl.store(host_token_to_device_ptr + host_idx, device_idx)
            tl.store(device_token_to_host_ptr + device_idx, host_idx)

            t_nope = tl.load(layer_host_kv_ptr + host_idx * total_dim + nope_range)
            tl.store(layer_device_kv_ptr + device_idx * total_dim + nope_range, t_nope)
            t_rope = tl.load(layer_host_kv_ptr + host_idx * total_dim + rope_range)
            tl.store(layer_device_kv_ptr + device_idx * total_dim + rope_range, t_rope)
            local_pos += 1

def recall_update(
    recall_host_indices: torch.Tensor,          # [q_len, topk]
    recall_device_indices: torch.Tensor,        # [device_pool.size]
    host_token_to_device: torch.Tensor,         # [size]
    device_token_to_host: torch.Tensor,         # [device_pool.size]
    layer_device_kv: torch.Tensor,              # [device_pool.size, 1, 576]
    layer_host_kv: torch.Tensor,                # [size, 1, 576]
    recall_counter: torch.Tensor,               # [1]
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
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
def _recall_update_with_prefetch_kernel(
    recall_counter_ptr,
    recall_host_indices_ptr,
    device_page_table_1_ptr,
    recall_device_indices_ptr,
    host_token_to_device_ptr,
    device_token_to_host_ptr,
    layer_device_kv_ptr,
    layer_host_kv_ptr,
    prefetch_host_loc_ptr,
    prefetch_kv_buf_ptr,
    recall_host_indices_stride: tl.constexpr,
    device_page_table_stride: tl.constexpr,
    prefetch_host_stride: tl.constexpr,
    prefetch_q_stride: tl.constexpr,
    prefetch_slot_stride: tl.constexpr,
    prefetch_dim_stride: tl.constexpr,
    prefetch_device_start: tl.constexpr,
    req_pf_max: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    BLOCK: tl.constexpr,
    I32_MAX: tl.constexpr,
):
    tl.static_assert(kv_lora_rank == 512 and qk_rope_head_dim == 64)

    qid = tl.program_id(0)
    blk = tl.program_id(1)

    nope_range = tl.arange(0, kv_lora_rank)
    rope_range = kv_lora_rank + tl.arange(0, qk_rope_head_dim)
    total_dim = kv_lora_rank + qk_rope_head_dim

    recall_off = qid * recall_host_indices_stride + blk * BLOCK
    device_page_off = qid * device_page_table_stride + blk * BLOCK
    col_mask = blk * BLOCK + tl.arange(0, BLOCK) < recall_host_indices_stride
    host_indices = tl.load(
        recall_host_indices_ptr + recall_off + tl.arange(0, BLOCK),
        mask=col_mask,
        other=-1,
    )
    valid_mask = col_mask & (host_indices >= 0)
    valid_i32 = valid_mask.to(tl.int32)
    num_valid = tl.sum(valid_i32, axis=0)
    if num_valid == 0:
        return

    start_pos = tl.atomic_add(recall_counter_ptr, num_valid)
    local_pos = start_pos
    for i in range(BLOCK):
        host_idx = tl.load(
            recall_host_indices_ptr + recall_off + i,
            mask=blk * BLOCK + i < recall_host_indices_stride,
            other=-1,
        )
        if host_idx >= 0:
            old_device_idx = tl.load(device_page_table_1_ptr + device_page_off + i)
            device_idx = tl.load(recall_device_indices_ptr + local_pos)
            tl.device_assert(device_idx > 0)
            tl.store(host_token_to_device_ptr + host_idx, device_idx)
            tl.store(device_token_to_host_ptr + device_idx, host_idx)
            tl.store(device_page_table_1_ptr + device_page_off + i, device_idx)

            is_prefetched = (old_device_idx >= prefetch_device_start) & (
                old_device_idx < I32_MAX
            )
            slot = old_device_idx - prefetch_device_start
            if is_prefetched & (slot < req_pf_max):
                prefetched_host_idx = tl.load(
                    prefetch_host_loc_ptr + qid * prefetch_host_stride + slot
                )
                is_prefetched = prefetched_host_idx == host_idx

            if is_prefetched:
                prefetch_base = (
                    qid * prefetch_q_stride
                    + slot * prefetch_slot_stride
                )
                t_nope = tl.load(
                    prefetch_kv_buf_ptr + prefetch_base + nope_range * prefetch_dim_stride
                )
                tl.store(layer_device_kv_ptr + device_idx * total_dim + nope_range, t_nope)
                t_rope = tl.load(
                    prefetch_kv_buf_ptr + prefetch_base + rope_range * prefetch_dim_stride
                )
                tl.store(layer_device_kv_ptr + device_idx * total_dim + rope_range, t_rope)
            else:
                t_nope = tl.load(layer_host_kv_ptr + host_idx * total_dim + nope_range)
                tl.store(layer_device_kv_ptr + device_idx * total_dim + nope_range, t_nope)
                t_rope = tl.load(layer_host_kv_ptr + host_idx * total_dim + rope_range)
                tl.store(layer_device_kv_ptr + device_idx * total_dim + rope_range, t_rope)
            local_pos += 1


def recall_update_with_prefetch(
    recall_host_indices: torch.Tensor,          # [q_len, topk]
    device_page_table_1: torch.Tensor,          # [q_len, topk], in & out
    recall_device_indices: torch.Tensor,        # [device_pool.size]
    host_token_to_device: torch.Tensor,         # [size]
    device_token_to_host: torch.Tensor,         # [device_pool.size]
    layer_device_kv: torch.Tensor,              # [device_pool.size, 1, 576]
    layer_host_kv: torch.Tensor,                # [size, 1, 576]
    prefetch_host_loc: torch.Tensor,            # [q_len, req_pf_max]
    prefetch_kv_buf: torch.Tensor,              # [q_len, req_pf_max, 576]
    recall_counter: torch.Tensor,               # [1]
    prefetch_device_start: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
    qlen, topk = recall_host_indices.shape
    BLOCK = 32
    grid = (qlen, triton.cdiv(topk, BLOCK))
    recall_counter.fill_(0)
    _recall_update_with_prefetch_kernel[grid](
        recall_counter,
        recall_host_indices,
        device_page_table_1,
        recall_device_indices,
        host_token_to_device,
        device_token_to_host,
        layer_device_kv,
        layer_host_kv,
        prefetch_host_loc,
        prefetch_kv_buf,
        recall_host_indices.stride(0),
        device_page_table_1.stride(0),
        prefetch_host_loc.stride(0),
        prefetch_kv_buf.stride(0),
        prefetch_kv_buf.stride(1),
        prefetch_kv_buf.stride(2),
        prefetch_device_start,
        prefetch_kv_buf.size(1),
        kv_lora_rank,
        qk_rope_head_dim,
        BLOCK,
        I32_MAX,
    )


@triton.jit
def _clear_decode_prefetch_mappings_kernel(
    prefetch_host_loc_ptr,
    host_token_to_device_ptr,
    query_recall_counter_ptr,
    prefetch_host_stride: tl.constexpr,
    prefetch_device_start: tl.constexpr,
    req_pf_max: tl.constexpr,
    BLOCK: tl.constexpr,
    I32_MAX: tl.constexpr,
):
    qid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    num_prefetched = tl.minimum(tl.load(query_recall_counter_ptr + qid), req_pf_max)
    mask = offsets < num_prefetched
    host_idx = tl.load(
        prefetch_host_loc_ptr + qid * prefetch_host_stride + offsets,
        mask=mask,
        other=-1,
    )
    current_device_idx = tl.load(
        host_token_to_device_ptr + host_idx,
        mask=mask & (host_idx > 0),
        other=I32_MAX,
    )
    expected_temp_idx = prefetch_device_start + offsets
    should_clear = mask & (host_idx > 0) & (current_device_idx == expected_temp_idx)
    tl.store(host_token_to_device_ptr + host_idx, I32_MAX, mask=should_clear)


def clear_decode_prefetch_mappings(
    prefetch_host_loc: torch.Tensor,
    host_token_to_device: torch.Tensor,
    query_recall_counter: torch.Tensor,
    prefetch_device_start: int,
):
    qlen, req_pf_max = prefetch_host_loc.shape
    _clear_decode_prefetch_mappings_kernel[(qlen,)](
        prefetch_host_loc,
        host_token_to_device,
        query_recall_counter,
        prefetch_host_loc.stride(0),
        prefetch_device_start,
        req_pf_max,
        triton.next_power_of_2(req_pf_max),
        I32_MAX,
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
    rope_range = kv_lora_rank + tl.arange(0, qk_rope_head_dim)
    total_dim = kv_lora_rank + qk_rope_head_dim

    tid = tl.program_id(0)
    for i in range(BLOCK):
        host_idx = tid * BLOCK + i
        if host_idx < host_size:
            need_recall = tl.load(host_need_recall_ptr + host_idx)
            if need_recall:
                poc = tl.atomic_add(recall_counter_ptr, 1)
                device_idx = tl.load(recall_device_indices_ptr + poc)
                tl.device_assert(device_idx > 0)
                tl.store(host_token_to_device_ptr + host_idx, device_idx)
                tl.store(device_token_to_host_ptr + device_idx, host_idx)

                t_nope = tl.load(layer_host_kv_ptr + host_idx * total_dim + nope_range)
                tl.store(layer_device_kv_ptr + device_idx * total_dim + nope_range, t_nope)
                t_rope = tl.load(layer_host_kv_ptr + host_idx * total_dim + rope_range)
                tl.store(layer_device_kv_ptr + device_idx * total_dim + rope_range, t_rope)


def recall_update_extend(
    host_need_recall: torch.Tensor,             # [host_pool.size]
    recall_device_indices: torch.Tensor,        # [device_pool.size]
    host_token_to_device: torch.Tensor,         # [size]
    device_token_to_host: torch.Tensor,         # [device_pool.size]
    layer_device_kv: torch.Tensor,              # [device_pool.size, 1, 576]
    layer_host_kv: torch.Tensor,                # [size, 1, 576]
    recall_counter: torch.Tensor,               # [1]
    kv_lora_rank: int,
    qk_rope_head_dim: int,
):
    host_size, = host_need_recall.shape
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

def get_free_loc(priority: torch.Tensor, loc: torch.Tensor, free_size: torch.Tensor):
    """Select the `free_size` indices with the smallest priority values.

    This is an argtopk (smallest) over `priority`. In the recall path the
    live priorities are bounded non-negative FIFO values, while free slots are
    remapped to `I32_MAX` by the caller before invoking this function. This
    path requires the dedicated bounded-range selector
    `sgl_kernel.fast_argmin_bounded`; we intentionally do not fall back to the
    generic `fast_argtopk(..., largest=False)` implementation.

    The kernel supports CUDA graph capture because `free_size` is passed as a
    device-side scalar tensor and the output buffer has fixed capacity.

    Note on perf: we keep this wrapper intentionally thin so the hot path
    is just one Python->C++ dispatcher call. Callers must ensure
    `priority`, `loc`, and `free_size` are all int32; the underlying
    C++ interface will raise otherwise.
    """
    if _fast_argmin_bounded is None:
        raise RuntimeError(
            "get_free_loc requires sgl_kernel.fast_argmin_bounded, but it could not be imported"
        )
    _fast_argmin_bounded(priority, loc, free_size)
