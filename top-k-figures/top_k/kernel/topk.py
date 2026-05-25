# Adapted from https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py

from dataclasses import dataclass

import tilelang
import tilelang.language as T
import torch


@dataclass
class TopKConfig:
    topk: int
    in_dtype: str = "float32"
    out_dtype: str = "int32"
    debug: bool = False


@dataclass
class TopKContext:
    input_tensor: torch.Tensor
    indexes: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    u16_offset: torch.Tensor
    debug_histograms: torch.Tensor
    debug_threshold_bin_id: torch.Tensor
    debug_num_elements_in_threshold_bin: torch.Tensor

    def as_input_tensors(self) -> list[torch.Tensor]:
        """
        Fit in the tilelang profiler interface
        """
        return [
            self.input_tensor,
            self.indexes,
            self.starts,
            self.ends,
            self.u16_offset,
            self.debug_histograms,
            self.debug_threshold_bin_id,
            self.debug_num_elements_in_threshold_bin,
        ]


pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


def convert_to_uint16(x, offset):
    hval = x + offset
    hval = T.Cast("float16", hval)
    bits_uint = T.reinterpret("uint16", hval)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & (0xFFFF),
        bits_uint | (0x8000),
    )
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret("uint32", x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast("uint32", (0xFFFFFFFF)),
        bits_uint | T.Cast("uint32", (0x80000000)),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_impl(
    topk,
    in_dtype="float32",
    out_dtype="int32",
    debug=False,
):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = 8192  # assume the threshold bucket size after first pass is less than 4K

    @T.prim_func
    def tl_topk_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
        starts: T.Tensor[(batch), out_dtype],
        ends: T.Tensor[(batch), out_dtype],
        u16_offset: T.Tensor[(batch,), in_dtype],
        debug_histogram: T.Tensor[(batch, 5, RADIX + 1), "int32"] = None,  # noqa: F821
        debug_threshold_bin_id: T.Tensor[(batch, 5), "int32"] = None,  # noqa: F821
        debug_num_elements_in_threshold_bin: T.Tensor[(batch, 5), "int32"] = None,  # noqa: F821
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], "int32")
            s_histogram = T.alloc_shared([RADIX + 1], "int32")
            s_num_input = T.alloc_shared([2], "int32")
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], "int32")

            l_threshold_bin_id = T.alloc_var("int32")
            l_new_topk = T.alloc_var("int32")
            l_num_input = T.alloc_var("int32")
            l_bin_id32 = T.alloc_var("int32")
            l_val = T.alloc_var("int32")
            l_start_pos = T.alloc_var("int32")
            l_start_idx = T.alloc_var("int32")
            l_end_idx = T.alloc_var("int32")
            l_out_pos = T.alloc_var("int32")
            l_u16_offset = T.alloc_var(in_dtype)

            l_u16_offset = u16_offset[bx]
            l_new_topk = topk
            l_start_idx = starts[bx]
            l_end_idx = ends[bx]

            # stage 1: use 8bit to do quick topk
            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)

            T.sync_threads()
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    inval_int16 = convert_to_uint16(input[bx, input_idx], offset=l_u16_offset)
                    T.atomic_add(s_histogram[inval_int16], 1)
            T.sync_threads()

            if debug:
                for i in T.serial(RADIX + 1):
                    debug_histogram[bx, 0, i] = s_histogram[i]

            # cumsum
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val
                # find threshold bin id
                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            if debug:
                if tx == 0:
                    debug_threshold_bin_id[bx, 0] = l_threshold_bin_id
                    debug_num_elements_in_threshold_bin[bx, 0] = l_new_topk
                T.sync_threads()
            # collect all elements with exponent ≥ threshold
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                T.sync_threads()
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    bin_id = convert_to_uint16(input[bx, input_idx], offset=l_u16_offset)
                    l_bin_id32 = T.Cast("int32", bin_id)
                    if l_bin_id32 > l_threshold_bin_id:
                        # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, pos] = input_idx

                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        # pos = s_num_input[0]
                        pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        s_input_idx[0, pos] = input_idx

            # stage 2: tail pass
            for round in T.serial(4):
                if l_new_topk <= 0:
                    T.loop_break()

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast(
                            "int32",
                            (
                                (
                                    convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]])
                                    >> (24 - round * 8)
                                )
                                & 0xFF
                            ),
                        )
                        T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()
                # cumsum
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    # find threshold bin id
                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast(
                            "int32",
                            (
                                (
                                    convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]])
                                    >> (24 - round * 8)
                                )
                                & 0xFF
                            ),
                        )
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            if round == 3:
                                l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

    return tl_topk_kernel


def make_topk_kernel(
    config: TopKConfig,
) -> tilelang.JITKernel:
    return tl_topk_impl(
        config.topk,
        in_dtype=config.in_dtype,
        out_dtype=config.out_dtype,
        debug=config.debug,
    )


def tl_topk(input, starts, ends, topk, debug=False, u16_offset=0.0):
    """
    Deprecated: use make_topk_kernel_and_forward instead
    """
    batch, seq_len = input.shape
    indexes = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)
    u16_offset_tensor = torch.full((batch,), u16_offset, dtype=torch.float32, device=input.device)
    debug_histograms = torch.zeros(batch, 5, 257, dtype=torch.int32, device=input.device)
    debug_threshold_bin_id = torch.zeros(batch, 5, dtype=torch.int32, device=input.device)
    debug_num_elements_in_threshold_bin = torch.zeros(batch, 5, dtype=torch.int32, device=input.device)
    kernel = tl_topk_impl(topk, debug=debug)
    kernel(
        input,
        indexes,
        starts,
        ends,
        u16_offset_tensor,
        debug_histograms,
        debug_threshold_bin_id,
        debug_num_elements_in_threshold_bin,
    )
    return TopKContext(
        input_tensor=input,
        indexes=indexes,
        starts=starts,
        ends=ends,
        u16_offset=u16_offset_tensor,
        debug_histograms=debug_histograms,
        debug_threshold_bin_id=debug_threshold_bin_id,
        debug_num_elements_in_threshold_bin=debug_num_elements_in_threshold_bin,
    )


def make_topk_context(
    config: TopKConfig,
    input: torch.Tensor,
    u16_offset: float = 0.0,
) -> TopKContext:
    """
    Prepare other tensors needed for topk kernel execution. The shape of these
    tensors depend on the input tensor shape.

    """
    batch, seq_len = input.shape
    out = torch.zeros(batch, config.topk, dtype=torch.int32, device=input.device)

    starts = torch.zeros(
        batch,
        dtype=torch.int32,
        device=input.device,
    )
    ends = torch.full(
        (batch,),
        seq_len,
        dtype=torch.int32,
        device=input.device,
    )
    u16_offset_tensor = torch.full((batch,), u16_offset, dtype=torch.float32, device=input.device)
    debug_histograms = torch.zeros(
        batch,
        5,
        257,
        dtype=torch.int32,
        device=input.device,
    )
    debug_threshold_bin_id = torch.zeros(
        batch,
        5,
        dtype=torch.int32,
        device=input.device,
    )
    debug_num_elements_in_threshold_bin = torch.zeros(batch, 5, dtype=torch.int32, device=input.device)

    ctx = TopKContext(
        input_tensor=input,
        indexes=out,
        starts=starts,
        ends=ends,
        u16_offset=u16_offset_tensor,
        debug_histograms=debug_histograms,
        debug_threshold_bin_id=debug_threshold_bin_id,
        debug_num_elements_in_threshold_bin=debug_num_elements_in_threshold_bin,
    )

    return ctx


def run_kernel_inplace(
    kernel: tilelang.JITKernel,
    input_tensor: torch.Tensor,
    ctx: TopKContext,
):
    kernel(
        input_tensor,
        ctx.indexes,
        ctx.starts,
        ctx.ends,
        ctx.u16_offset,
        ctx.debug_histograms,
        ctx.debug_threshold_bin_id,
        ctx.debug_num_elements_in_threshold_bin,
    )


def make_topk_kernel_and_forward(
    config: TopKConfig,
    input_tensor: torch.Tensor,
    u16_offset: float = 0.0,
) -> tuple[tilelang.JITKernel, TopKContext]:
    ctx = make_topk_context(config, input_tensor, u16_offset)
    kernel = make_topk_kernel(config)
    run_kernel_inplace(kernel, input_tensor, ctx)

    return kernel, ctx
