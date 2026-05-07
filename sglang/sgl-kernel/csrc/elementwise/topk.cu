/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 * We:
 * 1. adapt from tilelang to pure cuda
 * 2. optimize the performance a little
 * 3. fix the potential illegal memory access
 */
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr size_t kSmem = 32 * 1024 * sizeof(uint32_t);  // 128KB
constexpr size_t kArgTopKLargeSmem = 48 * 1024 * sizeof(uint32_t);  // 192KB on Hopper-class parts
constexpr int kGetFreeLocMaxPriority = 10'000'000;

struct FastTopKParams {
  const float* __restrict__ input;  // [B, input_stride]
  int32_t* __restrict__ indices;    // [B, TopK]
  int32_t* __restrict__ lengths;    // [B]
  int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table,
    float* __restrict__ topk_logit) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
  if (tid == 0 && topk_logit != nullptr) {
    *topk_logit = (length >= TopK) ? score[TopK - 1] : 0.0f;
  }
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ void fast_topk_cuda_tl(const float* __restrict__ input, int* __restrict__ index, int length) {
  // An optimized topk kernel copied from tilelang kernel
  // We assume length > TopK here, or it will crash
  int topk = TopK;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        /// NOTE: (dark) fuse the histogram computation here
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[TopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              /// NOTE: (dark) fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params) {
  const auto& [input, indices, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda_tl(score, indice, length);
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        float* __restrict__ topk_logits) {
  const auto& [input, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto topk_logit = (topk_logits == nullptr) ? nullptr : topk_logits + bid;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry, topk_logit);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, length);
    if (topk_logit != nullptr && tid == 0) {
      const auto topk_idx = s_indices[TopK - 1];
      *topk_logit = (topk_idx >= 0) ? score[topk_idx] : 0.0f;
    }
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        float* __restrict__ topk_logits,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto topk_logit = (topk_logits == nullptr) ? nullptr : topk_logits + bid;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry, topk_logit);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda_tl(score, s_indices, length);
    if (topk_logit != nullptr && tid == 0) {
      const auto topk_idx = s_indices[TopK - 1];
      *topk_logit = (topk_idx >= 0) ? score[topk_idx] : 0.0f;
    }
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

auto get_params(at::Tensor score, at::Tensor lengths, std::optional<at::Tensor> indices_opt = std::nullopt)
    -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(indices.size(1) == TopK);
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}

// ============ argtopk_m2048: int32 scores, batch=1, variable topk ============

// Monotone mapping int32 -> uint32 used by the radix selector.
// - Largest=true: XOR sign bit; ordering is preserved (smaller int -> smaller uint).
//   This is the classic "flip sign bit" trick for signed radix sort.
// - Largest=false: XOR with 0x7FFFFFFF; ordering is reversed (smaller int -> larger uint).
//   Radix selection still picks "largest uint" bins, which correspond to "smallest int".
template <bool Largest>
__device__ __forceinline__ auto convert_int_to_uint32(int32_t x) -> uint32_t {
  constexpr uint32_t kMask = Largest ? 0x80000000u : 0x7FFFFFFFu;
  return static_cast<uint32_t>(x) ^ kMask;
}

template <bool Largest>
__device__ __forceinline__ auto convert_int_to_uint8(int32_t x) -> uint8_t {
  return static_cast<uint8_t>(convert_int_to_uint32<Largest>(x) >> 24);
}

__device__ void naive_argtopk_int32(int32_t* __restrict__ indice, int32_t length, int topk) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < topk; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

template <bool Largest, size_t DynamicSmemBytes>
__device__ void fast_argtopk_int32_impl(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ index,
    int length,
    int topk_total) {
  // Local aliases so the radix body stays identical to the original code.
  auto to_u32 = [] __device__ (int32_t x) { return convert_int_to_uint32<Largest>(x); };
  auto to_u8  = [] __device__ (int32_t x) { return convert_int_to_uint8<Largest>(x); };
  // Radix-based topk selection for int32 scores.
  // Hybrid approach: uses shared-memory index buffers for refinement passes
  // when candidates fit, falls back to global memory re-scan when they overflow.
  // Best case: 2 global scans + 3 fast smem scans.
  // Worst case (narrow value range): up to 5 global scans.
  int topk = topk_total;
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = static_cast<int>(DynamicSmemBytes / (2 * sizeof(int)));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_bin_counter[RADIX];

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_arg_input_storage[];
  auto* s_input_idx = reinterpret_cast<int (*)[SMEM_INPUT_SIZE]>(s_arg_input_storage);

  const int tx = threadIdx.x;

  // Accumulated byte-prefix for global-memory fallback scans
  uint32_t acc_value = 0;
  uint32_t acc_mask = 0;

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  // ---- Stage 1: coarse histogram from global memory (byte 3) ----
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = static_cast<int>(to_u8(input[idx]));
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  {
    const auto threshold_bin = s_threshold_bin_id;
    const int above_count = s_histogram[threshold_bin + 1];
    topk -= above_count;
    acc_value = static_cast<uint32_t>(threshold_bin) << 24;
    acc_mask = 0xFF000000u;

    if (topk == 0) {
      // All top-k have byte3 > threshold — use per-bin counters to avoid s_counter contention
      if (tx < RADIX) s_bin_counter[tx] = 0;
      __syncthreads();
      for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto bin = static_cast<int>(to_u8(input[idx]));
        if (bin > threshold_bin) {
          const auto pos = s_histogram[bin + 1] + ::atomicAdd(&s_bin_counter[bin], 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      return;
    }

    // Fused scan: output above-threshold (per-bin) + store candidates in smem[0] + build byte-2 histogram
    // Keep s_histogram (buf[0]) as per-bin offsets; build byte-2 histogram in buf[1]
    if (tx < RADIX) s_bin_counter[tx] = 0;
    if (tx < RADIX + 1) s_histogram_buf[1][tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto val = input[idx];
      const auto bin = static_cast<int>(to_u8(val));
      if (bin > threshold_bin) {
        // Per-bin output: distributes atomics across ~N_bins addresses instead of 1
        const auto pos = s_histogram[bin + 1] + ::atomicAdd(&s_bin_counter[bin], 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        // Always build histogram, even if smem buffer overflows
        const auto sub_bin = static_cast<int>((to_u32(val) >> 16) & 0xFF);
        ::atomicAdd(&s_histogram_buf[1][sub_bin], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
        }
      }
    }

    // Ensure all threads finished the fused scan before overwriting buf[0]
    __syncthreads();
    // Set s_counter for refinement passes, copy byte-2 histogram from buf[1] to buf[0]
    if (tx == 0) s_counter = above_count;
    if (tx < RADIX + 1) s_histogram[tx] = s_histogram_buf[1][tx];
    __syncthreads();
  }

  // ---- Stage 2: 3 refinement passes (bytes 2, 1, 0) ----
  for (int round = 0; round < 3; ++round) {
    const auto r_idx = round & 1;
    const auto raw_num_input = s_num_input[r_idx];
    const bool smem_valid = (raw_num_input <= SMEM_INPUT_SIZE);
    const auto num_input = smem_valid ? raw_num_input : 0;
    const int offset = 16 - round * 8;

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    // Save prior prefix before updating
    const uint32_t prior_value = acc_value;
    const uint32_t prior_mask = acc_mask;
    acc_value |= (static_cast<uint32_t>(threshold_bin) << offset);
    acc_mask |= (0xFFu << offset);

    if (topk == 0) {
      // Output all elements with current byte > threshold
      if (smem_valid) {
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          if (static_cast<int>((to_u32(input[idx]) >> offset) & 0xFF) > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
          }
        }
      } else {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
          const auto uval = to_u32(input[idx]);
          if ((uval & prior_mask) != prior_value) continue;
          if (static_cast<int>((uval >> offset) & 0xFF) > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
          }
        }
      }
      __syncthreads();
      return;
    }

    if (round == 2) {
      // Last round (byte 0): output above-threshold + pick remaining ties
      __shared__ int s_last_remain;
      if (tx == 0) s_last_remain = topk;
      __syncthreads();
      if (smem_valid) {
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto bin = static_cast<int>((to_u32(input[idx]) >> offset) & 0xFF);
          if (bin > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
          } else if (bin == threshold_bin) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[topk_total - pos] = idx;
            }
          }
        }
      } else {
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
          const auto uval = to_u32(input[idx]);
          if ((uval & prior_mask) != prior_value) continue;
          const auto bin = static_cast<int>((uval >> offset) & 0xFF);
          if (bin > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
          } else if (bin == threshold_bin) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[topk_total - pos] = idx;
            }
          }
        }
      }
      __syncthreads();
    } else {
      // Rounds 0, 1: output + store candidates in next smem buffer + build next histogram
      __syncthreads();
      if (tx < RADIX + 1) s_histogram[tx] = 0;
      __syncthreads();

      if (smem_valid) {
        for (int i = tx; i < num_input; i += BLOCK_SIZE) {
          const auto idx = s_input_idx[r_idx][i];
          const auto val = input[idx];
          const auto uval = to_u32(val);
          const auto bin = static_cast<int>((uval >> offset) & 0xFF);
          if (bin > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
          } else if (bin == threshold_bin) {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            const auto sub_bin = static_cast<int>((uval >> (offset - 8)) & 0xFF);
            ::atomicAdd(&s_histogram[sub_bin], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
            }
          }
        }
      } else {
        // Fallback: scan global memory with accumulated prefix mask
        for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
          const auto uval = to_u32(input[idx]);
          if ((uval & prior_mask) != prior_value) continue;
          const auto bin = static_cast<int>((uval >> offset) & 0xFF);
          if (bin > threshold_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
          } else if (bin == threshold_bin) {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            const auto sub_bin = static_cast<int>((uval >> (offset - 8)) & 0xFF);
            ::atomicAdd(&s_histogram[sub_bin], 1);
            if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
              s_input_idx[r_idx ^ 1][pos] = idx;
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <bool Largest, size_t DynamicSmemBytes>
__global__ __launch_bounds__(kThreadsPerBlock)
    void argtopk_int32_kernel(
        const int32_t* __restrict__ input,
        int32_t* __restrict__ indices,
        const int32_t* __restrict__ topk_ptr,
        int32_t length,
        int32_t max_output_size) {
  const auto topk = *topk_ptr;
  if (topk <= 0) return;  // nothing to do
  if (topk > max_output_size) return;  // prevent out-of-bounds writes
  if (length <= topk) {
    return naive_argtopk_int32(indices, length, topk);
  } else {
    return fast_argtopk_int32_impl<Largest, DynamicSmemBytes>(input, indices, length, topk);
  }
}

template <size_t DynamicSmemBytes>
__device__ void fast_argmin_bounded_impl(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ index,
    int length,
    int topk_total) {
  constexpr auto BLOCK_SIZE = 1024;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = static_cast<int>(DynamicSmemBytes / (2 * sizeof(int)));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_bin_counter[RADIX];
  alignas(128) __shared__ int s_exact_base;
  alignas(128) __shared__ int s_exact_counter;

  auto& s_histogram = s_histogram_buf[0];
  extern __shared__ int s_bounded_input_storage[];
  auto* s_input_idx = reinterpret_cast<int (*)[SMEM_INPUT_SIZE]>(s_bounded_input_storage);

  const int tx = threadIdx.x;
  int topk = topk_total;
  if (topk <= 0) return;
  int threshold_byte2 = 0;
  int threshold_byte1 = 0;

  const auto to_bounded_u24 = [] __device__(int32_t x) -> uint32_t {
    return (0 <= x && x <= kGetFreeLocMaxPriority) ? static_cast<uint32_t>(x) : 0x00FFFFFFu;
  };

  const auto run_prefix_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (C10_LIKELY(tx < RADIX)) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx >= j) {
          value += s_histogram_buf[k][tx - j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  // Stage 1: histogram byte-2 across the bounded [0, 10_000_000] range.
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();
  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto uval = to_bounded_u24(input[idx]);
    const auto bin = static_cast<int>((uval >> 16) & 0xFF);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  run_prefix_cumsum();
  if (tx < RADIX && s_histogram[tx] >= topk && (tx == 0 || s_histogram[tx - 1] < topk)) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = (tx == 0) ? 0 : s_histogram[tx - 1];
  }
  __syncthreads();

  threshold_byte2 = s_threshold_bin_id;
  {
    if (tx < RADIX) s_bin_counter[tx] = 0;
    if (tx < RADIX + 1) s_histogram_buf[1][tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto uval = to_bounded_u24(input[idx]);
      const auto bin = static_cast<int>((uval >> 16) & 0xFF);
      if (bin < threshold_byte2) {
        const auto offset = (bin == 0) ? 0 : s_histogram[bin - 1];
        const auto pos = offset + ::atomicAdd(&s_bin_counter[bin], 1);
        index[pos] = idx;
      } else if (bin == threshold_byte2) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        const auto sub_bin = static_cast<int>((uval >> 8) & 0xFF);
        ::atomicAdd(&s_histogram_buf[1][sub_bin], 1);
        if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
        }
      }
    }

    __syncthreads();
    topk -= (threshold_byte2 == 0) ? 0 : s_histogram[threshold_byte2 - 1];
    if (tx < RADIX + 1) s_histogram[tx] = s_histogram_buf[1][tx];
    __syncthreads();
  }

  // Stage 2: refine within the selected byte-2 bucket using byte-1.
  {
    const auto raw_num_input = s_num_input[0];
    const bool smem_valid = raw_num_input <= SMEM_INPUT_SIZE;
    const auto num_input = smem_valid ? raw_num_input : 0;

    run_prefix_cumsum();
    if (tx < RADIX && s_histogram[tx] >= topk && (tx == 0 || s_histogram[tx - 1] < topk)) {
      s_threshold_bin_id = tx;
      s_num_input[1] = 0;
    }
    __syncthreads();

    threshold_byte1 = s_threshold_bin_id;
    topk -= (threshold_byte1 == 0) ? 0 : s_histogram[threshold_byte1 - 1];

    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    if (smem_valid) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[0][i];
        const auto uval = to_bounded_u24(input[idx]);
        const auto bin = static_cast<int>((uval >> 8) & 0xFF);
        if (bin < threshold_byte1) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_byte1) {
          const auto pos = ::atomicAdd(&s_num_input[1], 1);
          const auto sub_bin = static_cast<int>(uval & 0xFF);
          ::atomicAdd(&s_histogram[sub_bin], 1);
          if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
            s_input_idx[1][pos] = idx;
          }
        }
      }
    } else {
      for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto uval = to_bounded_u24(input[idx]);
        if (static_cast<int>((uval >> 16) & 0xFF) != threshold_byte2) continue;
        const auto bin = static_cast<int>((uval >> 8) & 0xFF);
        if (bin < threshold_byte1) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_byte1) {
          const auto pos = ::atomicAdd(&s_num_input[1], 1);
          const auto sub_bin = static_cast<int>(uval & 0xFF);
          ::atomicAdd(&s_histogram[sub_bin], 1);
          if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
            s_input_idx[1][pos] = idx;
          }
        }
      }
    }
    __syncthreads();
  }

  // Stage 3: final refine using byte-0 and pick the remaining ties.
  {
    const auto raw_num_input = s_num_input[1];
    const bool smem_valid = raw_num_input <= SMEM_INPUT_SIZE;
    const auto num_input = smem_valid ? raw_num_input : 0;

    run_prefix_cumsum();
    if (tx < RADIX && s_histogram[tx] >= topk && (tx == 0 || s_histogram[tx - 1] < topk)) {
      s_threshold_bin_id = tx;
    }
    __syncthreads();

    const auto threshold_byte0 = s_threshold_bin_id;
    const auto smaller_count = (threshold_byte0 == 0) ? 0 : s_histogram[threshold_byte0 - 1];
    topk -= smaller_count;
    if (tx == 0) {
      s_exact_base = s_counter + smaller_count;
      s_exact_counter = 0;
    }
    __syncthreads();

    if (smem_valid) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[1][i];
        const auto uval = to_bounded_u24(input[idx]);
        const auto bin = static_cast<int>(uval & 0xFF);
        if (bin < threshold_byte0) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_byte0) {
          const auto pos = ::atomicAdd(&s_exact_counter, 1);
          if (pos < topk) {
            index[s_exact_base + pos] = idx;
          }
        }
      }
    } else {
      for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
        const auto uval = to_bounded_u24(input[idx]);
        if (static_cast<int>((uval >> 16) & 0xFF) != threshold_byte2) continue;
        if (static_cast<int>((uval >> 8) & 0xFF) != threshold_byte1) continue;
        const auto bin = static_cast<int>(uval & 0xFF);
        if (bin < threshold_byte0) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_byte0) {
          const auto pos = ::atomicAdd(&s_exact_counter, 1);
          if (pos < topk) {
            index[s_exact_base + pos] = idx;
          }
        }
      }
    }
    __syncthreads();
  }
}

template <size_t DynamicSmemBytes>
__global__ __launch_bounds__(kThreadsPerBlock)
    void argmin_bounded_kernel(
        const int32_t* __restrict__ input,
        int32_t* __restrict__ indices,
        const int32_t* __restrict__ topk_ptr,
        int32_t length,
        int32_t max_output_size) {
  const auto topk = *topk_ptr;
  if (topk <= 0) return;
  if (topk > max_output_size) return;
  if (length <= topk) {
    return naive_argtopk_int32(indices, length, topk);
  } else {
    return fast_argmin_bounded_impl<DynamicSmemBytes>(input, indices, length, topk);
  }
}

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result =
      [] { return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem); }();
  TORCH_CHECK(result == cudaSuccess, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

auto get_argtopk_dynamic_smem(int device_index) -> size_t {
  int max_dynamic_smem = 0;
  const auto result = ::cudaDeviceGetAttribute(
      &max_dynamic_smem, ::cudaDevAttrMaxSharedMemoryPerBlockOptin, device_index);
  TORCH_CHECK(result == cudaSuccess, "query shared memory failed:", ::cudaGetErrorString(result));
  return max_dynamic_smem >= static_cast<int>(kArgTopKLargeSmem) ? kArgTopKLargeSmem : kSmem;
}

}  // namespace

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

void fast_topk_interface(at::Tensor score, at::Tensor indices, at::Tensor lengths) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  CHECK_CUDA(lengths);
  const auto params = get_params(score, lengths, indices);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_kernel, kSmem>();
  topk_kernel<<<grid, block, kSmem, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    at::Tensor score,
    at::Tensor lengths,
    at::Tensor dst_page_table,
    at::Tensor src_page_table,
    at::Tensor cu_seqlens_q,
    std::optional<at::Tensor> topk_logits_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  const auto params = get_params(score, lengths);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs
  float* topk_logits_ptr = nullptr;
  if (topk_logits_opt.has_value() && topk_logits_opt->defined()) {
    const auto& topk_logits = topk_logits_opt.value();
    CHECK_CUDA(topk_logits);
    TORCH_CHECK(topk_logits.dim() == 1 && topk_logits.is_contiguous());
    TORCH_CHECK(topk_logits.size(0) == B);
    TORCH_CHECK(topk_logits.scalar_type() == at::kFloat);
    topk_logits_ptr = topk_logits.data_ptr<float>();
  }

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  const auto is_decode = (prefill_bs == B);
  if (is_decode) {
    setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
    topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        topk_logits_ptr);
  } else {
    setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
    topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        topk_logits_ptr,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_argtopk_m2048_interface(at::Tensor score, at::Tensor indices, at::Tensor topk, bool largest) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  TORCH_CHECK(score.dim() == 1 && score.is_contiguous(), "score must be 1-D contiguous");
  TORCH_CHECK(indices.dim() == 1 && indices.is_contiguous(), "indices must be 1-D contiguous");
  TORCH_CHECK(topk.numel() == 1, "topk must be a scalar tensor");
  TORCH_CHECK(score.scalar_type() == at::kInt, "score must be int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(topk.scalar_type() == at::kInt, "topk must be int32");

  const auto N = static_cast<int32_t>(score.size(0));
  const auto output_capacity = static_cast<int32_t>(indices.size(0));
  TORCH_CHECK(N >= 0, "score.size(0) must be non-negative");
  TORCH_CHECK(output_capacity > 0, "indices must not be empty");

  // If topk is on CPU, perform full validation before launch
  if (topk.is_cpu()) {
    const auto k = topk.item<int32_t>();
    TORCH_CHECK(k >= 0, "topk must be non-negative, got ", k);
    TORCH_CHECK(k <= N, "topk (", k, ") must not exceed score.size(0) (", N, ")");
    TORCH_CHECK(output_capacity >= k,
        "indices.size(0) (", output_capacity, ") must be >= topk (", k, ")");
  }

  // Ensure topk is on the same device as score for the kernel
  at::Tensor topk_device = topk.is_cuda() ? topk : topk.to(score.device());

  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto argtopk_smem = get_argtopk_dynamic_smem(score.get_device());
  if (largest) {
    if (argtopk_smem == kArgTopKLargeSmem) {
      // Narrow integer ranges keep many candidates alive after the first radix
      // byte. On Hopper we can spend more shared memory to avoid the expensive
      // global-memory fallback path in later rounds.
      setup_kernel_smem_once<argtopk_int32_kernel<true, kArgTopKLargeSmem>, kArgTopKLargeSmem>();
      argtopk_int32_kernel<true, kArgTopKLargeSmem><<<1, kThreadsPerBlock, kArgTopKLargeSmem, stream>>>(
          score.data_ptr<int32_t>(),
          indices.data_ptr<int32_t>(),
          topk_device.data_ptr<int32_t>(),
          N,
          output_capacity);
    } else {
      setup_kernel_smem_once<argtopk_int32_kernel<true, kSmem>, kSmem>();
      argtopk_int32_kernel<true, kSmem><<<1, kThreadsPerBlock, kSmem, stream>>>(
          score.data_ptr<int32_t>(),
          indices.data_ptr<int32_t>(),
          topk_device.data_ptr<int32_t>(),
          N,
          output_capacity);
    }
  } else {
    if (argtopk_smem == kArgTopKLargeSmem) {
      setup_kernel_smem_once<argtopk_int32_kernel<false, kArgTopKLargeSmem>, kArgTopKLargeSmem>();
      argtopk_int32_kernel<false, kArgTopKLargeSmem><<<1, kThreadsPerBlock, kArgTopKLargeSmem, stream>>>(
          score.data_ptr<int32_t>(),
          indices.data_ptr<int32_t>(),
          topk_device.data_ptr<int32_t>(),
          N,
          output_capacity);
    } else {
      setup_kernel_smem_once<argtopk_int32_kernel<false, kSmem>, kSmem>();
      argtopk_int32_kernel<false, kSmem><<<1, kThreadsPerBlock, kSmem, stream>>>(
          score.data_ptr<int32_t>(),
          indices.data_ptr<int32_t>(),
          topk_device.data_ptr<int32_t>(),
          N,
          output_capacity);
    }
  }
  const auto result2 = cudaGetLastError();
  TORCH_CHECK(result2 == cudaSuccess, "argtopk kernel failed:", ::cudaGetErrorString(result2));
}

void fast_argtopk_interface(at::Tensor score, at::Tensor indices, at::Tensor topk, bool largest) {
  fast_argtopk_m2048_interface(score, indices, topk, largest);
}

void fast_argmin_bounded_interface(at::Tensor score, at::Tensor indices, at::Tensor topk) {
  CHECK_CUDA(score);
  CHECK_CUDA(indices);
  TORCH_CHECK(score.dim() == 1 && score.is_contiguous(), "score must be 1-D contiguous");
  TORCH_CHECK(indices.dim() == 1 && indices.is_contiguous(), "indices must be 1-D contiguous");
  TORCH_CHECK(topk.numel() == 1, "topk must be a scalar tensor");
  TORCH_CHECK(score.scalar_type() == at::kInt, "score must be int32");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(topk.scalar_type() == at::kInt, "topk must be int32");

  const auto N = static_cast<int32_t>(score.size(0));
  const auto output_capacity = static_cast<int32_t>(indices.size(0));
  TORCH_CHECK(N >= 0, "score.size(0) must be non-negative");
  TORCH_CHECK(output_capacity > 0, "indices must not be empty");

  if (topk.is_cpu()) {
    const auto k = topk.item<int32_t>();
    TORCH_CHECK(k >= 0, "topk must be non-negative, got ", k);
    TORCH_CHECK(k <= N, "topk (", k, ") must not exceed score.size(0) (", N, ")");
    TORCH_CHECK(output_capacity >= k,
        "indices.size(0) (", output_capacity, ") must be >= topk (", k, ")");
  }

  at::Tensor topk_device = topk.is_cuda() ? topk : topk.to(score.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto argtopk_smem = get_argtopk_dynamic_smem(score.get_device());
  if (argtopk_smem == kArgTopKLargeSmem) {
    setup_kernel_smem_once<argmin_bounded_kernel<kArgTopKLargeSmem>, kArgTopKLargeSmem>();
    argmin_bounded_kernel<kArgTopKLargeSmem><<<1, kThreadsPerBlock, kArgTopKLargeSmem, stream>>>(
        score.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(),
        topk_device.data_ptr<int32_t>(),
        N,
        output_capacity);
  } else {
    setup_kernel_smem_once<argmin_bounded_kernel<kSmem>, kSmem>();
    argmin_bounded_kernel<kSmem><<<1, kThreadsPerBlock, kSmem, stream>>>(
        score.data_ptr<int32_t>(),
        indices.data_ptr<int32_t>(),
        topk_device.data_ptr<int32_t>(),
        N,
        output_capacity);
  }
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "argmin_bounded kernel failed:", ::cudaGetErrorString(result));
}
