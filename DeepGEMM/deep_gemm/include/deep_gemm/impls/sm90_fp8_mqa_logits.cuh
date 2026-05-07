#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

// ReSharper disable once CppNotAllPathsReturnValue
template <uint32_t kHeadDim>
static constexpr int to_swizzle_cute_type() {
    DG_STATIC_ASSERT(kHeadDim == 32 or kHeadDim == 64 or kHeadDim == 128, "Invalid swizzling");
    if constexpr (kHeadDim == 32)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B32);
    if constexpr (kHeadDim == 64)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B64);
    if constexpr (kHeadDim == 128)
        return static_cast<int>(cute::SM90::GMMA::LayoutType::B128);
}

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_mqa_logits(const uint32_t seq_len, const uint32_t seq_len_kv, const uint64_t stride_kv,
                         uint32_t* cu_seq_len_k_start,
                         uint32_t* cu_seq_len_k_end,
                         float* logits,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                         const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    const auto& num_q_blocks = ceil_div(seq_len, BLOCK_Q);

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Data on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_q_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); });

    // Initialize barriers
    const bool& is_tma_load_warp = kNumMathThreads <= threadIdx.x and threadIdx.x < kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // Block scheduler
    uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
    const auto& get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + gridDim.x, q_iter_idx + 1};
    };
    const auto& load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto& q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            start = min(start, min(__ldg(cu_seq_len_k_start + q_idx), seq_len_kv));
            end = max(end, min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv));
        }
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, ceil_div(end - start, BLOCK_KV)};          // Task info
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto& get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Only the first warp remains
        if (not is_tma_load_warp)
            return;

        // Prefetch
        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma_copy(&tensor_map_weights, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma_copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma_copy(&tensor_map_kv_scales, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx = get_lane_idx();
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                float scale_kv_0 = ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                float scale_kv_1 = ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);

                // Issue WGMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = make_smem_desc(smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = make_smem_desc(smem_q[q_stage_idx] + k * WGMMA::K,
                                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Reduce over the head dim and store
                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    auto shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto& transform = [&](const uint32_t& j) {
                        return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                    };

                    // Intra-thread reduction
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k ++)
                            sum[k] += transform(j * 4 + k);
                    }
                    float v_0 = (sum[0] + sum[1]) * scale_kv_0;
                    float v_1 = (sum[2] + sum[3]) * scale_kv_1;

                    // Inter-thread reduction
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0 += __shfl_xor_sync(0xffffffffu, v_0, offset);
                        v_1 += __shfl_xor_sync(0xffffffffu, v_1, offset);
                    }

                    // Store into the global memory
                    // NOTES: we have redundant writes here, consider more carefully
                    const uint32_t& q_idx = block_q_idx * BLOCK_Q + i;
                    logits[q_idx * stride_kv + kv_offset + v_0_offset] = v_0;
                    logits[q_idx * stride_kv + kv_offset + v_1_offset] = v_1;
                }
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
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

template <uint32_t kNumHeads, uint32_t kHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTopkStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads, uint32_t kNumTopkThreads,
          uint32_t SMEM_SIZE_FOR_TOPK_INPUT>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads + kNumTopkThreads, 1)
void sm90_fp8_mqa_logits_fuse_topk(const uint32_t seq_len, const uint32_t seq_len_kv, const uint64_t stride_kv,
                                   uint32_t* cu_seq_len_k_start,
                                   uint32_t* cu_seq_len_k_end,
                                   float* logits,
                                   int* topk_index,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                                   const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    const auto& num_q_blocks = ceil_div(seq_len, BLOCK_Q);

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0 and kNumTopkThreads % 128 == 0, "Invalid threads");
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t SMEM_LOGITS_SIZE_PER_STAGE = BLOCK_Q * BLOCK_KV * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    constexpr auto SMEM_TOPK_INPUT_SIZE = SMEM_SIZE_FOR_TOPK_INPUT / (2 * sizeof(int));
    // In the shape of [2, SMEM_TOPK_INPUT_SIZE]
    auto s_input_idx = reinterpret_cast<int (*)[SMEM_SIZE_FOR_TOPK_INPUT]>(smem_buffer);

    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Data on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto smem_logits = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * kNumKVStages +
            SMEM_LOGITS_SIZE_PER_STAGE * i
        );
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_logits[kNumTopkStages]);
    auto full_q_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = PatternVisitor([&](const uint32_t& i) { 
        return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); 
    });
    auto full_logits_barriers = PatternVisitor([&](const uint32_t& i) { 
        return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); 
    });
    auto empty_logits_barriers = PatternVisitor([&](const uint32_t& i) { 
        return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumTopkStages + i); 
    });

    // Initialize barriers
    const bool& is_tma_load_warp = kNumTopkThreads + kNumMathThreads <= threadIdx.x 
                                    and threadIdx.x < kNumTopkThreads + kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTopkStages; ++ i) {
            full_logits_barriers[i]->init(kNumMathThreads);
            empty_logits_barriers[i]->init(kNumTopkThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTopkRegisters = 32;
    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // Block scheduler
    uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
    const auto& get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + gridDim.x, q_iter_idx + 1};
    };
    const auto& load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto& q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            start = min(start, min(__ldg(cu_seq_len_k_start + q_idx), seq_len_kv));
            end = max(end, min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv));
        }
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, ceil_div(end - start, BLOCK_KV)};          // Task info
    };
    const auto& topk_schedule = [&](const uint32_t& i) -> cute::tuple<uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        const auto& q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
        start = min(start, min(__ldg(cu_seq_len_k_start + q_idx), seq_len_kv));
        end = max(end, min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv));

        start = start / 4 * 4;
        return {start, ceil_div(end - start, BLOCK_KV)};
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto& get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };
    const auto& get_topk_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumTopkStages,         
            ((num_total_kv_blocks + kv_block_idx) / kNumTopkStages) & 1    
        };
    };

    if (threadIdx.x >= kNumMathThreads + kNumTopkThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Only the first warp remains
        if (not is_tma_load_warp)
            return;

        // Prefetch
        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma_copy(&tensor_map_weights, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma_copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma_copy(&tensor_map_kv_scales, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else if (threadIdx.x >= kNumTopkThreads) {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx = get_lane_idx();
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // M: KV, N: QH, K: head_dim
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                float scale_kv_0 = ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                float scale_kv_1 = ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);

                // Issue WGMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = make_smem_desc(smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = make_smem_desc(smem_q[q_stage_idx] + k * WGMMA::K,
                                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Reduce over the head dim and store
                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

                float v_0[BLOCK_Q], v_1[BLOCK_Q];
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    auto shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto& transform = [&](const uint32_t& j) {
                        return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                    };

                    // Intra-thread reduction
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k ++)
                            sum[k] += transform(j * 4 + k);
                    }
                    v_0[i] = (sum[0] + sum[1]) * scale_kv_0;
                    v_1[i] = (sum[2] + sum[3]) * scale_kv_1;

                    // Inter-thread reduction
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0[i] += __shfl_xor_sync(0xffffffffu, v_0[i], offset);
                        v_1[i] += __shfl_xor_sync(0xffffffffu, v_1[i], offset);
                    }
                }

                CUTE_TIE_DECL(get_topk_pipeline(kv_block_idx), topk_stage_idx, topk_phase);
                empty_logits_barriers[topk_stage_idx]->wait(topk_phase ^ 1);    // !!! IMPORTANT !!!
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Produce
                    smem_logits[topk_stage_idx][i * BLOCK_KV + warp_offset + v_0_offset] = v_0[i];
                    smem_logits[topk_stage_idx][i * BLOCK_KV + warp_offset + v_1_offset] = v_1[i];
                }

                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Store into the global memory
                    // NOTES: we have redundant writes here, consider more carefully
                    const uint32_t& q_idx = block_q_idx * BLOCK_Q + i;
                    logits[q_idx * stride_kv + kv_offset + v_0_offset] = v_0[i];
                    logits[q_idx * stride_kv + kv_offset + v_1_offset] = v_1[i];
                }
                full_logits_barriers[topk_stage_idx]->arrive();
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    } else {
        constexpr auto RADIX = 256, TOPK = 2048;
        DG_STATIC_ASSERT(BLOCK_KV <= kNumTopkThreads, "kNumTopkThreads too small");
        DG_STATIC_ASSERT((BLOCK_Q * BLOCK_KV) % kNumTopkThreads == 0, "unaligned");
        cutlass::arch::NamedBarrier topk_bar(kNumTopkThreads, 8);   // NOTE: 8 is max allowed id

        cutlass::arch::warpgroup_reg_dealloc<kNumTopkRegisters>();

        alignas(128) __shared__ int s_histogram_buf[2][BLOCK_Q][RADIX + 128];
        alignas(128) __shared__ int s_counter;
        alignas(128) __shared__ int s_threshold_bin_id;
        alignas(128) __shared__ int s_num_input[2];
        __shared__ int s_indices[TOPK];

        auto& s_histogram = s_histogram_buf[0];
        const int tx = threadIdx.x;
        // tx is also id in the topk warps

        const auto clear_histogram = [&](int i) {
            #pragma unroll
            for (uint32_t s_idx = tx; s_idx < RADIX + 1; s_idx += kNumTopkThreads) {
                s_histogram[i][s_idx] = 0;
            }
        };
        #pragma unroll
        for (int i = 0; i < BLOCK_Q; i++) {
            clear_histogram(i);
        }
        topk_bar.sync();

        const auto run_cumsum = [&](int q) {
            DG_STATIC_ASSERT(1 << 8 == RADIX, "Wrong RADIX");
            #pragma unroll 8
            for (int i = 0; i < 8; ++i) {
                if (tx < RADIX) {
                    const auto j = 1 << i;
                    const auto k = i & 1;
                    auto value = s_histogram_buf[k][q][tx];
                    if (tx < RADIX - j) {
                        value += s_histogram_buf[k][q][tx + j];
                    }
                    s_histogram_buf[k ^ 1][q][tx] = value;
                }
                topk_bar.sync();
            }
        };

        // Consume
        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_topk_pipeline(kv_block_idx), topk_stage_idx, topk_phase);
                full_logits_barriers[topk_stage_idx]->wait(topk_phase);
                // Build coarse buckets
                for (uint32_t idx = tx; idx < BLOCK_Q * BLOCK_KV; idx += kNumTopkThreads) {
                    const uint32_t local_kv_idx = tx % BLOCK_KV;
                    const uint32_t local_q_idx = tx / BLOCK_KV;
                    const auto bin = convert_to_uint8(smem_logits[topk_stage_idx][local_q_idx * BLOCK_KV + local_kv_idx]);
                    const uint32_t q_idx = block_q_idx * BLOCK_Q + local_q_idx;
                    const uint32_t global_kv_idx = kv_start + kv_block_idx * BLOCK_KV + local_kv_idx;
                    if (cu_seq_len_k_start[q_idx] <= global_kv_idx and global_kv_idx < cu_seq_len_k_end[q_idx]) {
                        ::atomicAdd(&s_histogram[local_q_idx][bin], 1);
                    }
                }
                
                empty_logits_barriers[topk_stage_idx]->arrive();
            }
            num_total_kv_blocks += num_kv_blocks;
            
            // Check coarse bucket for current qs
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                const uint32_t& q_idx = block_q_idx * BLOCK_Q + i;
                if (q_idx >= seq_len) {
                    break;
                }
                CUTE_TIE_DECL(topk_schedule(i), kv_start, num_kv_blocks);

                topk_bar.sync();
                run_cumsum(i);

                int topk = TOPK;
                if (tx < RADIX && s_histogram[i][tx] > topk && s_histogram[i][tx + 1] <= topk) {
                    s_threshold_bin_id = tx;
                    s_num_input[0] = 0;
                    s_counter = 0;
                }
                topk_bar.sync();

                const auto threshold_bin = s_threshold_bin_id;
                topk -= s_histogram[i][threshold_bin + 1];
                if (topk == 0) {
                    for (int idx = tx; idx < num_kv_blocks * BLOCK_KV; idx += kNumTopkThreads) {
                        const auto bin = static_cast<int>(convert_to_uint8(
                            logits[q_idx * stride_kv + kv_start + idx]
                        ));
                        if (bin > threshold_bin) {
                            const auto pos = ::atomicAdd(&s_counter, 1);
                            s_indices[pos] = idx;
                        }
                    }
                    topk_bar.sync();
                    return;
                } else {
                    clear_histogram(i);
                    topk_bar.sync();

                    for (int idx = tx; idx < num_kv_blocks * BLOCK_KV; idx += kNumTopkThreads) {
                        const auto raw_input = logits[q_idx * stride_kv + kv_start + idx];
                        const auto bin = static_cast<int>(convert_to_uint8(raw_input));
                        if (bin > threshold_bin) {
                            const auto pos = ::atomicAdd(&s_counter, 1);
                            s_indices[pos] = idx;
                        } else if (bin == threshold_bin) {
                            const auto pos = ::atomicAdd(&s_num_input[0], 1);
                            /// NOTE: (dark) fuse the histogram computation here
                            if (pos < SMEM_TOPK_INPUT_SIZE) {
                                s_input_idx[0][pos] = idx;
                                const auto bin = convert_to_uint32(raw_input);
                                const auto sub_bin = (bin >> 24) & 0xFF;
                                ::atomicAdd(&s_histogram[i][sub_bin], 1);
                            }
                        }
                    }
                    topk_bar.sync();
                }
                #pragma unroll 4
                for (int round = 0; round < 4; ++round) {
                    __shared__ int s_last_remain;
                    const auto r_idx = round % 2;

                    // clip here to prevent overflow
                    const auto _raw_num_input = s_num_input[r_idx];
                    const auto num_input = (_raw_num_input < int(SMEM_TOPK_INPUT_SIZE)) ? _raw_num_input : int(SMEM_TOPK_INPUT_SIZE);

                    run_cumsum(i);
                    if (tx < RADIX && s_histogram[i][tx] > topk && s_histogram[i][tx + 1] <= topk) {
                        s_threshold_bin_id = tx;
                        s_num_input[r_idx ^ 1] = 0;
                        s_last_remain = topk - s_histogram[i][tx + 1];
                    }
                    topk_bar.sync();

                    const auto threshold_bin = s_threshold_bin_id;
                    topk -= s_histogram[i][threshold_bin + 1];

                    if (topk == 0) {
                        for (int j = tx; j < num_input; j += kNumTopkThreads) {
                            const auto idx = s_input_idx[r_idx][j];
                            const auto offset = 24 - round * 8;
                            const auto bin = (convert_to_uint32(logits[q_idx * stride_kv + kv_start + idx]) >> offset) & 0xFF;
                            if (bin > threshold_bin) {
                                const auto pos = ::atomicAdd(&s_counter, 1);
                                s_indices[pos] = idx;
                            }
                        }
                        topk_bar.sync();
                        break;
                    } else {
                        clear_histogram(i);
                        topk_bar.sync();

                        for (int j = tx; j < num_input; j += kNumTopkThreads) {
                            const auto idx = s_input_idx[r_idx][j];
                            const auto raw_input = logits[q_idx * stride_kv + kv_start + idx];
                            const auto offset = 24 - round * 8;
                            const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
                            if (bin > threshold_bin) {
                                const auto pos = ::atomicAdd(&s_counter, 1);
                                s_indices[pos] = idx;
                            } else if (bin == threshold_bin) {
                                if (round == 3) {
                                    const auto pos = ::atomicAdd(&s_last_remain, -1);
                                    if (pos > 0) {
                                        s_indices[TOPK - pos] = idx;
                                    }
                                } else {
                                    const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
                                    if (pos < SMEM_TOPK_INPUT_SIZE) {
                                        /// NOTE: (dark) fuse the histogram computation here
                                        s_input_idx[r_idx ^ 1][pos] = idx;
                                        const auto bin = convert_to_uint32(raw_input);
                                        const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
                                        ::atomicAdd(&s_histogram[i][sub_bin], 1);
                                    }
                                }
                            }
                        }
                        topk_bar.sync();
                    }
                }
                // Output topk index to global memory
                for (int idx = tx; idx < TOPK; idx += kNumTopkThreads) {
                    topk_index[q_idx * TOPK + idx] = s_indices[idx];
                }
            }

            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
}

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t kMLAHeadDim,
          uint32_t BLOCK_Q, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumTopkStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads, uint32_t kNumTopkThreads,
          uint32_t SMEM_SIZE_FOR_TOPK_INPUT>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads + kNumTopkThreads, 1)
void sm90_fp8_mqa_logits_fuse_prefetch(const uint32_t seq_len, const uint32_t seq_len_kv, const uint64_t stride_kv,
                                       const uint64_t page_table_1_stride,
                                       uint32_t* cu_seq_len_k_start,
                                       uint32_t* cu_seq_len_k_end,
                                       float* logits,
                                       const int* page_table_1, const int* extend_seq_lens, const int* extend_seq_to_req,
                                       __nv_bfloat16* device_pool_buf, __nv_bfloat16* host_pool_buf,
                                       int* device_pool_loc_alloc_buf,
                                       const int* device_pool_loc_small_priority,
                                       int64_t* device_token_to_host, int* host_token_to_device,
                                       uint32_t* recall_counter, float* extend_logits_offsets,
                                       uint32_t device_pool_size_m1,
                                       const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                       const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                       const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                                       const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // TODO: consider TMA multicast
    // For one block, we process `[q_start:q_end, h, d] @ [kv_start:kv_end, d] -> [q_start:q_end, kv_start:kv_end]`
    // Q should be load only at once for a block
    const auto& num_q_blocks = ceil_div(seq_len, BLOCK_Q);

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_Q * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Prefetch TMA descriptors
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0 and kNumTopkThreads % 128 == 0, "Invalid threads");
    if (threadIdx.x / 32 == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    // NOTES: weight may be unaligned
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(float);
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t SMEM_LOGITS_SIZE_PER_STAGE = BLOCK_Q * BLOCK_KV * sizeof(float);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

    constexpr auto SMEM_TOPK_INPUT_SIZE = SMEM_SIZE_FOR_TOPK_INPUT / (2 * sizeof(int));
    // In the shape of [2, SMEM_TOPK_INPUT_SIZE]
    auto s_input_idx = reinterpret_cast<int (*)[SMEM_SIZE_FOR_TOPK_INPUT]>(smem_buffer);

    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Data on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT +
            SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT + (
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * i));
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto smem_logits = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SIZE_FOR_TOPK_INPUT +
            SMEM_Q_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
            SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_KV_SCALE_SIZE_PER_STAGE * kNumKVStages +
            SMEM_LOGITS_SIZE_PER_STAGE * i
        );
    });

    // TMA barriers
    auto barrier_ptr = reinterpret_cast<Barrier*>(smem_logits[kNumTopkStages]);
    auto full_q_barriers   = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
    auto empty_q_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
    auto full_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
    auto empty_kv_barriers = PatternVisitor([&](const uint32_t& i) { 
        return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i); 
    });
    auto full_logits_barriers = PatternVisitor([&](const uint32_t& i) { 
        return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i); 
    });
    auto empty_logits_barriers = PatternVisitor([&](const uint32_t& i) { 
        return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + kNumTopkStages + i); 
    });

    // Initialize barriers
    const bool& is_tma_load_warp = kNumTopkThreads + kNumMathThreads <= threadIdx.x 
                                    and threadIdx.x < kNumTopkThreads + kNumMathThreads + 32;
    if (is_tma_load_warp and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            full_q_barriers[i]->init(1);
            empty_q_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            full_kv_barriers[i]->init(1);
            empty_kv_barriers[i]->init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTopkStages; ++ i) {
            full_logits_barriers[i]->init(kNumMathThreads);
            empty_logits_barriers[i]->init(kNumTopkThreads);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTopkRegisters = 32;
    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 112;

    // Block scheduler
    uint32_t block_q_idx = blockIdx.x, q_iter_idx = 0;
    const auto& get_next_block_q_idx = [&]() -> cute::tuple<uint32_t, uint32_t> {
        return {block_q_idx + gridDim.x, q_iter_idx + 1};
    };
    const auto& load_schedule = [&](const uint32_t& q_iter_offset = 0) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        uint32_t start = cute::numeric_limits<uint32_t>::max();
        uint32_t end = cute::numeric_limits<uint32_t>::min();

        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
            const auto& q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
            start = min(start, min(__ldg(cu_seq_len_k_start + q_idx), seq_len_kv));
            end = max(end, min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv));
        }
        start = start / 4 * 4;
        return {(q_iter_idx + q_iter_offset) % kNumQStages,       // Q pipeline stage
                ((q_iter_idx + q_iter_offset) / kNumQStages) & 1, // Q pipeline phase
                start, ceil_div(end - start, BLOCK_KV)};          // Task info
    };
    const auto& topk_schedule = [&](const uint32_t& q_idx) -> cute::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        const uint32_t& start = min(__ldg(cu_seq_len_k_start + q_idx), seq_len_kv);
        const uint32_t& end = min(__ldg(cu_seq_len_k_end + q_idx), seq_len_kv);
        const uint32_t& len = end - start;

        const uint32_t& req_id = __ldg(extend_seq_to_req + q_idx);
        const uint32_t& cur_extend_len = __ldg(extend_seq_lens + req_id);

        return {start, len, req_id, cur_extend_len};
    };

    // KV pipeline
    uint32_t num_total_kv_blocks = 0;
    const auto& get_kv_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumKVStages,         // KV pipeline stage
            ((num_total_kv_blocks + kv_block_idx) / kNumKVStages) & 1    // KV pipeline phase
        };
    };
    const auto& get_topk_pipeline = [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {
            (num_total_kv_blocks + kv_block_idx) % kNumTopkStages,         
            ((num_total_kv_blocks + kv_block_idx) / kNumTopkStages) & 1    
        };
    };

    if (threadIdx.x >= kNumMathThreads + kNumTopkThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // Only the first warp remains
        if (not is_tma_load_warp)
            return;

        // Prefetch
        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const auto& block_idx) {
            tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, block_idx * BLOCK_Q * kNumHeads);
            tma_copy(&tensor_map_weights, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_weights[stage_idx], 0, block_idx * BLOCK_Q);
            full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
        };
        if (cute::elect_one_sync() and block_q_idx < num_q_blocks)
            issue_tma_q(0, block_q_idx);

        // Only the first lane persistently schedules over blocks
        if (cute::elect_one_sync()) {
            while (block_q_idx < num_q_blocks) {
                CUTE_TIE_DECL(load_schedule(1), q_stage_idx, q_phase, kv_start, num_kv_blocks);

                // Wait Q consumer release
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);

                // Issue TMA Q
                if (const auto& next_block_q_idx = cute::get<0>(get_next_block_q_idx()); next_block_q_idx < num_q_blocks)
                    issue_tma_q(q_stage_idx, next_block_q_idx);

                // Issue TMA KV
                #pragma unroll
                for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                    // Wait consumer release
                    CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                    empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

                    // Issue TMA KV
                    tma_copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
                    tma_copy(&tensor_map_kv_scales, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                             smem_kv_scales[kv_stage_idx], kv_start + kv_block_idx * BLOCK_KV, 0);
                    full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
                }
                num_total_kv_blocks += num_kv_blocks;

                // Jump to the next block
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
            }
        }
    } else if (threadIdx.x >= kNumTopkThreads) {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto& thread_idx = threadIdx.x % kNumMathThreads;
        const auto& warp_idx = __shfl_sync(0xffffffff, thread_idx / 32, 0);
        const auto& warpgroup_idx = warp_idx / 4;
        const auto& lane_idx = get_lane_idx();
        float accum[WGMMA::kNumAccum], weights[BLOCK_Q][kNumHeads / 4];

        const auto& warp_offset = warp_idx * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);

            // Wait TMA Q arrival
            full_q_barriers[q_stage_idx]->wait(q_phase);

            // Read weights
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                #pragma unroll
                for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                    weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
            }

            // Compute over KV blocks
            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                // Compute `[BLOCK_Q * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [BLOCK_Q, BLOCK_KV]`
                // M: KV, N: QH, K: head_dim
                // Wait TMA KV arrival
                CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
                full_kv_barriers[kv_stage_idx]->wait(kv_phase);

                // Read per-KV scales
                float scale_kv_0 = ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_0_offset);
                float scale_kv_1 = ld_shared(smem_kv_scales[kv_stage_idx] + warp_offset + v_1_offset);

                // Issue WGMMA
                DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads / 2, "Invalid block size");
                DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                #pragma unroll
                for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                    auto desc_a = make_smem_desc(smem_kv[kv_stage_idx] + (warpgroup_idx * WGMMA::M) * kHeadDim + k * WGMMA::K,
                                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    auto desc_b = make_smem_desc(smem_q[q_stage_idx] + k * WGMMA::K,
                                                 to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();

                // Release KV empty
                empty_kv_barriers[kv_stage_idx]->arrive();

                // Reduce over the head dim and store
                const auto& kv_offset = kv_start + kv_block_idx * BLOCK_KV + warp_offset;
                static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
                DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
                DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == BLOCK_Q, "Invalid accumulation");
                DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

                float v_0[BLOCK_Q], v_1[BLOCK_Q];
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    auto shifted_accum = accum + i * kNumAccumPerReduce;
                    const auto& transform = [&](const uint32_t& j) {
                        return fmaxf(shifted_accum[j], 0) * weights[i][(j / 4) * 2 + (j & 1)];
                    };

                    // Intra-thread reduction
                    float sum[4] = {transform(0), transform(1), transform(2), transform(3)};
                    #pragma unroll
                    for (uint32_t j = 1; j < kNumHeads / 8; ++ j) {
                        #pragma unroll
                        for (uint32_t k = 0; k < 4; k ++)
                            sum[k] += transform(j * 4 + k);
                    }
                    v_0[i] = (sum[0] + sum[1]) * scale_kv_0;
                    v_1[i] = (sum[2] + sum[3]) * scale_kv_1;

                    // Inter-thread reduction
                    #pragma unroll
                    for (uint32_t j = 0; j < 2; ++ j) {
                        const auto& offset = static_cast<int>(1u << j);
                        v_0[i] += __shfl_xor_sync(0xffffffffu, v_0[i], offset);
                        v_1[i] += __shfl_xor_sync(0xffffffffu, v_1[i], offset);
                    }
                }

                CUTE_TIE_DECL(get_topk_pipeline(kv_block_idx), topk_stage_idx, topk_phase);
                empty_logits_barriers[topk_stage_idx]->wait(topk_phase ^ 1);    // !!! IMPORTANT !!!
                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Produce
                    smem_logits[topk_stage_idx][i * BLOCK_KV + warp_offset + v_0_offset] = v_0[i];
                    smem_logits[topk_stage_idx][i * BLOCK_KV + warp_offset + v_1_offset] = v_1[i];
                }

                #pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                    // Store into the global memory
                    // NOTES: we have redundant writes here, consider more carefully
                    const uint32_t& q_idx = block_q_idx * BLOCK_Q + i;
                    logits[q_idx * stride_kv + kv_offset + v_0_offset] = v_0[i];
                    logits[q_idx * stride_kv + kv_offset + v_1_offset] = v_1[i];
                }
                full_logits_barriers[topk_stage_idx]->arrive();
            }
            num_total_kv_blocks += num_kv_blocks;

            // Release Q empty
            empty_q_barriers[q_stage_idx]->arrive();

            // Jump to the next block
            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    } else {
        constexpr auto RADIX = 256, TOPK = 2048;
        constexpr auto kMaxPrefetchTasks = 8192;
        constexpr auto kTransferItemBytes = kMLAHeadDim * sizeof(__nv_bfloat16);
        DG_STATIC_ASSERT(BLOCK_KV <= kNumTopkThreads, "kNumTopkThreads too small");
        DG_STATIC_ASSERT((BLOCK_Q * BLOCK_KV) % kNumTopkThreads == 0, "unaligned");
        cutlass::arch::NamedBarrier topk_bar(kNumTopkThreads, 8);   // NOTE: 8 is max allowed id

        cutlass::arch::warpgroup_reg_dealloc<kNumTopkRegisters>();

        alignas(128) __shared__ int s_histogram_buf[2][BLOCK_Q][RADIX + 128];
        alignas(128) __shared__ int s_threshold_bin_id;
        __shared__ int s_prefetch_enabled;
        __shared__ int s_prefetch_host_loc_temp[kNumTopkThreads];
        __shared__ int s_prefetch_device_loc_temp[kNumTopkThreads];

        auto& s_histogram = s_histogram_buf[0];
        const int tx = threadIdx.x;
        const int lane_idx = tx % 32;
        volatile uint32_t* vol_recall_counter = recall_counter;
        // tx is also id in the topk warps

        constexpr auto MAX_EXTEND_BS = 16;
        float extend_logits_offsets_val[MAX_EXTEND_BS];
        #pragma unroll
        for (int i = 0; i < MAX_EXTEND_BS; i++) {
            extend_logits_offsets_val[i] = __ldg(extend_logits_offsets + i);
        }

        const auto clear_histogram = [&](int i) {
            #pragma unroll
            for (uint32_t s_idx = tx; s_idx < RADIX + 1; s_idx += kNumTopkThreads) {
                s_histogram[i][s_idx] = 0;
            }
        };
        #pragma unroll
        for (int i = 0; i < BLOCK_Q; i++) {
            clear_histogram(i);
        }
        topk_bar.sync();

        const auto run_cumsum = [&](int q) {
            DG_STATIC_ASSERT(1 << 8 == RADIX, "Wrong RADIX");
            #pragma unroll 8
            for (int i = 0; i < 8; ++i) {
                if (tx < RADIX) {
                    const auto j = 1 << i;
                    const auto k = i & 1;
                    auto value = s_histogram_buf[k][q][tx];
                    if (tx < RADIX - j) {
                        value += s_histogram_buf[k][q][tx + j];
                    }
                    s_histogram_buf[k ^ 1][q][tx] = value;
                }
                topk_bar.sync();
            }
        };

        const auto& recall_func = [&](const bool is_candidate, const int kv_token_host_loc,
                                      const uint32_t& max_recall_tasks) {
            bool is_miss = false;
            if (is_candidate)
                is_miss = atomicCAS(host_token_to_device + kv_token_host_loc, INT32_MAX, -1) == INT32_MAX;

            const uint32_t miss_mask = __ballot_sync(0xffffffffu, is_miss);
            if (miss_mask == 0)
                return;

            const uint32_t miss_count = __popc(miss_mask);
            const int leader = __ffs(miss_mask) - 1;
            const uint32_t warp_temp_base = (tx / 32) * 32;
            uint32_t slot_base = 0;
            uint32_t valid_count = 0;
            if (lane_idx == leader) {
                slot_base = atomicAdd(recall_counter, miss_count);
                valid_count = slot_base < max_recall_tasks ? min(max_recall_tasks - slot_base, miss_count) : 0;
            }
            slot_base = __shfl_sync(0xffffffffu, slot_base, leader);
            valid_count = __shfl_sync(0xffffffffu, valid_count, leader);

            const uint32_t lower_lane_mask = lane_idx == 0 ? 0u : ((1u << lane_idx) - 1);
            const uint32_t miss_rank = __popc(miss_mask & lower_lane_mask);
            if (is_miss) {
                if (miss_rank < valid_count) {
                    const uint32_t priority_loc = slot_base + miss_rank;
                    const int device_loc = device_pool_loc_small_priority[priority_loc];
                    device_pool_loc_alloc_buf[priority_loc] = device_loc;
                    if (device_token_to_host[device_loc] < INT32_MAX)
                        host_token_to_device[device_token_to_host[device_loc]] = INT32_MAX;
                    device_token_to_host[device_loc] = kv_token_host_loc;
                    host_token_to_device[kv_token_host_loc] = device_loc;
                    s_prefetch_host_loc_temp[warp_temp_base + miss_rank] = kv_token_host_loc;
                    s_prefetch_device_loc_temp[warp_temp_base + miss_rank] = device_loc;
                } else {
                    host_token_to_device[kv_token_host_loc] = INT32_MAX;
                }
            }

            #pragma unroll
            for (uint32_t i = 0; i < 32; ++ i) {
                if (i >= valid_count)
                    break;
                if (i + 1 < valid_count) {
                    const int next_host_loc = s_prefetch_host_loc_temp[warp_temp_base + i + 1];
                    prefetch_item_warp<kTransferItemBytes>(&host_pool_buf[(size_t)next_host_loc * kMLAHeadDim]);
                }
                const int copy_host_loc = s_prefetch_host_loc_temp[warp_temp_base + i];
                const int copy_device_loc = s_prefetch_device_loc_temp[warp_temp_base + i];
                transfer_item_warp_v2<kTransferItemBytes>(
                    &host_pool_buf[(size_t)copy_host_loc * kMLAHeadDim],
                    &device_pool_buf[(size_t)copy_device_loc * kMLAHeadDim]
                );
            }
        };

        // Consume
        while (block_q_idx < num_q_blocks) {
            CUTE_TIE_DECL(load_schedule(), q_stage_idx, q_phase, kv_start, num_kv_blocks);
            const uint32_t left_pool_size = device_pool_size_m1 - seq_len;
            const uint32_t max_recall_tasks = left_pool_size > kMaxPrefetchTasks ? kMaxPrefetchTasks : left_pool_size;

            if (tx == 0) {
                s_prefetch_enabled = (*vol_recall_counter < max_recall_tasks);
            }
            topk_bar.sync();

            #pragma unroll
            for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++ kv_block_idx) {
                CUTE_TIE_DECL(get_topk_pipeline(kv_block_idx), topk_stage_idx, topk_phase);
                full_logits_barriers[topk_stage_idx]->wait(topk_phase);
                // Build coarse buckets
                if (s_prefetch_enabled) {
                    #pragma unroll
                    for (uint32_t local_q_idx = 0; local_q_idx < BLOCK_Q; ++ local_q_idx) {
                        const uint32_t q_idx = block_q_idx * BLOCK_Q + local_q_idx;
                        if (q_idx >= seq_len) {
                            break;
                        }
                        const int req_id = __ldg(extend_seq_to_req + q_idx);
                        for (uint32_t idx = tx; idx < BLOCK_KV; idx += kNumTopkThreads) {
                            const uint32_t local_kv_idx = idx;
                            const auto bin = convert_to_uint8(
                                smem_logits[topk_stage_idx][local_q_idx * BLOCK_KV + local_kv_idx] -
                                extend_logits_offsets_val[req_id]
                            );
                            const uint32_t global_kv_idx = kv_start + kv_block_idx * BLOCK_KV + local_kv_idx;
                            if (cu_seq_len_k_start[q_idx] <= global_kv_idx and global_kv_idx < cu_seq_len_k_end[q_idx]) {
                                ::atomicAdd(&s_histogram[local_q_idx][bin], 1);
                            }
                        }
                    }
                }
                empty_logits_barriers[topk_stage_idx]->arrive();
            }
            num_total_kv_blocks += num_kv_blocks;

            if (!s_prefetch_enabled) {
                CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
                continue;
            }
            
            // Check coarse bucket for current qs
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                if (tx == 0) {
                    s_prefetch_enabled = (*vol_recall_counter < max_recall_tasks);
                }
                topk_bar.sync();
                if (!s_prefetch_enabled) {
                    break;
                }

                const uint32_t& q_idx = block_q_idx * BLOCK_Q + i;
                if (q_idx >= seq_len) {
                    break;
                }
                CUTE_TIE_DECL(topk_schedule(q_idx), kv_start, kv_len, req_id, cur_extend_len);

                topk_bar.sync();
                run_cumsum(i);

                int topk = TOPK;
                if (tx < RADIX && s_histogram[i][tx] > topk && s_histogram[i][tx + 1] <= topk) {
                    s_threshold_bin_id = tx;
                }
                topk_bar.sync();

                const auto threshold_bin = s_threshold_bin_id;
                // Prefetch for tokens larger than threshold_bin
                if (kv_len < cur_extend_len) { // e.g., token 1 for prefill with 3 tokens
                    continue;
                }
                const int total_kv_items = kv_len - cur_extend_len;
                const int aligned_kv_items = (total_kv_items + kNumTopkThreads - 1) / kNumTopkThreads * kNumTopkThreads;
                for (int kv_idx = tx; kv_idx < aligned_kv_items; kv_idx += kNumTopkThreads) {
                    bool is_candidate = false;
                    int kv_token_host_loc = 0;
                    if (kv_idx < total_kv_items and *vol_recall_counter < max_recall_tasks) {
                        const auto bin = static_cast<int>(convert_to_uint8(
                            logits[q_idx * stride_kv + kv_start + kv_idx] - extend_logits_offsets_val[req_id]
                        ));
                        if (bin > threshold_bin) {
                            const auto page_table_ptr = page_table_1 + req_id * page_table_1_stride + kv_idx;
                            asm volatile("prefetch.global.L2 [%0];" :: "l"(page_table_ptr));
                            kv_token_host_loc = __ldg(page_table_ptr);
                            is_candidate = true;
                        }
                    }

                    recall_func(is_candidate, kv_token_host_loc, max_recall_tasks);
                 }
             }

            CUTE_TIE(get_next_block_q_idx(), block_q_idx, q_iter_idx);
        }
    }
}

} // namespace deep_gemm
