#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/sm90_utils.cuh>
#include <deep_gemm/impls/sm90_fp8_mqa_logits.cuh>

#ifndef DG_ENABLE_SM90_FP8_PAGED_MQA_PREFETCH
#define DG_ENABLE_SM90_FP8_PAGED_MQA_PREFETCH 0
#endif

namespace deep_gemm {

template <uint32_t kAlignedBatchSize, uint32_t SPLIT_KV, uint32_t kNumSMs>
__global__ __launch_bounds__(32, 1)
void smxx_paged_mqa_logits_metadata(const uint32_t batch_size, const uint32_t* context_lens, uint32_t* schedule_metadata) {
    DG_STATIC_ASSERT(kAlignedBatchSize % 32 == 0, "Invalid aligned batch size");
    const uint32_t lane_idx = get_lane_idx();

    uint32_t num_segs[kAlignedBatchSize / 32];
    #pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++ k) {
        const uint32_t& context_len = (k * 32 + lane_idx < batch_size ? __ldg(context_lens + k * 32 + lane_idx) : 0);
        num_segs[k] = ceil_div(context_len, SPLIT_KV);
    }

    __shared__ uint32_t prefix_sum[kAlignedBatchSize];
    uint32_t sum = 0;
    #pragma unroll
    for (uint32_t k = 0; k < kAlignedBatchSize / 32; ++ k) {
        uint32_t x = num_segs[k];
        #pragma unroll
        for (uint32_t offset = 1; offset < 32; offset <<= 1) {
            const uint32_t& y = __shfl_up_sync(0xffffffff, x, offset);
            x += (lane_idx >= offset ? y : 0);
        }
        x += sum;
        prefix_sum[k * 32 + lane_idx] = x;
        sum = __shfl_sync(0xffffffff, x, 31);
    }

    const uint32_t& q = sum / kNumSMs, r = sum % kNumSMs;
    for (uint32_t sm_idx = lane_idx; sm_idx <= kNumSMs; sm_idx += 32) {
        uint32_t seg_starts = sm_idx * q + min(sm_idx, r);
        uint32_t q_idx = 0;
        while (q_idx < batch_size and prefix_sum[q_idx] <= seg_starts)
            ++ q_idx;
        const uint32_t& kv_split_idx = (q_idx == 0 ? seg_starts : seg_starts - prefix_sum[q_idx - 1]);
        __syncwarp();

        schedule_metadata[sm_idx * 2] = q_idx;
        schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
    }
}

template <uint32_t BLOCK_KV, uint32_t kNumMathWarpGroups>
struct PagedMQALogitsScheduler {
    uint32_t batch_size;
    const uint32_t* context_lens;

    uint32_t current_q_idx, current_kv_idx;
    uint32_t end_q_idx, end_kv_idx;
    uint32_t current_num_kv;

    __device__ __forceinline__ explicit PagedMQALogitsScheduler(const uint32_t& batch_size, const uint32_t& sm_idx,
                                                                const uint32_t* context_lens, const uint32_t* schedule_meta) {
        this->batch_size = batch_size;
        this->context_lens = context_lens;

        const auto& current_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + sm_idx);
        const auto& end_pack = __ldg(reinterpret_cast<const uint2*>(schedule_meta) + sm_idx + 1);
        current_q_idx = current_pack.x, current_kv_idx = current_pack.y * kNumMathWarpGroups;
        end_q_idx = end_pack.x, end_kv_idx = end_pack.y * kNumMathWarpGroups;

        current_num_kv = current_q_idx < batch_size ? ceil_div(__ldg(this->context_lens + current_q_idx), BLOCK_KV) : 0;
    }

    __device__ __forceinline__ bool fetch_next_task(uint32_t &q_idx, uint32_t &kv_idx, uint32_t &num_kv) {
        q_idx = current_q_idx;
        kv_idx = current_kv_idx;
        num_kv = current_num_kv;

        if (q_idx == end_q_idx and kv_idx == end_kv_idx)
            return false;

        current_kv_idx += kNumMathWarpGroups;
        if (current_kv_idx >= current_num_kv) {
            ++ current_q_idx;
            current_kv_idx = 0;
            current_num_kv = current_q_idx < batch_size ? ceil_div(__ldg(this->context_lens + current_q_idx), BLOCK_KV) : 0;
        }

        return true;
    }

    __device__ __forceinline__ bool exist_q_idx(const uint32_t& q_idx) const {
        return q_idx < end_q_idx or q_idx == end_q_idx and 0 < end_kv_idx;
    }
};

using namespace deep_gemm::sm90;

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1)
void sm90_fp8_paged_mqa_logits(const uint32_t batch_size,
                               const uint64_t logits_stride, const uint64_t block_table_stride,
                               const uint32_t* context_lens, float* logits,
                               const uint32_t* block_table, const uint32_t* schedule_meta,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // Types
    using WGMMA = typename FP8MMASelector<kNextN * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();

    // Prefetch TMA descriptors
    static constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumMathWarpGroups, "Invalid `SPLIT_KV`");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextN * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextN * kNumHeads * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE = constexpr_align(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_Q_PIPE_SIZE = kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
                                                 constexpr_align(kNumQStages * 8 * 2, kSwizzleAlignment);

    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_PIPE_SIZE = kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE) +
                                                  constexpr_align(kNumKVStages * 8 * 2, kSwizzleAlignment);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q data and barriers on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto q_barrier_ptr = reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
    auto full_q_barriers  = PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + i; });
    auto empty_q_barriers = PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + (kNumQStages + i); });

    // Separate math warpgroups and tma load warps into KV groups
    // Each math warpgroup corresponds to a tma load warp
    const auto& kv_group_idx = __shfl_sync(0xffffffff, threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warpgroup_idx, 0);

    // Per group KV data and barriers on shared memory
    const auto& smem_offset = SMEM_Q_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales =  PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });
    auto kv_barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + i; });
    auto empty_kv_barriers = PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + kNumKVStages + i; });

    // Initialize barriers
    if (warp_idx >= kNumMathThreads / 32 and cute::elect_one_sync()) {
        if (kv_group_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumQStages; ++ i) {
                full_q_barriers[i]->init(1);
                empty_q_barriers[i]->init(kNumMathThreads);
            }
        }
        if (kv_group_idx < kNumMathWarpGroups) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumKVStages; ++ i) {
                full_kv_barriers[i]->init(1);
                empty_kv_barriers[i]->init(128);
            }
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 64;
    constexpr uint32_t kNumMathRegisters = 104;

    // Scheduler
    auto scheduler = PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups>(batch_size, blockIdx.x, context_lens, schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    // Q and KV pipeline
    const auto& get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1}; // Q pipeline stage and phase
    };
    const auto& get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1}; // KV pipeline stage and phase
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    if (warp_idx >= kNumMathThreads / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (kv_group_idx >= kNumMathWarpGroups)
            return;

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, q_idx * kNextN * kNumHeads);
                tma_copy(&tensor_map_weights, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_weights[stage_idx], 0, q_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first Q
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_idx), q_iter_idx = 1;

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch next Q when current Q changes
            bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_idx(next_q_idx + 1));
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            // Wait Q consumer release and issue TMA Q
            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + 1);
            }

            // Read KV block index
            // TODO: deal with `-1`?
            if (kv_idx == 0 or kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage = (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups < num_kv ?
                    __ldg(block_table + q_idx * block_table_stride + (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups)) : 0);
            }
            const auto& kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr ++);

            // Wait KV consumer release
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            // Issue TMA KV
            if (cute::elect_one_sync()) {
                tma_3d_copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                            smem_kv[kv_stage_idx], 0, 0, kv_block_idx);
                tma_copy(&tensor_map_kv_scales, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                            smem_kv_scales[kv_stage_idx], 0, kv_block_idx);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            // Fetch next task
            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        float accum[WGMMA::kNumAccum], weights[kNextN][kNumHeads / 4];
        const auto& sub_warp_offset = (warp_idx % 4) * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            // Current Q changes
            if (q_idx != next_q_idx) {
                // Release Last Q empty
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                // Wait TMA Q arrival
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextN; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                        weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                }
            }

            // Get current Q and KV index
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            // Calculate KV offset in advance
            auto kv_offset = q_idx * kNextN * logits_stride + ((kv_idx + kv_group_idx) * BLOCK_KV + sub_warp_offset);

            // Compute `[kNextN * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [kNextN, BLOCK_KV]`
            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Issue WGMMA
            DG_STATIC_ASSERT(BLOCK_KV == 64, "Invalid block size");
            DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                auto desc_a = make_smem_desc(smem_kv[kv_stage_idx] + k * WGMMA::K, to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                auto desc_b = make_smem_desc(smem_q[q_stage_idx] + k * WGMMA::K, to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                WGMMA::wgmma(desc_a, desc_b, accum, k);
            }
            warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                warpgroup_fence_operand(accum[i]);

            // Read per-KV scales
            float scale_kv_0 = ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_0_offset);
            float scale_kv_1 = ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_1_offset);

            // Wait WGMMA
            warpgroup_wait<0>();

            // Release KV empty
            empty_kv_barriers[kv_stage_idx]->arrive();

            // Reduce over the head dim and store
            static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
            DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
            DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == kNextN, "Invalid accumulation");
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");
            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
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
                logits[kv_offset + i * logits_stride + v_0_offset] = v_0;
                logits[kv_offset + i * logits_stride + v_1_offset] = v_1;
            }
        }
    }
}

template <uint32_t kNextN, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t kMLAHeadDim, uint32_t BLOCK_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages, uint32_t kNumPrefetchStages,
          uint32_t SPLIT_KV,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads, uint32_t kNumPrefetchThreads>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads + kNumPrefetchThreads, 1)
void sm90_fp8_paged_mqa_logits_fused_v2(const uint32_t batch_size,
                                     const uint64_t logits_stride, const uint64_t block_table_stride,
                                     const uint64_t page_table_1_stride,
                                     const uint32_t* context_lens, float* logits,
                                     const uint32_t* block_table, const uint32_t* schedule_meta,
                                     const uint32_t* page_table_1,
                                     __nv_bfloat16* device_pool_buf, __nv_bfloat16* host_pool_buf,
                                     int* prefetch_host_loc, __nv_bfloat16* prefetch_kv_buf,
                                     int* host_token_to_device, uint32_t* query_recall_counter,
                                     const float* decode_topk_logits, 
                                     uint32_t device_pool_size_m1,
                                     const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                                     const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                                     const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
                                     const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    // Types
    using WGMMA = typename FP8MMASelector<kNextN * kNumHeads>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
    const auto& warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const auto& warpgroup_idx = warp_idx / 4;
    const auto& lane_idx = get_lane_idx();

    // Prefetch TMA descriptors
    static constexpr uint32_t kNumMathWarpGroups = kNumMathThreads / 128;
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == BLOCK_KV * kNumMathWarpGroups, "Invalid `SPLIT_KV`");
    DG_STATIC_ASSERT(kNumPrefetchThreads > 0 and kNumPrefetchThreads % 32 == 0, "Invalid threads");
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
    }
    __syncwarp();

    // Shared memory configs
    static constexpr uint32_t kSwizzleAlignment = kHeadDim * 8;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = kNextN * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = kNextN * kNumHeads * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE = constexpr_align(SMEM_WEIGHT_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_Q_PIPE_SIZE = kNumQStages * (SMEM_Q_SIZE_PER_STAGE + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE) +
                                                 constexpr_align(kNumQStages * 8 * 2, kSwizzleAlignment);

    static constexpr uint32_t SMEM_LOGITS_SIZE_PER_STAGE = kNextN * kNumMathWarpGroups * BLOCK_KV * sizeof(float);
    static constexpr uint32_t SMEM_PF_PIPE_SIZE = kNumPrefetchStages * SMEM_LOGITS_SIZE_PER_STAGE  +
                                                 constexpr_align(kNumPrefetchStages * 8 * 2, kSwizzleAlignment);

    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE = BLOCK_KV * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE = constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, kSwizzleAlignment);
    static constexpr uint32_t SMEM_KV_PIPE_SIZE = kNumKVStages * (SMEM_KV_SIZE_PER_STAGE + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE) +
                                                  constexpr_align(kNumKVStages * 8 * 2, kSwizzleAlignment);

    // Align to swizzling alignment bytes
    extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");
    DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % kSwizzleAlignment == 0, "Unaligned TMA swizzling");

    // Q data and barriers on shared memory
    auto smem_q = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * i);
    });
    auto smem_weights = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * i);
    });
    auto smem_logits = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages + ALIGNED_SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages + SMEM_LOGITS_SIZE_PER_STAGE * i);
    });
    auto q_barrier_ptr = reinterpret_cast<Barrier*>(smem_logits[kNumPrefetchStages]);
    auto full_q_barriers  = PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + i; });
    auto empty_q_barriers = PatternVisitor([&](const uint32_t& i) { return q_barrier_ptr + (kNumQStages + i); });
    auto full_logits_barriers = PatternVisitor([&](const uint32_t& i) { 
        return q_barrier_ptr + kNumQStages * 2 + i;
    });
    auto empty_logits_barriers = PatternVisitor([&](const uint32_t& i) { 
        return q_barrier_ptr + kNumQStages * 2 + kNumPrefetchStages + i;
    });

    // Separate math warpgroups and tma load warps into KV groups
    // Each math warpgroup corresponds to a tma load warp
    // const auto& kv_group_idx = __shfl_sync(0xffffffff, threadIdx.x >= kNumMathThreads ? (threadIdx.x - kNumMathThreads) / 32 : warpgroup_idx, 0);
    uint32_t kv_group_idx = 0;
    if (threadIdx.x >= kNumMathThreads + kNumTMAThreads) {
        kv_group_idx = __shfl_sync(0xffffffff, (threadIdx.x - kNumMathThreads - kNumTMAThreads) / 32, 0);
    }
    else if (threadIdx.x >= kNumMathThreads) {
        kv_group_idx = __shfl_sync(0xffffffff, (threadIdx.x - kNumMathThreads) / 32, 0);
    }
    else {
        kv_group_idx = __shfl_sync(0xffffffff, warpgroup_idx, 0);
    }

    // Per group KV data and barriers on shared memory
    const auto& smem_offset = SMEM_Q_PIPE_SIZE + SMEM_PF_PIPE_SIZE + SMEM_KV_PIPE_SIZE * kv_group_idx;
    auto smem_kv = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * i);
    });
    auto smem_kv_scales =  PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + smem_offset + SMEM_KV_SIZE_PER_STAGE * kNumKVStages + ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
    });

    auto kv_barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
    auto full_kv_barriers  = PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + i; });
    auto empty_kv_barriers = PatternVisitor([&](const uint32_t& i) { return kv_barrier_ptr + kNumKVStages + i; });


    // Initialize barriers
    if (warp_idx >= kNumMathThreads / 32 and warp_idx < (kNumMathThreads + kNumTMAThreads) / 32 and cute::elect_one_sync()) {
        if (kv_group_idx == 0) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumQStages; ++ i) {
                full_q_barriers[i]->init(1);
                empty_q_barriers[i]->init(kNumMathThreads);
            }
            #pragma unroll
            for (uint32_t i = 0; i < kNumPrefetchStages; ++ i) {
                full_logits_barriers[i]->init(kNumMathThreads);
                empty_logits_barriers[i]->init(kNumPrefetchThreads);
            }
        }
        if (kv_group_idx < kNumMathWarpGroups) {
            #pragma unroll
            for (uint32_t i = 0; i < kNumKVStages; ++ i) {
                full_kv_barriers[i]->init(1);
                empty_kv_barriers[i]->init(128);
            }
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = 32;
    constexpr uint32_t kNumMathRegisters = 104;
    constexpr uint32_t kNumPfRegisters = 32;

    // Scheduler
    auto scheduler = PagedMQALogitsScheduler<BLOCK_KV, kNumMathWarpGroups>(batch_size, blockIdx.x, context_lens, schedule_meta);
    DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0, "Unaligned SPLIT_KV");

    // Q and KV pipeline
    const auto& get_q_pipeline = [=](const uint32_t& q_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {q_iter_idx % kNumQStages, (q_iter_idx / kNumQStages) & 1}; // Q pipeline stage and phase
    };
    const auto& get_kv_pipeline = [=](const uint32_t& kv_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {kv_iter_idx % kNumKVStages, (kv_iter_idx / kNumKVStages) & 1}; // KV pipeline stage and phase
    };
    uint32_t q_iter_idx = 0, kv_iter_idx = 0;

    const auto& get_pf_pipeline = [=](const uint32_t& pf_iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {pf_iter_idx % kNumPrefetchStages, (pf_iter_idx / kNumPrefetchStages) & 1};
    };
    uint32_t pf_iter_idx = 0;

    constexpr auto req_pf_max = 64;
    constexpr auto kTransferItemBytes = kMLAHeadDim * sizeof(__nv_bfloat16);

    if (warp_idx >= kNumMathThreads / 32 and warp_idx < (kNumMathThreads + kNumTMAThreads) / 32) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        if (kv_group_idx >= kNumMathWarpGroups)
            return;

        const auto& issue_tma_q = [&](const uint32_t& stage_idx, const uint32_t& q_idx) {
            if (kv_group_idx == 0 and cute::elect_one_sync()) {
                tma_copy(&tensor_map_q, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_q[stage_idx], 0, q_idx * kNextN * kNumHeads);
                tma_copy(&tensor_map_weights, reinterpret_cast<uint64_t*>(full_q_barriers[stage_idx]), smem_weights[stage_idx], 0, q_idx);
                full_q_barriers[stage_idx]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        };

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx, num_kv;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        bool fetched_next_task;

        // Prefetch the first Q
        if ((fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)))
            issue_tma_q(0, next_q_idx), q_iter_idx = 1;

        int kv_block_idx_ptr = 32;
        uint32_t kv_block_idx_storage;

        while (fetched_next_task) {
            // Prefetch next Q when current Q changes
            bool prefetch_q = (q_idx != next_q_idx and scheduler.exist_q_idx(next_q_idx + 1));
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;
            num_kv = next_num_kv;

            // Wait Q consumer release and issue TMA Q
            if (prefetch_q) {
                CUTE_TIE_DECL(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
                issue_tma_q(q_stage_idx, q_idx + 1);
            }

            // Read KV block index
            // TODO: deal with `-1`?
            if (kv_idx == 0 or kv_block_idx_ptr == 32) {
                kv_block_idx_ptr = 0;
                kv_block_idx_storage = (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups < num_kv ?
                    __ldg(block_table + q_idx * block_table_stride + (kv_idx + kv_group_idx + lane_idx * kNumMathWarpGroups)) : 0);
            }
            const auto& kv_block_idx = __shfl_sync(0xffffffff, kv_block_idx_storage, kv_block_idx_ptr ++);

            // Wait KV consumer release
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

            // Issue TMA KV
            if (cute::elect_one_sync()) {
                tma_3d_copy(&tensor_map_kv, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                            smem_kv[kv_stage_idx], 0, 0, kv_block_idx);
                tma_copy(&tensor_map_kv_scales, reinterpret_cast<uint64_t*>(full_kv_barriers[kv_stage_idx]),
                            smem_kv_scales[kv_stage_idx], 0, kv_block_idx);
                full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
            }

            // Fetch next task
            fetched_next_task = scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv);
        }
    } else if (warp_idx < kNumMathThreads / 32) {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        float accum[WGMMA::kNumAccum], weights[kNextN][kNumHeads / 4];
        const auto& sub_warp_offset = (warp_idx % 4) * 16;
        const auto& v_0_offset = lane_idx / 4 + 0;
        const auto& v_1_offset = lane_idx / 4 + 8;

        // Initialize `q_idx` outside `[0, batch_size)` to indicate it was none
        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;
        uint32_t q_stage_idx, q_phase;

        while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
            // Current Q changes
            if (q_idx != next_q_idx) {
                // Release Last Q empty
                if (q_iter_idx > 0)
                    empty_q_barriers[(q_iter_idx - 1) % kNumQStages]->arrive();

                // Wait TMA Q arrival
                CUTE_TIE(get_q_pipeline(q_iter_idx ++), q_stage_idx, q_phase);
                full_q_barriers[q_stage_idx]->wait(q_phase);

                // Read weights
                #pragma unroll
                for (uint32_t i = 0; i < kNextN; ++ i) {
                    #pragma unroll
                    for (uint32_t j = 0; j < kNumHeads / 4; ++ j)
                        weights[i][j] = ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + (j / 2) * 8 + (j & 1) + (lane_idx % 4) * 2);
                }
            }

            // Get current Q and KV index
            q_idx = next_q_idx;
            kv_idx = next_kv_idx;

            // Calculate KV offset in advance
            auto kv_offset = q_idx * kNextN * logits_stride + ((kv_idx + kv_group_idx) * BLOCK_KV + sub_warp_offset);

            // Compute `[kNextN * kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim] -> [kNextN, BLOCK_KV]`
            // Wait TMA KV arrival
            CUTE_TIE_DECL(get_kv_pipeline(kv_iter_idx ++), kv_stage_idx, kv_phase);
            full_kv_barriers[kv_stage_idx]->wait(kv_phase);

            // Issue WGMMA
            DG_STATIC_ASSERT(BLOCK_KV == 64, "Invalid block size");
            DG_STATIC_ASSERT(kHeadDim % WGMMA::K == 0, "Invalid head dim");
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                warpgroup_fence_operand(accum[i]);
            warpgroup_arrive();
            #pragma unroll
            for (uint32_t k = 0; k < kHeadDim / WGMMA::K; ++ k) {
                auto desc_a = make_smem_desc(smem_kv[kv_stage_idx] + k * WGMMA::K, to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                auto desc_b = make_smem_desc(smem_q[q_stage_idx] + k * WGMMA::K, to_swizzle_cute_type<kHeadDim>(), 0, kHeadDim * 8);
                WGMMA::wgmma(desc_a, desc_b, accum, k);
            }
            warpgroup_commit_batch();
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                warpgroup_fence_operand(accum[i]);

            // Read per-KV scales
            float scale_kv_0 = ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_0_offset);
            float scale_kv_1 = ld_shared(smem_kv_scales[kv_stage_idx] + sub_warp_offset + v_1_offset);

            // Wait WGMMA
            warpgroup_wait<0>();

            // Release KV empty
            empty_kv_barriers[kv_stage_idx]->arrive();

            // Reduce over the head dim and store
            static constexpr uint32_t kNumAccumPerReduce = kNumHeads / 2;
            DG_STATIC_ASSERT(WGMMA::kNumAccum % kNumAccumPerReduce == 0, "Invalid accumulation");
            DG_STATIC_ASSERT(WGMMA::kNumAccum / kNumAccumPerReduce == kNextN, "Invalid accumulation");
            DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

            float v_0[kNextN], v_1[kNextN];
            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
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

            CUTE_TIE_DECL(get_pf_pipeline(pf_iter_idx++), pf_stage_idx, pf_phase);
            empty_logits_barriers[pf_stage_idx]->wait(pf_phase ^ 1);    // !!! IMPORTANT !!!

            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
                smem_logits[pf_stage_idx][i * kNumMathWarpGroups * BLOCK_KV + kv_group_idx * BLOCK_KV + sub_warp_offset + v_0_offset] = v_0[i];
                smem_logits[pf_stage_idx][i * kNumMathWarpGroups * BLOCK_KV + kv_group_idx * BLOCK_KV + sub_warp_offset + v_1_offset] = v_1[i];
            }
            full_logits_barriers[pf_stage_idx]->arrive();

            #pragma unroll
            for (uint32_t i = 0; i < kNextN; ++ i) {
                // Store into the global memory
                // NOTES: we have redundant writes here, consider more carefully
                logits[kv_offset + i * logits_stride + v_0_offset] = v_0[i];
                logits[kv_offset + i * logits_stride + v_1_offset] = v_1[i];
            }
        }
    } else if (threadIdx.x < kNumMathThreads + kNumTMAThreads + kNumPrefetchThreads) {
        // Prefetch warp group
        // All warps scan logits and issue their own warp-cooperative transfer
        // requests.  This removes the dedicated polling transfer warp.
        if constexpr (kNumPrefetchThreads >= 128)
            cutlass::arch::warpgroup_reg_dealloc<kNumPfRegisters>();
        const auto pf_tx = threadIdx.x - (kNumMathThreads + kNumTMAThreads);
        __shared__ int prefetch_host_loc_temp[kNumPrefetchThreads];
        uint32_t q_idx = batch_size, kv_idx;
        uint32_t next_q_idx, next_kv_idx, next_num_kv;

        // Number of scan threads
        static constexpr uint32_t kNumScanThreads = kNumPrefetchThreads;

        {
            // ==================== Scan + transfer warps ====================
            uint32_t pf_iter_idx = 0;
            // `recall_func` is called in a CONVERGED warp context (all 32 lanes
            // enter the lambda simultaneously).  Lanes with `is_miss == false`
            // pass their no-op through the same control flow so that
            // `__ballot_sync(0xffffffff, ...)` does not force divergent
            // reconvergence — that was the main cost observed when the ballot
            // was placed inside a divergent `if (need_copy)` branch.
            const auto& recall_func = [&](const bool is_miss, const int kv_token_host_loc) {
                const uint32_t miss_mask = __ballot_sync(0xffffffffu, is_miss);
                if (miss_mask == 0) return;
                const uint32_t miss_count = __popc(miss_mask);
                const int leader = __ffs(miss_mask) - 1;
                const uint32_t warp_temp_base = (pf_tx / 32) * 32;
                uint32_t slot_base = 0;
                uint32_t valid_count = 0;
                if (static_cast<int>(lane_idx) == leader) {
                    slot_base = atomicAdd(query_recall_counter + q_idx, miss_count);
                    valid_count = slot_base < req_pf_max ? min(req_pf_max - slot_base, miss_count) : 0;
                }
                slot_base = __shfl_sync(0xffffffffu, slot_base, leader);
                valid_count = __shfl_sync(0xffffffffu, valid_count, leader);
                const uint32_t lower_lane_mask = lane_idx == 0 ? 0u : ((1u << lane_idx) - 1);
                const uint32_t miss_rank = __popc(miss_mask & lower_lane_mask);
                if (is_miss && miss_rank < valid_count) {
                    prefetch_host_loc_temp[warp_temp_base + miss_rank] = kv_token_host_loc;
                }

                if (lane_idx < valid_count) {
                    const int host_loc = prefetch_host_loc_temp[warp_temp_base + lane_idx];
                    const uint32_t slot_x = slot_base + lane_idx;
                    prefetch_host_loc[q_idx * req_pf_max + slot_x] = host_loc;
                    host_token_to_device[host_loc] = static_cast<int>(device_pool_size_m1 + 1 + slot_x);
                }
                __syncwarp();

                #pragma unroll
                for (uint32_t i = 0; i < 32; ++ i) {
                    if (i >= valid_count)
                        break;

                    const int host_loc = prefetch_host_loc_temp[warp_temp_base + i];
                    const uint32_t slot_x = slot_base + i;
                    transfer_item_warp_v2<kTransferItemBytes>(
                        &host_pool_buf[(size_t)host_loc * kMLAHeadDim],
                        &prefetch_kv_buf[((size_t)q_idx * req_pf_max + (size_t)slot_x) * kMLAHeadDim]
                    );
                }
            };

            static constexpr uint32_t total_items = kNumMathWarpGroups * BLOCK_KV;
            static constexpr uint32_t aligned_total = (total_items + kNumScanThreads - 1) / kNumScanThreads * kNumScanThreads;
            static constexpr uint32_t kMaxIters = aligned_total / kNumScanThreads;

            while (scheduler.fetch_next_task(next_q_idx, next_kv_idx, next_num_kv)) {
                q_idx = next_q_idx;
                kv_idx = next_kv_idx;

                const int context_len = __ldg(context_lens + q_idx);
                const float topk_logits = __ldg(decode_topk_logits + q_idx);

                // const uint32_t rc = *(volatile uint32_t*)recall_counter;
                // const bool prefetch = (rc < total_pf_max);
                const uint32_t qrc = *(volatile uint32_t*)(query_recall_counter + q_idx);
                const bool prefetch = (qrc < req_pf_max);

                // --- Phase 1: read logits from smem (barrier held) ---
                CUTE_TIE_DECL(get_pf_pipeline(pf_iter_idx++), pf_stage_idx, pf_phase);
                full_logits_barriers[pf_stage_idx]->wait(pf_phase);

                bool need_copy_buf[kMaxIters];
                if (prefetch) {
                    #pragma unroll
                    for (uint32_t iter = 0; iter < kMaxIters; ++ iter) {
                        const uint32_t idx = pf_tx + iter * kNumScanThreads;
                        bool need_copy = false;
                        if (prefetch and idx < total_items) {
                            const int kv_token_offset = kv_idx * BLOCK_KV + idx;
                            if (kv_token_offset < context_len - 1) {
                                #pragma unroll
                                for (uint32_t i = 0; i < kNextN; ++ i) {
                                    const float logit = smem_logits[pf_stage_idx][
                                        i * kNumMathWarpGroups * BLOCK_KV + idx
                                    ];
                                    if (logit > topk_logits) {
                                        need_copy = true;
                                    }
                                }
                                // Issue L2 prefetch for page_table_1 so Phase 2 hits cache
                                if (need_copy) {
                                    asm volatile("prefetch.global.L2 [%0];" :: "l"(
                                        page_table_1 + q_idx * page_table_1_stride + kv_token_offset));
                                }
                            }
                        }
                        need_copy_buf[iter] = need_copy;
                    }
                }

                // Release the logits smem buffer -- math warps can proceed
                empty_logits_barriers[pf_stage_idx]->arrive();

                // --- Phase 2: recall decisions ---
                // Re-check global budget before doing expensive global reads.
                // The per-iter work is performed in CONVERGED warp context: every
                // lane evaluates `is_miss` (lanes with `need_copy=false` or OOB
                // contribute `false`) and passes it to `recall_func`, so the
                // `__ballot_sync` inside runs on a full 0xffffffff mask and
                // does not pay divergent-reconvergence cost.

                // const uint32_t qrc = *(volatile uint32_t*)(query_recall_counter + q_idx);
                // const bool prefetch = (qrc < total_pf_max);

                if (prefetch) {
                    bool is_miss_buf[kMaxIters];
                    int kv_token_host_loc_buf[kMaxIters];
                    uint32_t lane_miss_mask = 0;
                    #pragma unroll
                    for (uint32_t iter = 0; iter < kMaxIters; ++ iter) {
                        bool is_miss = false;
                        int kv_token_host_loc = 0;
                        if (need_copy_buf[iter]) {
                            const uint32_t idx = pf_tx + iter * kNumScanThreads;
                            const int kv_token_offset = kv_idx * BLOCK_KV + idx;
                            kv_token_host_loc = __ldg(
                                page_table_1 + q_idx * page_table_1_stride + kv_token_offset
                            );
                            is_miss = host_token_to_device[kv_token_host_loc] == INT32_MAX;
                        }
                        is_miss_buf[iter] = is_miss;
                        kv_token_host_loc_buf[iter] = kv_token_host_loc;
                        lane_miss_mask |= (static_cast<uint32_t>(is_miss) << iter);
                    }
                    // Single cheap warp-vote: skip the whole reservation +
                    // ballot loop when no lane in this warp has any miss.
                    if (__any_sync(0xffffffffu, lane_miss_mask != 0)) {
                        #pragma unroll
                        for (uint32_t iter = 0; iter < kMaxIters; ++ iter) {
                            recall_func(is_miss_buf[iter], kv_token_host_loc_buf[iter]);
                        }
                    }
                }
            }
        }
    }
}

} // namespace deep_gemm
