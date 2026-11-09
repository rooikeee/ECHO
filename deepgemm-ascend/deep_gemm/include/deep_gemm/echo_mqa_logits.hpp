// Copyright (c) 2026. ECHO experimental Ascend backend; MIT license.
#pragma once
#include <deep_gemm/echo_prefetch.hpp>
#include <deep_gemm/mqa_logits.hpp>

namespace deep_gemm {
// First supported contract: MXFP8, H=32, D=128, page=128, row-major page payload.
// The upstream scheduler partitions KV splits across AIC/AIV blocks. Each row's
// staging cursor is atomic; persistent resident mappings are read-only snapshots.
__global__ __mix__(1, 2) void echo_paged_mqa_logits_impl(
    uint32_t num_q_tokens, uint32_t table_stride, uint64_t kv_page_stride,
    gm_ptr<uint8_t, Major::K> q, __gm__ int16_t* sf_q,
    gm_ptr<uint8_t, Major::K> fused_kv, gm_ptr<bfloat16_t, Major::K> weights,
    gm_ptr<bfloat16_t, Major::K> logits,
    __gm__ uint32_t* context_lens, __gm__ uint32_t* block_table,
    __gm__ uint32_t* indices, __gm__ uint32_t* metadata, EchoPrefetchArgs prefetch)
{
    asc_init();
    const auto sf_kv = (fused_kv + 128U * fused_kv.stride_outer).template ptr<int16_t>();
    const auto scheduler = [&](uint32_t core) __aicore__ {
        return PagedMQALogitsScheduler<16U, 512U, 128U, 32U>(core, num_q_tokens, context_lens,
            indices, metadata, block_table, table_stride, kv_page_stride);
    };
    mqa_logits_core_impl<false,32U,128U,512U,64U,2U,2U,4U,3U,3U,128U,false,
                         decltype(scheduler),EchoPrefetch>(q, fused_kv, sf_q, sf_kv, weights, logits,
                                                          scheduler, EchoPrefetch{prefetch});
}
} // namespace deep_gemm
