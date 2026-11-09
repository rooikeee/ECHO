// Copyright (c) 2026. ECHO experimental Ascend backend; MIT license.
#pragma once
#include <deep_gemm/ascend.hpp>
#include <deep_gemm/echo_prefetch_select.hpp>

namespace deep_gemm {
// ECHO staging, not cache insertion. Resident entries are preserved, while
// missing entries acquire a transient staging address. No LRU/eviction is done.
// Slots and host claims owned by different score blocks are disjoint.
struct EchoPrefetch {
    EchoPrefetchArgs args;
    __ubuf__ int32_t* scratch = nullptr;
    __ubuf__ uint8_t* buffers = nullptr;
    bool pending[2] = {false, false};
    uint32_t next = 0;
    static constexpr uint32_t kUBBytes = echo::kScratchBytes + 2U * echo::kCopyBytes;

    __aicore__ void init(__ubuf__ uint8_t* storage) {
        scratch = reinterpret_cast<__ubuf__ int32_t*>(storage);
        buffers = storage + echo::kScratchBytes;
    }
    __aicore__ void consume(uint32_t row, uint32_t token_begin, uint32_t count,
                            __ubuf__ bfloat16_t* logits) {
        // Event 7 in V->S is private to the ECHO hook; MQA uses MTE2->S 0.
        asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID7);
        asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID7);
        asc_vf_call<echo::select_vf>(dim3(echo::kThreads), args,
            reinterpret_cast<__ubuf__ uint16_t*>(logits), scratch, row, token_begin, count);
        asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID7);
        asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID7);
        const uint32_t first = static_cast<uint32_t>(scratch[4]);
        const uint32_t selected = static_cast<uint32_t>(scratch[5]);
        for (uint32_t i = 0; i < selected; ++i) {
            const uint32_t host = static_cast<uint32_t>(scratch[echo::kHeader + i]);
            const uint64_t src = args.source_dva + static_cast<uint64_t>(host) * args.token_bytes;
            const uint64_t dst = args.staging + (static_cast<uint64_t>(row) * args.capacity + first + i) * args.token_bytes;
            for (uint32_t offset = 0; offset < args.token_bytes; offset += echo::kCopyBytes) {
                const uint32_t bytes = min(echo::kCopyBytes, args.token_bytes - offset);
                const event_t event = static_cast<event_t>(6U + next);
                if (pending[next]) asc_sync_wait(PIPE_MTE3, PIPE_MTE2, event);
                auto* local = buffers + next * echo::kCopyBytes;
                copy_gm_to_ub(local, reinterpret_cast<__gm__ uint8_t*>(src + offset), 1, bytes, bytes,
                              asc_load_l2_cache_mode::NOTALLOC_KEEP);
                asc_sync_notify(PIPE_MTE2, PIPE_MTE3, event);
                asc_sync_wait(PIPE_MTE2, PIPE_MTE3, event);
                asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(dst + offset), local, 1, bytes,
                                    asc_store_l2_cache_mode::NOTALLOC_CLEAN, bytes, bytes);
                asc_sync_notify(PIPE_MTE3, PIPE_MTE2, event);
                pending[next] = true;
                next ^= 1U;
            }
        }
    }
    __aicore__ void finish() {
        for (uint32_t i = 0; i < 2; ++i)
            if (pending[i]) asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(6U + i));
        // Kernel completion is the publication boundary for staging and count.
        asc_sync_notify(PIPE_MTE3, PIPE_S, EVENT_ID7);
        asc_sync_wait(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
};
} // namespace deep_gemm
