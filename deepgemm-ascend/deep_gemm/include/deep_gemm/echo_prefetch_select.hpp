// Copyright (c) 2026. ECHO experimental Ascend backend; MIT license.
#pragma once
#include <deep_gemm/echo_prefetch_types.hpp>
#include <simt_api/asc_simt.h>
#include <simt_api/device_atomic_functions.h>
#include <simt_api/device_sync_functions.h>
#include <simt_api/device_warp_functions.h>

namespace deep_gemm::echo {
// Shared by the real VF and the multi-threaded CPU algorithm tests.
__simt_callee__ __aicore__ inline void select(
    EchoPrefetchArgs a, __ubuf__ uint16_t* logits, __ubuf__ int32_t* scratch,
    uint32_t row, uint32_t token_begin, uint32_t count)
{
    const uint32_t lane = threadIdx.x, warp = lane / 32U, warp_lane = lane % 32U;
    auto* counters = reinterpret_cast<__gm__ int32_t*>(a.counts);
    auto* mapping = reinterpret_cast<__gm__ int32_t*>(a.host_to_device);
    const int32_t active = *reinterpret_cast<__gm__ int32_t*>(a.active_rows);
    const int32_t stable = reinterpret_cast<__gm__ int32_t*>(a.stable_lens)[row];
    const int32_t requested = reinterpret_cast<__gm__ int32_t*>(a.budgets)[row];
    const uint32_t budget = requested <= 0 ? 0U :
        (static_cast<uint32_t>(requested) < a.capacity ? static_cast<uint32_t>(requested) : a.capacity);
    const float threshold = reinterpret_cast<__gm__ float*>(a.thresholds)[row];
    const bool row_valid = active > 0 && row < static_cast<uint32_t>(active) && row < a.rows;
    if (lane == 0) {
        const int32_t used = asc_atomic_cas(counters + row, 0, 0);
        scratch[6] = row_valid && budget && used >= 0 && static_cast<uint32_t>(used) < budget;
        scratch[4] = scratch[5] = 0;
    }
    asc_syncthreads();
    // Block-uniform early exit keeps every barrier converged.
    if (!scratch[6]) return;
    int32_t host[4] = {-1, -1, -1, -1};
    uint32_t mask = 0, selected = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t offset = lane * 4U + i, token = token_begin + offset;
        if (!row_valid || !budget || offset >= count || token >= a.context_capacity ||
            stable <= 0 || token >= static_cast<uint32_t>(stable)) continue;
        union { uint32_t bits; float value; } score;
        score.bits = static_cast<uint32_t>(logits[offset]) << 16U;
        if (!(score.value > threshold)) continue; // Strict >; NaNs select nothing.
        const int32_t h = reinterpret_cast<__gm__ int32_t*>(a.host_ids)[
            static_cast<uint64_t>(row) * a.context_capacity + token];
        if (!valid_host(h, a.host_tokens)) continue;
        if (reinterpret_cast<__gm__ int32_t*>(a.written)[h] == 0) continue;
        // Claim before reserving slots. All concurrent map accesses are atomic:
        // another tile/query seeing a claim or an encoded staging slot skips it.
        // No spin waiting on another block, so an unscheduled owner cannot deadlock us.
        if (asc_atomic_cas(mapping + h, kMiss, kClaimed) != kMiss) continue;
        host[i] = h; mask |= 1U << i; ++selected;
    }
    // Four warps compact one score tile; one bounded reservation per tile/row.
    uint32_t inclusive = selected;
    for (uint32_t d = 1; d < 32; d <<= 1U) {
        const uint32_t previous = asc_shfl_up(inclusive, d, 32);
        if (warp_lane >= d) inclusive += previous;
    }
    if (warp_lane == 31) scratch[warp] = static_cast<int32_t>(inclusive);
    asc_syncthreads();
    uint32_t rank = inclusive - selected;
    for (uint32_t w = 0; w < warp; ++w) rank += static_cast<uint32_t>(scratch[w]);
    if (lane == 0) {
        uint32_t wanted = 0;
        for (uint32_t w = 0; w < kWarps; ++w) wanted += static_cast<uint32_t>(scratch[w]);
        uint32_t base = 0, accepted = 0;
        if (wanted) {
            int32_t old = asc_atomic_cas(counters + row, 0, 0);
            while (old >= 0) {
                base = static_cast<uint32_t>(old);
                accepted = grant(base, wanted, budget);
                if (!accepted) break;
                const int32_t seen = asc_atomic_cas(counters + row, old, old + static_cast<int32_t>(accepted));
                if (seen == old) break;
                old = seen;
            }
        }
        scratch[4] = static_cast<int32_t>(base);
        scratch[5] = static_cast<int32_t>(accepted);
    }
    asc_syncthreads();
    const uint32_t base = static_cast<uint32_t>(scratch[4]);
    const uint32_t accepted = static_cast<uint32_t>(scratch[5]);
    for (uint32_t i = 0; i < 4; ++i) {
        if (!(mask & (1U << i))) continue;
        if (rank < accepted) {
            const uint32_t token = token_begin + lane * 4U + i;
            scratch[kHeader + rank] = host[i];
            const uint64_t output = static_cast<uint64_t>(row) * a.capacity + base + rank;
            reinterpret_cast<__gm__ int32_t*>(a.selected_tokens)[output] = static_cast<int32_t>(token);
            reinterpret_cast<__gm__ int32_t*>(a.selected_hosts)[output] = host[i];
            // This encodes ownership/location, not DMA readiness. Consumers must
            // wait for the entire fused launch before following this address.
            asc_atomic_cas(mapping + host[i], kClaimed, staging_location(a.resident_capacity, output));
        } else {
            // A concurrent reservation exhausted the row budget. Return the
            // unallocated claim to MISS; never publish a slot without a copy.
            asc_atomic_cas(mapping + host[i], kClaimed, kMiss);
        }
        ++rank;
    }
    asc_syncthreads();
}
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void select_vf(
    EchoPrefetchArgs a, __ubuf__ uint16_t* logits, __ubuf__ int32_t* scratch,
    uint32_t row, uint32_t token_begin, uint32_t count)
{
    select(a, logits, scratch, row, token_begin, count);
}
} // namespace deep_gemm::echo
