// Copyright (c) 2026. ECHO experimental Ascend backend; MIT license.
#pragma once
#include <deep_gemm/echo_prefetch_types.hpp>
#include <simt_api/asc_simt.h>
#include <simt_api/device_atomic_functions.h>
#include <simt_api/device_sync_functions.h>

namespace deep_gemm::echo {
// Called in a separate same-stream launch BEFORE clearing the old output list.
// Do not use counts: reset of a row counter must not race another lane's scan.
// Every hosts entry was initialized to -1 or written by a previous valid launch.
__simt_callee__ __aicore__ inline void clear_slot(EchoPrefetchArgs a, uint64_t slot) {
    auto* hosts = reinterpret_cast<__gm__ int32_t*>(a.selected_hosts);
    const int32_t host = hosts[slot];
    if (valid_host(host, a.host_tokens)) {
        auto* mapping = reinterpret_cast<__gm__ int32_t*>(a.host_to_device);
        // Preserve a real resident slot installed by the downstream consumer,
        // or an already-invalidated/replaced entry; only retire our exact tag.
        asc_atomic_cas(mapping + host, staging_location(a.resident_capacity, slot), kMiss);
    }
    reinterpret_cast<__gm__ int32_t*>(a.selected_tokens)[slot] = -1;
    hosts[slot] = -1;
}
__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void clear_vf(
    EchoPrefetchArgs a, uint32_t core, uint32_t cores) {
    const uint64_t first = static_cast<uint64_t>(core) * kThreads + threadIdx.x;
    const uint64_t stride = static_cast<uint64_t>(cores) * kThreads;
    for (uint64_t slot = first; slot < static_cast<uint64_t>(a.rows) * a.capacity; slot += stride)
        clear_slot(a, slot);
    for (uint64_t row = first; row < a.rows; row += stride)
        reinterpret_cast<__gm__ int32_t*>(a.counts)[row] = 0;
    asc_syncthreads();
}
} // namespace deep_gemm::echo
