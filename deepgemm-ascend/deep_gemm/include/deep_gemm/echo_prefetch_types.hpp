// Copyright (c) 2026. ECHO experimental Ascend backend; MIT license.
#pragma once
#include <cstdint>

#if defined(__CCE__)
#define ECHO_HD __simt_callee__ __aicore__
#else
#define ECHO_HD
#endif
namespace deep_gemm {
// Raw addresses keep the ABI usable by the host JIT and the device compiler.
struct EchoPrefetchArgs {
    uintptr_t thresholds, stable_lens, active_rows, budgets;
    uintptr_t host_ids, host_to_device, written;
    uintptr_t counts, selected_tokens, selected_hosts, staging;
    uint64_t source_dva;
    uint32_t rows, context_capacity, host_tokens, capacity, token_bytes, resident_capacity;
};
namespace echo {
constexpr uint32_t kTile = 512, kThreads = 128, kWarps = 4;
constexpr uint32_t kHeader = 16;
constexpr uint32_t kScratchBytes = (kHeader + kTile) * sizeof(int32_t);
constexpr uint32_t kCopyBytes = 2048;
constexpr int32_t kMiss = -1;
constexpr int32_t kClaimed = -2; // Internal transient state, never visible after successful completion.
// The host checks resident_capacity + rows * capacity <= INT32_MAX.
ECHO_HD constexpr int32_t staging_location(uint32_t resident_capacity, uint64_t flat_slot) {
    return static_cast<int32_t>(static_cast<uint64_t>(resident_capacity) + flat_slot);
}
ECHO_HD constexpr bool is_resident(int32_t location, uint32_t resident_capacity) {
    return location >= 0 && static_cast<uint32_t>(location) < resident_capacity;
}
ECHO_HD constexpr bool is_staging(int32_t location, uint32_t resident_capacity, uint64_t slots) {
    return location >= 0 && static_cast<uint64_t>(location) >= resident_capacity &&
           static_cast<uint64_t>(location) - resident_capacity < slots;
}

// Count is a bounded allocation cursor, NOT a readiness notification. Consumers
// must run after the whole launch on the same stream (or wait on its completion).
ECHO_HD constexpr uint32_t grant(uint32_t used, uint32_t wanted, uint32_t capacity) {
    return used >= capacity ? 0U : (wanted < capacity - used ? wanted : capacity - used);
}
ECHO_HD constexpr bool valid_host(int32_t host, uint32_t host_tokens) {
    return host > 0 && static_cast<uint32_t>(host) < host_tokens; // ID zero is ECHO padding.
}
} // namespace echo
} // namespace deep_gemm
#undef ECHO_HD
