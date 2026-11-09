// Copyright (c) 2026. ECHO experimental Ascend backend; MIT license.
#pragma once
#include <deep_gemm/ascend.hpp>
#include <deep_gemm/echo_prefetch_clear.hpp>
namespace deep_gemm {
__global__ __vector__ void echo_prefetch_clear_impl(EchoPrefetchArgs a, uint32_t cores) {
    asc_init();
    asc_vf_call<echo::clear_vf>(cce::dim3(echo::kThreads), a, block_idx, cores);
    asc_sync_pipe(PIPE_ALL);
}
} // namespace deep_gemm
