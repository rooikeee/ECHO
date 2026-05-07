#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cute/container/tuple.hpp>

#include "cute_tie.cuh"

#ifdef __CLION_IDE__

__host__ __device__ __forceinline__ void host_device_printf(const char* format, ...) {
    asm volatile("trap;");
}

#define printf host_device_printf
#endif

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) \
        asm("trap;"); \
} while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

namespace deep_gemm {

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__
    explicit PatternVisitor(FuncT&& func): func(std::forward<FuncT>(func)) {}

    __device__ __host__
    auto operator [](const uint32_t& i) {
        return func(i);
    }
};

template <typename T>
__device__ __host__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ T align(T a, T b) {
    return ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_align(T a, T b) {
    return constexpr_ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
    return b == 0 ? a : constexpr_gcd(b, a % b);
}

template<typename T>
__forceinline__ __device__ void swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

__forceinline__ __device__ uint32_t get_sm_idx() {
    uint32_t sm_idx;
    asm ("mov.u32 %0, %%smid;" : "=r"(sm_idx));
    return sm_idx;
}

__forceinline__ __device__ uint32_t get_lane_idx() {
    uint32_t lane_id;
    asm ("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__  __forceinline__ uint32_t ld_shared(const uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float2 ld_shared(const float2* ptr) {
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float4 ld_shared(const float4* ptr) {
    float4 ret;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ uint4 ld_shared(const uint4* ptr) {
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float ld_shared(const float* ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}

__device__ __forceinline__ void st_shared(const float2* ptr, float2 val) {
    asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "l"(ptr), "f"(val.x), "f"(val.y));
}

__device__ __forceinline__ void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(ptr), "r"(val));
}

__device__  __forceinline__ void st_shared(const void* ptr, uint32_t x, uint32_t y) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "l"(ptr), "r"(x), "r"(y));
}

__device__  __forceinline__ void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(ptr), "r"(x), "r"(y), "r"(z), "r"(w));
}

template <typename old_t>
__device__ __forceinline__ int cast_into_bf16_and_pack(old_t& x, old_t& y) {
    auto bf16x2 = __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
    return *reinterpret_cast<int*>(&bf16x2);
}

__device__ __forceinline__ void prefetch_l1(void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

__device__ __forceinline__ void
transfer_item_thread(const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
    auto* src_u8 = static_cast<const uint8_t*>(src_addr);
    auto* dst_u8 = static_cast<uint8_t*>(dst_addr);
    int64_t copied = 0;

    // Prefer 16B moves when both pointers are 16-byte aligned.
    const auto src_ptr = reinterpret_cast<cuda::std::uintptr_t>(src_addr);
    const auto dst_ptr = reinterpret_cast<cuda::std::uintptr_t>(dst_addr);
    if (((src_ptr | dst_ptr) & 0xf) == 0) {
        const int64_t total_chunks_16b = item_size_bytes / 16;
#pragma unroll
        for (int64_t j = 0; j < total_chunks_16b; ++ j) {
            uint32_t x, y, z, w;
            asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                         : "l"(src_u8 + j * 16)
                         : "memory");
            asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                         :
                         : "l"(dst_u8 + j * 16), "r"(x), "r"(y), "r"(z), "r"(w)
                         : "memory");
        }
        copied = total_chunks_16b * 16;
    }

    const int64_t remaining_after_16b = item_size_bytes - copied;
    const int64_t total_chunks_8b = remaining_after_16b / 8;
#pragma unroll
    for (int64_t j = 0; j < total_chunks_8b; ++ j) {
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src_u8 + copied + j * 8) : "memory");
        asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst_u8 + copied + j * 8), "l"(tmp) : "memory");
    }
    copied += total_chunks_8b * 8;

    const int64_t remaining_after_8b = item_size_bytes - copied;
    const int64_t total_chunks_4b = remaining_after_8b / 4;
#pragma unroll
    for (int64_t j = 0; j < total_chunks_4b; ++ j) {
        uint32_t tmp;
        asm volatile("ld.global.nc.b32 %0,[%1];" : "=r"(tmp) : "l"(src_u8 + copied + j * 4) : "memory");
        asm volatile("st.global.cg.b32 [%0],%1;" ::"l"(dst_u8 + copied + j * 4), "r"(tmp) : "memory");
    }
}

template <int kItemSizeBytes>
__device__ __forceinline__ void
transfer_item_warp(const void* src_addr, void* dst_addr) {
    auto* src_u8 = static_cast<const uint8_t*>(src_addr);
    auto* dst_u8 = static_cast<uint8_t*>(dst_addr);
    int64_t copied = 0;
    const int lane_idx = threadIdx.x % 32;

    // Prefer 16B moves when both pointers are 16-byte aligned.
    const auto src_ptr = reinterpret_cast<cuda::std::uintptr_t>(src_addr);
    const auto dst_ptr = reinterpret_cast<cuda::std::uintptr_t>(dst_addr);
    if (((src_ptr | dst_ptr) & 0xf) == 0) {
        const int64_t total_chunks_16b = kItemSizeBytes / 16;
        for (int64_t j = lane_idx; j < total_chunks_16b; j += 32) {
            uint32_t x, y, z, w;
            asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                         : "l"(src_u8 + j * 16)
                         : "memory");
            asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                         :
                         : "l"(dst_u8 + j * 16), "r"(x), "r"(y), "r"(z), "r"(w)
                         : "memory");
        }
        copied = total_chunks_16b * 16;
    }

    const int64_t remaining_after_16b = kItemSizeBytes - copied;
    const int64_t total_chunks_8b = remaining_after_16b / 8;
    for (int64_t j = lane_idx; j < total_chunks_8b; j += 32) {
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src_u8 + copied + j * 8) : "memory");
        asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst_u8 + copied + j * 8), "l"(tmp) : "memory");
    }
    copied += total_chunks_8b * 8;

    const int64_t remaining_after_8b = kItemSizeBytes - copied;
    const int64_t total_chunks_4b = remaining_after_8b / 4;
    for (int64_t j = lane_idx; j < total_chunks_4b; j += 32) {
        uint32_t tmp;
        asm volatile("ld.global.nc.b32 %0,[%1];" : "=r"(tmp) : "l"(src_u8 + copied + j * 4) : "memory");
        asm volatile("st.global.cg.b32 [%0],%1;" ::"l"(dst_u8 + copied + j * 4), "r"(tmp) : "memory");
    }
}

__device__ __forceinline__ uint4 ld_global_nc_uint4(const uint4* addr) {
    uint4 val;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr)
                 : "memory");
    return val;
}

// Improvement 1: Templated size + load-store separation.
// All loads are issued before any store so the hardware can have all PCIe
// read requests in flight simultaneously rather than serializing on each
// load->store dependency.
template <int kItemSizeBytes>
__device__ __forceinline__ void
transfer_item_warp_v2_ldg(const void* __restrict__ src_addr, void* __restrict__ dst_addr) {
    static_assert(kItemSizeBytes > 0 && kItemSizeBytes % 16 == 0,
                  "item size must be positive and 16-byte aligned");
    constexpr int kChunks = kItemSizeBytes / 16;
    constexpr int kChunksPerLane = (kChunks + 31) / 32;

    const int lane_idx = threadIdx.x % 32;
    const auto* src = static_cast<const uint4*>(src_addr);
    auto* dst = static_cast<uint4*>(dst_addr);

    uint4 regs[kChunksPerLane];
    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            regs[i] = __ldg(src + idx);
        }
    }

    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "l"(dst + idx),
                            "r"(regs[i].x), "r"(regs[i].y),
                            "r"(regs[i].z), "r"(regs[i].w)
                         : "memory");
        }
    }
}

template <int kItemSizeBytes>
__device__ __forceinline__ void
transfer_item_warp_v2(const void* __restrict__ src_addr, void* __restrict__ dst_addr) {
    static_assert(kItemSizeBytes > 0 && kItemSizeBytes % 16 == 0,
                  "item size must be positive and 16-byte aligned");
    constexpr int kChunks = kItemSizeBytes / 16;
    constexpr int kChunksPerLane = (kChunks + 31) / 32;

    const int lane_idx = threadIdx.x % 32;
    const auto* src = static_cast<const uint4*>(src_addr);
    auto* dst = static_cast<uint4*>(dst_addr);

    uint4 regs[kChunksPerLane];
    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            regs[i] = ld_global_nc_uint4(src + idx);
        }
    }

    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "l"(dst + idx),
                            "r"(regs[i].x), "r"(regs[i].y),
                            "r"(regs[i].z), "r"(regs[i].w)
                         : "memory");
        }
    }
}

template <>
__device__ __forceinline__ void
transfer_item_warp_v2<1152>(const void* __restrict__ src_addr, void* __restrict__ dst_addr) {
    constexpr int kChunks = 1152 / 16;
    constexpr int kChunksPerLane = (kChunks + 31) / 32;

    const int lane_idx = threadIdx.x % 32;
    const auto* src = static_cast<const uint4*>(src_addr);
    auto* dst = static_cast<uint4*>(dst_addr);

    uint4 regs[kChunksPerLane];
    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            regs[i] = __ldg(src + idx);
        }
    }

    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            const auto& val = regs[i];
            asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "l"(dst + idx),
                            "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                         : "memory");
        }
    }
}

// Helper: prefetch an item's source data into L2 (one 128-byte cache line
// per active lane).  Call this for the *next* item while transferring the
// current one to overlap PCIe latency across items.
template <int kItemSizeBytes>
__device__ __forceinline__ void
prefetch_item_warp(const void* src_addr) {
    static_assert(kItemSizeBytes > 0 && kItemSizeBytes % 16 == 0);
    constexpr int kCacheLines = (kItemSizeBytes + 127) / 128;
    constexpr int kLinesPerLane = (kCacheLines + 31) / 32;
    const int lane_idx = threadIdx.x % 32;
    const auto* src = static_cast<const uint8_t*>(src_addr);

    #pragma unroll
    for (int i = 0; i < kLinesPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kCacheLines) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(src + idx * 128));
        }
    }
}

// Improvement 3: cp.async staging through shared memory.
// Uses the async copy engine to transfer src → smem without occupying
// registers during the PCIe read, then stores from smem → dst.
// Requires kItemSizeBytes of shared memory per warp as staging area.
template <int kItemSizeBytes>
__device__ __forceinline__ void
transfer_item_warp_v3(const void* __restrict__ src_addr, void* __restrict__ dst_addr,
                      void* smem_staging) {
    static_assert(kItemSizeBytes > 0 && kItemSizeBytes % 16 == 0);
    constexpr int kChunks = kItemSizeBytes / 16;
    constexpr int kChunksPerLane = (kChunks + 31) / 32;

    const int lane_idx = threadIdx.x % 32;
    auto* staging = static_cast<uint4*>(smem_staging);
    auto* dst = static_cast<uint4*>(dst_addr);

    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            uint32_t smem_addr = __cvta_generic_to_shared(staging + idx);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                         :: "r"(smem_addr),
                            "l"(static_cast<const uint4*>(src_addr) + idx));
        }
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();

    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            uint4 val = ld_shared(staging + idx);
            asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "l"(dst + idx),
                            "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                         : "memory");
        }
    }
}

// Improvement 4: pipelined cp.async with multiple in-flight commit groups.
// Issues H→smem copies for a window of `kPipeDepth` items back-to-back so
// the PCIe round-trips overlap, then drains in FIFO order.  The caller
// streams (host_ptr, device_ptr) pairs via `issue()` and must call
// `drain()` at the end.  Requires `kPipeDepth * kItemSizeBytes` of shared
// memory as staging (passed in `smem_staging_base`).
template <int kItemSizeBytes, int kPipeDepth>
struct WarpTransferPipe {
    static_assert(kItemSizeBytes > 0 && kItemSizeBytes % 16 == 0);
    static_assert(kPipeDepth > 0);
    static constexpr int kChunks = kItemSizeBytes / 16;
    static constexpr int kChunksPerLane = (kChunks + 31) / 32;

    uint4* smem_base;      // [kPipeDepth][kChunks]
    void* dst_ring[kPipeDepth];
    int stored = 0;        // next item index to issue (monotonic)
    int consumed = 0;      // next item index to store-back (monotonic)

    __device__ __forceinline__ WarpTransferPipe(void* smem_staging_base)
        : smem_base(static_cast<uint4*>(smem_staging_base)) {}

    __device__ __forceinline__ void issue(const void* src_addr, void* dst_addr) {
        // If the pipeline is full, drain the oldest stage first (copy its
        // shared buffer out to global so we can reuse the slot).
        if (stored - consumed == kPipeDepth) {
            drain_one();
        }
        const int slot = stored % kPipeDepth;
        const int lane_idx = threadIdx.x % 32;
        uint4* staging = smem_base + slot * kChunks;

        #pragma unroll
        for (int i = 0; i < kChunksPerLane; i++) {
            const int idx = lane_idx + i * 32;
            if (idx < kChunks) {
                uint32_t smem_addr = __cvta_generic_to_shared(staging + idx);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                             :: "r"(smem_addr),
                                "l"(static_cast<const uint4*>(src_addr) + idx));
            }
        }
        asm volatile("cp.async.commit_group;");
        dst_ring[slot] = dst_addr;
        stored++;
    }

    // Wait for the oldest outstanding group and store it to its device dst.
    __device__ __forceinline__ void drain_one() {
        if (consumed == stored) return;
        const int outstanding = stored - consumed;
        // wait_group N means "wait until at most N groups remain in-flight"
        switch (outstanding - 1) {
            case 0: asm volatile("cp.async.wait_group 0;"); break;
            default: {
                // Fallback: wait for all but the newest (outstanding-1) groups.
                // PTX accepts immediate operand only, so we inline a small switch.
                const int remain = outstanding - 1;
                if (remain == 1) asm volatile("cp.async.wait_group 1;");
                else if (remain == 2) asm volatile("cp.async.wait_group 2;");
                else if (remain == 3) asm volatile("cp.async.wait_group 3;");
                else asm volatile("cp.async.wait_group 4;");
                break;
            }
        }
        __syncwarp();

        const int slot = consumed % kPipeDepth;
        const int lane_idx = threadIdx.x % 32;
        uint4* staging = smem_base + slot * kChunks;
        auto* dst = static_cast<uint4*>(dst_ring[slot]);

        #pragma unroll
        for (int i = 0; i < kChunksPerLane; i++) {
            const int idx = lane_idx + i * 32;
            if (idx < kChunks) {
                uint4 val = ld_shared(staging + idx);
                asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                             :: "l"(dst + idx),
                                "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
                             : "memory");
            }
        }
        // Synchronize the warp before the slot may be reused by a future
        // `issue()` cp.async: without this, under Volta+ Independent Thread
        // Scheduling a lagging lane's `ld_shared` could race with a new
        // async H->smem write from the copy engine into the same staging
        // slot.  `ld_shared` is a synchronous SM instruction, but lanes may
        // reach this point at different cycles, so the sync is mandatory.
        __syncwarp();
        consumed++;
    }

    __device__ __forceinline__ void drain() {
        while (consumed < stored) {
            drain_one();
        }
    }
};

// Direct sparse-prefetch copy path: issue sysmem reads into registers, then
// store to device memory, avoiding the shared-memory cp.async staging pipe.
template <int kItemSizeBytes>
__device__ __forceinline__ void
transfer_item_pair_warp_nc(const void* __restrict__ src_addr_0, void* __restrict__ dst_addr_0,
                           const void* __restrict__ src_addr_1, void* __restrict__ dst_addr_1,
                           const bool copy_second) {
    static_assert(kItemSizeBytes > 0 && kItemSizeBytes % 16 == 0,
                  "item size must be positive and 16-byte aligned");
    constexpr int kChunks = kItemSizeBytes / 16;
    constexpr int kChunksPerLane = (kChunks + 31) / 32;

    const int lane_idx = threadIdx.x % 32;
    const auto* src_0 = static_cast<const uint4*>(src_addr_0);
    auto* dst_0 = static_cast<uint4*>(dst_addr_0);
    const auto* src_1 = static_cast<const uint4*>(src_addr_1);
    auto* dst_1 = static_cast<uint4*>(dst_addr_1);

    uint4 regs_0[kChunksPerLane];
    uint4 regs_1[kChunksPerLane];

    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            uint4 val_0;
            asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(val_0.x), "=r"(val_0.y), "=r"(val_0.z), "=r"(val_0.w)
                         : "l"(src_0 + idx)
                         : "memory");
            regs_0[i] = val_0;
            if (copy_second) {
                uint4 val_1;
                asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(val_1.x), "=r"(val_1.y), "=r"(val_1.z), "=r"(val_1.w)
                             : "l"(src_1 + idx)
                             : "memory");
                regs_1[i] = val_1;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < kChunksPerLane; i++) {
        const int idx = lane_idx + i * 32;
        if (idx < kChunks) {
            const auto& val_0 = regs_0[i];
            asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "l"(dst_0 + idx),
                            "r"(val_0.x), "r"(val_0.y), "r"(val_0.z), "r"(val_0.w)
                         : "memory");
            if (copy_second) {
                const auto& val_1 = regs_1[i];
                asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                             :: "l"(dst_1 + idx),
                                "r"(val_1.x), "r"(val_1.y), "r"(val_1.z), "r"(val_1.w)
                             : "memory");
            }
        }
    }
}


__device__ __forceinline__ void
transfer_item_thread_v0(const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
    const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
    uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
    const int total_chunks = item_size_bytes / sizeof(uint64_t);

#pragma unroll
    for (int j = 0; j < total_chunks; j += 1) {
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src + j) : "memory");
        asm volatile("st.global.cg.b64 [%0],%1;" ::"l"(dst + j), "l"(tmp) : "memory");
    }
}


template <uint32_t kNumBytes>
struct Vectorized {
    static auto zeros() {
        // TODO: add `ulonglong4` for SM100 once `__ldg` support this
        if constexpr (kNumBytes > 0 and kNumBytes % 16 == 0) {
            return make_uint4(0, 0, 0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 8 == 0) {
            return make_uint2(0, 0);
        } else if constexpr (kNumBytes > 0 and kNumBytes % 4 == 0) {
            return 0;
        } else {
            DG_STATIC_ASSERT(kNumBytes > 0 and kNumBytes % 4 == 0, "Invalid vectorization");
        }
    }

    using vec_t = decltype(zeros());
};

} // namespace `deep_gemm`
