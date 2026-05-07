#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <deep_gemm/common/utils.cuh>

#define CUDA_CHECK(expr)                                                                          \
    do {                                                                                          \
        cudaError_t _err = (expr);                                                                \
        if (_err != cudaSuccess) {                                                                \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                         cudaGetErrorString(_err));                                                \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (0)

namespace {

static constexpr int kDefaultItemBytes = 576 * 2;

struct BenchmarkConfig {
    int64_t num_items = 512;
    int item_size_bytes = kDefaultItemBytes;
    int warmup_iters = 200;
    int test_iters = 2000;
    int threads = 256;
};

// =========================== Warp-level kernels ===========================
// Real use case: a single warp processes all items sequentially.

enum WarpVariant {
    W_BASE = 0,
    W_LDST_LDG = 1,
    W_LDST_NC = 2,
    W_PREF_NC = 3,
    W_CPA = 4
};

template <WarpVariant V, int kItemBytes>
__global__ void transfer_warp_kernel(const uint8_t* src, uint8_t* dst,
                                     int item_size_bytes, int64_t num_items) {
    extern __shared__ uint8_t smem[];

    for (int64_t i = 0; i < num_items; i++) {
        auto* s = src + i * item_size_bytes;
        auto* d = dst + i * item_size_bytes;

        if constexpr (V == W_BASE) {
            deep_gemm::transfer_item_warp<kItemBytes>(s, d);
        } else if constexpr (V == W_LDST_LDG) {
            deep_gemm::transfer_item_warp_v2_ldg<kItemBytes>(s, d);
        } else if constexpr (V == W_LDST_NC) {
            deep_gemm::transfer_item_warp_v2<kItemBytes>(s, d);
        } else if constexpr (V == W_PREF_NC) {
            if (i + 1 < num_items) {
                deep_gemm::prefetch_item_warp<kItemBytes>(
                    src + (i + 1) * item_size_bytes);
            }
            deep_gemm::transfer_item_warp_v2<kItemBytes>(s, d);
        } else if constexpr (V == W_CPA) {
            deep_gemm::transfer_item_warp_v3<kItemBytes>(s, d, smem);
        }
    }
}

// ========================== Benchmark helpers =============================

template <WarpVariant V, int kItemBytes>
double benchmark_warp(const uint8_t* src, uint8_t* dst, const BenchmarkConfig& cfg) {
    const int smem_bytes = (V == W_CPA) ? kItemBytes : 0;

    CUDA_CHECK(cudaMemset(dst, 0xA5, cfg.num_items * cfg.item_size_bytes));

    for (int i = 0; i < cfg.warmup_iters; ++i) {
        transfer_warp_kernel<V, kItemBytes><<<1, 32, smem_bytes>>>(
            src, dst, cfg.item_size_bytes, cfg.num_items);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < cfg.test_iters; ++i) {
        transfer_warp_kernel<V, kItemBytes><<<1, 32, smem_bytes>>>(
            src, dst, cfg.item_size_bytes, cfg.num_items);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<double>(elapsed_ms) * 1e-3 / static_cast<double>(cfg.test_iters);
}

bool sanity_check(const uint8_t* src, const uint8_t* dst, const BenchmarkConfig& cfg,
                  const char* name) {
    constexpr int64_t kSampleItems = 16;
    const int64_t sample_items = cfg.num_items < kSampleItems ? cfg.num_items : kSampleItems;
    const int64_t sample_bytes = sample_items * cfg.item_size_bytes;

    std::vector<uint8_t> src_host(sample_bytes);
    std::vector<uint8_t> dst_host(sample_bytes);
    std::memcpy(src_host.data(), src, static_cast<size_t>(sample_bytes));
    CUDA_CHECK(cudaMemcpy(dst_host.data(), dst, static_cast<size_t>(sample_bytes),
                          cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int64_t i = 0; i < sample_bytes; ++i) {
        if (src_host[i] != dst_host[i]) {
            ++mismatches;
        }
    }

    if (mismatches > 0) {
        std::printf("  [%s] FAIL: %d/%lld byte mismatches\n", name,
                    mismatches, static_cast<long long>(sample_bytes));
    }
    return mismatches == 0;
}

void parse_args(int argc, char** argv, BenchmarkConfig* cfg) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--items") == 0 && i + 1 < argc) {
            cfg->num_items = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--item-size") == 0 && i + 1 < argc) {
            cfg->item_size_bytes = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg->warmup_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg->test_iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg->threads = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr,
                         "Unknown or incomplete argument: %s\n"
                         "Usage: %s [--items N] [--item-size B] [--warmup N] [--iters N] [--threads N]\n",
                         argv[i], argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
}

template <int kItemBytes>
void run_warp_sweep(const uint8_t* src_managed, uint8_t* dst_device,
                    const BenchmarkConfig& base_cfg) {
    const int64_t item_counts[] = {100, 200, 500, 1000};
    constexpr int n_counts = sizeof(item_counts) / sizeof(item_counts[0]);

    std::printf("\n=== Warp-level benchmark sweep (1 warp, item_size=%d) ===\n", kItemBytes);
    std::printf("  %-8s  %-23s  %-27s  %-27s  %-27s  %-23s\n",
                "items", "baseline", "ldg_split", "nc_split", "nc+prefetch", "cpasync");
    std::printf("  %-8s  %-23s  %-27s  %-27s  %-27s  %-23s\n",
                "-----", "--------", "---------", "--------", "-----------", "-------");

    for (int ci = 0; ci < n_counts; ci++) {
        BenchmarkConfig cfg = base_cfg;
        cfg.num_items = item_counts[ci];
        if (cfg.num_items * cfg.item_size_bytes >
            base_cfg.num_items * base_cfg.item_size_bytes) {
            continue;
        }

        const double tw_base = benchmark_warp<W_BASE, kItemBytes>(src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_baseline");

        const double tw_ldg = benchmark_warp<W_LDST_LDG, kItemBytes>(src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_ldg_split");

        const double tw_nc = benchmark_warp<W_LDST_NC, kItemBytes>(src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_nc_split");

        const double tw_pref = benchmark_warp<W_PREF_NC, kItemBytes>(src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_nc_split+prefetch");

        const double tw_cpa  = benchmark_warp<W_CPA, kItemBytes>(src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_cpasync");

        const double bytes = static_cast<double>(cfg.num_items) * cfg.item_size_bytes;
        const double bw_base = bytes / tw_base / 1e9;
        const double bw_ldg = bytes / tw_ldg / 1e9;
        const double bw_nc = bytes / tw_nc / 1e9;
        const double bw_pref = bytes / tw_pref / 1e9;
        const double bw_cpa = bytes / tw_cpa / 1e9;

        std::printf("  %-8lld  %7.1f us %5.2f GB/s   %7.1f us %5.2f GB/s %4.2fx   %7.1f us %5.2f GB/s %4.2fx   %7.1f us %5.2f GB/s %4.2fx   %7.1f us %5.2f GB/s %4.2fx\n",
                    static_cast<long long>(cfg.num_items),
                    tw_base * 1e6, bw_base,
                    tw_ldg * 1e6, bw_ldg, tw_base / tw_ldg,
                    tw_nc * 1e6, bw_nc, tw_base / tw_nc,
                    tw_pref * 1e6, bw_pref, tw_base / tw_pref,
                    tw_cpa  * 1e6, bw_cpa, tw_base / tw_cpa);
    }
}

}  // namespace

int main(int argc, char** argv) {
    BenchmarkConfig cfg;
    parse_args(argc, argv, &cfg);

    if (cfg.num_items <= 0 || cfg.item_size_bytes <= 0 || cfg.warmup_iters < 0 ||
        cfg.test_iters <= 0 || cfg.threads <= 0) {
        std::fprintf(stderr, "Invalid benchmark configuration.\n");
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    const int64_t max_items = 1000;
    const int64_t alloc_items = std::max(cfg.num_items, max_items);
    const int64_t total_bytes = alloc_items * cfg.item_size_bytes;
    uint8_t* src_managed = nullptr;
    uint8_t* dst_device = nullptr;
    CUDA_CHECK(cudaMallocManaged(&src_managed, static_cast<size_t>(total_bytes)));
    CUDA_CHECK(cudaMalloc(&dst_device, static_cast<size_t>(total_bytes)));

    for (int64_t i = 0; i < total_bytes; ++i) {
        src_managed[i] = static_cast<uint8_t>(i & 0xFF);
    }
    CUDA_CHECK(cudaMemAdvise(src_managed, static_cast<size_t>(total_bytes),
                             cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(src_managed, static_cast<size_t>(total_bytes),
                             cudaMemAdviseSetAccessedBy, device));
    CUDA_CHECK(cudaMemPrefetchAsync(src_managed, static_cast<size_t>(total_bytes), cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Allocated: %lld items, item_size=%d bytes, total=%.2f MiB\n",
                static_cast<long long>(alloc_items), cfg.item_size_bytes,
                static_cast<double>(total_bytes) / (1024.0 * 1024.0));

    // =================== Single-config warp benchmark ====================
    if (cfg.item_size_bytes == kDefaultItemBytes) {
        std::printf("\n=== Warp-level benchmark (1 warp, items=%lld, item_size=%d) ===\n",
                    static_cast<long long>(cfg.num_items), kDefaultItemBytes);

        const double tw_base = benchmark_warp<W_BASE, kDefaultItemBytes>(
            src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_baseline");

        const double tw_ldg = benchmark_warp<W_LDST_LDG, kDefaultItemBytes>(
            src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_ldg_split");

        const double tw_nc = benchmark_warp<W_LDST_NC, kDefaultItemBytes>(
            src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_nc_split");

        const double tw_pref = benchmark_warp<W_PREF_NC, kDefaultItemBytes>(
            src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_nc_split+prefetch");

        const double tw_cpa = benchmark_warp<W_CPA, kDefaultItemBytes>(
            src_managed, dst_device, cfg);
        sanity_check(src_managed, dst_device, cfg, "warp_cpasync");

        const double bytes = static_cast<double>(cfg.num_items) * cfg.item_size_bytes;
        std::printf("  %-30s: %10.1f us, %8.2f GB/s\n", "warp_baseline",
                    tw_base * 1e6, bytes / tw_base / 1e9);
        std::printf("  %-30s: %10.1f us, %8.2f GB/s  (%.2fx)\n", "warp_ldg_split",
                    tw_ldg * 1e6, bytes / tw_ldg / 1e9, tw_base / tw_ldg);
        std::printf("  %-30s: %10.1f us, %8.2f GB/s  (%.2fx)\n", "warp_nc_split",
                    tw_nc * 1e6, bytes / tw_nc / 1e9, tw_base / tw_nc);
        std::printf("  %-30s: %10.1f us, %8.2f GB/s  (%.2fx)\n", "warp_nc_split+prefetch",
                    tw_pref * 1e6, bytes / tw_pref / 1e9, tw_base / tw_pref);
        std::printf("  %-30s: %10.1f us, %8.2f GB/s  (%.2fx)\n", "warp_cpasync",
                    tw_cpa * 1e6, bytes / tw_cpa / 1e9, tw_base / tw_cpa);

        // ===================== Sweep across item counts ======================
        BenchmarkConfig sweep_cfg = cfg;
        sweep_cfg.num_items = alloc_items;
        run_warp_sweep<kDefaultItemBytes>(src_managed, dst_device, sweep_cfg);
    } else {
        std::printf("Skipping warp benchmarks (compiled for item_size=%d, got %d)\n",
                    kDefaultItemBytes, cfg.item_size_bytes);
    }

    CUDA_CHECK(cudaFree(dst_device));
    CUDA_CHECK(cudaFree(src_managed));
    return EXIT_SUCCESS;
}
// /usr/local/cuda/bin/nvcc -std=c++17 -O3 -arch=sm_90 tests/bench_transfer_item_thread.cu -Ideep_gemm/include -Ithird-party/cutlass/include -o /tmp/bench_transfer_item_thread
