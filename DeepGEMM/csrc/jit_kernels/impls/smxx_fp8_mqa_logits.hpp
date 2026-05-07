#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90FP8MQALogitsRuntime final: public LaunchRuntime<SM90FP8MQALogitsRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int stride_kv;
        int num_heads, head_dim;
        int num_q_stages;
        int num_kv_stages;

        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        float* logits;
        float softmax_scale;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto& arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {}
    >);
}};
)", arch, arch,
    args.num_heads, args.head_dim,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages,
    args.num_specialized_threads, args.num_math_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv, static_cast<int64_t>(args.stride_kv),
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

class SM90FP8MQALogitsFuseTopkRuntime final: public LaunchRuntime<SM90FP8MQALogitsFuseTopkRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int stride_kv;
        int num_heads, head_dim;
        int num_q_stages;
        int num_kv_stages;
        int num_topk_stages;

        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        float* logits;
        int* topk_index;
        float softmax_scale;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;

        int num_specialized_threads;
        int num_math_threads;
        int num_topk_threads;

        int smem_size_for_topk_input;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto& arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_mqa_logits_fuse_topk<
        {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {}, 
        {}
    >);
}};
)", arch, arch,
    args.num_heads, args.head_dim,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages, args.num_topk_stages,
    args.num_specialized_threads, args.num_math_threads, args.num_topk_threads,
    args.smem_size_for_topk_input);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv, static_cast<int64_t>(args.stride_kv),
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.topk_index,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

class SM90FP8MQALogitsFusePrefetchRuntime final: public LaunchRuntime<SM90FP8MQALogitsFusePrefetchRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int stride_kv;
        int num_heads, head_dim;
        int num_q_stages;
        int num_kv_stages;
        int num_topk_stages;

        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        float* logits;
        float softmax_scale;

        int* page_table_1;
        uint64_t page_table_1_stride;
        int* extend_seq_lens;
        int* extend_seq_to_req;

        at::BFloat16* device_pool_buf;
        at::BFloat16* host_pool_buf;
        int* device_pool_loc_alloc_buf;
        int* device_pool_loc_small_priority;
        int64_t* device_token_to_host;
        int* host_token_to_device;
        uint32_t* recall_counter;
        float* extend_logits_offsets;

        uint32_t device_pool_size_m1;
        int mla_head_dim;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;

        int num_specialized_threads;
        int num_math_threads;
        int num_topk_threads;

        int smem_size_for_topk_input;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto& arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_mqa_logits_fuse_prefetch<
        {}, {}, {},
        {}, {},
        {}, {}, {},
        {}, {}, {}, 
        {}
    >);
}};
)", arch, arch,
    args.num_heads, args.head_dim, args.mla_head_dim,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages, args.num_topk_stages,
    args.num_specialized_threads, args.num_math_threads, args.num_topk_threads,
    args.smem_size_for_topk_input);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv, static_cast<int64_t>(args.stride_kv),
            args.page_table_1_stride,
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.page_table_1, args.extend_seq_lens, args.extend_seq_to_req,
            args.device_pool_buf,
            args.host_pool_buf,
            args.device_pool_loc_alloc_buf,
            args.device_pool_loc_small_priority,
            args.device_token_to_host, args.host_token_to_device,
            args.recall_counter, args.extend_logits_offsets,
            args.device_pool_size_m1,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
};

static void smxx_fp8_mqa_logits(const torch::Tensor& q,
                                const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                const torch::Tensor& weights,
                                const torch::Tensor& cu_seq_len_k_start,
                                const torch::Tensor& cu_seq_len_k_end,
                                const torch::Tensor& logits,
                                const int& seq_len, const int& seq_len_kv, const int& stride_kv,
                                const int& num_heads, const int& head_dim,
                                const int& seq_len_alignment) {
    constexpr int block_qh = 128;
    constexpr int block_kv = 256;
    constexpr int num_specialized_threads = 128;
    constexpr int num_math_threads = 512;
    constexpr int num_q_stages = 3, num_kv_stages = 3;
    const int block_q = block_qh / num_heads;
    DG_HOST_ASSERT(block_qh % num_heads == 0);
    DG_HOST_ASSERT(seq_len_alignment % block_q == 0);

    // Construct TMAs
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto& tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                                head_dim, block_qh, head_dim, head_dim);
    const auto& tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                 head_dim, block_kv, head_dim, head_dim);
    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    const auto& tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                        get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                        1, block_kv, 1, 0, 0);
    const auto& tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                      num_heads, block_q, num_heads, 0);

    // Calculate shared memory size
    int smem_size = 0;
    const int smem_q_size_per_stage = block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    const int smem_weight_size_per_stage = block_q * num_heads * static_cast<int>(weights.element_size());
    const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv.element_size());
    const int kv_scale_size_per_stage = block_kv * static_cast<int>(kv_scales.element_size());
    smem_size += num_q_stages * smem_q_size_per_stage;
    smem_size += num_kv_stages * smem_kv_size_per_stage;
    smem_size += num_q_stages * smem_weight_size_per_stage;
    smem_size += num_kv_stages * kv_scale_size_per_stage;
    smem_size += (num_q_stages * 2 + num_kv_stages * 2 + (num_math_threads / 128) * 2) * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);

    // Launch
    const SM90FP8MQALogitsRuntime::Args& args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .stride_kv = stride_kv,
        .num_heads = num_heads, .head_dim = head_dim,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr<float>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads,
                                  smem_size)
    };
    const auto& code = SM90FP8MQALogitsRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_fp8_mqa_logits", code);
    SM90FP8MQALogitsRuntime::launch(runtime, args);
}

static void smxx_fp8_mqa_logits_fuse_topk(const torch::Tensor& q,
                                          const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                          const torch::Tensor& weights,
                                          const torch::Tensor& cu_seq_len_k_start,
                                          const torch::Tensor& cu_seq_len_k_end,
                                          const torch::Tensor& logits,
                                          const torch::Tensor& topk_index,
                                          const int& seq_len, const int& seq_len_kv, const int& stride_kv,
                                          const int& num_heads, const int& head_dim,
                                          const int& seq_len_alignment) {
    constexpr int block_qh = 128;
    constexpr int block_kv = 128;
    constexpr int num_specialized_threads = 128;
    constexpr int num_math_threads = 256;
    // constexpr int num_topk_threads = 128;
    constexpr int num_topk_threads = 256;
    constexpr int smem_size_for_topk_input = 16 * 1024; // 16KB, to make [2, 2K] i32 buffer

    constexpr int num_q_stages = 3, num_kv_stages = 3, num_topk_stages = 3;
    const int block_q = block_qh / num_heads;
    DG_HOST_ASSERT(block_qh % num_heads == 0);
    DG_HOST_ASSERT(seq_len_alignment % block_q == 0);

    // Construct TMAs
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto& tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                                head_dim, block_qh, head_dim, head_dim);
    const auto& tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                 head_dim, block_kv, head_dim, head_dim);
    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    const auto& tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                        get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                        1, block_kv, 1, 0, 0);
    const auto& tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                      num_heads, block_q, num_heads, 0);

    // Calculate shared memory size
    int smem_size = 0;
    smem_size += smem_size_for_topk_input;
    const int smem_q_size_per_stage = block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    const int smem_weight_size_per_stage = block_q * num_heads * static_cast<int>(weights.element_size());
    const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv.element_size());
    const int kv_scale_size_per_stage = block_kv * static_cast<int>(kv_scales.element_size());
    const int logits_size_per_stage = block_q * block_kv * static_cast<int>(logits.element_size());
    smem_size += num_q_stages * smem_q_size_per_stage;
    smem_size += num_kv_stages * smem_kv_size_per_stage;
    smem_size += num_q_stages * smem_weight_size_per_stage;
    smem_size += num_kv_stages * kv_scale_size_per_stage;
    smem_size += num_topk_stages * logits_size_per_stage;
    smem_size += (num_q_stages * 2 + num_kv_stages * 2 + num_topk_stages * 2 + (num_math_threads / 128) * 2) * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);

    // Launch
    const SM90FP8MQALogitsFuseTopkRuntime::Args& args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .stride_kv = stride_kv,
        .num_heads = num_heads, .head_dim = head_dim,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .num_topk_stages = num_topk_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr<float>(),
        .topk_index = topk_index.data_ptr<int>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .num_topk_threads = num_topk_threads,
        .smem_size_for_topk_input = smem_size_for_topk_input,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads + num_topk_threads,
                                  smem_size)
    };
    const auto& code = SM90FP8MQALogitsFuseTopkRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_fp8_mqa_logits_fuse_topk", code);
    SM90FP8MQALogitsFuseTopkRuntime::launch(runtime, args);
}

static void smxx_fp8_mqa_logits_fuse_prefetch(const torch::Tensor& q,
                                              const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                              const torch::Tensor& weights,
                                              const torch::Tensor& cu_seq_len_k_start,
                                              const torch::Tensor& cu_seq_len_k_end,
                                              const torch::Tensor& logits,
                                              const torch::Tensor& page_table_1,
                                              const torch::Tensor& extend_seq_lens,
                                              const torch::Tensor& extend_seq_to_req,
                                              torch::Tensor& device_pool_buf,
                                              torch::Tensor& host_pool_buf,
                                              torch::Tensor& device_pool_loc_alloc_buf,
                                              torch::Tensor& device_pool_priority,
                                              const torch::Tensor& device_pool_loc_small_priority,
                                              torch::Tensor& device_token_to_host,
                                              torch::Tensor& host_token_to_device,
                                              torch::Tensor& recall_counter,
                                              const torch::Tensor& extend_logits_offsets,
                                              const uint32_t& device_pool_size_m1,
                                              const int& mla_head_dim,
                                              const int& seq_len, const int& seq_len_kv, const int& stride_kv,
                                              const uint64_t& page_table_1_stride,
                                              const int& num_heads, const int& head_dim,
                                              const int& seq_len_alignment) {
    constexpr int block_qh = 128;
    constexpr int block_kv = 128;
    constexpr int num_specialized_threads = 128;
    constexpr int num_math_threads = 256;
    // constexpr int num_topk_threads = 128;
    constexpr int num_topk_threads = 256;
    constexpr int smem_size_for_topk_input = 16 * 1024; // 16KB, to make [2, 2K] i32 buffer

    constexpr int num_q_stages = 3, num_kv_stages = 3;
    const int block_q = block_qh / num_heads;
    DG_HOST_ASSERT(block_qh % num_heads == 0);
    DG_HOST_ASSERT(seq_len_alignment % block_q == 0);

    // Construct TMAs
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
    const auto& tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                                head_dim, block_qh, head_dim, head_dim);
    const auto& tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                 head_dim, block_kv, head_dim, head_dim);
    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    const auto& tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                        get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                        1, block_kv, 1, 0, 0);
    const auto& tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                      num_heads, block_q, num_heads, 0);

    // Calculate shared memory size
    const int smem_q_size_per_stage = block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    const int smem_weight_size_per_stage = block_q * num_heads * static_cast<int>(weights.element_size());
    const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv.element_size());
    const int kv_scale_size_per_stage = block_kv * static_cast<int>(kv_scales.element_size());
    const int logits_size_per_stage = block_q * block_kv * static_cast<int>(logits.element_size());

    // The prefetch top-k branch also has statically allocated shared memory.
    // Reserve it before maximizing the dynamic logits pipeline stages.
    constexpr int radix = 256;
    const int static_topk_smem_size = align(
        align(2 * block_q * (radix + 128) * static_cast<int>(sizeof(int)), 128) +
        align(static_cast<int>(sizeof(int)), 128) +
        align(static_cast<int>(sizeof(int)), 128) +
        2 * num_topk_threads * static_cast<int>(sizeof(int)),
        1024);
    const int fixed_dynamic_smem_size =
        smem_size_for_topk_input +
        num_q_stages * smem_q_size_per_stage +
        num_kv_stages * smem_kv_size_per_stage +
        num_q_stages * smem_weight_size_per_stage +
        num_kv_stages * kv_scale_size_per_stage +
        (num_q_stages * 2 + num_kv_stages * 2 + (num_math_threads / 128) * 2) * 8 +
        4;
    const int topk_dynamic_smem_size_per_stage = logits_size_per_stage + 2 * 8;
    DG_HOST_ASSERT(SM90ArchSpec::smem_capacity > fixed_dynamic_smem_size + static_topk_smem_size);
    const int num_topk_stages =
        (SM90ArchSpec::smem_capacity - static_topk_smem_size - fixed_dynamic_smem_size) /
        topk_dynamic_smem_size_per_stage;
    DG_HOST_ASSERT(num_topk_stages >= 3);

    const int smem_size = fixed_dynamic_smem_size + num_topk_stages * topk_dynamic_smem_size_per_stage;
    DG_HOST_ASSERT(smem_size + static_topk_smem_size <= SM90ArchSpec::smem_capacity);

    // Launch
    const SM90FP8MQALogitsFusePrefetchRuntime::Args& args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .stride_kv = stride_kv,
        .num_heads = num_heads, .head_dim = head_dim,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .num_topk_stages = num_topk_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr<float>(),
        .page_table_1 = page_table_1.data_ptr<int>(),
        .page_table_1_stride = page_table_1_stride,
        .extend_seq_lens = extend_seq_lens.data_ptr<int>(),
        .extend_seq_to_req = extend_seq_to_req.data_ptr<int>(),
        .device_pool_buf = device_pool_buf.data_ptr<at::BFloat16>(),
        .host_pool_buf = host_pool_buf.data_ptr<at::BFloat16>(),
        .device_pool_loc_alloc_buf = device_pool_loc_alloc_buf.data_ptr<int>(),
        .device_pool_loc_small_priority = device_pool_loc_small_priority.data_ptr<int>(),
        .device_token_to_host = device_token_to_host.data_ptr<int64_t>(),
        .host_token_to_device = host_token_to_device.data_ptr<int>(),
        .recall_counter = recall_counter.data_ptr<uint32_t>(),
        .extend_logits_offsets = extend_logits_offsets.data_ptr<float>(),
        .device_pool_size_m1 = device_pool_size_m1,
        .mla_head_dim = mla_head_dim,
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .num_topk_threads = num_topk_threads,
        .smem_size_for_topk_input = smem_size_for_topk_input,
        .launch_args = LaunchArgs(device_runtime->get_num_sms(),
                                  num_specialized_threads + num_math_threads + num_topk_threads,
                                  smem_size)
    };
    const auto& code = SM90FP8MQALogitsFusePrefetchRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_fp8_mqa_logits_fuse_prefetch", code);
    SM90FP8MQALogitsFusePrefetchRuntime::launch(runtime, args);
}

} // namespace deep_gemm
