#pragma once

#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/smxx_fp8_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_fp8_paged_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_clean_logits.hpp"

#include "layout.hpp"

namespace deep_gemm::attention {

static void fp8_gemm_nt_skip_head_mid(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const std::tuple<int, int, int> &head_splits,
                                      std::optional<std::tuple<int, int, int>> recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto& major_a = get_major_type_ab(a.first);
    const auto& major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto& [m , k ] = get_shape<2>(a.first);
    const auto& [n , k_] = get_shape<2>(b.first);
    const auto& [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Check head splits and N
    const auto& [left, mid, right] = head_splits;
    DG_HOST_ASSERT(n % (left + right) == 0 and n_ == n + n / (left + right) * mid);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    if (not recipe.has_value())
        recipe = get_default_recipe(a.second.scalar_type(), b.second.scalar_type());
    DG_HOST_ASSERT(recipe.value() == std::make_tuple(1, 1, 128) or recipe.value() == std::make_tuple(1, 128, 128));
    const auto& sfa = layout::transform_sf_into_required_layout(a.second, m, k, recipe.value(), std::nullopt,  true, disable_ue8m0_cast);
    const auto& sfb = layout::transform_sf_into_required_layout(b.second, n, k, recipe.value(), std::nullopt, false, disable_ue8m0_cast);

    // Dispatch into different implements
    const auto& arch_major = device_runtime->get_arch_major();
    const auto& epilogue_type = fmt::format("EpilogueHeadSplits<{}, {}, {}>", left, mid, right);
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat and std::get<1>(recipe.value()) != 1) {
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k, major_a, major_b, compiled_dims, epilogue_type);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        sm100_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k, major_a, major_b, compiled_dims, epilogue_type);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kFloat) {
        sm100_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k, major_a, major_b, compiled_dims, epilogue_type);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static torch::Tensor fp8_mqa_logits(const torch::Tensor& q,
                                    const std::pair<torch::Tensor, torch::Tensor>& kv,
                                    const torch::Tensor& weights,
                                    const torch::Tensor& cu_seq_len_k_start,
                                    const torch::Tensor& cu_seq_len_k_end,
                                    const bool& clean_logits) {
    const auto& [seq_len, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_kv, head_dim_] = get_shape<2>(kv.first);
    const auto& [seq_len_, num_heads_] = get_shape<2>(weights);
    const auto& [seq_len_kv_] = get_shape<1>(kv.second);

    DG_HOST_ASSERT(seq_len == seq_len_);
    DG_HOST_ASSERT(num_heads == num_heads_ and head_dim == head_dim_);
    DG_HOST_ASSERT(seq_len_kv == seq_len_kv_);
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);

    DG_HOST_ASSERT(q.is_contiguous() and kv.first.is_contiguous());
    DG_HOST_ASSERT(kv.second.is_contiguous());
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());

    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    constexpr int seq_len_alignment = 4;
    constexpr int block_kv = 256;
    const auto aligned_seq_len = align(seq_len, seq_len_alignment);
    const auto aligned_seq_len_kv = align(seq_len_kv + block_kv, 4);
    auto logits = torch::empty({aligned_seq_len, aligned_seq_len_kv}, q.options().dtype(torch::kFloat));
    logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len_kv)});

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_fp8_mqa_logits(q, kv.first, kv.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits,
                            seq_len, seq_len_kv, aligned_seq_len_kv, num_heads, head_dim, seq_len_alignment);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, cu_seq_len_k_start, cu_seq_len_k_end, 1, seq_len, seq_len_kv, aligned_seq_len_kv);
    return logits;
}

static std::pair<torch::Tensor, torch::Tensor> fp8_mqa_logits_fuse_topk(
    const torch::Tensor& q,
    const std::pair<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const bool& clean_logits
) {
    const auto& [seq_len, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_kv, head_dim_] = get_shape<2>(kv.first);
    const auto& [seq_len_, num_heads_] = get_shape<2>(weights);
    const auto& [seq_len_kv_] = get_shape<1>(kv.second);

    DG_HOST_ASSERT(seq_len == seq_len_);
    DG_HOST_ASSERT(num_heads == num_heads_ and head_dim == head_dim_);
    DG_HOST_ASSERT(seq_len_kv == seq_len_kv_);
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);

    DG_HOST_ASSERT(q.is_contiguous() and kv.first.is_contiguous());
    DG_HOST_ASSERT(kv.second.is_contiguous());
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());

    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    constexpr int seq_len_alignment = 4;
    constexpr int block_kv = 128;
    const auto aligned_seq_len = align(seq_len, seq_len_alignment);
    const auto aligned_seq_len_kv = align(seq_len_kv + block_kv, 4);
    auto logits = torch::empty({aligned_seq_len, aligned_seq_len_kv}, q.options().dtype(torch::kFloat));
    logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len_kv)});
    constexpr int TOPK = 2048;
    auto topk_index = torch::empty({aligned_seq_len, TOPK}, q.options().dtype(torch::kInt));
    topk_index = topk_index.index({torch::indexing::Slice(0, seq_len)});

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        smxx_fp8_mqa_logits_fuse_topk(q, kv.first, kv.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, 
                                      logits, topk_index,
                                      seq_len, seq_len_kv, aligned_seq_len_kv, num_heads, head_dim, seq_len_alignment);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, cu_seq_len_k_start, cu_seq_len_k_end, 1, seq_len, seq_len_kv, aligned_seq_len_kv);
    return {logits, topk_index};
}

static torch::Tensor fp8_mqa_logits_fuse_prefetch(
    const torch::Tensor& q,
    const std::pair<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    // recall params
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
    const bool& clean_logits
) {
    const auto& [seq_len, num_heads, head_dim] = get_shape<3>(q);
    const auto& [seq_len_kv, head_dim_] = get_shape<2>(kv.first);
    const auto& [seq_len_, num_heads_] = get_shape<2>(weights);
    const auto& [seq_len_kv_] = get_shape<1>(kv.second);

    get_shape<2>(page_table_1);
    const auto& [device_pool_size, mla_num_heads_kv, mla_head_dim] = get_shape<3>(device_pool_buf);
    const auto& [host_pool_size, mla_num_heads_kv_, mla_head_dim_] = get_shape<3>(host_pool_buf);
    const auto& [device_pool_size_m1] = get_shape<1>(device_pool_loc_alloc_buf);

    DG_HOST_ASSERT(seq_len == seq_len_);
    DG_HOST_ASSERT(num_heads == num_heads_ and head_dim == head_dim_);
    DG_HOST_ASSERT(seq_len_kv == seq_len_kv_);
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);

    DG_HOST_ASSERT(q.is_contiguous() and kv.first.is_contiguous());
    DG_HOST_ASSERT(kv.second.is_contiguous());
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());

    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(kv.second.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    DG_HOST_ASSERT(page_table_1.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(extend_seq_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(extend_seq_to_req.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(device_pool_buf.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(host_pool_buf.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(device_pool_loc_alloc_buf.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(device_pool_priority.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(device_pool_loc_small_priority.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(device_token_to_host.scalar_type() == torch::kInt64);
    DG_HOST_ASSERT(host_token_to_device.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(recall_counter.scalar_type() == torch::kUInt32);
    DG_HOST_ASSERT(extend_logits_offsets.scalar_type() == torch::kFloat32);

    DG_HOST_ASSERT(device_pool_size == device_pool_size_m1 + 1);
    DG_HOST_ASSERT(mla_num_heads_kv == 1 and mla_num_heads_kv_ == 1);
    DG_HOST_ASSERT(mla_head_dim_ == mla_head_dim);

    constexpr int seq_len_alignment = 4;
    constexpr int block_kv = 128;
    const auto aligned_seq_len = align(seq_len, seq_len_alignment);
    const auto aligned_seq_len_kv = align(seq_len_kv + block_kv, 4);
    auto logits = torch::empty({aligned_seq_len, aligned_seq_len_kv}, q.options().dtype(torch::kFloat));
    logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len_kv)});

    const auto& page_table_1_stride = page_table_1.stride(0);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9) {
        smxx_fp8_mqa_logits_fuse_prefetch(q, kv.first, kv.second, weights, cu_seq_len_k_start, cu_seq_len_k_end, 
                                          logits, 
                                          page_table_1, extend_seq_lens, extend_seq_to_req,
                                          device_pool_buf, host_pool_buf, device_pool_loc_alloc_buf, 
                                          device_pool_priority, device_pool_loc_small_priority,
                                          device_token_to_host, host_token_to_device,
                                          recall_counter, extend_logits_offsets,
                                          device_pool_size_m1, mla_head_dim,
                                          seq_len, seq_len_kv, aligned_seq_len_kv, 
                                          page_table_1_stride,
                                          num_heads, head_dim, seq_len_alignment);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, cu_seq_len_k_start, cu_seq_len_k_end, 1, seq_len, seq_len_kv, aligned_seq_len_kv);
    return logits;
}

static torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms) {
    const auto& [batch_size] = get_shape<1>(context_lens);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(context_lens.is_contiguous());

    auto schedule_metadata = torch::empty({num_sms + 1, 2}, context_lens.options());

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, block_kv, num_sms);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    return schedule_metadata;
}

static torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q,
                                          const torch::Tensor& fused_kv_cache,
                                          const torch::Tensor& weights,
                                          const torch::Tensor& context_lens,
                                          const torch::Tensor& block_table,
                                          const torch::Tensor& schedule_meta,
                                          const int& max_context_len,
                                          const bool& clean_logits) {
    const auto& [batch_size, next_n, num_heads, head_dim] = get_shape<4>(q);
    const auto& [num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf] = get_shape<4>(fused_kv_cache);
    const auto& [batch_size_] = get_shape<1>(context_lens);
    const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
    const auto& [batch_size__, max_block_len] = get_shape<2>(block_table);
    const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta);
    const auto& num_sms = device_runtime->get_num_sms();
    const auto& kv_cache_stride_bytes = fused_kv_cache.stride(0);
    const auto& block_table_stride = block_table.stride(0);

    DG_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    DG_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
    DG_HOST_ASSERT(num_heads == num_heads_ and num_heads_kv == 1);
    DG_HOST_ASSERT(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
    DG_HOST_ASSERT(schedule_meta_size == num_sms + 1 and meta_info_size == 2);

    DG_HOST_ASSERT(next_n == 1 or next_n == 2);
    DG_HOST_ASSERT(block_kv == 64);

    DG_HOST_ASSERT(q.is_contiguous());
    DG_HOST_ASSERT(kv_cache_stride_bytes % sizeof(float) == 0);
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf);
    DG_HOST_ASSERT(fused_kv_cache.stride(2) == head_dim_with_sf);
    DG_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(context_lens.is_contiguous());
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(schedule_meta.is_contiguous());

    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt);

    // Derive FP8 values and SF tensor from KV cache
    const auto& kv_cache = torch::from_blob(
        fused_kv_cache.data_ptr(),
        {num_kv_blocks, block_kv, head_dim},
        {kv_cache_stride_bytes, head_dim, 1},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn)
    );
    const auto& kv_cache_scales = torch::from_blob(
        fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim,
        {num_kv_blocks, block_kv},
        {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
        torch::TensorOptions().dtype(torch::kFloat32)
    );

    // Allocate output
    constexpr int num_math_warp_groups = 4;
    const auto& aligned_max_context_len = align(max_context_len, num_math_warp_groups * block_kv);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q.options().dtype(torch::kFloat));
    logits = logits.slice(-1, 0, max_context_len);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_fp8_paged_mqa_logits(q, kv_cache, kv_cache_scales, weights, context_lens, logits, block_table, schedule_meta,
                                  batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv,
                                  kv_cache_stride_bytes, aligned_max_context_len, block_table_stride, num_sms, num_math_warp_groups);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, std::nullopt, context_lens, next_n, batch_size * next_n, max_context_len, aligned_max_context_len);
    return logits;
}

static torch::Tensor fp8_paged_mqa_logits_fused_v2(const torch::Tensor& q,
                                                const torch::Tensor& fused_kv_cache,
                                                const torch::Tensor& weights,
                                                const torch::Tensor& context_lens,
                                                const torch::Tensor& block_table,
                                                const torch::Tensor& schedule_meta,
                                                const int& max_context_len,
                                                // recall params
                                                const torch::Tensor& page_table_1,
                                                torch::Tensor& device_pool_buf,
                                                torch::Tensor& host_pool_buf,
                                                torch::Tensor& prefetch_host_loc,
                                                torch::Tensor& prefetch_kv_buf,
                                                torch::Tensor& host_token_to_device,
                                                torch::Tensor& query_recall_counter,
                                                const torch::Tensor& decode_topk_logits,
                                                const bool& clean_logits) {
    const auto& [batch_size, next_n, num_heads, head_dim] = get_shape<4>(q);
    const auto& [num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf] = get_shape<4>(fused_kv_cache);
    const auto& [batch_size_] = get_shape<1>(context_lens);
    const auto& [batch_size_next_n, num_heads_] = get_shape<2>(weights);
    const auto& [batch_size__, max_block_len] = get_shape<2>(block_table);
    const auto& [schedule_meta_size, meta_info_size] = get_shape<2>(schedule_meta);

    const auto& [batch_size___, max_kv_len] = get_shape<2>(page_table_1);
    const auto& [device_pool_size, mla_num_heads_kv, mla_head_dim] = get_shape<3>(device_pool_buf);
    const auto& [host_pool_size, mla_num_heads_kv_, mla_head_dim_] = get_shape<3>(host_pool_buf);
    const auto& [batch_size_pf, req_pf_max] = get_shape<2>(prefetch_host_loc);
    const auto& [batch_size_pf_kv, req_pf_max_kv, mla_head_dim_pf] = get_shape<3>(prefetch_kv_buf);
    const auto& [host_pool_size_] = get_shape<1>(host_token_to_device);
    const auto& [batch_size____] = get_shape<1>(decode_topk_logits);

    const auto& num_sms = device_runtime->get_num_sms();
    const auto& kv_cache_stride_bytes = fused_kv_cache.stride(0);
    const auto& block_table_stride = block_table.stride(0);
    const auto& page_table_1_stride = page_table_1.stride(0);

    DG_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    DG_HOST_ASSERT(batch_size == batch_size___ and batch_size == batch_size____ and batch_size == batch_size_pf);
    DG_HOST_ASSERT(batch_size == batch_size_pf_kv);
    DG_HOST_ASSERT(batch_size_next_n == batch_size * next_n);
    DG_HOST_ASSERT(num_heads == num_heads_ and num_heads_kv == 1);
    DG_HOST_ASSERT(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
    DG_HOST_ASSERT(schedule_meta_size == num_sms + 1 and meta_info_size == 2);

    DG_HOST_ASSERT(next_n == 1 or next_n == 2);
    DG_HOST_ASSERT(block_kv == 64);

    DG_HOST_ASSERT(q.is_contiguous());
    DG_HOST_ASSERT(kv_cache_stride_bytes % sizeof(float) == 0);
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf);
    DG_HOST_ASSERT(fused_kv_cache.stride(2) == head_dim_with_sf);
    DG_HOST_ASSERT(fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(weights.is_contiguous());
    DG_HOST_ASSERT(context_lens.is_contiguous());
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(schedule_meta.is_contiguous());

    DG_HOST_ASSERT(q.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt);

    DG_HOST_ASSERT(page_table_1.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(device_pool_buf.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(host_pool_buf.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(prefetch_host_loc.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(prefetch_kv_buf.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(host_token_to_device.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(decode_topk_logits.scalar_type() == torch::kFloat32);

    DG_HOST_ASSERT(max_kv_len <= num_kv_blocks * block_kv);
    DG_HOST_ASSERT(req_pf_max == req_pf_max_kv);
    DG_HOST_ASSERT(host_pool_size + 1 == host_pool_size_);  // host_token_to_device need additional for -1
    DG_HOST_ASSERT(mla_num_heads_kv == 1 and mla_num_heads_kv_ == 1);
    DG_HOST_ASSERT(mla_head_dim_ == mla_head_dim and mla_head_dim_pf == mla_head_dim);

    // Derive FP8 values and SF tensor from KV cache
    const auto& kv_cache = torch::from_blob(
        fused_kv_cache.data_ptr(),
        {num_kv_blocks, block_kv, head_dim},
        {kv_cache_stride_bytes, head_dim, 1},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn)
    );
    const auto& kv_cache_scales = torch::from_blob(
        fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim,
        {num_kv_blocks, block_kv},
        {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
        torch::TensorOptions().dtype(torch::kFloat32)
    );

    // Allocate output
    constexpr int num_math_warp_groups = 4;
    const auto& aligned_max_context_len = align(max_context_len, num_math_warp_groups * block_kv);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q.options().dtype(torch::kFloat));
    logits = logits.slice(-1, 0, max_context_len);

    // Dispatch implementation
    const auto& arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        smxx_fp8_paged_mqa_logits_fused_v2(
            q, kv_cache, kv_cache_scales, weights, context_lens, logits, block_table, schedule_meta,
            page_table_1, device_pool_buf, host_pool_buf, prefetch_host_loc,
            prefetch_kv_buf, host_token_to_device, query_recall_counter, decode_topk_logits, device_pool_size - 1, mla_head_dim, 
            batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv,
            kv_cache_stride_bytes, aligned_max_context_len, block_table_stride, page_table_1_stride,
            num_sms, num_math_warp_groups
        );
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, std::nullopt, context_lens, next_n, batch_size * next_n, max_context_len, aligned_max_context_len);
    return logits;
}

static void register_apis(pybind11::module_& m) {
    m.def("fp8_gemm_nt_skip_head_mid", &fp8_gemm_nt_skip_head_mid,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("head_splits"),
          py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_mqa_logits", &fp8_mqa_logits,
      py::arg("q"), py::arg("kv"), py::arg("weights"),
      py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
      py::arg("clean_logits") = true);
    m.def("fp8_mqa_logits_fuse_topk", &fp8_mqa_logits_fuse_topk,
      py::arg("q"), py::arg("kv"), py::arg("weights"),
      py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
      py::arg("clean_logits") = true);
    m.def("fp8_mqa_logits_fuse_prefetch", &fp8_mqa_logits_fuse_prefetch,
      py::arg("q"), py::arg("kv"), py::arg("weights"),
      py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
      py::arg("page_table_1"),py::arg("extend_seq_lens"), py::arg("extend_seq_to_req"), 
      py::arg("device_pool_buf"), py::arg("host_pool_buf"),
      py::arg("device_pool_loc_alloc_buf"),
      py::arg("device_pool_priority"), py::arg("device_pool_loc_small_priority"), 
      py::arg("device_token_to_host"), py::arg("host_token_to_device"),
      py::arg("recall_counter"), py::arg("extend_logits_offsets"),
      py::arg("clean_logits") = true);
    m.def("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_kv"), py::arg("num_sms"));
    m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("clean_logits") = false);
    m.def("fp8_paged_mqa_logits_fused_v2", &fp8_paged_mqa_logits_fused_v2,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), 
          py::arg("page_table_1"), 
          py::arg("device_pool_buf"), py::arg("host_pool_buf"),
          py::arg("prefetch_host_loc"), py::arg("prefetch_kv_buf"),
          py::arg("host_token_to_device"), py::arg("query_recall_counter"), 
          py::arg("decode_topk_logits"),
          py::arg("clean_logits") = false);
}

} // namespace deep_gemm::attention
