// Copyright (c) 2026. ECHO experimental Ascend backend; MIT license.
#pragma once
#include <climits>
#include <vector>
#include <c10/core/DeviceGuard.h>
#include <deep_gemm/echo_prefetch_types.hpp>
#include "attention.hpp"

namespace deep_gemm::echo_api {
namespace py = pybind11;

static void require_tensor(const torch::Tensor& t, const torch::Tensor& q,
                           at::ScalarType dtype, bool contiguous = true) {
    DJ_HOST_ASSERT(t.device() == q.device() && t.scalar_type() == dtype);
    DJ_HOST_ASSERT(!contiguous || t.is_contiguous());
}

// Caller-owned outputs and already packed scales: no tensor allocation or host
// reads of device state. Warm up JIT before graph capture, retain all buffers.
static void paged_mqa_prefetch_out(
    const torch::Tensor& q, const torch::Tensor& q_scales, const torch::Tensor& kv,
    const torch::Tensor& weights, const torch::Tensor& context_lens, const torch::Tensor& pages,
    const torch::Tensor& indices, const torch::Tensor& metadata,
    const torch::Tensor& thresholds, const torch::Tensor& stable, const torch::Tensor& active,
    const torch::Tensor& budgets, const torch::Tensor& host_ids,
    const torch::Tensor& host_to_device, const torch::Tensor& written,
    torch::Tensor& logits, torch::Tensor& counts, torch::Tensor& tokens,
    torch::Tensor& hosts, torch::Tensor& staging, uint64_t source_dva)
{
    DJ_HOST_ASSERT(q.device().type() == c10::DeviceType::PrivateUse1, "ECHO DeepGEMM requires Ascend NPU");
    DJ_HOST_ASSERT(q.dim() == 4 && q.size(0) > 0 && q.size(1) == 1 && q.size(2) == 32 && q.size(3) == 128);
    require_tensor(q, q, at::kFloat8_e4m3fn);
    const auto rows = q.size(0);
    DJ_HOST_ASSERT(rows <= INT32_MAX / 32);
    require_tensor(q_scales,q,at::kShort,false);
    DJ_HOST_ASSERT(q_scales.dim() == 2 && q_scales.size(0) == rows * 32 && q_scales.size(1) == 2 &&
                   q_scales.stride(0) == 1 && q_scales.stride(1) == rows * 32,
                   "Q scales must be prepacked int16 MX scales in MN-major layout");
    require_tensor(kv,q,at::kByte);
    DJ_HOST_ASSERT(kv.dim() == 4 && kv.size(0) > 0 && kv.size(0) <= INT32_MAX && kv.size(1) == 128 &&
                   kv.size(2) == 1 && kv.size(3) == 132);
    require_tensor(weights,q,at::kBFloat16);
    DJ_HOST_ASSERT(weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == 32);
    require_tensor(context_lens,q,at::kInt);
    DJ_HOST_ASSERT(context_lens.dim() == 2 && context_lens.size(0) == rows && context_lens.size(1) == 1);
    require_tensor(indices,q,at::kInt);
    DJ_HOST_ASSERT(indices.dim() == 1 && indices.numel() == rows);
    require_tensor(pages,q,at::kInt);
    DJ_HOST_ASSERT(pages.dim() == 2 && pages.size(0) == rows && pages.size(1) > 0 && pages.size(1) <= INT32_MAX);
    for (const auto& t : {stable,budgets,counts}) {
        require_tensor(t,q,at::kInt); DJ_HOST_ASSERT(t.dim() == 1 && t.numel() == rows);
    }
    require_tensor(thresholds,q,at::kFloat);
    DJ_HOST_ASSERT(thresholds.dim() == 1 && thresholds.numel() == rows);
    require_tensor(active,q,at::kInt); DJ_HOST_ASSERT(active.numel() == 1);
    require_tensor(host_ids,q,at::kInt);
    DJ_HOST_ASSERT(host_ids.dim() == 2 && host_ids.size(0) == rows && host_ids.size(1) > 0 && host_ids.size(1) <= INT32_MAX);
    const auto context = host_ids.size(1);
    DJ_HOST_ASSERT(pages.size(1) >= (context + 127) / 128);
    require_tensor(host_to_device,q,at::kInt); require_tensor(written,q,at::kInt);
    DJ_HOST_ASSERT(host_to_device.dim() == 1 && host_to_device.numel() > 1 && host_to_device.numel() <= INT32_MAX &&
                   written.dim() == 1 && written.numel() == host_to_device.numel());
    require_tensor(logits,q,at::kBFloat16,false);
    DJ_HOST_ASSERT(logits.dim() == 2 && logits.size(0) == rows && logits.size(1) == context &&
                   logits.stride(1) == 1 && logits.stride(0) >= context && logits.stride(0) <= UINT32_MAX &&
                   logits.stride(0) * sizeof(uint16_t) % 512 == 0);
    require_tensor(tokens,q,at::kInt); require_tensor(hosts,q,at::kInt); require_tensor(staging,q,at::kByte);
    DJ_HOST_ASSERT(tokens.dim() == 2 && tokens.size(0) == rows && tokens.size(1) > 0 && tokens.size(1) <= 2048 &&
                   hosts.sizes() == tokens.sizes());
    DJ_HOST_ASSERT(staging.dim() == 3 && staging.size(0) == rows && staging.size(1) == tokens.size(1) &&
                   staging.size(2) > 0 && staging.size(2) <= 65536);
    DJ_HOST_ASSERT(source_dva != 0 && source_dva <= INT64_MAX &&
                   static_cast<uint64_t>(host_to_device.numel()) * staging.size(2) <= INT64_MAX - source_dva);
    const c10::OptionalDeviceGuard guard(q.device());
    DJ_HOST_ASSERT(jit->device.get_npu_arch() == 3510,
                   "ECHO DeepGEMM targets Ascend 950 (dav-3510); architecture overrides do not enable 910");
    const uint32_t cores = runtime->get_num_sms();
    require_tensor(metadata,q,at::kInt);
    DJ_HOST_ASSERT(metadata.dim() == 2 && metadata.size(0) == cores+1 && metadata.size(1) == 2);
    const std::vector<torch::Tensor> inputs{q,q_scales,kv,weights,context_lens,pages,indices,
        thresholds,stable,active,budgets,host_ids,host_to_device,written};
    const std::vector<torch::Tensor> outputs{metadata,logits,counts,tokens,hosts,staging};
    for (size_t i=0;i<outputs.size();++i) {
        for (const auto& input:inputs) DJ_HOST_ASSERT(!outputs[i].is_alias_of(input),"ECHO outputs cannot alias inputs");
        for (size_t j=0;j<i;++j) DJ_HOST_ASSERT(!outputs[i].is_alias_of(outputs[j]),"ECHO outputs cannot alias each other");
    }
    // Rebuild the schedule on device every replay; never reuse a schedule from
    // different lengths/request groups. These are ordinary same-stream launches.
    launch_paged_mqa_logits_metadata(rows,16,512,2,cores,context_lens.data_ptr(),indices.data_ptr(),metadata.data_ptr());
    EchoPrefetchArgs args{
        reinterpret_cast<uintptr_t>(thresholds.data_ptr()),reinterpret_cast<uintptr_t>(stable.data_ptr()),
        reinterpret_cast<uintptr_t>(active.data_ptr()),reinterpret_cast<uintptr_t>(budgets.data_ptr()),
        reinterpret_cast<uintptr_t>(host_ids.data_ptr()),reinterpret_cast<uintptr_t>(host_to_device.data_ptr()),
        reinterpret_cast<uintptr_t>(written.data_ptr()),reinterpret_cast<uintptr_t>(counts.data_ptr()),
        reinterpret_cast<uintptr_t>(tokens.data_ptr()),reinterpret_cast<uintptr_t>(hosts.data_ptr()),
        reinterpret_cast<uintptr_t>(staging.data_ptr()),source_dva,static_cast<uint32_t>(rows),
        static_cast<uint32_t>(context),static_cast<uint32_t>(host_to_device.numel()),
        static_cast<uint32_t>(tokens.size(1)),static_cast<uint32_t>(staging.size(2))};
    const std::string code=R"(
#include <deep_gemm/echo_mqa_logits.hpp>
static void __instantiate() {
    auto p = reinterpret_cast<void*>(&deep_gemm::echo_paged_mqa_logits_impl);
}
)";
    if (runtime->get_dry_run()) { jit->compile_without_load("echo_paged_mqa_logits",code); return; }
    const auto kernel=jit->compile("echo_paged_mqa_logits",code);
    jit->launch(kernel,{.num_blocks=static_cast<int>(cores),.num_ubuf_bytes=224*1024},
        static_cast<uint32_t>(rows),static_cast<uint32_t>(pages.stride(0)),static_cast<uint64_t>(kv.stride(0)),
        gm_ptr<uint8_t>{128,reinterpret_cast<uintptr_t>(q.data_ptr())},q_scales.data_ptr(),
        gm_ptr<uint8_t>{128,reinterpret_cast<uintptr_t>(kv.data_ptr())},
        gm_ptr<uint8_t>{32,reinterpret_cast<uintptr_t>(weights.data_ptr())},
        gm_ptr<uint8_t>{static_cast<uint64_t>(logits.stride(0)),reinterpret_cast<uintptr_t>(logits.data_ptr())},
        context_lens.data_ptr(),pages.data_ptr(),indices.data_ptr(),metadata.data_ptr(),args);
}
static void register_apis(py::module_& m) {
    m.def("echo_paged_mqa_prefetch_out",paged_mqa_prefetch_out,
        py::arg("q"),py::arg("q_scales"),py::arg("kv"),py::arg("weights"),py::arg("context_lens"),
        py::arg("pages"),py::arg("indices"),py::arg("metadata"),py::arg("thresholds"),py::arg("stable"),
        py::arg("active"),py::arg("budgets"),py::arg("host_ids"),py::arg("host_to_device"),py::arg("written"),
        py::arg("logits"),py::arg("counts"),py::arg("tokens"),py::arg("hosts"),py::arg("staging"),py::arg("source_dva"));
}
} // namespace deep_gemm::echo_api
